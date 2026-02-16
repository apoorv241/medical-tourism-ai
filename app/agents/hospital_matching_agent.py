from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.tools.hospital_tools import search_hospitals_tool, pick_best_hospitals_tool


class HospitalMatchState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    constraints: Dict[str, Any]
    hospitals_raw: List[Dict[str, Any]]
    hospitals_ranked: List[Dict[str, Any]]

    needs_clarification: bool
    clarification_question: Optional[str]


class HospitalMatchingAgent:
    """
    LLM decides which tools to call.
    Inputs: constraints (from language agent)
    Output: hospitals_ranked OR clarification_question
    """

    def __init__(self, llm):
        self.tools = [search_hospitals_tool, pick_best_hospitals_tool]
        self.llm = llm.bind_tools(self.tools) 
        self.checkpointer = MemorySaver()
        self.graph = self._build()

    def _safe_get_destination(self, constraints: Dict[str, Any]) -> Dict[str, Optional[str]]:
        dest = constraints.get("destination") or {}
        return {
            "country": (dest.get("country") or "").strip() or None,
            "city": (dest.get("city") or "").strip() or None,
        }

    def _init_state(self, state: HospitalMatchState) -> HospitalMatchState:
        state.setdefault("constraints", {})
        state.setdefault("hospitals_raw", [])
        state.setdefault("hospitals_ranked", [])
        state.setdefault("needs_clarification", False)
        state.setdefault("clarification_question", None)
        return state

    def _build_prompt_from_constraints(self, constraints: Dict[str, Any]) -> str:
        """
        We instruct the LLM to call tools in the right order.
        City may be missing -> still call search_hospitals_tool with city=None.
        """
        procedure = (constraints.get("procedure") or "").strip()
        budget = constraints.get("budget") or {}
        budget_max = budget.get("max", None)

        dest = self._safe_get_destination(constraints)
        country = dest["country"]
        city = dest["city"]

        return f"""
You are the Hospital Matching Agent.

You MUST do:
1) If procedure or country is missing, ask ONE clarification question and do NOT call tools.
2) Otherwise call:
   - search_hospitals_tool(procedure, country, city?)  (city can be null)
   - then call pick_best_hospitals_tool(hospitals=<search results>, budget_max=<budget_max>)
3) Finally respond with STRICT JSON only:
   {{
     "needs_clarification": false,
     "clarification_question": null,
     "hospitals": [... top 5 ...]
   }}

Constraints:
procedure={procedure or None}
country={country or None}
city={city or None}
budget_max={budget_max}
""".strip()

    def llm_node(self, state: HospitalMatchState, config: RunnableConfig) -> HospitalMatchState:
        constraints = state.get("constraints") or {}
        prompt = self._build_prompt_from_constraints(constraints)

        msgs = [
            SystemMessage(content="You are a tool-using assistant. Follow instructions exactly."),
            HumanMessage(content=prompt),
        ]

        # ✅ This returns an AIMessage which MAY include tool calls
        ai_msg = self.llm.invoke(msgs, config=config)

        return {"messages": [ai_msg]}

    def finalize_node(self, state: HospitalMatchState, config: RunnableConfig) -> HospitalMatchState:
        """
        After the tool loop ends, the last assistant message should contain final JSON.
        We parse it into structured outputs for your API.
        """
        last = None
        msgs = state.get("messages") or []
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            # AIMessage has .content, but it can be a dict/list too depending on libs.
            content = getattr(m, "content", None) if hasattr(m, "content") else m.get("content")
            if content:
                last = content
                break

        # Default fallback
        out = {
            "needs_clarification": False,
            "clarification_question": None,
            "hospitals": [],
        }

        if isinstance(last, str):
            try:
                out = json.loads(last)
            except Exception:
                # if model didn't follow JSON-only instruction, return raw as clarification
                out = {
                    "needs_clarification": True,
                    "clarification_question": last,
                    "hospitals": [],
                }

        state_update: HospitalMatchState = {
            "needs_clarification": bool(out.get("needs_clarification")),
            "clarification_question": out.get("clarification_question"),
            "hospitals_ranked": out.get("hospitals") or [],
        }
        return state_update

    def _build(self):
        builder = StateGraph(HospitalMatchState)

        tool_node = ToolNode(self.tools)  # ✅ executes @tool calls

        builder.add_node("init", self._init_state)
        builder.add_node("llm", self.llm_node)
        builder.add_node("tools", tool_node)
        builder.add_node("finalize", self.finalize_node)

        builder.add_edge(START, "init")
        builder.add_edge("init", "llm")

        # ✅ Router: if LLM emitted tool calls -> go to tools, else finalize
        builder.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: "finalize"})
        # After tools execute, go back to LLM for next step (e.g., ranking)
        builder.add_edge("tools", "llm")

        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, constraints: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        state_in: HospitalMatchState = {
            "constraints": constraints or {},
            "messages": [],  # LLM node will create messages
        }

        out: HospitalMatchState = self.graph.invoke(
            state_in,
            config={"configurable": {"thread_id": thread_id}},
        )

        return {
            "needs_clarification": bool(out.get("needs_clarification")),
            "clarification_question": out.get("clarification_question"),
            "hospitals": out.get("hospitals_ranked") or [],
        }