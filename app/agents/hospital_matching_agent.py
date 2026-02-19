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
import sys
import re
from langchain_core.messages import AIMessage, ToolMessage

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
        
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Handles:
        - plain JSON: {...}
        - fenced JSON: ```json {...} ```
        - extra text around JSON (best effort)
        """
        if not text:
            return {}

        s = text.strip()

        # If fenced: ```json ... ```
        if s.startswith("```"):
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
            if m:
                s = m.group(1).strip()

        # If extra junk, grab first {...}
        if not s.startswith("{"):
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                s = s[start : end + 1]

        return json.loads(s)

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
   - search_hospitals_tool(procedure, country, city?)
   - then call pick_best_hospitals_tool(hospitals=<search results>, budget_max=<budget_max>)

3) Finally respond with STRICT JSON only.

IMPORTANT:
- Each hospital MUST contain the REAL hospital name.
- DO NOT generate descriptive titles like "Best Lasik Hospital in Turkey".
- Use realistic hospital names.
- Output format MUST be:

{{
  "needs_clarification": false,
  "clarification_question": null,
  "hospitals": [
    {{
      "name": "Real Hospital Name",
      "rating": 4.5,
      "price": 1800
    }}
  ]
}}

Constraints:
procedure={procedure or None}
country={country or None}
city={city or None}
budget_max={budget_max}
"""

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
        After tool loop ends, parse the final assistant JSON and write it into state.
        IMPORTANT: The last message might be:
        - AIMessage with ```json fenced block
        - ToolMessage content
        - Multiple messages (tool traces + final)
        We search backwards for the most recent AIMessage content that contains '{'.
        """
        msgs = state.get("messages") or []

        last_text: Optional[str] = None

        # Walk backward to find a likely final JSON answer
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                if isinstance(m.content, str) and "{" in m.content:
                    last_text = m.content
                    break
            elif isinstance(m, ToolMessage):
                # tool outputs can be JSON strings too, but we prefer final AIMessage
                continue
            elif isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, str) and "{" in c:
                    last_text = c
                    break

        # Default fallback
        out_obj = {
            "needs_clarification": False,
            "clarification_question": None,
            "hospitals": [],
        }

        if last_text:
            try:
                out_obj = self._extract_json_from_text(last_text)
            except Exception:
                # If model didn't return parseable JSON, treat it as clarification text
                out_obj = {
                    "needs_clarification": True,
                    "clarification_question": last_text,
                    "hospitals": [],
                }
        else:
            out_obj = {
                "needs_clarification": True,
                "clarification_question": "No final response found from hospital agent.",
                "hospitals": [],
            }

        return {
            "needs_clarification": bool(out_obj.get("needs_clarification")),
            "clarification_question": out_obj.get("clarification_question"),
            "hospitals_ranked": out_obj.get("hospitals") or [],
        }
    def _build(self):
        builder = StateGraph(HospitalMatchState)

        tool_node = ToolNode(self.tools)  # ✅ executes @tool calls

        builder.add_node("init", self._init_state)
        builder.add_node("llm", self.llm_node)
        builder.add_node("tools", tool_node)
        builder.add_node("finalize", self.finalize_node)

        builder.add_edge(START, "init")
        builder.add_edge("init", "llm")

        builder.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: "finalize"})
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
        
        print("========", file=sys.stderr, flush=True)
        print(out, file=sys.stderr, flush=True)
        print("========", file=sys.stderr, flush=True)

        return {
            "needs_clarification": bool(out.get("needs_clarification")),
            "clarification_question": out.get("clarification_question"),
            "hospitals": out.get("hospitals_ranked") or [],
        }