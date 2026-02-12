from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


# -------------------------
# Tools (STUBS) - Replace internals with real API calls later
# -------------------------

@tool
def search_hospitals_tool(procedure: str, city: str, country: str) -> Dict[str, Any]:
    """
    Return a shortlist of hospitals/clinics relevant to a procedure.
    Replace stub with: Google Places / government registry / accreditation directory API.
    """
    # STUB response
    return {
        "query": {"procedure": procedure, "city": city, "country": country},
        "results": [
            {
                "name": "Example Eye Hospital",
                "address": f"{city}, {country}",
                "rating": 4.6,
                "approx_price_usd": 1800,
                "why": "High ratings + specializes in LASIK",
            },
            {
                "name": "Example Multispecialty Clinic",
                "address": f"{city}, {country}",
                "rating": 4.4,
                "approx_price_usd": 1600,
                "why": "Good outcomes + shorter wait times",
            },
        ],
    }


@tool
def pick_best_hospitals_tool(hospitals: List[Dict[str, Any]], budget_max: Optional[float]) -> Dict[str, Any]:
    """
    Rank hospitals by price-fit + rating. Replace with your own scoring logic.
    """
    if not hospitals:
        return {"ranked": []}

    ranked = sorted(
        hospitals,
        key=lambda x: (
            0 if (budget_max is None or (x.get("approx_price_usd") or 10**9) <= budget_max) else 1,
            -(x.get("rating") or 0),
        ),
    )
    return {"ranked": ranked[:5]}


# -------------------------
# LangGraph State
# -------------------------

class HospitalMatchState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    constraints: Dict[str, Any]
    hospitals_raw: List[Dict[str, Any]]
    hospitals_ranked: List[Dict[str, Any]]

    needs_clarification: bool
    clarification_question: Optional[str]


class HospitalMatchingAgent:
    """
    Inputs: constraints { procedure, destination (country/city), budget }
    Output: ranked hospitals + optional clarification if missing destination city, etc.
    """

    def __init__(self, llm):
        self.llm = llm
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

    def _missing_question(self, missing: List[str], language: str = "English") -> str:
        # deterministic and short
        if "destination.city" in missing:
            return "Which city are you traveling to for treatment?"
        if "procedure" in missing:
            return "What procedure are you looking for (e.g., LASIK, knee replacement)?"
        return "Can you share the missing details so I can match hospitals?"

    def match_node(self, state: HospitalMatchState, config: RunnableConfig) -> HospitalMatchState:
        constraints = state.get("constraints") or {}

        procedure = (constraints.get("procedure") or "").strip()
        budget = constraints.get("budget") or {}
        budget_max = budget.get("max", None)
        if isinstance(budget_max, str):
            try:
                budget_max = float(budget_max)
            except Exception:
                budget_max = None

        dest = self._safe_get_destination(constraints)
        country = dest["country"]
        city = dest["city"]

        missing: List[str] = []
        if not procedure:
            missing.append("procedure")
        if not country:
            missing.append("destination.country")
        if not city:
            missing.append("destination.city")

        if missing:
            q = self._missing_question(missing)
            return {
                "needs_clarification": True,
                "clarification_question": q,
                "messages": [{"role": "assistant", "content": q}],
            }

        # 1) hospital search tool
        raw = search_hospitals_tool.invoke(
            {"procedure": procedure, "city": city, "country": country},
            config=config,
        )
        hospitals = (raw or {}).get("results") or []

        # 2) ranking tool
        ranked = pick_best_hospitals_tool.invoke(
            {"hospitals": hospitals, "budget_max": budget_max},
            config=config,
        )
        hospitals_ranked = (ranked or {}).get("ranked") or []

        payload = {
            "hospitals": hospitals_ranked,
            "count": len(hospitals_ranked),
        }

        return {
            "hospitals_raw": hospitals,
            "hospitals_ranked": hospitals_ranked,
            "needs_clarification": False,
            "clarification_question": None,
            "messages": [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}],
        }

    def _build(self):
        builder = StateGraph(HospitalMatchState)
        builder.add_node("init", self._init_state)
        builder.add_node("match", self.match_node)

        builder.add_edge(START, "init")
        builder.add_edge("init", "match")
        builder.add_edge("match", END)

        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, constraints: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        state_in: HospitalMatchState = {
            "constraints": constraints or {},
            "messages": [{"role": "user", "content": "match hospitals"}],
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
