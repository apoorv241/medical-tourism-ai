from __future__ import annotations

import json
import datetime as dt
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


SUPPORTED_SET = {
    "thailand",
    "mexico",
    "india",
    "colombia",
    "turkey",
    "costa rica",
    "brazil",
    "malaysia",
    "singapore",
}


class LangGraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    language: Optional[str]
    translation: Optional[str]

    constraints: Dict[str, Any]

    missing_fields: List[str]
    needs_clarification: bool
    clarification_question: Optional[str]

    # destination check
    destination_supported: bool
    unsupported_destinations: List[str]
    supported_destinations: List[str]
    destination_check_message: Optional[str]


# -------------------------
# Tool: check destination support
# -------------------------
@tool("check_destination_supported")
def check_destination_tool(destinations: List[str]) -> Dict[str, Any]:
    """
    Check whether destination countries are supported by our database.
    Input: list of country names from the user (preferred_countries).
    Output: supported/unsupported lists and a boolean.
    """

    def norm(x: str) -> str:
        return " ".join((x or "").strip().lower().split())

    dest_norm: List[str] = []
    for d in destinations or []:
        dn = norm(d)
        if dn:
            dest_norm.append(dn)

    # unique preserve order
    seen = set()
    dest_u: List[str] = []
    for d in dest_norm:
        if d not in seen:
            seen.add(d)
            dest_u.append(d)

    supported = [d for d in dest_u if d in SUPPORTED_SET]
    unsupported = [d for d in dest_u if d not in SUPPORTED_SET]

    destination_supported = (len(dest_u) > 0 and len(unsupported) == 0)

    return {
        "destination_supported": destination_supported,
        "supported_destinations": supported,
        "unsupported_destinations": unsupported,
    }


class LanguageGraphDetector:
    REQUIRED_FIELDS = ["procedure", "budget", "origin_location", "travel_dates"]

    def __init__(self, llm, *, today: Optional[dt.date] = None):
        self.llm = llm
        self.llm_with_tools = self.llm.bind_tools([check_destination_tool])

        # determinism: use an injected "today" for testing, else system date
        self.today: dt.date = today or dt.date.today()

        self.checkpointer = MemorySaver()
        self.graph = self._build()

        # optional: render graph image once at init
        try:
            lg_graph = self.graph.get_graph()
            lg_graph.draw_mermaid_png(output_file_path="medical_tourism__lang_agent_graph.png")
        except Exception:
            # don't break runtime if drawing isn't available in the environment
            pass

    # -------------------------
    # Helpers
    # -------------------------
    def _is_destination_prompt(self, q: Optional[str]) -> bool:
        if not q:
            return False
        q = q.lower()
        return (
            ("destination country" in q)
            or ("which destination" in q)
            or ("supported destination" in q)
            or ("pick a supported destination" in q)
            or ("choose one of these destinations" in q)
            or ("i don’t have destination data" in q)
            or ("i don't have destination data" in q)
        )

    def _safe_json_load(self, raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            s = raw.find("{")
            e = raw.rfind("}")
            if s == -1 or e == -1:
                raise ValueError(f"LLM did not return JSON. Got: {raw}")
            return json.loads(raw[s : e + 1])

    def _clean_one_line(self, s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return None
        return " ".join(s.split())

    def _last_user_text(self, messages: list) -> str:
        for m in reversed(messages or []):
            if isinstance(m, BaseMessage) and m.type == "human":
                return (m.content or "").strip()
            if isinstance(m, dict) and m.get("role") == "user":
                return (m.get("content") or "").strip()
        return ""

    def _normalize_budget(self, budget: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(budget, dict):
            return None
        maxv = budget.get("max")
        cur = budget.get("currency")

        if isinstance(maxv, str):
            try:
                maxv = float(maxv)
            except Exception:
                maxv = None

        if maxv is None and (cur is None or str(cur).strip() == ""):
            return None

        return {"max": maxv, "currency": cur}

    def _normalize_travel_dates(self, td: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(td, dict):
            return None

        out = {
            "start": td.get("start"),
            "end": td.get("end"),
            "notes": td.get("notes"),
            "flexible": td.get("flexible"),
        }

        flex = out.get("flexible", None)
        has_any = bool(out["start"] or out["end"] or out["notes"] or (flex is not None))
        return out if has_any else None

    def _merge_constraints(self, old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(old or {})

        if new.get("procedure"):
            merged["procedure"] = new["procedure"]

        if new.get("origin_location"):
            merged["origin_location"] = new["origin_location"]

        old_c = merged.get("preferred_countries") or []
        new_c = new.get("preferred_countries") or []
        if isinstance(old_c, list) and isinstance(new_c, list):
            merged["preferred_countries"] = sorted(
                set([str(x).strip() for x in (old_c + new_c) if str(x).strip()])
            )

        nb = self._normalize_budget(new.get("budget"))
        if nb:
            merged["budget"] = nb
        else:
            merged.setdefault("budget", {"max": None, "currency": None})

        ntd = self._normalize_travel_dates(new.get("travel_dates"))
        if ntd:
            merged["travel_dates"] = ntd
        else:
            merged.setdefault(
                "travel_dates",
                {"start": None, "end": None, "notes": None, "flexible": None},
            )

        return merged

    # -------------------------
    # NEW: deterministic future-date validation (don’t rely on LLM)
    # -------------------------
    def _parse_iso_date(self, s: Any) -> Optional[dt.date]:
        """Parse YYYY-MM-DD. Anything else => None (forces clarification)."""
        if not s:
            return None
        if not isinstance(s, str):
            return None
        s = s.strip()
        try:
            return dt.date.fromisoformat(s)  # expects YYYY-MM-DD
        except Exception:
            return None

    def _enforce_future_travel_dates(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rules enforced:
        - If start/end are present, they must parse as YYYY-MM-DD
        - start >= today, end >= today
        - end >= start
        If invalid, wipe start/end/notes/flexible to force re-ask.
        """
        td = constraints.get("travel_dates") or {}
        start_raw = td.get("start")
        end_raw = td.get("end")
        flex = td.get("flexible", None)
        notes = td.get("notes")

        # If user is using flexible range (flex=True + notes), we still require it to be future-ish,
        # but since notes are free text, we just accept it and do NOT attempt date math.
        has_flexible_range = (flex is True) and bool(notes and str(notes).strip())
        if has_flexible_range:
            return constraints

        # Strict dates path: require parseable ISO start+end
        start = self._parse_iso_date(start_raw)
        end = self._parse_iso_date(end_raw)

        if start is None or end is None:
            # If they provided something but not ISO, force re-ask.
            if start_raw or end_raw:
                constraints = dict(constraints)
                constraints["travel_dates"] = {"start": None, "end": None, "notes": None, "flexible": None}
            return constraints

        # Validate future + ordering
        if start < self.today or end < self.today or end < start:
            constraints = dict(constraints)
            constraints["travel_dates"] = {"start": None, "end": None, "notes": None, "flexible": None}
            return constraints

        return constraints

    def _compute_missing(self, constraints: Dict[str, Any]) -> List[str]:
        missing: List[str] = []

        if not constraints.get("procedure"):
            missing.append("procedure")

        if not constraints.get("origin_location"):
            missing.append("origin_location")

        budget = constraints.get("budget") or {}
        if not budget.get("max"):
            missing.append("budget")

        td = constraints.get("travel_dates") or {}
        start = td.get("start")
        end = td.get("end")
        notes = td.get("notes")
        flex = td.get("flexible", None)

        has_strict_dates = bool(start and end)
        has_flexible_range = (flex is True) and bool(notes and str(notes).strip())

        if not (has_strict_dates or has_flexible_range):
            missing.append("travel_dates")

        return missing

    def _messages_to_json(self, messages: list) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages or []:
            if isinstance(m, BaseMessage):
                role = "user" if m.type == "human" else ("assistant" if m.type == "ai" else m.type)
                out.append({"role": role, "content": m.content})
            elif isinstance(m, dict):
                out.append({"role": m.get("role", "unknown"), "content": m.get("content", "")})
            else:
                out.append({"role": "unknown", "content": str(m)})
        return out

    def _init_state(self, state: LangGraphState) -> LangGraphState:
        state.setdefault("constraints", {})
        state.setdefault("missing_fields", [])
        state.setdefault("needs_clarification", False)
        state.setdefault("language", None)
        state.setdefault("translation", None)
        state.setdefault("clarification_question", None)

        state.setdefault("destination_supported", True)
        state.setdefault("unsupported_destinations", [])
        state.setdefault("supported_destinations", [])
        state.setdefault("destination_check_message", None)
        return state

    # -------------------------
    # Deterministic travel date question
    # -------------------------
    def _travel_dates_question(self, constraints: Dict[str, Any]) -> str:
        # now explicitly future + ISO format
        return (
            f"What are your travel dates (start and end) in YYYY-MM-DD format, "
            f"and they must be on/after {self.today.isoformat()}? "
            f"(Example: {self.today.isoformat()} to {(self.today + dt.timedelta(days=7)).isoformat()})"
        )

    # -------------------------
    # Intake node
    # -------------------------
    def intake_node(self, state: LangGraphState, config: RunnableConfig) -> LangGraphState:
        user_text = self._last_user_text(state.get("messages", []))
        if not user_text:
            return state

        prev_constraints = state.get("constraints") or {}

        # HITL destination correction:
        if state.get("needs_clarification") and self._is_destination_prompt(state.get("clarification_question")):
            merged_constraints = dict(prev_constraints)
            merged_constraints["preferred_countries"] = [user_text.strip()]

            # enforce dates again (in case they were already present)
            merged_constraints = self._enforce_future_travel_dates(merged_constraints)

            missing = self._compute_missing(merged_constraints)
            needs = bool(missing)

            assistant_text = "Thanks—checking destination availability..."

            return {
                "messages": [{"role": "assistant", "content": assistant_text}],
                "language": state.get("language"),
                "translation": state.get("translation"),
                "constraints": merged_constraints,
                "missing_fields": missing,
                "needs_clarification": needs,
                "clarification_question": None,
            }

        # IMPORTANT: include today's date so the model has context,
        # but we STILL validate in code.
        prompt = (
            "You are a medical tourism intake agent.\n\n"
            "Return ONLY valid JSON in this exact schema:\n"
            "{\n"
            '  "language": "English|French|Hindi|...",\n'
            '  "translation": "English translation (or original if already English)",\n'
            '  "constraints": {\n'
            '    "procedure": null,\n'
            '    "budget": { "max": null, "currency": null },\n'
            '    "origin_location": null,\n'
            '    "preferred_countries": [],\n'
            '    "travel_dates": { "start": null, "end": null, "notes": null, "flexible": null }\n'
            "  }\n"
            "}\n\n"
            "Rules:\n"
            f"- Today is {self.today.isoformat()}.\n"
            "- origin_location = where the user starts from (city/state/country), NOT the destination.\n"
            "- preferred_countries = destination countries mentioned.\n"
            "- If user provides strict dates, output travel_dates.start and travel_dates.end as ISO YYYY-MM-DD.\n"
            "- If dates are vague, put that in travel_dates.notes and set flexible=true.\n"
            "- budget.max MUST be a number when present.\n"
            "- Output JSON only.\n\n"
            f"User input: {user_text}\n"
        )

        raw = self.llm.invoke(prompt, config=config).content
        data = self._safe_json_load(raw)

        language = self._clean_one_line(data.get("language")) or "Unknown"
        translation = self._clean_one_line(data.get("translation")) or user_text

        new_constraints = data.get("constraints") or {}
        merged_constraints = self._merge_constraints(prev_constraints, new_constraints)

        # ✅ enforce future + ordering deterministically
        merged_constraints = self._enforce_future_travel_dates(merged_constraints)

        missing = self._compute_missing(merged_constraints)
        needs = bool(missing)

        clarification_question: Optional[str] = None
        if needs:
            parts: List[str] = []

            if "travel_dates" in missing:
                parts.append(self._travel_dates_question(merged_constraints))

            other_missing = [m for m in missing if m != "travel_dates"]
            if other_missing:
                missing_list = ", ".join(other_missing)
                clarify_prompt = (
                    "Ask ONLY about the missing fields listed.\n"
                    "Ask ONE short question per missing field.\n"
                    "Do NOT ask about travel dates.\n"
                    "Do NOT repeat previous questions.\n"
                    "For budget, ALWAYS ask for the MAXIMUM amount the user is willing to spend.\n"
                    "Return plain text only.\n\n"
                    f"Missing fields: {missing_list}\n"
                    f"Current constraints:\n{json.dumps(merged_constraints, ensure_ascii=False)}\n"
                    f"Respond in the user's language: {language}.\n"
                )

                llm_q = self.llm.invoke(clarify_prompt, config=config).content
                llm_q = self._clean_one_line(llm_q)
                if llm_q:
                    parts.append(llm_q)

            clarification_question = self._clean_one_line(" ".join(parts))

        assistant_text = clarification_question if needs else "Got it. I have all required details."

        return {
            "messages": [{"role": "assistant", "content": assistant_text}],
            "language": language,
            "translation": translation,
            "constraints": merged_constraints,
            "missing_fields": missing,
            "needs_clarification": needs,
            "clarification_question": clarification_question,
        }

    # -------------------------
    # Destination check (tool-enabled LLM calls tool)
    # -------------------------
    def dest_check_llm_node(self, state: LangGraphState, config: RunnableConfig) -> LangGraphState:
        constraints = state.get("constraints") or {}
        preferred = constraints.get("preferred_countries") or []
        if not isinstance(preferred, list):
            preferred = []

        if len(preferred) == 0:
            msg = "Which destination country are you considering? (Example: Thailand, India, Mexico)"
            return {
                "messages": [{"role": "assistant", "content": msg}],
                "destination_supported": False,
                "needs_clarification": True,
                "clarification_question": msg,
                "destination_check_message": msg,
            }

        prompt = (
            "Always call the tool `check_destination_supported` with the list of destinations.\n"
            f"Destinations: {json.dumps(preferred, ensure_ascii=False)}"
        )

        ai = self.llm_with_tools.invoke(prompt, config=config)
        return {"messages": [ai]}

    def _route_to_tools(self, state: LangGraphState) -> str:
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "apply"

    def tools_node(self, state: LangGraphState, config: RunnableConfig) -> LangGraphState:
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
            return state

        tool_messages: List[ToolMessage] = []

        for tc in last.tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}
            tool_call_id = tc.get("id")

            if name == "check_destination_supported":
                # expects args like {"destinations":[...]}
                result = check_destination_tool.invoke(args)
                tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id))

        return {"messages": tool_messages}

    def dest_check_apply_node(self, state: LangGraphState, config: RunnableConfig) -> LangGraphState:
        msgs = state.get("messages") or []

        tool_msg = None
        for m in reversed(msgs):
            if isinstance(m, ToolMessage):
                tool_msg = m
                break

        if tool_msg is None:
            return state

        data = json.loads(tool_msg.content or "{}")

        unsupported = data.get("unsupported_destinations") or []
        supported = data.get("supported_destinations") or []
        ok = data.get("destination_supported") is True

        if ok:
            msg = "Destination supported. Proceeding."
            return {
                "messages": [{"role": "assistant", "content": msg}],
                "destination_supported": True,
                "unsupported_destinations": [],
                "supported_destinations": supported,
                "destination_check_message": msg,
                "needs_clarification": False,
                "clarification_question": None,
            }

        supported_list = ", ".join(sorted(SUPPORTED_SET))
        msg = (
            "I don’t have destination data for that country yet. "
            f"Pick one of these supported destinations: {supported_list}."
        )

        return {
            "messages": [{"role": "assistant", "content": msg}],
            "destination_supported": False,
            "unsupported_destinations": unsupported,
            "supported_destinations": supported,
            "destination_check_message": msg,
            "needs_clarification": True,
            "clarification_question": msg,
        }

    # -------------------------
    # Build graph
    # -------------------------
    def _route_after_intake(self, state: LangGraphState) -> str:
        if state.get("needs_clarification"):
            return "end"
        return "dest_check_llm"

    def _build(self):
        builder = StateGraph(LangGraphState)

        builder.add_node("init", self._init_state)
        builder.add_node("intake", self.intake_node)
        builder.add_node("dest_check_llm", self.dest_check_llm_node)
        builder.add_node("tools", self.tools_node)
        builder.add_node("apply", self.dest_check_apply_node)

        builder.add_edge(START, "init")
        builder.add_edge("init", "intake")

        builder.add_conditional_edges(
            "intake",
            self._route_after_intake,
            {"dest_check_llm": "dest_check_llm", "end": END},
        )

        builder.add_conditional_edges(
            "dest_check_llm",
            self._route_to_tools,
            {"tools": "tools", "apply": "apply"},
        )

        builder.add_edge("tools", "apply")
        builder.add_edge("apply", END)

        return builder.compile(checkpointer=self.checkpointer)

    # -------------------------
    # Public invoke
    # -------------------------
    def invoke(self, user_text: str, thread_id: str) -> Dict[str, Any]:
        state_in: LangGraphState = {
            "messages": [{"role": "user", "content": user_text}],
        }

        out: LangGraphState = self.graph.invoke(
            state_in,
            config={"configurable": {"thread_id": thread_id}},
        )

        return {
            "language": out.get("language"),
            "translation": out.get("translation"),
            "constraints": out.get("constraints") or {},
            "missing_fields": out.get("missing_fields") or [],
            "needs_clarification": bool(out.get("needs_clarification")),
            "clarification_question": out.get("clarification_question"),
            "destination_supported": bool(out.get("destination_supported")),
            "unsupported_destinations": out.get("unsupported_destinations") or [],
            "supported_destinations": out.get("supported_destinations") or [],
            "destination_check_message": out.get("destination_check_message"),
            "messages": self._messages_to_json(out.get("messages", [])),
        }
