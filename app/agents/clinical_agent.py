from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from app.ThirdPartyTools.procedure_lookup_tool import lookup_procedure


# -------------------------
# State
# -------------------------
class ClinicalState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    # carried from language agent
    constraints: Dict[str, Any]

    # procedure normalization
    procedure_normalized: Optional[str]
    procedure_category: Optional[str]

    # safety outputs
    is_procedure_safe: bool
    safety_flags: List[str]

    # intake / clarification
    missing_fields: List[str]          # stable field IDs (NOT raw questions)
    missing_questions: List[str]       # questions aligned to missing_fields
    needs_clinical_clarification: bool
    clarification_question: Optional[str]


# -------------------------
# Agent
# -------------------------
class ClinicalScopeSafetyAgent:
    """
    Agent 2: Clinical Scope & Safety Agent

    Responsibilities:
    - Normalize medical procedure name (via CTSS tool)
    - Reject unsafe / illegal / non-medical requests
    - Maintain a clinical_profile of what user already provided
    - Ask ONLY unresolved clinically critical clarifications
    """

    def __init__(self, llm):
        self.llm = llm
        self.checkpointer = MemorySaver()
        self.graph = self._build()

    # -------------------------
    # Helpers
    # -------------------------
    def _last_user_text(self, messages: list) -> str:
        for m in reversed(messages or []):
            if isinstance(m, BaseMessage) and m.type == "human":
                return m.content or ""
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content") or ""
        return ""

    def _safe_json_load(self, raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        try:
            return json.loads(raw)
        except Exception:
            s = raw.find("{")
            e = raw.rfind("}")
            if s == -1 or e == -1:
                raise ValueError(f"Invalid JSON from LLM: {raw}")
            return json.loads(raw[s : e + 1])

    def _messages_to_json(self, messages: list) -> List[Dict[str, Any]]:
        out = []
        for m in messages or []:
            if isinstance(m, BaseMessage):
                role = "user" if m.type == "human" else "assistant"
                out.append({"role": role, "content": m.content})
            elif isinstance(m, dict):
                out.append(m)
        return out

    def _ensure_profile(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        constraints.setdefault("clinical_profile", {})
        prof = constraints["clinical_profile"]
        if not isinstance(prof, dict):
            prof = {}
            constraints["clinical_profile"] = prof
        return prof

    def _is_filled(self, v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str) and v.strip() == "":
            return False
        if isinstance(v, (list, dict)) and len(v) == 0:
            return False
        return True

    def _filter_missing_already_known(
        self,
        missing_fields: List[str],
        missing_questions: List[str],
        profile: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        filtered_fields: List[str] = []
        filtered_questions: List[str] = []

        for i, fid in enumerate(missing_fields or []):
            already = fid in (profile or {}) and self._is_filled(profile.get(fid))
            if already:
                continue
            filtered_fields.append(fid)
            if i < len(missing_questions):
                filtered_questions.append(missing_questions[i])

        return filtered_fields, filtered_questions

    # -------------------------
    # Init
    # -------------------------
    def _init_state(self, state: ClinicalState) -> ClinicalState:
        state.setdefault("procedure_normalized", None)
        state.setdefault("procedure_category", None)
        state.setdefault("is_procedure_safe", True)
        state.setdefault("safety_flags", [])
        state.setdefault("missing_fields", [])
        state.setdefault("missing_questions", [])
        state.setdefault("needs_clinical_clarification", False)
        state.setdefault("clarification_question", None)
        state.setdefault("constraints", {})
        return state

    # -------------------------
    # Node: normalize procedure via CTSS tool
    # -------------------------
    def normalize_procedure_node(self, state: ClinicalState, config: RunnableConfig) -> ClinicalState:
        constraints = state.get("constraints") or {}
        procedure = (constraints.get("procedure") or "").strip()
        if not procedure:
            return {"constraints": constraints}

        # IMPORTANT: lookup_procedure is a StructuredTool; call it via .invoke()
        api_result = lookup_procedure.invoke(
            {"term": procedure, "max_results": 5, "timeout_s": 6.0}
        )

        best = (api_result or {}).get("best")
        normalized = None
        if isinstance(best, dict):
            normalized = best.get("consumer_name") or best.get("primary_name")

        constraints["procedure_normalized"] = normalized
        constraints["procedure_candidates"] = (api_result or {}).get("candidates", [])

        return {
            "constraints": constraints,
            "procedure_normalized": normalized,
        }

    # -------------------------
    # Node: extract clinical profile (what user already said)
    # -------------------------
    def extract_profile_node(self, state: ClinicalState, config: RunnableConfig) -> ClinicalState:
        constraints = state.get("constraints") or {}
        profile = self._ensure_profile(constraints)

        user_text = self._last_user_text(state.get("messages", []))
        if not user_text:
            return {"constraints": constraints}

        prompt = (
            "You are a clinical intake extractor.\n"
            "Given the user's latest message, extract any clearly stated clinical facts.\n"
            "Return ONLY valid JSON with ONLY the keys below when present.\n\n"
            "Allowed keys:\n"
            "{\n"
            '  "age": string|null,\n'
            '  "diagnosis_or_reason": string|null,\n'
            '  "duration_severity": string|null,\n'
            '  "other_conditions": string|null,\n'
            '  "daily_activities": string|null,\n'
            '  "medications": string|null,\n'
            '  "allergies": string|null,\n'
            '  "prior_surgeries": string|null,\n'
            '  "imaging_or_labs": string|null\n'
            "}\n\n"
            "Rules:\n"
            "- Only extract what the user explicitly states. Do not infer.\n"
            "- Keep values short.\n\n"
            f"User message: {user_text}\n"
        )

        raw = self.llm.invoke(prompt, config=config).content
        data = self._safe_json_load(raw)

        # merge into profile (do not overwrite filled fields)
        for k, v in (data or {}).items():
            if k not in profile or not self._is_filled(profile.get(k)):
                if self._is_filled(v):
                    profile[k] = v

        constraints["clinical_profile"] = profile
        return {"constraints": constraints}

    # -------------------------
    # Node: clinical scope + safety + missing fields
    # -------------------------
    def clinical_scope_node(self, state: ClinicalState, config: RunnableConfig) -> ClinicalState:
        constraints = state.get("constraints") or {}
        profile = self._ensure_profile(constraints)

        procedure = constraints.get("procedure_normalized") or constraints.get("procedure")
        if not procedure:
            return {"constraints": constraints}

        prompt = (
            "You are a medical safety & clinical scope expert.\n\n"
            "Analyze the requested medical procedure and return ONLY valid JSON.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "procedure_category": string|null,\n'
            '  "is_procedure_safe": boolean,\n'
            '  "safety_flags": [string],\n'
            '  "missing_fields": [string],\n'
            '  "missing_questions": [string]\n'
            "}\n\n"
            "Rules:\n"
            "- If illegal, non-medical, or clearly unsafe/unethical: set is_procedure_safe=false and explain in safety_flags.\n"
            "- Determine the minimal clinically critical intake needed to assess basic suitability/contraindications.\n"
            "- Use these STANDARD missing field IDs when relevant:\n"
            '  ["age","diagnosis_or_reason","duration_severity","other_conditions","daily_activities","medications","allergies","prior_surgeries","imaging_or_labs"]\n'
            "- missing_fields must be ONLY from that list.\n"
            "- missing_questions must align 1:1 with missing_fields.\n"
            "- Do NOT include a field if it is already present in this profile:\n"
            f"  profile={json.dumps(profile, ensure_ascii=False)}\n"
            "- Keep missing_fields to max 5.\n\n"
            f"Procedure: {procedure}\n"
        )

        raw = self.llm.invoke(prompt, config=config).content
        data = self._safe_json_load(raw)

        missing_fields = data.get("missing_fields") or []
        missing_questions = data.get("missing_questions") or []

        # sanitize types
        if not isinstance(missing_fields, list):
            missing_fields = []
        if not isinstance(missing_questions, list):
            missing_questions = []

        # force 1:1 alignment
        if len(missing_questions) != len(missing_fields):
            missing_questions = missing_questions[: len(missing_fields)]
            while len(missing_questions) < len(missing_fields):
                missing_questions.append(f"Please provide: {missing_fields[len(missing_questions)]}.")

        # HARD FILTER (prevents repeats even if model re-asks)
        missing_fields, missing_questions = self._filter_missing_already_known(
            missing_fields=missing_fields,
            missing_questions=missing_questions,
            profile=profile,
        )

        needs = bool(missing_fields)

        clarification_question = None
        if needs:
            # show questions (not IDs)
            clarification_question = "To assess medical suitability, I need:\n- " + "\n- ".join(missing_questions)

        assistant_text = (
            clarification_question
            if needs
            else "Clinically: the procedure appears valid to proceed to cost/clinic matching. No additional intake needed right now."
        )

        return {
            "constraints": constraints,
            "messages": [{"role": "assistant", "content": assistant_text}],
            "procedure_normalized": constraints.get("procedure_normalized"),
            "procedure_category": data.get("procedure_category"),
            "is_procedure_safe": bool(data.get("is_procedure_safe", True)),
            "safety_flags": data.get("safety_flags") or [],
            "missing_fields": missing_fields,
            "missing_questions": missing_questions,
            "needs_clinical_clarification": needs,
            "clarification_question": clarification_question,
        }

    # -------------------------
    # Graph
    # -------------------------
    def _build(self):
        builder = StateGraph(ClinicalState)

        builder.add_node("init", self._init_state)
        builder.add_node("normalize_procedure", self.normalize_procedure_node)
        builder.add_node("extract_profile", self.extract_profile_node)
        builder.add_node("clinical_scope", self.clinical_scope_node)

        builder.add_edge(START, "init")
        builder.add_edge("init", "normalize_procedure")
        builder.add_edge("normalize_procedure", "extract_profile")
        builder.add_edge("extract_profile", "clinical_scope")
        builder.add_edge("clinical_scope", END)

        return builder.compile(checkpointer=self.checkpointer)

    # -------------------------
    # Public invoke
    # -------------------------
    def invoke(
        self,
        constraints: Dict[str, Any],
        thread_id: str,
        user_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        state_in: ClinicalState = {"constraints": constraints or {}}

        if user_text:
            state_in["messages"] = [{"role": "user", "content": user_text}]

        out: ClinicalState = self.graph.invoke(
            state_in,
            config={"configurable": {"thread_id": thread_id}},
        )

        return {
            "procedure_normalized": out.get("procedure_normalized"),
            "procedure_category": out.get("procedure_category"),
            "is_procedure_safe": out.get("is_procedure_safe"),
            "safety_flags": out.get("safety_flags") or [],
            "missing_fields": out.get("missing_fields") or [],
            "missing_questions": out.get("missing_questions") or [],
            "needs_clinical_clarification": bool(out.get("needs_clinical_clarification")),
            "clarification_question": out.get("clarification_question"),
            "messages": self._messages_to_json(out.get("messages", [])),
            "clinical_profile": (out.get("constraints") or {}).get("clinical_profile", {}),
        }
