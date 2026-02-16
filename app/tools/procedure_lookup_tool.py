# app/tools/procedure_lookup_tool.py
from __future__ import annotations

from dataclasses import dataclass
from langchain.tools import tool
from typing import Any, Dict, List, Optional
import requests
import logging
from flask import current_app, has_app_context


CTSS_PROCEDURES_SEARCH_URL = "https://clinicaltables.nlm.nih.gov/api/procedures/v3/search"


@dataclass
class ProcedureMatch:
    key_id: str
    primary_name: str
    consumer_name: Optional[str] = None
    term_icd9_code: Optional[str] = None
    term_icd9_text: Optional[str] = None
    synonyms: Optional[List[str]] = None
    info_link_data: Optional[List[List[str]]] = None

def get_logger():
    """
    Returns Flask app logger if available, otherwise a module-level logger.
    Safe to call from LangChain tools.
    """
    if has_app_context():
        return current_app.logger
    return logging.getLogger(__name__)


@tool
def lookup_procedure(term: str, max_results: int = 5, timeout_s: float = 6.0) -> Dict[str, Any]:
    """
    Calls NIH Clinical Table Search Service: Major Surgeries & Implants (procedures) API.
    Returns a normalized best-match plus top candidates.
    """

    logger = get_logger()

    q = (term or "").strip()
    if not q:
        logger.info("lookup_procedure: empty term received")
        return {"query": term, "best": None, "candidates": []}

    params = {
        "terms": q,
        "maxList": str(max_results),
        "df": "primary_name,consumer_name",
        "ef": "term_icd9_code,term_icd9_text,synonyms,info_link_data",
        "sf": "consumer_name,primary_name,word_synonyms,synonyms",
        "cf": "key_id",
    }

    logger.info(
        "lookup_procedure REQUEST | term=%s max_results=%s timeout=%ss",
        q,
        max_results,
        timeout_s,
    )

    try:
        r = requests.get(
            CTSS_PROCEDURES_SEARCH_URL,
            params=params,
            timeout=timeout_s,
        )
        r.raise_for_status()
    except Exception as e:
        logger.exception(
            "lookup_procedure ERROR | term=%s url=%s",
            q,
            CTSS_PROCEDURES_SEARCH_URL,
        )
        raise

    data = r.json()

    total = int(data[0]) if len(data) > 0 else 0
    codes = data[1] if len(data) > 1 else []
    extra = data[2] if len(data) > 2 else None
    display_rows = data[3] if len(data) > 3 else []

    candidates: List[Dict[str, Any]] = []
    for i, code in enumerate(codes):
        display = display_rows[i] if i < len(display_rows) else []
        primary_name = display[0] if len(display) > 0 else None
        consumer_name = display[1] if len(display) > 1 else None

        term_icd9_code = (extra.get("term_icd9_code", []) if isinstance(extra, dict) else [])
        term_icd9_text = (extra.get("term_icd9_text", []) if isinstance(extra, dict) else [])
        synonyms = (extra.get("synonyms", []) if isinstance(extra, dict) else [])
        info_link_data = (extra.get("info_link_data", []) if isinstance(extra, dict) else [])

        candidates.append(
            {
                "key_id": code,
                "primary_name": primary_name,
                "consumer_name": consumer_name,
                "term_icd9_code": term_icd9_code[i] if i < len(term_icd9_code) else None,
                "term_icd9_text": term_icd9_text[i] if i < len(term_icd9_text) else None,
                "synonyms": synonyms[i] if i < len(synonyms) else None,
                "info_link_data": info_link_data[i] if i < len(info_link_data) else None,
            }
        )

    best = candidates[0] if candidates else None

    logger.info(
        "lookup_procedure RESPONSE | term=%s total=%d best=%s",
        q,
        total,
        best.get("primary_name") if best else None,
    )

    return {
        "query": term,
        "total": total,
        "best": best,
        "candidates": candidates,
    }
