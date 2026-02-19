from __future__ import annotations

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import tool


# -------------------------------------------------
# Logging setup (DEBUG ON)
# -------------------------------------------------
logger = logging.getLogger("hospital_agent")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.DEBUG)


# -----------------------------
# Providers
# -----------------------------
# Google Places Text Search (New): POST https://places.googleapis.com/v1/places:searchText
# Docs: https://developers.google.com/maps/documentation/places/web-service/text-search
#
# Foursquare Places Search: GET https://api.foursquare.com/v3/places/search


def _normalize_hospital_result(
    *,
    name: str,
    address: str,
    rating: Optional[float],
    maps_url: Optional[str] = None,
    website: Optional[str] = None,
    phone: Optional[str] = None,
    raw: Optional[dict] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "address": address,
        "rating": rating,
        "approx_price_usd": None,  # Places APIs don't have procedure pricing
        "why": "Matched from places provider",
        "links": {
            "maps": maps_url,
            "website": website,
        },
        "phone": phone,
        "raw": raw or {},
    }


def _safe_trunc(s: Any, n: int = 2000) -> str:
    try:
        txt = s if isinstance(s, str) else str(s)
    except Exception:
        txt = "<unprintable>"
    return txt[:n]


async def _google_places_search(
    *,
    procedure: str,
    country: str,
    city: Optional[str],
    limit: int = 10,
    timeout_s: float = 15.0,
) -> Dict[str, Any]:
    logger.info("Using provider: GOOGLE PLACES (Text Search New)")

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_MAPS_API_KEY in environment.")

    text_bits = []
    if procedure:
        # Example: "knee replacement hospital"
        text_bits.append(procedure.strip())
    text_bits.append("hospital")
    if city:
        text_bits.append(city.strip())
    if country:
        text_bits.append(country.strip())

    text_query = " ".join(text_bits)

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        # FieldMask is REQUIRED for Text Search (New)
        "X-Goog-FieldMask": ",".join(
            [
                "places.displayName",
                "places.formattedAddress",
                "places.rating",
                "places.userRatingCount",
                "places.googleMapsUri",
                "places.websiteUri",
                "places.nationalPhoneNumber",
                "places.primaryType",
                "places.types",
            ]
        ),
    }

    payload: Dict[str, Any] = {
        "textQuery": text_query,
        "pageSize": max(1, min(limit, 20)),
        # Strongly bias to real hospitals
        "includedType": "hospital",
        "strictTypeFiltering": True,
        "rankPreference": "RELEVANCE",
        "languageCode": "en",
    }

    # DEBUG logs (don’t print api_key)
    safe_headers = dict(headers)
    if "X-Goog-Api-Key" in safe_headers:
        safe_headers["X-Goog-Api-Key"] = "***REDACTED***"

    logger.debug(f"Google API URL: {url}")
    logger.debug(f"Google Request Headers: {safe_headers}")
    logger.debug(f"Google Request Payload: {_safe_trunc(payload, 4000)}")

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, headers=headers, json=payload)

        logger.info(f"Google Response Status: {resp.status_code}")
        logger.debug(f"Google Raw Response (truncated): {_safe_trunc(resp.text, 4000)}")

        resp.raise_for_status()
        data = resp.json()

    places = data.get("places", []) or []

    results: List[Dict[str, Any]] = []
    for p in places[:limit]:
        display = (p.get("displayName") or {}).get("text") or p.get("displayName") or "Unknown"
        addr = p.get("formattedAddress") or ""
        rating = p.get("rating")
        maps_url = p.get("googleMapsUri")
        website = p.get("websiteUri")
        phone = p.get("nationalPhoneNumber")

        results.append(
            _normalize_hospital_result(
                name=str(display),
                address=str(addr),
                rating=float(rating) if rating is not None else None,
                maps_url=str(maps_url) if maps_url else None,
                website=str(website) if website else None,
                phone=str(phone) if phone else None,
                raw=p,
            )
        )

    return {
        "provider": "google_places",
        "query": {"procedure": procedure, "country": country, "city": city, "textQuery": text_query},
        "mode": "city" if city else "country",
        "cities_used": [city] if city else [],
        "results": results,
        "raw": {"nextPageToken": data.get("nextPageToken")},
    }


async def _foursquare_search(
    *,
    procedure: str,
    country: str,
    city: Optional[str],
    limit: int = 10,
    timeout_s: float = 15.0,
) -> Dict[str, Any]:
    logger.info("Using provider: FOURSQUARE PLACES")

    api_key = os.getenv("FOURSQUARE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FOURSQUARE_API_KEY in environment.")

    # Foursquare uses "near" for city text search, or "ll" for coordinates
    near = ", ".join([x for x in [city, country] if x])

    query_terms = []
    if procedure:
        query_terms.append(procedure.strip())
    query_terms.append("hospital")
    query = " ".join(query_terms)

    url = "https://api.foursquare.com/v3/places/search"
    headers = {"Authorization": api_key, "Accept": "application/json"}

    params: Dict[str, Any] = {
        "query": query,
        "limit": max(1, min(limit, 50)),
    }
    if near:
        params["near"] = near

    safe_headers = dict(headers)
    if "Authorization" in safe_headers:
        safe_headers["Authorization"] = "***REDACTED***"

    logger.debug(f"Foursquare API URL: {url}")
    logger.debug(f"Foursquare Request Headers: {safe_headers}")
    logger.debug(f"Foursquare Request Params: {_safe_trunc(params, 4000)}")

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(url, headers=headers, params=params)

        logger.info(f"Foursquare Response Status: {resp.status_code}")
        logger.debug(f"Foursquare Raw Response (truncated): {_safe_trunc(resp.text, 4000)}")

        resp.raise_for_status()
        data = resp.json()

    fsq_results = data.get("results", []) or []

    results: List[Dict[str, Any]] = []
    for r in fsq_results[:limit]:
        name = r.get("name") or "Unknown"
        loc = r.get("location") or {}
        address = loc.get("formatted_address") or ", ".join(
            [x for x in [loc.get("address"), loc.get("locality"), loc.get("country")] if x]
        )

        # Foursquare ratings aren't always present on all plans/data
        rating = r.get("rating")

        results.append(
            _normalize_hospital_result(
                name=str(name),
                address=str(address),
                rating=float(rating) if rating is not None else None,
                maps_url=None,
                website=(r.get("website") or None),
                phone=(r.get("tel") or None),
                raw=r,
            )
        )

    return {
        "provider": "foursquare",
        "query": {"procedure": procedure, "country": country, "city": city, "near": near, "queryText": query},
        "mode": "city" if city else "country",
        "cities_used": [city] if city else [],
        "results": results,
        "raw": {},
    }


# -----------------------------
# Tools (LangChain)
# -----------------------------
@tool
def search_hospitals_tool(
    procedure: str,
    country: str,
    city: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return hospitals using a real API.
    Provider selection:
      - If GOOGLE_MAPS_API_KEY is set -> Google Places Text Search (New)
      - Else if FOURSQUARE_API_KEY is set -> Foursquare Places Search
    """
    logger.info(
        f"Hospital search requested | procedure={procedure!r} | country={country!r} | city={city!r}"
    )

    # Decide provider based on env (simple & practical)
    use_google = bool(os.getenv("GOOGLE_MAPS_API_KEY"))
    use_fsq = bool(os.getenv("FOURSQUARE_API_KEY"))

    if use_google:
        logger.info("Provider selected: GOOGLE (GOOGLE_MAPS_API_KEY present)")
    elif use_fsq:
        logger.info("Provider selected: FOURSQUARE (FOURSQUARE_API_KEY present)")
    else:
        logger.warning("No provider keys found. Set GOOGLE_MAPS_API_KEY or FOURSQUARE_API_KEY.")
        return {
            "error": "No provider keys found. Set GOOGLE_MAPS_API_KEY or FOURSQUARE_API_KEY.",
            "query": {"procedure": procedure, "country": country, "city": city},
            "results": [],
        }

    # Run async without forcing your whole stack to be async
    try:
        # NOTE: asyncio.run() will fail if there's already a running event loop
        # (e.g., in some async environments). For Flask sync routes it’s usually fine.
        if use_google:
            out = asyncio.run(
                _google_places_search(procedure=procedure, country=country, city=city, limit=10)
            )
        else:
            out = asyncio.run(
                _foursquare_search(procedure=procedure, country=country, city=city, limit=10)
            )

        logger.debug(f"Tool output summary: provider={out.get('provider')} results={len(out.get('results') or [])}")
        return out

    except RuntimeError as e:
        # This catches missing env keys AND "asyncio.run() cannot be called from a running event loop"
        logger.exception("RuntimeError in search_hospitals_tool")
        return {"error": str(e), "query": {"procedure": procedure, "country": country, "city": city}, "results": []}

    except httpx.HTTPStatusError as e:
        logger.exception("Provider HTTPStatusError in search_hospitals_tool")
        body = ""
        try:
            body = e.response.text
        except Exception:
            body = ""
        return {
            "error": f"Provider HTTP error: {e.response.status_code}",
            "details": _safe_trunc(body, 2000),
            "query": {"procedure": procedure, "country": country, "city": city},
            "results": [],
        }

    except Exception as e:
        logger.exception("Unexpected error in search_hospitals_tool")
        return {"error": f"Unexpected error: {e}", "query": {"procedure": procedure, "country": country, "city": city}, "results": []}


@tool
def pick_best_hospitals_tool(
    hospitals: List[Dict[str, Any]],
    budget_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Rank hospitals by (budget-fit if price exists) + rating."""
    logger.debug(
        f"pick_best_hospitals_tool called | hospitals={len(hospitals or [])} | budget_max={budget_max}"
    )

    if not hospitals:
        return {"ranked": []}

    def price(x: Dict[str, Any]) -> float:
        p = x.get("approx_price_usd")
        return float(p) if isinstance(p, (int, float)) else 10**9  # unknown price -> worst for budget

    def rating(x: Dict[str, Any]) -> float:
        r = x.get("rating")
        return float(r) if isinstance(r, (int, float)) else 0.0

    ranked = sorted(
        hospitals,
        key=lambda x: (
            0 if (budget_max is None or price(x) <= budget_max) else 1,  # budget-fit first
            -rating(x),  # higher rating first
            price(x),  # cheaper first (when present)
        ),
    )

    logger.debug("Ranking complete. Returning top 5.")
    return {"ranked": ranked[:5]}