from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool


@tool
def search_hospitals_tool(
    procedure: str,
    country: str,
    city: Optional[str] = None,
) -> Dict[str, Any]:
    """Return hospitals. City optional."""
    # STUB for now (replace internals later)
    city_label = city or "top cities"
    return {
        "query": {"procedure": procedure, "country": country, "city": city},
        "mode": "city" if city else "country",
        "cities_used": [city] if city else ["CityA", "CityB", "CityC"],
        "results": [
            {
                "name": "Example Hospital 1",
                "address": f"{city_label}, {country}",
                "rating": 4.6,
                "approx_price_usd": 1800,
                "why": "Good ratings",
            },
            {
                "name": "Example Hospital 2",
                "address": f"{city_label}, {country}",
                "rating": 4.4,
                "approx_price_usd": 1600,
                "why": "Budget friendly",
            },
        ],
    }


@tool
def pick_best_hospitals_tool(
    hospitals: List[Dict[str, Any]],
    budget_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Rank hospitals by budget-fit + rating."""
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