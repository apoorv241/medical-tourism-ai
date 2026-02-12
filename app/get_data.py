import os
import re
import requests
from typing import Any, Dict, List, Optional, Tuple

DATA_GOV_BASE = "https://api.gsa.gov/technology/datagov/v3/action"
DEFAULT_API_KEY = os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")  # DEMO_KEY works for sample/testing

session = requests.Session()
session.headers.update({"x-api-key": DEFAULT_API_KEY})


def datagov_package_search(query: str, rows: int = 10, start: int = 0) -> Dict[str, Any]:
    """
    Search the Data.gov catalog (CKAN) for datasets matching query.
    """
    url = f"{DATA_GOV_BASE}/package_search"
    params = {"q": query, "rows": rows, "start": start}
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_socrata_dataset_id(url: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Extract (domain, dataset_id) from a Socrata URL if present.
    Example dataset_id: ypvj-urp2
    """
    if not url or not isinstance(url, str):
        return None

    m = re.search(
        r"https?://([^/]+)/.*?/([a-z0-9]{4}-[a-z0-9]{4})(?:[/?#]|$)",
        url,
        re.I,
    )
    if not m:
        return None
    return m.group(1), m.group(2)



def socrata_query(domain: str, dataset_id: str, limit: int = 1000, offset: int = 0,
                 where: Optional[str] = None, select: Optional[str] = None,
                 order: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Query a Socrata dataset using SODA (no key required for many public datasets).
    Uses $limit/$offset for paging. Defaults to 1000 rows per request (typical default).
    """
    url = f"https://{domain}/resource/{dataset_id}.json"
    params = {"$limit": limit, "$offset": offset}
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select
    if order:
        params["$order"] = order

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def pick_best_resource(package: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pick a resource that looks easiest to consume (JSON/CSV/API).
    """
    resources = package.get("resources", []) or []
    preferred_formats = {"JSON", "CSV"}
    for res in resources:
        fmt = (res.get("format") or "").upper()
        if fmt in preferred_formats and res.get("url"):
            return res
    # fallback: anything with a url
    for res in resources:
        if res.get("url"):
            return res
    return None


def demo_flow(search_term: str = "telemedicine", rows: int = 5) -> None:
    """
    1) Search Data.gov
    2) For each dataset found, try to:
       - show title
       - find a good resource
       - if it’s a healthdata.gov Socrata page, pull sample rows via SODA
    """
    result = datagov_package_search(search_term, rows=rows)
    if not result.get("success"):
        raise RuntimeError(f"Data.gov search failed: {result}")

    packages = result["result"]["results"]
    print(f"\nFound {len(packages)} datasets for: {search_term}\n")

    for i, pkg in enumerate(packages, 1):
        title = pkg.get("title", "(no title)")
        notes = (pkg.get("notes") or "").strip().replace("\n", " ")
        print(f"{i}. {title}")
        if notes:
            print(f"   {notes[:140]}{'...' if len(notes) > 140 else ''}")

        res = pick_best_resource(pkg)
        if res:
            print(f"   Resource: {res.get('format')} | {res.get('url')}")
            soc = extract_socrata_dataset_id(res.get("url"))
            if not soc:
                soc = extract_socrata_dataset_id(pkg.get("url"))
            if soc:
                domain, dataset_id = soc
                print(f"   Socrata detected: {domain} / {dataset_id}")
                try:
                    rows_data = socrata_query(domain, dataset_id, limit=5, offset=0)
                    print(f"   Sample rows (5): keys={list(rows_data[0].keys()) if rows_data else 'none'}")
                except Exception as e:
                    print(f"   (Could not pull sample rows) {e}")
        print()


if __name__ == "__main__":
    # Change the query terms depending on your medical tourism focus:
    # "hospital", "medicare", "quality measures", "emergency department", "wait times", "telemedicine"
    demo_flow(search_term="hospital quality", rows=5)
