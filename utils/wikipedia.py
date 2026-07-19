import requests
from typing import Optional


def _fetch_summary(title: str) -> Optional[dict]:
    wiki_url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        resp = requests.get(wiki_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            content_urls = data.get("content_urls")
            if isinstance(content_urls, dict):
                desktop = content_urls.get("desktop")
                if isinstance(desktop, dict):
                    page_url = desktop.get("page")
                    if page_url:
                        data["page_url"] = page_url
            return data
    except Exception:
        pass
    return None


def get_wikipedia_summary(scientific_name: str) -> Optional[dict]:
    if not scientific_name:
        return None
    # 1. Try full scientific name
    data = _fetch_summary(scientific_name)
    if data:
        return data

    # 2. Fallback: try querying just the genus (first word)
    parts = scientific_name.strip().split()
    if len(parts) > 1:
        genus = parts[0]
        data = _fetch_summary(genus)
        if data:
            return data

    return None
