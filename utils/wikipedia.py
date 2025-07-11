import requests
import re
from typing import Optional

def _clean_wikipedia_html(html: str) -> str:
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    html = re.sub(r'<sup[^>]*>.*?</sup>', '', html, flags=re.DOTALL)
    html = re.sub(r'<a [^>]+>(.*?)</a>', r'\1', html, flags=re.DOTALL)
    clean_text = re.sub('<[^<]+?>', '', html)
    clean_text = re.sub(r'\[[^\]]*\]', '', clean_text)
    clean_text = '\n'.join([line for line in clean_text.splitlines() if not line.strip().startswith('^')])
    clean_text = re.sub(r'/\*.*?\*/', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'\{[^\}]*\}', '', clean_text, flags=re.DOTALL)
    clean_text = '\n'.join([line for line in clean_text.splitlines() if line.strip()])
    return clean_text.strip()

def get_wikipedia_summary(scientific_name: str) -> Optional[dict]:
    wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{scientific_name.replace(' ', '_')}"
    try:
        resp = requests.get(wiki_url)
        if resp.status_code == 200:
            data = resp.json()
            # Add the canonical Wikipedia page URL if available
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
            if page_url:
                data["page_url"] = page_url
            return data
    except Exception:
        pass
    return None
