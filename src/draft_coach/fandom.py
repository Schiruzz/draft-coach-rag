"""Download pages from the Brawl Stars Fandom wiki through its API (?action=raw is blocked with 403)."""
import requests

API = "https://brawlstars.fandom.com/api.php"
HEADERS = {"User-Agent": "draft-coach-rag/0.1 (non-commercial, educational)"}  # wikis ask clients to identify


def fetch_wikitext(title: str) -> str:
    """Raw wikitext of the current version of a page; redirects are followed (e.g. Jae-Yong -> Jae-yong)."""
    params = {"action": "parse", "page": title, "prop": "wikitext", "redirects": 1, "format": "json", "formatversion": 2}
    response = requests.get(API, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()
    if "parse" not in data:
        raise LookupError(f"page not found on Fandom: {title}")
    return data["parse"]["wikitext"]