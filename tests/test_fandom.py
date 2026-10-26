"""Fandom API wrapper, with the network replaced by a fake response."""
import pytest

from draft_coach import fandom


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_returns_wikitext_and_follows_redirects(monkeypatch):
    sent = {}

    def fake_get(url, params, headers, timeout):
        sent.update(params)
        return FakeResponse({"parse": {"wikitext": "{{Brawler Infobox}}"}})

    monkeypatch.setattr(fandom.requests, "get", fake_get)
    assert fandom.fetch_wikitext("Jae-Yong") == "{{Brawler Infobox}}"
    assert sent["redirects"] == 1 and sent["page"] == "Jae-Yong"


def test_missing_page_raises(monkeypatch):
    monkeypatch.setattr(fandom.requests, "get", lambda *a, **k: FakeResponse({"error": {"code": "missingtitle"}}))
    with pytest.raises(LookupError):
        fandom.fetch_wikitext("Nobody")



def test_page_image_url(monkeypatch):
    data = {"query": {"pages": [{"title": "Hot Potato", "original": {"source": "https://static/x/Hot_Potato-Map.png"}}]}}
    monkeypatch.setattr(fandom.requests, "get", lambda *a, **k: FakeResponse(data))
    assert fandom.fetch_page_image("Hot Potato") == "https://static/x/Hot_Potato-Map.png"


def test_page_without_image_raises(monkeypatch):
    data = {"query": {"pages": [{"title": "Nobody", "missing": True}]}}
    monkeypatch.setattr(fandom.requests, "get", lambda *a, **k: FakeResponse(data))
    with pytest.raises(LookupError):
        fandom.fetch_page_image("Nobody")
        