"""Backup request: a slow first call is overtaken by the copy; a fast first call sends no copy."""
import time

import draft_coach.gemini as gemini


def fake_calls(monkeypatch, delays: list[float]):
    """Each call sleeps the next delay, then answers with its own number."""
    calls = []
    def fake(client, model, prompt, schema, level, label=""):
        n = len(calls)
        calls.append(label)
        time.sleep(delays[n])
        return f"answer {n}"
    monkeypatch.setattr(gemini, "generate_json", fake)
    return calls


def test_slow_first_call_is_overtaken_by_the_backup(monkeypatch):
    calls = fake_calls(monkeypatch, [1.0, 0.0])
    start = time.monotonic()
    answer = gemini.generate_json_hedged(None, "m", "p", None, "low", hedge_after_s=0.05, label="s")
    assert answer == ("answer 1", True, True) and time.monotonic() - start < 0.5
    assert calls == ["s", "s (backup)"]


def test_fast_first_call_sends_no_backup(monkeypatch):
    calls = fake_calls(monkeypatch, [0.0, 0.0])
    assert gemini.generate_json_hedged(None, "m", "p", None, "low", hedge_after_s=0.5) == ("answer 0", False, False)
    assert len(calls) == 1