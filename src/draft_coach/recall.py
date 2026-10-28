"""Retrieval metrics on the evaluation set. A chunk finds a gold interval if it is from the same source
and contains at least half of the interval (touching it for a few seconds is not enough)."""
from math import comb


def hits_gold(chunk: dict, gold: dict) -> bool:
    if chunk["source_id"] != gold["source"]:
        return False
    overlap = min(chunk["end_s"], gold["end_s"]) - max(chunk["start_s"], gold["start_s"])
    return overlap >= (gold["end_s"] - gold["start_s"]) / 2


def first_hit(chunks: list[dict], golds: list[dict]) -> int | None:
    """Rank (1-based) of the first chunk that finds any gold interval; None if no chunk does."""
    for rank, chunk in enumerate(chunks, 1):
        if any(hits_gold(chunk, g) for g in golds):
            return rank
    return None


def coverage(chunks: list[dict], golds: list[dict]) -> float:
    """Share of the gold intervals found by at least one of the chunks."""
    return sum(any(hits_gold(c, g) for c in chunks) for g in golds) / len(golds)


def sign_test_p(gains: int, losses: int) -> float:
    """Two-sided sign test: probability of a split at least this uneven if both methods were equal."""
    n, k = gains + losses, min(gains, losses)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)