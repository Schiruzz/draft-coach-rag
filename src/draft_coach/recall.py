"""Retrieval metrics on the evaluation set. A chunk finds a gold interval if it is from the same source
and contains at least half of the interval (touching it for a few seconds is not enough)."""
from math import comb, sqrt

Z_95 = 1.96  # 95% confidence
USEFUL = ("yes", "partial")  # judge verdicts that bring something to the answer


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


def useful_count(verdicts: list[str | None]) -> int:
    """Chunks judged yes or partial (None, a missing verdict, counts as not useful)."""
    return sum(v in USEFUL for v in verdicts)


def normalized_ctx(useful_read: int, useful_available: int, read: int) -> float | None:
    """Useful chunks read over the most that could be read: min(chunks read, useful chunks available deeper in the
    list). A question with 2 useful passages in the corpus scores 1.0 with both in its 5 chunks, not 0.4.
    None when nothing useful is available: that question says nothing about the context."""
    best = min(read, useful_available)
    return useful_read / best if best else None


def overlapping(chunks: list[dict]) -> int:
    """Chunks that share time with an earlier chunk of the same source: places spent on almost the same text."""
    return sum(any(e["source_id"] == c["source_id"] and min(e["end_s"], c["end_s"]) > max(e["start_s"], c["start_s"])
                   for e in chunks[:i]) for i, c in enumerate(chunks))

def wilson_interval(hits: int, n: int) -> tuple[float, float]:
    """95% confidence interval of a proportion (Wilson): it stays honest near 0% and 100% and with few questions."""
    if n == 0:
        return 0.0, 1.0
    p, z2 = hits / n, Z_95 ** 2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z_95 * sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return centre - half, centre + half