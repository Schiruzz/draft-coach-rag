"""Retrieval metrics on the evaluation set, chunking methods side by side.
Run: python scripts/measure_recall.py                        (all sources)
     python scripts/measure_recall.py --formats guide counters  (only sources of these formats)
The review of disagreements goes to reports/recall_review.md."""
import argparse
from pathlib import Path
from statistics import median

from google import genai
from google.genai import types
from tqdm import tqdm

from draft_coach.chunks import CHAPTER_METHOD, METHOD
from draft_coach.eval_set import load_eval
from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers
from draft_coach.recall import coverage, first_hit, sign_test_p
from draft_coach.retrieval import Retriever, minute
from draft_coach.scenarios import scenario_question
from draft_coach.sources import FORMATS, load_sources, sources_with_formats

METHODS = [METHOD, CHAPTER_METHOD]
TOP_K = 10  # the answer reads 5 chunks; @10 tells whether more chunks or a reranker would help
REPORT = Path("reports/recall_review.md")

all_questions = load_eval(Path("data/eval/v1.yaml"))
questions = [q for q in all_questions if q["type"] != "abstain"]  # abstain has no gold
abstain = [q for q in all_questions if q["type"] == "abstain"]
scenarios = [q for q in questions if "scenario" in q]
names = {b["id"]: b["en"] for b in load_brawlers(Path("data/brawlers.yaml"))}
maps = {m["id"]: m for m in load_maps(Path("data/maps.yaml"))}
sources = load_sources(Path("data/sources.yaml"))
with_chapters = {s["id"] for s in sources if s.get("chapters")}

parser = argparse.ArgumentParser()
parser.add_argument("--formats", nargs="+", choices=sorted(FORMATS), default=sorted(FORMATS))
searched = sources_with_formats(sources, set(parser.parse_args().formats))  # same filter for both methods
print(f"searching {len(searched)} sources: {', '.join(sorted(parser.parse_args().formats))}")


def question_text(q: dict) -> str:
    if "scenario" in q:
        return scenario_question(q["scenario"], names, maps, q.get("candidate"))
    return q["question"]


def groups(q: dict) -> list[str]:
    """The rows of the table this question counts in."""
    gold_sources = {g["source"] for g in q["gold"]}
    chapters = ("chapters" if gold_sources <= with_chapters
                else "no chapters" if not gold_sources & with_chapters else "mixed")
    return ["all", f"type {q['type']}", f"lang {q.get('lang', 'scenario')}", f"gold in {chapters}"]


def at5(rank: int | None) -> bool:
    return rank is not None and rank <= 5


client = genai.Client(http_options=types.HttpOptions(timeout=30_000))
runs = {}  # method -> {"ranks", "cov5", "top_score", "top_chunk", "reference_ranks", "abstain_scores"}
for method in METHODS:
    retriever = Retriever(client, method=method)
    run = runs[method] = {"ranks": {}, "cov5": {}, "top_score": {}, "top_chunk": {}, "reference_ranks": {}}
    for q in tqdm(questions, desc=f"{method} questions"):
        chunks = retriever.search(question_text(q), top_k=TOP_K, sources=searched)
        run["ranks"][q["id"]] = first_hit(chunks, q["gold"])
        run["cov5"][q["id"]] = coverage(chunks[:5], q["gold"])
        run["top_score"][q["id"]], run["top_chunk"][q["id"]] = chunks[0]["score"], chunks[0]
    # Upper bound for scenarios: the concept written by us as the query (index problem or query problem?)
    for q in tqdm(scenarios, desc=f"{method} reference"):
        run["reference_ranks"][q["id"]] = first_hit(retriever.search(q["reference"], top_k=TOP_K, sources=searched),
                                                    q["gold"])
    run["abstain_scores"] = [retriever.search(q["question"], top_k=1, sources=searched)[0]["score"]
                             for q in tqdm(abstain, desc=f"{method} abstain")]

# 1. Table: recall@5, recall@10, MRR, coverage@5 per group
order = ["all", "type", "lang", "gold"]  # rows grouped by kind, in this order
rows = sorted(dict.fromkeys(g for q in questions for g in groups(q)), key=lambda r: (order.index(r.split()[0]), r))
print(f"\n{'':<28}" + "".join(f"{m:^28}" for m in METHODS))
print(f"{'group':<24}{'n':>4}" + f"{'R@5':>7}{'R@10':>7}{'MRR':>7}{'Cov@5':>7}" * len(METHODS))
for row in rows:
    ids = [q["id"] for q in questions if row in groups(q)]
    cells = ""
    for m in METHODS:
        ranks = [runs[m]["ranks"][i] for i in ids]
        cells += (f"{sum(map(at5, ranks)) / len(ids):>7.0%}{sum(r is not None for r in ranks) / len(ids):>7.0%}"
                  f"{sum(1 / r for r in ranks if r) / len(ids):>7.2f}"
                  f"{sum(runs[m]['cov5'][i] for i in ids) / len(ids):>7.0%}")
    print(f"{row:<24}{len(ids):>4}{cells}")

# 2. Paired comparison at @5, with the sign test
a, b = METHODS
found = {m: {i for i, r in runs[m]["ranks"].items() if at5(r)} for m in METHODS}
gains, losses = len(found[b] - found[a]), len(found[a] - found[b])
print(f"\n{b} gains {gains} and loses {losses} against {a}: sign test p = {sign_test_p(gains, losses):.2f}"
      f"  (below 0.05: a real difference)")

# 3. Scenarios: rendered question against the reference as query
print(f"\n{'scenarios R@5':<28}" + "".join(f"{m:^28}" for m in METHODS))
for label, key in [("rendered question", "ranks"), ("reference as query", "reference_ranks")]:
    print(f"{label:<28}" + "".join(
        f"{sum(at5(runs[m][key][q['id']]) for q in scenarios) / len(scenarios):^28.0%}" for m in METHODS))

# 4. Top score: can a threshold separate abstain questions from answerable ones?
print("\nTop score of the first chunk (a threshold for abstention?)")
for m in METHODS:
    answerable, refused = list(runs[m]["top_score"].values()), runs[m]["abstain_scores"]
    below = sum(s <= max(refused) for s in answerable)
    print(f"{m:<28}answerable median {median(answerable):.3f} (min {min(answerable):.3f}), "
          f"abstain median {median(refused):.3f} (max {max(refused):.3f}); "
          f"answerable at or below the abstain max: {below}/{len(answerable)}")

# 5. Review of the disagreements: is the first chunk of the method that missed useful anyway?
lines = ["# Recall review: questions where the two methods disagree at @5",
         "For each one: is the first chunk of the method that missed useful for the reference? (useful / not useful)", ""]
for q in questions:
    if (q["id"] in found[a]) == (q["id"] in found[b]):
        continue
    missed = b if q["id"] in found[a] else a
    top = runs[missed]["top_chunk"][q["id"]]
    lines += [f"## {q['id']} ({q['type']}): found by {a if missed == b else b}, missed by {missed}",
              f"Question: {question_text(q)}", f"Reference: {q['reference']}",
              f"Gold: {', '.join(f'{g['source']} {minute(g['start_s'])}-{minute(g['end_s'])}' for g in q['gold'])}",
              f"First chunk of {missed}: `{top['source_id']}` [{minute(top['start_s'])}]({top['url']})",
              f"> {top['text']}", "", "Verdict: ", ""]
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReview: {REPORT}")