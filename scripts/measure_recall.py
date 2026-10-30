"""Retrieval metrics on the evaluation set, chunking methods side by side: gold intervals and the retrieval judge.
Run: python scripts/measure_recall.py                        (all sources)
     python scripts/measure_recall.py --formats guide counters  (only sources of these formats)
Judgments are cached in MongoDB (collection judgments): the first run asks Gemini, the next ones read them.
Reports: reports/recall_review.md (disagreements), reports/recall_misses.md (misses),
reports/judge_check.md (new verdicts to check by hand)."""
import argparse
import random
from pathlib import Path
from statistics import median

import yaml
from google import genai
from google.genai import types
from pymongo import MongoClient
from tqdm import tqdm

from draft_coach.chunks import METHOD, WINDOW_METHOD
from draft_coach.config import MONGO_URI
from draft_coach.eval_set import load_eval
from draft_coach.judge import MODELS, PROMPT_VERSION, PromptBuilder, judge_cached, judgment_id
from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers
from draft_coach.recall import coverage, first_hit, sign_test_p, wilson_interval
from draft_coach.retrieval import Retriever, minute
from draft_coach.sources import FORMATS, load_sources, sources_with_formats

METHODS = [WINDOW_METHOD, METHOD]  # the method in use against A (30/10: chapter60-plain-v1 won)
TOP_K = 10  # the answer reads 5 chunks; @10 tells whether more chunks or a reranker would help
JUDGE_MODEL = "gemini-3.5-flash-lite"  # validated 28/10 (judge-v2): precision 86-93%, recall 90-100%
WRITTEN = {"concept", "habit", "plan"}  # the stop rule looks only at these; scenarios wait for step 8
STOP_RECALL = 0.85  # judge recall@5 on written questions that stops the retrieval experiments
REPORT = Path("reports/recall_review.md")
MISSES = Path("reports/recall_misses.md")
CHECK = Path("reports/judge_check.md")
N_MISSES = 20  # misses at @5 sampled to check whether the gold is incomplete
N_CHECK_YES, N_CHECK_NO = 10, 5  # new verdicts for Federico: decisive yes of found questions, top chunk of missed ones

all_questions = load_eval(Path("data/eval/v1.yaml"))
questions = [q for q in all_questions if q["type"] != "abstain"]  # abstain has no gold
abstain = [q for q in all_questions if q["type"] == "abstain"]
scenarios = [q for q in questions if "scenario" in q]
brawler_list = load_brawlers(Path("data/brawlers.yaml"))
sources = load_sources(Path("data/sources.yaml"))
with_chapters = {s["id"] for s in sources if s.get("chapters")}
source_format = {s["id"]: s["format"] for s in sources}  # main format (format_changes inside a video not split yet)

db = MongoClient(MONGO_URI)["draft_coach"]
wiki_class = {c["_id"]: c.get("wiki_class") for c in db["brawler_cards"].find({}, {"wiki_class": 1})}
builder = PromptBuilder(brawler_list, load_maps(Path("data/maps.yaml")), wiki_class)
question_text = builder.question_text

parser = argparse.ArgumentParser()
parser.add_argument("--formats", nargs="+", choices=sorted(FORMATS), default=sorted(FORMATS))
searched = sources_with_formats(sources, set(parser.parse_args().formats))  # same filter for both methods
print(f"searching {len(searched)} sources: {', '.join(sorted(parser.parse_args().formats))}")


def groups(q: dict) -> list[str]:
    """The rows of the tables this question counts in."""
    gold_sources = {g["source"] for g in q["gold"]}
    chapters = ("chapters" if gold_sources <= with_chapters
                else "no chapters" if not gold_sources & with_chapters else "mixed")
    rows = ["all", "written"] if q["type"] in WRITTEN else ["all"]
    rows += [f"type {q['type']}", f"lang {q.get('lang', 'scenario')}", f"gold in {chapters}"]
    return rows + [f"format {f}" for f in sorted({source_format[s] for s in gold_sources})]


def table_rows() -> list[str]:
    order = ["all", "written", "type", "lang", "gold", "format"]  # rows grouped by kind, in this order
    return sorted(dict.fromkeys(g for q in questions for g in groups(q)), key=lambda r: (order.index(r.split()[0]), r))


def at5(rank: int | None) -> bool:
    return rank is not None and rank <= 5


client = genai.Client(http_options=types.HttpOptions(timeout=30_000))
runs = {}  # method -> {"ranks", "cov5", "top_score", "top5", "reference_ranks", "abstain_scores"}
for method in METHODS:
    retriever = Retriever(client, method=method)
    run = runs[method] = {"ranks": {}, "cov5": {}, "top_score": {}, "top5": {}, "reference_ranks": {}}
    for q in tqdm(questions, desc=f"{method} questions"):
        chunks = retriever.search(question_text(q), top_k=TOP_K, sources=searched)
        run["ranks"][q["id"]] = first_hit(chunks, q["gold"])
        run["cov5"][q["id"]] = coverage(chunks[:5], q["gold"])
        run["top_score"][q["id"]], run["top5"][q["id"]] = chunks[0]["score"], chunks[:5]
    # Upper bound for scenarios: the concept written by us as the query (index problem or query problem?)
    for q in tqdm(scenarios, desc=f"{method} reference"):
        run["reference_ranks"][q["id"]] = first_hit(retriever.search(q["reference"], top_k=TOP_K, sources=searched),
                                                    q["gold"])
    run["abstain_scores"] = [retriever.search(q["question"], top_k=1, sources=searched)[0]["score"]
                             for q in tqdm(abstain, desc=f"{method} abstain")]

# Judge the first 5 chunks of every question, both methods in one batch (identical pairs are asked once)
pairs = {}  # (method, question id) -> the 5 pairs, in rank order
for method in METHODS:
    for q in questions:
        pairs[method, q["id"]] = []
        for c in runs[method]["top5"][q["id"]]:
            prompt = builder.prompt(q, c["text"], c["mentions"])
            pairs[method, q["id"]].append({"key": judgment_id(JUDGE_MODEL, prompt), "prompt": prompt,
                                           "excerpt": c["text"], "question_id": q["id"], "chunk_id": c["chunk_id"]})
judge_client = genai.Client(http_options=types.HttpOptions(timeout=MODELS[JUDGE_MODEL][1]))
verdicts = judge_cached(judge_client, JUDGE_MODEL, [p for five in pairs.values() for p in five], db["judgments"])
judged = {key: [verdicts[p["key"]] for p in five] for key, five in pairs.items()}  # (method, id) -> 5 verdicts

# 1. Table: recall@5, recall@10, MRR, coverage@5 per group (gold rule: the chunk holds at least half the interval)
print(f"\nGold intervals\n{'':<28}" + "".join(f"{m:^28}" for m in METHODS))
print(f"{'group':<24}{'n':>4}" + f"{'R@5':>7}{'R@10':>7}{'MRR':>7}{'Cov@5':>7}" * len(METHODS))
for row in table_rows():
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

# 3. Judge table: J@5 = at least one yes in the first 5 (with its 95% interval), J+P@5 = yes or partial,
#    Ctx = share of the 5 chunks that are yes or partial (how much of the context is useful)
yes5 = {key: "yes" in v for key, v in judged.items()}
print(f"\nRetrieval judge ({JUDGE_MODEL})\n{'':<28}" + "".join(f"{m:^31}" for m in METHODS))
print(f"{'group':<24}{'n':>4}" + f"{'J@5':>5}{'95% interval':>14}{'J+P@5':>6}{'Ctx':>6}" * len(METHODS))
for row in table_rows():
    ids = [q["id"] for q in questions if row in groups(q)]
    cells = ""
    for m in METHODS:
        hits = sum(yes5[m, i] for i in ids)
        low, high = wilson_interval(hits, len(ids))
        partial5 = sum(any(v in ("yes", "partial") for v in judged[m, i]) for i in ids)
        useful = sum(v in ("yes", "partial") for i in ids for v in judged[m, i]) / (5 * len(ids))
        cells += f"{hits / len(ids):>5.0%}{f'{low:.0%}-{high:.0%}':>14}{partial5 / len(ids):>6.0%}{useful:>6.0%}"
    print(f"{row:<24}{len(ids):>4}{cells}")
missing = {m: sum(v is None for (method, _), five in judged.items() if method == m for v in five) for m in METHODS}
if any(missing.values()):
    print(f"pairs without a verdict (counted as no, run again to fill): {missing}")

# 4. Judge: paired comparison and stop rule
for label, ids in [("all", [q["id"] for q in questions]), ("written", [q["id"] for q in questions if q["type"] in WRITTEN])]:
    gains = sum(yes5[b, i] and not yes5[a, i] for i in ids)
    losses = sum(yes5[a, i] and not yes5[b, i] for i in ids)
    print(f"judge, {label:<8} {b} gains {gains} and loses {losses} against {a}: sign test p = {sign_test_p(gains, losses):.2f}")
written = [q["id"] for q in questions if q["type"] in WRITTEN]
for m in METHODS:
    hits = sum(yes5[m, i] for i in written)
    low, high = wilson_interval(hits, len(written))
    print(f"stop rule {m:<15} J@5 on written {hits / len(written):.0%} ({low:.0%}-{high:.0%}), target {STOP_RECALL:.0%}")

# 5. Scenarios: rendered question against the reference as query
print(f"\n{'scenarios R@5':<28}" + "".join(f"{m:^28}" for m in METHODS))
for label, key in [("rendered question", "ranks"), ("reference as query", "reference_ranks")]:
    print(f"{label:<28}" + "".join(
        f"{sum(at5(runs[m][key][q['id']]) for q in scenarios) / len(scenarios):^28.0%}" for m in METHODS))

# 6. Top score: can a threshold separate abstain questions from answerable ones?
print("\nTop score of the first chunk (a threshold for abstention?)")
for m in METHODS:
    answerable, refused = list(runs[m]["top_score"].values()), runs[m]["abstain_scores"]
    below = sum(s <= max(refused) for s in answerable)
    print(f"{m:<28}answerable median {median(answerable):.3f} (min {min(answerable):.3f}), "
          f"abstain median {median(refused):.3f} (max {max(refused):.3f}); "
          f"answerable at or below the abstain max: {below}/{len(answerable)}")

# 7. Review of the disagreements: is the first chunk of the method that missed useful anyway?
lines = ["# Recall review: questions where the two methods disagree at @5",
         "For each one: is the first chunk of the method that missed useful for the reference? (useful / not useful)", ""]
for q in questions:
    if (q["id"] in found[a]) == (q["id"] in found[b]):
        continue
    missed = b if q["id"] in found[a] else a
    top = runs[missed]["top5"][q["id"]][0]
    lines += [f"## {q['id']} ({q['type']}): found by {a if missed == b else b}, missed by {missed}",
              f"Question: {question_text(q)}", f"Reference: {q['reference']}",
              f"Gold: {', '.join(f'{g['source']} {minute(g['start_s'])}-{minute(g['end_s'])}' for g in q['gold'])}",
              f"First chunk of {missed}: `{top['source_id']}` [{minute(top['start_s'])}]({top['url']})",
              f"> {top['text']}", "", "Verdict: ", ""]
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReview: {REPORT}")


# 8. Sample of misses at @5 (first method): real misses, or valid passages that the gold does not list?
segments = {n["_id"]: sorted(n["segments"], key=lambda s: s["start_s"]) for n in db["normalized"].find()}


def gold_text(g: dict) -> str:
    return " ".join(s["text"] for s in segments[g["source"]] if g["start_s"] <= s["start_s"] < g["end_s"])


missed = [q for q in questions if not at5(runs[a]["ranks"][q["id"]])]
sample = random.Random(0).sample(missed, min(N_MISSES, len(missed)))  # fixed seed: the same sample every run
lines = [f"# Misses at @5 of {a}: {len(sample)} of {len(missed)}, random sample",
         "For each chunk: does it answer the reference? (yes / no)", ""]
for q in sample:
    lines += [f"## {q['id']} ({q['type']})", f"Question: {question_text(q)}", f"Reference: {q['reference']}"]
    lines += [f"Gold {g['source']} {minute(g['start_s'])}-{minute(g['end_s'])}: {gold_text(g)}" for g in q["gold"]]
    for rank, c in enumerate(runs[a]["top5"][q["id"]], 1):
        lines += ["", f"{rank}. `{c['source_id']}` [{minute(c['start_s'])}]({c['url']})", f"> {c['text']}", "Verdict: "]
    lines.append("")
MISSES.write_text("\n".join(lines), encoding="utf-8")
print(f"Misses: {MISSES}")

# 9. New verdicts for Federico (first method), never seen in the judge validation: the decisive yes of found
#    questions (a false yes inflates the number) and the top chunk of questions without yes (a false no lowers it)
validated = {(p["question"], p["chunk"]) for p in yaml.safe_load(Path("data/eval/judge_validation.yaml").read_text(encoding="utf-8"))}
by_id = {q["id"]: q for q in questions}
decisive, top_of_missed = [], []
for q in questions:
    five = list(zip(range(1, 6), runs[a]["top5"][q["id"]], judged[a, q["id"]]))  # (rank, chunk, verdict)
    if yes5[a, q["id"]]:
        rank, c, verdict = next(x for x in five if x[2] == "yes")  # the yes that makes the question found
        bucket = decisive
    else:
        rank, c, verdict = five[0]  # the chunk most likely to be a useful one judged no
        bucket = top_of_missed
    if (q["id"], c["chunk_id"]) not in validated and verdict is not None:
        bucket.append((q["id"], rank, c, verdict))
rng = random.Random(0)  # fixed seed: the same sample until the judgments change
to_check = (rng.sample(decisive, min(N_CHECK_YES, len(decisive)))
            + rng.sample(top_of_missed, min(N_CHECK_NO, len(top_of_missed))))
saved = {d["_id"]: d for d in db["judgments"].find({"_id": {"$in": [pairs[a, i][r - 1]["key"] for i, r, _, _ in to_check]}})}
lines = [f"# Judge check: {len(to_check)} new verdicts of {a} ({JUDGE_MODEL}, prompt {PROMPT_VERSION})",
         "Is the verdict right? (right / wrong). A wrong yes inflates the judge recall, a wrong no lowers it.", ""]
for qid, rank, c, verdict in to_check:
    q, d = by_id[qid], saved[pairs[a, qid][rank - 1]["key"]]
    lines += [f"## {qid} ({q['type']}), chunk {rank}: judge {verdict}",
              f"Question: {question_text(q)}", f"Reference: {q['reference']}",
              f"Chunk: `{c['source_id']}` [{minute(c['start_s'])}]({c['url']})", f"> {c['text']}",
              f"Judge reason: {d['reason']}", f"Judge quote: {d['quote']}", "", "Right or wrong: ", ""]
CHECK.write_text("\n".join(lines), encoding="utf-8")
print(f"Judge check: {CHECK} ({len(to_check)} verdicts: from {len(decisive)} decisive yes and {len(top_of_missed)} missed)")