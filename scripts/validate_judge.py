"""Validate the retrieval judge on data/eval/judge_validation.yaml (labels by Federico, Claude and the gold).
Run: python scripts/validate_judge.py --limit 10                  (timing: 10 pairs, one at a time)
     python scripts/validate_judge.py                             (all pairs, 3 runs: accuracy and stability)
     python scripts/validate_judge.py --model gemini-3.8-flash    (another model)
     python scripts/validate_judge.py --grep pick02               (only the questions whose id contains the text)
Disagreements go to reports/judge_disagreements.md, the most stable first."""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median

import yaml
from google import genai
from google.genai import types
from pymongo import MongoClient
from tqdm import tqdm

from draft_coach.config import MONGO_URI
from draft_coach.eval_set import load_eval
from draft_coach.judge import MODELS, PROMPT_VERSION, ask, brawler_facts, build_prompt, final_verdict, quote_found
from draft_coach.maps import load_maps
from draft_coach.names import NameNormalizer, load_brawlers
from draft_coach.scenarios import scenario_question

PARALLEL_CALLS = 8   # as in the riddle project; lower it if 429 errors appear
THRESHOLD = 0.85     # precision and recall on "yes" (partial counts as no), fixed before measuring
REPORT = Path("reports/judge_disagreements.md")

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=sorted(MODELS), default="gemini-3.5-flash-lite")
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--limit", type=int, help="judge only the first N pairs, one at a time, once, and print the timing")
parser.add_argument("--grep", default="", help="judge only the questions whose id contains this text")
args = parser.parse_args()

pairs = [p for p in yaml.safe_load(Path("data/eval/judge_validation.yaml").read_text(encoding="utf-8"))
         if args.grep in p["question"]][:args.limit]
questions = {q["id"]: q for q in load_eval(Path("data/eval/v1.yaml"))}
brawler_list = load_brawlers(Path("data/brawlers.yaml"))
brawlers = {b["id"]: b for b in brawler_list}
normalizer = NameNormalizer(brawler_list)
maps = {m["id"]: m for m in load_maps(Path("data/maps.yaml"))}
names = {i: b["en"] for i, b in brawlers.items()}

db = MongoClient(MONGO_URI)["draft_coach"]
wiki_class = {c["_id"]: c.get("wiki_class") for c in db["brawler_cards"].find({}, {"wiki_class": 1})}
chunks = {c["_id"]: c for c in db["chunks"].find({"_id": {"$in": [p["chunk"] for p in pairs]}})}
missing = [p["chunk"] for p in pairs if p["chunk"] not in chunks]
assert not missing, f"chunks not in MongoDB: {missing}"


def question_text(q: dict) -> str:
    return scenario_question(q["scenario"], names, maps, q.get("candidate")) if "scenario" in q else q["question"]


def prompt_for(pair: dict) -> str:
    q, chunk = questions[pair["question"]], chunks[pair["chunk"]]
    if "scenario" in q:
        s = q["scenario"]
        in_question = s.get("bans", []) + s.get("ours", []) + s.get("theirs", []) + [q.get("candidate")] * bool(q.get("candidate"))
        mode = s["quiz_mode"]
    else:
        in_question, mode = normalizer.mentions(q["question"]), None
    return build_prompt(question_text(q), mode, brawler_facts(in_question, brawlers, wiki_class), q["reference"],
                        chunk["text"], brawler_facts(chunk["mentions"], brawlers, wiki_class))


prompts = [prompt_for(p) for p in pairs]
client = genai.Client(http_options=types.HttpOptions(timeout=MODELS[args.model][1]))


def judge_one(i: int) -> dict:
    """One pair: the judgment, the final verdict after the quote check, the seconds it took."""
    start = time.monotonic()
    judgment = ask(client, args.model, prompts[i], label=f"{pairs[i]['question']} {pairs[i]['chunk']}")
    text = chunks[pairs[i]["chunk"]]["text"]
    return {"judgment": judgment, "seconds": time.monotonic() - start,
            "verdict": final_verdict(judgment, text) if judgment else None,
            "quote_ok": judgment is None or judgment.verdict == "no" or quote_found(judgment.quote, text)}


def run(name: str, workers: int) -> list[dict]:
    """Judge every pair once; one line per call as soon as it finishes."""
    results = [None] * len(pairs)
    with ThreadPoolExecutor(workers) as pool, tqdm(total=len(pairs), desc=name) as bar:
        futures = {pool.submit(judge_one, i): i for i in range(len(pairs))}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = r = future.result()
            mark = "" if r["verdict"] is None or (r["verdict"] == "yes") == (pairs[i]["label"] == "yes") else "  <- disagrees"
            tqdm.write(f"{r['seconds']:5.1f} s  {pairs[i]['question']:<11} label {pairs[i]['label']:<4} judge {r['verdict']}{mark}")
            bar.update()
    return results


print(f"model {args.model} (thinking {MODELS[args.model][0]}, timeout {MODELS[args.model][1] // 1000} s), "
      f"prompt {PROMPT_VERSION}, {len(pairs)} pairs ({sum(p['label'] == 'yes' for p in pairs)} yes)")

if args.limit:  # timing only: one call at a time, once
    results = run("timing", workers=1)
    times = sorted(r["seconds"] for r in results)
    print(f"\n{'calls':<18} {len(times)}   failed {sum(r['judgment'] is None for r in results)}")
    print(f"{'seconds per call':<18} median {median(times):.1f}   p90 {times[int(0.9 * (len(times) - 1))]:.1f}   max {times[-1]:.1f}")
    raise SystemExit


def metrics(results: list[dict]) -> dict:
    judged = [(p, r) for p, r in zip(pairs, results) if r["verdict"]]
    tp = sum(p["label"] == "yes" and r["verdict"] == "yes" for p, r in judged)
    said_yes = sum(r["verdict"] == "yes" for _, r in judged)
    true_yes = sum(p["label"] == "yes" for p, _ in judged)
    return {"judged": len(judged), "precision": tp / said_yes if said_yes else 0, "recall": tp / true_yes if true_yes else 0,
            "agreement": sum((r["verdict"] == "yes") == (p["label"] == "yes") for p, r in judged) / len(judged),
            "partial": sum(r["verdict"] == "partial" for _, r in judged),
            "quote not found": sum(not r["quote_ok"] for r in results),
            "seconds (median)": median(r["seconds"] for r in results)}


runs = [run(f"run {n}", PARALLEL_CALLS) for n in range(1, args.runs + 1)]
table = [metrics(results) for results in runs]
print(f"\n{'':<18}" + "".join(f"{'run ' + str(n):>10}" for n in range(1, args.runs + 1)))
for key in table[0]:
    print(f"{key:<18}" + "".join(f"{m[key]:>10.0%}" if key in ("precision", "recall", "agreement")
                                 else f"{m[key]:>10.1f}" if isinstance(m[key], float) else f"{m[key]:>10}" for m in table))

# Stability: how many runs disagree with the label, per pair (3/3 = a stable error, worth a prompt fix)
wrong = [sum(r[i]["verdict"] is not None and (r[i]["verdict"] == "yes") != (p["label"] == "yes") for r in runs)
         for i, p in enumerate(pairs)]
changed = sum(len({r[i]["verdict"] for r in runs}) > 1 for i in range(len(pairs)))
print(f"\npairs with a different verdict across runs: {changed}/{len(pairs)}")
for k in range(args.runs, 0, -1):
    print(f"disagree with the label in {k}/{args.runs} runs: {sum(w == k for w in wrong)}")
for by in ["federico", "claude", "gold"]:  # whose labels the judge disagrees with
    rows = [w for w, p in zip(wrong, pairs) if p["by"] == by]
    print(f"stable disagreements with {by:<9} {sum(w == args.runs for w in rows)}/{len(rows)}")
low = min(min(m["precision"], m["recall"]) for m in table)
print(f"\nthreshold {THRESHOLD:.0%} on precision and recall in every run: {'passed' if low >= THRESHOLD else 'NOT passed'} "
      f"(mean precision {mean(m['precision'] for m in table):.0%}, recall {mean(m['recall'] for m in table):.0%})")

lines = [f"# Judge disagreements ({args.model}, prompt {PROMPT_VERSION}, {args.runs} runs), most stable first",
         "Who is right: the label or the judge?", ""]
for i in sorted(range(len(pairs)), key=lambda i: -wrong[i]):
    if not wrong[i]:
        continue
    p, q = pairs[i], questions[pairs[i]["question"]]
    first = next(r[i]["judgment"] for r in runs if r[i]["judgment"])
    lines += [f"## {p['question']} -> `{p['chunk']}`  (disagrees in {wrong[i]}/{args.runs} runs)",
              f"Label: {p['label']} ({p['by']})   Judge: {', '.join(str(r[i]['verdict']) for r in runs)}",
              f"Question: {question_text(q)}", f"Reference: {q['reference']}",
              f"Judge reason: {first.reason}", f"Judge quote: {first.quote}",
              f"> {chunks[p['chunk']]['text']}", "", "Who is right: ", ""]
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"Disagreements: {REPORT}")