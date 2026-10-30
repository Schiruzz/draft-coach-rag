"""Step 7b: does reordering the 20 nearest chunks with the ranking API put the useful ones in the first 5?
Written questions only (scenarios wait for step 8). Every variant hands the judge chunks of the method in use:
  chunks@5    the 5 nearest chunks (the value to beat)
  chunks@10   the 10 nearest (the simplest alternative: more context, no reranker)
  <model>@5   the 20 nearest reordered by the ranker, first 5
Reordered chunks are among the 20 nearest, already judged on 30/10: only pairs missing from the cache are asked.
Metrics corrected on 30/10 (the first version tested only J, and Ctx against 100%):
  NCtx    useful chunks read / most that could be read (useful chunks among the 20 nearest, up to the chunks read)
  useful  sign test per question on the number of useful chunks, not only on J
  overlap chunks sharing time with an earlier chunk of the same source (places spent on the same text)
Decision rule, fixed before the corrected run: the reranker enters if it improves the useful chunks per question
with p < 0.05 and does not lose more J questions than it gains.
Run: python scripts/measure_rerank.py"""
import time
from pathlib import Path
from statistics import median, quantiles

from google import genai
from google.genai import types
from pymongo import MongoClient
from tqdm import tqdm

from draft_coach.claims import EXCLUDED_FORMATS
from draft_coach.config import GOOGLE_CLOUD_PROJECT, MONGO_URI
from draft_coach.eval_set import load_eval
from draft_coach.judge import MODELS, PromptBuilder, judge_cached, judgment_id
from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers
from draft_coach.recall import normalized_ctx, overlapping, sign_test_p, useful_count, wilson_interval
from draft_coach.rerank import RANK_MODELS, Reranker
from draft_coach.retrieval import Retriever, minute
from draft_coach.sources import FORMATS, load_sources, sources_with_formats

JUDGE_MODEL = "gemini-3.5-flash-lite"  # as measure_recall.py and measure_claims.py
WRITTEN = {"concept", "habit", "plan"}
DEPTH = 20  # measured 30/10: J@20 97% on written questions, the ceiling of a reranker
VARIANTS = ["chunks@5", "chunks@10"] + [f"{m}@5" for m in RANK_MODELS]

brawler_list = load_brawlers(Path("data/brawlers.yaml"))
db = MongoClient(MONGO_URI)["draft_coach"]
wiki_class = {c["_id"]: c.get("wiki_class") for c in db["brawler_cards"].find({}, {"wiki_class": 1})}
builder = PromptBuilder(brawler_list, load_maps(Path("data/maps.yaml")), wiki_class)
written = [q for q in load_eval(Path("data/eval/v1.yaml")) if q["type"] in WRITTEN]
searched = sources_with_formats(load_sources(Path("data/sources.yaml")), FORMATS - EXCLUDED_FORMATS)

retriever = Retriever(genai.Client(http_options=types.HttpOptions(timeout=30_000)))
rerankers = {m: Reranker(GOOGLE_CLOUD_PROJECT, m) for m in RANK_MODELS}
top, latency = {}, {m: [] for m in RANK_MODELS}  # (variant, question id) -> chunks; model -> seconds per call
for q in tqdm(written, desc="questions"):
    text = builder.question_text(q)
    nearest = retriever.search(text, top_k=DEPTH, sources=searched)
    top["chunks@5", q["id"]], top["chunks@10", q["id"]] = nearest[:5], nearest[:10]
    top["chunks@20", q["id"]] = nearest  # what is available: the ceiling of every variant
    for m, reranker in rerankers.items():
        start = time.monotonic()
        top[f"{m}@5", q["id"]] = reranker.rerank(text, nearest)
        latency[m].append(time.monotonic() - start)

# Judge every chunk of every variant in one batch: pairs already judged come from the cache
pairs = {}
for (variant, qid), chunks in top.items():
    q = next(x for x in written if x["id"] == qid)
    pairs[variant, qid] = [{"key": judgment_id(JUDGE_MODEL, p), "prompt": p, "excerpt": c["text"], "question_id": qid,
                            "chunk_id": c["chunk_id"]} for c in chunks for p in [builder.prompt(q, c["text"], c["mentions"])]]
judge_client = genai.Client(http_options=types.HttpOptions(timeout=MODELS[JUDGE_MODEL][1]))
verdicts = judge_cached(judge_client, JUDGE_MODEL, [p for ps in pairs.values() for p in ps], db["judgments"])
judged = {key: [verdicts[p["key"]] for p in ps] for key, ps in pairs.items()}
found = {key: "yes" in v for key, v in judged.items()}

ids = [q["id"] for q in written]
lang = {q["id"]: q.get("lang", "-") for q in written}
useful = {key: useful_count(v) for key, v in judged.items()}
available = {i: useful["chunks@20", i] for i in ids}
print(f"\nRetrieval judge ({JUDGE_MODEL}), written questions ({len(ids)}); useful = yes or partial")
print(f"{sum(available[i] == 0 for i in ids)} questions have no useful chunk among the 20 nearest: left out of NCtx")
print(f"{'variant':<30}{'J':>5}{'95% int.':>10}{'Ctx':>5}{'NCtx':>6}{'useful':>7}{'overlap':>8}"
      f"{'J vs @5':>10}{'p':>6}{'useful vs @5':>14}{'p':>6}")
for v in VARIANTS:
    hits = sum(found[v, i] for i in ids)
    low, high = wilson_interval(hits, len(ids))
    read = len(top[v, ids[0]])
    ctx = sum(useful[v, i] for i in ids) / (read * len(ids))
    nctx = [x for i in ids if (x := normalized_ctx(useful[v, i], available[i], read)) is not None]
    cells = f"{v:<30}{hits / len(ids):>5.0%}{f'{low:.0%}-{high:.0%}':>10}{ctx:>5.0%}{sum(nctx) / len(nctx):>6.0%}"
    cells += f"{sum(useful[v, i] for i in ids) / len(ids):>7.1f}{sum(overlapping(top[v, i]) for i in ids) / len(ids):>8.2f}"
    if v != "chunks@5":
        j_gain = sum(found[v, i] and not found["chunks@5", i] for i in ids)
        j_loss = sum(found["chunks@5", i] and not found[v, i] for i in ids)
        u_gain = sum(useful[v, i] > useful["chunks@5", i] for i in ids)
        u_loss = sum(useful[v, i] < useful["chunks@5", i] for i in ids)
        cells += f"{f'+{j_gain} -{j_loss}':>10}{sign_test_p(j_gain, j_loss):>6.2f}"
        cells += f"{f'+{u_gain} -{u_loss}':>14}{sign_test_p(u_gain, u_loss):>6.2f}"
    print(cells)

print(f"\n{'ranker latency (s)':<34}{'median':>8}{'p95':>8}{'max':>8}")
for m, seconds in latency.items():
    print(f"{m:<34}{median(seconds):>8.2f}{quantiles(seconds, n=20)[18]:>8.2f}{max(seconds):>8.2f}")

# Error analysis: the J questions a reranker lost, with the useful chunk it dropped and the chunk it put first
for m in RANK_MODELS:
    lost = [i for i in ids if found["chunks@5", i] and not found[f"{m}@5", i]]
    print(f"\n{m}: {len(lost)} questions lost")
    for i in lost:
        kept = {c["chunk_id"] for c in top[f"{m}@5", i]}
        q = next(x for x in written if x["id"] == i)
        print(f"\n  {i} ({lang[i]}): {q['question']}")
        for c, verdict in zip(top["chunks@5", i], judged["chunks@5", i]):
            if verdict == "yes" and c["chunk_id"] not in kept:
                print(f"    dropped yes  {c['source_id']} {minute(c['start_s'])}: {c['text'][:110]}...")
        first, verdict = top[f"{m}@5", i][0], judged[f"{m}@5", i][0]
        print(f"    put first ({verdict})  {first['source_id']} {minute(first['start_s'])}: {first['text'][:110]}...")
    print(f"  by language, gained / lost: " + ", ".join(
        f"{l} +{sum(found[f'{m}@5', i] and not found['chunks@5', i] for i in ids if lang[i] == l)}"
        f" -{sum(i in lost for i in ids if lang[i] == l)}" for l in sorted(set(lang.values()))))

missing = sum(v is None for five in judged.values() for v in five)
if missing:
    print(f"pairs without a verdict (counted as no, run again to fill): {missing}")