"""Step 6: does searching the claims find better chunks than searching the chunks? Same questions, same judge.
Every variant hands the judge 5 distinct chunks of the method in use, so it reads the same unit as the reference:
  chunks          the chunks nearest to the question (the reference, on today's chunk texts)
  claims:<text>   the claims nearest to the question, each replaced by its chunk, until 5 distinct chunks
  union           chunks and claims (concept_reason) ranked together by score, until 5 distinct chunks
Judgments are shared with measure_recall.py through the cache: only new (question, chunk) pairs are asked.
Run: python scripts/measure_claims.py"""
from pathlib import Path

from google import genai
from google.genai import types
from pymongo import MongoClient
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from draft_coach.chunks import METHOD
from draft_coach.claims import EXTRACT_VERSION
from draft_coach.config import EMBEDDING_MODEL, MONGO_URI, QDRANT_URL
from draft_coach.eval_set import load_eval
from draft_coach.judge import MODELS, PromptBuilder, judge_cached, judgment_id
from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers
from draft_coach.recall import first_hit, sign_test_p, wilson_interval
from draft_coach.sources import FORMATS, load_sources, sources_with_formats
from draft_coach.vectors import collection_name

JUDGE_MODEL = "gemini-3.5-flash-lite"  # as measure_recall.py
WRITTEN = {"concept", "habit", "plan"}  # the decision looks at these; scenarios wait for step 8
EXCLUDED = {"gameplay", "quiz", "wiki"}  # as the 83% of 28/10 and as the extraction
CLAIM_HITS = 50  # claims read to fill 5 distinct chunks (one chunk often holds several claims)
TEXTS = ["concept"]  # concept_reason measured equal to concept on extract-v4 (30/10): dropped
VARIANTS = ["chunks"] + [f"claims:{t}" for t in TEXTS] + ["union"]

questions = [q for q in load_eval(Path("data/eval/v1.yaml")) if q["type"] != "abstain"]  # abstain has no gold
brawler_list = load_brawlers(Path("data/brawlers.yaml"))
searched = sources_with_formats(load_sources(Path("data/sources.yaml")), FORMATS - EXCLUDED)
db = MongoClient(MONGO_URI)["draft_coach"]
wiki_class = {c["_id"]: c.get("wiki_class") for c in db["brawler_cards"].find({}, {"wiki_class": 1})}
builder = PromptBuilder(brawler_list, load_maps(Path("data/maps.yaml")), wiki_class)
chunks = {c["_id"]: c for c in db["chunks"].find({"method": METHOD, "layer": {"$exists": False}})}

qdrant = QdrantClient(url=QDRANT_URL)
chunk_collection = collection_name(METHOD, EMBEDDING_MODEL)
claim_collection = {t: f"claims__{EXTRACT_VERSION}__{t}__{EMBEDDING_MODEL}" for t in TEXTS}
guides = models.Filter(must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="layer")),
                             models.FieldCondition(key="source_id", match=models.MatchAny(any=searched))])


def nearest(collection: str, vector: list[float], limit: int, query_filter=None) -> list[tuple[float, str]]:
    """(score, chunk id) of the nearest points; for claims, the id of their chunk."""
    hits = qdrant.query_points(collection, query=vector, query_filter=query_filter, limit=limit,
                               with_payload=["chunk_id"]).points
    return [(h.score, h.payload["chunk_id"]) for h in hits if h.payload["chunk_id"]]


def first_five(ranked: list[tuple[float, str]]) -> list[dict]:
    """The first 5 distinct chunks of a ranking."""
    ids = list(dict.fromkeys(chunk_id for _, chunk_id in ranked))[:5]
    return [chunks[i] for i in ids]


client = genai.Client(http_options=types.HttpOptions(timeout=30_000))
query_config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
top5 = {}  # (variant, question id) -> 5 chunks
for q in tqdm(questions, desc="questions"):
    vector = client.models.embed_content(model=EMBEDDING_MODEL, contents=builder.question_text(q),
                                         config=query_config).embeddings[0].values  # one embedding, every variant
    by_chunk = nearest(chunk_collection, vector, 10, guides)
    by_claim = {t: nearest(claim_collection[t], vector, CLAIM_HITS) for t in TEXTS}
    top5["chunks", q["id"]] = first_five(by_chunk)
    for t in TEXTS:
        top5[f"claims:{t}", q["id"]] = first_five(by_claim[t])
    top5["union", q["id"]] = first_five(sorted(by_chunk + by_claim["concept"], reverse=True))

# Judge the 5 chunks of every variant in one batch: pairs already judged come from the cache
pairs = {}
for key, five in top5.items():
    q = next(x for x in questions if x["id"] == key[1])
    pairs[key] = []
    for c in five:
        prompt = builder.prompt(q, c["text"], c["mentions"])
        pairs[key].append({"key": judgment_id(JUDGE_MODEL, prompt), "prompt": prompt, "excerpt": c["text"],
                           "question_id": q["id"], "chunk_id": c["_id"]})
judge_client = genai.Client(http_options=types.HttpOptions(timeout=MODELS[JUDGE_MODEL][1]))
verdicts = judge_cached(judge_client, JUDGE_MODEL, [p for five in pairs.values() for p in five], db["judgments"])
judged = {key: [verdicts[p["key"]] for p in five] for key, five in pairs.items()}
yes5 = {key: "yes" in v for key, v in judged.items()}


def groups(q: dict) -> list[str]:
    rows = ["all", "written"] if q["type"] in WRITTEN else ["all"]
    return rows + [f"type {q['type']}", f"lang {q.get('lang', 'scenario')}"]


rows = list(dict.fromkeys(g for q in questions for g in groups(q)))
print(f"\nRetrieval judge ({JUDGE_MODEL}): J@5 (95% interval) / Ctx = share of the 5 chunks that are yes or partial; "
      f"R@5 = gold rule\n{'group':<18}{'n':>4}" + "".join(f"{v:>30}" for v in VARIANTS))
for row in rows:
    ids = [q["id"] for q in questions if row in groups(q)]
    cells = ""
    for v in VARIANTS:
        hits = sum(yes5[v, i] for i in ids)
        low, high = wilson_interval(hits, len(ids))
        ctx = sum(x in ("yes", "partial") for i in ids for x in judged[v, i]) / (5 * len(ids))
        gold = sum((r := first_hit(top5[v, i], next(q for q in questions if q["id"] == i)["gold"])) is not None
                   and r <= 5 for i in ids) / len(ids)
        cells += f"{f'{hits / len(ids):.0%} ({low:.0%}-{high:.0%}) {ctx:.0%} {gold:.0%}':>30}"
    print(f"{row:<18}{len(ids):>4}{cells}")

print("\nAgainst chunks (J@5): questions gained and lost, sign test (below 0.05: a real difference)")
for label, ids in [("written", [q["id"] for q in questions if q["type"] in WRITTEN]), ("all", [q["id"] for q in questions])]:
    for v in VARIANTS[1:]:
        gains = sum(yes5[v, i] and not yes5["chunks", i] for i in ids)
        losses = sum(yes5["chunks", i] and not yes5[v, i] for i in ids)
        print(f"  {label:<8} {v:<22} +{gains:<3} -{losses:<3} p = {sign_test_p(gains, losses):.2f}")
missing = sum(v is None for five in judged.values() for v in five)
if missing:
    print(f"pairs without a verdict (counted as no, run again to fill): {missing}")