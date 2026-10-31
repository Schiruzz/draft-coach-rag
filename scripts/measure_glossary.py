"""Step 7d: does replacing the Italian game terms (data/glossary.yaml, Italian brawler names) before the search
help the retrieval in use? Written questions only; only the questions the glossary changes are run: on the
others the retrieval is identical by construction.
Both variants: the 20 nearest chunks reordered by the ranker, first 5 (Retriever.retrieve). The glossary variant
searches and reranks the translated text; the judge always reads the original question, as the player wrote it.
Decision rule, fixed before the run (31/10): the glossary enters Retriever.retrieve if, on the changed questions,
it loses no J question and the questions with fewer useful chunks are not more than those with more.
With ~16 changed questions this is a non-regression check on a deterministic rewrite, not a significance test.
Run: python scripts/measure_glossary.py"""
from pathlib import Path

from google import genai
from google.genai import types
from pymongo import MongoClient
from tqdm import tqdm

from draft_coach.claims import EXCLUDED_FORMATS
from draft_coach.config import GOOGLE_CLOUD_PROJECT, MONGO_URI
from draft_coach.eval_set import load_eval
from draft_coach.glossary import brawler_terms, load_glossary, to_english
from draft_coach.judge import MODELS, PromptBuilder, judge_cached, judgment_id
from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers
from draft_coach.recall import sign_test_p, useful_count
from draft_coach.rerank import RANK_MODEL, Reranker
from draft_coach.retrieval import CANDIDATES, Retriever
from draft_coach.sources import FORMATS, load_sources, sources_with_formats

JUDGE_MODEL = "gemini-3.5-flash-lite"  # as measure_rerank.py: most baseline pairs come from the cache
WRITTEN = {"concept", "habit", "plan"}

brawler_list = load_brawlers(Path("data/brawlers.yaml"))
terms = {**load_glossary(Path("data/glossary.yaml")), **brawler_terms(brawler_list)}
db = MongoClient(MONGO_URI)["draft_coach"]
wiki_class = {c["_id"]: c.get("wiki_class") for c in db["brawler_cards"].find({}, {"wiki_class": 1})}
builder = PromptBuilder(brawler_list, load_maps(Path("data/maps.yaml")), wiki_class)
written = [q for q in load_eval(Path("data/eval/v1.yaml")) if q["type"] in WRITTEN]
changed = [q for q in written if to_english(q["question"], terms) != q["question"]]
searched = sources_with_formats(load_sources(Path("data/sources.yaml")), FORMATS - EXCLUDED_FORMATS)
print(f"{len(changed)} of {len(written)} written questions changed by the glossary")

retriever = Retriever(genai.Client(http_options=types.HttpOptions(timeout=30_000)))
reranker = Reranker(GOOGLE_CLOUD_PROJECT, RANK_MODEL)
top = {}  # (variant, question id) -> the 5 chunks read
for q in tqdm(changed, desc="questions"):
    for variant, text in (("original", q["question"]), ("glossary", to_english(q["question"], terms))):
        top[variant, q["id"]] = reranker.rerank(text, retriever.search(text, top_k=CANDIDATES, sources=searched))

# The judge reads the original question in both variants: the same chunk gets the same cached verdict
pairs = {}
for (variant, qid), chunks in top.items():
    q = next(x for x in changed if x["id"] == qid)
    pairs[variant, qid] = [{"key": judgment_id(JUDGE_MODEL, p), "prompt": p, "excerpt": c["text"], "question_id": qid,
                            "chunk_id": c["chunk_id"]} for c in chunks for p in [builder.prompt(q, c["text"], c["mentions"])]]
judge_client = genai.Client(http_options=types.HttpOptions(timeout=MODELS[JUDGE_MODEL][1]))
verdicts = judge_cached(judge_client, JUDGE_MODEL, [p for ps in pairs.values() for p in ps], db["judgments"])
found = {key: "yes" in [verdicts[p["key"]] for p in ps] for key, ps in pairs.items()}
useful = {key: useful_count([verdicts[p["key"]] for p in ps]) for key, ps in pairs.items()}

print(f"\n{'question':<12}{'J orig':>7}{'J gloss':>8}{'useful':>8}  glossary text")
for q in changed:
    i = q["id"]
    print(f"{i:<12}{'yes' if found['original', i] else 'no':>7}{'yes' if found['glossary', i] else 'no':>8}"
          f"{useful['original', i]:>4} ->{useful['glossary', i]}  {to_english(q['question'], terms)}")

ids = [q["id"] for q in changed]
j_gain = sum(found["glossary", i] and not found["original", i] for i in ids)
j_loss = sum(found["original", i] and not found["glossary", i] for i in ids)
u_gain = sum(useful["glossary", i] > useful["original", i] for i in ids)
u_loss = sum(useful["glossary", i] < useful["original", i] for i in ids)
print(f"\nJ@5: original {sum(found['original', i] for i in ids)}/{len(ids)}, "
      f"glossary {sum(found['glossary', i] for i in ids)}/{len(ids)} (+{j_gain} -{j_loss})")
print(f"useful chunks per question: +{u_gain} -{u_loss} (sign test p = {sign_test_p(u_gain, u_loss):.2f}, for information)")
print(f"rule (no J loss, useful losses <= gains): {'ENTERS' if j_loss == 0 and u_loss <= u_gain else 'STAYS OUT'}")
missing = sum(v is None for v in verdicts.values())
if missing:
    print(f"pairs without a verdict (counted as no, run again to fill): {missing}")