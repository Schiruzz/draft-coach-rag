"""Ask the skeleton questions to the vector index and print the 5 nearest chunks of each.
Run: python scripts/search.py            (all questions in data/skeleton_questions.yaml)
     python scripts/search.py "question"  (one free question)
     add --method window60-v1 (or chapter60-v1) to search the chunks of another method
     add --layer wiki to search the wiki layer (mode pages and map Layouts) instead of the guides"""
import argparse
from pathlib import Path

import yaml
from google import genai
from google.genai import types

from draft_coach.chunks import METHOD, METHODS
from draft_coach.retrieval import Retriever, locator
from draft_coach.wiki_chunks import LAYER as WIKI_LAYER

REPORT = Path("reports/skeleton_search.md")  # full results, easier to read than the terminal (git-ignored)

parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="?")  # no question: all the skeleton questions
parser.add_argument("--method", choices=METHODS, default=METHOD)
parser.add_argument("--layer", choices=[WIKI_LAYER])  # none: the guides
args = parser.parse_args()

if args.question:
    questions = [{"id": "free", "question": args.question, "expected": "-"}]
else:
    questions = yaml.safe_load(Path("data/skeleton_questions.yaml").read_text(encoding="utf-8"))

retriever = Retriever(genai.Client(http_options=types.HttpOptions(timeout=30_000)), method=args.method)
lines = []
for q in questions:
    chunks = retriever.search(q["question"], layer=args.layer)
    lines += [f"## {q['id']}: {q['question']}", f"Expected: {q['expected']}", ""]
    print(f"\n{q['id']}: {q['question']}")
    for rank, c in enumerate(chunks, 1):
        print(f"  {rank}. {c['score']:.3f}  {c['source_id']:<32} {locator(c):>8}  {c['text'][:90]}...")
        lines += [f"{rank}. **{c['score']:.3f}** `{c['source_id']}` [{locator(c)}]({c['url']})", f"   > {c['text']}", ""]
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nFull results: {REPORT}")