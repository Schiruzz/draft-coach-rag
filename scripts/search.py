"""Ask the skeleton questions to the vector index and print the 5 nearest chunks of each.
Run: python scripts/search.py            (all questions in data/skeleton_questions.yaml)
     python scripts/search.py "question"  (one free question)
     add --method chapter60-v1 to search the chunks of the other method"""
import argparse
from pathlib import Path

import yaml
from google import genai
from google.genai import types

from draft_coach.chunks import CHAPTER_METHOD, METHOD
from draft_coach.retrieval import Retriever, minute

REPORT = Path("reports/skeleton_search.md")  # full results, easier to read than the terminal (git-ignored)

parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="?")  # no question: all the skeleton questions
parser.add_argument("--method", choices=[METHOD, CHAPTER_METHOD], default=METHOD)
args = parser.parse_args()

if args.question:
    questions = [{"id": "free", "question": args.question, "expected": "-"}]
else:
    questions = yaml.safe_load(Path("data/skeleton_questions.yaml").read_text(encoding="utf-8"))

retriever = Retriever(genai.Client(http_options=types.HttpOptions(timeout=30_000)), method=args.method)
lines = []
for q in questions:
    chunks = retriever.search(q["question"])
    lines += [f"## {q['id']}: {q['question']}", f"Expected: {q['expected']}", ""]
    print(f"\n{q['id']}: {q['question']}")
    for rank, c in enumerate(chunks, 1):
        print(f"  {rank}. {c['score']:.3f}  {c['source_id']:<32} {minute(c['start_s']):>6}  {c['text'][:90]}...")
        lines += [f"{rank}. **{c['score']:.3f}** `{c['source_id']}` [{minute(c['start_s'])}]({c['url']})", f"   > {c['text']}", ""]
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nFull results: {REPORT}")