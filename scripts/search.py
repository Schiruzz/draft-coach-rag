"""Ask the skeleton questions to the vector index and print the 5 nearest chunks of each.
Run: python scripts/search.py            (all questions in data/skeleton_questions.yaml)
     python scripts/search.py "question"  (one free question)"""
import sys
from pathlib import Path

import yaml
from google import genai
from google.genai import types

from draft_coach.retrieval import Retriever, minute

REPORT = Path("reports/skeleton_search.md")  # full results, easier to read than the terminal (git-ignored)

if len(sys.argv) > 1:
    questions = [{"id": "free", "question": sys.argv[1], "expected": "-"}]
else:
    questions = yaml.safe_load(Path("data/skeleton_questions.yaml").read_text(encoding="utf-8"))

retriever = Retriever(genai.Client(http_options=types.HttpOptions(timeout=30_000)))
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