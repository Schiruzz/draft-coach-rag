"""Answer questions with Gemini using only the 5 chunks of the retrieval in use, with citations (RAG skeleton, step 3s.4).
Run: python scripts/answer.py            (all questions in data/skeleton_questions.yaml)
     python scripts/answer.py "question"  (one free question)"""
import sys
from pathlib import Path

import yaml
from google import genai
from google.genai import types

from draft_coach.config import GEMINI_MODEL
from draft_coach.retrieval import Retriever, minute

PROMPT_VERSION = "v1"  # bump when the prompt changes
REPORT = Path("reports/skeleton_answers.md")
PROMPT = """You are a draft coach for Brawl Stars ranked. Answer the question using ONLY the excerpts below,
which are pieces of YouTube video transcripts by pro players (automatic transcripts: names may be misspelled).

Rules:
- Every statement must come from the excerpts and end with its citation, e.g. [2] or [1][4].
- If the excerpts do not answer the question, say so in one sentence and stop. Do not use your own knowledge.
- If the excerpts answer only part of it, answer that part and say what is missing.
- Answer in the language of the question, in at most 8 lines.

Question: {question}

Excerpts:
{excerpts}"""

if len(sys.argv) > 1:
    questions = [{"id": "free", "question": sys.argv[1]}]
else:
    questions = yaml.safe_load(Path("data/skeleton_questions.yaml").read_text(encoding="utf-8"))

client = genai.Client(http_options=types.HttpOptions(timeout=60_000))  # ms; an answer takes longer than an embedding
retriever = Retriever(client)
config = types.GenerateContentConfig(temperature=0)  # same question -> same answer, easier to compare runs

lines = [f"Prompt {PROMPT_VERSION}, model {GEMINI_MODEL}", ""]
for q in questions:
    chunks = retriever.retrieve(q["question"])  # 20 nearest, reordered by the ranker, first 5
    excerpts = "\n\n".join(f"[{i}] ({c['source_id']}, {minute(c['start_s'])}) {c['text']}" for i, c in enumerate(chunks, 1))
    answer = client.models.generate_content(
        model=GEMINI_MODEL, contents=PROMPT.format(question=q["question"], excerpts=excerpts), config=config).text.strip()
    sources = [f"[{i}] {c['source_id']} {minute(c['start_s'])} {c['url']}" for i, c in enumerate(chunks, 1)]
    print(f"\n{q['id']}: {q['question']}\n{answer}\n" + "\n".join(sources))
    lines += [f"## {q['id']}: {q['question']}", "", answer, "", *[f"- {s}" for s in sources], ""]
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nFull results: {REPORT}")