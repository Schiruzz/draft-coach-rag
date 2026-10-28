import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Literal

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from pymongo.collection import Collection
from tqdm import tqdm

from draft_coach.names import NameNormalizer
from draft_coach.scenarios import scenario_question

PROMPT_VERSION = "judge-v2"  # bump when the prompt changes; saved with every cached judgment as a label
PARALLEL_CALLS = 8  # as in the riddle project; lower it if 429 errors appear
# model -> (thinking level, timeout in ms); measured 28/10 in "eu": Lite answers a short call in under 1 s,
# Flash in 2-4 s but 20 s median on the judge prompt, with 504 after 57 s on 2 calls out of 10
MODELS = {"gemini-3.5-flash-lite": ("high", 12_000),  # Lite needs high reasoning for hard cases (riddle project)
          "gemini-3.8-flash": ("low", 20_000)}
ATTEMPTS = 5
RETRY_CODES = {408, 429, 499, 500, 502, 503, 504}  # temporary: queue on Vertex, quota, server errors

QUIZ_MODES = {
    "diamond": "Diamond: blind picks, no draft (you only see your teammates' picks)",
    "mythic_plus": "Mythic+: full draft with bans, pick order 1, 2-3, 4-5, 6",
    None: "general question, not a draft scenario",
}

GLOSSARY = """Our draft classes are the canonical ones. Sources may use other words or the game's official roles:
- space_maker: assassin, aggro, diver, gap closer (game role Assassin)
- anti_tank: DPS, damage dealer, tank counter (game role Damage Dealer)
- thrower: thrower, artillery (game role Artillery)
- sniper: sniper, sharpshooter, long range (game role Marksman)
- control: control, controller, zone holder (game role Controller)
- tank: tank (game role Tank); support: support, healer (game role Support)
A brawler fits a class if any of its classes matches.
Pick order in Mythic+: 1, 2-3, 4-5, 6 (6 = last pick). Overdrive = hypercharge."""

EXAMPLES = """Examples (invented, not from this test):
1. Reference: "4-5 picks: if the enemy has only snipers and controls, take a space maker that jumps on them."
   Excerpt: "...they went Piper and Belle on Bounty, so I just sent an Edgar on them and it was over."
   Brawlers in the excerpt: Edgar: space_maker (draft), Assassin (game)
   -> reason: same rule shown on another map with other brawlers; verdict: yes; quote: "so I just sent an Edgar on them"
2. Reference: "A thrower is a good last pick only if the enemy has nobody who closes the gap."
   Excerpt: "Throwers are great last picks on maps with a lot of walls."
   -> reason: gives the rule without the required condition on gap closers; verdict: partial; quote: "Throwers are great last picks on maps with a lot of walls."
3. Reference: "In Diamond, ban the brawlers you hate facing the most."
   Excerpt: "I was told to just ban whatever feels annoying to you. But I'm here to tell you that is not at all correct."
   -> reason: the speaker mentions the idea only to reject it; verdict: no; quote: \"\""""

PROMPT = """You judge whether an excerpt (a short piece cut from a longer Brawl Stars video transcript) supports
the reference answer to a draft question. Transcripts are automatic: brawler names may be misspelled.
The question may be in Italian; the excerpt is in English. The excerpt may start or end mid-sentence:
judge only what it says, do not assume how it continues.

Glossary:
{glossary}

Format of the question: {quiz_mode}
Question: {question}
Brawlers in the question: {question_facts}
Reference answer: {reference}
Excerpt: {excerpt}
Brawlers in the excerpt: {excerpt_facts}

Verdict:
- yes: the excerpt states the key point of the reference, or the same rule applied to a different
  example (other map or brawlers), as long as the rule fits this situation and this format.
  If the reference lists alternatives or several valid answers, one of them with its reason is enough.
  If the reference gives different answers for different ranks or formats, one of them is enough.
  If the excerpt recommends brawlers, use the brawler facts: recommending a brawler of the class the
  reference asks for, in the same situation, applies the rule.
- partial: the excerpt gives the rule but misses a condition the reference requires
  ("only if...", "unless..."), or it is related but too vague to act on.
- no: anything else, including generic advice that would fit almost any question, and ideas the
  speaker mentions only to reject them.
When the excerpt lists several separate examples (for example the drafts of a quiz), judge each one in
its own situation (map, pick position, teams); never combine details from different examples.
Judge the rule, not whether the brawlers in the example are still strong today.
When unsure between yes and partial, choose partial.

{examples}

Answer with: reason (one short sentence), then verdict, then quote (the shortest sentence of the
excerpt, word for word, that contains the key point; empty for no)."""


class Judgment(BaseModel):
    reason: str  # first: the model writes its reasoning before the verdict
    verdict: Literal["yes", "partial", "no"]
    quote: str


def brawler_facts(ids: list[str], brawlers: dict, wiki_class: dict) -> str:
    """'Bull: space_maker (draft), Tank (game); ...' from brawlers.yaml (canonical) and the Fandom cards."""
    facts = []
    for i in dict.fromkeys(ids):
        b = brawlers[i]
        name = b["en"] if b["it"] == b["en"] else f"{b['en']} (it: {b['it']})"
        game = f", {wiki_class[i]} (game)" if wiki_class.get(i) else ""
        facts.append(f"{name}: {b['class']} (draft){game}")
    return "; ".join(facts) or "none"


def build_prompt(question: str, quiz_mode: str | None, question_facts: str, reference: str,
                 excerpt: str, excerpt_facts: str) -> str:
    return PROMPT.format(glossary=GLOSSARY, quiz_mode=QUIZ_MODES[quiz_mode], question=question,
                         question_facts=question_facts, reference=reference, excerpt=excerpt,
                         excerpt_facts=excerpt_facts, examples=EXAMPLES)


def _plain(text: str) -> str:
    """Lowercase words only: the quote check ignores case, punctuation and spacing."""
    return " ".join(re.findall(r"\w+", text.lower()))


def quote_found(quote: str, excerpt: str) -> bool:
    return bool(_plain(quote)) and _plain(quote) in _plain(excerpt)


def final_verdict(judgment: Judgment, excerpt: str) -> str:
    """A yes or partial whose quote is not in the excerpt becomes no: the judge must point at real text."""
    if judgment.verdict != "no" and not quote_found(judgment.quote, excerpt):
        return "no"
    return judgment.verdict


def judgment_id(model: str, prompt: str) -> str:
    """Same model and same full prompt -> same id: a judgment is asked once.
    Any change in the input (question text, facts, glossary, prompt version) gives a new id by itself."""
    return hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()


class PromptBuilder:
    """The judge prompt of a (question, excerpt) pair, built from our tables; shared by validation and measurement."""

    def __init__(self, brawler_list: list[dict], maps: list[dict], wiki_class: dict):
        self.brawlers = {b["id"]: b for b in brawler_list}
        self.names = {b["id"]: b["en"] for b in brawler_list}
        self.maps = {m["id"]: m for m in maps}
        self.normalizer = NameNormalizer(brawler_list)
        self.wiki_class = wiki_class  # brawler id -> official game class (Fandom cards)

    def question_text(self, q: dict) -> str:
        """Scenarios are rendered from their data; written questions are used as they are."""
        if "scenario" in q:
            return scenario_question(q["scenario"], self.names, self.maps, q.get("candidate"))
        return q["question"]

    def prompt(self, q: dict, excerpt: str, excerpt_mentions: list[str]) -> str:
        if "scenario" in q:
            s = q["scenario"]
            candidate = [q["candidate"]] if q.get("candidate") else []
            in_question, mode = s.get("bans", []) + s.get("ours", []) + s.get("theirs", []) + candidate, s["quiz_mode"]
        else:
            in_question, mode = self.normalizer.mentions(q["question"]), None
        return build_prompt(self.question_text(q), mode, brawler_facts(in_question, self.brawlers, self.wiki_class),
                            q["reference"], excerpt, brawler_facts(excerpt_mentions, self.brawlers, self.wiki_class))


def ask(client: genai.Client, model: str, prompt: str, label: str = "") -> Judgment | None:
    """One judgment, up to ATTEMPTS calls; None if it keeps failing (asked again on the next run).
    The client carries the timeout of the model (MODELS)."""
    for attempt in range(1, ATTEMPTS + 1):
        start = time.monotonic()
        try:
            response = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=Judgment, temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_level=MODELS[model][0])))
            return response.parsed
        except (errors.APIError, httpx.TransportError) as error:  # TransportError: network timeout, dropped connection
            if isinstance(error, errors.APIError) and error.code not in RETRY_CODES:
                raise
            reason = f"{error.code} {error.status}" if isinstance(error, errors.APIError) else type(error).__name__
            print(f"{label}: {reason} after {time.monotonic() - start:.0f} s, attempt {attempt}/{ATTEMPTS}")
            if attempt < ATTEMPTS:
                time.sleep(2 ** attempt)  # 2, 4, 8, 16 s
    return None


def judge_cached(client: genai.Client, model: str, pairs: list[dict], cache: Collection) -> dict[str, str | None]:
    """pairs: dicts with key (judgment_id), prompt, excerpt, question_id, chunk_id. Returns key -> final verdict
    (None = still failing after ATTEMPTS calls: not saved, asked again on the next run).
    Saved judgments are read from MongoDB; the others are asked in parallel and saved as soon as they arrive,
    so an interrupted run resumes where it stopped."""
    todo = {p["key"]: p for p in pairs}  # the same pair can come from both methods: asked once
    verdicts = {d["_id"]: d["verdict"] for d in cache.find({"_id": {"$in": list(todo)}}, {"verdict": 1})}
    missing = [p for key, p in todo.items() if key not in verdicts]
    print(f"judgments: {len(todo)} distinct pairs, {len(verdicts)} cached, {len(missing)} to ask")
    with ThreadPoolExecutor(PARALLEL_CALLS) as pool, tqdm(total=len(missing), desc="judging") as bar:
        futures = {pool.submit(ask, client, model, p["prompt"], f"{p['question_id']} {p['chunk_id']}"): p
                   for p in missing}
        try:
            for future in as_completed(futures):
                p, judgment = futures[future], future.result()
                bar.update()
                if judgment is None:
                    verdicts[p["key"]] = None
                    continue
                verdicts[p["key"]] = final_verdict(judgment, p["excerpt"])
                cache.replace_one({"_id": p["key"]}, {
                    "model": model, "prompt_version": PROMPT_VERSION, "question_id": p["question_id"],
                    "chunk_id": p["chunk_id"], "verdict": verdicts[p["key"]], "raw_verdict": judgment.verdict,
                    "reason": judgment.reason, "quote": judgment.quote, "judged_at": datetime.now(UTC)}, upsert=True)
        except KeyboardInterrupt:  # Ctrl+C: drop the queued calls instead of waiting for all of them
            pool.shutdown(cancel_futures=True)
            raise
    return verdicts