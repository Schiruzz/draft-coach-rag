"""Retrieval judge: does an excerpt (a chunk) support the reference answer of an evaluation question?
Builds the prompt with facts from our tables, asks Gemini, and checks that the quote is really in the excerpt."""
import hashlib
import re
import time
from typing import Literal

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

PROMPT_VERSION = "judge-v2"  # bump when the prompt changes: cached judgments of older versions are not reused
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


def judgment_id(question_id: str, reference: str, excerpt: str) -> str:
    """Same question, reference, excerpt and prompt version -> same id: a judgment is asked once."""
    key = "|".join([PROMPT_VERSION, question_id, reference, excerpt])
    return hashlib.sha256(key.encode()).hexdigest()


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