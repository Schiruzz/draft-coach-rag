"""Parse a transcript copied from YouTube's transcript panel (two copy formats)."""
import re

# Format A: "1:04" alone on a line, text on the following lines.
# Format B: "1:041 minute, 4 secondsSo, Edgar..." (timestamp, spoken label and text on one line).
TIMESTAMP = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{2})"                              # h:mm:ss or m:ss
    r"(?:\d+ hours?,? ?)?(?:\d+ minutes?,? ?)?(?:\d+ seconds?)?"  # format B spoken label
    r"(.*)$"                                                      # format B text (empty in format A)
)
CHAPTER = re.compile(r"^Chapter \d+:")  # format B chapter lines: chapters live in sources.yaml


def parse_transcript(raw: str) -> list[dict]:
    """Return [{"start_s": int, "text": str}]; text before the first timestamp starts at 0."""
    segments = [{"start_s": 0, "text": ""}]
    for line in raw.splitlines():
        line = line.strip()
        if not line or CHAPTER.match(line):
            continue
        match = TIMESTAMP.match(line)
        if match:
            hours, minutes, seconds, text = match.groups()
            start = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
            segments.append({"start_s": start, "text": text.strip()})
        else:
            segments[-1]["text"] = f"{segments[-1]['text']} {line}".strip()
    return [segment for segment in segments if segment["text"]]
