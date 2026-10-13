"""Parse a transcript copied from YouTube's 'Show transcript' panel."""
import re

# A line made only of m:ss or h:mm:ss marks the start of the next segment
TIMESTAMP = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")


def parse_transcript(raw: str) -> list[dict]:
    """Return [{"start_s": int, "text": str}]; text before the first timestamp starts at 0."""
    segments = [{"start_s": 0, "text": ""}]
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        match = TIMESTAMP.match(line)
        if match:
            hours, minutes, seconds = match.groups()
            start = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
            segments.append({"start_s": start, "text": ""})
        else:
            segments[-1]["text"] = f"{segments[-1]['text']} {line}".strip()
    return [segment for segment in segments if segment["text"]]