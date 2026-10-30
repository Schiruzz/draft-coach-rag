"""Cut a transcript into time windows: fixed (step 3s) or inside the chapters (step 5, the method in use since 30/10)."""

WINDOW_METHOD = "window60-v1"                # A: fixed windows over the whole transcript (main until 30/10)
CHAPTER_METHOD = "chapter60-v1"              # B: windows that never cross a chapter, title in the text (lost 28/10)
PLAIN_CHAPTER_METHOD = "chapter60-plain-v1"  # B-plain: same windows, title kept out of the text (won 30/10)
METHOD = PLAIN_CHAPTER_METHOD                # the method in use: stored on every chunk, default of every script
METHODS = [WINDOW_METHOD, CHAPTER_METHOD, PLAIN_CHAPTER_METHOD]  # all kept in MongoDB and Qdrant, reproducible
WINDOW_S = 60           # a window covers segments starting within 60 s of its first segment
OVERLAP_S = 15          # the next window starts 15 s before the end of the previous one


def make_windows(segments: list[dict], window_s: int = WINDOW_S, overlap_s: int = OVERLAP_S) -> list[dict]:
    """[{start_s, text, mentions}] -> [{start_s, end_s, text, mentions}]; segments must be sorted by start_s."""
    windows, i = [], 0
    while i < len(segments):
        start = segments[i]["start_s"]
        j = i
        while j < len(segments) and segments[j]["start_s"] < start + window_s:
            j += 1
        part = segments[i:j]
        end = segments[j]["start_s"] if j < len(segments) else part[-1]["start_s"]  # last window: end unknown
        windows.append({
            "start_s": start,
            "end_s": end,
            "text": " ".join(segment["text"] for segment in part),
            "mentions": list(dict.fromkeys(m for segment in part for m in segment["mentions"])),
        })
        if j == len(segments):
            break
        # first segment inside the overlap; at least one step forward, so the loop always ends
        i = next(k for k in range(i + 1, j + 1) if segments[k]["start_s"] >= end - overlap_s)
    return windows


def make_chapter_windows(segments: list[dict], chapters: list[dict], title_in_text: bool = True,
                         window_s: int = WINDOW_S, overlap_s: int = OVERLAP_S) -> list[dict]:
    """Time windows that never cross a chapter; with title_in_text, each text starts with its chapter title.
    segments sorted by start_s; chapters [{start_s, title}] sorted, the first at 0."""
    windows = []
    for k, chapter in enumerate(chapters):
        end = chapters[k + 1]["start_s"] if k + 1 < len(chapters) else None  # last chapter: end unknown
        part = [s for s in segments if s["start_s"] >= chapter["start_s"] and (end is None or s["start_s"] < end)]
        if not part:
            continue
        chapter_windows = make_windows(part, window_s, overlap_s)
        if end is not None:
            chapter_windows[-1]["end_s"] = end  # the chapter end is known: better than the last segment start
        for window in chapter_windows:
            window["chapter"] = chapter["title"]
            if title_in_text:
                window["text"] = f"{chapter['title']}: {window['text']}"  # context for the embedding
        windows += chapter_windows
    return windows