"""Cut a transcript into fixed time windows (step 3s: the simplest chunking, to be compared at step 5)."""

METHOD = "window60-v1"  # stored on every chunk; change it when the windowing changes
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