"""Pure logic: timestamp parsing, SRT rendering, chunk planning.

No I/O, no torch — importable and testable without a GPU or ffmpeg.
"""
from __future__ import annotations

import re

# "MM:SS.d - MM:SS.d: description", accepting en-dash, em-dash or hyphen.
LINE_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*[–—-]\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*:\s*(.+?)\s*$"
)


def _mmss_to_seconds(minutes: str, seconds: str) -> float:
    return int(minutes) * 60 + float(seconds)


def format_mmss(t: float) -> str:
    if t < 0:
        t = 0.0
    minutes = int(t // 60)
    return f"{minutes:02d}:{t - minutes * 60:05.2f}"


def parse_hand_ego_lines(text: str) -> list[tuple[float, float, str]]:
    """Model output -> [(start_s, end_s, caption)], junk lines dropped."""
    text = text.split("</think>", 1)[-1]  # drop reasoning if the model emitted any
    events = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        events.append((_mmss_to_seconds(m.group(1), m.group(2)),
                       _mmss_to_seconds(m.group(3), m.group(4)),
                       m.group(5).strip()))
    return events


def _stamp(t: float) -> str:  # HH:MM:SS,mmm
    if t < 0:
        t = 0.0
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def events_to_srt(events) -> str:
    return "\n\n".join(f"{i}\n{_stamp(s)} --> {_stamp(e)}\n{c}"
                       for i, (s, e, c) in enumerate(events, start=1))


def plan_chunks(duration: float, chunk_seconds: float, overlap_seconds: float):
    """(start, end) per chunk, consecutive chunks overlapping. A would-be tiny
    final chunk (< 2x overlap left) is merged into the previous one."""
    if chunk_seconds <= overlap_seconds:
        raise ValueError("chunk_seconds must exceed overlap_seconds")
    chunks = []
    start = 0.0
    while True:
        end = min(start + chunk_seconds, duration)
        if 0 < duration - end < overlap_seconds * 2:
            end = duration
        chunks.append((start, end))
        if end >= duration - 1e-6:
            break
        start = end - overlap_seconds
    return chunks


def build_chunk_prompt(chunk_user_text: str, global_caption: str, previous_summary: str | None,
                       offset: float, is_last_chunk: bool) -> str:
    parts = [chunk_user_text,
             f"\n\nGlobal context for the entire video: {global_caption}"]
    if previous_summary:
        parts.append(
            "\n\nContext from the immediately preceding chunk (for continuity "
            f"only, do not repeat it as its own line): {previous_summary}")
    if offset > 0:
        parts.append(
            f"\n\nThe first {offset:.2f} seconds of this clip overlap with the "
            "previous chunk and were already captioned there. Do NOT output "
            f"any line that lies entirely within [00:00.00, {format_mmss(offset)}); "
            f"your first timestamped line must start at or after {format_mmss(offset)}.")
    if not is_last_chunk:
        parts.append(
            "\n\nThis clip is cut from a longer video. If the final action is "
            "still in progress and does not clearly conclude before the clip "
            "ends, omit that last line entirely — it will be captioned in "
            "full in the next chunk.")
    return "".join(parts)


def selfcheck():
    """assert-based check of the pure logic (no model, no ffmpeg, no torch)."""
    chunks = plan_chunks(24.7, 20.0, 1.5)
    assert chunks[0] == (0.0, 20.0) and abs(chunks[1][0] - 18.5) < 1e-9
    assert plan_chunks(21.0, 20.0, 1.5) == [(0.0, 21.0)]
    assert plan_chunks(10.0, 20.0, 1.5) == [(0.0, 10.0)]

    txt = ("preamble\n00:01.2 – 00:03.4: [left hand] grasp cup | [ego] stay still\n"
           "00:03.4 - 00:05.0: [right hand] pour | [ego] stay still\nnot a line")
    evs = parse_hand_ego_lines("thinking...</think>" + txt)
    assert len(evs) == 2 and evs[0][0] == 1.2 and evs[1][1] == 5.0

    srt = events_to_srt([(0.0, 2.831, "[left hand] grasp domino | [ego] stay still")])
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,831\n[left hand] grasp domino")
    assert format_mmss(61.25) == "01:01.25"

    assert "overlap" not in build_chunk_prompt("d", "g", None, 0.0, False)
    assert "00:01.50" in build_chunk_prompt("d", "g", "prev line", 1.5, False)

    print("core.timing selfcheck OK")
