"""The 2-stage Cosmos3-Nano captioning pipeline as a reusable class.

Load once (CosmosCaptioner(config)), then call .caption(...) as many times as
needed — this is what lets infer.py batch over a manifest and server/app.py serve
many uploads without reloading the ~33GB model each time.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .config import CHUNK_USER_TEXT, PipelineConfig
from .ffmpeg_utils import extract_chunk, get_duration
from .model import caption_chunk, get_global_caption, load_model
from .timing import build_chunk_prompt, events_to_srt, parse_hand_ego_lines, plan_chunks


class CosmosCaptioner:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.processor = None
        self.model = None

    def load(self) -> None:
        if self.model is None:
            self.processor, self.model = load_model(self.config.model_id)

    @property
    def device(self):
        return self.model.device if self.model is not None else None

    def caption(self, video_path: str | Path, system_prompt: str,
               instruction: str = "", duration_s: float | None = None,
               on_chunk=None) -> dict:
        """Caption one video end to end.

        video_path    path to the .mp4
        system_prompt output-format contract (e.g. system_prompt.txt contents)
        instruction   task description appended as episode context (optional)
        duration_s    skips ffprobe if already known
        on_chunk      optional callback(i, n_chunks) for progress reporting

        Returns {"pred": <SRT text>, "n_events": N, "n_chunks": M}. Raises if
        no chunk produced a parseable line.
        """
        self.load()
        video_path = Path(video_path)
        cfg = self.config
        duration = duration_s or get_duration(video_path)

        system_content = system_prompt
        if instruction:
            system_content += (
                "\n\n## Episode context\n"
                "The camera-wearer is performing this task (background reference for "
                f"identifying objects and the setting):\n{instruction}")

        with tempfile.TemporaryDirectory(prefix="dvc_cosmos_nano_") as tmp_dir:
            global_caption = get_global_caption(
                self.processor, self.model, video_path, duration, cfg, Path(tmp_dir))

            chunks = plan_chunks(duration, cfg.chunk_seconds, cfg.overlap_seconds)
            all_events = []
            previous_summary = None

            for i, (start, end) in enumerate(chunks):
                is_last_chunk = (i == len(chunks) - 1)
                offset = cfg.overlap_seconds if i > 0 else 0.0

                if start == 0.0 and end >= duration - 1e-6:
                    chunk_path = video_path          # whole video fits in one chunk
                else:
                    chunk_path = Path(tmp_dir) / f"chunk_{i:03d}.mp4"
                    extract_chunk(video_path, start, end, chunk_path)

                user_text = build_chunk_prompt(
                    CHUNK_USER_TEXT, global_caption, previous_summary, offset, is_last_chunk)
                raw = caption_chunk(
                    self.processor, self.model, system_content, chunk_path, user_text, cfg)

                lines = parse_hand_ego_lines(raw)
                if on_chunk:
                    on_chunk(i, len(chunks))
                if not lines:
                    continue
                for local_start, local_end, caption in lines:
                    if local_end <= offset:          # entirely inside the overlap
                        continue
                    global_start = min(start + max(local_start, offset), end)
                    global_end = min(max(start + local_end, global_start), end)
                    all_events.append((global_start, global_end, caption))
                previous_summary = lines[-1][2]

        all_events.sort(key=lambda ev: ev[0])
        if not all_events:
            raise RuntimeError("no parseable timestamped lines in any chunk")
        return {"pred": events_to_srt(all_events),
                "n_events": len(all_events), "n_chunks": len(chunks),
                "global_caption": global_caption, "duration_s": duration}
