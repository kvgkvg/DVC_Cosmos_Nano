"""Pipeline defaults and knobs, in one place so infer.py and server/app.py agree.

Values are the same 2-stage Cosmos3-Nano rollout as demo_cosmos/infer_cosmos.py:
Stage 1 samples frames uniformly for a short global caption; stage 2 chunks the
video with overlap and captions each chunk with that global caption as context.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Sampling params, fixed seed: keeps numbers comparable across runs.
GEN_KWARGS = dict(do_sample=True, temperature=0.7, top_p=0.8)
SEED = 0

DEFAULT_MODEL_ID = "nvidia/Cosmos3-Nano"

GLOBAL_SYSTEM_PROMPT = "You are a helpful assistant specialized in video captioning."
GLOBAL_USER_PROMPT = (
    "These images are frames sampled uniformly, in chronological order, across "
    "an entire video. Based on them, write a single global caption describing "
    "the overall content of the video. Use at most 3 sentences. Do not include "
    "timestamps."
)

# Per-chunk task text. The full output-format contract lives in the system prompt file.
CHUNK_USER_TEXT = "Describe the person's actions."


@dataclass
class PipelineConfig:
    chunk_seconds: float = 20.0
    overlap_seconds: float = 1.5
    chunk_fps: float = 2.0
    global_num_frames: int = 10
    global_max_tokens: int = 256
    chunk_max_tokens: int = 1024
    model_id: str = DEFAULT_MODEL_ID
    gen_kwargs: dict = field(default_factory=lambda: dict(GEN_KWARGS))
    seed: int = SEED

    def __post_init__(self):
        if self.chunk_seconds <= self.overlap_seconds:
            raise ValueError("chunk_seconds must exceed overlap_seconds")
