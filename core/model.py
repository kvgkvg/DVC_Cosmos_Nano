"""Model loading and generation. torch/transformers imported lazily so the rest
of core/ (and --selfcheck) works without a GPU or those packages installed."""
from __future__ import annotations

from pathlib import Path

from .config import GLOBAL_SYSTEM_PROMPT, GLOBAL_USER_PROMPT, PipelineConfig
from .ffmpeg_utils import extract_n_frames


def load_model(model_id: str):
    import torch
    from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Cosmos3OmniForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return processor, model


def run_reasoner(processor, model, messages, max_new_tokens: int,
                 seed: int, gen_kwargs: dict, fps: float | None = None) -> str:
    import torch

    kw = dict(tokenize=True, add_generation_prompt=True, return_dict=True,
              return_tensors="pt")
    if fps is not None:
        kw["fps"] = fps
    inputs = processor.apply_chat_template(messages, **kw).to(model.device, torch.bfloat16)
    torch.manual_seed(seed)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def get_global_caption(processor, model, video_path: Path, duration: float,
                       config: PipelineConfig, tmp_dir: Path) -> str:
    frames = extract_n_frames(video_path, duration, config.global_num_frames, tmp_dir)
    content = [{"type": "image", "path": str(f)} for f in frames]
    content.append({"type": "text", "text": GLOBAL_USER_PROMPT})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": GLOBAL_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return run_reasoner(processor, model, messages, config.global_max_tokens,
                        config.seed, config.gen_kwargs).strip()


def caption_chunk(processor, model, system_content: str, chunk_path: Path,
                  user_text: str, config: PipelineConfig) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_content}]},
        {"role": "user", "content": [
            {"type": "video", "path": str(chunk_path)},
            {"type": "text", "text": user_text},
        ]},
    ]
    return run_reasoner(processor, model, messages, config.chunk_max_tokens,
                        config.seed, config.gen_kwargs, fps=config.chunk_fps)
