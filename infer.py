#!/usr/bin/env python3
"""CLI over core/: run the Cosmos3-Nano captioning pipeline from the command line.

Batch mode (a manifest of episodes -> a predictions JSONL, resumable):

  python infer.py batch --manifest val.jsonl --prompt system_prompt.txt \\
      --out preds/cosmos3-nano_val.jsonl

Single-video mode (one .mp4 -> one .srt on stdout or --out):

  python infer.py caption --video clip.mp4 --instruction "fold the towel" \\
      --prompt system_prompt.txt

Logic-only check, no model and no ffmpeg needed:

  python infer.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from core.config import PipelineConfig
from core.pipeline import CosmosCaptioner
from core.timing import selfcheck as timing_selfcheck


def load_system_prompt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_manifest(path: Path, sample: int | None, limit: int | None, seed: int) -> list[dict]:
    episodes = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if sample is not None:
        episodes = random.Random(seed).sample(episodes, min(sample, len(episodes)))
    elif limit is not None:
        episodes = episodes[:limit]
    return episodes


def done_uids(out: Path) -> set[str]:
    """episode_uids already written to --out, so a re-run resumes."""
    if not out.exists():
        return set()
    return {json.loads(l)["episode_uid"] for l in out.read_text().splitlines() if l.strip()}


def remap_videos(episodes: list[dict], mappings: list[str]) -> None:
    """Rewrite video path prefixes in place (OLD=NEW), longest prefix first."""
    pairs = []
    for mapping in mappings:
        if "=" not in mapping:
            sys.exit(f"--video-root-map needs OLD=NEW, got {mapping!r}")
        pairs.append(tuple(mapping.split("=", 1)))
    pairs.sort(key=lambda p: -len(p[0]))
    for e in episodes:
        for old, new in pairs:
            if e["video"].startswith(old):
                e["video"] = new + e["video"][len(old):]
                break


def config_from_args(args) -> PipelineConfig:
    return PipelineConfig(
        chunk_seconds=args.chunk_seconds, overlap_seconds=args.overlap_seconds,
        chunk_fps=args.chunk_fps, global_num_frames=args.global_num_frames,
        global_max_tokens=args.global_max_tokens, chunk_max_tokens=args.chunk_max_tokens,
        model_id=args.model)


def add_pipeline_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--prompt", type=Path, default=Path(__file__).with_name("system_prompt.txt"),
                    help="chunk system prompt file (default: ./system_prompt.txt)")
    ap.add_argument("--model", default="nvidia/Cosmos3-Nano",
                    help="HF id or local dir (default: %(default)s)")
    ap.add_argument("--chunk-seconds", type=float, default=20.0)
    ap.add_argument("--overlap-seconds", type=float, default=1.5)
    ap.add_argument("--chunk-fps", type=float, default=2.0,
                    help="fps passed to the processor for chunk videos")
    ap.add_argument("--global-num-frames", type=int, default=10)
    ap.add_argument("--global-max-tokens", type=int, default=256)
    ap.add_argument("--chunk-max-tokens", type=int, default=1024)


def cmd_batch(args) -> None:
    sys_prompt = load_system_prompt(args.prompt)
    episodes = load_manifest(args.manifest, args.sample, args.limit, args.seed)
    remap_videos(episodes, args.video_root_map)

    missing = [e["video"] for e in episodes if not Path(e["video"]).exists()]
    if missing:
        sys.exit(f"{len(missing)}/{len(episodes)} videos do not exist, e.g. {missing[0]}\n"
                 "Manifests store absolute paths from the curating machine; remap them "
                 "with --video-root-map OLD=NEW (repeatable).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    already = done_uids(args.out)
    todo = [e for e in episodes if e["episode_uid"] not in already]
    print(f"manifest={args.manifest.name} prompt={args.prompt.name} "
          f"episodes={len(episodes)} done={len(already)} todo={len(todo)}")
    if not todo:
        print("nothing to do (all episodes already in --out)")
        return

    print(f"loading {args.model} (may take a while)...")
    captioner = CosmosCaptioner(config_from_args(args))
    captioner.load()
    print(f"model on {captioner.device} | chunk={args.chunk_seconds}s "
          f"overlap={args.overlap_seconds}s fps={args.chunk_fps}")

    n_ok = n_err = 0
    t0 = time.time()
    with open(args.out, "a") as fh:
        for i, e in enumerate(todo, 1):
            try:
                result = captioner.caption(
                    e["video"], sys_prompt, instruction=e.get("instruction", ""),
                    duration_s=e.get("duration_s"))
                rec = {"episode_uid": e["episode_uid"], "pred": result["pred"],
                       "n_events": result["n_events"], "n_chunks": result["n_chunks"]}
            except Exception as ex:
                rec = {"episode_uid": e["episode_uid"], "pred": "",
                       "error": f"{type(ex).__name__}: {ex}"}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()                            # crash-safe: never lose finished episodes
            if rec.get("error"):
                n_err += 1
                print(f"  [{i}/{len(todo)}] ERR {rec['episode_uid']}: {rec['error']}", flush=True)
            else:
                n_ok += 1
                print(f"  [{i}/{len(todo)}] ok={n_ok} err={n_err} "
                      f"events={rec['n_events']} chunks={rec['n_chunks']} "
                      f"({(time.time() - t0) / i:.1f} s/ep)", flush=True)

    print(f"done: {n_ok} ok, {n_err} err -> {args.out}")


def cmd_caption(args) -> None:
    sys_prompt = load_system_prompt(args.prompt)
    if not Path(args.video).exists():
        sys.exit(f"video does not exist: {args.video}")

    print(f"loading {args.model} (may take a while)...", file=sys.stderr)
    captioner = CosmosCaptioner(config_from_args(args))
    captioner.load()

    def on_chunk(i, n):
        print(f"  chunk {i + 1}/{n} done", file=sys.stderr)

    result = captioner.caption(args.video, sys_prompt, instruction=args.instruction,
                               on_chunk=on_chunk)
    print(f"events={result['n_events']} chunks={result['n_chunks']} "
          f"global_caption={result['global_caption']!r}", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(result["pred"])
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(result["pred"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true", help="logic assertions only, then exit")
    sub = ap.add_subparsers(dest="command")

    p_batch = sub.add_parser("batch", help="manifest JSONL -> predictions JSONL")
    p_batch.add_argument("--manifest", type=Path, required=True,
                         help="episodes JSONL (episode_uid, video, instruction, duration_s)")
    p_batch.add_argument("--out", type=Path, required=True, help="predictions JSONL (appended)")
    p_batch.add_argument("--sample", type=int, help="random K episodes (reproducible via --seed)")
    p_batch.add_argument("--limit", type=int, help="first N episodes (ignored if --sample)")
    p_batch.add_argument("--seed", type=int, default=42)
    p_batch.add_argument("--video-root-map", action="append", default=[], metavar="OLD=NEW",
                         help="rewrite manifest video path prefixes (repeatable)")
    add_pipeline_flags(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    p_caption = sub.add_parser("caption", help="one .mp4 -> one .srt")
    p_caption.add_argument("--video", required=True, help="path to the .mp4")
    p_caption.add_argument("--instruction", default="", help="task description (optional)")
    p_caption.add_argument("--out", type=Path, help="write SRT here instead of stdout")
    add_pipeline_flags(p_caption)
    p_caption.set_defaults(func=cmd_caption)

    args = ap.parse_args()

    if args.selfcheck:
        timing_selfcheck()
        return
    if not args.command:
        ap.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
