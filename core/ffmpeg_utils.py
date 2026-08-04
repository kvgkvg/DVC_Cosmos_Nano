"""ffmpeg / ffprobe wrappers: duration probing, frame extraction, chunk cutting."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def get_duration(video_path: Path) -> float:
    out = _run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ])
    return float(out.strip())


def extract_n_frames(video_path: Path, duration: float, n: int, out_dir: Path) -> list[Path]:
    """Exactly n frames spread uniformly (midpoint of each of n equal spans) —
    a fixed-fps filter cannot guarantee an exact count."""
    frame_paths = []
    for i in range(n):
        t = duration * (i + 0.5) / n
        out_path = out_dir / f"frame_{i:03d}.jpg"
        _run_cmd([
            "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out_path),
        ])
        frame_paths.append(out_path)
    return frame_paths


def extract_chunk(video_path: Path, start: float, end: float, out_path: Path) -> None:
    # -ss after -i for frame-accurate cuts (slower than -ss before -i, fine here)
    _run_cmd([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an",
        str(out_path),
    ])
