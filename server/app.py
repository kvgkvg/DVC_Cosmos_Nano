#!/usr/bin/env python3
"""FastAPI server around core/: upload a video, get back timestamped captions.

  uvicorn server.app:app --host 0.0.0.0 --port 8000

The model loads once at startup and is reused across requests. Uploads are
processed one at a time by a single background worker (server/jobs.py) — safe
for one GPU, no concurrent model.generate() calls.
"""
from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from core.config import PipelineConfig  # noqa: E402
from core.pipeline import CosmosCaptioner  # noqa: E402
from server.jobs import Job, JobQueue  # noqa: E402

UPLOAD_DIR = HERE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SYSTEM_PROMPT_PATH = ROOT / "system_prompt.txt"

app = FastAPI(title="dvc_cosmos_nano")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

captioner = CosmosCaptioner(PipelineConfig())
system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _run_job(job: Job) -> dict:
    return captioner.caption(job.video_path, system_prompt, instruction=job.instruction)


job_queue = JobQueue(_run_job)


@app.on_event("startup")
def _startup():
    print("loading model (may take a while)...")
    captioner.load()
    print(f"model on {captioner.device}, ready")
    job_queue.start()


@app.post("/api/jobs")
async def create_job(video: UploadFile = File(...), instruction: str = Form("")):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(video.file, fh)

    job = job_queue.submit(dest, instruction)
    return {"job_id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job id")
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "video_url": f"/api/jobs/{job.id}/video",
    }


@app.get("/api/jobs/{job_id}/video")
def get_job_video(job_id: str):
    job = job_queue.get(job_id)
    if job is None or not job.video_path.exists():
        raise HTTPException(404, "unknown job id")
    return FileResponse(job.video_path)


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
