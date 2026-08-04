"""Sequential in-process job queue.

One GPU, one model instance -> jobs run one at a time in a single worker
thread. Good enough for a small internal demo; swap for Celery/RQ if this ever
needs to scale past one process.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Job:
    id: str
    video_path: Path
    instruction: str
    status: str = "queued"          # queued -> running -> done | error
    created_at: float = field(default_factory=time.time)
    progress: str = ""
    result: dict | None = None
    error: str | None = None


class JobQueue:
    def __init__(self, worker_fn: Callable[[Job], dict]):
        """worker_fn(job) -> result dict, called on the single worker thread."""
        self._worker_fn = worker_fn
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def submit(self, video_path: Path, instruction: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], video_path=video_path, instruction=instruction)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs[job_id]
                job.status = "running"
            try:
                result = self._worker_fn(job)
                with self._lock:
                    job.result = result
                    job.status = "done"
            except Exception as ex:
                with self._lock:
                    job.error = f"{type(ex).__name__}: {ex}"
                    job.status = "error"
