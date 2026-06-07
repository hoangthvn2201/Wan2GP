"""In-memory job model and store.

One process == one job store. Jobs are processed strictly one at a time
by the engine worker (the underlying WanGP session refuses concurrent
generations), so the store also answers queue-position queries.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def media_type_of(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "unknown"


@dataclass
class Job:
    task: str                                  # "text_to_image" | "text_to_video" | "image_to_video" | "preload"
    model: str                                 # preset id
    settings: Dict[str, Any]                   # resolved WanGP task settings
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.QUEUED
    progress: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)   # absolute output paths
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # internals
    _done_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)
    _session_job: Any = field(default=None, repr=False)  # shared.api.SessionJob once running

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the job reaches a terminal state. Returns True if it did."""
        return self._done_event.wait(timeout)

    def finish(self, status: JobStatus, *, files: Optional[List[str]] = None, error: Optional[str] = None) -> None:
        self.status = status
        if files:
            self.files = list(files)
        self.error = error
        self.finished_at = time.time()
        self._done_event.set()


class JobStore:
    def __init__(self, max_jobs: int = 500):
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            # Evict oldest terminal jobs beyond the cap
            while len(self._order) > self._max_jobs:
                for jid in self._order:
                    if self._jobs[jid].status.terminal:
                        self._order.remove(jid)
                        del self._jobs[jid]
                        break
                else:
                    break  # nothing evictable

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> List[Job]:
        with self._lock:
            return [self._jobs[jid] for jid in reversed(self._order[-limit:])]

    def queue_position(self, job: Job) -> Optional[int]:
        """0-based position among queued jobs; None unless status=queued."""
        if job.status is not JobStatus.QUEUED:
            return None
        with self._lock:
            queued = [jid for jid in self._order if self._jobs[jid].status is JobStatus.QUEUED]
        try:
            return queued.index(job.id)
        except ValueError:
            return None

    def queued_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status is JobStatus.QUEUED)

    def active_job(self) -> Optional[Job]:
        with self._lock:
            for j in self._jobs.values():
                if j.status is JobStatus.RUNNING:
                    return j
        return None
