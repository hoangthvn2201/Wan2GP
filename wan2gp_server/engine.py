"""Generation engine: a lazy WanGP session + a single worker thread.

WanGP's `shared.api.WanGPSession` refuses to run two generations at once
(`RuntimeError: WanGP session already has a generation in progress`), and
the GPU could not serve them anyway — so the engine serializes everything
through one queue + one worker thread. The session (and the model weights
it keeps in VRAM) lives for the whole process.
"""

import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ServerConfig
from .jobs import Job, JobStatus, JobStore

logger = logging.getLogger("wan2gp_server")


class _JobCallbacks:
    """Bridge WanGP progress events into the Job's progress dict."""

    def __init__(self, job: Job):
        self._job = job

    def on_progress(self, update: Any) -> None:
        self._job.progress = {
            "phase": getattr(update, "phase", None),
            "status": getattr(update, "status", None),
            "percent": int(getattr(update, "progress", 0) or 0),
            "current_step": getattr(update, "current_step", None),
            "total_steps": getattr(update, "total_steps", None),
        }

    def on_status(self, text: str) -> None:
        self._job.progress = {**self._job.progress, "status": str(text or "").strip()}


class Wan2GPEngine:
    def __init__(self, config: ServerConfig, store: JobStore):
        self.config = config
        self.store = store
        self._session = None
        self._session_lock = threading.Lock()
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()
        self._shutdown = threading.Event()

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    @property
    def runtime_loaded(self) -> bool:
        return self._session is not None

    def ensure_session(self):
        """Get or lazily create the WanGP session (thread-safe, slow on first call)."""
        with self._session_lock:
            if self._session is None:
                root = self.config.wan2gp_root
                if not (root / "wgp.py").exists():
                    raise RuntimeError(
                        f"Wan2GP installation not found at '{root}' (wgp.py is missing). "
                        f"Set the WAN2GP_ROOT environment variable."
                    )
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))

                from shared.api import init  # WanGP official Python API

                logger.info("Initializing WanGP session (root=%s, cli_args=%s)", root, self.config.cli_args)
                init_kwargs: Dict[str, Any] = {"root": root, "cli_args": self.config.cli_args}
                if self.config.output_dir is not None:
                    self.config.output_dir.mkdir(parents=True, exist_ok=True)
                    init_kwargs["output_dir"] = self.config.output_dir
                self._session = init(**init_kwargs)
                logger.info("WanGP session ready")
            return self._session

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._worker_lock:
            if self._worker is None or not self._worker.is_alive():
                self._shutdown.clear()
                self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="wan2gp-engine")
                self._worker.start()
        if self.config.eager_init:
            threading.Thread(target=self._eager_init, daemon=True, name="wan2gp-eager-init").start()

    def _eager_init(self) -> None:
        try:
            self.ensure_session()
        except Exception:
            logger.exception("Eager WanGP session init failed (will retry on first job)")

    def stop(self) -> None:
        self._shutdown.set()

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job._cancel_requested:
                job.finish(JobStatus.CANCELLED, error="Cancelled while queued")
                continue
            self._run_job(job)

    def _run_job(self, job: Job) -> None:
        import time

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress = {"phase": "loading_model", "status": "Preparing generation...", "percent": 0}
        logger.info(
            "Job %s started: task=%s model_type=%s resolution=%s video_length=%s",
            job.id, job.task, job.settings.get("model_type"),
            job.settings.get("resolution"), job.settings.get("video_length", "-"),
        )
        try:
            session = self.ensure_session()
            session_job = session.submit_task(dict(job.settings), callbacks=_JobCallbacks(job))
            job._session_job = session_job
            if job._cancel_requested:  # cancel arrived between queue pop and submit
                session_job.cancel()
            result = session_job.result()

            if getattr(result, "cancelled", False) or job._cancel_requested:
                job.finish(JobStatus.CANCELLED, files=list(result.generated_files), error="Cancelled")
            elif result.success:
                if job.task == "preload":
                    # Warmup output is a throwaway: the point was downloading
                    # the checkpoint and loading the weights into VRAM.
                    for path in result.generated_files:
                        try:
                            Path(path).unlink(missing_ok=True)
                        except OSError:
                            logger.warning("Could not remove warmup output %s", path)
                    job.finish(JobStatus.SUCCEEDED)
                    logger.info("Job %s succeeded: model '%s' preloaded", job.id, job.model)
                elif not result.generated_files:
                    job.finish(JobStatus.FAILED, error="Generation completed but produced no output files")
                else:
                    job.finish(JobStatus.SUCCEEDED, files=list(result.generated_files))
                    logger.info("Job %s succeeded: %s", job.id, job.files)
            else:
                messages = "; ".join(e.message for e in result.errors) or "Unknown generation error"
                job.finish(JobStatus.FAILED, files=list(result.generated_files), error=messages)
                logger.warning("Job %s failed: %s", job.id, messages)
        except Exception as exc:
            logger.exception("Job %s crashed", job.id)
            job.finish(JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
        finally:
            job._session_job = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, job: Job) -> Job:
        self.store.add(job)
        self._queue.put(job)
        return job

    def cancel(self, job: Job) -> Job:
        job._cancel_requested = True
        if job.status is JobStatus.RUNNING and job._session_job is not None:
            try:
                job._session_job.cancel()  # cooperative; worker observes the result
            except Exception:
                logger.exception("Cancellation of job %s raised", job.id)
        # Queued jobs are finalized by the worker when popped; nothing else to do.
        return job
