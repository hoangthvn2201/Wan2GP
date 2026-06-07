"""FastAPI application exposing WanGP generation as a REST API.

Endpoints (all JSON unless noted):

    GET  /health                              Liveness + runtime/queue state
    GET  /v1/models                           List model presets (?task=t2i|t2v|i2v)
    POST /v1/models/preload                   Warm up models (download checkpoints + load weights)
    POST /v1/generations/text-to-image        Submit a text-to-image job
    POST /v1/generations/text-to-video        Submit a text-to-video job
    POST /v1/generations/image-to-video       Submit an image-to-video job
    POST /v1/assets                           Upload an input image (multipart) -> asset_id
    GET  /v1/jobs                             List recent jobs
    GET  /v1/jobs/{job_id}                    Job status + progress + files
    POST /v1/jobs/{job_id}/cancel             Cancel a queued/running job
    GET  /v1/jobs/{job_id}/files/{index}      Download an output file

Designed for agent use: every submit returns a Job object whose `files[].url`
can be fetched once `status == "succeeded"`. Set `"wait": true` in the request
body to block until completion instead of polling.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from . import __version__ as VERSION
from .config import ServerConfig
from .engine import Wan2GPEngine
from .jobs import Job, JobStatus, JobStore, media_type_of
from .media import ImageInputError, resolve_image_input, save_upload
from .presets import ModelPreset, PresetRegistry, build_settings, build_warmup_settings
from .schemas import (
    AssetInfo,
    HealthResponse,
    ImageToVideoRequest,
    JobFile,
    JobInfo,
    JobListResponse,
    JobProgress,
    ModelInfo,
    ModelListResponse,
    PreloadRequest,
    PreloadResponse,
    TextToImageRequest,
    TextToVideoRequest,
)

logger = logging.getLogger("wan2gp_server")


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    config = config or ServerConfig.from_env()
    store = JobStore()
    engine = Wan2GPEngine(config, store)
    presets = PresetRegistry(config.presets_dir)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.assets_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        yield
        engine.stop()

    app = FastAPI(
        title="Wan2GP Server",
        version=VERSION,
        description=__doc__,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.engine = engine
    app.state.store = store
    app.state.presets = presets

    # ------------------------------------------------------------------
    # Auth (optional)
    # ------------------------------------------------------------------

    async def require_api_key(request: Request) -> None:
        if config.api_key and request.headers.get("X-API-Key") != config.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def job_info(job: Job) -> JobInfo:
        return JobInfo(
            id=job.id,
            task=job.task,
            status=job.status.value,
            model=job.model,
            queue_position=store.queue_position(job),
            progress=JobProgress(**job.progress) if job.progress else JobProgress(),
            files=[
                JobFile(
                    index=i,
                    media_type=media_type_of(path),
                    filename=Path(path).name,
                    path=path,
                    url=f"/v1/jobs/{job.id}/files/{i}",
                )
                for i, path in enumerate(job.files)
            ],
            error=job.error,
            settings={k: v for k, v in job.settings.items() if k != "_api"},
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    def resolve_preset(model: Optional[str], task: str) -> ModelPreset:
        preset_id = model or config.default_models[task]
        try:
            return presets.get(preset_id, task=task)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def submit_and_maybe_wait(job: Job, wait: bool, wait_timeout: float) -> JobInfo:
        engine.submit(job)
        if wait:
            finished = await run_in_threadpool(job.wait, wait_timeout)
            if not finished:
                logger.info("Job %s wait timed out after %.0fs (job keeps running)", job.id, wait_timeout)
        return job_info(job)

    # ------------------------------------------------------------------
    # Meta endpoints
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        active = store.active_job()
        return HealthResponse(
            version=VERSION,
            wan2gp_root=str(config.wan2gp_root),
            runtime_loaded=engine.runtime_loaded,
            active_job_id=active.id if active else None,
            queued_jobs=store.queued_count(),
        )

    @app.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(require_api_key)])
    async def list_models(
        task: Optional[str] = Query(None, pattern="^(t2i|t2v|i2v)$", description="Filter by task"),
    ) -> ModelListResponse:
        return ModelListResponse(
            models=[
                ModelInfo(
                    id=p.id,
                    task=p.task,
                    model_type=p.model_type,
                    description=p.description,
                    media_type=p.media_type,
                    fps=p.fps,
                    max_pixels=p.max_pixels,
                    max_frames=p.max_frames,
                    is_default=(config.default_models.get(p.task) == p.id),
                )
                for p in presets.list(task)
            ]
        )

    @app.post("/v1/models/preload", response_model=PreloadResponse, dependencies=[Depends(require_api_key)])
    async def preload_models(req: PreloadRequest) -> PreloadResponse:
        """Preload models by running a minimal warmup generation per model.

        Each warmup goes through WanGP's normal generation path, which
        downloads the checkpoint (first time) and loads the weights into
        VRAM. Models are warmed FIFO in the order given; only the LAST one
        stays in VRAM (WanGP keeps a single model loaded), but every listed
        model ends up with its checkpoint cached on disk. The throwaway
        warmup output is deleted automatically.
        """
        # Resolve preset ids: explicit list, or the server defaults
        preset_ids = req.models or [
            config.default_models[task] for task in ("t2i", "t2v", "i2v")
        ]
        resolved: list[ModelPreset] = []
        seen_model_types = set()
        for preset_id in preset_ids:
            try:
                preset = presets.get(preset_id)
            except KeyError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if preset.model_type in seen_model_types:
                continue  # same weights; warming once is enough
            seen_model_types.add(preset.model_type)
            resolved.append(preset)

        # i2v models need a start image even for a warmup
        warmup_image: Optional[Path] = None
        if any(p.task == "i2v" for p in resolved):
            warmup_image = config.data_dir / "inputs" / "warmup.png"
            if not warmup_image.is_file():
                from PIL import Image

                warmup_image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (256, 256), "black").save(warmup_image)

        jobs = []
        for preset in resolved:
            settings = build_warmup_settings(
                preset,
                image_start=str(warmup_image) if preset.task == "i2v" else None,
            )
            jobs.append(engine.submit(Job(task="preload", model=preset.id, settings=settings)))

        if req.wait:
            deadline = req.wait_timeout_seconds
            import time as _time

            start = _time.monotonic()
            for job in jobs:
                remaining = max(0.1, deadline - (_time.monotonic() - start))
                finished = await run_in_threadpool(job.wait, remaining)
                if not finished:
                    logger.info("Preload wait timed out at job %s (jobs keep running)", job.id)
                    break
        return PreloadResponse(jobs=[job_info(j) for j in jobs])

    # ------------------------------------------------------------------
    # Generation endpoints
    # ------------------------------------------------------------------

    @app.post("/v1/generations/text-to-image", response_model=JobInfo, dependencies=[Depends(require_api_key)])
    async def text_to_image(req: TextToImageRequest) -> JobInfo:
        preset = resolve_preset(req.model, "t2i")
        settings = build_settings(
            preset,
            req.prompt,
            width=req.width,
            height=req.height,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            extra_settings=req.settings,
        )
        job = Job(task="text_to_image", model=preset.id, settings=settings)
        return await submit_and_maybe_wait(job, req.wait, req.wait_timeout_seconds)

    @app.post("/v1/generations/text-to-video", response_model=JobInfo, dependencies=[Depends(require_api_key)])
    async def text_to_video(req: TextToVideoRequest) -> JobInfo:
        preset = resolve_preset(req.model, "t2v")
        settings = build_settings(
            preset,
            req.prompt,
            width=req.width,
            height=req.height,
            duration_seconds=req.duration_seconds,
            num_frames=req.num_frames,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            extra_settings=req.settings,
        )
        job = Job(task="text_to_video", model=preset.id, settings=settings)
        return await submit_and_maybe_wait(job, req.wait, req.wait_timeout_seconds)

    @app.post("/v1/generations/image-to-video", response_model=JobInfo, dependencies=[Depends(require_api_key)])
    async def image_to_video(req: ImageToVideoRequest) -> JobInfo:
        preset = resolve_preset(req.model, "i2v")
        try:
            image_path, (img_w, img_h) = await run_in_threadpool(
                lambda: resolve_image_input(
                    data_dir=config.data_dir,
                    image_b64=req.image_b64,
                    image_url=req.image_url,
                    image_path=req.image_path,
                    image_asset_id=req.image_asset_id,
                )
            )
        except ImageInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        settings = build_settings(
            preset,
            req.prompt,
            width=req.width or img_w,
            height=req.height or img_h,
            duration_seconds=req.duration_seconds,
            num_frames=req.num_frames,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            image_start=str(image_path),
            extra_settings=req.settings,
        )
        job = Job(task="image_to_video", model=preset.id, settings=settings)
        return await submit_and_maybe_wait(job, req.wait, req.wait_timeout_seconds)

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------

    @app.post("/v1/assets", response_model=AssetInfo, dependencies=[Depends(require_api_key)])
    async def upload_asset(file: UploadFile) -> AssetInfo:
        data = await file.read()
        try:
            asset_id, path, _size = await run_in_threadpool(
                save_upload, data, config.assets_dir, file.filename
            )
        except ImageInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AssetInfo(
            asset_id=asset_id,
            filename=path.name,
            path=str(path),
            size_bytes=len(data),
        )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def get_job_or_404(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job

    @app.get("/v1/jobs", response_model=JobListResponse, dependencies=[Depends(require_api_key)])
    async def list_jobs(limit: int = Query(50, ge=1, le=500)) -> JobListResponse:
        return JobListResponse(jobs=[job_info(j) for j in store.list(limit)])

    @app.get("/v1/jobs/{job_id}", response_model=JobInfo, dependencies=[Depends(require_api_key)])
    async def get_job(job_id: str) -> JobInfo:
        return job_info(get_job_or_404(job_id))

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobInfo, dependencies=[Depends(require_api_key)])
    async def cancel_job(job_id: str) -> JobInfo:
        job = get_job_or_404(job_id)
        if job.status.terminal:
            raise HTTPException(status_code=409, detail=f"Job is already {job.status.value}")
        engine.cancel(job)
        return job_info(job)

    @app.get("/v1/jobs/{job_id}/files/{index}", dependencies=[Depends(require_api_key)])
    async def download_file(job_id: str, index: int) -> FileResponse:
        job = get_job_or_404(job_id)
        if index < 0 or index >= len(job.files):
            raise HTTPException(
                status_code=404,
                detail=f"Job '{job_id}' has {len(job.files)} file(s); index {index} is out of range",
            )
        path = Path(job.files[index])
        if not path.is_file():
            raise HTTPException(status_code=410, detail=f"Output file no longer exists: {path}")
        media = {
            "image": "image/png" if path.suffix.lower() == ".png" else "image/jpeg",
            "video": "video/mp4",
            "audio": "audio/mpeg",
        }.get(media_type_of(str(path)), "application/octet-stream")
        return FileResponse(path, media_type=media, filename=path.name)

    return app
