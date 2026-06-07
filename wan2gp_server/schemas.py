"""Pydantic request / response schemas for the wan2gp_server API."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

JobStatusLiteral = Literal["queued", "running", "succeeded", "failed", "cancelled"]


# ----------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------

class _BaseGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Generation prompt")
    model: Optional[str] = Field(
        None,
        description="Model preset id (see GET /v1/models). Omit to use the server default.",
    )
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    steps: Optional[int] = Field(None, ge=1, le=100, description="Inference steps (model default if omitted)")
    seed: Optional[int] = Field(None, description="Random seed; -1 or omitted = random")
    guidance_scale: Optional[float] = Field(None, ge=0, description="CFG / guidance scale")
    settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw WanGP task settings merged last (expert use; same keys as WanGP 'Export Settings')",
    )
    wait: bool = Field(
        False,
        description="If true, the request blocks until the job finishes and returns the final job state. "
        "If false (default), it returns immediately with a queued job to poll via GET /v1/jobs/{id}.",
    )
    wait_timeout_seconds: float = Field(
        3600, gt=0, le=24 * 3600,
        description="Max time to block when wait=true (the job keeps running server-side on timeout)",
    )


class TextToImageRequest(_BaseGenerationRequest):
    width: int = Field(1024, ge=64, le=4096)
    height: int = Field(1024, ge=64, le=4096)


class TextToVideoRequest(_BaseGenerationRequest):
    width: int = Field(832, ge=64, le=4096)
    height: int = Field(480, ge=64, le=4096)
    duration_seconds: Optional[float] = Field(
        5.0, gt=0, le=60,
        description="Target clip duration; converted to the model's frame grid (rounded up)",
    )
    num_frames: Optional[int] = Field(
        None, ge=1,
        description="Exact frame count (overrides duration_seconds; should be quant*n+1 for the model)",
    )


class ImageToVideoRequest(TextToVideoRequest):
    """Image-to-video: provide the start image as exactly ONE of
    image_b64 / image_url / image_path / image_asset_id."""

    width: Optional[int] = Field(  # type: ignore[assignment]
        None, ge=64, le=4096,
        description="Output width; omitted = derived from the input image",
    )
    height: Optional[int] = Field(  # type: ignore[assignment]
        None, ge=64, le=4096,
        description="Output height; omitted = derived from the input image",
    )
    image_b64: Optional[str] = Field(
        None, description="Start image as base64 (raw or data: URI)")
    image_url: Optional[str] = Field(
        None, description="Start image URL (http/https); downloaded by the server")
    image_path: Optional[str] = Field(
        None, description="Start image path on the server's filesystem")
    image_asset_id: Optional[str] = Field(
        None, description="Asset id returned by POST /v1/assets")

    @model_validator(mode="after")
    def _exactly_one_image_source(self) -> "ImageToVideoRequest":
        sources = [self.image_b64, self.image_url, self.image_path, self.image_asset_id]
        if sum(s is not None for s in sources) != 1:
            raise ValueError(
                "Provide exactly one of image_b64, image_url, image_path, image_asset_id"
            )
        return self


# ----------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------

class ModelInfo(BaseModel):
    id: str
    task: Literal["t2i", "t2v", "i2v"]
    model_type: str
    description: str
    media_type: Literal["image", "video"]
    fps: int
    max_pixels: Optional[int] = None
    max_frames: Optional[int] = None
    is_default: bool = False


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class JobProgress(BaseModel):
    phase: Optional[str] = Field(None, description="loading_model | encoding_text | inference | decoding | ...")
    status: Optional[str] = Field(None, description="Human-readable status from WanGP")
    percent: int = Field(0, ge=0, le=100)
    current_step: Optional[int] = None
    total_steps: Optional[int] = None


class JobFile(BaseModel):
    index: int
    media_type: Literal["image", "video", "audio", "unknown"]
    filename: str
    path: str = Field(description="Absolute path on the server")
    url: str = Field(description="Download URL on this server")


class JobInfo(BaseModel):
    id: str
    task: Literal["text_to_image", "text_to_video", "image_to_video", "preload"]
    status: JobStatusLiteral
    model: str
    queue_position: Optional[int] = Field(
        None, description="0 = running next / now; only set while status=queued")
    progress: JobProgress = Field(default_factory=JobProgress)
    files: List[JobFile] = Field(default_factory=list)
    error: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict, description="Resolved WanGP task settings")
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class JobListResponse(BaseModel):
    jobs: List[JobInfo]


class PreloadRequest(BaseModel):
    models: List[str] = Field(
        default_factory=list,
        description="Model preset ids to preload, in order. Empty/omitted = the server's "
        "default t2i/t2v/i2v presets. The LAST model stays loaded in VRAM; "
        "earlier ones get their checkpoints downloaded and warmed.",
    )
    wait: bool = Field(
        False,
        description="If true, block until every preload job finishes; "
        "otherwise return queued jobs to poll via GET /v1/jobs/{id}.",
    )
    wait_timeout_seconds: float = Field(
        7200, gt=0, le=24 * 3600,
        description="Max time to block when wait=true (covers checkpoint downloads)",
    )


class PreloadResponse(BaseModel):
    jobs: List["JobInfo"] = Field(
        description="One warmup job per (deduplicated) model, queued FIFO")


class AssetInfo(BaseModel):
    asset_id: str
    filename: str
    path: str
    size_bytes: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    wan2gp_root: str
    runtime_loaded: bool = Field(description="Whether the WanGP runtime/session has been initialized")
    active_job_id: Optional[str] = None
    queued_jobs: int = 0
