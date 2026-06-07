"""Model presets: friendly ids that map to WanGP model types + tuned defaults.

A preset plays the same role as Pixelle-Video's `workflows/wan2gp/*.json`
descriptors: it bundles a WanGP `model_type` (any file name from
`defaults/*.json` in the Wan2GP repo) with the grid constraints needed to
build valid task settings:

    fps                  Output fps (16 for Wan 2.1/2.2 14B, 24 for LTX-2 / Wan 2.2 5B)
    frame_quant          Temporal grid: frame count snaps UP to quant*n+1
    resolution_multiple  Spatial grid for width/height snapping
    max_pixels           Approximate area cap (larger sizes scale down, keeping aspect)
    max_frames           Hard cap on frame count
    settings             Extra WanGP task settings merged into every job
                         (same keys as WanGP's "Export Settings" output)

Extra presets can be dropped as *.json files into the folder pointed to by
WAN2GP_SERVER_PRESETS_DIR — same fields as below plus "id" and "task".
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

TASKS = ("t2i", "t2v", "i2v")


@dataclass(frozen=True)
class ModelPreset:
    id: str
    task: str                      # "t2i" | "t2v" | "i2v"
    model_type: str                # WanGP model id (a defaults/*.json file name)
    description: str
    media_type: str                # "image" | "video"
    fps: int = 16
    frame_quant: int = 4
    resolution_multiple: int = 16
    max_pixels: Optional[int] = None
    max_frames: Optional[int] = None
    settings: Dict[str, Any] = field(default_factory=dict)


_BUILTIN: List[ModelPreset] = [
    # ------------------------------------------------------------------
    # Text to image
    # ------------------------------------------------------------------
    ModelPreset(
        id="z-image-turbo",
        task="t2i",
        model_type="z_image",
        description="Z-Image Turbo 6B — fast (8 steps), low VRAM, fits a free Colab T4",
        media_type="image",
        max_pixels=1_572_864,
        settings={"image_mode": 1, "batch_size": 1, "num_inference_steps": 8},
    ),
    ModelPreset(
        id="qwen-image",
        task="t2i",
        model_type="qwen_image_20B",
        description="Qwen Image 20B — high quality text-to-image, strong text rendering",
        media_type="image",
        max_pixels=1_572_864,
        settings={"image_mode": 1, "batch_size": 1},
    ),
    ModelPreset(
        id="flux-dev",
        task="t2i",
        model_type="flux",
        description="Flux 1 Dev 12B — high quality text-to-image",
        media_type="image",
        max_pixels=1_572_864,
        settings={"image_mode": 1, "batch_size": 1},
    ),
    # ------------------------------------------------------------------
    # Text to video
    # ------------------------------------------------------------------
    ModelPreset(
        id="wan21-t2v-1.3b",
        task="t2v",
        model_type="t2v_1.3B",
        description="Wan 2.1 Text2Video 1.3B — low VRAM, fits a free Colab T4, 16 fps",
        media_type="video",
        fps=16,
        frame_quant=4,
        max_pixels=399_360,
        max_frames=81,
    ),
    ModelPreset(
        id="wan21-fusionix",
        task="t2v",
        model_type="t2v_fusionix",
        description="Wan 2.1 FusioniX 14B text-to-video — fast (8 steps), 16 fps",
        media_type="video",
        fps=16,
        frame_quant=4,
        max_pixels=399_360,
        max_frames=129,
        settings={"num_inference_steps": 8, "guidance_scale": 1},
    ),
    ModelPreset(
        id="wan22-t2v",
        task="t2v",
        model_type="t2v_2_2",
        description="Wan 2.2 Text2Video 14B — high quality, 16 fps",
        media_type="video",
        fps=16,
        frame_quant=4,
        max_pixels=399_360,
        max_frames=129,
    ),
    ModelPreset(
        id="ltx2-distilled",
        task="t2v",
        model_type="ltx2_22B_distilled",
        description="LTX-2 2.3 Distilled 22B — text-to-video WITH audio, 8 steps, 24 fps",
        media_type="video",
        fps=24,
        frame_quant=8,
        resolution_multiple=32,
        max_pixels=921_600,
        max_frames=241,
        settings={"num_inference_steps": 8},
    ),
    # ------------------------------------------------------------------
    # Image to video
    # ------------------------------------------------------------------
    ModelPreset(
        id="wan21-fun-inp-1.3b",
        task="i2v",
        model_type="fun_inp_1.3B",
        description="Wan 2.1 Fun InP 1.3B image-to-video — low VRAM, fits a free Colab T4, 16 fps",
        media_type="video",
        fps=16,
        frame_quant=4,
        max_pixels=399_360,
        max_frames=81,
    ),
    ModelPreset(
        id="wan21-fusionix-i2v",
        task="i2v",
        model_type="i2v_fusionix",
        description="Wan 2.1 FusioniX 14B image-to-video 480p — fast (8 steps), 16 fps",
        media_type="video",
        fps=16,
        frame_quant=4,
        max_pixels=399_360,
        max_frames=129,
        settings={"num_inference_steps": 8, "guidance_scale": 1},
    ),
    ModelPreset(
        id="wan22-i2v",
        task="i2v",
        model_type="i2v_2_2",
        description="Wan 2.2 Image2Video 14B — high quality, 16 fps",
        media_type="video",
        fps=16,
        frame_quant=4,
        max_pixels=399_360,
        max_frames=129,
    ),
    ModelPreset(
        id="wan22-i2v-lightning",
        task="i2v",
        model_type="i2v_2_2_Enhanced_Lightning_v2",
        description="Wan 2.2 Image2Video Enhanced Lightning v2 14B — fast (4 steps), 16 fps",
        media_type="video",
        fps=16,
        frame_quant=4,
        max_pixels=399_360,
        max_frames=129,
    ),
    ModelPreset(
        id="wan22-ti2v-5b",
        task="i2v",
        model_type="ti2v_2_2",
        description="Wan 2.2 TextImage2Video 5B — 720p 24 fps, medium VRAM",
        media_type="video",
        fps=24,
        frame_quant=4,
        resolution_multiple=32,
        max_pixels=921_600,
        max_frames=121,
        settings={"image_prompt_type": "S"},
    ),
    ModelPreset(
        id="ltx2-distilled-i2v",
        task="i2v",
        model_type="ltx2_22B_distilled",
        description="LTX-2 2.3 Distilled 22B image-to-video WITH audio — 8 steps, 24 fps",
        media_type="video",
        fps=24,
        frame_quant=8,
        resolution_multiple=32,
        max_pixels=921_600,
        max_frames=241,
        settings={"num_inference_steps": 8, "image_prompt_type": "S"},
    ),
]


class PresetRegistry:
    """Built-in presets plus optional user presets loaded from a folder."""

    def __init__(self, presets_dir: Optional[Path] = None):
        self._presets: Dict[str, ModelPreset] = {p.id: p for p in _BUILTIN}
        if presets_dir and presets_dir.is_dir():
            for path in sorted(presets_dir.glob("*.json")):
                preset = self._load_file(path)
                self._presets[preset.id] = preset

    @staticmethod
    def _load_file(path: Path) -> ModelPreset:
        data = json.loads(path.read_text())
        if data.get("task") not in TASKS:
            raise ValueError(f"{path}: 'task' must be one of {TASKS}")
        return ModelPreset(
            id=data.get("id") or path.stem,
            task=data["task"],
            model_type=data["model_type"],
            description=data.get("description", f"Custom preset from {path.name}"),
            media_type=data.get("media_type", "image" if data["task"] == "t2i" else "video"),
            fps=int(data.get("fps", 16)),
            frame_quant=int(data.get("frame_quant", 4)),
            resolution_multiple=int(data.get("resolution_multiple", 16)),
            max_pixels=data.get("max_pixels"),
            max_frames=data.get("max_frames"),
            settings=dict(data.get("settings") or {}),
        )

    def list(self, task: Optional[str] = None) -> List[ModelPreset]:
        presets = list(self._presets.values())
        if task:
            presets = [p for p in presets if p.task == task]
        return presets

    def get(self, preset_id: str, task: Optional[str] = None) -> ModelPreset:
        preset = self._presets.get(preset_id)
        if preset is None:
            available = ", ".join(p.id for p in self.list(task))
            raise KeyError(f"Unknown model '{preset_id}'. Available: {available}")
        if task and preset.task != task:
            raise KeyError(f"Model '{preset_id}' is a {preset.task} model, not {task}")
        return preset


# ----------------------------------------------------------------------
# Settings construction (mirrors Pixelle-Video's proven Wan2GPClient logic)
# ----------------------------------------------------------------------

def fit_resolution(
    width: int,
    height: int,
    multiple: int = 16,
    max_pixels: Optional[int] = None,
) -> str:
    """Scale (if needed) and snap a resolution to the model grid."""
    w, h = float(width), float(height)
    if max_pixels and w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        w, h = w * scale, h * scale
    w_snapped = max(multiple, int(round(w / multiple)) * multiple)
    h_snapped = max(multiple, int(round(h / multiple)) * multiple)
    return f"{w_snapped}x{h_snapped}"


def fit_video_length(
    duration: float,
    fps: int,
    frame_quant: int = 4,
    max_frames: Optional[int] = None,
) -> int:
    """Convert a duration (seconds) to a frame count on the model's
    temporal grid (quant * n + 1), rounding up."""
    frames = max(1, int(round(duration * fps)))
    frames = ((max(frames - 1, 1) + frame_quant - 1) // frame_quant) * frame_quant + 1
    if max_frames:
        frames = min(frames, max_frames)
    return frames


def build_settings(
    preset: ModelPreset,
    prompt: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    num_frames: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    image_start: Optional[str] = None,
    extra_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a WanGP task settings dict from a preset + request parameters."""
    settings: Dict[str, Any] = dict(preset.settings)
    settings["model_type"] = preset.model_type
    settings["prompt"] = prompt
    settings.setdefault("seed", -1)

    if width and height:
        settings["resolution"] = fit_resolution(
            int(width),
            int(height),
            multiple=preset.resolution_multiple,
            max_pixels=preset.max_pixels,
        )

    if preset.media_type == "video":
        if num_frames is not None:
            frames = max(1, int(num_frames))
            if preset.max_frames:
                frames = min(frames, preset.max_frames)
            settings["video_length"] = frames
        elif duration_seconds is not None:
            settings["video_length"] = fit_video_length(
                float(duration_seconds),
                preset.fps,
                frame_quant=preset.frame_quant,
                max_frames=preset.max_frames,
            )

    if negative_prompt is not None:
        settings["negative_prompt"] = negative_prompt
    if steps is not None:
        settings["num_inference_steps"] = int(steps)
    if seed is not None:
        settings["seed"] = int(seed)
    if guidance_scale is not None:
        settings["guidance_scale"] = float(guidance_scale)
    if image_start is not None:
        settings["image_start"] = str(Path(image_start).resolve())

    # Raw passthrough wins over everything (advanced/expert use)
    if extra_settings:
        settings.update(extra_settings)

    return settings


def build_warmup_settings(preset: ModelPreset, *, image_start: Optional[str] = None) -> Dict[str, Any]:
    """Build the cheapest valid task for a preset.

    Used by the preload endpoint: running a minimal generation goes through
    WanGP's normal path, which downloads the checkpoint and loads the
    weights into VRAM — exactly what a real job would do, minus the cost.
    """
    settings = build_settings(
        preset,
        "warmup",
        width=256,
        height=256,
        num_frames=preset.frame_quant + 1 if preset.media_type == "video" else None,
        steps=2,
        seed=0,
        image_start=image_start,
    )
    return settings
