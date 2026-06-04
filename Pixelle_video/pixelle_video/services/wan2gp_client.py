# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Wan2GP Client - In-process media generation through the WanGP Python API

Instead of sending workflows to a remote ComfyUI / RunningHub instance,
this client loads the image / video generation models directly in the
current process using WanGP's official `shared.api` wrapper
(see docs/API.md in the Wan2GP repository).

Workflow descriptors live in `workflows/wan2gp/*.json` and map a Pixelle
workflow to a WanGP `model_type` plus default settings:

    {
        "source": "wan2gp",
        "model_type": "t2v_fusionix",
        "media_type": "video",
        "fps": 16,
        "frame_quant": 4,
        "resolution_multiple": 16,
        "max_pixels": 399360,
        "max_frames": 161,
        "settings": {"num_inference_steps": 8, "guidance_scale": 1}
    }

The WanGP session is created lazily on first use (model weights are only
loaded when a generation is actually requested) and is kept alive across
requests so the model stays in VRAM between frames.
"""

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Pixelle-Video lives inside the Wan2GP repository:
#   <wan2gp_root>/Pixelle_video/pixelle_video/services/wan2gp_client.py
_DEFAULT_WAN2GP_ROOT = Path(__file__).resolve().parents[3]


class Wan2GPClient:
    """
    Thin wrapper around `shared.api.WanGPSession`.

    - Lazy, thread-safe session initialization (one session per process)
    - Maps Pixelle media parameters (prompt/width/height/duration/...) to
      WanGP task settings
    - Runs the blocking WanGP job in a worker thread so async pipelines
      are not blocked
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Full application config dict (the `wan2gp` section is used)
        """
        wan2gp_config = (config or {}).get("wan2gp") or {}

        root = wan2gp_config.get("root")
        self.root = Path(root).resolve() if root else _DEFAULT_WAN2GP_ROOT
        self.cli_args: List[str] = list(wan2gp_config.get("cli_args") or [])
        output_dir = wan2gp_config.get("output_dir")
        self.output_dir = Path(output_dir).resolve() if output_dir else None

        self._session = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self):
        """Get or lazily create the WanGP session (thread-safe)."""
        with self._lock:
            if self._session is None:
                if not (self.root / "wgp.py").exists():
                    raise RuntimeError(
                        f"Wan2GP installation not found at '{self.root}' "
                        f"(wgp.py is missing). Set 'wan2gp.root' in config.yaml."
                    )
                if str(self.root) not in sys.path:
                    sys.path.insert(0, str(self.root))

                from shared.api import init  # WanGP official Python API

                logger.info(f"🚀 Initializing WanGP session (root={self.root}, cli_args={self.cli_args})")
                init_kwargs: Dict[str, Any] = {
                    "root": self.root,
                    "cli_args": self.cli_args,
                }
                if self.output_dir is not None:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    init_kwargs["output_dir"] = self.output_dir
                self._session = init(**init_kwargs)
                logger.info("✅ WanGP session ready")
            return self._session

    @property
    def is_available(self) -> bool:
        """Check whether a Wan2GP installation is reachable (no model load)."""
        return (self.root / "wgp.py").exists()

    # ------------------------------------------------------------------
    # Settings construction
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_resolution(
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

    @staticmethod
    def _fit_video_length(
        duration: float,
        fps: int,
        frame_quant: int = 4,
        max_frames: Optional[int] = None,
    ) -> int:
        """
        Convert a target duration (seconds) to a frame count compatible with
        the model's temporal grid (quant * n + 1 frames).

        Rounds up so the generated video is at least as long as the
        narration audio.
        """
        frames = max(1, int(round(duration * fps)))
        # ceil to quant * n + 1
        frames = ((max(frames - 1, 1) + frame_quant - 1) // frame_quant) * frame_quant + 1
        if max_frames:
            frames = min(frames, max_frames)
        return frames

    def build_settings(
        self,
        descriptor: Dict[str, Any],
        prompt: str,
        *,
        media_type: str = "image",
        width: Optional[int] = None,
        height: Optional[int] = None,
        duration: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        cfg: Optional[float] = None,
        image_start: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the WanGP task settings dict from a descriptor + parameters."""
        model_type = descriptor.get("model_type")
        if not model_type:
            raise ValueError("wan2gp workflow descriptor is missing 'model_type'")

        # Descriptor-provided default settings first, overrides afterwards
        settings: Dict[str, Any] = dict(descriptor.get("settings") or {})
        settings["model_type"] = model_type
        settings["prompt"] = prompt
        settings.setdefault("seed", -1)

        if width and height:
            settings["resolution"] = self._fit_resolution(
                int(width),
                int(height),
                multiple=int(descriptor.get("resolution_multiple", 16)),
                max_pixels=descriptor.get("max_pixels"),
            )

        if media_type == "video":
            fps = int(descriptor.get("fps", 16))
            if duration:
                settings["video_length"] = self._fit_video_length(
                    float(duration),
                    fps,
                    frame_quant=int(descriptor.get("frame_quant", 4)),
                    max_frames=descriptor.get("max_frames"),
                )

        # Optional overrides
        if negative_prompt is not None:
            settings["negative_prompt"] = negative_prompt
        if steps is not None:
            settings["num_inference_steps"] = int(steps)
        if seed is not None:
            settings["seed"] = int(seed)
        if cfg is not None:
            settings["guidance_scale"] = float(cfg)
        if image_start is not None:
            settings["image_start"] = str(Path(image_start).resolve())

        return settings

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_sync(self, settings: Dict[str, Any]) -> List[str]:
        """Submit a task to WanGP and wait for completion (blocking)."""
        session = self._get_session()

        logger.info(
            f"🎨 WanGP generation: model={settings.get('model_type')} "
            f"resolution={settings.get('resolution')} "
            f"video_length={settings.get('video_length', '-')} "
            f"steps={settings.get('num_inference_steps', 'default')}"
        )

        job = session.submit_task(settings)
        result = job.result()

        if not result.success:
            messages = "; ".join(e.message for e in result.errors) or "unknown error"
            raise RuntimeError(f"WanGP generation failed: {messages}")
        if not result.generated_files:
            raise RuntimeError("WanGP generation completed but returned no output files")

        return list(result.generated_files)

    async def generate(
        self,
        descriptor: Dict[str, Any],
        prompt: str,
        **kwargs,
    ) -> str:
        """
        Generate media with WanGP and return the local output file path.

        Args:
            descriptor: Parsed wan2gp workflow descriptor (workflows/wan2gp/*.json)
            prompt: Generation prompt
            **kwargs: See `build_settings` (media_type, width, height, duration,
                      negative_prompt, steps, seed, cfg, image_start)

        Returns:
            Absolute path to the generated media file
        """
        settings = self.build_settings(descriptor, prompt, **kwargs)
        # WanGP's job.result() blocks; run it in a worker thread
        files = await asyncio.to_thread(self._generate_sync, settings)
        output = files[-1]
        logger.info(f"✅ WanGP generated: {output}")
        return output


# ----------------------------------------------------------------------
# Module-level shared client (mirrors the shared ComfyKit pattern)
# ----------------------------------------------------------------------

_client: Optional[Wan2GPClient] = None
_client_lock = threading.Lock()


def get_wan2gp_client(config: Optional[dict] = None) -> Wan2GPClient:
    """
    Get the shared Wan2GP client.

    The underlying WanGP session keeps models loaded in VRAM, so a single
    client instance is reused for the whole process. Changing the `wan2gp`
    config section (root / cli_args) requires an app restart.
    """
    global _client
    with _client_lock:
        if _client is None:
            if config is None:
                from Pixelle_video.pixelle_video.config import config_manager
                config = config_manager.config.to_dict()
            _client = Wan2GPClient(config)
        return _client
