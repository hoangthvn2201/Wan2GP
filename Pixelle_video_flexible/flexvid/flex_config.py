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
Module-local configuration for the flexible (media-search) pipeline.

Lives in ``Pixelle_video_flexible/flex_config.yaml`` — NOT in the shared
``Pixelle_video/config.yaml``: the core ``PixelleVideoConfig`` is a Pydantic
model that silently drops unknown top-level keys, and the core must stay
unchanged (same rule the PDF module follows).

Environment variables ``PEXELS_API_KEY`` / ``PIXABAY_API_KEY`` override the
file, so keys never need to be committed.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

# Module root: .../Pixelle_video_flexible
_MODULE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _MODULE_ROOT / "flex_config.yaml"


@dataclass
class FlexConfig:
    """Media-search settings (providers + search behavior)."""

    pexels_enabled: bool = True
    pexels_api_key: str = ""
    pixabay_enabled: bool = True
    pixabay_api_key: str = ""

    candidates_per_scene: int = 6     # total candidates shown per scene
    min_resolution: int = 720         # min(width, height) of acceptable results
    allow_fallback: bool = True       # empty search -> fall back to generation
    search_only: bool = False         # stock-only: plan every scene as search and
                                      # never fall back to AI generation

    @property
    def pexels_active(self) -> bool:
        return self.pexels_enabled and bool(self.pexels_api_key.strip())

    @property
    def pixabay_active(self) -> bool:
        return self.pixabay_enabled and bool(self.pixabay_api_key.strip())

    @property
    def any_provider_active(self) -> bool:
        return self.pexels_active or self.pixabay_active


def load_flex_config(path: Optional[str] = None) -> FlexConfig:
    """Read flex_config.yaml (if present) and apply env-var overrides."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    data: dict = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to read {config_path}: {e} — using defaults")
    section = data.get("media_search") or {}

    pexels = section.get("pexels") or {}
    pixabay = section.get("pixabay") or {}

    config = FlexConfig(
        pexels_enabled=bool(pexels.get("enabled", True)),
        pexels_api_key=str(pexels.get("api_key") or ""),
        pixabay_enabled=bool(pixabay.get("enabled", True)),
        pixabay_api_key=str(pixabay.get("api_key") or ""),
        candidates_per_scene=int(section.get("candidates_per_scene", 6)),
        min_resolution=int(section.get("min_resolution", 720)),
        allow_fallback=bool(section.get("allow_fallback", True)),
        search_only=bool(section.get("search_only", False)),
    )

    # Env vars beat the file (useful for deployments / keeping keys out of git)
    env_pexels = os.environ.get("PEXELS_API_KEY")
    if env_pexels:
        config.pexels_api_key = env_pexels
    env_pixabay = os.environ.get("PIXABAY_API_KEY")
    if env_pixabay:
        config.pixabay_api_key = env_pixabay

    return config


def save_flex_config(config: FlexConfig, path: Optional[str] = None):
    """Write the config back to flex_config.yaml (used by the settings UI)."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = {
        "media_search": {
            "pexels": {"enabled": config.pexels_enabled, "api_key": config.pexels_api_key},
            "pixabay": {"enabled": config.pixabay_enabled, "api_key": config.pixabay_api_key},
            "candidates_per_scene": config.candidates_per_scene,
            "min_resolution": config.min_resolution,
            "allow_fallback": config.allow_fallback,
            "search_only": config.search_only,
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"💾 Saved media-search config: {config_path}")
