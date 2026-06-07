# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0.

"""
Flexible video pipeline: LLM-orchestrated generate-or-search media per scene,
on top of the scene-by-scene engine (sbs) and the Pixelle-Video core.
"""

from flexvid.engine import FlexibleVideoEngine
from flexvid.flex_config import FlexConfig, load_flex_config, save_flex_config
from flexvid.models import FlexScene, MediaCandidate
from flexvid.search import MediaSearchAggregator

__all__ = [
    "FlexibleVideoEngine",
    "FlexConfig",
    "FlexScene",
    "MediaCandidate",
    "MediaSearchAggregator",
    "load_flex_config",
    "save_flex_config",
]
