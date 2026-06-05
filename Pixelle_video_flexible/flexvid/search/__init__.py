# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0.

"""Stock media search (Pexels + Pixabay) for the flexible pipeline."""

from flexvid.search.aggregator import MediaSearchAggregator
from flexvid.search.base import MediaSearchProvider
from flexvid.search.pexels import PexelsProvider
from flexvid.search.pixabay import PixabayProvider

__all__ = [
    "MediaSearchAggregator",
    "MediaSearchProvider",
    "PexelsProvider",
    "PixabayProvider",
]
