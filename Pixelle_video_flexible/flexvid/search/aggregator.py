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

"""Fan-out search across all enabled stock providers, with dedup."""

import asyncio
import math
from typing import List, Optional

from loguru import logger

from flexvid.flex_config import FlexConfig
from flexvid.models import MediaCandidate
from flexvid.search.base import MediaSearchProvider
from flexvid.search.pexels import PexelsProvider
from flexvid.search.pixabay import PixabayProvider


class MediaSearchAggregator:
    """Queries every enabled provider concurrently and merges the results."""

    def __init__(self, providers: List[MediaSearchProvider]):
        self.providers = providers

    @classmethod
    def from_config(cls, config: FlexConfig) -> "MediaSearchAggregator":
        providers: List[MediaSearchProvider] = []
        if config.pexels_active:
            providers.append(PexelsProvider(config.pexels_api_key))
        if config.pixabay_active:
            providers.append(PixabayProvider(config.pixabay_api_key))
        return cls(providers)

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    @property
    def provider_names(self) -> List[str]:
        return [p.name for p in self.providers]

    async def search(
        self,
        query: str,
        media_type: str,
        n_total: int = 6,
        orientation: Optional[str] = None,
        min_width: Optional[int] = None,
    ) -> List[MediaCandidate]:
        """
        Search all providers concurrently; a failing provider is logged and
        skipped, never failing the whole search. Results are interleaved
        (provider round-robin) so no single provider dominates the gallery,
        deduplicated, and truncated to n_total.
        """
        if not self.providers:
            return []

        n_per_provider = max(1, math.ceil(n_total / len(self.providers)))
        results = await asyncio.gather(
            *(p.search(query, media_type, n=n_per_provider,
                       orientation=orientation, min_width=min_width)
              for p in self.providers),
            return_exceptions=True,
        )

        per_provider: List[List[MediaCandidate]] = []
        for provider, result in zip(self.providers, results):
            if isinstance(result, BaseException):
                logger.warning(f"Stock search on {provider.name} failed: {result}")
                continue
            per_provider.append(result)

        # Round-robin interleave + dedup (by id and by download URL)
        merged: List[MediaCandidate] = []
        seen = set()
        for rank in range(max((len(r) for r in per_provider), default=0)):
            for result in per_provider:
                if rank >= len(result):
                    continue
                candidate = result[rank]
                if candidate.id in seen or candidate.download_url in seen:
                    continue
                seen.add(candidate.id)
                seen.add(candidate.download_url)
                merged.append(candidate)

        logger.info(
            f"🔎 Stock search '{query}' [{media_type}]: {len(merged)} candidate(s) "
            f"from {', '.join(self.provider_names) or 'no providers'}"
        )
        return merged[:n_total]
