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

"""Stock media search provider abstraction."""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx
from loguru import logger

from flexvid.models import MediaCandidate
from flexvid.search.cache import get_search_cache


class MediaSearchProvider(ABC):
    """One stock media API (Pexels, Pixabay, ...) normalized to MediaCandidate."""

    name: str = "provider"

    def __init__(self, api_key: str, *, timeout: float = 20.0):
        self.api_key = api_key
        self.timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(connect=10.0, read=self.timeout,
                                write=self.timeout, pool=self.timeout)
        return httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def _request_json(self, url: str, params: dict,
                            headers: Optional[dict] = None,
                            max_retries: int = 1) -> dict:
        """
        GET → JSON with quota-aware 429 handling: on 429 wait Retry-After
        (capped — Pexels' quota is hourly, waiting that out is pointless) and
        retry once; a persistent 429 raises, which the aggregator logs and
        skips so the other provider still serves the scene.
        """
        async with self._client() as client:
            for attempt in range(max_retries + 1):
                response = await client.get(url, params=params, headers=headers)
                if response.status_code == 429 and attempt < max_retries:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = min(float(retry_after), 30.0) if retry_after else 5.0
                    except ValueError:
                        delay = 5.0
                    logger.warning(
                        f"{self.name}: rate limited (429), retrying in {delay:.0f}s "
                        f"(remaining quota: "
                        f"{response.headers.get('X-RateLimit-Remaining', 'unknown')})"
                    )
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()

    @abstractmethod
    async def search_images(
        self,
        query: str,
        n: int = 6,
        orientation: Optional[str] = None,   # "portrait" | "landscape" | "square"
        min_width: Optional[int] = None,
    ) -> List[MediaCandidate]:
        """Search stock photos; returns up to n normalized candidates."""

    @abstractmethod
    async def search_videos(
        self,
        query: str,
        n: int = 6,
        orientation: Optional[str] = None,
        min_width: Optional[int] = None,
    ) -> List[MediaCandidate]:
        """Search stock video clips; returns up to n normalized candidates."""

    async def search(
        self,
        query: str,
        media_type: str,
        n: int = 6,
        orientation: Optional[str] = None,
        min_width: Optional[int] = None,
    ) -> List[MediaCandidate]:
        """
        Dispatch on media_type ("image" | "video"), behind a 24h on-disk
        cache. Pixabay's terms REQUIRE caching search responses for 24 hours;
        for Pexels it just saves quota. Failures are never cached.
        """
        cache = get_search_cache()
        key = (self.name, media_type, (query or "").strip().lower(),
               n, orientation or "", min_width or 0)
        cached = cache.get(key)
        if cached is not None:
            logger.debug(f"{self.name}: search cache hit for '{query}' [{media_type}]")
            return cached

        if media_type == "video":
            results = await self.search_videos(query, n=n, orientation=orientation,
                                               min_width=min_width)
        else:
            results = await self.search_images(query, n=n, orientation=orientation,
                                               min_width=min_width)
        cache.set(key, results)
        return results


def matches_min_resolution(width: int, height: int, min_resolution: Optional[int]) -> bool:
    """Quality gate: the SHORT side must reach min_resolution (when known)."""
    if not min_resolution:
        return True
    if not width or not height:
        return True  # unknown dimensions -> don't discard, the ranker can decide
    return min(width, height) >= min_resolution
