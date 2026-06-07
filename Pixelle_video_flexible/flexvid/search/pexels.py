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
Pexels provider (https://www.pexels.com/api/).

Free API; auth via an ``Authorization: <key>`` header. Photos and videos use
separate endpoints. License: "Pexels License" (free to use, attribution
appreciated).
"""

from typing import List, Optional

from loguru import logger

from flexvid.models import MediaCandidate
from flexvid.search.base import MediaSearchProvider, matches_min_resolution

_PHOTO_URL = "https://api.pexels.com/v1/search"
_VIDEO_URL = "https://api.pexels.com/videos/search"
_LICENSE = "Pexels License"


class PexelsProvider(MediaSearchProvider):
    name = "pexels"

    async def _get(self, url: str, params: dict) -> dict:
        return await self._request_json(url, params,
                                        headers={"Authorization": self.api_key})

    async def search_images(
        self,
        query: str,
        n: int = 6,
        orientation: Optional[str] = None,
        min_width: Optional[int] = None,
    ) -> List[MediaCandidate]:
        params = {"query": query, "per_page": min(max(n, 1), 80)}
        if orientation in ("portrait", "landscape", "square"):
            params["orientation"] = orientation
        data = await self._get(_PHOTO_URL, params)

        candidates = []
        for photo in data.get("photos", []):
            width, height = photo.get("width", 0), photo.get("height", 0)
            if not matches_min_resolution(width, height, min_width):
                continue
            src = photo.get("src") or {}
            download_url = src.get("large2x") or src.get("large") or src.get("original")
            if not download_url:
                continue
            candidates.append(MediaCandidate(
                id=f"pexels:photo:{photo.get('id')}",
                source=self.name,
                media_type="image",
                thumbnail_url=src.get("medium") or src.get("tiny") or download_url,
                download_url=download_url,
                width=width,
                height=height,
                description=photo.get("alt") or "",
                photographer=photo.get("photographer") or "",
                page_url=photo.get("url") or "",
                license=_LICENSE,
            ))
            if len(candidates) >= n:
                break
        logger.debug(f"Pexels images '{query}': {len(candidates)} candidate(s)")
        return candidates

    async def search_videos(
        self,
        query: str,
        n: int = 6,
        orientation: Optional[str] = None,
        min_width: Optional[int] = None,
    ) -> List[MediaCandidate]:
        params = {"query": query, "per_page": min(max(n, 1), 80)}
        if orientation in ("portrait", "landscape", "square"):
            params["orientation"] = orientation
        data = await self._get(_VIDEO_URL, params)

        candidates = []
        for video in data.get("videos", []):
            best = self._pick_video_file(video.get("video_files") or [], min_width)
            if best is None:
                continue
            user = video.get("user") or {}
            candidates.append(MediaCandidate(
                id=f"pexels:video:{video.get('id')}",
                source=self.name,
                media_type="video",
                thumbnail_url=video.get("image") or "",
                download_url=best["link"],
                width=best.get("width") or video.get("width", 0),
                height=best.get("height") or video.get("height", 0),
                duration=float(video.get("duration") or 0) or None,
                # Pexels videos carry no alt text; the page URL slug is the
                # only textual hint (e.g. ".../video/woman-running-on-beach-123/")
                description=_describe_from_url(video.get("url") or ""),
                photographer=user.get("name") or "",
                page_url=video.get("url") or "",
                license=_LICENSE,
            ))
            if len(candidates) >= n:
                break
        logger.debug(f"Pexels videos '{query}': {len(candidates)} candidate(s)")
        return candidates

    @staticmethod
    def _pick_video_file(files: list, min_width: Optional[int]) -> Optional[dict]:
        """
        Pick the mp4 rendition to download: the SMALLEST one that still meets
        the resolution floor (stock 4K originals are slow to download and get
        downscaled by normalization anyway). Falls back to the largest file.
        """
        mp4s = [f for f in files if (f.get("file_type") or "").endswith("mp4") and f.get("link")]
        if not mp4s:
            return None
        ok = [f for f in mp4s
              if matches_min_resolution(f.get("width") or 0, f.get("height") or 0, min_width)]
        if ok:
            return min(ok, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))
        return max(mp4s, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))


def _describe_from_url(page_url: str) -> str:
    """'https://www.pexels.com/video/woman-running-on-beach-857134/' -> 'woman running on beach'."""
    slug = page_url.rstrip("/").rsplit("/", 1)[-1]
    words = [w for w in slug.split("-") if not w.isdigit()]
    return " ".join(words)
