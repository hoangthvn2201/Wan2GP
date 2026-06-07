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
Pixabay provider (https://pixabay.com/api/docs/).

Free API; auth via a ``key=`` query parameter. The API has no orientation
filter for videos (and only a coarse one for images), so orientation is
enforced client-side from the result dimensions. License: "Pixabay Content
License" (free to use, no attribution required).
"""

from typing import List, Optional

from loguru import logger

from flexvid.models import MediaCandidate
from flexvid.search.base import MediaSearchProvider, matches_min_resolution

_PHOTO_URL = "https://pixabay.com/api/"
_VIDEO_URL = "https://pixabay.com/api/videos/"
_LICENSE = "Pixabay Content License"


def _matches_orientation(width: int, height: int, orientation: Optional[str]) -> bool:
    if not orientation or not width or not height:
        return True
    if orientation == "portrait":
        return height > width
    if orientation == "landscape":
        return width > height
    if orientation == "square":
        return abs(width - height) <= max(width, height) * 0.05
    return True


class PixabayProvider(MediaSearchProvider):
    name = "pixabay"

    async def _get(self, url: str, params: dict) -> dict:
        return await self._request_json(url, {"key": self.api_key, **params})

    async def search_images(
        self,
        query: str,
        n: int = 6,
        orientation: Optional[str] = None,
        min_width: Optional[int] = None,
    ) -> List[MediaCandidate]:
        params = {
            "q": query,
            "image_type": "photo",
            "safesearch": "true",
            # Over-fetch: orientation + resolution are filtered client-side
            "per_page": min(max(n * 3, 10), 200),
        }
        if orientation in ("vertical", "portrait"):
            params["orientation"] = "vertical"
        elif orientation == "landscape":
            params["orientation"] = "horizontal"
        data = await self._get(_PHOTO_URL, params)

        candidates = []
        for hit in data.get("hits", []):
            width = hit.get("imageWidth", 0)
            height = hit.get("imageHeight", 0)
            if not _matches_orientation(width, height, orientation):
                continue
            if not matches_min_resolution(width, height, min_width):
                continue
            download_url = hit.get("largeImageURL") or hit.get("webformatURL")
            if not download_url:
                continue
            candidates.append(MediaCandidate(
                id=f"pixabay:photo:{hit.get('id')}",
                source=self.name,
                media_type="image",
                thumbnail_url=hit.get("webformatURL") or hit.get("previewURL") or download_url,
                download_url=download_url,
                width=width,
                height=height,
                description=hit.get("tags") or "",
                photographer=hit.get("user") or "",
                page_url=hit.get("pageURL") or "",
                license=_LICENSE,
            ))
            if len(candidates) >= n:
                break
        logger.debug(f"Pixabay images '{query}': {len(candidates)} candidate(s)")
        return candidates

    async def search_videos(
        self,
        query: str,
        n: int = 6,
        orientation: Optional[str] = None,
        min_width: Optional[int] = None,
    ) -> List[MediaCandidate]:
        params = {
            "q": query,
            "safesearch": "true",
            "per_page": min(max(n * 3, 10), 200),
        }
        data = await self._get(_VIDEO_URL, params)

        candidates = []
        for hit in data.get("hits", []):
            best = self._pick_rendition(hit.get("videos") or {}, min_width)
            if best is None:
                continue
            width, height = best.get("width", 0), best.get("height", 0)
            if not _matches_orientation(width, height, orientation):
                continue
            candidates.append(MediaCandidate(
                id=f"pixabay:video:{hit.get('id')}",
                source=self.name,
                media_type="video",
                thumbnail_url=best.get("thumbnail") or "",
                download_url=best["url"],
                width=width,
                height=height,
                duration=float(hit.get("duration") or 0) or None,
                description=hit.get("tags") or "",
                photographer=hit.get("user") or "",
                page_url=hit.get("pageURL") or "",
                license=_LICENSE,
            ))
            if len(candidates) >= n:
                break
        logger.debug(f"Pixabay videos '{query}': {len(candidates)} candidate(s)")
        return candidates

    @staticmethod
    def _pick_rendition(videos: dict, min_width: Optional[int]) -> Optional[dict]:
        """
        Pixabay returns named renditions (large/medium/small/tiny). Pick the
        SMALLEST one meeting the resolution floor (normalization downscales
        anyway); fall back to the largest available.
        """
        renditions = [v for v in videos.values() if v and v.get("url")]
        if not renditions:
            return None
        ok = [v for v in renditions
              if matches_min_resolution(v.get("width") or 0, v.get("height") or 0, min_width)]
        if ok:
            return min(ok, key=lambda v: (v.get("width") or 0) * (v.get("height") or 0))
        return max(renditions, key=lambda v: (v.get("width") or 0) * (v.get("height") or 0))
