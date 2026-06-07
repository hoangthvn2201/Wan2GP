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
24-hour on-disk cache for stock search responses.

Pixabay's API terms REQUIRE search requests to be cached for 24 hours; for
Pexels it's not required but saves the 200/hour / 20,000/month quota. The TTL
also matches Pixabay's URL lifetime (returned URLs expire after ~24h), so a
cache hit never serves longer-dead links than a live response would.

One JSON file per (provider, media_type, query, params) key under
``Pixelle_video_flexible/.search_cache/``. Set ``FLEXVID_SEARCH_CACHE=0`` to
disable (e.g. in tests).
"""

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from loguru import logger

from flexvid.models import MediaCandidate

_MODULE_ROOT = Path(__file__).resolve().parents[2]   # .../Pixelle_video_flexible
DEFAULT_CACHE_DIR = _MODULE_ROOT / ".search_cache"
DEFAULT_TTL_SECONDS = 24 * 3600


class SearchCache:
    """Tiny file-per-key JSON cache with TTL (best effort, never raises)."""

    def __init__(self, cache_dir: Optional[Path] = None,
                 ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.ttl_seconds = ttl_seconds

    @property
    def enabled(self) -> bool:
        return os.environ.get("FLEXVID_SEARCH_CACHE", "1") != "0"

    def _path(self, key: tuple) -> Path:
        digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: tuple) -> Optional[List[MediaCandidate]]:
        """Cached candidates for the key, or None on miss/expiry/any error."""
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            if time.time() - entry["saved_at"] > self.ttl_seconds:
                path.unlink(missing_ok=True)
                return None
            return [MediaCandidate(**c) for c in entry["candidates"]]
        except Exception as e:
            logger.debug(f"Search cache read failed for {path}: {e}")
            return None

    def set(self, key: tuple, candidates: List[MediaCandidate]):
        """Store candidates (empty results included — same 24h rule applies)."""
        if not self.enabled:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "saved_at": time.time(),
                "key": list(key),
                "candidates": [asdict(c) for c in candidates],
            }
            path = self._path(key)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
            self._prune()
        except Exception as e:
            logger.debug(f"Search cache write failed: {e}")

    def _prune(self, max_files: int = 500):
        """Opportunistically drop expired entries (and cap total file count)."""
        try:
            files = sorted(self.cache_dir.glob("*.json"),
                           key=lambda p: p.stat().st_mtime)
            now = time.time()
            for path in files:
                if now - path.stat().st_mtime > self.ttl_seconds:
                    path.unlink(missing_ok=True)
            files = [p for p in files if p.exists()]
            for path in files[:max(0, len(files) - max_files)]:
                path.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Search cache prune failed: {e}")


_cache: Optional[SearchCache] = None


def get_search_cache() -> SearchCache:
    global _cache
    if _cache is None:
        _cache = SearchCache()
    return _cache
