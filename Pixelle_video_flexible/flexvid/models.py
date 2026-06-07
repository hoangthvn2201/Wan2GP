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
Data models for the flexible (generate-or-search) workflow.

`FlexScene` extends the scene-by-scene `Scene` with the media-sourcing plan:
which source the scene uses ("generate" via WanGP / ComfyUI, or "search" on
stock providers), the LLM's search query / reasoning, the fetched candidates
and the picked one. All added fields have defaults, so every inherited engine
method (`render_segment`, `compose_final`, `_to_storyboard`, ...) keeps
working on a `FlexScene` unchanged.

`MediaCandidate` is the provider-normalized stock search result: lightweight
(URLs + metadata only, no bytes) so candidate lists can live in
`st.session_state` safely.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from sbs.models import Scene


@dataclass
class MediaCandidate:
    """One stock media search result, normalized across providers."""

    id: str                                  # "<provider>:<native id>"
    source: str                              # "pexels" | "pixabay"
    media_type: str                          # "image" | "video"
    thumbnail_url: str
    download_url: str
    width: int = 0
    height: int = 0
    duration: Optional[float] = None         # seconds, videos only
    description: str = ""                    # alt text / tags
    photographer: str = ""
    page_url: str = ""
    license: str = ""

    @property
    def orientation(self) -> str:
        """'portrait' | 'landscape' | 'square' from the candidate dimensions."""
        if not self.width or not self.height:
            return "unknown"
        if self.width > self.height:
            return "landscape"
        if self.height > self.width:
            return "portrait"
        return "square"

    def attribution(self) -> dict:
        """Licensing info persisted with the scene / task metadata."""
        return {
            "source": self.source,
            "photographer": self.photographer,
            "page_url": self.page_url,
            "license": self.license,
            "media_type": self.media_type,
        }

    def meta_line(self) -> str:
        """One-line metadata summary for LLM ranking prompts."""
        parts = [
            f"source={self.source}",
            f"type={self.media_type}",
            f"size={self.width}x{self.height} ({self.orientation})",
        ]
        if self.duration:
            parts.append(f"duration={self.duration:.1f}s")
        if self.description:
            parts.append(f'description="{self.description[:160]}"')
        return ", ".join(parts)


@dataclass
class FlexScene(Scene):
    """Scene with a media-sourcing plan (generate vs stock search)."""

    # --- LLM media plan -------------------------------------------------
    source: str = "generate"                 # "generate" | "search" | "import"
    plan_media_type: str = "image"           # intended kind when searching: "image" | "video"
    search_query: Optional[str] = None       # short English stock keywords (search only)
    plan_reason: str = ""                    # LLM's justification, shown in the UI

    # --- Search state ---------------------------------------------------
    candidates: List[MediaCandidate] = field(default_factory=list)
    picked_candidate_id: Optional[str] = None
    search_attempted: bool = False
    fell_back_to_generate: bool = False

    # --- Licensing (copied from the picked candidate) --------------------
    attribution: Optional[dict] = None

    @classmethod
    def from_scene(cls, scene: Scene) -> "FlexScene":
        """Upgrade a base Scene (e.g. from create_project) in place-compatible form."""
        if isinstance(scene, cls):
            return scene
        return cls(
            narration=scene.narration,
            prompt=scene.prompt,
            uid=scene.uid,
            audio_path=scene.audio_path,
            media_type=scene.media_type,
            image_path=scene.image_path,
            video_path=scene.video_path,
            composed_path=scene.composed_path,
            segment_path=scene.segment_path,
            duration=scene.duration,
            created_at=scene.created_at,
        )

    @property
    def picked_candidate(self) -> Optional[MediaCandidate]:
        for c in self.candidates:
            if c.id == self.picked_candidate_id:
                return c
        return None

    @property
    def is_search(self) -> bool:
        """The scene still sources its media from stock search."""
        return self.source == "search" and not self.fell_back_to_generate

    @property
    def is_import(self) -> bool:
        """The scene uses user-imported media."""
        return self.source == "import"

    # ---- invalidation helpers ----

    def invalidate_search(self):
        """Search query / source changed -> candidates and pick are stale."""
        self.candidates = []
        self.picked_candidate_id = None
        self.search_attempted = False
        self.fell_back_to_generate = False
        self.attribution = None
        self.invalidate_segment()

    def invalidate_media(self):
        """Prompt / sourcing changed -> media, attribution and segment are stale."""
        super().invalidate_media()
        self.attribution = None
