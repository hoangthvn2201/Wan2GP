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
Data models for the scene-by-scene workflow.

A `Scene` is the step-wise equivalent of a `StoryboardFrame`: it carries the
same generated-asset paths, but is identified by a stable `uid` (instead of a
positional index) so scenes can be edited, regenerated, added or removed
without filename collisions inside the task directory.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from Pixelle_video.pixelle_video.models.storyboard import StoryboardConfig


def new_scene_uid() -> str:
    """Short stable id used in generated asset filenames."""
    return uuid.uuid4().hex[:8]


@dataclass
class Scene:
    """Single scene in the step-by-step workflow."""
    narration: str
    prompt: Optional[str] = None               # image/video prompt (None for static templates)
    uid: str = field(default_factory=new_scene_uid)

    # Generated resource paths (None = not generated yet / invalidated by edits)
    audio_path: Optional[str] = None
    media_type: Optional[str] = None           # "image" or "video"
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    composed_path: Optional[str] = None        # HTML-rendered frame (with subtitles)
    segment_path: Optional[str] = None         # final per-scene video segment

    duration: float = 0.0                      # seconds (from TTS audio or generated video)
    created_at: datetime = field(default_factory=datetime.now)

    # ---- invalidation helpers (edits make downstream assets stale) ----

    def invalidate_audio(self):
        """Narration changed -> audio, and everything depending on it, is stale."""
        self.audio_path = None
        self.duration = 0.0
        self.invalidate_segment()

    def invalidate_media(self):
        """Prompt changed -> generated media and the segment are stale."""
        self.image_path = None
        self.video_path = None
        self.media_type = None
        self.invalidate_segment()

    def invalidate_video(self):
        """
        The animated clip is stale (e.g. narration audio changed, so the clip
        length no longer matches) but the start image is still valid.
        Used by the image→video (i2v) mode.
        """
        self.video_path = None
        if self.media_type == "video":
            self.media_type = "image" if self.image_path else None
        self.invalidate_segment()

    def invalidate_segment(self):
        """Audio or media changed -> rendered frame + segment are stale."""
        self.composed_path = None
        self.segment_path = None

    @property
    def has_media(self) -> bool:
        return self.image_path is not None or self.video_path is not None


@dataclass
class SceneProject:
    """State of one step-by-step video generation session."""
    title: str
    task_id: str
    task_dir: str
    config: StoryboardConfig                   # reused core config (tts/media/template params)
    params: Dict[str, Any]                     # raw UI params (kept for persistence/history)
    scenes: List[Scene] = field(default_factory=list)

    # 'static' | 'image' | 'video' — derived from the frame template
    media_requirement: str = "image"

    # How scene media is produced:
    #   'none'  – static template, no media
    #   'image' – one still image per scene (image template)
    #   't2v'   – text → video with the style-config video workflow
    #   'i2v'   – image → video: a still is generated first, then animated by
    #             an i2v-capable workflow (workflows/<source>/i2v_*.json)
    media_mode: str = "image"

    final_video_path: Optional[str] = None
    total_duration: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def needs_media(self) -> bool:
        return self.media_requirement in ("image", "video")

    @property
    def is_i2v(self) -> bool:
        return self.media_mode == "i2v"

    @property
    def is_video_workflow(self) -> bool:
        """Scene media ends up as a video clip (t2v or i2v)."""
        if self.media_mode in ("t2v", "i2v"):
            return True
        if self.media_mode in ("none", "image"):
            return False
        # Fallback for projects created before media_mode existed
        workflow_name = self.config.media_workflow or ""
        return "video_" in workflow_name.lower()

    def scene_media_ready(self, scene: "Scene") -> bool:
        """Is this scene's media complete for segment rendering?"""
        if not self.needs_media:
            return True
        if self.is_video_workflow:
            return scene.video_path is not None
        return scene.image_path is not None

    @property
    def all_segments_ready(self) -> bool:
        return bool(self.scenes) and all(s.segment_path for s in self.scenes)

    def scene_by_uid(self, uid: str) -> Optional[Scene]:
        for scene in self.scenes:
            if scene.uid == uid:
                return scene
        return None
