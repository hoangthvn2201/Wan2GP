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
Disk persistence for in-progress flexible projects.

Streamlit keeps the wizard state in `st.session_state`, which dies with the
browser session — and over a Colab/Cloudflare tunnel the websocket regularly
drops during long generation runs, resetting the whole wizard. The wizard
therefore checkpoints the project into `<task_dir>/flex_project.json` after
every step, and the Setup step offers to resume the latest unfinished one.

Asset paths are validated on load: files that no longer exist (e.g. a fresh
Colab runtime reusing an old output dir) are dropped so the wizard simply
regenerates them instead of failing on ghosts.
"""

import dataclasses
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from Pixelle_video.pixelle_video.models.storyboard import StoryboardConfig
from sbs.models import SceneProject

from flexvid.models import FlexScene, MediaCandidate

SAVE_FILENAME = "flex_project.json"

_SCENE_FIELDS = {f.name for f in dataclasses.fields(FlexScene)}
_CONFIG_FIELDS = {f.name for f in dataclasses.fields(StoryboardConfig)}
_CANDIDATE_FIELDS = {f.name for f in dataclasses.fields(MediaCandidate)}

# Per-scene asset paths whose files must exist to survive a load
_ASSET_ATTRS = ("audio_path", "image_path", "video_path", "composed_path", "segment_path")


def save_project(project: SceneProject, step: Optional[int] = None) -> Optional[str]:
    """Checkpoint the project into its task dir (atomic, best effort)."""
    try:
        data = {
            "version": 1,
            "pipeline": "flexible_video",
            "saved_at": datetime.now().isoformat(),
            "step": step,
            "title": project.title,
            "task_id": project.task_id,
            "task_dir": project.task_dir,
            "config": asdict(project.config),
            "params": project.params,
            "media_requirement": project.media_requirement,
            "media_mode": project.media_mode,
            "final_video_path": project.final_video_path,
            "total_duration": project.total_duration,
            "created_at": project.created_at.isoformat(),
            "scenes": [asdict(s) for s in project.scenes],
        }
        path = Path(project.task_dir) / SAVE_FILENAME
        tmp = path.with_suffix(".json.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        os.replace(tmp, path)
        return str(path)
    except Exception as e:
        # A checkpoint failure must never break the wizard
        logger.warning(f"Failed to checkpoint flex project: {e}")
        return None


def _parse_dt(value) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return datetime.now()


def _load_scene(data: dict) -> FlexScene:
    candidates = [
        MediaCandidate(**{k: v for k, v in c.items() if k in _CANDIDATE_FIELDS})
        for c in (data.get("candidates") or [])
    ]
    created_at = _parse_dt(data.get("created_at"))
    fields = {k: v for k, v in data.items()
              if k in _SCENE_FIELDS and k not in ("candidates", "created_at")}
    scene = FlexScene(**fields)
    scene.candidates = candidates
    scene.created_at = created_at

    # Drop asset paths whose files are gone (fresh runtime / cleaned output)
    for attr in _ASSET_ATTRS:
        value = getattr(scene, attr)
        if value and not os.path.exists(value):
            setattr(scene, attr, None)
    if scene.video_path is None and scene.media_type == "video":
        scene.media_type = "image" if scene.image_path else None
    if scene.image_path is None and scene.media_type == "image":
        scene.media_type = "video" if scene.video_path else None
    if scene.audio_path is None:
        scene.duration = 0.0
    if not scene.has_media:
        scene.attribution = None
    return scene


def load_project(path: str) -> Tuple[SceneProject, Optional[int]]:
    """Load a checkpoint; returns (project, saved wizard step)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = StoryboardConfig(**{k: v for k, v in (data.get("config") or {}).items()
                                 if k in _CONFIG_FIELDS})
    project = SceneProject(
        title=data["title"],
        task_id=data["task_id"],
        task_dir=data["task_dir"],
        config=config,
        params=data.get("params") or {},
        scenes=[_load_scene(s) for s in (data.get("scenes") or [])],
        media_requirement=data.get("media_requirement", "image"),
        media_mode=data.get("media_mode", "image"),
    )
    project.final_video_path = data.get("final_video_path")
    project.total_duration = float(data.get("total_duration") or 0.0)
    project.created_at = _parse_dt(data.get("created_at"))
    step = data.get("step")
    return project, (int(step) if step is not None else None)


def _output_root() -> Path:
    root = os.environ.get("PIXELLE_VIDEO_ROOT") or os.getcwd()
    return Path(root) / "output"


def find_latest_saved_project(output_root: Optional[str] = None) -> Optional[dict]:
    """
    Newest UNFINISHED checkpoint (no final video yet), as a summary dict:
    {path, title, saved_at, ready, total} — or None.
    """
    root = Path(output_root) if output_root else _output_root()
    try:
        checkpoints = sorted(root.glob(f"*/{SAVE_FILENAME}"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None
    for path in checkpoints:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("final_video_path"):
                continue
            scenes = data.get("scenes") or []
            return {
                "path": str(path),
                "title": data.get("title") or "?",
                "saved_at": str(data.get("saved_at") or "")[:16].replace("T", " "),
                "ready": sum(1 for s in scenes if s.get("segment_path")),
                "total": len(scenes),
            }
        except Exception as e:
            logger.debug(f"Skipping unreadable checkpoint {path}: {e}")
    return None
