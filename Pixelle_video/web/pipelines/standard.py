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
Standard Pipeline UI

Implements the classic 3-column layout for the Standard Pipeline.

Optionally, the script can be PREPARED first (generated / split without
rendering anything) so the user can attach their own image or video to
individual scenes before the actual generation runs ("per-scene media
import"). Scenes without an upload are generated as usual.
"""

import os
import uuid
from typing import Any

import streamlit as st

from web.i18n import tr
from web.utils.async_helpers import run_async

from web.pipelines.base import PipelineUI, register_pipeline_ui

# Import components
from web.components.content_input import render_content_input, render_bgm_section, render_version_info
from web.components.style_config import render_style_config
from web.components.output_preview import render_output_preview

# Session-state keys for the per-scene media import flow
K_PREPARED = "std_import_prepared"   # {"text", "mode", "n_scenes", "split_mode", "narrations"}
K_FILES = "std_import_files"         # {scene_index: {"sig": (name, size), "path": str}}
K_TOKEN = "std_import_token"         # unique-ish token for temp file names


def _prepare_script(pixelle_video, content_params: dict) -> list:
    """Generate (or split) the narrations exactly like StandardPipeline would."""
    from pixelle_video.utils.content_generators import (
        generate_narrations_from_topic,
        split_narration_script,
    )

    if content_params.get("mode", "generate") == "generate":
        return run_async(generate_narrations_from_topic(
            pixelle_video.llm,
            topic=content_params["text"],
            n_scenes=content_params.get("n_scenes", 5),
            min_words=5,
            max_words=20,
        ))
    return run_async(split_narration_script(
        content_params["text"],
        split_mode=content_params.get("split_mode", "paragraph"),
    ))


def _prepared_is_stale(prepared: dict, content_params: dict) -> bool:
    """The input changed since the script was prepared."""
    if (prepared.get("text") or "").strip() != (content_params.get("text") or "").strip():
        return True
    if prepared.get("mode") != content_params.get("mode"):
        return True
    if prepared.get("mode") == "generate":
        return prepared.get("n_scenes") != content_params.get("n_scenes")
    return prepared.get("split_mode") != content_params.get("split_mode")


def _store_upload(index: int, uploaded) -> str:
    """
    Persist an uploaded file into the temp dir (only when it actually changed,
    so reruns don't rewrite large videos) and return its path.
    """
    from pixelle_video.utils.os_util import get_temp_path

    files = st.session_state.setdefault(K_FILES, {})
    sig = (uploaded.name, uploaded.size)
    entry = files.get(index)
    if entry and entry["sig"] == sig and os.path.exists(entry["path"]):
        return entry["path"]

    token = st.session_state.setdefault(K_TOKEN, uuid.uuid4().hex[:8])
    ext = os.path.splitext(uploaded.name)[1].lower()
    path = get_temp_path("scene_imports", f"{token}_scene_{index + 1:02d}{ext}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    files[index] = {"sig": sig, "path": path}
    return path


def _render_scene_import_section(pixelle_video, content_params: dict) -> dict:
    """
    Optional per-scene media import: prepare the script first, then attach an
    image/video to any scene. Returns {} (feature unused) or
    {"narrations": [...], "scene_media": {index: path}} to merge into the
    generation params.
    """
    from pixelle_video.utils.imported_media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

    if content_params.get("batch_mode"):
        return {}

    prepared = st.session_state.get(K_PREPARED)

    with st.expander(tr("std.import.expander", "📥 Per-scene media import (optional)"),
                     expanded=prepared is not None):
        st.caption(tr("std.import.hint",
                      "Prepare the script first, then attach your own image or video "
                      "to any scene. Scenes without an upload are generated as usual."))

        # ---- Prepare / re-prepare / clear ----
        if prepared is None:
            if st.button(tr("std.import.prepare_btn", "📝 Prepare scenes"),
                         use_container_width=True,
                         disabled=not (content_params.get("text") or "").strip()):
                try:
                    with st.spinner(tr("std.import.preparing", "Preparing the script...")):
                        narrations = _prepare_script(pixelle_video, content_params)
                    st.session_state[K_PREPARED] = {
                        "text": content_params.get("text"),
                        "mode": content_params.get("mode"),
                        "n_scenes": content_params.get("n_scenes"),
                        "split_mode": content_params.get("split_mode"),
                        "narrations": narrations,
                    }
                    st.session_state[K_FILES] = {}
                    st.rerun()
                except Exception as e:
                    st.error(tr("status.error", error=str(e)))
            return {}

        stale = _prepared_is_stale(prepared, content_params)
        if stale:
            st.warning(tr("std.import.stale_warning",
                          "The input changed since the script was prepared — "
                          "re-prepare it, or the imports below will be ignored."))

        c_re, c_clear = st.columns(2)
        with c_re:
            if st.button(tr("std.import.reprepare_btn", "🔄 Re-prepare"),
                         use_container_width=True):
                try:
                    with st.spinner(tr("std.import.preparing", "Preparing the script...")):
                        narrations = _prepare_script(pixelle_video, content_params)
                    st.session_state[K_PREPARED] = {
                        "text": content_params.get("text"),
                        "mode": content_params.get("mode"),
                        "n_scenes": content_params.get("n_scenes"),
                        "split_mode": content_params.get("split_mode"),
                        "narrations": narrations,
                    }
                    st.session_state[K_FILES] = {}
                    st.rerun()
                except Exception as e:
                    st.error(tr("status.error", error=str(e)))
        with c_clear:
            if st.button(tr("std.import.clear_btn", "🗑️ Clear"),
                         use_container_width=True):
                st.session_state.pop(K_PREPARED, None)
                st.session_state.pop(K_FILES, None)
                st.rerun()

        # ---- Per-scene uploaders ----
        upload_types = [ext.lstrip(".") for ext in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS]
        scene_media = {}
        for i, narration in enumerate(prepared["narrations"]):
            st.markdown(f"**{tr('std.import.scene_label', 'Scene {n}', n=i + 1)}**")
            st.caption(f"🗣️ {narration}")
            uploaded = st.file_uploader(
                tr("std.import.uploader_label", "Image or video for this scene"),
                type=upload_types,
                key=f"std_import_file_{i}",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                scene_media[i] = _store_upload(i, uploaded)
            else:
                st.session_state.get(K_FILES, {}).pop(i, None)

        if scene_media:
            st.caption(tr("std.import.summary",
                          "{count} scene(s) will use imported media",
                          count=len(scene_media)))

        if stale:
            return {}
        overrides = {"narrations": list(prepared["narrations"])}
        if scene_media:
            overrides["scene_media"] = scene_media
        return overrides


class StandardPipelineUI(PipelineUI):
    """
    UI for the Standard Video Generation Pipeline.
    Implements the classic 3-column layout.
    """
    name = "quick_create"
    icon = "⚡"

    @property
    def display_name(self):
        return tr("pipeline.quick_create.name")

    @property
    def description(self):
        return tr("pipeline.quick_create.description")

    def render(self, pixelle_video: Any):
        # Three-column layout
        left_col, middle_col, right_col = st.columns([1, 1, 1])

        # ====================================================================
        # Left Column: Content Input & BGM
        # ====================================================================
        with left_col:
            # Content input (mode, text, title, n_scenes)
            content_params = render_content_input()

            # BGM selection (bgm_path, bgm_volume)
            bgm_params = render_bgm_section()

            # Per-scene media import (optional, prepares the script first)
            import_params = _render_scene_import_section(pixelle_video, content_params)

            # Version info & GitHub link
            render_version_info()

        # ====================================================================
        # Middle Column: Style Configuration
        # ====================================================================
        with middle_col:
            # Style configuration (TTS, template, workflow, etc.)
            style_params = render_style_config(pixelle_video)

        # ====================================================================
        # Right Column: Output Preview
        # ====================================================================
        with right_col:
            # Combine all parameters
            video_params = {
                "pipeline": self.name,
                **content_params,
                **bgm_params,
                **style_params,
                **import_params
            }

            # Render output preview (generate button, progress, video preview)
            render_output_preview(pixelle_video, video_params)


# Register self
register_pipeline_ui(StandardPipelineUI)
