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
Scene-by-Scene Wizard UI

A 5-step wizard on top of `sbs.SceneBySceneEngine`:

    ① Setup    – topic / fixed script + style (TTS, template, workflow, ...)
    ② Script   – review / edit / AI-rewrite / add / delete narrations
    ③ Prompts  – review / edit / regenerate per-scene media prompts
    ④ Scenes   – generate & preview audio + image/video + segment per scene
    ⑤ Final    – pick BGM, compose the final video, preview & download

All intermediate state lives in `st.session_state` (prefix `sbs_`), so the
user can freely move back and forth; edits invalidate only the affected
downstream assets of that scene.
"""

import os

import streamlit as st
from loguru import logger

from web.i18n import tr
from web.utils.async_helpers import run_async
from web.components.content_input import render_bgm_section
from web.components.style_config import render_style_config

from pixelle_video.config import config_manager
from pixelle_video.utils.imported_media import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    cleanup_raw,
)
from pixelle_video.utils.os_util import get_task_path

from sbs import Scene, SceneBySceneEngine

# Session-state keys
K_STEP = "sbs_step"
K_PROJECT = "sbs_project"
K_RESULT = "sbs_result"

STEPS = ["setup", "script", "prompts", "scenes", "final"]
STEP_ICONS = ["⚙️", "📝", "🎨", "🎬", "🏁"]


# ============================================================================
# Widget-state helpers
#
# Model state (the project / scenes) is the single source of truth; widgets
# are bound to it through _bind_text below, which detects whether the MODEL
# or the WIDGET moved since the last render — so a stale widget copy can
# never silently overwrite a model value changed elsewhere.
# ============================================================================

def _clear_project_widget_state():
    """Drop per-scene widget state (and binder companions) from a previous project."""
    for key in list(st.session_state.keys()):
        if key.startswith(("sbs_narr_", "sbs_prompt_", "sbs_scene_narr_",
                           "sbs_scene_prompt_", "sbs_title_w", "sbs_import_")):
            del st.session_state[key]
    st.session_state.pop(K_RESULT, None)


def _reset_wizard():
    """Start a brand-new project (keeps the setup form values)."""
    _clear_project_widget_state()
    st.session_state.pop(K_PROJECT, None)
    st.session_state[K_STEP] = 0


def _read_bytes(path):
    """Read a file as bytes for cache-proof previews (regenerated files keep the same path)."""
    try:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    except Exception as e:
        logger.warning(f"Failed to read preview file {path}: {e}")
    return None


def _goto(step: int):
    st.session_state[K_STEP] = step
    st.rerun()


# ============================================================================
# Stepper
# ============================================================================

def _render_stepper(step: int, project):
    """Clickable breadcrumb of the wizard steps."""
    cols = st.columns(len(STEPS))
    for i, (name, icon) in enumerate(zip(STEPS, STEP_ICONS)):
        label = f"{icon} {tr(f'sbs.stepper.{name}')}"
        skipped = (i == 2 and project is not None and not project.needs_media)
        with cols[i]:
            if i == step:
                st.button(label, key=f"sbs_step_btn_{i}", type="primary",
                          use_container_width=True, disabled=True)
            elif project is not None and i < step and not skipped:
                # Completed steps are clickable to go back
                if st.button(label, key=f"sbs_step_btn_{i}", use_container_width=True):
                    _goto(i)
            else:
                help_text = tr("sbs.stepper.skipped_static") if skipped else None
                st.button(label, key=f"sbs_step_btn_{i}", use_container_width=True,
                          disabled=True, help=help_text)
    st.markdown("")


# ============================================================================
# Step ① Setup
# ============================================================================

def _render_video_mode_section(engine: SceneBySceneEngine, style_params: dict) -> dict:
    """
    Video-generation mode selector — only for video templates.

    Modes:
      t2v: text → video with the style-config video workflow (default)
      i2v: image → video — generate a still first, then animate it. Only
           i2v-capable workflows (workflows/<source>/i2v_*.json) can take a
           start image, so the option is gated on at least one being present.
    """
    frame_template = style_params.get("frame_template")
    if engine.media_requirement(frame_template) != "video":
        return {}

    with st.container(border=True):
        st.markdown(f"**{tr('sbs.setup.video_mode_label')}**")

        i2v_workflows = engine.list_i2v_workflows()
        if not i2v_workflows:
            st.caption(tr("sbs.setup.i2v_unavailable"))
            return {"media_mode": "t2v"}

        mode = st.radio(
            "video mode",
            ["t2v", "i2v"],
            horizontal=True,
            format_func=lambda x: tr(f"sbs.setup.video_mode_{x}"),
            label_visibility="collapsed",
            help=tr("sbs.setup.video_mode_help"),
            key="sbs_setup_video_mode",
        )
        if mode != "i2v":
            return {"media_mode": "t2v"}

        st.caption(tr("sbs.setup.i2v_hint"))

        i2v_workflow = st.selectbox(
            tr("sbs.setup.i2v_workflow_label"),
            options=i2v_workflows,
            key="sbs_setup_i2v_workflow",
        )

        # Image workflow for the start frame (default = config's image workflow)
        image_workflows = engine.list_image_workflows()
        default_label = tr("sbs.setup.i2v_image_workflow_default")
        image_choice = st.selectbox(
            tr("sbs.setup.i2v_image_workflow_label"),
            options=[default_label] + image_workflows,
            key="sbs_setup_i2v_image_workflow",
        )

        return {
            "media_mode": "i2v",
            "i2v_workflow": i2v_workflow,
            "i2v_image_workflow": None if image_choice == default_label else image_choice,
        }


def _render_setup(engine: SceneBySceneEngine):
    pixelle_video = engine.core
    project = st.session_state.get(K_PROJECT)

    left_col, right_col = st.columns([1, 1])

    # ---- Content input ----
    with left_col:
        with st.container(border=True):
            st.markdown(f"**{tr('section.content_input')}**")

            mode = st.radio(
                "Processing Mode",
                ["generate", "fixed"],
                horizontal=True,
                format_func=lambda x: tr(f"mode.{x}"),
                label_visibility="collapsed",
                key="sbs_setup_mode",
            )

            text_placeholder = (
                tr("input.topic_placeholder") if mode == "generate"
                else tr("input.content_placeholder")
            )
            text = st.text_area(
                tr("input.text"),
                placeholder=text_placeholder,
                height=140 if mode == "generate" else 220,
                key="sbs_setup_text",
            )

            if mode == "fixed":
                split_mode_options = {
                    "paragraph": tr("split.mode_paragraph"),
                    "line": tr("split.mode_line"),
                    "sentence": tr("split.mode_sentence"),
                }
                split_mode = st.selectbox(
                    tr("split.mode_label"),
                    options=list(split_mode_options.keys()),
                    format_func=lambda x: split_mode_options[x],
                    index=0,
                    help=tr("split.mode_help"),
                    key="sbs_setup_split_mode",
                )
            else:
                split_mode = "paragraph"

            title = st.text_input(
                tr("input.title"),
                placeholder=tr("input.title_placeholder"),
                help=tr("input.title_help"),
                key="sbs_setup_title",
            )

            if mode == "generate":
                n_scenes = st.slider(
                    tr("video.frames"),
                    min_value=3, max_value=30, value=5,
                    help=tr("video.frames_help"),
                    key="sbs_setup_n_scenes",
                )
                st.caption(tr("video.frames_label", n=n_scenes))
            else:
                n_scenes = 5
                st.info(tr("video.frames_fixed_mode_hint"))

    # ---- Style configuration (reused component: TTS / template / workflow) ----
    with right_col:
        style_params = render_style_config(pixelle_video)

        # Video templates can additionally run in image→video (i2v) mode:
        # generate a still first, then animate it with an i2v-capable model.
        video_mode_params = _render_video_mode_section(engine, style_params)

    st.caption(tr("sbs.setup.params_frozen_hint"))

    # ---- Actions ----
    c1, c2 = st.columns([2, 1]) if project is not None else (st.container(), None)
    with c1:
        if st.button(tr("sbs.setup.generate_btn"), type="primary", use_container_width=True):
            if not config_manager.validate():
                st.error(tr("settings.not_configured"))
                st.stop()
            if not text or not text.strip():
                st.error(tr("error.input_required"))
                st.stop()

            try:
                with st.spinner(tr("sbs.setup.generating")):
                    final_title, narrations = run_async(engine.generate_script(
                        text=text,
                        mode=mode,
                        n_scenes=n_scenes,
                        split_mode=split_mode,
                        title=title.strip() or None,
                    ))
                params = {
                    "text": text,
                    "mode": mode,
                    "n_scenes": n_scenes,
                    "split_mode": split_mode,
                    "title": title.strip() or None,
                    **style_params,
                    **video_mode_params,
                }
                _clear_project_widget_state()
                st.session_state[K_PROJECT] = engine.create_project(
                    final_title, narrations, [None] * len(narrations), params
                )
                _goto(1)
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))

    if project is not None and c2 is not None:
        with c2:
            if st.button(tr("sbs.setup.resume_btn"), use_container_width=True):
                _goto(1)
        st.warning(tr("sbs.setup.new_project_warning"))


# ============================================================================
# Step ② Script
# ============================================================================

def _bind_text(key: str, model_value: str, *, height: int = 80, label: str = "text",
               input_widget: bool = False, label_visibility: str = "collapsed"):
    """
    Text widget two-way bound to a model value, immune to stale widget state.

    A plain `if widget != model: model = widget` treats ANY mismatch as a user
    edit — but a mismatch can equally mean the MODEL was changed elsewhere
    (an edit on another step, AI rewrite, regenerated prompt) while this
    widget kept a stale copy; writing that stale copy back silently REVERTS
    the change (lost-script bug). A companion key remembers the model value
    this widget was last synced to, so the two directions can be told apart:
    model moved -> push the model into the widget (never apply the stale
    copy); widget moved -> return the edit for the caller to apply.
    """
    sync_key = key + "__synced"
    model_value = model_value or ""
    last_synced = st.session_state.get(sync_key)
    if key not in st.session_state:
        # First render (or Streamlit dropped the unrendered widget's state)
        st.session_state[key] = model_value
    elif last_synced is None or _texts_differ(last_synced, model_value):
        # The model changed since this widget last rendered: its copy is stale
        st.session_state[key] = model_value
    st.session_state[sync_key] = model_value
    if input_widget:
        return st.text_input(label, key=key, label_visibility=label_visibility)
    return st.text_area(label, key=key, height=height, label_visibility=label_visibility)


def _texts_differ(a: str, b: str) -> bool:
    """
    True only on a REAL content change.

    The browser can return a text_area value that differs from the model only
    in line endings / leading-trailing whitespace (sent along with ANY button
    click). Treating that as an edit used to invalidate generated assets —
    e.g. the i2v start image got wiped when the audio button was clicked.
    """
    norm = lambda s: (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return norm(a) != norm(b)


def _apply_narration_change(project, scene, value: str):
    """
    Narration changed -> audio is stale; video clips are length-synced to the
    audio, so they go stale too (i2v keeps its still-valid start image).
    Imported media is NOT length-synced (the segment step pads/trims it),
    so it survives narration edits.
    """
    scene.narration = value
    scene.invalidate_audio()
    if scene.media_origin == "imported":
        return
    if project.is_i2v:
        scene.invalidate_video()
    elif project.is_video_workflow:
        scene.invalidate_media()


def _render_script(engine: SceneBySceneEngine):
    project = st.session_state.get(K_PROJECT)
    if project is None:
        _goto(0)

    st.markdown(f"#### {tr('sbs.script.heading')}")
    st.caption(tr("sbs.script.hint"))

    # ---- Title ----
    new_title = _bind_text("sbs_title_w", project.title,
                           label=tr("sbs.script.title_label"),
                           input_widget=True, label_visibility="visible")
    if new_title.strip() and _texts_differ(new_title, project.title):
        project.title = new_title.strip()

    # ---- Scenes ----
    for i, scene in enumerate(list(project.scenes)):
        with st.container(border=True):
            st.markdown(f"**{tr('sbs.script.scene_label', n=i + 1)}**")

            key = f"sbs_narr_{scene.uid}"
            value = _bind_text(key, scene.narration, label="narration")
            if _texts_differ(value, scene.narration):
                _apply_narration_change(project, scene, value)

            c1, c2, _sp = st.columns([1, 1, 2])
            with c1:
                if st.button(tr("sbs.script.rewrite"), key=f"sbs_rw_{scene.uid}",
                             use_container_width=True):
                    try:
                        with st.spinner(tr("sbs.script.rewriting")):
                            rewritten = run_async(engine.rewrite_narration(
                                scene.narration,
                                topic=project.params.get("text"),
                            ))
                        # The binders push the new model value into every
                        # widget (this one included) on the next run.
                        _apply_narration_change(project, scene, rewritten)
                        st.rerun()
                    except Exception as e:
                        logger.exception(e)
                        st.error(tr("sbs.common.error", error=str(e)))
            with c2:
                if st.button(tr("sbs.script.delete"), key=f"sbs_del_{scene.uid}",
                             use_container_width=True,
                             disabled=len(project.scenes) <= 1):
                    project.scenes.remove(scene)
                    st.rerun()

    if st.button(tr("sbs.script.add"), use_container_width=True):
        project.scenes.append(Scene(narration=""))
        st.rerun()

    st.markdown("---")

    # ---- Navigation ----
    c_back, c_regen, c_next = st.columns([1, 1, 2])
    with c_back:
        if st.button(tr("sbs.nav.back"), key="sbs_script_back", use_container_width=True):
            _goto(0)
    with c_regen:
        if st.button(tr("sbs.script.regen_all"), key="sbs_script_regen", use_container_width=True):
            try:
                with st.spinner(tr("sbs.setup.generating")):
                    p = project.params
                    final_title, narrations = run_async(engine.generate_script(
                        text=p.get("text", ""),
                        mode=p.get("mode", "generate"),
                        n_scenes=p.get("n_scenes", 5),
                        split_mode=p.get("split_mode", "paragraph"),
                        title=p.get("title"),
                    ))
                _clear_project_widget_state()
                project.title = final_title
                project.scenes = [Scene(narration=n) for n in narrations]
                st.rerun()
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))
    with c_next:
        if st.button(tr("sbs.script.continue"), key="sbs_script_next",
                     type="primary", use_container_width=True):
            if any(not s.narration.strip() for s in project.scenes):
                st.error(tr("sbs.script.empty_warning"))
            else:
                _goto(2 if project.needs_media else 3)


# ============================================================================
# Step ③ Prompts
# ============================================================================

def _effective_prefix(project) -> str:
    """The style prefix to use for prompt generation ('' when toggled off)."""
    if not project.params.get("use_prompt_prefix", True):
        return ""
    return project.params.get("prompt_prefix") or ""


def _strip_prompt_prefix(prompt: str, prefix: str) -> str:
    """Inverse of build_image_prompt: remove a leading 'prefix, ' / bare prefix."""
    prompt = (prompt or "").strip()
    prefix = (prefix or "").strip()
    if not prefix:
        return prompt
    if prompt == prefix:
        return ""
    if prompt.startswith(prefix + ", "):
        return prompt[len(prefix) + 2:]
    return prompt


def _render_prefix_toggle(project):
    """
    Toggle: apply the style prefix to the media prompts or not.

    Flipping it rewrites the existing prompts in place (no LLM call): the
    prefix is prepended / stripped deterministically, and the affected scenes'
    media is invalidated since the effective prompt changed.
    """
    prefix_text = (project.params.get("prompt_prefix") or "").strip()
    if not prefix_text:
        return  # nothing to toggle

    from pixelle_video.utils.prompt_helper import build_image_prompt

    enabled = st.toggle(
        tr("sbs.prompts.use_prefix"),
        value=project.params.get("use_prompt_prefix", True),
        key="sbs_use_prefix",
        help=tr("sbs.prompts.use_prefix_help"),
    )
    st.caption(tr("sbs.prompts.prefix_preview", prefix=prefix_text))

    previous = project.params.get("use_prompt_prefix", True)
    if enabled == previous:
        return

    project.params["use_prompt_prefix"] = enabled
    for scene in project.scenes:
        if not scene.prompt:
            continue
        if enabled:
            new_prompt = build_image_prompt(_strip_prompt_prefix(scene.prompt, prefix_text), prefix_text)
        else:
            new_prompt = _strip_prompt_prefix(scene.prompt, prefix_text)
        if new_prompt != scene.prompt:
            scene.prompt = new_prompt
            scene.invalidate_media()
    st.rerun()   # the binders push the rewritten prompts into the widgets


def _render_prompts(engine: SceneBySceneEngine):
    project = st.session_state.get(K_PROJECT)
    if project is None:
        _goto(0)
    if not project.needs_media:
        _goto(3)

    st.markdown(f"#### {tr('sbs.prompts.heading')}")
    st.caption(tr("sbs.prompts.hint"))

    _render_prefix_toggle(project)
    prompt_prefix = _effective_prefix(project)

    # ---- Auto-generate missing prompts on entry ----
    missing = [s for s in project.scenes if not s.prompt]
    if missing:
        try:
            with st.spinner(tr("sbs.prompts.generating", count=len(missing))):
                prompts = run_async(engine.generate_prompts(
                    [s.narration for s in missing],
                    prompt_prefix=prompt_prefix,
                ))
            for scene, prompt in zip(missing, prompts):
                scene.prompt = prompt
            # The widget binders below pick the fresh prompts up from the scenes
        except Exception as e:
            logger.exception(e)
            st.error(tr("sbs.common.error", error=str(e)))
            if st.button(tr("sbs.prompts.retry"), type="primary"):
                st.rerun()
            return

    # ---- Per-scene prompt editing ----
    for i, scene in enumerate(project.scenes):
        with st.container(border=True):
            st.markdown(f"**{tr('sbs.script.scene_label', n=i + 1)}**")
            st.caption(f"🗣️ {scene.narration}")

            key = f"sbs_prompt_{scene.uid}"
            value = _bind_text(key, scene.prompt or "", height=100, label="prompt")
            if _texts_differ(value, scene.prompt):
                scene.prompt = value
                scene.invalidate_media()

            if st.button(tr("sbs.prompts.regen_one"), key=f"sbs_rp_{scene.uid}"):
                try:
                    with st.spinner(tr("sbs.prompts.regenerating")):
                        new_prompt = run_async(engine.generate_prompt_for(
                            scene.narration, prompt_prefix=prompt_prefix
                        ))
                    scene.prompt = new_prompt
                    scene.invalidate_media()
                    st.rerun()   # binders push the new prompt into the widgets
                except Exception as e:
                    logger.exception(e)
                    st.error(tr("sbs.common.error", error=str(e)))

    st.markdown("---")

    # ---- Navigation ----
    c_back, c_regen, c_next = st.columns([1, 1, 2])
    with c_back:
        if st.button(tr("sbs.nav.back"), key="sbs_prompts_back", use_container_width=True):
            _goto(1)
    with c_regen:
        if st.button(tr("sbs.prompts.regen_all"), key="sbs_prompts_regen",
                     use_container_width=True):
            try:
                with st.spinner(tr("sbs.prompts.generating", count=len(project.scenes))):
                    prompts = run_async(engine.generate_prompts(
                        [s.narration for s in project.scenes],
                        prompt_prefix=prompt_prefix,
                    ))
                for scene, prompt in zip(project.scenes, prompts):
                    scene.prompt = prompt
                    scene.invalidate_media()
                st.rerun()   # binders push the new prompts into the widgets
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))
    with c_next:
        if st.button(tr("sbs.prompts.continue"), key="sbs_prompts_next",
                     type="primary", use_container_width=True):
            if any(not (s.prompt or "").strip() for s in project.scenes):
                st.error(tr("sbs.prompts.empty_warning"))
            else:
                _goto(3)


# ============================================================================
# Step ④ Scenes
# ============================================================================

def _scene_status_icons(project, scene) -> str:
    parts = [f"🎤{'✅' if scene.audio_path else '⬜'}"]
    if scene.media_origin == "imported" and scene.has_media:
        parts.append("📥✅")
    elif project.is_i2v:
        # image → video: the still and the animated clip are separate steps
        parts.append(f"🖼️{'✅' if scene.image_path else '⬜'}")
        parts.append(f"🎬{'✅' if scene.video_path else '⬜'}")
    elif project.needs_media:
        media_icon = "🎬" if project.is_video_workflow else "🖼️"
        parts.append(f"{media_icon}{'✅' if scene.has_media else '⬜'}")
    parts.append(f"🎞️{'✅' if scene.segment_path else '⬜'}")
    return " ".join(parts)


def _save_uploaded_media(project, scene, uploaded) -> str:
    """Write an upload next to the scene assets as a raw temp file."""
    ext = os.path.splitext(uploaded.name)[1].lower() or ".bin"
    raw_path = get_task_path(project.task_id, "frames", f"{scene.uid}_import_raw{ext}")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return raw_path


def _render_import_control(engine, project, scene, index: int, key_prefix: str = "sbs"):
    """
    Upload control: use your own image/video for this scene instead of
    generating it (the engine normalizes the file into the canonical
    scene asset; clips are padded/trimmed to the narration at segment time).
    """
    imported = scene.media_origin == "imported" and scene.has_media
    with st.expander(tr("sbs.scenes.import_label", "📥 Use my own image/video"),
                     expanded=False):
        st.caption(tr("sbs.scenes.import_hint",
                      "The file is cropped/re-encoded to the project size; video "
                      "clips are padded or trimmed to the narration length."))
        uploaded = st.file_uploader(
            "import media",
            type=[e.lstrip(".") for e in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS],
            key=f"{key_prefix}_import_file_{scene.uid}",
            label_visibility="collapsed",
        )
        if st.button(tr("sbs.scenes.import_apply", "📥 Use this file"),
                     key=f"{key_prefix}_import_btn_{scene.uid}",
                     use_container_width=True, disabled=uploaded is None):
            try:
                with st.spinner(tr("sbs.scenes.import_applying", "Importing media...")):
                    raw_path = _save_uploaded_media(project, scene, uploaded)
                    try:
                        run_async(engine.import_media(
                            project, scene, index,
                            raw_path, original_name=uploaded.name,
                        ))
                    finally:
                        cleanup_raw(raw_path)
                st.rerun()
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))
        if imported:
            st.caption(tr("sbs.scenes.import_active", "✅ Currently using imported media"))


def _render_scene_card(engine: SceneBySceneEngine, project, scene, index: int):
    with st.container(border=True):
        head_l, head_r = st.columns([3, 1])
        with head_l:
            st.markdown(f"**{tr('sbs.script.scene_label', n=index + 1)}**  "
                        f"{_scene_status_icons(project, scene)}")
        with head_r:
            if scene.duration:
                st.caption(tr("sbs.scenes.duration", seconds=f"{scene.duration:.1f}"))

        text_col, preview_col = st.columns([1, 1])

        # ---- Editable inputs ----
        with text_col:
            st.caption(tr("sbs.scenes.narration_label"))
            nkey = f"sbs_scene_narr_{scene.uid}"
            nval = _bind_text(nkey, scene.narration, label="narration")
            if _texts_differ(nval, scene.narration):
                _apply_narration_change(project, scene, nval)

            if project.needs_media:
                st.caption(tr("sbs.scenes.prompt_label"))
                pkey = f"sbs_scene_prompt_{scene.uid}"
                pval = _bind_text(pkey, scene.prompt or "", height=100, label="prompt")
                if _texts_differ(pval, scene.prompt):
                    scene.prompt = pval
                    scene.invalidate_media()

            # ---- Action buttons ----
            def _run_step(spinner_key: str, coro_factory):
                try:
                    with st.spinner(tr(spinner_key, current=index + 1)):
                        run_async(coro_factory())
                    st.rerun()
                except Exception as e:
                    logger.exception(e)
                    st.error(tr("sbs.common.error", error=str(e)))

            cols = st.columns(4 if project.is_i2v else 3)

            with cols[0]:
                audio_label = tr("sbs.scenes.audio_regen") if scene.audio_path else tr("sbs.scenes.audio_btn")
                if st.button(audio_label, key=f"sbs_a_{scene.uid}", use_container_width=True,
                             disabled=not scene.narration.strip()):
                    _run_step("sbs.scenes.stage_audio",
                              lambda: engine.generate_audio(project, scene, index))

            if project.is_i2v:
                # i2v: strict order — audio → start image → animate.
                # (The audio comes first so the whole chain flows one way and the
                # clip length is always synced to the narration.)
                with cols[1]:
                    img_label = (tr("sbs.scenes.image_start_regen") if scene.image_path
                                 else tr("sbs.scenes.image_start_btn"))
                    img_needs_audio = not scene.audio_path
                    img_disabled = img_needs_audio or not (scene.prompt or "").strip()
                    if st.button(img_label, key=f"sbs_m_{scene.uid}", use_container_width=True,
                                 disabled=img_disabled,
                                 help=tr("sbs.scenes.need_audio_first") if img_needs_audio else None):
                        _run_step("sbs.scenes.stage_image_start",
                                  lambda: engine.generate_start_image(project, scene, index))

                with cols[2]:
                    anim_label = (tr("sbs.scenes.animate_regen") if scene.video_path
                                  else tr("sbs.scenes.animate_btn"))
                    missing = []
                    if not scene.image_path:
                        missing.append(tr("sbs.scenes.need_image_first"))
                    if not scene.audio_path:
                        missing.append(tr("sbs.scenes.need_audio_first"))
                    if st.button(anim_label, key=f"sbs_v_{scene.uid}", use_container_width=True,
                                 disabled=bool(missing),
                                 help=" / ".join(missing) if missing else None):
                        _run_step("sbs.scenes.stage_animate",
                                  lambda: engine.animate_image(project, scene, index))

            elif project.needs_media:
                with cols[1]:
                    media_label = (
                        tr("sbs.scenes.media_regen") if scene.has_media
                        else (tr("sbs.scenes.media_btn_video") if project.is_video_workflow
                              else tr("sbs.scenes.media_btn_image"))
                    )
                    # Video workflows need the audio duration first
                    needs_audio_first = project.is_video_workflow and not scene.audio_path
                    disabled = needs_audio_first or not (scene.prompt or "").strip()
                    help_text = tr("sbs.scenes.need_audio_first") if needs_audio_first else None
                    if st.button(media_label, key=f"sbs_m_{scene.uid}", use_container_width=True,
                                 disabled=disabled, help=help_text):
                        _run_step("sbs.scenes.stage_media",
                                  lambda: engine.generate_media(project, scene, index))

            with cols[-1]:
                seg_label = tr("sbs.scenes.segment_regen") if scene.segment_path else tr("sbs.scenes.segment_btn")
                seg_ready = scene.audio_path and project.scene_media_ready(scene)
                if st.button(seg_label, key=f"sbs_s_{scene.uid}", use_container_width=True,
                             disabled=not seg_ready,
                             help=None if seg_ready else tr("sbs.scenes.segment_requirements")):
                    _run_step("sbs.scenes.stage_segment",
                              lambda: engine.render_segment(project, scene, index))

            # ---- Import your own media (alternative to generation) ----
            if project.needs_media:
                _render_import_control(engine, project, scene, index)

        # ---- Previews (bytes-based so regenerated files refresh properly) ----
        with preview_col:
            audio_bytes = _read_bytes(scene.audio_path)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

            if project.is_i2v:
                # Show both the start frame and the animated clip
                video_bytes = _read_bytes(scene.video_path)
                if video_bytes:
                    st.video(video_bytes)
                image_bytes = _read_bytes(scene.image_path)
                if image_bytes:
                    if video_bytes:
                        with st.expander(tr("sbs.scenes.start_frame_caption"), expanded=False):
                            st.image(image_bytes, use_container_width=True)
                    else:
                        st.caption(tr("sbs.scenes.start_frame_caption"))
                        st.image(image_bytes, use_container_width=True)
            elif scene.media_type == "video":
                video_bytes = _read_bytes(scene.video_path)
                if video_bytes:
                    st.video(video_bytes)
            else:
                image_bytes = _read_bytes(scene.image_path)
                if image_bytes:
                    st.image(image_bytes, use_container_width=True)

            segment_bytes = _read_bytes(scene.segment_path)
            if segment_bytes:
                with st.expander(tr("sbs.scenes.preview_segment"), expanded=False):
                    st.video(segment_bytes)


def _render_scenes(engine: SceneBySceneEngine):
    project = st.session_state.get(K_PROJECT)
    if project is None:
        _goto(0)

    st.markdown(f"#### {tr('sbs.scenes.heading')}")
    st.caption(tr("sbs.scenes.hint"))

    ready = sum(1 for s in project.scenes if s.segment_path)
    total = len(project.scenes)
    st.markdown(f"**{tr('sbs.scenes.progress', ready=ready, total=total)}**")

    # ---- Generate everything that's still missing ----
    pending = [s for s in project.scenes if not s.segment_path]
    if pending and st.button(tr("sbs.scenes.generate_all", count=len(pending)),
                             type="primary", use_container_width=True):
        progress_bar = st.progress(ready / total if total else 0.0)
        status = st.empty()
        failed = False
        for i, scene in enumerate(project.scenes):
            if scene.segment_path:
                continue

            def stage_cb(stage, _i=i):
                status.text(tr(f"sbs.scenes.stage_{stage}", current=_i + 1))

            try:
                run_async(engine.process_scene(project, scene, i, progress_callback=stage_cb))
            except Exception as e:
                logger.exception(e)
                status.empty()
                st.error(tr("sbs.scenes.scene_failed", n=i + 1, error=str(e)))
                failed = True
                break
            done = sum(1 for s in project.scenes if s.segment_path)
            progress_bar.progress(done / total)
        if not failed:
            status.text(tr("sbs.scenes.all_done"))
            st.rerun()
        # On failure: no rerun, so the error message stays visible.
        # Partial progress is kept and shown on the next interaction.

    # ---- Per-scene cards ----
    for i, scene in enumerate(project.scenes):
        _render_scene_card(engine, project, scene, i)

    st.markdown("---")

    # ---- Navigation ----
    c_back, c_next = st.columns([1, 2])
    with c_back:
        if st.button(tr("sbs.nav.back"), key="sbs_scenes_back", use_container_width=True):
            _goto(2 if project.needs_media else 1)
    with c_next:
        all_ready = project.all_segments_ready
        if st.button(tr("sbs.scenes.continue"), key="sbs_scenes_next", type="primary",
                     use_container_width=True, disabled=not all_ready,
                     help=None if all_ready else tr("sbs.scenes.not_all_ready")):
            _goto(4)


# ============================================================================
# Step ⑤ Final
# ============================================================================

def _render_final(engine: SceneBySceneEngine):
    project = st.session_state.get(K_PROJECT)
    if project is None:
        _goto(0)

    st.markdown(f"#### {tr('sbs.final.heading')}")

    if not project.all_segments_ready:
        st.warning(tr("sbs.final.not_ready"))
        if st.button(tr("sbs.nav.back"), use_container_width=True):
            _goto(3)
        return

    total_duration = sum(s.duration for s in project.scenes)
    st.caption(tr("sbs.final.summary", count=len(project.scenes),
                  duration=f"{total_duration:.1f}"))

    # ---- BGM (reused component) ----
    bgm_params = render_bgm_section(key_prefix="sbs_final_")

    # ---- Compose ----
    if st.button(tr("sbs.final.compose_btn"), type="primary", use_container_width=True):
        try:
            import time
            start_time = time.time()
            with st.spinner(tr("sbs.final.composing")):
                result = run_async(engine.compose_final(
                    project,
                    bgm_path=bgm_params.get("bgm_path"),
                    bgm_volume=bgm_params.get("bgm_volume", 0.2),
                ))
            result["generation_time"] = time.time() - start_time
            st.session_state[K_RESULT] = result
            st.rerun()
        except Exception as e:
            logger.exception(e)
            st.error(tr("sbs.common.error", error=str(e)))

    # ---- Result ----
    result = st.session_state.get(K_RESULT)
    if result and project.final_video_path and os.path.exists(project.final_video_path):
        st.success(tr("status.video_generated", path=result["video_path"]))

        file_size_mb = result["file_size"] / (1024 * 1024)
        info = (
            f"⏱️ {result.get('generation_time', 0):.1f}s   "
            f"📦 {file_size_mb:.2f}MB   "
            f"🎬 {result['n_scenes']}{tr('info.scenes_unit')}   "
            f"⏳ {result['duration']:.1f}s"
        )
        st.caption(info)

        video_bytes = _read_bytes(result["video_path"])
        if video_bytes:
            st.video(video_bytes)
            st.download_button(
                label=tr("sbs.final.download"),
                data=video_bytes,
                file_name=os.path.basename(result["video_path"]),
                mime="video/mp4",
                use_container_width=True,
            )

    st.markdown("---")

    c_back, c_new = st.columns(2)
    with c_back:
        if st.button(tr("sbs.nav.back"), key="sbs_final_back", use_container_width=True):
            _goto(3)
    with c_new:
        if st.button(tr("sbs.final.new_project"), key="sbs_final_new", use_container_width=True):
            _reset_wizard()
            st.rerun()


# ============================================================================
# Entry point
# ============================================================================

def render_scene_wizard(pixelle_video):
    """Render the scene-by-scene wizard (call after settings/header)."""
    engine = SceneBySceneEngine(pixelle_video)
    step = st.session_state.setdefault(K_STEP, 0)
    project = st.session_state.get(K_PROJECT)

    # A project is required for every step after Setup
    if step > 0 and project is None:
        step = 0
        st.session_state[K_STEP] = 0

    _render_stepper(step, project)

    if step == 0:
        _render_setup(engine)
    elif step == 1:
        _render_script(engine)
    elif step == 2:
        _render_prompts(engine)
    elif step == 3:
        _render_scenes(engine)
    else:
        _render_final(engine)
