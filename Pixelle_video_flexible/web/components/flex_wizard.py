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
Flexible (generate-or-search) Video Wizard UI

A 6-step wizard on top of `flexvid.FlexibleVideoEngine`:

    ① Setup    – topic + scene count / language / style (TTS, template, ...)
                 + stock-provider status (Pexels / Pixabay)
    ② Script   – review / edit / AI-rewrite / add / delete narrations
    ③ Plan     – review the LLM's per-scene media plan: AI generation vs
                 stock search, with editable keywords / prompts and overrides
    ④ Source   – search scenes: candidate gallery (AI pick highlighted,
                 click to override) + download & normalize the chosen media
    ⑤ Scenes   – generate & preview audio + media + segment per scene
    ⑥ Final    – pick BGM, compose the final video, stock credits, download

Steps ②⑤⑥ mirror the scene-by-scene wizard (the engine inherits those steps
from `sbs.SceneBySceneEngine` unchanged); ③ and ④ are the flexible-pipeline
additions. All intermediate state lives in `st.session_state` (prefix
`flxw_`); edits invalidate only the affected downstream assets of that scene.
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

from flexvid import FlexibleVideoEngine, FlexScene, load_flex_config
from flexvid.project_io import find_latest_saved_project, load_project, save_project

# Session-state keys
K_STEP = "flxw_step"
K_PROJECT = "flxw_project"
K_RESULT = "flxw_result"
K_SETUP = "flxw_setup_params"   # frozen setup params (topic, style, language, ...)
K_GENALL = "flxw_genall"        # "Generate All" runs one scene per rerun while set

STEPS = ["setup", "script", "plan", "source", "scenes", "final"]
STEP_ICONS = ["🧠", "📝", "🧭", "🛒", "🎬", "🏁"]
PLAN_STEP = 2                   # media plan (skipped for static templates)
SOURCE_STEP = 3                 # source media (skipped for static templates)
SCENES_STEP = 4


# ============================================================================
# Widget-state helpers
#
# Model state (the project / scenes) is the single source of truth; widgets
# are bound to it through _bind_text/_bind_choice below, which detect whether
# the MODEL or the WIDGET moved since the last render. The older wizards'
# override-queue pattern is not needed here.
# ============================================================================

def _clear_project_widget_state():
    """Drop per-scene widget state (and binder companions) from a previous project."""
    for key in list(st.session_state.keys()):
        if key.startswith(("flxw_narr_", "flxw_query_", "flxw_prompt_", "flxw_src_",
                           "flxw_mt_", "flxw_scene_narr_", "flxw_scene_prompt_",
                           "flxw_title_w", "flxw_import_")):
            del st.session_state[key]
    st.session_state.pop(K_RESULT, None)


def _reset_wizard():
    """Start a brand-new project (keeps the setup form values)."""
    _clear_project_widget_state()
    for key in (K_PROJECT, K_SETUP, K_GENALL):
        st.session_state.pop(key, None)
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


def _bind_text(key: str, model_value: str, *, height: int = 80, label: str = "text",
               input_widget: bool = False, label_visibility: str = "collapsed"):
    """
    Text widget two-way bound to a model value, immune to stale widget state.

    A plain `if widget != model: model = widget` treats ANY mismatch as a user
    edit — but a mismatch can equally mean the MODEL was changed elsewhere
    (an edit on another step, AI rewrite, re-plan, resume) while this widget
    kept a stale copy; writing that stale copy back silently REVERTS the
    change (the "lost my script" bug). A companion key remembers the model
    value this widget was last synced to, so the two directions can be told
    apart: model moved -> push the model into the widget (never apply the
    stale copy); widget moved -> return the edit for the caller to apply.
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


def _bind_choice(key: str, model_value):
    """
    Pre-sync a choice widget (radio, ...) exactly like _bind_text: a stale
    widget copy must never overwrite a model value that changed elsewhere
    (e.g. a re-planned source). Call BEFORE instantiating the widget.
    """
    sync_key = key + "__synced"
    last_synced = st.session_state.get(sync_key)
    if key not in st.session_state:
        st.session_state[key] = model_value
    elif last_synced is None or last_synced != model_value:
        st.session_state[key] = model_value
    st.session_state[sync_key] = model_value


def _texts_differ(a: str, b: str) -> bool:
    """
    True only on a REAL content change (the browser may normalize line endings
    / edge whitespace on any button click, which must not count as an edit).
    """
    norm = lambda s: (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return norm(a) != norm(b)


def _effective_prefix(project) -> str:
    """The style prefix to use for prompt generation ('' when toggled off)."""
    if not project.params.get("use_prompt_prefix", True):
        return ""
    return project.params.get("prompt_prefix") or ""


# ============================================================================
# Stepper
# ============================================================================

def _render_stepper(step: int, project):
    """Clickable breadcrumb of the wizard steps."""
    cols = st.columns(len(STEPS))
    for i, (name, icon) in enumerate(zip(STEPS, STEP_ICONS)):
        label = f"{icon} {tr(f'flxw.stepper.{name}')}"
        skipped = (i in (PLAN_STEP, SOURCE_STEP)
                   and project is not None and not project.needs_media) \
            or (i == SOURCE_STEP and project is not None
                and project.params.get("generate_only"))
        revisitable = i < step and not skipped and project is not None
        with cols[i]:
            if i == step:
                st.button(label, key=f"flxw_step_btn_{i}", type="primary",
                          use_container_width=True, disabled=True)
            elif revisitable:
                if st.button(label, key=f"flxw_step_btn_{i}", use_container_width=True):
                    _goto(i)
            else:
                help_text = tr("sbs.stepper.skipped_static") if skipped else None
                st.button(label, key=f"flxw_step_btn_{i}", use_container_width=True,
                          disabled=True, help=help_text)
    st.markdown("")


# ============================================================================
# Step ① Setup
# ============================================================================

def _render_video_mode_section(engine: FlexibleVideoEngine, style_params: dict) -> dict:
    """
    Video-generation mode selector — only for video templates (same behavior
    as the scene-by-scene / PDF wizards: t2v by default, i2v offered when an
    i2v-capable workflow is installed).
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
            key="flxw_setup_video_mode",
        )
        if mode != "i2v":
            return {"media_mode": "t2v"}

        st.caption(tr("sbs.setup.i2v_hint"))

        i2v_workflow = st.selectbox(
            tr("sbs.setup.i2v_workflow_label"),
            options=i2v_workflows,
            key="flxw_setup_i2v_workflow",
        )

        image_workflows = engine.list_image_workflows()
        default_label = tr("sbs.setup.i2v_image_workflow_default")
        image_choice = st.selectbox(
            tr("sbs.setup.i2v_image_workflow_label"),
            options=[default_label] + image_workflows,
            key="flxw_setup_i2v_image_workflow",
        )

        return {
            "media_mode": "i2v",
            "i2v_workflow": i2v_workflow,
            "i2v_image_workflow": None if image_choice == default_label else image_choice,
        }


_LANGUAGE_CHOICES = [
    "auto", "English", "Vietnamese", "Chinese", "Japanese",
    "Korean", "French", "German", "Spanish",
]


def _render_providers_panel(engine: FlexibleVideoEngine) -> dict:
    """Stock-search status (which providers are configured) + sourcing mode."""
    with st.container(border=True):
        st.markdown(f"**{tr('flxw.setup.providers_label')}**")
        if engine.aggregator.enabled:
            st.success(tr("flxw.setup.providers_ok",
                          providers=", ".join(engine.aggregator.provider_names)))
        else:
            st.warning(tr("flxw.setup.providers_none"))
        st.caption(tr("flxw.setup.providers_hint"))

        # 🤖 auto (LLM decides per scene) / 🔒 stock-only / 🎨 generate-only
        if engine.flex_config.generate_only:
            default_mode = "generate"
        elif engine.flex_config.search_only and engine.aggregator.enabled:
            default_mode = "stock"
        else:
            default_mode = "auto"
        mode_key = "flxw_setup_sourcing"
        if mode_key not in st.session_state:
            st.session_state[mode_key] = default_mode
        mode = st.radio(
            tr("flxw.setup.sourcing_label"),
            ["auto", "stock", "generate"],
            format_func=lambda x: tr(f"flxw.setup.sourcing_{x}"),
            help=tr("flxw.setup.sourcing_help"),
            key=mode_key,
        )
        if mode == "stock" and not engine.aggregator.enabled:
            st.caption(tr("flxw.setup.sourcing_stock_needs_keys"))
    return {
        "search_only": mode == "stock" and engine.aggregator.enabled,
        "generate_only": mode == "generate",
    }


def _render_resume_saved(engine: FlexibleVideoEngine):
    """Offer to resume the latest unfinished checkpoint after a session reset."""
    saved = find_latest_saved_project()
    if not saved:
        return
    with st.container(border=True):
        st.info(tr("flxw.setup.resume_saved_info",
                   title=saved["title"], ready=saved["ready"],
                   total=saved["total"], saved=saved["saved_at"]))
        if st.button(tr("flxw.setup.resume_saved_btn"), use_container_width=True):
            try:
                project, step = load_project(saved["path"])
                _clear_project_widget_state()
                st.session_state[K_PROJECT] = project
                st.session_state[K_SETUP] = {
                    "topic": project.params.get("text"),
                    "n_scenes": project.params.get("n_scenes", len(project.scenes)),
                    "language": project.params.get("flex_language"),
                }
                # Resume on a sensible step: never past Scenes (the final step
                # recomposes), never before Script.
                _goto(min(max(step or SCENES_STEP, 1), SCENES_STEP))
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))


def _render_setup(engine: FlexibleVideoEngine):
    pixelle_video = engine.core
    project = st.session_state.get(K_PROJECT)

    if project is None:
        _render_resume_saved(engine)

    left_col, right_col = st.columns([1, 1])

    # ---- Topic input ----
    with left_col:
        with st.container(border=True):
            st.markdown(f"**{tr('flxw.setup.topic_label')}**")
            topic = st.text_area(
                "topic",
                placeholder=tr("flxw.setup.topic_placeholder"),
                help=tr("flxw.setup.topic_help"),
                height=100,
                label_visibility="collapsed",
                key="flxw_setup_topic",
            )

            n_scenes = st.slider(
                tr("video.frames"),
                min_value=3, max_value=30, value=5,
                help=tr("video.frames_help"),
                key="flxw_setup_n_scenes",
            )
            st.caption(tr("video.frames_label", n=n_scenes))

            language = st.selectbox(
                tr("flxw.setup.language_label"),
                options=_LANGUAGE_CHOICES,
                format_func=lambda x: tr("flxw.setup.language_auto") if x == "auto" else x,
                help=tr("flxw.setup.language_help"),
                key="flxw_setup_language",
            )

        provider_params = _render_providers_panel(engine)

        # ---- Narration loudness (TTS output is often quiet; fixed post-TTS) ----
        with st.container(border=True):
            st.markdown(f"**{tr('flxw.setup.audio_label')}**")
            tts_normalize = st.toggle(
                tr("flxw.setup.tts_normalize_label"),
                value=True,
                help=tr("flxw.setup.tts_normalize_help"),
                key="flxw_setup_tts_normalize",
            )
            tts_volume = st.slider(
                tr("flxw.setup.tts_volume_label"),
                min_value=0.5, max_value=3.0, value=1.0, step=0.1,
                help=tr("flxw.setup.tts_volume_help"),
                key="flxw_setup_tts_volume",
            )

    # ---- Style configuration (reused component: TTS / template / workflow) ----
    with right_col:
        style_params = render_style_config(pixelle_video)
        video_mode_params = _render_video_mode_section(engine, style_params)

    st.caption(tr("sbs.setup.params_frozen_hint"))

    # ---- Actions ----
    c1, c2 = st.columns([2, 1]) if project is not None else (st.container(), None)
    with c1:
        if st.button(tr("sbs.setup.generate_btn"), type="primary", use_container_width=True):
            if not config_manager.validate():
                st.error(tr("settings.not_configured"))
                st.stop()
            if not (topic or "").strip():
                st.error(tr("flxw.setup.no_topic_warning"))
                st.stop()

            script_topic = topic.strip()
            if language != "auto":
                # generate_script has no language parameter; the topic prompt
                # preserves the input language, so the instruction rides along.
                script_topic = f"{script_topic}\n\n(Write the narration in {language}.)"

            try:
                with st.spinner(tr("flxw.setup.scripting")):
                    title, narrations = run_async(engine.generate_script(
                        script_topic, mode="generate", n_scenes=n_scenes,
                    ))

                params = {
                    "text": topic.strip(),
                    "mode": "generate",
                    "n_scenes": n_scenes,
                    "title": title,
                    **style_params,
                    **video_mode_params,
                    # --- Narration loudness (applied post-TTS by the engine) ---
                    "tts_volume": tts_volume,
                    "tts_normalize": tts_normalize,
                    # --- Media sourcing ---
                    "search_only": provider_params.get("search_only", False),
                    "generate_only": provider_params.get("generate_only", False),
                    # --- Provenance ---
                    "flex_language": None if language == "auto" else language,
                    "flex_providers": engine.aggregator.provider_names,
                }
                st.session_state[K_SETUP] = {
                    "topic": script_topic,
                    "n_scenes": n_scenes,
                    "language": None if language == "auto" else language,
                }
                _clear_project_widget_state()
                st.session_state[K_PROJECT] = engine.create_project(
                    title, narrations, [None] * len(narrations), params
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

def _apply_narration_change(project, scene, value: str):
    """
    Narration changed -> audio is stale. GENERATED video clips are
    length-synced to the audio, so they go stale too (i2v keeps its
    still-valid start image). Stock and imported media are NOT length-synced
    (the segment step pads/trims them to the narration), so they survive.
    """
    scene.narration = value
    scene.invalidate_audio()
    if isinstance(scene, FlexScene) and (scene.is_search or scene.is_import):
        return
    if scene.media_origin == "imported":
        return
    if project.is_i2v:
        scene.invalidate_video()
    elif project.is_video_workflow:
        scene.invalidate_media()


def _render_script(engine: FlexibleVideoEngine):
    project = st.session_state.get(K_PROJECT)
    setup = st.session_state.get(K_SETUP) or {}
    if project is None:
        _goto(0)

    st.markdown(f"#### {tr('sbs.script.heading')}")
    st.caption(tr("flxw.script.hint"))

    # ---- Title ----
    new_title = _bind_text("flxw_title_w", project.title,
                           label=tr("sbs.script.title_label"),
                           input_widget=True, label_visibility="visible")
    if new_title.strip() and _texts_differ(new_title, project.title):
        project.title = new_title.strip()

    # ---- Scenes ----
    for i, scene in enumerate(list(project.scenes)):
        with st.container(border=True):
            st.markdown(f"**{tr('sbs.script.scene_label', n=i + 1)}**")

            key = f"flxw_narr_{scene.uid}"
            value = _bind_text(key, scene.narration)
            if _texts_differ(value, scene.narration):
                _apply_narration_change(project, scene, value)

            c1, c2, _sp = st.columns([1, 1, 2])
            with c1:
                if st.button(tr("sbs.script.rewrite"), key=f"flxw_rw_{scene.uid}",
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
                if st.button(tr("sbs.script.delete"), key=f"flxw_del_{scene.uid}",
                             use_container_width=True,
                             disabled=len(project.scenes) <= 1):
                    project.scenes.remove(scene)
                    st.rerun()

    if st.button(tr("sbs.script.add"), use_container_width=True):
        project.scenes.append(FlexScene(narration=""))
        st.rerun()

    st.markdown("---")

    # ---- Navigation ----
    c_back, c_regen, c_next = st.columns([1, 1, 2])
    with c_back:
        if st.button(tr("sbs.nav.back"), key="flxw_script_back", use_container_width=True):
            _goto(0)
    with c_regen:
        if st.button(tr("sbs.script.regen_all"), key="flxw_script_regen",
                     use_container_width=True, disabled=not setup.get("topic")):
            try:
                with st.spinner(tr("flxw.setup.scripting")):
                    title, narrations = run_async(engine.generate_script(
                        setup.get("topic"), mode="generate",
                        n_scenes=setup.get("n_scenes", len(project.scenes)),
                    ))
                _clear_project_widget_state()
                project.title = title
                project.scenes = [FlexScene(narration=n) for n in narrations]
                st.rerun()
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))
    with c_next:
        if st.button(tr("sbs.script.continue"), key="flxw_script_next",
                     type="primary", use_container_width=True):
            if any(not s.narration.strip() for s in project.scenes):
                st.error(tr("sbs.script.empty_warning"))
            else:
                _goto(PLAN_STEP if project.needs_media else SCENES_STEP)


# ============================================================================
# Step ③ Media plan
# ============================================================================

def _scene_is_planned(scene) -> bool:
    return bool(getattr(scene, "plan_reason", "") or getattr(scene, "search_query", None)
                or scene.prompt)


def _apply_source_change(scene: FlexScene, source: str):
    """User flipped generate <-> search: sourcing state is stale, text survives."""
    scene.source = source
    scene.invalidate_search()
    scene.invalidate_media()


def _render_ai_prompt_button(engine: FlexibleVideoEngine, project, scene, key: str):
    """
    "🎲 AI prompt" — (re)write this scene's generation prompt from its
    narration. The main use case: a scene flipped from stock search to AI
    generate has no prompt yet (search scenes carry only keywords).
    """
    label = (tr("flxw.plan.gen_prompt_regen_btn") if (scene.prompt or "").strip()
             else tr("flxw.plan.gen_prompt_btn"))
    if st.button(label, key=key, disabled=not scene.narration.strip(),
                 help=tr("flxw.plan.gen_prompt_help")):
        try:
            with st.spinner(tr("sbs.prompts.regenerating")):
                new_prompt = run_async(engine.generate_prompt_for(
                    scene.narration, prompt_prefix=_effective_prefix(project)
                ))
            scene.prompt = new_prompt
            scene.invalidate_media()
            st.rerun()   # the binders push the new prompt into the widgets
        except Exception as e:
            logger.exception(e)
            st.error(tr("sbs.common.error", error=str(e)))


def _render_plan(engine: FlexibleVideoEngine):
    project = st.session_state.get(K_PROJECT)
    if project is None:
        _goto(0)
    if not project.needs_media:
        _goto(SCENES_STEP)

    st.markdown(f"#### {tr('flxw.plan.heading')}")
    st.caption(tr("flxw.plan.hint"))
    generate_only = engine._generate_only(project)
    if generate_only:
        st.info(tr("flxw.plan.generate_only_info"))
    elif not engine.aggregator.enabled:
        st.info(tr("flxw.plan.no_providers_info"))
    elif project.params.get("search_only"):
        st.info(tr("flxw.plan.search_only_info"))

    prompt_prefix = _effective_prefix(project)

    # ---- Auto-plan on first entry ----
    if not any(_scene_is_planned(s) for s in project.scenes):
        try:
            with st.spinner(tr("flxw.script.planning")):
                run_async(engine.generate_scene_plan(project, prompt_prefix=prompt_prefix))
            # The widget binders below pick the fresh plan up from the scenes.
        except Exception as e:
            logger.exception(e)
            st.error(tr("sbs.common.error", error=str(e)))
            if st.button(tr("sbs.prompts.retry"), type="primary"):
                st.rerun()
            return

    is_video_project = project.is_video_workflow

    # ---- Per-scene plan cards ----
    for i, scene in enumerate(project.scenes):
        with st.container(border=True):
            st.markdown(f"**{tr('sbs.script.scene_label', n=i + 1)}**")
            st.caption(f"🗣️ {scene.narration}")

            src_col, mt_col, btn_col = st.columns([2, 2, 1])
            with src_col:
                src_key = f"flxw_src_{scene.uid}"
                _bind_choice(src_key, scene.source)
                source = st.radio(
                    tr("flxw.plan.source_label"),
                    ["generate", "search"],
                    horizontal=True,
                    format_func=lambda x: tr(f"flxw.plan.source_{x}"),
                    key=src_key,
                    disabled=not engine.aggregator.enabled or generate_only,
                )
                if source != scene.source:
                    _apply_source_change(scene, source)

            with mt_col:
                if is_video_project and scene.source == "search":
                    mt_key = f"flxw_mt_{scene.uid}"
                    _bind_choice(mt_key, scene.plan_media_type)
                    media_type = st.radio(
                        tr("flxw.plan.media_type_label"),
                        ["video", "image"],
                        horizontal=True,
                        format_func=lambda x: tr(f"flxw.plan.media_{x}"),
                        key=mt_key,
                    )
                    if media_type != scene.plan_media_type:
                        scene.plan_media_type = media_type
                        scene.invalidate_search()

            with btn_col:
                if st.button(tr("flxw.plan.replan_one"), key=f"flxw_rp_{scene.uid}",
                             use_container_width=True):
                    try:
                        with st.spinner(tr("flxw.plan.replanning")):
                            run_async(engine.regenerate_plan_for(
                                project, scene, prompt_prefix=prompt_prefix
                            ))
                        st.rerun()   # binders push the new plan into the widgets
                    except Exception as e:
                        logger.exception(e)
                        st.error(tr("sbs.common.error", error=str(e)))

            if scene.source == "search":
                st.caption(tr("flxw.plan.query_label"))
                qkey = f"flxw_query_{scene.uid}"
                qval = _bind_text(qkey, scene.search_query or "",
                                  label="query", input_widget=True)
                if _texts_differ(qval, scene.search_query or ""):
                    scene.search_query = qval.strip() or None
                    scene.invalidate_search()
            else:
                st.caption(tr("flxw.plan.prompt_label"))
                pkey = f"flxw_prompt_{scene.uid}"
                pval = _bind_text(pkey, scene.prompt or "", height=100, label="prompt")
                if _texts_differ(pval, scene.prompt or ""):
                    scene.prompt = pval
                    scene.invalidate_media()
                # Switched from search to generate? The scene has no prompt
                # yet — write one from the narration with a click.
                _render_ai_prompt_button(engine, project, scene,
                                         key=f"flxw_genprompt_{scene.uid}")

            if scene.plan_reason:
                st.caption(tr("flxw.plan.reason_caption", reason=scene.plan_reason))

    st.markdown("---")

    # ---- Navigation ----
    c_back, c_regen, c_next = st.columns([1, 1, 2])
    with c_back:
        if st.button(tr("sbs.nav.back"), key="flxw_plan_back", use_container_width=True):
            _goto(1)
    with c_regen:
        if st.button(tr("flxw.plan.replan_all"), key="flxw_plan_regen",
                     use_container_width=True):
            try:
                with st.spinner(tr("flxw.script.planning")):
                    run_async(engine.generate_scene_plan(project, prompt_prefix=prompt_prefix))
                st.rerun()   # binders push the new plan into the widgets
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))
    with c_next:
        if st.button(tr("flxw.plan.continue"), key="flxw_plan_next",
                     type="primary", use_container_width=True):
            search_scenes = [s for s in project.scenes if s.source == "search"]
            if any(not (s.search_query or "").strip() for s in search_scenes):
                st.error(tr("flxw.plan.empty_query_warning"))
            elif any(s.source == "generate" and not (s.prompt or "").strip()
                     for s in project.scenes):
                st.error(tr("flxw.plan.empty_prompt_warning"))
            else:
                # Generate-only: nothing to source, go straight to Scenes
                _goto(SCENES_STEP if generate_only else SOURCE_STEP)


# ============================================================================
# Step ④ Source media
# ============================================================================

def _candidate_caption(candidate, picked: bool) -> str:
    parts = []
    if picked:
        parts.append(tr("flxw.source.auto_pick_caption"))
    parts.append(f"{candidate.source} · {candidate.width}x{candidate.height}")
    if candidate.duration:
        parts.append(f"{candidate.duration:.0f}s")
    if candidate.photographer:
        parts.append(candidate.photographer)
    return " · ".join(parts)


def _render_candidate_gallery(engine: FlexibleVideoEngine, project, scene, index: int):
    """Thumbnail grid with the AI pick highlighted and click-to-override."""
    st.caption(tr("flxw.source.candidates_label"))
    n_cols = 3
    for row_start in range(0, len(scene.candidates), n_cols):
        cols = st.columns(n_cols)
        for col, candidate in zip(cols, scene.candidates[row_start:row_start + n_cols]):
            with col:
                picked = candidate.id == scene.picked_candidate_id
                with st.container(border=picked):
                    if candidate.thumbnail_url:
                        st.image(candidate.thumbnail_url, use_container_width=True)
                    st.caption(_candidate_caption(candidate, picked))
                    label = (tr("flxw.source.picked_btn") if picked
                             else tr("flxw.source.pick_btn"))
                    if st.button(label, key=f"flxw_pick_{scene.uid}_{candidate.id}",
                                 use_container_width=True, disabled=picked):
                        scene.picked_candidate_id = candidate.id
                        st.rerun()


def _render_generate_prompt_editor(engine: FlexibleVideoEngine, project, scene,
                                   label_key: str = "flxw.plan.prompt_label"):
    """Prompt editor for generate / fallen-back scenes on the Source step."""
    st.caption(tr(label_key))
    pkey = f"flxw_prompt_{scene.uid}"
    pval = _bind_text(pkey, scene.prompt or "", height=100, label="prompt")
    if _texts_differ(pval, scene.prompt or ""):
        scene.prompt = pval
        scene.invalidate_media()
    _render_ai_prompt_button(engine, project, scene,
                             key=f"flxw_src_genprompt_{scene.uid}")


def _save_uploaded_media(project, scene, uploaded) -> str:
    """Write an upload next to the scene assets as a raw temp file."""
    ext = os.path.splitext(uploaded.name)[1].lower() or ".bin"
    raw_path = get_task_path(project.task_id, "frames", f"{scene.uid}_import_raw{ext}")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return raw_path


def _render_import_control(engine: FlexibleVideoEngine, project, scene, index: int):
    """
    Upload control: use your own image/video for this scene instead of
    generating or searching (the engine normalizes the file into the
    canonical scene asset and flips the scene's source to 'import').
    """
    with st.expander(tr("flxw.source.import_expander", "📥 Use my own image/video"),
                     expanded=False):
        st.caption(tr("flxw.source.import_hint",
                      "The file is cropped/re-encoded to the project size; video "
                      "clips are padded or trimmed to the narration length."))
        uploaded = st.file_uploader(
            "import media",
            type=[e.lstrip(".") for e in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS],
            key=f"flxw_import_file_{scene.uid}",
            label_visibility="collapsed",
        )
        if st.button(tr("flxw.source.import_apply", "📥 Use this file"),
                     key=f"flxw_import_btn_{scene.uid}",
                     use_container_width=True, disabled=uploaded is None):
            try:
                with st.spinner(tr("flxw.source.import_applying", "Importing media...")):
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


def _source_badge(scene) -> str:
    if scene.is_import:
        return tr("flxw.scenes.source_import_badge", "📥 Imported")
    if scene.is_search:
        return tr("flxw.scenes.source_search_badge")
    return tr("flxw.scenes.source_generate_badge")


def _render_source_scene(engine: FlexibleVideoEngine, project, scene, index: int):
    with st.container(border=True):
        st.markdown(f"**{tr('sbs.script.scene_label', n=index + 1)}**  "
                    f"{_source_badge(scene)}")
        st.caption(f"🗣️ {scene.narration}")

        # Importing your own file is an alternative to BOTH generate and search
        _render_import_control(engine, project, scene, index)

        # ---- Imported scenes: show the applied media ----
        if scene.is_import:
            if scene.has_media:
                st.success(tr("flxw.source.import_ready", "✅ Imported media applied"))
                if scene.media_type == "video":
                    video_bytes = _read_bytes(scene.video_path)
                    if video_bytes:
                        st.video(video_bytes)
                else:
                    image_bytes = _read_bytes(scene.image_path)
                    if image_bytes:
                        st.image(image_bytes, use_container_width=True)
            return

        # ---- Generate scenes: nothing to source here ----
        if not scene.is_search:
            if scene.fell_back_to_generate:
                st.info(tr("flxw.source.fallback_notice"))
            else:
                st.caption(tr("flxw.source.generate_scene_caption"))
            _render_generate_prompt_editor(engine, project, scene)
            return

        # ---- Search scenes ----
        st.caption(f"{tr('flxw.plan.query_label')}: `{scene.search_query}`")

        applied = scene.attribution is not None and scene.has_media

        c_search, c_apply, _sp = st.columns([1, 1, 2])
        with c_search:
            search_label = (tr("flxw.source.re_search_btn") if scene.search_attempted
                            else tr("flxw.source.search_btn"))
            if st.button(search_label, key=f"flxw_search_{scene.uid}",
                         use_container_width=True,
                         disabled=not (scene.search_query or "").strip()):
                try:
                    with st.spinner(tr("flxw.source.searching")):
                        run_async(engine.search_scene_media(
                            project, scene, index,
                            prompt_prefix=_effective_prefix(project),
                        ))
                    st.rerun()
                except Exception as e:
                    logger.exception(e)
                    st.error(tr("sbs.common.error", error=str(e)))
        with c_apply:
            if scene.candidates and st.button(
                    tr("flxw.source.use_pick_btn"), key=f"flxw_apply_{scene.uid}",
                    type="primary", use_container_width=True,
                    disabled=scene.picked_candidate_id is None):
                try:
                    with st.spinner(tr("flxw.source.applying")):
                        run_async(engine.apply_picked_candidate(project, scene, index))
                    st.rerun()
                except Exception as e:
                    logger.exception(e)
                    st.error(tr("sbs.common.error", error=str(e)))

        if scene.search_attempted and not scene.candidates:
            st.warning(tr("flxw.source.no_results"))

        if scene.candidates:
            _render_candidate_gallery(engine, project, scene, index)

        # ---- Applied media preview + attribution ----
        if applied:
            st.success(tr("flxw.source.media_ready"))
            attribution = scene.attribution or {}
            st.caption(tr("flxw.source.attribution_caption",
                          photographer=attribution.get("photographer") or "unknown",
                          source=attribution.get("source") or "",
                          license=attribution.get("license") or ""))
            if scene.media_type == "video":
                video_bytes = _read_bytes(scene.video_path)
                if video_bytes:
                    st.video(video_bytes)
            else:
                image_bytes = _read_bytes(scene.image_path)
                if image_bytes:
                    st.image(image_bytes, use_container_width=True)


def _render_source(engine: FlexibleVideoEngine):
    project = st.session_state.get(K_PROJECT)
    if project is None:
        _goto(0)
    if not project.needs_media or engine._generate_only(project):
        _goto(SCENES_STEP)

    st.markdown(f"#### {tr('flxw.source.heading')}")
    st.caption(tr("flxw.source.hint"))

    # ---- Search everything that hasn't been searched yet ----
    pending = [s for s in project.scenes if s.is_search and not s.search_attempted
               and (s.search_query or "").strip()]
    if pending and st.button(tr("flxw.source.search_all_btn", count=len(pending)),
                             use_container_width=True):
        progress_bar = st.progress(0.0)
        prefix = _effective_prefix(project)
        for done, scene in enumerate(pending, start=1):
            index = project.scenes.index(scene)
            try:
                run_async(engine.search_scene_media(project, scene, index,
                                                    prompt_prefix=prefix))
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.scenes.scene_failed", n=index + 1, error=str(e)))
                break
            progress_bar.progress(done / len(pending))
        st.rerun()

    # ---- Per-scene cards ----
    for i, scene in enumerate(project.scenes):
        _render_source_scene(engine, project, scene, i)

    st.markdown("---")

    not_sourced = [s for s in project.scenes
                   if s.is_search and not (s.attribution and s.has_media)]
    if not_sourced:
        st.caption(tr("flxw.source.not_sourced_warning"))

    # ---- Navigation ----
    c_back, c_next = st.columns([1, 2])
    with c_back:
        if st.button(tr("sbs.nav.back"), key="flxw_source_back", use_container_width=True):
            _goto(PLAN_STEP)
    with c_next:
        if st.button(tr("flxw.source.continue"), key="flxw_source_next",
                     type="primary", use_container_width=True):
            _goto(SCENES_STEP)


# ============================================================================
# Step ⑤ Scenes
# ============================================================================

def _scene_media_ready(project, scene) -> bool:
    """Search scenes count any applied media; generate scenes follow the project mode."""
    if not project.needs_media:
        return True
    if isinstance(scene, FlexScene) and scene.is_search:
        return scene.has_media
    return project.scene_media_ready(scene)


def _scene_status_icons(project, scene) -> str:
    parts = [f"🎤{'✅' if scene.audio_path else '⬜'}"]
    is_search = isinstance(scene, FlexScene) and scene.is_search
    is_import = isinstance(scene, FlexScene) and scene.is_import
    if is_import:
        parts.append(f"📥{'✅' if scene.has_media else '⬜'}")
    elif is_search:
        parts.append(f"🔎{'✅' if scene.has_media else '⬜'}")
    elif project.is_i2v:
        # image → video: the still and the animated clip are separate steps
        parts.append(f"🖼️{'✅' if scene.image_path else '⬜'}")
        parts.append(f"🎬{'✅' if scene.video_path else '⬜'}")
    elif project.needs_media:
        media_icon = "🎬" if project.is_video_workflow else "🖼️"
        parts.append(f"{media_icon}{'✅' if scene.has_media else '⬜'}")
    parts.append(f"🎞️{'✅' if scene.segment_path else '⬜'}")
    return " ".join(parts)


def _stage_text(stage: str, current: int) -> str:
    """Progress text for a process_scene stage (flex stages + inherited ones)."""
    if stage in ("search", "rank", "download"):
        return tr(f"flxw.scenes.stage_{stage}", current=current)
    return tr(f"sbs.scenes.stage_{stage}", current=current)


async def _source_stock_media(engine: FlexibleVideoEngine, project, scene, index: int,
                              prompt_prefix: str):
    """Search → rank → download for one scene (used by the Stock button)."""
    scene.invalidate_search()
    await engine.search_scene_media(project, scene, index, prompt_prefix=prompt_prefix)
    if scene.is_search and scene.picked_candidate_id:
        await engine.apply_picked_candidate(project, scene, index)


def _render_scene_card(engine: FlexibleVideoEngine, project, scene, index: int):
    is_search = isinstance(scene, FlexScene) and scene.is_search
    is_import = isinstance(scene, FlexScene) and scene.is_import

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
            nkey = f"flxw_scene_narr_{scene.uid}"
            nval = _bind_text(nkey, scene.narration, label="narration")
            if _texts_differ(nval, scene.narration):
                _apply_narration_change(project, scene, nval)

            if is_import:
                st.caption(tr("flxw.scenes.import_caption",
                              "📥 Uses your imported media (re-import on the Source step)"))
            elif project.needs_media and not is_search:
                st.caption(tr("sbs.scenes.prompt_label"))
                pkey = f"flxw_scene_prompt_{scene.uid}"
                pval = _bind_text(pkey, scene.prompt or "", height=100, label="prompt")
                if _texts_differ(pval, scene.prompt):
                    scene.prompt = pval
                    scene.invalidate_media()
            elif is_search:
                st.caption(f"{tr('flxw.plan.query_label')}: `{scene.search_query}`")
                if scene.attribution:
                    st.caption(tr("flxw.source.attribution_caption",
                                  photographer=scene.attribution.get("photographer") or "unknown",
                                  source=scene.attribution.get("source") or "",
                                  license=scene.attribution.get("license") or ""))

            # ---- Action buttons ----
            def _run_step(spinner_text: str, coro_factory):
                try:
                    with st.spinner(spinner_text):
                        run_async(coro_factory())
                    st.rerun()
                except Exception as e:
                    logger.exception(e)
                    st.error(tr("sbs.common.error", error=str(e)))

            cols = st.columns(4 if (project.is_i2v and not is_search and not is_import) else 3)

            with cols[0]:
                audio_label = tr("sbs.scenes.audio_regen") if scene.audio_path else tr("sbs.scenes.audio_btn")
                if st.button(audio_label, key=f"flxw_a_{scene.uid}", use_container_width=True,
                             disabled=not scene.narration.strip()):
                    _run_step(_stage_text("audio", index + 1),
                              lambda: engine.generate_audio(project, scene, index))

            if is_import:
                pass  # imported media: audio + segment only (re-import on Source step)

            elif is_search:
                with cols[1]:
                    stock_label = (tr("flxw.source.re_search_btn") if scene.has_media
                                   else tr("flxw.source.search_btn"))
                    if st.button(stock_label, key=f"flxw_m_{scene.uid}", use_container_width=True,
                                 disabled=not (scene.search_query or "").strip()):
                        _run_step(_stage_text("search", index + 1),
                                  lambda: _source_stock_media(
                                      engine, project, scene, index,
                                      _effective_prefix(project)))

            elif project.is_i2v:
                # i2v: strict order — audio → start image → animate.
                with cols[1]:
                    img_label = (tr("sbs.scenes.image_start_regen") if scene.image_path
                                 else tr("sbs.scenes.image_start_btn"))
                    img_needs_audio = not scene.audio_path
                    img_disabled = img_needs_audio or not (scene.prompt or "").strip()
                    if st.button(img_label, key=f"flxw_m_{scene.uid}", use_container_width=True,
                                 disabled=img_disabled,
                                 help=tr("sbs.scenes.need_audio_first") if img_needs_audio else None):
                        _run_step(_stage_text("image_start", index + 1),
                                  lambda: engine.generate_start_image(project, scene, index))

                with cols[2]:
                    anim_label = (tr("sbs.scenes.animate_regen") if scene.video_path
                                  else tr("sbs.scenes.animate_btn"))
                    missing = []
                    if not scene.image_path:
                        missing.append(tr("sbs.scenes.need_image_first"))
                    if not scene.audio_path:
                        missing.append(tr("sbs.scenes.need_audio_first"))
                    if st.button(anim_label, key=f"flxw_v_{scene.uid}", use_container_width=True,
                                 disabled=bool(missing),
                                 help=" / ".join(missing) if missing else None):
                        _run_step(_stage_text("animate", index + 1),
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
                    if st.button(media_label, key=f"flxw_m_{scene.uid}", use_container_width=True,
                                 disabled=disabled, help=help_text):
                        _run_step(_stage_text("media", index + 1),
                                  lambda: engine.generate_media(project, scene, index))

            with cols[-1]:
                seg_label = tr("sbs.scenes.segment_regen") if scene.segment_path else tr("sbs.scenes.segment_btn")
                seg_ready = scene.audio_path and _scene_media_ready(project, scene)
                if st.button(seg_label, key=f"flxw_s_{scene.uid}", use_container_width=True,
                             disabled=not seg_ready,
                             help=None if seg_ready else tr("sbs.scenes.segment_requirements")):
                    _run_step(_stage_text("segment", index + 1),
                              lambda: engine.render_segment(project, scene, index))

        # ---- Previews (bytes-based so regenerated files refresh properly) ----
        with preview_col:
            audio_bytes = _read_bytes(scene.audio_path)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

            if project.is_i2v and not is_search:
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


def _render_scenes(engine: FlexibleVideoEngine):
    project = st.session_state.get(K_PROJECT)
    if project is None:
        _goto(0)

    st.markdown(f"#### {tr('sbs.scenes.heading')}")
    st.caption(tr("sbs.scenes.hint"))

    ready = sum(1 for s in project.scenes if s.segment_path)
    total = len(project.scenes)
    st.markdown(f"**{tr('sbs.scenes.progress', ready=ready, total=total)}**")

    # ---- Generate everything that's still missing ----
    # One scene per script run: each completed scene is checkpointed to disk
    # and triggers a rerun, so the previews update live and a dropped
    # browser/tunnel session can be resumed from the Setup step instead of
    # losing the whole run (long single-run loops are exactly when remote
    # websockets die).
    pending = [s for s in project.scenes if not s.segment_path]
    generating = st.session_state.get(K_GENALL, False)

    if pending and not generating:
        if st.button(tr("sbs.scenes.generate_all", count=len(pending)),
                     type="primary", use_container_width=True):
            st.session_state[K_GENALL] = True
            st.rerun()

    if generating:
        if not pending:
            st.session_state[K_GENALL] = False
            st.success(tr("sbs.scenes.all_done"))
        elif st.button(tr("flxw.scenes.genall_stop"), use_container_width=True):
            st.session_state[K_GENALL] = False
            st.rerun()
        else:
            scene = pending[0]
            index = project.scenes.index(scene)
            st.caption(tr("flxw.scenes.genall_running",
                          ready=ready, total=total, current=index + 1))
            progress_bar = st.progress(ready / total if total else 0.0)
            status = st.empty()

            def stage_cb(stage, _i=index):
                status.text(_stage_text(stage, _i + 1))

            try:
                run_async(engine.process_scene(project, scene, index,
                                               progress_callback=stage_cb))
                save_project(project, step=SCENES_STEP)
            except Exception as e:
                logger.exception(e)
                st.session_state[K_GENALL] = False
                status.empty()
                st.error(tr("sbs.scenes.scene_failed", n=index + 1, error=str(e)))
                # No rerun: the error stays visible; partial progress is kept.
            else:
                progress_bar.progress((ready + 1) / total)
                st.rerun()   # next pending scene (or the all-done state)

    # ---- Per-scene cards ----
    for i, scene in enumerate(project.scenes):
        _render_scene_card(engine, project, scene, i)

    st.markdown("---")

    # ---- Navigation ----
    c_back, c_next = st.columns([1, 2])
    with c_back:
        if st.button(tr("sbs.nav.back"), key="flxw_scenes_back", use_container_width=True):
            if not project.needs_media:
                _goto(1)
            elif engine._generate_only(project):
                _goto(PLAN_STEP)
            else:
                _goto(SOURCE_STEP)
    with c_next:
        all_ready = project.all_segments_ready
        if st.button(tr("sbs.scenes.continue"), key="flxw_scenes_next", type="primary",
                     use_container_width=True, disabled=not all_ready,
                     help=None if all_ready else tr("sbs.scenes.not_all_ready")):
            _goto(5)


# ============================================================================
# Step ⑥ Final
# ============================================================================

def _render_credits(project):
    """Stock media credits (Pexels/Pixabay attribution duty)."""
    credited = [(i, s) for i, s in enumerate(project.scenes)
                if getattr(s, "attribution", None)]
    if not credited:
        return
    with st.expander(tr("flxw.final.credits_label"), expanded=False):
        st.caption(tr("flxw.final.credits_hint"))
        for i, scene in credited:
            attribution = scene.attribution or {}
            line = tr("flxw.final.credits_line",
                      n=i + 1,
                      photographer=attribution.get("photographer") or "unknown",
                      source=attribution.get("source") or "",
                      license=attribution.get("license") or "")
            page_url = attribution.get("page_url")
            st.markdown(f"- {line}" + (f" — [{page_url}]({page_url})" if page_url else ""))


def _render_final(engine: FlexibleVideoEngine):
    project = st.session_state.get(K_PROJECT)
    if project is None:
        _goto(0)

    st.markdown(f"#### {tr('sbs.final.heading')}")

    if not project.all_segments_ready:
        st.warning(tr("sbs.final.not_ready"))
        if st.button(tr("sbs.nav.back"), use_container_width=True):
            _goto(SCENES_STEP)
        return

    total_duration = sum(s.duration for s in project.scenes)
    st.caption(tr("sbs.final.summary", count=len(project.scenes),
                  duration=f"{total_duration:.1f}"))

    _render_credits(project)

    # ---- BGM (reused component) ----
    bgm_params = render_bgm_section(key_prefix="flxw_final_")

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
        if st.button(tr("sbs.nav.back"), key="flxw_final_back", use_container_width=True):
            _goto(SCENES_STEP)
    with c_new:
        if st.button(tr("flxw.final.new_project"), key="flxw_final_new", use_container_width=True):
            _reset_wizard()
            st.rerun()


# ============================================================================
# Entry point
# ============================================================================

def render_flex_wizard(pixelle_video):
    """Render the flexible (generate-or-search) wizard (call after settings/header)."""
    engine = FlexibleVideoEngine(pixelle_video, load_flex_config())
    step = st.session_state.setdefault(K_STEP, 0)
    project = st.session_state.get(K_PROJECT)

    # Steps past Setup need a project
    if step > 0 and project is None:
        step = 0
        st.session_state[K_STEP] = 0

    # Checkpoint to disk every run: session state dies with the browser/tunnel
    # session, the task dir doesn't — Setup offers to resume from there.
    if project is not None:
        save_project(project, step=step)

    _render_stepper(step, project)

    if step == 0:
        _render_setup(engine)
    elif step == 1:
        _render_script(engine)
    elif step == PLAN_STEP:
        _render_plan(engine)
    elif step == SOURCE_STEP:
        _render_source(engine)
    elif step == SCENES_STEP:
        _render_scenes(engine)
    else:
        _render_final(engine)
