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
PDF → Video Wizard UI

A 6-step wizard on top of `pdfv.PdfVideoEngine`:

    ① Setup    – upload a PDF + page range / focus / style (TTS, template, ...)
    ② Digest   – review / edit the AI's understanding of the document
                 (core message, grounded key insights, visual world, tone)
    ③ Script   – review / edit / AI-rewrite / add / delete narrations
    ④ Prompts  – review / edit / regenerate per-scene media prompts
                 (all living in the digest's shared visual world)
    ⑤ Scenes   – generate & preview audio + image/video + segment per scene
    ⑥ Final    – pick BGM, compose the final video, preview & download

Steps ③–⑥ mirror the scene-by-scene wizard (the engine inherits those steps
from `sbs.SceneBySceneEngine` unchanged); steps ① and ② are PDF-specific.
All intermediate state lives in `st.session_state` (prefix `pdfw_`), so the
user can freely move back and forth; edits invalidate only the affected
downstream assets of that scene.
"""

import os
import re
from pathlib import Path

import streamlit as st
from loguru import logger

from web.i18n import tr
from web.utils.async_helpers import run_async
from web.components.content_input import render_bgm_section
from web.components.style_config import render_style_config

from pixelle_video.config import config_manager

from sbs import Scene
from pdfv import PdfVideoEngine

# Session-state keys
K_STEP = "pdfw_step"
K_PROJECT = "pdfw_project"
K_RESULT = "pdfw_result"
K_DOC = "pdfw_doc"              # PdfDocument
K_DIGEST = "pdfw_digest"        # DocumentDigest
K_SETUP = "pdfw_setup_params"   # frozen setup params (style, focus, language, ...)
K_PDF_PATH = "pdfw_pdf_path"    # saved upload path

STEPS = ["setup", "digest", "script", "prompts", "scenes", "final"]
STEP_ICONS = ["📄", "🧠", "📝", "🎨", "🎬", "🏁"]
PROMPTS_STEP = 3                # index of the (skippable) prompts step


# ============================================================================
# Widget-state helpers
#
# Model state (the digest / project / scenes) is the single source of truth;
# widgets are bound to it through _bind_text below, which detects whether the
# MODEL or the WIDGET moved since the last render — so a stale widget copy
# can never silently overwrite a model value changed elsewhere.
# ============================================================================

def _clear_digest_widget_state():
    """Drop digest-editing widget state (used on re-digest / new document)."""
    for key in list(st.session_state.keys()):
        if key.startswith(("pdfw_dg_", "pdfw_dgi_")):
            del st.session_state[key]


def _clear_project_widget_state():
    """Drop per-scene widget state (and binder companions) from a previous project."""
    for key in list(st.session_state.keys()):
        if key.startswith(("pdfw_narr_", "pdfw_prompt_", "pdfw_scene_narr_",
                           "pdfw_scene_prompt_", "pdfw_title_w")):
            del st.session_state[key]
    st.session_state.pop(K_RESULT, None)


def _reset_wizard():
    """Start a brand-new document (keeps the setup form values)."""
    _clear_project_widget_state()
    _clear_digest_widget_state()
    for key in (K_PROJECT, K_DOC, K_DIGEST, K_SETUP):
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


# ============================================================================
# Stepper
# ============================================================================

def _render_stepper(step: int, project):
    """Clickable breadcrumb of the wizard steps."""
    cols = st.columns(len(STEPS))
    for i, (name, icon) in enumerate(zip(STEPS, STEP_ICONS)):
        label = f"{icon} {tr(f'pdfw.stepper.{name}')}"
        skipped = (i == PROMPTS_STEP and project is not None and not project.needs_media)
        # Digest is revisitable as soon as it exists; later steps need a project
        revisitable = (
            i < step
            and not skipped
            and (project is not None or (i <= 1 and st.session_state.get(K_DIGEST) is not None))
        )
        with cols[i]:
            if i == step:
                st.button(label, key=f"pdfw_step_btn_{i}", type="primary",
                          use_container_width=True, disabled=True)
            elif revisitable:
                # Completed steps are clickable to go back
                if st.button(label, key=f"pdfw_step_btn_{i}", use_container_width=True):
                    _goto(i)
            else:
                help_text = tr("sbs.stepper.skipped_static") if skipped else None
                st.button(label, key=f"pdfw_step_btn_{i}", use_container_width=True,
                          disabled=True, help=help_text)
    st.markdown("")


# ============================================================================
# Step ① Setup
# ============================================================================

def _render_video_mode_section(engine: PdfVideoEngine, style_params: dict) -> dict:
    """
    Video-generation mode selector — only for video templates.

    Same behavior as the scene-by-scene wizard: t2v by default, i2v offered
    when at least one i2v-capable workflow (workflows/<source>/i2v_*.json)
    is installed.
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
            key="pdfw_setup_video_mode",
        )
        if mode != "i2v":
            return {"media_mode": "t2v"}

        st.caption(tr("sbs.setup.i2v_hint"))

        i2v_workflow = st.selectbox(
            tr("sbs.setup.i2v_workflow_label"),
            options=i2v_workflows,
            key="pdfw_setup_i2v_workflow",
        )

        image_workflows = engine.list_image_workflows()
        default_label = tr("sbs.setup.i2v_image_workflow_default")
        image_choice = st.selectbox(
            tr("sbs.setup.i2v_image_workflow_label"),
            options=[default_label] + image_workflows,
            key="pdfw_setup_i2v_image_workflow",
        )

        return {
            "media_mode": "i2v",
            "i2v_workflow": i2v_workflow,
            "i2v_image_workflow": None if image_choice == default_label else image_choice,
        }


def _parse_page_range(text: str):
    """'3-12' / '5' -> (first, last) tuple, or None for the whole document."""
    text = (text or "").strip()
    if not text:
        return None
    match = re.fullmatch(r"(\d+)\s*[-–~]\s*(\d+)", text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    if text.isdigit():
        return (int(text), int(text))
    raise ValueError(tr("pdfw.setup.page_range_invalid", value=text))


def _save_uploaded_pdf(uploaded) -> str:
    """Persist the uploaded PDF under output/uploads/ (cwd = Pixelle_video)."""
    uploads_dir = Path("output") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w.\-]+", "_", uploaded.name) or "document.pdf"
    path = uploads_dir / safe_name
    data = uploaded.getvalue()
    # Skip the rewrite when the same file is already on disk (every rerun
    # passes through here while the uploader holds a file)
    if not (path.exists() and path.stat().st_size == len(data)):
        path.write_bytes(data)
        logger.info(f"📥 Saved uploaded PDF: {path} ({len(data) / 1e6:.1f} MB)")
    return str(path)


_LANGUAGE_CHOICES = [
    "auto", "English", "Vietnamese", "Chinese", "Japanese",
    "Korean", "French", "German", "Spanish",
]


def _render_setup(engine: PdfVideoEngine):
    pixelle_video = engine.core
    project = st.session_state.get(K_PROJECT)

    left_col, right_col = st.columns([1, 1])

    # ---- Document input ----
    with left_col:
        with st.container(border=True):
            st.markdown(f"**{tr('pdfw.setup.document_label')}**")

            uploaded = st.file_uploader(
                tr("pdfw.setup.upload_label"),
                type=["pdf"],
                key="pdfw_setup_upload",
                help=tr("pdfw.setup.upload_help"),
            )
            if uploaded is not None:
                st.session_state[K_PDF_PATH] = _save_uploaded_pdf(uploaded)
            pdf_path = st.session_state.get(K_PDF_PATH)
            if pdf_path and os.path.exists(pdf_path):
                size_mb = os.path.getsize(pdf_path) / 1e6
                st.caption(tr("pdfw.setup.uploaded_info",
                              name=os.path.basename(pdf_path), size=f"{size_mb:.1f}"))

            page_range_text = st.text_input(
                tr("pdfw.setup.page_range_label"),
                placeholder=tr("pdfw.setup.page_range_placeholder"),
                help=tr("pdfw.setup.page_range_help"),
                key="pdfw_setup_page_range",
            )

            focus = st.text_input(
                tr("pdfw.setup.focus_label"),
                placeholder=tr("pdfw.setup.focus_placeholder"),
                help=tr("pdfw.setup.focus_help"),
                key="pdfw_setup_focus",
            )

            n_scenes = st.slider(
                tr("video.frames"),
                min_value=3, max_value=30, value=5,
                help=tr("video.frames_help"),
                key="pdfw_setup_n_scenes",
            )
            st.caption(tr("video.frames_label", n=n_scenes))

            language = st.selectbox(
                tr("pdfw.setup.language_label"),
                options=_LANGUAGE_CHOICES,
                format_func=lambda x: tr("pdfw.setup.language_auto") if x == "auto" else x,
                help=tr("pdfw.setup.language_help"),
                key="pdfw_setup_language",
            )

        # ---- Narration loudness (TTS output is often quiet; fixed post-TTS) ----
        with st.container(border=True):
            st.markdown(f"**{tr('pdfw.setup.audio_label')}**")
            tts_normalize = st.toggle(
                tr("pdfw.setup.tts_normalize_label"),
                value=True,
                help=tr("pdfw.setup.tts_normalize_help"),
                key="pdfw_setup_tts_normalize",
            )
            tts_volume = st.slider(
                tr("pdfw.setup.tts_volume_label"),
                min_value=0.5, max_value=3.0, value=1.0, step=0.1,
                help=tr("pdfw.setup.tts_volume_help"),
                key="pdfw_setup_tts_volume",
            )

    # ---- Style configuration (reused component: TTS / template / workflow) ----
    with right_col:
        style_params = render_style_config(pixelle_video)
        video_mode_params = _render_video_mode_section(engine, style_params)

    st.caption(tr("pdfw.setup.params_frozen_hint"))

    # ---- Actions ----
    c1, c2 = st.columns([2, 1]) if project is not None else (st.container(), None)
    with c1:
        if st.button(tr("pdfw.setup.ingest_btn"), type="primary", use_container_width=True):
            if not config_manager.validate():
                st.error(tr("settings.not_configured"))
                st.stop()
            pdf_path = st.session_state.get(K_PDF_PATH)
            if not pdf_path or not os.path.exists(pdf_path):
                st.error(tr("pdfw.setup.no_pdf_warning"))
                st.stop()

            try:
                page_range = _parse_page_range(page_range_text)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            try:
                with st.spinner(tr("pdfw.setup.ingesting")):
                    doc = engine.ingest_pdf(pdf_path, page_range=page_range)
                if doc.n_chars < 200:
                    st.error(tr("pdfw.setup.too_little_text"))
                    st.stop()

                status = st.status(tr("pdfw.setup.digesting"), expanded=True)

                def on_progress(done, total, message):
                    status.write(f"[{done}/{total}] {message}")

                digest = run_async(engine.digest_document(
                    doc, focus=focus.strip() or None, progress_callback=on_progress,
                ))
                status.update(label=tr("pdfw.setup.digest_done"), state="complete",
                              expanded=False)

                # Freeze the setup so later steps (script / project) reuse it
                st.session_state[K_SETUP] = {
                    "n_scenes": n_scenes,
                    "focus": focus.strip() or None,
                    "language": None if language == "auto" else language,
                    "style_params": style_params,
                    "video_mode_params": video_mode_params,
                    "pdf_path": pdf_path,
                    "page_range": page_range,
                    "tts_volume": tts_volume,
                    "tts_normalize": tts_normalize,
                }
                st.session_state[K_DOC] = doc
                st.session_state[K_DIGEST] = digest
                # A new document discards the previous project + digest edits
                _clear_project_widget_state()
                _clear_digest_widget_state()
                st.session_state.pop(K_PROJECT, None)
                _goto(1)
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))

    if project is not None and c2 is not None:
        with c2:
            if st.button(tr("sbs.setup.resume_btn"), use_container_width=True):
                _goto(2)
        st.warning(tr("pdfw.setup.new_project_warning"))


# ============================================================================
# Step ② Digest
# ============================================================================

def _bind_text(key: str, model_value: str, *, height: int = 80, label: str = "text",
               input_widget: bool = False, label_visibility: str = "collapsed"):
    """
    Text widget two-way bound to a model value, immune to stale widget state.

    A plain `if widget != model: model = widget` treats ANY mismatch as a user
    edit — but a mismatch can equally mean the MODEL was changed elsewhere
    (an edit on another step, AI rewrite, regenerated prompt, re-digest)
    while this widget kept a stale copy; writing that stale copy back
    silently REVERTS the change (lost-script bug). A companion key remembers
    the model value this widget was last synced to, so the two directions can
    be told apart: model moved -> push the model into the widget (never apply
    the stale copy); widget moved -> return the edit for the caller to apply.
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
    True only on a REAL content change (same rationale as the scene-by-scene
    wizard: the browser may normalize line endings / edge whitespace on any
    button click, which must not count as an edit).
    """
    norm = lambda s: (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return norm(a) != norm(b)


def _render_digest(engine: PdfVideoEngine):
    doc = st.session_state.get(K_DOC)
    digest = st.session_state.get(K_DIGEST)
    setup = st.session_state.get(K_SETUP) or {}
    if doc is None or digest is None:
        _goto(0)

    st.markdown(f"#### {tr('pdfw.digest.heading')}")
    st.caption(tr("pdfw.digest.hint"))

    # ---- Document facts ----
    first, last = doc.page_range
    st.info(tr(
        "pdfw.digest.doc_info",
        name=os.path.basename(doc.path),
        first=first, last=last, total=doc.n_pages_total,
        chars=f"{doc.n_chars:,}",
        doc_type=digest.doc_type, language=digest.language,
    ))

    # ---- Editable digest fields ----
    new_title = _bind_text("pdfw_dg_title", digest.title,
                           label=tr("pdfw.digest.title_label"),
                           input_widget=True, label_visibility="visible")
    if new_title.strip() and _texts_differ(new_title, digest.title):
        digest.title = new_title.strip()

    st.markdown(f"**{tr('pdfw.digest.core_label')}**")
    core = _bind_text("pdfw_dg_core", digest.core_message, height=80)
    if _texts_differ(core, digest.core_message):
        digest.core_message = core.strip()

    c_world, c_tone = st.columns([2, 1])
    with c_world:
        st.markdown(f"**{tr('pdfw.digest.visual_world_label')}**")
        st.caption(tr("pdfw.digest.visual_world_help"))
        world = _bind_text("pdfw_dg_world", digest.visual_world, height=80)
        if _texts_differ(world, digest.visual_world):
            digest.visual_world = world.strip()
    with c_tone:
        st.markdown(f"**{tr('pdfw.digest.tone_label')}**")
        tone = _bind_text("pdfw_dg_tone", digest.tone, label="tone", input_widget=True)
        if tone.strip() and _texts_differ(tone, digest.tone):
            digest.tone = tone.strip()

    # ---- Key insights (editable; grounding shown read-only) ----
    st.markdown(f"**{tr('pdfw.digest.insights_label')}**")
    st.caption(tr("pdfw.digest.insights_hint"))
    for i, ki in enumerate(list(digest.key_insights)):
        with st.container(border=True):
            text_col, btn_col = st.columns([6, 1])
            with text_col:
                key = f"pdfw_dgi_{i}"
                value = _bind_text(key, ki.insight, height=70)
                if _texts_differ(value, ki.insight):
                    ki.insight = value.strip()
                if ki.grounding:
                    st.caption(tr("pdfw.digest.grounding_caption", grounding=ki.grounding))
            with btn_col:
                if st.button("🗑️", key=f"pdfw_dgi_del_{i}",
                             disabled=len(digest.key_insights) <= 1,
                             help=tr("pdfw.digest.delete_insight")):
                    digest.key_insights.remove(ki)
                    _clear_digest_widget_state()
                    st.rerun()

    # ---- Hooks (informational) ----
    if digest.hook_ideas:
        with st.expander(tr("pdfw.digest.hooks_label"), expanded=False):
            for hook in digest.hook_ideas:
                st.markdown(f"- {hook}")

    st.markdown("---")

    # ---- Navigation ----
    c_back, c_regen, c_next = st.columns([1, 1, 2])
    with c_back:
        if st.button(tr("sbs.nav.back"), key="pdfw_digest_back", use_container_width=True):
            _goto(0)
    with c_regen:
        if st.button(tr("pdfw.digest.redigest_btn"), key="pdfw_digest_regen",
                     use_container_width=True):
            try:
                status = st.status(tr("pdfw.setup.digesting"), expanded=True)

                def on_progress(done, total, message):
                    status.write(f"[{done}/{total}] {message}")

                new_digest = run_async(engine.digest_document(
                    doc, focus=setup.get("focus"), progress_callback=on_progress,
                ))
                status.update(label=tr("pdfw.setup.digest_done"), state="complete",
                              expanded=False)
                st.session_state[K_DIGEST] = new_digest
                _clear_digest_widget_state()
                st.rerun()
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))
    with c_next:
        if st.button(tr("pdfw.digest.continue"), key="pdfw_digest_next",
                     type="primary", use_container_width=True):
            if not any(ki.insight.strip() for ki in digest.key_insights):
                st.error(tr("pdfw.digest.no_insights_warning"))
                st.stop()
            try:
                with st.spinner(tr("pdfw.digest.scripting")):
                    title, narrations = run_async(engine.generate_pdf_script(
                        digest,
                        n_scenes=setup.get("n_scenes", 5),
                        focus=setup.get("focus"),
                        language=setup.get("language"),
                    ))
                params = {
                    "text": f"PDF: {digest.title}",
                    "mode": "pdf",
                    "n_scenes": setup.get("n_scenes", 5),
                    "split_mode": "paragraph",
                    "title": title,
                    **(setup.get("style_params") or {}),
                    **(setup.get("video_mode_params") or {}),
                    # --- Narration loudness (applied post-TTS by the engine) ---
                    "tts_volume": setup.get("tts_volume", 1.0),
                    "tts_normalize": setup.get("tts_normalize", True),
                    # --- PDF provenance (persisted with the task) ---
                    "pdf_source": doc.path,
                    "pdf_pages": list(doc.page_range),
                    "pdf_focus": setup.get("focus"),
                    "pdf_language": setup.get("language") or digest.language,
                    "pdf_visual_world": digest.visual_world,
                }
                _clear_project_widget_state()
                st.session_state[K_PROJECT] = engine.create_project(
                    title, narrations, [None] * len(narrations), params
                )
                _goto(2)
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))

    if st.session_state.get(K_PROJECT) is not None:
        st.warning(tr("pdfw.digest.new_script_warning"))


# ============================================================================
# Step ③ Script
# ============================================================================

def _apply_narration_change(project, scene, value: str):
    """
    Narration changed -> audio is stale; video clips are length-synced to the
    audio, so they go stale too (i2v keeps its still-valid start image).
    """
    scene.narration = value
    scene.invalidate_audio()
    if project.is_i2v:
        scene.invalidate_video()
    elif project.is_video_workflow:
        scene.invalidate_media()


def _render_script(engine: PdfVideoEngine):
    project = st.session_state.get(K_PROJECT)
    digest = st.session_state.get(K_DIGEST)
    setup = st.session_state.get(K_SETUP) or {}
    if project is None:
        _goto(1 if digest is not None else 0)

    st.markdown(f"#### {tr('sbs.script.heading')}")
    st.caption(tr("pdfw.script.hint"))

    # ---- Title ----
    new_title = _bind_text("pdfw_title_w", project.title,
                           label=tr("sbs.script.title_label"),
                           input_widget=True, label_visibility="visible")
    if new_title.strip() and _texts_differ(new_title, project.title):
        project.title = new_title.strip()

    # ---- Scenes ----
    for i, scene in enumerate(list(project.scenes)):
        with st.container(border=True):
            st.markdown(f"**{tr('sbs.script.scene_label', n=i + 1)}**")

            key = f"pdfw_narr_{scene.uid}"
            value = _bind_text(key, scene.narration, label="narration")
            if _texts_differ(value, scene.narration):
                _apply_narration_change(project, scene, value)

            c1, c2, _sp = st.columns([1, 1, 2])
            with c1:
                if st.button(tr("sbs.script.rewrite"), key=f"pdfw_rw_{scene.uid}",
                             use_container_width=True):
                    try:
                        with st.spinner(tr("sbs.script.rewriting")):
                            rewritten = run_async(engine.rewrite_narration(
                                scene.narration,
                                topic=(digest.core_message if digest else None),
                            ))
                        # The binders push the new model value into every
                        # widget (this one included) on the next run.
                        _apply_narration_change(project, scene, rewritten)
                        st.rerun()
                    except Exception as e:
                        logger.exception(e)
                        st.error(tr("sbs.common.error", error=str(e)))
            with c2:
                if st.button(tr("sbs.script.delete"), key=f"pdfw_del_{scene.uid}",
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
        if st.button(tr("sbs.nav.back"), key="pdfw_script_back", use_container_width=True):
            _goto(1)
    with c_regen:
        if st.button(tr("sbs.script.regen_all"), key="pdfw_script_regen",
                     use_container_width=True, disabled=digest is None):
            try:
                with st.spinner(tr("pdfw.digest.scripting")):
                    title, narrations = run_async(engine.generate_pdf_script(
                        digest,
                        n_scenes=project.params.get("n_scenes", len(project.scenes)),
                        focus=setup.get("focus"),
                        language=setup.get("language"),
                    ))
                _clear_project_widget_state()
                project.title = title
                project.scenes = [Scene(narration=n) for n in narrations]
                st.rerun()
            except Exception as e:
                logger.exception(e)
                st.error(tr("sbs.common.error", error=str(e)))
    with c_next:
        if st.button(tr("sbs.script.continue"), key="pdfw_script_next",
                     type="primary", use_container_width=True):
            if any(not s.narration.strip() for s in project.scenes):
                st.error(tr("sbs.script.empty_warning"))
            else:
                _goto(PROMPTS_STEP if project.needs_media else 4)


# ============================================================================
# Step ④ Prompts
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
        key="pdfw_use_prefix",
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


def _render_prompts(engine: PdfVideoEngine):
    project = st.session_state.get(K_PROJECT)
    digest = st.session_state.get(K_DIGEST)
    if project is None or digest is None:
        _goto(0)
    if not project.needs_media:
        _goto(4)

    st.markdown(f"#### {tr('sbs.prompts.heading')}")
    st.caption(tr("pdfw.prompts.hint"))
    if digest.visual_world:
        st.caption(tr("pdfw.prompts.visual_world_caption", world=digest.visual_world))

    _render_prefix_toggle(project)
    prompt_prefix = _effective_prefix(project)

    # ---- Auto-generate missing prompts on entry (digest-aware) ----
    missing = [s for s in project.scenes if not s.prompt]
    if missing:
        try:
            with st.spinner(tr("sbs.prompts.generating", count=len(missing))):
                prompts = run_async(engine.generate_visual_prompts(
                    [s.narration for s in missing],
                    digest,
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

            key = f"pdfw_prompt_{scene.uid}"
            value = _bind_text(key, scene.prompt or "", height=100, label="prompt")
            if _texts_differ(value, scene.prompt):
                scene.prompt = value
                scene.invalidate_media()

            if st.button(tr("sbs.prompts.regen_one"), key=f"pdfw_rp_{scene.uid}"):
                try:
                    with st.spinner(tr("sbs.prompts.regenerating")):
                        new_prompt = run_async(engine.generate_visual_prompt_for(
                            scene.narration, digest, prompt_prefix=prompt_prefix
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
        if st.button(tr("sbs.nav.back"), key="pdfw_prompts_back", use_container_width=True):
            _goto(2)
    with c_regen:
        if st.button(tr("sbs.prompts.regen_all"), key="pdfw_prompts_regen",
                     use_container_width=True):
            try:
                with st.spinner(tr("sbs.prompts.generating", count=len(project.scenes))):
                    prompts = run_async(engine.generate_visual_prompts(
                        [s.narration for s in project.scenes],
                        digest,
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
        if st.button(tr("sbs.prompts.continue"), key="pdfw_prompts_next",
                     type="primary", use_container_width=True):
            if any(not (s.prompt or "").strip() for s in project.scenes):
                st.error(tr("sbs.prompts.empty_warning"))
            else:
                _goto(4)


# ============================================================================
# Step ⑤ Scenes  (identical behavior to the scene-by-scene wizard)
# ============================================================================

def _scene_status_icons(project, scene) -> str:
    parts = [f"🎤{'✅' if scene.audio_path else '⬜'}"]
    if project.is_i2v:
        # image → video: the still and the animated clip are separate steps
        parts.append(f"🖼️{'✅' if scene.image_path else '⬜'}")
        parts.append(f"🎬{'✅' if scene.video_path else '⬜'}")
    elif project.needs_media:
        media_icon = "🎬" if project.is_video_workflow else "🖼️"
        parts.append(f"{media_icon}{'✅' if scene.has_media else '⬜'}")
    parts.append(f"🎞️{'✅' if scene.segment_path else '⬜'}")
    return " ".join(parts)


def _render_scene_card(engine: PdfVideoEngine, project, scene, index: int):
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
            nkey = f"pdfw_scene_narr_{scene.uid}"
            nval = _bind_text(nkey, scene.narration, label="narration")
            if _texts_differ(nval, scene.narration):
                _apply_narration_change(project, scene, nval)

            if project.needs_media:
                st.caption(tr("sbs.scenes.prompt_label"))
                pkey = f"pdfw_scene_prompt_{scene.uid}"
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
                if st.button(audio_label, key=f"pdfw_a_{scene.uid}", use_container_width=True,
                             disabled=not scene.narration.strip()):
                    _run_step("sbs.scenes.stage_audio",
                              lambda: engine.generate_audio(project, scene, index))

            if project.is_i2v:
                # i2v: strict order — audio → start image → animate.
                with cols[1]:
                    img_label = (tr("sbs.scenes.image_start_regen") if scene.image_path
                                 else tr("sbs.scenes.image_start_btn"))
                    img_needs_audio = not scene.audio_path
                    img_disabled = img_needs_audio or not (scene.prompt or "").strip()
                    if st.button(img_label, key=f"pdfw_m_{scene.uid}", use_container_width=True,
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
                    if st.button(anim_label, key=f"pdfw_v_{scene.uid}", use_container_width=True,
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
                    if st.button(media_label, key=f"pdfw_m_{scene.uid}", use_container_width=True,
                                 disabled=disabled, help=help_text):
                        _run_step("sbs.scenes.stage_media",
                                  lambda: engine.generate_media(project, scene, index))

            with cols[-1]:
                seg_label = tr("sbs.scenes.segment_regen") if scene.segment_path else tr("sbs.scenes.segment_btn")
                seg_ready = scene.audio_path and project.scene_media_ready(scene)
                if st.button(seg_label, key=f"pdfw_s_{scene.uid}", use_container_width=True,
                             disabled=not seg_ready,
                             help=None if seg_ready else tr("sbs.scenes.segment_requirements")):
                    _run_step("sbs.scenes.stage_segment",
                              lambda: engine.render_segment(project, scene, index))

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


def _render_scenes(engine: PdfVideoEngine):
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
        if st.button(tr("sbs.nav.back"), key="pdfw_scenes_back", use_container_width=True):
            _goto(PROMPTS_STEP if project.needs_media else 2)
    with c_next:
        all_ready = project.all_segments_ready
        if st.button(tr("sbs.scenes.continue"), key="pdfw_scenes_next", type="primary",
                     use_container_width=True, disabled=not all_ready,
                     help=None if all_ready else tr("sbs.scenes.not_all_ready")):
            _goto(5)


# ============================================================================
# Step ⑥ Final  (identical behavior to the scene-by-scene wizard)
# ============================================================================

def _render_final(engine: PdfVideoEngine):
    project = st.session_state.get(K_PROJECT)
    if project is None:
        _goto(0)

    st.markdown(f"#### {tr('sbs.final.heading')}")

    if not project.all_segments_ready:
        st.warning(tr("sbs.final.not_ready"))
        if st.button(tr("sbs.nav.back"), use_container_width=True):
            _goto(4)
        return

    total_duration = sum(s.duration for s in project.scenes)
    st.caption(tr("sbs.final.summary", count=len(project.scenes),
                  duration=f"{total_duration:.1f}"))

    # ---- BGM (reused component) ----
    bgm_params = render_bgm_section(key_prefix="pdfw_final_")

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
        if st.button(tr("sbs.nav.back"), key="pdfw_final_back", use_container_width=True):
            _goto(4)
    with c_new:
        if st.button(tr("pdfw.final.new_document"), key="pdfw_final_new", use_container_width=True):
            _reset_wizard()
            st.rerun()


# ============================================================================
# Entry point
# ============================================================================

def render_pdf_wizard(pixelle_video):
    """Render the PDF → video wizard (call after settings/header)."""
    engine = PdfVideoEngine(pixelle_video)
    step = st.session_state.setdefault(K_STEP, 0)
    project = st.session_state.get(K_PROJECT)
    digest = st.session_state.get(K_DIGEST)

    # Required state per step: digest for ②, a project for ③+
    if step == 1 and digest is None:
        step = 0
        st.session_state[K_STEP] = 0
    elif step > 1 and project is None:
        step = 1 if digest is not None else 0
        st.session_state[K_STEP] = step

    _render_stepper(step, project)

    if step == 0:
        _render_setup(engine)
    elif step == 1:
        _render_digest(engine)
    elif step == 2:
        _render_script(engine)
    elif step == PROMPTS_STEP:
        _render_prompts(engine)
    elif step == 4:
        _render_scenes(engine)
    else:
        _render_final(engine)
