"""WanGP "Video Creator" plugin.

A guided, end-to-end pipeline: LLM script -> start images -> LTX-2 video ->
narration TTS -> stitched master video. Generation is driven through the WanGP
API session (one task at a time, sequentially).
"""

import json
import os
import time

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import assembly, llm_client, model_registry, orchestrator, scene_model

PlugIn_Name = "Video Creator"
PlugIn_Id = "VideoCreator"

_STATUS_ICON = {
    "pending": "⚪",
    "running": "🟡",
    "done": "🟢",
    "error": "🔴",
    "skipped": "⚫",
}


def _badges(scene: dict) -> str:
    st = scene.get("status", {})
    parts = [f"{_STATUS_ICON.get(st.get(s, 'pending'), '⚪')} {s}" for s in scene_model.STAGES]
    line = "  ".join(parts)
    if scene.get("error"):
        line += f"  —  {scene['error'][:120]}"
    return line


class VideoCreatorPlugin(WAN2GPPlugin):
    def setup_ui(self):
        # globals from wgp.py
        self.request_global("get_model_def")
        self.request_global("get_base_model_type")
        self.request_global("get_model_name")
        self.request_global("displayed_model_types")
        self.request_global("server_config")
        self.request_global("server_config_filename")
        self.request_global("save_path")
        # components
        self.request_component("state")
        self.request_component("main_tabs")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)

    # ----- config helpers -------------------------------------------------
    def _vc_config(self) -> dict:
        cfg = {}
        try:
            cfg = (self.server_config or {}).get("video_creator", {}) or {}
        except Exception:
            cfg = {}
        return {
            "llm_base_url": cfg.get("llm_base_url", ""),
            "llm_api_key": cfg.get("llm_api_key", ""),
            "llm_model": cfg.get("llm_model", ""),
        }

    def _save_vc_config(self, base_url, api_key, model):
        try:
            if self.server_config is None:
                return "Config unavailable; not saved."
            self.server_config["video_creator"] = {
                "llm_base_url": (base_url or "").strip(),
                "llm_api_key": (api_key or "").strip(),
                "llm_model": (model or "").strip(),
            }
            filename = getattr(self, "server_config_filename", "") or ""
            if filename:
                with open(filename, "w", encoding="utf-8") as writer:
                    writer.write(json.dumps(self.server_config, indent=4))
                return "LLM settings saved."
            return "Saved in memory (no config file path)."
        except Exception as e:  # noqa: BLE001
            return f"Save failed: {e}"

    # ----- model dropdown choices ----------------------------------------
    def _modality_choices(self):
        dmt = getattr(self, "displayed_model_types", []) or []
        gmd = self.get_model_def
        gmn = getattr(self, "get_model_name", None)
        gbt = getattr(self, "get_base_model_type", None)
        images = model_registry.list_image_models(dmt, gmd, gmn)
        tts = model_registry.list_tts_models(dmt, gmd, gmn)
        ltx2 = model_registry.list_ltx2_video_models(dmt, gmd, gbt, gmn)
        return images, tts, ltx2

    def _out_dir(self) -> str:
        base = getattr(self, "save_path", None) or "outputs"
        out = os.path.join(base, "video_creator")
        os.makedirs(out, exist_ok=True)
        return out

    # ----- lifecycle ------------------------------------------------------
    def on_tab_select(self, state: dict):
        images, tts, ltx2 = self._modality_choices()
        vc = self._vc_config()
        return (
            gr.update(choices=images, value=(images[0][1] if images else None)),
            # Leave the exact-model dropdown empty so the Dev/Distilled radio drives
            # the choice unless the user explicitly overrides it here.
            gr.update(choices=ltx2, value=None),
            gr.update(choices=tts, value=(tts[0][1] if tts else None)),
            gr.update(value=vc["llm_base_url"]),
            gr.update(value=vc["llm_api_key"]),
            gr.update(value=vc["llm_model"]),
        )

    # ----- UI -------------------------------------------------------------
    def create_ui(self, api_session):
        active_job = {"job": None}
        cancel_flag = {"cancel": False}

        pipeline = gr.State(scene_model.empty_pipeline())

        with gr.Column():
            gr.Markdown(
                "## Video Creator\n"
                "Guided pipeline: **LLM script → start images → LTX-2 video → narration → master**."
            )

            # ---------------- LLM settings ----------------
            with gr.Accordion("LLM endpoint (OpenAI-compatible)", open=False):
                llm_base_url = gr.Textbox(label="Base URL", placeholder="http://localhost:8000  or  https://api.openai.com")
                llm_api_key = gr.Textbox(label="API key", type="password")
                llm_model = gr.Textbox(label="Model name", placeholder="Qwen2.5-7B-Instruct / gpt-4o-mini")
                with gr.Row():
                    test_btn = gr.Button("Test connection")
                    save_llm_btn = gr.Button("Save LLM settings")
                llm_status = gr.Markdown()

            # ---------------- Stage A: script ----------------
            with gr.Accordion("Stage A — Script", open=True):
                with gr.Tabs():
                    with gr.Tab("LLM mode"):
                        brief = gr.Textbox(label="Brief / topic", lines=4, placeholder="A 4-scene promo for an eco-friendly water bottle...")
                        num_scenes_llm = gr.Slider(1, 12, value=4, step=1, label="Number of scenes")
                        gen_script_btn = gr.Button("Generate script", variant="primary")
                    with gr.Tab("Direct import"):
                        num_scenes_manual = gr.Slider(1, 12, value=3, step=1, label="Number of scenes")
                        build_blank_btn = gr.Button("Create blank scenes")
                overall_script = gr.Textbox(label="Overall script", lines=4)

            # ---------------- Models ----------------
            with gr.Accordion("Models", open=True):
                image_model_dd = gr.Dropdown(label="Image model (Stage B — start frames)", choices=[], interactive=True)
                video_variant = gr.Radio(["Distilled", "Dev"], value="Distilled", label="LTX-2 variant (Stage C)")
                with gr.Accordion("Advanced: pick exact LTX-2 model", open=False):
                    video_model_dd = gr.Dropdown(label="LTX-2 model", choices=[], interactive=True)
                narration_mode = gr.Radio(
                    ["Separate TTS (muxed)", "LTX-2 native audio"],
                    value="Separate TTS (muxed)",
                    label="Narration mode",
                )
                tts_model_dd = gr.Dropdown(label="TTS / audio model (Stage D)", choices=[], interactive=True)
                with gr.Row():
                    resolution = gr.Textbox(label="Resolution", value=scene_model.DEFAULT_RESOLUTION)
                    video_length = gr.Number(label="Video length (frames)", value=scene_model.DEFAULT_VIDEO_LENGTH, precision=0)

            # ---------------- Per-scene editor ----------------
            gr.Markdown("### Scenes")

            @gr.render(inputs=pipeline)
            def render_scenes(pl):
                scenes = (pl or {}).get("scenes", [])
                if not scenes:
                    gr.Markdown("_No scenes yet. Generate a script or create blank scenes above._")
                    return
                for sc in scenes:
                    i = sc["index"]
                    with gr.Group():
                        gr.Markdown(f"**Scene {i + 1}** — {_badges(sc)}")
                        summ = gr.Textbox(value=sc["scene_summary"], label="Summary", lines=1)
                        imgp = gr.Textbox(value=sc["image_prompt"], label="Image prompt", lines=2)
                        vidp = gr.Textbox(value=sc["video_prompt"], label="Video prompt", lines=2)
                        narr = gr.Textbox(value=sc["narration_text"], label="Narration text", lines=2)
                        ttsp = gr.Textbox(value=sc["tts_prompt"], label="Voice hint", lines=1)
                        use_img = gr.Checkbox(value=sc.get("use_start_image", True), label="Generate & use start image")
                        with gr.Row():
                            img_prev = gr.Image(value=sc.get("image_path"), label="Start image", height=140)
                            vid_prev = gr.Video(value=sc.get("video_path"), label="Clip", height=180)
                            aud_prev = gr.Audio(value=sc.get("audio_path"), label="Narration")
                        with gr.Row():
                            rb_img = gr.Button("Regenerate image", size="sm")
                            rb_vid = gr.Button("Regenerate video", size="sm")
                            rb_tts = gr.Button("Regenerate narration", size="sm")

                        edit_inputs = [pipeline, summ, imgp, vidp, narr, ttsp, use_img]

                        def _persist(pl, summ_v, imgp_v, vidp_v, narr_v, ttsp_v, use_v, _i=i):
                            s = pl["scenes"][_i]
                            s["scene_summary"] = summ_v
                            s["image_prompt"] = imgp_v
                            s["video_prompt"] = vidp_v
                            s["narration_text"] = narr_v
                            s["tts_prompt"] = ttsp_v
                            s["use_start_image"] = bool(use_v)
                            return pl

                        # Persist edits on blur (avoids re-render mid-typing).
                        for comp in (summ, imgp, vidp, narr, ttsp):
                            comp.blur(_persist, inputs=edit_inputs, outputs=[pipeline])
                        use_img.change(_persist, inputs=edit_inputs, outputs=[pipeline])

                        def _make_regen(stage, _i=i):
                            def _regen(pl, summ_v, imgp_v, vidp_v, narr_v, ttsp_v, use_v,
                                       img_model, vid_var, vid_model_exact, narr_mode, tts_model, res, vlen,
                                       progress=gr.Progress(track_tqdm=False)):
                                pl = _persist(pl, summ_v, imgp_v, vidp_v, narr_v, ttsp_v, use_v, _i)
                                self._sync_run_config(pl, img_model, vid_var, vid_model_exact, narr_mode, tts_model, res, vlen)
                                err = scene_model.validate_pipeline_for_stage(pl, stage)
                                if err:
                                    raise gr.Error(err)
                                cancel_flag["cancel"] = False
                                orchestrator.run_stage_over_scenes(
                                    api_session, pl, stage, [_i], active_job,
                                    lambda r, d: progress(r, desc=d), cancel_flag,
                                )
                                return pl
                            return _regen

                        regen_inputs = edit_inputs + [
                            image_model_dd, video_variant, video_model_dd,
                            narration_mode, tts_model_dd, resolution, video_length,
                        ]
                        rb_img.click(_make_regen("image"), inputs=regen_inputs, outputs=[pipeline], queue=False)
                        rb_vid.click(_make_regen("video"), inputs=regen_inputs, outputs=[pipeline], queue=False)
                        rb_tts.click(_make_regen("tts"), inputs=regen_inputs, outputs=[pipeline], queue=False)

            # ---------------- Run controls ----------------
            with gr.Row():
                run_images_btn = gr.Button("Generate all images")
                run_videos_btn = gr.Button("Generate all videos")
                run_tts_btn = gr.Button("Generate all narration")
                run_all_btn = gr.Button("Run full pipeline", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="stop")
            progress_md = gr.Markdown()

            # ---------------- Final assembly ----------------
            with gr.Accordion("Final assembly", open=True):
                assemble_btn = gr.Button("Concatenate clips + mux narration")
                final_video = gr.Video(label="Master video")
                assemble_status = gr.Markdown()

        self.on_tab_outputs = [image_model_dd, video_model_dd, tts_model_dd, llm_base_url, llm_api_key, llm_model]

        run_config_inputs = [
            pipeline, image_model_dd, video_variant, video_model_dd,
            narration_mode, tts_model_dd, resolution, video_length,
        ]

        # ---- LLM events ----
        def _do_test(base_url, api_key, model):
            cfg = llm_client.LLMConfig(base_url, api_key, model)
            ok, msg = llm_client.test_connection(cfg)
            return f"{'✅' if ok else '❌'} {msg}"

        test_btn.click(_do_test, inputs=[llm_base_url, llm_api_key, llm_model], outputs=[llm_status], queue=False)
        save_llm_btn.click(self._save_vc_config, inputs=[llm_base_url, llm_api_key, llm_model], outputs=[llm_status], queue=False)

        # ---- Stage A events ----
        def _gen_script(base_url, api_key, model, brief_v, n, *run_cfg):
            cfg = llm_client.LLMConfig(base_url, api_key, model)
            if not cfg.base_url or not cfg.model:
                raise gr.Error("Set the LLM base URL and model name first.")
            try:
                script = llm_client.generate_script(cfg, brief_v, int(n))
            except Exception as e:  # noqa: BLE001
                raise gr.Error(f"Script generation failed: {e}")
            pl = scene_model.scenes_from_llm(script)
            self._sync_run_config(pl, *run_cfg[1:])  # run_cfg[0] would be old pipeline
            warn = script.get("_warning")
            status = "✅ Script generated." + (f"  ⚠️ {warn}" if warn else "")
            return pl, pl["overall_script"], status

        gen_script_btn.click(
            _gen_script,
            inputs=[llm_base_url, llm_api_key, llm_model, brief, num_scenes_llm] + run_config_inputs,
            outputs=[pipeline, overall_script, progress_md],
            queue=False,
        )

        def _build_blank(n, *run_cfg):
            pl = scene_model.empty_pipeline()
            pl["scenes"] = scene_model.build_scenes(int(n))
            pl["num_scenes"] = int(n)
            self._sync_run_config(pl, *run_cfg[1:])
            return pl, f"Created {int(n)} blank scene(s)."

        build_blank_btn.click(
            _build_blank,
            inputs=[num_scenes_manual] + run_config_inputs,
            outputs=[pipeline, progress_md],
            queue=False,
        )

        # ---- overall_script edit persists ----
        def _persist_overall(pl, text):
            pl["overall_script"] = text
            return pl
        overall_script.blur(_persist_overall, inputs=[pipeline, overall_script], outputs=[pipeline])

        # ---- run-all events ----
        def _make_run_stage(stage):
            def _run(pl, img_model, vid_var, vid_model_exact, narr_mode, tts_model, res, vlen,
                     progress=gr.Progress(track_tqdm=False)):
                self._sync_run_config(pl, img_model, vid_var, vid_model_exact, narr_mode, tts_model, res, vlen)
                err = scene_model.validate_pipeline_for_stage(pl, stage)
                if err:
                    raise gr.Error(err)
                cancel_flag["cancel"] = False
                all_idx = list(range(len(pl["scenes"])))
                orchestrator.run_stage_over_scenes(
                    api_session, pl, stage, all_idx, active_job,
                    lambda r, d: progress(r, desc=d), cancel_flag,
                )
                return pl, f"Stage '{stage}' complete."
            return _run

        run_images_btn.click(_make_run_stage("image"), inputs=run_config_inputs, outputs=[pipeline, progress_md], queue=False)
        run_videos_btn.click(_make_run_stage("video"), inputs=run_config_inputs, outputs=[pipeline, progress_md], queue=False)
        run_tts_btn.click(_make_run_stage("tts"), inputs=run_config_inputs, outputs=[pipeline, progress_md], queue=False)

        def _run_all(pl, img_model, vid_var, vid_model_exact, narr_mode, tts_model, res, vlen,
                     progress=gr.Progress(track_tqdm=False)):
            self._sync_run_config(pl, img_model, vid_var, vid_model_exact, narr_mode, tts_model, res, vlen)
            if not pl.get("scenes"):
                raise gr.Error("No scenes to process.")
            cancel_flag["cancel"] = False
            orchestrator.run_full_pipeline(
                api_session, pl, active_job, lambda r, d: progress(r, desc=d), cancel_flag,
            )
            master, warnings = assembly.assemble(pl, self._out_dir())
            msg = "✅ Pipeline complete." if master else "⚠️ Pipeline ran; assembly incomplete."
            if warnings:
                msg += "  " + " ".join(warnings)
            return pl, msg, master

        run_all_btn.click(
            _run_all,
            inputs=run_config_inputs,
            outputs=[pipeline, progress_md, final_video],
            queue=False,
        )

        def _cancel():
            cancel_flag["cancel"] = True
            job = active_job.get("job")
            if job is not None and not job.done:
                job.cancel()
            return "Cancellation requested."
        cancel_btn.click(_cancel, outputs=[progress_md], queue=False)

        # ---- assembly event ----
        def _assemble(pl):
            master, warnings = assembly.assemble(pl, self._out_dir())
            if master:
                return master, "✅ Master created. " + " ".join(warnings)
            return gr.update(), "❌ " + " ".join(warnings or ["Assembly failed."])
        assemble_btn.click(_assemble, inputs=[pipeline], outputs=[final_video, assemble_status], queue=False)

    # ----- shared: fold UI run-config into the pipeline state -------------
    def _sync_run_config(self, pl, img_model, vid_var, vid_model_exact, narr_mode, tts_model, res, vlen):
        models = pl.setdefault("models", {})
        models["image_model"] = img_model
        models["tts_model"] = tts_model
        # Exact LTX-2 model wins if chosen; otherwise map the Dev/Distilled radio.
        if vid_model_exact:
            models["video_model"] = vid_model_exact
        else:
            _, _, ltx2 = self._modality_choices()
            models["video_model"] = model_registry.resolve_variant_key(vid_var, ltx2)
        pl["narration_mode"] = "ltx2_native" if narr_mode == "LTX-2 native audio" else "tts"
        if res:
            pl["resolution"] = res
        try:
            if vlen:
                pl["video_length"] = int(vlen)
        except Exception:
            pass
        return pl
