# 🎞️ Pixelle-Video · Scene by Scene

A **step-by-step (scene-by-scene)** variant of the [Pixelle-Video](../Pixelle_video)
pipeline. Instead of one fully-automatic "topic → final video" run, the user
reviews and controls the input/output of **every stage and every scene**:

```
① Setup     topic / fixed script + voice / template / workflow settings
② Script    review · edit · ✨AI-rewrite · add / delete scene narrations
③ Prompts   review · edit · 🎲regenerate the per-scene media prompts
④ Scenes    per scene: 🎤 audio → 🖼️/🎬 image-or-video → 🎞️ segment
            (preview everything, regenerate any piece, edit text inline)
⑤ Final     pick BGM → compose → preview → ⬇️ download
```

For video templates there are two generation modes (chosen in ① Setup):

- **📝 Text → Video (t2v)** — each clip is generated directly from the prompt
  with the video workflow selected in the visual settings.
- **🖼️ Image → Video (i2v)** — a still image is generated first (image
  workflow), previewed/regenerable per scene, then **animated from that start
  frame** by an i2v-capable model, with the clip length synced to the
  narration audio. Per scene: 🎤 audio → 🖼️ image → 🎬 animate → 🎞️ segment.
  Only workflows named `i2v_*.json` accept a start image (e.g.
  `wan2gp/i2v_wan2.2.json`, `runninghub/i2v_LTX2.json`); the mode is offered
  only when at least one is installed.

Edits automatically invalidate only the affected downstream assets of that
scene (e.g. changing a narration invalidates its audio + segment — and any
length-synced video clip; changing a prompt invalidates its media + segment;
in i2v mode regenerating the start image invalidates the animation but
changing only the narration keeps the still and re-animates).

## Relationship with `../Pixelle_video`

- The **core is reused, unchanged**: `pixelle_video` (LLM / TTS / media /
  video / HTML frame rendering / persistence) is imported from
  `../Pixelle_video` — nothing there is modified.
- **Data is shared** with the original app: `config.yaml`, `workflows/`,
  `templates/`, `bgm/` and `output/` all resolve to `../Pixelle_video`
  (via `PIXELLE_VIDEO_ROOT`). No re-configuration needed, and finished
  videos appear in both apps' History pages.
- Only this folder contains the new logic:
  - `sbs/` — the step-wise engine (`SceneBySceneEngine`), modeled on
    `StandardPipeline` + `FrameProcessor` but exposed as individual steps
    with per-scene `uid`-based asset paths (safe add/delete/regenerate).
  - `web/` — the Streamlit wizard UI (some components are copies of the
    original ones: settings, style config, header, i18n, History page).

## Running

Same environment as the original app (Wan2GP conda env, see
[`../Pixelle_video/WAN2GP_BACKEND.md`](../Pixelle_video/WAN2GP_BACKEND.md)):

```bash
conda activate wan2gp
cd Pixelle_video_scene_by_scene
./start_web.sh            # serves on port 8502 (original app uses 8501)
```

Both apps can run at the same time — they are separate Streamlit processes.

## Notes

- The in-process **Wan2GP backend** (`wan2gp/*` workflows) works here too:
  the model stays loaded in VRAM between scene generations.
- Per-scene asset files live in `output/<task_id>/frames/<scene_uid>_*.{mp3,png,mp4}`.
- Voice / template / workflow settings are captured into the project when
  the script is generated; generate a new script to change them.
