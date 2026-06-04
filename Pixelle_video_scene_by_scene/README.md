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

Edits automatically invalidate only the affected downstream assets of that
scene (e.g. changing a narration invalidates its audio + segment; changing a
prompt invalidates its media + segment).

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
