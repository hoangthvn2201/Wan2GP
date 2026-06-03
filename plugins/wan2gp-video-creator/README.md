# Video Creator plugin

End-to-end guided pipeline for WanGP:

**LLM script → start images → LTX-2 video → narration TTS → stitched master.**

## Enable

1. Launch WanGP, open the **Plugins** tab (Plugin Manager).
2. Enable **Video Creator** (it's bundled but off by default).
3. A new **Video Creator** tab appears.

## Workflow

1. **LLM endpoint** (collapsible): enter an OpenAI-compatible Base URL, API key, and
   model name. Works for self-hosted vLLM (`http://host:8000`) and hosted GPT
   (`https://api.openai.com`). Click **Test connection**, then **Save LLM settings**
   (persisted to `wgp_config.json` under `"video_creator"`).
2. **Stage A — Script:**
   - *LLM mode*: enter a brief + scene count → **Generate script**. The model returns,
     per scene: summary, image prompt, video prompt, narration text, voice hint.
   - *Direct import*: **Create blank scenes** and fill everything in by hand.
3. **Models:** pick the image model, the LTX-2 variant (**Distilled**/**Dev**, or an
   exact model under *Advanced*), the narration mode, and the TTS model.
4. **Scenes:** edit any per-scene field; toggle *Generate & use start image* off for
   direct text-to-video. Regenerate any single stage per scene.
5. **Run controls:** generate per stage, or **Run full pipeline** (images → videos →
   narration → auto-stitch). **Cancel** aborts the in-flight job and stops the loop.
6. **Final assembly:** the master video is produced automatically by *Run full
   pipeline*; the **Concatenate clips + mux narration** button re-runs assembly on
   demand. Output lands in `<save_path>/video_creator/`.

### Narration modes
- **Separate TTS (muxed)** *(default)*: LTX-2 produces silent video; the TTS model
  generates narration; ffmpeg muxes it onto each clip.
- **LTX-2 native audio**: LTX-2 generates audio inline; the TTS stage is skipped.

## Constraints / notes
- WanGP runs **one generation at a time**, so the pipeline is strictly sequential.
- Stage-major order (all images, then all videos, then all narration) minimises
  costly model reloads.
- Modality dropdowns are filtered via the live model registry (`get_model_def`),
  not by parsing `defaults/*.json`.
- Final assembly needs `ffmpeg` on PATH; without it, per-scene clips are still
  produced.

## Verifying the LLM stage with a mock server

No GPU needed. Run a tiny OpenAI-compatible mock, then point the plugin at it:

```bash
python plugins/wan2gp-video-creator/tests/mock_llm_server.py  # serves on :8000
```

In the plugin: Base URL `http://localhost:8000`, model `mock`, any key →
**Test connection** → **Generate script** → confirm N scenes render with all fields.

## Unit tests (pure logic, no GPU/Gradio)

```bash
python plugins/wan2gp-video-creator/tests/test_logic.py
```
