# Pixelle-Video · Flexible Video (generate-or-search)

Turns a **topic** into a narrated short video where an **LLM orchestrator decides
per scene** whether the visual is:

- 🎨 **AI-generated** — the existing WanGP / ComfyUI generation path, or
- 🔎 **Real stock media** — searched on **Pexels** and **Pixabay**, with an LLM
  ranking the candidates against the scene narration, a thumbnail gallery for
  manual override, and automatic fallback to generation when search comes up dry.

```
topic ─► script ─► media plan ─► source media ─► scenes ─► final video
         (LLM)     (LLM: per      (stock gallery   (TTS +     (concat +
                    scene gen      + AI pick +      segment     BGM +
                    vs search)     normalize)       render)     credits)
```

## Architecture

This app follows the same pattern as `../Pixelle_video_pdf`: it REUSES the
Pixelle-Video core (`pixelle_video`) and the scene-by-scene engine (`sbs`)
**unchanged**, adding only:

- `flexvid/` — the flexible pipeline package
  - `engine.py` — `FlexibleVideoEngine(SceneBySceneEngine)`: scene plan,
    stock search, candidate ranking, download + normalization, fallback
  - `models.py` — `FlexScene(Scene)` (per-scene sourcing plan + candidates),
    `MediaCandidate` (provider-normalized search result)
  - `search/` — `PexelsProvider`, `PixabayProvider`, fan-out aggregator
  - `prompts.py` — scene-plan + candidate-ranking LLM prompts
  - `normalize.py` — ffmpeg re-encode of stock media to the project
    size/fps (cover-crop, yuv420p, silent) so the final concat (`-c copy`
    demuxer) stays safe
  - `flex_config.py` — module-local config loader
- `web/` — Streamlit wizard (6 steps, session prefix `flxw_`)

Stock media enters the exact same per-scene asset contract the base engine
uses (`frames/<uid>_image.png` / `<uid>_video.mp4`), so audio, segment
rendering, composition, History and persistence are all inherited.

## Setup

1. Configure the shared core as usual (`../Pixelle_video/config.yaml`:
   LLM + workflows + TTS).
2. Add free stock API keys (either place):

   ```yaml
   # Pixelle_video_flexible/flex_config.yaml
   media_search:
     pexels:  {enabled: true, api_key: "YOUR_PEXELS_KEY"}    # pexels.com/api
     pixabay: {enabled: true, api_key: "YOUR_PIXABAY_KEY"}   # pixabay.com/api/docs
     candidates_per_scene: 6
     min_resolution: 720
     allow_fallback: true     # empty search -> fall back to AI generation
     search_only: false       # true = stock-only, AI generation never used
   ```

   or environment variables: `PEXELS_API_KEY` / `PIXABAY_API_KEY`.

   With **no keys configured the app still works** — every scene is simply
   AI-generated.

   **Stock-only mode (no image/video generator):** set `search_only: true`
   (or use the 🔒 toggle on the wizard's Setup step). The media plan then
   forces every scene to search — abstract ideas are rephrased into concrete
   stock-findable imagery — and an empty search retries once with
   LLM-broadened keywords instead of falling back to generation. Since WanGP
   models load lazily on first generation, a stock-only video never downloads
   a model and doesn't need the GPU (TTS + ffmpeg + Chromium only).

3. Start (port **8504**; 8501 = original app, 8502 = scene-by-scene, 8503 = PDF):

   ```bash
   ./start_web.sh
   ```

No new Python dependencies: `httpx`, `pyyaml`, `ffmpeg-python` are already
core requirements; `ffmpeg` must be on PATH (already required by the core).

## Wizard steps

| Step | What happens |
|------|--------------|
| ① Setup | topic + scene count + language + voice/template/workflow + provider status |
| ② Script | editable narrations (AI rewrite / add / delete) |
| ③ Media Plan | per scene: LLM's generate-vs-search decision (+ its reasoning), editable search keywords / generation prompts, per-scene re-plan |
| ④ Source Media | search scenes: candidate gallery (AI pick highlighted, click to override) → download + normalize; fallback notice when search finds nothing |
| ⑤ Scenes | audio + media + segment per scene (search scenes auto-complete search→rank→download here too) |
| ⑥ Final | BGM, compose, preview/download + **stock credits** (photographer / source / license per scene) |

## API quotas & caching

| Provider | Free quota | On 429 |
|----------|-----------|--------|
| Pexels   | 200 req/hour, 20,000 req/month (unlimited on request with visible attribution) | retried once honoring `Retry-After` (capped 30s) |
| Pixabay  | 100 req/60s | retried once honoring `Retry-After` (capped 30s) |

Search responses are **cached on disk for 24 hours** (`.search_cache/`,
git-ignored) — required by Pixabay's API terms and quota-friendly for Pexels.
The TTL matches Pixabay's URL lifetime, so cached thumbnails stay valid as
long as live ones. Picked media is always downloaded into the task directory
immediately (no permanent hotlinking, also a Pixabay requirement). Set
`FLEXVID_SEARCH_CACHE=0` to disable the cache. If one provider is rate-limited
or down, the aggregator logs it and serves results from the other.

## Licensing / attribution

Pexels and Pixabay content is free to use. The picked candidate's
photographer, page URL and license are stored on each scene, persisted in the
task metadata (`input.scene_sources`) and shown in a credits expander on the
Final step.
