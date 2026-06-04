# Wan2GP In-Process Backend

Pixelle-Video normally generates its images / videos by sending ComfyUI
workflows to a remote backend (a self-hosted ComfyUI server or the
RunningHub cloud). Since this copy of Pixelle-Video lives **inside the
[Wan2GP](https://github.com/deepbeepmeep/Wan2GP) repository**, a third
backend is available: load the image and video generation models
**directly in the Pixelle-Video process** through WanGP's official Python
API (`shared/api.py`, see `../docs/API.md`).

> This integration uses **WanGP by DeepBeepMeep**. Use of the WanGP API is
> subject to the WanGP Terms and Conditions.

## How it works

```
Pixelle pipeline (storyboard → frames)
   └── MediaService (pixelle_video/services/media.py)
         ├── source == "selfhost"   → ComfyKit → local ComfyUI server
         ├── source == "runninghub" → ComfyKit → RunningHub cloud
         └── source == "wan2gp"     → Wan2GPClient → shared.api (in-process)
                                        (pixelle_video/services/wan2gp_client.py)
```

- The WanGP session is created **lazily** on the first generation and kept
  alive for the whole process, so the model stays loaded in VRAM between
  storyboard frames.
- Model checkpoints are **downloaded automatically** by WanGP on first use.
- Generation results are returned as **local file paths** (no download step).
- TTS keeps using the existing `local` (edge-tts) or ComfyUI modes.

## Workflow descriptors

`workflows/wan2gp/*.json` files are not ComfyUI graphs — they are small
descriptors that map a Pixelle workflow to a WanGP `model_type`
(any file name from `../defaults/*.json` is a valid `model_type`):

| Field | Meaning |
|---|---|
| `source` | Always `"wan2gp"` |
| `model_type` | WanGP model id, e.g. `t2v_fusionix`, `qwen_image_20B`, `ltx2_22B_distilled` |
| `media_type` | `"image"` or `"video"` |
| `fps` | Output fps of the model (16 for Wan, 24 for LTX-2) — used to convert the TTS audio duration into a frame count |
| `frame_quant` | Temporal grid: frame count is snapped **up** to `quant*n+1` so the video covers the narration |
| `resolution_multiple` | Spatial grid for width/height snapping |
| `max_pixels` | Approximate area cap; larger template sizes are scaled down preserving aspect ratio |
| `max_frames` | Hard cap on the frame count |
| `settings` | Extra WanGP task settings merged into every task (steps, guidance, loras, ...) — same keys as WanGP's *Export Settings* |

Bundled descriptors:

- `image_qwen.json` – Qwen Image 20B
- `image_flux.json` – Flux 1 Dev 12B
- `image_z_image.json` – Z-Image Turbo 6B (fast)
- `video_wan2.1_fusionx.json` – Wan 2.1 FusioniX 14B t2v (8 steps)
- `video_wan2.2_t2v.json` – Wan 2.2 t2v 14B
- `video_ltx2_distilled.json` – LTX-2 2.3 Distilled 22B (video + audio)
- `i2v_wan2.2.json` – Wan 2.2 i2v 14B (used by the Image-To-Video pipeline)

Add your own by copying one of these and changing `model_type` /
`settings`. Custom descriptors can also go to `data/workflows/wan2gp/`.

## Configuration

`config.yaml`:

```yaml
comfyui:
  image:
    default_workflow: wan2gp/image_qwen.json
  video:
    default_workflow: wan2gp/video_wan2.1_fusionx.json

wan2gp:
  root: null            # auto-detected (parent repository); set to override
  cli_args: []          # WanGP startup flags, e.g. ["--attention", "sdpa", "--profile", "4"]
  output_dir: null      # optional override for WanGP's outputs folder
```

The `wan2gp/...` workflows also appear in the Web UI dropdowns next to the
selfhost / runninghub ones — no ComfyUI URL or RunningHub key is needed to
use them.

> Note: the WanGP session is created once per process. Changing the
> `wan2gp` config section requires restarting the app.

## Running

The app runs in the **Wan2GP environment**. Wan2GP's `../requirements.txt`
already covers the shared packages (loguru, pydantic, moviepy,
ffmpeg-python, httpx, fastapi, pyyaml, ...); `requirements.txt` in this
folder adds only the Pixelle-specific extras. Install everything **once**:

```bash
conda activate wan2gp                  # the Wan2GP environment
cd Wan2GP
pip install -r requirements.txt -r Pixelle_video/requirements.txt
playwright install --with-deps chromium   # for HTML frame template rendering

cd Pixelle_video
./start_web.sh
```

For Google Colab there is a ready-made end-to-end notebook at the repo
root: **`pixelle_video_wan2gp.ipynb`** (setup → config → topic → final
video, plus an optional Streamlit web UI tunnel).

VRAM notes:

- Image and video use different models; a standard run loads one media
  model at a time (TTS is CPU-side edge-tts by default).
- Use `wan2gp.cli_args` to pass WanGP low-VRAM options (e.g.
  `["--profile", "4"]`) exactly as you would on the `wgp.py` command line.
