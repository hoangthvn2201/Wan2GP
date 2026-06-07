# Wan2GP Server

A clean FastAPI service that exposes [WanGP](https://github.com/deepbeepmeep/Wan2GP)'s
in-process Python API (`shared/api.py`) as a REST API with three generation
tasks:

| Task | Endpoint |
|---|---|
| Text → Image | `POST /v1/generations/text-to-image` |
| Text → Video | `POST /v1/generations/text-to-video` |
| Image → Video | `POST /v1/generations/image-to-video` |

Models are loaded **in the server process** by WanGP and kept in VRAM
between requests. Checkpoints are downloaded automatically on first use.
Jobs run **one at a time** (single GPU); submissions queue up and are
processed in order.

> This server uses **WanGP by DeepBeepMeep**. Use of the WanGP API is
> subject to the WanGP Terms and Conditions.

## Quick start

```bash
conda activate wan2gp                      # the Wan2GP environment
cd Wan2GP
pip install -r requirements.txt -r wan2gp_server/requirements.txt

python -m wan2gp_server --port 8000        # add --eager to preload the runtime
```

Interactive OpenAPI docs: `http://localhost:8000/docs`

```bash
# 1. Submit (returns immediately with a queued job)
curl -s localhost:8000/v1/generations/text-to-image \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a red bicycle in front of a bakery", "width": 768, "height": 768}'
# -> {"id": "a1b2c3d4e5f6", "status": "queued", ...}

# 2. Poll
curl -s localhost:8000/v1/jobs/a1b2c3d4e5f6
# -> {"status": "running", "progress": {"phase": "inference", "percent": 54, ...}, ...}

# 3. Download when status == "succeeded"
curl -s -o out.png localhost:8000/v1/jobs/a1b2c3d4e5f6/files/0
```

Or block until done in a single call with `"wait": true`:

```bash
curl -s localhost:8000/v1/generations/text-to-video \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a paper boat drifting down a rainy gutter", "duration_seconds": 4, "wait": true}'
```

### Python client

```python
from wan2gp_server.client import Wan2GPServerClient

client = Wan2GPServerClient("http://localhost:8000")
client.wait_until_ready()

job = client.text_to_image("a red bicycle in front of a bakery")
job = client.wait_for(job["id"])
client.download(job, "bicycle.png")

job = client.image_to_video("the bicycle rides away", image="bicycle.png", duration_seconds=4)
job = client.wait_for(job["id"])
client.download(job, "bicycle.mp4")
```

## API reference

### `GET /health`

```json
{"status": "ok", "version": "0.1.0", "wan2gp_root": "/path/to/Wan2GP",
 "runtime_loaded": true, "active_job_id": null, "queued_jobs": 0}
```

### `GET /v1/models?task=t2i|t2v|i2v`

Lists model presets. Each preset bundles a WanGP `model_type` with tuned
defaults (steps, fps, resolution/frame grids). `is_default: true` marks the
preset used when a request omits `model`.

Built-in presets:

| id | task | model | notes |
|---|---|---|---|
| `z-image-turbo` | t2i | Z-Image Turbo 6B | fast, low VRAM (default) |
| `qwen-image` | t2i | Qwen Image 20B | high quality |
| `flux-dev` | t2i | Flux 1 Dev 12B | high quality |
| `wan21-t2v-1.3b` | t2v | Wan 2.1 1.3B | low VRAM (default) |
| `wan21-fusionix` | t2v | Wan 2.1 FusioniX 14B | fast, 8 steps |
| `wan22-t2v` | t2v | Wan 2.2 14B | high quality |
| `ltx2-distilled` | t2v | LTX-2 22B Distilled | video **with audio**, 24 fps |
| `wan21-fun-inp-1.3b` | i2v | Wan 2.1 Fun InP 1.3B | low VRAM (default) |
| `wan21-fusionix-i2v` | i2v | Wan 2.1 FusioniX 14B | fast, 8 steps |
| `wan22-i2v` | i2v | Wan 2.2 14B | high quality |
| `wan22-i2v-lightning` | i2v | Wan 2.2 Lightning v2 14B | fast, 4 steps |
| `wan22-ti2v-5b` | i2v | Wan 2.2 TI2V 5B | 720p 24 fps, medium VRAM |
| `ltx2-distilled-i2v` | i2v | LTX-2 22B Distilled | video **with audio**, 24 fps |

### `POST /v1/models/preload`

Warm up models before the first real request: each listed model runs a
**minimal warmup generation** (256×256, 2 steps, ~5 frames for video)
through WanGP's normal path — that downloads the checkpoint (first time)
and loads the weights into VRAM. The throwaway warmup output is deleted
automatically.

```json
{"models": ["z-image-turbo", "wan21-t2v-1.3b"], "wait": false, "wait_timeout_seconds": 7200}
```

- `models` empty/omitted → the server's default t2i/t2v/i2v presets.
- Presets sharing the same underlying `model_type` are deduplicated.
- Returns `{"jobs": [JobInfo, ...]}` — one `task: "preload"` job per model,
  queued FIFO like any generation; poll them via `GET /v1/jobs/{id}`.
- WanGP keeps **one** model in VRAM, so the *last* listed model stays
  loaded; earlier ones end up checkpoint-cached on disk (still a big win:
  no multi-GB download on first use).

```python
client.preload()                                   # server defaults, blocks + prints progress
client.preload(["qwen-image", "ltx2-distilled"])   # custom list, last one stays in VRAM
```

Add your own: drop a JSON file into `WAN2GP_SERVER_PRESETS_DIR` with
`{"id", "task", "model_type", "media_type", "fps", "frame_quant",
"resolution_multiple", "max_pixels", "max_frames", "settings"}` —
`model_type` is any file name from the Wan2GP `defaults/*.json` folder.

### `POST /v1/generations/text-to-image`

```json
{
  "prompt": "required",
  "model": "z-image-turbo",
  "width": 1024, "height": 1024,
  "negative_prompt": null, "steps": null, "seed": null, "guidance_scale": null,
  "settings": {},
  "wait": false, "wait_timeout_seconds": 3600
}
```

- `width`/`height` are snapped to the model's grid and capped by its
  `max_pixels` (aspect ratio preserved) — the actual value is echoed back in
  `settings.resolution`.
- `settings` is a raw passthrough of WanGP task settings (the keys produced
  by WanGP's *Export Settings* button) merged last — expert use.

### `POST /v1/generations/text-to-video`

Adds to the above:

```json
{"duration_seconds": 5.0, "num_frames": null}
```

- `duration_seconds` is converted to the model's frame grid (`quant*n+1`),
  rounding **up**, capped at the model's `max_frames`.
- `num_frames` overrides `duration_seconds` when set.

### `POST /v1/generations/image-to-video`

Adds the start image, as **exactly one** of:

```json
{"image_b64": "<base64 or data: URI>",
 "image_url": "https://...",
 "image_path": "/path/on/server.png",
 "image_asset_id": "from POST /v1/assets"}
```

`width`/`height` default to the input image's size (then grid-snapped and
`max_pixels`-capped).

### `POST /v1/assets`

Multipart upload of an input image (`file` field). Returns
`{"asset_id": "...", "path": "..."}` for use in `image_asset_id`.

### Jobs

- `GET /v1/jobs?limit=50` — recent jobs, newest first
- `GET /v1/jobs/{id}` — status / progress / outputs:

```json
{
  "id": "a1b2c3d4e5f6",
  "task": "text_to_video",
  "status": "running",
  "model": "wan21-t2v-1.3b",
  "queue_position": null,
  "progress": {"phase": "inference", "status": "Denoising | 12s", "percent": 54,
               "current_step": 4, "total_steps": 8},
  "files": [{"index": 0, "media_type": "video", "filename": "...mp4",
             "path": "/abs/path.mp4", "url": "/v1/jobs/a1b2c3d4e5f6/files/0"}],
  "error": null,
  "settings": {"model_type": "t2v_1.3B", "resolution": "832x480", "video_length": 81, "...": "..."},
  "created_at": 1770000000.0, "started_at": 1770000001.0, "finished_at": null
}
```

  Status lifecycle: `queued → running → succeeded | failed | cancelled`.
  `progress.phase` is one of `loading_model`, `encoding_text`, `inference`,
  `decoding`, `downloading_output`, `cancelled`.

- `POST /v1/jobs/{id}/cancel` — cooperative cancel (queued or running)
- `GET /v1/jobs/{id}/files/{index}` — stream/download an output file

## Agent usage notes

1. `GET /v1/models?task=t2v` to discover model ids; omit `model` for a safe default.
2. Submit with `wait: false` and poll `GET /v1/jobs/{id}` every few seconds —
   first-time model download can take minutes (`progress.phase: "loading_model"`).
3. Generations are queued FIFO; `queue_position` tells you how many jobs are ahead.
4. On `succeeded`, fetch `files[*].url`. `failed` puts the reason in `error` —
   the job JSON never throws away the resolved `settings`, so retries can be adjusted.
5. Switching presets between requests is allowed but unloads/reloads model
   weights — batching same-model requests is much faster.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `WAN2GP_ROOT` | parent of this package | Wan2GP installation folder |
| `WAN2GP_CLI_ARGS` | – | WanGP startup flags, e.g. `--attention sdpa --profile 4` |
| `WAN2GP_SERVER_HOST` | `0.0.0.0` | Bind host |
| `WAN2GP_SERVER_PORT` | `8000` | Bind port |
| `WAN2GP_SERVER_OUTPUT_DIR` | WanGP default | Where generated files are written |
| `WAN2GP_SERVER_DATA_DIR` | `wan2gp_server/data` | Uploaded / downloaded input images |
| `WAN2GP_SERVER_T2I_MODEL` | `z-image-turbo` | Default t2i preset |
| `WAN2GP_SERVER_T2V_MODEL` | `wan21-t2v-1.3b` | Default t2v preset |
| `WAN2GP_SERVER_I2V_MODEL` | `wan21-fun-inp-1.3b` | Default i2v preset |
| `WAN2GP_SERVER_PRESETS_DIR` | – | Folder with extra `*.json` presets |
| `WAN2GP_SERVER_EAGER_INIT` | `0` | `1` = load the WanGP runtime at startup |
| `WAN2GP_SERVER_API_KEY` | – | If set, `/v1/*` requires the `X-API-Key` header |

Low-VRAM GPUs (e.g. Colab T4): keep the default presets and set
`WAN2GP_CLI_ARGS="--profile 5"`.

## Notebook serving

`wan2gp_server.ipynb` at the repository root is a Colab-ready notebook:
setup → start the server in the background → call all three endpoints →
optional public URL via a Cloudflare quick tunnel (with
`WAN2GP_SERVER_API_KEY` recommended when tunneling).
