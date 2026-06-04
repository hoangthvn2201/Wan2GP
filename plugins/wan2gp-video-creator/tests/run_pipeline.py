"""Headless end-to-end test runner for the Video Creator pipeline.

Run from the WanGP repo root, with the WanGP environment active for real runs:

    # 1) Dry run — NO GPU, no models. Validates orchestration, settings mappers,
    #    VieNeu routing and (if ffmpeg is present) real mux+concat assembly using
    #    synthetic test clips:
    python plugins/wan2gp-video-creator/tests/run_pipeline.py --dry-run

    # 2) Real single-scene smoke run (image -> video -> narration -> master).
    #    Downloads/loads real models; needs GPU:
    python plugins/wan2gp-video-creator/tests/run_pipeline.py \
        --scenes 1 --image-model flux_dev_chroma_hd --video-variant distilled

    # 3) Only some stages (e.g. validate the start-image fix on an existing image):
    python plugins/wan2gp-video-creator/tests/run_pipeline.py --stages video --scenes 1

    # 4) Scenes written by an LLM instead of the built-in sample script:
    python plugins/wan2gp-video-creator/tests/run_pipeline.py \
        --llm-base-url http://localhost:8000 --llm-model mock

Outputs land in <out-dir> (default: outputs/video_creator_test). Exit code 0 only
if every requested stage succeeded for every scene (and assembly, when run).
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PLUGINS_DIR = os.path.join(REPO_ROOT, "plugins")
sys.path.insert(0, PLUGINS_DIR)
sys.path.insert(0, REPO_ROOT)

PKG = "wan2gp-video-creator"
scene_model = importlib.import_module(f"{PKG}.scene_model")
orchestrator = importlib.import_module(f"{PKG}.orchestrator")
assembly = importlib.import_module(f"{PKG}.assembly")
vieneu_tts = importlib.import_module(f"{PKG}.vieneu_tts")
model_registry = importlib.import_module(f"{PKG}.model_registry")

SAMPLE_SCENES = [
    {
        "scene_summary": "A red vintage car parked on a misty mountain road at dawn.",
        "image_prompt": "A red vintage convertible parked on a winding mountain road, "
                        "morning mist, golden sunrise light, cinematic, photorealistic.",
        "video_prompt": "Slow cinematic dolly-in toward the red vintage car as the mist "
                        "drifts across the road, sunrise glow intensifying.",
        "narration_text": "Every journey begins with a single quiet morning.",
        "tts_prompt": "calm male narrator, warm tone",
    },
    {
        "scene_summary": "The car driving along a coastal cliff road.",
        "image_prompt": "A red vintage convertible driving along a coastal cliff road, "
                        "turquoise sea below, clear blue sky, aerial view, photorealistic.",
        "video_prompt": "Aerial tracking shot following the car along the cliff road, "
                        "waves crashing below, smooth camera motion.",
        "narration_text": "The open road carries us toward the horizon.",
        "tts_prompt": "calm male narrator, warm tone",
    },
]


# ---------------------------------------------------------------------------
# Dry-run stubs
# ---------------------------------------------------------------------------

def _ffmpeg_test_media(out_dir, kind, name, seconds=2):
    """Create a tiny real video/image/audio file with ffmpeg for dry runs."""
    path = os.path.join(out_dir, name)
    if kind == "video":
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=steelblue:s=320x180:d={seconds}",
               "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path]
    elif kind == "image":
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=tomato:s=320x180:d=0.1",
               "-frames:v", "1", path]
    else:  # audio
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", path]
    subprocess.run(cmd, capture_output=True)
    return path if os.path.exists(path) else None


class DryRunResult:
    def __init__(self, files):
        self.success = bool(files)
        self.generated_files = files
        self.errors = []
        self.cancelled = False


class DryRunJob:
    def __init__(self, result):
        self._result = result
        self.done = True

    def result(self, timeout=None):
        return self._result

    def cancel(self):
        pass


class DryRunSession:
    """Stands in for WanGPSession: records settings, emits synthetic media."""

    def __init__(self, out_dir, have_ffmpeg):
        self.out_dir = out_dir
        self.have_ffmpeg = have_ffmpeg
        self.submitted = []

    def submit_task(self, settings, callbacks=None):
        self.submitted.append(settings)
        n = len(self.submitted)
        image_mode = settings.get("image_mode", 0)
        if callbacks is not None and hasattr(callbacks, "on_status"):
            callbacks.on_status(f"[dry-run] task {n}: model={settings.get('model_type')}")
        if image_mode > 0:
            path = (self._media("image", f"dry_image_{n}.png")
                    or self._touch(f"dry_image_{n}.png"))
        else:
            path = (self._media("video", f"dry_video_{n}.mp4")
                    or self._touch(f"dry_video_{n}.mp4"))
        return DryRunJob(DryRunResult([path]))

    def _media(self, kind, name):
        return _ffmpeg_test_media(self.out_dir, kind, name) if self.have_ffmpeg else None

    def _touch(self, name):
        path = os.path.join(self.out_dir, name)
        open(path, "w").write("dry-run placeholder")
        return path


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(args):
    if args.llm_base_url:
        llm_client = importlib.import_module(f"{PKG}.llm_client")
        cfg = llm_client.LLMConfig(args.llm_base_url, args.llm_api_key, args.llm_model)
        ok, msg = llm_client.test_connection(cfg)
        print(f"LLM test connection: {'OK' if ok else 'FAILED'} - {msg}")
        if not ok:
            sys.exit(2)
        script = llm_client.generate_script(cfg, args.brief, args.scenes)
        if script.get("_warning"):
            print(f"  warning: {script['_warning']}")
        pipeline = scene_model.scenes_from_llm(script)
    else:
        pipeline = scene_model.empty_pipeline()
        pipeline["scenes"] = scene_model.build_scenes(args.scenes)
        for sc, sample in zip(pipeline["scenes"], SAMPLE_SCENES * args.scenes):
            sc.update({k: sample[k] for k in
                       ("scene_summary", "image_prompt", "video_prompt",
                        "narration_text", "tts_prompt")})
        pipeline["num_scenes"] = args.scenes

    video_model = args.video_model or (
        model_registry.LTX2_DISTILLED_KEY if args.video_variant == "distilled"
        else model_registry.LTX2_DEV_KEY)
    pipeline["models"] = {
        "image_model": args.image_model,
        "video_model": video_model,
        "tts_model": args.tts_model,
    }
    pipeline["resolution"] = args.resolution
    pipeline["video_length"] = args.video_length
    pipeline["narration_mode"] = args.narration_mode
    return pipeline


def report(ratio, desc):
    print(f"  [{ratio * 100:5.1f}%] {desc}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="no GPU/models; stub generation, exercise orchestration + assembly")
    p.add_argument("--scenes", type=int, default=2)
    p.add_argument("--stages", default="image,video,tts", help="comma list from: image,video,tts")
    p.add_argument("--image-model", default="qwen_image_20B")
    p.add_argument("--video-variant", choices=["distilled", "dev"], default="distilled")
    p.add_argument("--video-model", default=None, help="exact LTX-2 model_type (overrides --video-variant)")
    p.add_argument("--tts-model", default=vieneu_tts.VIENEU_MODEL_KEY,
                   help=f"WanGP TTS model_type, or '{vieneu_tts.VIENEU_MODEL_KEY}' for VieNeu default voice")
    p.add_argument("--narration-mode", choices=["tts", "ltx2_native"], default="tts")
    p.add_argument("--resolution", default="1280x720")
    p.add_argument("--video-length", type=int, default=121, help="frames (default short for testing)")
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "outputs", "video_creator_test"))
    p.add_argument("--skip-assembly", action="store_true")
    # LLM (optional; built-in sample script otherwise)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default="")
    p.add_argument("--llm-model", default="mock")
    p.add_argument("--brief", default="A 2-scene cinematic short about a road trip along the coast.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    pipeline = build_pipeline(args)

    have_ffmpeg = shutil.which("ffmpeg") is not None
    if args.dry_run:
        session = DryRunSession(args.out_dir, have_ffmpeg)
        # stub VieNeu so the dry run needs no install
        def _fake_vieneu(text, out_path, voice_id=None, mode="standard"):
            real = _ffmpeg_test_media(args.out_dir, "audio", os.path.basename(out_path)) if have_ffmpeg else None
            if real is None:
                open(out_path, "w").write("dry-run narration")
                real = out_path
            return real
        vieneu_tts.synthesize = _fake_vieneu
        print(f"[dry-run] ffmpeg={'yes' if have_ffmpeg else 'NO (placeholder files, assembly will fail gracefully)'}")
    else:
        from shared.api import init  # heavy: boots the WanGP runtime
        print("Initializing WanGP session (this loads the runtime)...")
        session = init(root=REPO_ROOT, console_isatty=False)

    cancel_flag = {"cancel": False}
    active_job = {"job": None}
    try:
        for stage in stages:
            err = scene_model.validate_pipeline_for_stage(pipeline, stage)
            if err:
                print(f"-- stage {stage}: SKIPPED ({err})")
                continue
            print(f"-- stage {stage}: {len(pipeline['scenes'])} scene(s)")
            orchestrator.run_stage_over_scenes(
                session, pipeline, stage, list(range(len(pipeline["scenes"]))),
                active_job, report, cancel_flag, args.out_dir)
    except KeyboardInterrupt:
        cancel_flag["cancel"] = True
        job = active_job.get("job")
        if job is not None and not job.done:
            job.cancel()
        print("\nCancelled.")

    master = None
    if not args.skip_assembly and "video" in stages:
        master, warnings = assembly.assemble(pipeline, args.out_dir, base_name="test_master")
        for w in warnings:
            print(f"  assembly warning: {w}")

    # ---- summary ----
    print("\n================ SUMMARY ================")
    failed = False
    for sc in pipeline["scenes"]:
        st = sc["status"]
        line = f"Scene {sc['index'] + 1}: " + "  ".join(f"{k}={st[k]}" for k in scene_model.STAGES)
        if sc.get("error"):
            line += f"  ERROR: {sc['error']}"
        print(line)
        for key in ("image_path", "video_path", "audio_path", "final_clip_path"):
            if sc.get(key):
                print(f"    {key}: {sc[key]}")
        failed = failed or any(st[s] == "error" for s in scene_model.STAGES)
    if master:
        print(f"MASTER: {master}")
    elif not args.skip_assembly and "video" in stages:
        print("MASTER: not produced")
        failed = True

    if args.dry_run and isinstance(session, DryRunSession):
        print("\n[dry-run] submitted settings:")
        print(json.dumps(session.submitted, indent=2, default=str))
        # sanity assertions for the two real-run bugs that were fixed
        for s in session.submitted:
            mt = s.get("model_type", "")
            if s.get("image_mode") == 1:
                assert "image_start" not in s, "image task should not carry image_start"
            if mt.startswith("ltx2") and "image_start" in s:
                assert s.get("image_prompt_type") == "S", "video task with start image must set image_prompt_type='S'"
        print("[dry-run] settings sanity checks passed (image_mode=1 / image_prompt_type='S').")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
