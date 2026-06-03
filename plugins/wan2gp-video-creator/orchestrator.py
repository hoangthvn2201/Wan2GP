"""Sequential generation engine over scenes x stages.

WanGP's API session allows only ONE generation at a time (`_submit_tasks` raises
RuntimeError otherwise), so every stage runs strictly sequentially: submit a task,
block on `job.result()`, then move on. This mirrors the pattern in
`plugins/wan2gp-sample/plugin.py`.
"""

from typing import Any, Callable, Dict, List, Optional

from . import scene_model


class ProgressAdapter:
    """Bridges WanGP generation callbacks to a gr.Progress-like callable.

    `report(ratio, desc)` is any callable; pass a gr.Progress instance's __call__
    or a no-op for headless use.
    """

    def __init__(self, report: Optional[Callable[[float, str], None]], label: str):
        self._report = report
        self._label = label
        self.ratio = 0.0

    def _emit(self, desc: str):
        if self._report is not None:
            try:
                self._report(self.ratio, f"{self._label} - {desc}" if desc else self._label)
            except Exception:
                pass

    def on_status(self, status):
        status = str(status or "").strip()
        if status:
            self._emit(status)

    def on_progress(self, update):
        try:
            self.ratio = max(0.0, min(1.0, float(getattr(update, "progress", 0)) / 100.0))
        except Exception:
            self.ratio = 0.0
        self._emit(str(getattr(update, "status", "") or "Generating..."))


def run_stage_over_scenes(
    api_session,
    pipeline: Dict[str, Any],
    stage: str,
    scene_indices: List[int],
    active_job: Dict[str, Any],
    report: Optional[Callable[[float, str], None]],
    cancel_flag: Dict[str, bool],
) -> Dict[str, Any]:
    """Run a single stage ("image" | "video" | "tts") over the given scenes.

    Mutates `pipeline` in place (status + output paths) and returns it.
    """
    settings_fn = scene_model.settings_fn_for(stage)
    total = len(scene_indices)
    for n, i in enumerate(scene_indices):
        if cancel_flag.get("cancel"):
            break
        scene = pipeline["scenes"][i]

        # Honour the optional image stage / skipped narration.
        if stage == "image" and not scene.get("use_start_image", True):
            scene["status"]["image"] = "skipped"
            continue

        scene["status"][stage] = "running"
        scene["error"] = None

        settings = settings_fn(pipeline, scene)
        cb = ProgressAdapter(report, f"Scene {i + 1}/{total} [{stage}]")
        try:
            job = api_session.submit_task(settings, callbacks=cb)
        except Exception as e:  # e.g. another generation already in progress
            scene["status"][stage] = "error"
            scene["error"] = f"submit failed: {e}"
            continue

        active_job["job"] = job
        try:
            result = job.result()
        finally:
            if active_job.get("job") is job:
                active_job["job"] = None

        if getattr(result, "cancelled", False):
            scene["status"][stage] = "pending"
            break
        if not result.success or not result.generated_files:
            scene["status"][stage] = "error"
            errs = list(result.errors or [])
            scene["error"] = str(errs[0]) if errs else "WanGP returned no output file."
            continue

        scene_model.store_output(scene, stage, result.generated_files[0])
        scene["status"][stage] = "done"

    return pipeline


def run_full_pipeline(
    api_session,
    pipeline: Dict[str, Any],
    active_job: Dict[str, Any],
    report: Optional[Callable[[float, str], None]],
    cancel_flag: Dict[str, bool],
) -> Dict[str, Any]:
    """Stage-major run (all images -> all videos -> all narration).

    Stage-major minimises costly model_type reloads (one model load per stage
    instead of three per scene).
    """
    all_idx = list(range(len(pipeline["scenes"])))

    # Stage B - start images (only scenes that opted in).
    if pipeline["models"].get("image_model"):
        run_stage_over_scenes(api_session, pipeline, "image", all_idx, active_job, report, cancel_flag)
    if cancel_flag.get("cancel"):
        return pipeline

    # Stage C - video.
    run_stage_over_scenes(api_session, pipeline, "video", all_idx, active_job, report, cancel_flag)
    if cancel_flag.get("cancel"):
        return pipeline

    # Stage D - narration (skip entirely in LTX-2 native audio mode).
    if pipeline.get("narration_mode", "tts") == "tts" and pipeline["models"].get("tts_model"):
        run_stage_over_scenes(api_session, pipeline, "tts", all_idx, active_job, report, cancel_flag)

    return pipeline
