"""Per-scene data model and scene -> generation-settings mappers.

The pipeline state lives in a plugin-owned gr.State (separate from the main app
`state`). Generation is driven through the WanGP API session, so each scene is
converted into a flat settings dict keyed like the `generate_video` parameters;
unknown keys fall back to the model's `defaults/<model_type>.json`.
"""

import copy
from typing import Any, Dict, List, Optional

# Sane LTX-2 distilled defaults (mirrors defaults/ltx2_22B_distilled.json and
# shared/deepy/settings/gen_video/LTX-2 2.3 Distilled.json).
LTX2_DISTILLED_DEFAULTS = {
    "num_inference_steps": 8,
    "sliding_window_size": 501,
    "sliding_window_overlap": 17,
}
LTX2_DEV_DEFAULTS = {
    "num_inference_steps": 30,
}

DEFAULT_RESOLUTION = "1280x720"
DEFAULT_VIDEO_LENGTH = 241

STAGES = ("image", "video", "tts")

# Field names that hold the spoken text for each TTS architecture.
# qwen3_tts_base/voicedesign and most TTS handlers read the spoken text from
# `prompt`; the voice/style hint (when supported) goes to `alt_prompt`.
TTS_SPOKEN_FIELD = "prompt"
TTS_VOICE_HINT_FIELD = "alt_prompt"


def new_scene(index: int) -> Dict[str, Any]:
    return {
        "index": index,
        # Stage A (script)
        "scene_summary": "",
        "image_prompt": "",
        "video_prompt": "",
        "narration_text": "",
        "tts_prompt": "",
        # toggles
        "use_start_image": True,
        # outputs
        "image_path": None,
        "video_path": None,
        "audio_path": None,
        "final_clip_path": None,
        "image_seed": -1,
        "video_seed": -1,
        # per-stage status: "pending" | "running" | "done" | "error" | "skipped"
        "status": {"image": "pending", "video": "pending", "tts": "pending"},
        "error": None,
    }


def empty_pipeline() -> Dict[str, Any]:
    return {
        "brief": "",
        "overall_script": "",
        "num_scenes": 0,
        "narration_mode": "tts",  # "tts" | "ltx2_native"
        "models": {
            "image_model": None,
            "video_model": "ltx2_22B_distilled",
            "tts_model": None,
        },
        "resolution": DEFAULT_RESOLUTION,
        "video_length": DEFAULT_VIDEO_LENGTH,
        "scenes": [],
    }


def build_scenes(n: int) -> List[Dict[str, Any]]:
    return [new_scene(i) for i in range(max(0, int(n)))]


def scenes_from_llm(script: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a validated LLM script dict into a fresh pipeline-scenes structure."""
    pipeline = empty_pipeline()
    pipeline["overall_script"] = str(script.get("overall_script", "") or "")
    raw_scenes = script.get("scenes", []) or []
    scenes = []
    for i, raw in enumerate(raw_scenes):
        sc = new_scene(i)
        sc["scene_summary"] = str(raw.get("scene_summary", "") or "")
        sc["image_prompt"] = str(raw.get("image_prompt", "") or "")
        sc["video_prompt"] = str(raw.get("video_prompt", "") or "")
        sc["narration_text"] = str(raw.get("narration_text", "") or "")
        sc["tts_prompt"] = str(raw.get("tts_prompt", "") or "")
        scenes.append(sc)
    pipeline["scenes"] = scenes
    pipeline["num_scenes"] = len(scenes)
    return pipeline


# ---------------------------------------------------------------------------
# Scene -> settings mappers (one per modality)
# ---------------------------------------------------------------------------

def scene_to_image_settings(pipeline: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_type": pipeline["models"]["image_model"],
        "prompt": scene["image_prompt"],
        "resolution": pipeline.get("resolution", DEFAULT_RESOLUTION),
        "seed": scene.get("image_seed", -1),
        "batch_size": 1,
    }


def scene_to_video_settings(pipeline: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
    video_model = pipeline["models"]["video_model"] or ""
    settings: Dict[str, Any] = {
        "model_type": video_model,
        "prompt": scene["video_prompt"],
        "resolution": pipeline.get("resolution", DEFAULT_RESOLUTION),
        "video_length": pipeline.get("video_length", DEFAULT_VIDEO_LENGTH),
        "seed": scene.get("video_seed", -1),
    }
    # Start frame only when the optional image stage was used and produced one.
    if scene.get("use_start_image") and scene.get("image_path"):
        settings["image_start"] = scene["image_path"]
    # Variant-specific tuning.
    if "distilled" in video_model:
        settings.update(copy.deepcopy(LTX2_DISTILLED_DEFAULTS))
    else:
        settings.update(copy.deepcopy(LTX2_DEV_DEFAULTS))
    # When narration comes from a separate TTS track, ask LTX-2 for silent video
    # (README "Silent Movie Mode": leave control audio empty -> no lip motion).
    if pipeline.get("narration_mode", "tts") == "tts":
        settings["audio_prompt_type"] = ""
    return settings


def scene_to_tts_settings(pipeline: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "model_type": pipeline["models"]["tts_model"],
        TTS_SPOKEN_FIELD: scene["narration_text"],
    }
    hint = scene.get("tts_prompt", "")
    if hint:
        settings[TTS_VOICE_HINT_FIELD] = hint
    return settings


def settings_fn_for(stage: str):
    return {
        "image": scene_to_image_settings,
        "video": scene_to_video_settings,
        "tts": scene_to_tts_settings,
    }[stage]


def store_output(scene: Dict[str, Any], stage: str, path: str) -> None:
    scene[{"image": "image_path", "video": "video_path", "tts": "audio_path"}[stage]] = path


def validate_pipeline_for_stage(pipeline: Dict[str, Any], stage: str) -> Optional[str]:
    """Return an error string if the pipeline can't run `stage`, else None."""
    models = pipeline.get("models", {})
    if stage == "image" and not models.get("image_model"):
        return "No image model selected."
    if stage == "video" and not models.get("video_model"):
        return "No LTX-2 video model selected."
    if stage == "tts":
        if pipeline.get("narration_mode") == "ltx2_native":
            return "Narration mode is LTX-2 native; TTS stage is skipped."
        if not models.get("tts_model"):
            return "No TTS model selected."
    if not pipeline.get("scenes"):
        return "No scenes to process."
    return None
