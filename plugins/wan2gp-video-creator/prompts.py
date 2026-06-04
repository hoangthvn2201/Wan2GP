"""System / user prompt templates and the target JSON schema for script generation."""

import json

SYSTEM_PROMPT = (
    "You are a professional short-form video scriptwriter and prompt engineer. "
    "Given a creative brief, you produce a cohesive narrative split into a fixed "
    "number of scenes. For EACH scene you write:\n"
    "  - scene_summary: one sentence describing what happens.\n"
    "  - image_prompt: a vivid, concrete text-to-image prompt for the opening frame "
    "(subject, setting, lighting, style, composition).\n"
    "  - video_prompt: a text-to-video prompt describing the motion and camera work "
    "that animates that opening frame.\n"
    "  - narration_text: the exact words spoken over the scene (1-3 short sentences).\n"
    "  - tts_prompt: a short voice/tone hint for the narrator (e.g. 'calm male "
    "narrator, warm tone').\n"
    "Keep scenes visually distinct but narratively connected. "
    "Respond with a SINGLE JSON object only. No prose, no markdown code fences."
)

# Schema shown to the model as a skeleton it must match.
_SCHEMA_SKELETON = {
    "overall_script": "string - 2-4 sentence overview of the whole video",
    "scenes": [
        {
            "scene_summary": "string",
            "image_prompt": "string",
            "video_prompt": "string",
            "narration_text": "string",
            "tts_prompt": "string",
        }
    ],
}


def build_user_prompt(brief: str, num_scenes: int) -> str:
    n = int(num_scenes)
    return (
        f"Brief:\n{brief}\n\n"
        f"Produce exactly {n} scenes.\n"
        f"Return a JSON object matching this schema (the `scenes` array MUST contain "
        f"exactly {n} items):\n"
        f"{json.dumps(_SCHEMA_SKELETON, indent=2)}"
    )
