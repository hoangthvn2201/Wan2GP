"""Modality filtering over the live WanGP model registry.

Modality flags (``audio_only`` / ``image_outputs``) are injected onto the model
definition at runtime by the model handlers, so they are only reliable when read
through the live ``get_model_def`` callable from wgp.py -- NOT by parsing the raw
``defaults/*.json`` files (which lack them for most models).
"""

from typing import Callable, List, Optional, Tuple

# (display_label, model_type) pairs ready to feed a gr.Dropdown's `choices`.
Choice = Tuple[str, str]


def classify(model_type: str, get_model_def: Callable[[str], Optional[dict]]) -> str:
    """Return one of "audio" | "image" | "video" for a model_type."""
    d = get_model_def(model_type) or {}
    if d.get("audio_only", False):
        return "audio"
    if d.get("image_outputs", False):
        return "image"
    return "video"


def _label(model_type: str, get_model_name: Optional[Callable[[str], str]]) -> str:
    if get_model_name is not None:
        try:
            name = get_model_name(model_type)
            if name:
                return name
        except Exception:
            pass
    return model_type


def list_models_by_modality(
    displayed_model_types: List[str],
    get_model_def: Callable[[str], Optional[dict]],
    get_model_name: Optional[Callable[[str], str]],
    modality: str,
) -> List[Choice]:
    out: List[Choice] = []
    for mt in displayed_model_types or []:
        try:
            if classify(mt, get_model_def) == modality:
                out.append((_label(mt, get_model_name), mt))
        except Exception:
            continue
    return sorted(out, key=lambda c: c[0].lower())


def list_image_models(displayed_model_types, get_model_def, get_model_name) -> List[Choice]:
    return list_models_by_modality(displayed_model_types, get_model_def, get_model_name, "image")


def list_tts_models(displayed_model_types, get_model_def, get_model_name) -> List[Choice]:
    return list_models_by_modality(displayed_model_types, get_model_def, get_model_name, "audio")


def list_ltx2_video_models(
    displayed_model_types: List[str],
    get_model_def: Callable[[str], Optional[dict]],
    get_base_model_type: Optional[Callable[[str], str]],
    get_model_name: Optional[Callable[[str], str]],
) -> List[Choice]:
    """All LTX-2 family video models (dev, distilled, 1.1, gguf, nvfp4, edit_anything...)."""
    out: List[Choice] = []
    for mt in displayed_model_types or []:
        try:
            if classify(mt, get_model_def) != "video":
                continue
            base = ""
            if get_base_model_type is not None:
                try:
                    base = get_base_model_type(mt) or ""
                except Exception:
                    base = ""
            if base.startswith("ltx2") or str(mt).startswith("ltx2"):
                out.append((_label(mt, get_model_name), mt))
        except Exception:
            continue
    return sorted(out, key=lambda c: c[0].lower())


# Canonical dev / distilled keys for the simple Dev/Distilled radio.
LTX2_DEV_KEY = "ltx2_22B"
LTX2_DISTILLED_KEY = "ltx2_22B_distilled"


def resolve_variant_key(
    variant: str,
    available: List[Choice],
) -> Optional[str]:
    """Map a "Dev"/"Distilled" radio choice to a concrete, available model_type.

    Falls back to any available LTX2 model if the canonical key isn't present.
    """
    keys = [mt for _, mt in available]
    if variant == "Distilled":
        if LTX2_DISTILLED_KEY in keys:
            return LTX2_DISTILLED_KEY
        for mt in keys:
            if "distilled" in mt:
                return mt
    else:  # Dev
        if LTX2_DEV_KEY in keys:
            return LTX2_DEV_KEY
        for mt in keys:
            if "distilled" not in mt:
                return mt
    return keys[0] if keys else None
