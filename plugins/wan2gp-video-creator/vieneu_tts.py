"""Direct VieNeu-TTS integration for the Video Creator plugin.

VieNeu-TTS (https://github.com/pnnbao97/VieNeu-TTS) is a bilingual (Vietnamese /
English) TTS with **built-in preset voices**, so it needs NO reference audio
sample. We call it directly here — outside WanGP's model pipeline — so the
narration stage works out of the box with a default voice.

Install on the WanGP host:  pip install vieneu        (CPU / turbo)
                            pip install "vieneu[gpu]"  (standard GPU model)

The engine is loaded lazily and cached process-wide (first call downloads the
weights from HuggingFace, repo `pnnbao-ump/VieNeu-TTS-v2`).
"""

import os
import threading

# Sentinel used as the "model_type" in the plugin's TTS dropdown. It is NOT a
# WanGP model — the orchestrator routes this value to synthesize() below instead
# of api_session.submit_task().
VIENEU_MODEL_KEY = "vieneu_tts"
VIENEU_LABEL = "VieNeu-TTS (default voice)"

# "standard" (highest quality, needs vieneu[gpu]) or "turbo" (lightweight, CPU).
DEFAULT_MODE = "standard"

_lock = threading.Lock()
_engine = None


def is_importable() -> bool:
    """True if the `vieneu` package is installed (no model load)."""
    try:
        import importlib.util

        return importlib.util.find_spec("vieneu") is not None
    except Exception:
        return False


def _get_engine(mode: str = DEFAULT_MODE, emotion: str = "natural"):
    global _engine
    with _lock:
        if _engine is not None:
            return _engine
        from vieneu import Vieneu  # imported lazily; may download weights

        if mode == "turbo":
            _engine = Vieneu(mode="turbo")
        else:
            _engine = Vieneu(emotion=emotion)
        return _engine


def synthesize(text: str, out_path: str, voice_id=None, mode: str = DEFAULT_MODE) -> str:
    """Synthesize `text` to a WAV at `out_path` using a preset/default voice.

    Returns the output path. Raises with a clear message if `vieneu` is missing.
    """
    if not text or not str(text).strip():
        raise ValueError("Narration text is empty.")
    if not is_importable():
        raise RuntimeError(
            "VieNeu-TTS is not installed. On the WanGP host run: pip install vieneu "
            "(or 'pip install \"vieneu[gpu]\"' for the standard GPU model)."
        )
    engine = _get_engine(mode=mode)
    if voice_id:
        voice = engine.get_preset_voice(voice_id)
        audio = engine.infer(text=text, voice=voice)
    else:
        audio = engine.infer(text=text)  # built-in default voice (no reference)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    engine.save(audio, out_path)
    return out_path


def list_preset_voices():
    """List (description, voice_id) preset voices. Loads the engine — avoid calling
    during UI build."""
    try:
        return _get_engine().list_preset_voices()
    except Exception:
        return []
