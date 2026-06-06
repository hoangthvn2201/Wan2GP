# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VieNeu TTS Utility - Local Vietnamese TTS (https://github.com/pnnbao97/VieNeu-TTS)

Runs the VieNeu-TTS model fully on-device, no API key needed. Model weights
are downloaded automatically from HuggingFace on first use.

Three quality tiers (config: tts.vieneu.mode), each falling back to the next
one if its dependencies/VRAM are missing, so a bad config degrades quality
instead of crashing the pipeline:

  "max"      — full-precision VieNeu-TTS-v2 PyTorch backbone
               (pnnbao-ump/VieNeu-TTS-v2 safetensors) + full NeuCodec codec
               (neuphonic/neucodec). Best audio quality and high-fidelity
               voice cloning (needs ref_text). Needs torch/transformers
               (already in Wan2GP) plus the "max quality" extras of
               requirements-vieneu.txt (neucodec, torchtune, torchao,
               local-attention). GPU recommended.
  "standard" — VieNeu-TTS-v2 Q4_K_M GGUF backbone (llama-cpp-python) +
               int8 ONNX codec decoder (CPU). Better than turbo, no extra
               deps beyond the base vieneu install. No voice cloning
               (decoder-only codec).
  "turbo"    — VieNeu-TTS-v2-Turbo GGUF + ONNX codec
               (pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF + pnnbao-ump/VieNeu-Codec).
               Fastest/CPU-friendliest, lowest quality (upstream warns about
               artifacts on very short sentences).

Base install (from the Wan2GP repo root, inside the Wan2GP env):
    pip install -r Pixelle_video/requirements-vieneu.txt
    pip install vieneu==2.7.0 --no-deps
(--no-deps avoids vieneu's gradio>=5.49.1 / CPU onnxruntime pins clobbering
Wan2GP's gradio==5.29.0 / onnxruntime-gpu; see requirements-vieneu.txt for
build prerequisites and the optional "max quality" extras.)
"""

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# Preset voices of the turbo model (mirrors voices.json in the HuggingFace
# repo pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF, which the engine downloads and
# which OVERRIDES the voices bundled inside the vieneu package — the voice
# IDs there are the full display names, diacritics included).
# Kept as a static list so the web UI can render the voice selector without
# loading the model (engine init downloads weights and is expensive).
VIENEU_VOICES: List[Dict[str, Any]] = [
    {"id": "Xuân Vĩnh (Nam - Miền Nam)", "label": "Xuân Vĩnh (Nam - Miền Nam)", "locale": "vi-VN", "gender": "male"},
    {"id": "Phạm Tuyên (Nam - Miền Bắc)", "label": "Phạm Tuyên (Nam - Miền Bắc)", "locale": "vi-VN", "gender": "male"},
    {"id": "Bích Ngọc (Nữ - Miền Bắc)", "label": "Bích Ngọc (Nữ - Miền Bắc)", "locale": "vi-VN", "gender": "female"},
    {"id": "Thục Đoan (Nữ - Miền Nam)", "label": "Thục Đoan (Nữ - Miền Nam)", "locale": "vi-VN", "gender": "female"},
]

# Default voice of the turbo model (its voices.json "default_voice")
DEFAULT_VIENEU_VOICE = "Xuân Vĩnh (Nam - Miền Nam)"

# Singleton engine (model stays loaded between calls) + lock:
# llama-cpp inference is not thread-safe, serialize synthesis.
_engine = None
_engine_key = None        # (requested_mode, requested_device) the engine was built for
_engine_mode = None       # effective mode after fallback ("max"/"standard"/"turbo")
_engine_lock = threading.Lock()

# Quality fallback chain: if a tier fails to load (missing optional deps,
# no GPU, OOM...), drop one tier instead of failing the whole pipeline.
_MODE_FALLBACK = {"max": "standard", "standard": "turbo"}

# Cache encoded reference audio by (path, mtime) to avoid re-encoding
# the same cloning sample for every frame.
_ref_audio_cache: Dict[tuple, Any] = {}


def is_vieneu_available() -> bool:
    """Check if the vieneu package is installed (without importing it fully)"""
    import importlib.util
    return importlib.util.find_spec("vieneu") is not None


def list_vieneu_voices() -> List[Dict[str, Any]]:
    """
    List available VieNeu preset voices.

    Returns the static bundled list (cheap, no model load). If the engine is
    already loaded, prefers its live preset list (covers custom voices.json).
    """
    if _engine is not None:
        try:
            return [
                {"id": voice_id, "label": description, "locale": "vi-VN", "gender": ""}
                for description, voice_id in _engine.list_preset_voices()
            ]
        except Exception as e:
            logger.warning(f"Failed to list voices from loaded VieNeu engine: {e}")
    return VIENEU_VOICES


def get_vieneu_voice_label(voice_id: str) -> str:
    """Get display label for a VieNeu voice ID (falls back to the ID itself)"""
    for voice in list_vieneu_voices():
        if voice["id"] == voice_id:
            return voice["label"]
    return voice_id


def _resolve_device(device: str) -> str:
    """Downgrade 'cuda' to 'cpu' when CUDA is unavailable (avoid hard crash)"""
    if device != "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                return device
        except ImportError:
            pass
        logger.warning(f"⚠️  VieNeu device '{device}' requested but CUDA is not available; using CPU")
        return "cpu"
    return device


def _build_engine(mode: str, device: str):
    """
    Construct a VieNeu engine for one quality tier (see module docstring).

    Upstream constructor kwargs differ per backend: only the turbo backend
    takes `device=`; the standard backend takes backbone_device/codec_device
    (passing `device=` to it is a TypeError — hence this mapping).
    """
    from vieneu import Vieneu

    if mode == "max":
        return Vieneu(
            mode="standard",
            backbone_repo="pnnbao-ump/VieNeu-TTS-v2",
            gguf_filename=None,                 # None = full-precision safetensors backbone
            backbone_device=device,
            codec_repo="neuphonic/neucodec",    # full PyTorch codec (encode + decode)
            codec_device=device,
        )
    if mode == "standard":
        # Upstream defaults: Q4_K_M GGUF backbone + int8 ONNX decoder (CPU-only codec)
        return Vieneu(mode="standard", backbone_device=device, codec_device="cpu")
    return Vieneu(mode="turbo", device=device)


def _get_engine(mode: str = "turbo", device: str = "cpu"):
    """
    Get or create the shared VieNeu engine (lazy, downloads weights on first use)

    Args:
        mode: Quality tier — "max", "standard" or "turbo" (see module
              docstring). Unknown values are treated as "turbo".
        device: "cpu" or "cuda"

    A tier that fails to initialize falls back down the chain
    (max → standard → turbo) with a warning instead of raising, so a missing
    optional dependency degrades audio quality, not the whole pipeline.
    """
    global _engine, _engine_key, _engine_mode

    key = (mode, device)
    if _engine is not None:
        if _engine_key == key:
            return _engine
        # Config changed: release the old engine before building a new one
        try:
            _engine.close()
        except Exception:
            pass
        _engine = None
        _engine_mode = None
        _ref_audio_cache.clear()

    if not is_vieneu_available():
        raise ImportError(
            "VieNeu-TTS is not installed. Install it with:\n"
            "  pip install -r Pixelle_video/requirements-vieneu.txt\n"
            "  pip install vieneu==2.7.0 --no-deps"
        )

    device = _resolve_device(device)
    current = mode if mode in ("max", "standard", "turbo") else "turbo"
    if current != mode:
        logger.warning(f"⚠️  Unknown VieNeu mode '{mode}', using 'turbo'")

    while True:
        try:
            logger.info(f"🔄 Loading VieNeu-TTS engine (mode={current}, device={device})... "
                        f"(first run downloads model weights from HuggingFace)")
            engine = _build_engine(current, device)
            break
        except Exception as e:
            fallback = _MODE_FALLBACK.get(current)
            if not fallback:
                raise
            logger.warning(
                f"⚠️  VieNeu mode '{current}' failed to load ({type(e).__name__}: {e}); "
                f"falling back to '{fallback}'."
                + (" For 'max' install the max-quality extras listed in "
                   "Pixelle_video/requirements-vieneu.txt." if current == "max" else "")
            )
            current = fallback

    # Cache under the *requested* key so the next call with the same config
    # reuses the (possibly demoted) engine without retrying the failed tier.
    _engine = engine
    _engine_key = key
    _engine_mode = current
    logger.info(f"✅ VieNeu-TTS engine ready (effective mode: {current})")
    return _engine


def _demote_engine() -> bool:
    """
    Drop the loaded engine one quality tier (used when synthesis itself
    fails, e.g. CUDA OOM mid-run). Returns False when already at the bottom.
    """
    global _engine, _engine_key, _engine_mode

    fallback = _MODE_FALLBACK.get(_engine_mode or "turbo")
    if _engine is None or not fallback:
        return False

    requested_key = _engine_key
    device = requested_key[1] if requested_key else "cpu"
    logger.warning(f"⚠️  Demoting VieNeu engine '{_engine_mode}' → '{fallback}' after synthesis failure")

    try:
        _engine.close()
    except Exception:
        pass
    _engine = None
    _engine_mode = None
    _engine_key = None
    _ref_audio_cache.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    _engine = _build_engine(fallback, _resolve_device(device))
    _engine_key = requested_key
    _engine_mode = fallback
    return True


def _normalize_voice_name(name: str) -> str:
    """Diacritic/case-insensitive form for fuzzy voice matching"""
    import unicodedata
    decomposed = unicodedata.normalize("NFD", name)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.replace("đ", "d").replace("Đ", "D").casefold().strip()


def _resolve_voice(engine, voice: Optional[str], ref_audio: Optional[str],
                   ref_text: Optional[str] = None):
    """
    Resolve voice parameter: cloned reference audio > preset voice > default.

    Voice cloning contracts differ per backend: the turbo backend takes raw
    reference codes, while the standard backend (modes "standard"/"max")
    requires a {"codes", "text"} dict — its prompt format needs the
    transcript of the reference audio (ref_text). Cloning problems (missing
    ref_text, encode-incapable codec, bad audio) fall back to the preset
    voice with a warning instead of failing the whole pipeline.

    Preset lookup is forgiving: the model's voices.json (downloaded from
    HuggingFace) may use different IDs across model versions, so an exact
    miss falls back to a diacritic-insensitive substring match, and an
    unknown voice falls back to the engine's default voice with a warning.
    """
    if ref_audio:
        is_standard_backend = _engine_mode in ("max", "standard")
        if is_standard_backend and not (ref_text and ref_text.strip()):
            logger.warning(
                f"⚠️  VieNeu '{_engine_mode}' mode needs the transcript of the reference "
                "audio (ref_text) for voice cloning; falling back to preset voice. "
                "Provide ref_text, or use turbo mode for audio-only cloning."
            )
        else:
            ref_path = Path(ref_audio)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
            cache_key = (str(ref_path.resolve()), ref_path.stat().st_mtime, ref_text or "")
            try:
                if cache_key not in _ref_audio_cache:
                    logger.info(f"🎤 Encoding reference audio for voice cloning: {ref_audio}")
                    codes = engine.encode_reference(str(ref_path))
                    _ref_audio_cache[cache_key] = (
                        {"codes": codes, "text": ref_text.strip()}
                        if is_standard_backend else codes
                    )
                return _ref_audio_cache[cache_key]
            except Exception as e:
                # e.g. "standard" mode's int8 ONNX codec is decoder-only
                logger.warning(
                    f"⚠️  Voice cloning unavailable in '{_engine_mode}' mode "
                    f"({type(e).__name__}: {e}); falling back to preset voice."
                )

    if voice:
        available = [voice_id for _desc, voice_id in engine.list_preset_voices()]

        # 1. Exact ID match
        if voice in available:
            return engine.get_preset_voice(voice)

        # 2. Fuzzy match: diacritic/case-insensitive equality, then substring
        wanted = _normalize_voice_name(voice)
        matches = [v for v in available if _normalize_voice_name(v) == wanted]
        if not matches:
            matches = [
                v for v in available
                if wanted in _normalize_voice_name(v) or _normalize_voice_name(v) in wanted
            ]
        if len(matches) == 1:
            logger.info(f"🔁 VieNeu voice '{voice}' resolved to preset '{matches[0]}'")
            return engine.get_preset_voice(matches[0])

        # 3. Unknown/ambiguous -> engine default, keep the pipeline alive
        logger.warning(
            f"⚠️  VieNeu voice '{voice}' not found"
            + (f" (ambiguous: {matches})" if matches else "")
            + f"; falling back to the model's default voice. Available: {available}"
        )

    # No voice given (or fallback): the engine's own default voice
    return engine.get_preset_voice()


def _apply_speed(audio, speed: float):
    """Time-stretch audio (pitch-preserving) for speed != 1.0"""
    if speed is None or abs(speed - 1.0) < 0.01:
        return audio

    import librosa
    import numpy as np
    return librosa.effects.time_stretch(y=audio.astype(np.float32), rate=float(speed))


def _save_audio(engine, audio, output_path: str) -> str:
    """
    Save audio to output_path. VieNeu outputs 24kHz float PCM; soundfile
    handles .wav directly, other formats (.mp3, ...) go through ffmpeg.
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_path.lower().endswith(".wav"):
        engine.save(audio, output_path)
        return output_path

    # Write temp wav, then convert with ffmpeg (same dependency used elsewhere)
    tmp_wav = f"{output_path}.tmp.wav"
    try:
        engine.save(audio, tmp_wav)
        import ffmpeg
        (
            ffmpeg
            .input(tmp_wav)
            .output(output_path)
            .overwrite_output()
            .run(quiet=True)
        )
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    return output_path


def _vieneu_tts_sync(
    text: str,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    output_path: str = None,
    mode: str = "turbo",
    device: str = "cpu",
) -> str:
    """Synchronous VieNeu synthesis (run via asyncio.to_thread)"""
    with _engine_lock:
        engine = _get_engine(mode=mode, device=device)

        # Synthesis failures (e.g. CUDA OOM mid-run while sharing the GPU
        # with video generation) demote one quality tier and retry instead
        # of failing the pipeline.
        while True:
            try:
                voice_data = _resolve_voice(engine, voice, ref_audio, ref_text)
                logger.debug(f"VieNeu synthesizing {len(text)} chars (voice={voice or 'default'})")
                audio = engine.infer(text=text, voice=voice_data, show_progress=False)
                break
            except FileNotFoundError:
                raise  # bad ref_audio path — not an engine problem, don't demote
            except Exception as e:
                logger.warning(f"⚠️  VieNeu synthesis failed ({type(e).__name__}: {e})")
                if not _demote_engine():
                    raise
                engine = _engine

        audio = _apply_speed(audio, speed)
        return _save_audio(engine, audio, output_path)


async def vieneu_tts(
    text: str,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    output_path: str = None,
    mode: str = "turbo",
    device: str = "cpu",
) -> str:
    """
    Convert text to speech using local VieNeu-TTS (Vietnamese, on-device)

    Args:
        text: Text to synthesize (Vietnamese, English also supported)
        voice: Preset voice ID (e.g. "Xuân Vĩnh (Nam - Miền Nam)", see
               VIENEU_VOICES; fuzzy-matched, None = model default)
        speed: Speech speed multiplier (1.0 = normal), applied as a
               pitch-preserving time stretch
        ref_audio: Optional reference audio (3-5s wav/mp3) for zero-shot
                   voice cloning; takes precedence over `voice`
        ref_text: Transcript of ref_audio — required for cloning in
                  "max"/"standard" modes (ignored in turbo mode)
        output_path: Output file path (.wav saved directly, .mp3 etc.
                     converted via ffmpeg)
        mode: Quality tier — "max" (full-precision backbone + full codec,
              best quality, GPU recommended), "standard" (Q4 GGUF + int8
              ONNX) or "turbo" (fastest, lowest quality, default). Each
              tier falls back to the next on failure.
        device: "cpu" or "cuda"

    Returns:
        Generated audio file path

    Example:
        audio_path = await vieneu_tts(
            text="Xin chào, đây là giọng đọc tiếng Việt.",
            voice="Bích Ngọc (Nữ - Miền Bắc)",
            mode="max",
            device="cuda",
            output_path="output/hello.mp3"
        )
    """
    if not output_path:
        import uuid
        Path("output").mkdir(parents=True, exist_ok=True)
        output_path = f"output/{uuid.uuid4().hex}.wav"

    return await asyncio.to_thread(
        _vieneu_tts_sync,
        text=text,
        voice=voice,
        speed=speed,
        ref_audio=ref_audio,
        ref_text=ref_text,
        output_path=output_path,
        mode=mode,
        device=device,
    )
