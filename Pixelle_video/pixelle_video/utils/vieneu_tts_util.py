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

Runs the VieNeu-TTS model fully on-device in "turbo" mode (GGUF backbone via
llama-cpp-python + ONNX codec), so it works on CPU and needs no API key.
Model weights are downloaded automatically from HuggingFace on first use
(pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF + pnnbao-ump/VieNeu-Codec).

Install (from the Wan2GP repo root, inside the Wan2GP env):
    pip install -r Pixelle_video/requirements-vieneu.txt
    pip install vieneu==2.7.0 --no-deps
(--no-deps avoids vieneu's gradio>=5.49.1 / CPU onnxruntime pins clobbering
Wan2GP's gradio==5.29.0 / onnxruntime-gpu; see requirements-vieneu.txt for
build prerequisites — cmake + C compiler for llama-cpp-python, Rust for sea-g2p.)
"""

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# Preset voices bundled with the vieneu package (mirrors vieneu/assets/voices.json).
# Kept as a static list so the web UI can render the voice selector without
# loading the model (engine init downloads weights and is expensive).
VIENEU_VOICES: List[Dict[str, Any]] = [
    {"id": "Binh", "label": "Bình (nam miền Bắc)", "locale": "vi-VN", "gender": "male"},
    {"id": "Tuyen", "label": "Tuyên (nam miền Bắc)", "locale": "vi-VN", "gender": "male"},
    {"id": "Ly", "label": "Ly (nữ miền Bắc)", "locale": "vi-VN", "gender": "female"},
    {"id": "Ngoc", "label": "Ngọc (nữ miền Bắc)", "locale": "vi-VN", "gender": "female"},
    {"id": "Vinh", "label": "Vĩnh (nam miền Nam)", "locale": "vi-VN", "gender": "male"},
    {"id": "Doan", "label": "Đoan (nữ miền Nam)", "locale": "vi-VN", "gender": "female"},
]

DEFAULT_VIENEU_VOICE = "Binh"

# Singleton engine (model stays loaded between calls) + lock:
# llama-cpp inference is not thread-safe, serialize synthesis.
_engine = None
_engine_key = None
_engine_lock = threading.Lock()

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


def _get_engine(mode: str = "turbo", device: str = "cpu"):
    """
    Get or create the shared VieNeu engine (lazy, downloads weights on first use)

    Args:
        mode: VieNeu backend ("turbo" recommended: GGUF+ONNX, runs on CPU;
              "standard"/"fast" need the GPU extras of vieneu)
        device: "cpu" or "cuda"
    """
    global _engine, _engine_key

    key = (mode, device)
    if _engine is not None and _engine_key == key:
        return _engine

    try:
        from vieneu import Vieneu
    except ImportError as e:
        raise ImportError(
            "VieNeu-TTS is not installed. Install it with:\n"
            "  pip install -r Pixelle_video/requirements-vieneu.txt\n"
            "  pip install vieneu==2.7.0 --no-deps"
        ) from e

    logger.info(f"🔄 Loading VieNeu-TTS engine (mode={mode}, device={device})... "
                f"(first run downloads model weights from HuggingFace)")
    _engine = Vieneu(mode=mode, device=device)
    _engine_key = key
    logger.info("✅ VieNeu-TTS engine ready")
    return _engine


def _resolve_voice(engine, voice: Optional[str], ref_audio: Optional[str]):
    """Resolve voice parameter: cloned reference audio > preset voice > default"""
    if ref_audio:
        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
        cache_key = (str(ref_path.resolve()), ref_path.stat().st_mtime)
        if cache_key not in _ref_audio_cache:
            logger.info(f"🎤 Encoding reference audio for voice cloning: {ref_audio}")
            _ref_audio_cache[cache_key] = engine.encode_reference(str(ref_path))
        return _ref_audio_cache[cache_key]

    if voice:
        return engine.get_preset_voice(voice)

    return engine.get_preset_voice(DEFAULT_VIENEU_VOICE)


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
    output_path: str = None,
    mode: str = "turbo",
    device: str = "cpu",
) -> str:
    """Synchronous VieNeu synthesis (run via asyncio.to_thread)"""
    with _engine_lock:
        engine = _get_engine(mode=mode, device=device)
        voice_data = _resolve_voice(engine, voice, ref_audio)

        logger.debug(f"VieNeu synthesizing {len(text)} chars (voice={voice or 'default'})")
        audio = engine.infer(text=text, voice=voice_data, show_progress=False)

        audio = _apply_speed(audio, speed)
        return _save_audio(engine, audio, output_path)


async def vieneu_tts(
    text: str,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    ref_audio: Optional[str] = None,
    output_path: str = None,
    mode: str = "turbo",
    device: str = "cpu",
) -> str:
    """
    Convert text to speech using local VieNeu-TTS (Vietnamese, on-device)

    Args:
        text: Text to synthesize (Vietnamese, English also supported)
        voice: Preset voice ID (e.g. "Binh", "Ly", see VIENEU_VOICES)
        speed: Speech speed multiplier (1.0 = normal), applied as a
               pitch-preserving time stretch
        ref_audio: Optional reference audio (3-5s wav/mp3) for zero-shot
                   voice cloning; takes precedence over `voice`
        output_path: Output file path (.wav saved directly, .mp3 etc.
                     converted via ffmpeg)
        mode: VieNeu backend ("turbo" = CPU-friendly GGUF/ONNX, default)
        device: "cpu" or "cuda"

    Returns:
        Generated audio file path

    Example:
        audio_path = await vieneu_tts(
            text="Xin chào, đây là giọng đọc tiếng Việt.",
            voice="Ly",
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
        output_path=output_path,
        mode=mode,
        device=device,
    )
