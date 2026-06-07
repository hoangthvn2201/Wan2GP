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
Normalization of downloaded stock media into the canonical scene assets.

Stock files arrive in arbitrary codecs / fps / resolutions / aspect ratios.
Generated media doesn't have this problem (workflows are configured for the
project size), so only the search path needs this step:

- Videos are re-encoded to the project size (cover-crop, no letterboxing) at
  the project fps with libx264 + yuv420p and the audio stripped. This matters
  for correctness, not just looks: `compose_final` concatenates the per-scene
  segments with ffmpeg's concat DEMUXER (`-c copy`), which requires identical
  codec / resolution / fps / pixel format across segments. The narration audio
  is added later by `merge_audio_video(replace_audio=True)`.
- Images are cover-cropped to the project size and converted to PNG, keeping
  the `<uid>_image.png` contract of `scene_asset_path`.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger


def pick_orientation(width: Optional[int], height: Optional[int]) -> Optional[str]:
    """'portrait' | 'landscape' | 'square' from the project media size."""
    if not width or not height:
        return None
    if height > width:
        return "portrait"
    if width > height:
        return "landscape"
    return "square"


def parse_template_size(frame_template: str) -> Optional[Tuple[int, int]]:
    """'1080x1920/image_default.html' -> (1080, 1920)."""
    match = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*/", frame_template or "")
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def target_media_size(config) -> Tuple[int, int]:
    """
    The concrete pixel size stock media must be normalized to: the project's
    explicit media size when set, else the frame template's size (parsed from
    its '1080x1920/...' directory name), else a portrait default.
    """
    if config.media_width and config.media_height:
        return int(config.media_width), int(config.media_height)
    parsed = parse_template_size(getattr(config, "frame_template", "") or "")
    if parsed:
        return parsed
    return 1080, 1920


def _cover_filter(stream, width: int, height: int):
    """Scale-to-cover + center-crop to exactly width x height (no letterbox)."""
    import ffmpeg

    return (
        stream
        .filter("scale", width, height, force_original_aspect_ratio="increase")
        .filter("crop", width, height)
    )


def normalize_stock_video(src: str, dst: str, *, fps: int, width: int, height: int) -> str:
    """
    Re-encode a raw stock clip into a concat-demuxer-safe canonical clip:
    exact project size (cover-crop), constant project fps, libx264 + yuv420p,
    no audio, faststart. Returns dst.
    """
    import ffmpeg

    stream = _cover_filter(ffmpeg.input(src).video, width, height)
    (
        ffmpeg.output(
            stream,
            dst,
            vcodec="libx264",
            pix_fmt="yuv420p",
            r=fps,
            crf=23,
            preset="medium",
            an=None,                      # narration is merged in later
            movflags="+faststart",
        )
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info(f"🧰 Normalized stock video → {width}x{height}@{fps}fps: {dst}")
    return dst


def normalize_stock_image(src: str, dst: str, *, width: int, height: int) -> str:
    """Cover-crop a stock photo to the project size and write it as PNG. Returns dst."""
    import ffmpeg

    stream = _cover_filter(ffmpeg.input(src), width, height)
    (
        ffmpeg.output(stream, dst, vframes=1)
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info(f"🧰 Normalized stock image → {width}x{height}: {dst}")
    return dst


def cleanup_raw(path: str):
    """Best-effort removal of a temporary raw download."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as e:
        logger.debug(f"Failed to remove raw download {path}: {e}")
