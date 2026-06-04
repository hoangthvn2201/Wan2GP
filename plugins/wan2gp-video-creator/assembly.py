"""Final assembly: mux per-scene narration onto each clip, then concat to a master.

Uses ffmpeg via subprocess. Degrades gracefully when ffmpeg is missing or a clip
fails -- the caller can still offer per-scene downloads.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(cmd: List[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, proc.stderr[-500:]
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def mux_narration(video_path: str, audio_path: str, out_path: str) -> Tuple[bool, str]:
    """Replace the clip's audio track with the narration track.

    Re-encodes audio to AAC; copies video stream. Trims to whichever is shorter.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    return _run(cmd)


def concat_clips(clip_paths: List[str], out_path: str, *, reencode: bool = True) -> Tuple[bool, str]:
    """Concatenate clips into one master video.

    Re-encodes by default (concat demuxer is fragile across differing fps/codecs).
    """
    if not clip_paths:
        return False, "No clips to concatenate."
    if len(clip_paths) == 1:
        try:
            shutil.copyfile(clip_paths[0], out_path)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    # Build a concat list file.
    fd, list_file = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for p in clip_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
        if reencode:
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                out_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                "-c", "copy", out_path,
            ]
        return _run(cmd)
    finally:
        try:
            os.remove(list_file)
        except OSError:
            pass


def assemble(
    pipeline: Dict[str, Any],
    out_dir: str,
    *,
    base_name: str = "video_creator_master",
) -> Tuple[Optional[str], List[str]]:
    """Mux narration per scene (tts mode) then concat into a master.

    Returns (master_path_or_None, warnings).
    """
    warnings: List[str] = []
    if not ffmpeg_available():
        return None, ["ffmpeg not found on PATH; cannot assemble. Per-scene clips are still available."]

    os.makedirs(out_dir, exist_ok=True)
    tts_mode = pipeline.get("narration_mode", "tts") == "tts"
    clip_paths: List[str] = []

    for scene in pipeline["scenes"]:
        video_path = scene.get("video_path")
        if not video_path or not os.path.exists(video_path):
            warnings.append(f"Scene {scene['index'] + 1}: no video, skipped.")
            continue
        audio_path = scene.get("audio_path")
        if tts_mode and audio_path and os.path.exists(audio_path):
            muxed = os.path.join(out_dir, f"{base_name}_scene{scene['index'] + 1}.mp4")
            ok, err = mux_narration(video_path, audio_path, muxed)
            if ok:
                scene["final_clip_path"] = muxed
                clip_paths.append(muxed)
            else:
                warnings.append(f"Scene {scene['index'] + 1}: mux failed ({err}); used silent clip.")
                scene["final_clip_path"] = video_path
                clip_paths.append(video_path)
        else:
            scene["final_clip_path"] = video_path
            clip_paths.append(video_path)

    if not clip_paths:
        return None, warnings + ["No usable clips to assemble."]

    master = os.path.join(out_dir, f"{base_name}.mp4")
    ok, err = concat_clips(clip_paths, master, reencode=True)
    if not ok:
        return None, warnings + [f"Concatenation failed: {err}"]
    return master, warnings
