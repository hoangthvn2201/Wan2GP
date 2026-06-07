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
Scene-by-Scene Engine

Exposes the Pixelle-Video generation pipeline as individual steps so the UI
can pause between them for user review / edit / regeneration:

    1. generate_script()       -> title + narrations          (editable)
    2. generate_prompts()      -> per-scene media prompts     (editable)
    3. per scene:
         generate_audio()      -> TTS audio                   (preview / regen)
         generate_media()      -> image or video              (preview / regen)
         render_segment()      -> subtitled frame + segment   (preview / regen)
    4. compose_final()         -> concat + BGM + persistence

The generation logic mirrors `StandardPipeline` + `FrameProcessor` from the
Pixelle_video core (kept unchanged in ../Pixelle_video); only the asset paths
are different: files are named after the scene `uid` instead of the frame
index, so scenes can be edited / inserted / deleted / regenerated safely.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import httpx
from loguru import logger

from Pixelle_video.pixelle_video.models.storyboard import (
    Storyboard,
    StoryboardConfig,
    StoryboardFrame,
)
from Pixelle_video.pixelle_video.utils.content_generators import (
    generate_image_prompts,
    generate_narrations_from_topic,
    generate_title,
    split_narration_script,
)
from Pixelle_video.pixelle_video.utils.os_util import (
    create_task_output_dir,
    get_resource_path,
    get_task_final_video_path,
    get_task_path,
    list_resource_dirs,
    list_resource_files,
)
from Pixelle_video.pixelle_video.utils.prompt_helper import build_image_prompt
from Pixelle_video.pixelle_video.utils.template_util import (
    get_template_type,
    resolve_template_path,
)
from Pixelle_video.pixelle_video.services.video import VideoService

from sbs.models import Scene, SceneProject


# File extension per asset type (same convention as os_util.get_task_frame_path)
_EXT_MAP = {
    "audio": "mp3",
    "image": "png",
    "video": "mp4",
    "composed": "png",
    "segment": "mp4",
}


class SceneBySceneEngine:
    """Step-wise orchestration on top of PixelleVideoCore services."""

    def __init__(self, core):
        """
        Args:
            core: initialized PixelleVideoCore instance (provides llm/tts/media/...)
        """
        self.core = core

    # ==================== Step 1: Script ====================

    async def generate_script(
        self,
        text: str,
        mode: str = "generate",
        n_scenes: int = 5,
        split_mode: str = "paragraph",
        title: Optional[str] = None,
        min_narration_words: int = 5,
        max_narration_words: int = 20,
    ) -> Tuple[str, List[str]]:
        """
        Generate (or split) the narration script and determine the title.

        Mirrors StandardPipeline.generate_content + determine_title.
        """
        if mode == "generate":
            narrations = await generate_narrations_from_topic(
                self.core.llm,
                topic=text,
                n_scenes=n_scenes,
                min_words=min_narration_words,
                max_words=max_narration_words,
            )
        else:  # fixed script
            narrations = await split_narration_script(text, split_mode=split_mode)

        if title:
            final_title = title
        elif mode == "generate":
            final_title = await generate_title(self.core.llm, text, strategy="auto")
        else:
            final_title = await generate_title(self.core.llm, text, strategy="llm")

        logger.info(f"📝 Script ready: title='{final_title}', {len(narrations)} scene(s)")
        return final_title, narrations

    async def rewrite_narration(
        self,
        narration: str,
        topic: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """AI-rewrite a single scene narration (keeps language and length)."""
        context = f'The video is about: "{topic}".\n' if topic else ""
        extra = f"Follow this instruction: {instruction}\n" if instruction else ""
        prompt = (
            "You are a short-video copywriter.\n"
            f"{context}"
            "Rewrite the following narration line for one scene of the video. "
            "Keep the SAME language as the original, a similar length, and the same core meaning, "
            "but make it more vivid and engaging.\n"
            f"{extra}"
            "Return ONLY the rewritten narration text, with no quotes and no explanations.\n\n"
            f"Original narration: {narration}"
        )
        response = await self.core.llm(prompt, temperature=0.9, max_tokens=1024)
        rewritten = response.strip().strip('"').strip("'").strip()
        # Some reasoning models leak a <think> block — content_generators strips it
        # for JSON; here we just take the last non-empty line as a fallback.
        if "</think>" in rewritten:
            rewritten = rewritten.split("</think>")[-1].strip()
        lines = [l.strip() for l in rewritten.splitlines() if l.strip()]
        return lines[-1] if lines else narration

    # ==================== Step 2: Prompts ====================

    @staticmethod
    def media_requirement(frame_template: str) -> str:
        """'static' | 'image' | 'video' from the template naming convention."""
        return get_template_type(Path(frame_template or "1080x1920/default.html").name)

    # ==================== Workflow capabilities ====================

    @staticmethod
    def list_i2v_workflows() -> List[str]:
        """
        Workflows that can animate a start image (image → video).

        Only `i2v_*.json` workflow files support a start image — regular
        `video_*` (t2v) workflows cannot take one. Honors `data/workflows/`
        overrides, like all other resources.

        Returns keys like "wan2gp/i2v_wan2.2.json".
        """
        keys = []
        try:
            for source in list_resource_dirs("workflows"):
                for fname in list_resource_files("workflows", source):
                    if fname.startswith("i2v_") and fname.endswith(".json"):
                        keys.append(f"{source}/{fname}")
        except Exception as e:
            logger.warning(f"Failed to scan i2v workflows: {e}")
        return sorted(keys)

    def list_image_workflows(self) -> List[str]:
        """Image workflows usable for the i2v start frame (keys like "wan2gp/image_qwen.json")."""
        try:
            return [k for k in self.core.media.available if Path(k).name.startswith("image_")]
        except Exception as e:
            logger.warning(f"Failed to scan image workflows: {e}")
            return []

    async def generate_prompts(
        self,
        narrations: List[str],
        prompt_prefix: str = "",
        min_words: int = 30,
        max_words: int = 60,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """
        Generate one media prompt per narration (mirrors StandardPipeline.plan_visuals).

        Returns final prompts with the style prefix already applied.
        """
        base_prompts = await generate_image_prompts(
            self.core.llm,
            narrations=narrations,
            min_words=min_words,
            max_words=max_words,
            progress_callback=progress_callback,
        )
        return [build_image_prompt(p, prompt_prefix or "") for p in base_prompts]

    async def generate_prompt_for(
        self,
        narration: str,
        prompt_prefix: str = "",
        min_words: int = 30,
        max_words: int = 60,
    ) -> str:
        """Regenerate the media prompt for a single scene."""
        prompts = await self.generate_prompts(
            [narration],
            prompt_prefix=prompt_prefix,
            min_words=min_words,
            max_words=max_words,
        )
        return prompts[0]

    # ==================== Project creation ====================

    def create_project(
        self,
        title: str,
        narrations: List[str],
        prompts: List[Optional[str]],
        params: dict,
    ) -> SceneProject:
        """
        Create the task directory + StoryboardConfig and wrap each narration
        into a Scene. Mirrors StandardPipeline.setup_environment +
        initialize_storyboard (TTS parameter handling included).
        """
        task_dir, task_id = create_task_output_dir()
        logger.info(f"📁 Scene-by-scene task directory created: {task_dir}")

        # --- TTS parameter compatibility (same rules as StandardPipeline) ---
        tts_inference_mode = params.get("tts_inference_mode") or "local"
        if tts_inference_mode == "local":
            voice_id = params.get("tts_voice") or "zh-CN-YunjianNeural"
            tts_workflow = None
        elif tts_inference_mode == "vieneu":
            # None falls back to config vieneu.voice, then to the model default
            voice_id = params.get("tts_voice")
            tts_workflow = None
        else:  # comfyui
            voice_id = None
            tts_workflow = params.get("tts_workflow")

        frame_template = params.get("frame_template") or "1080x1920/default.html"

        config = StoryboardConfig(
            task_id=task_id,
            n_storyboard=len(narrations),
            min_narration_words=params.get("min_narration_words", 5),
            max_narration_words=params.get("max_narration_words", 20),
            min_image_prompt_words=params.get("min_image_prompt_words", 30),
            max_image_prompt_words=params.get("max_image_prompt_words", 60),
            video_fps=params.get("video_fps", 30),
            tts_inference_mode=tts_inference_mode,
            voice_id=voice_id,
            tts_workflow=tts_workflow,
            tts_speed=params.get("tts_speed", 1.2),
            ref_audio=params.get("ref_audio"),
            ref_text=params.get("ref_text"),
            media_width=params.get("media_width"),
            media_height=params.get("media_height"),
            media_workflow=params.get("media_workflow"),
            frame_template=frame_template,
            template_params=params.get("template_params"),
        )

        scenes = [
            Scene(narration=narration, prompt=prompt)
            for narration, prompt in zip(narrations, prompts)
        ]

        # --- Media mode -----------------------------------------------------
        # static template -> 'none'; image template -> 'image';
        # video template  -> 't2v' (default) or 'i2v' if the user opted in AND
        # an i2v-capable workflow is selected (workflows/<source>/i2v_*.json).
        media_requirement = self.media_requirement(frame_template)
        media_mode = params.get("media_mode")
        if media_requirement == "static":
            media_mode = "none"
        elif media_requirement == "image":
            media_mode = "image"
        elif media_mode != "i2v":
            media_mode = "t2v"
        if media_mode == "i2v" and not params.get("i2v_workflow"):
            raise ValueError("media_mode='i2v' requires an 'i2v_workflow' param "
                             "(an i2v_*.json workflow descriptor)")

        return SceneProject(
            title=title,
            task_id=task_id,
            task_dir=task_dir,
            config=config,
            params=dict(params),
            scenes=scenes,
            media_requirement=media_requirement,
            media_mode=media_mode,
        )

    # ==================== Per-scene steps ====================

    def scene_asset_path(self, project: SceneProject, scene: Scene, file_type: str) -> str:
        """uid-based asset path inside the task frames/ directory."""
        filename = f"{scene.uid}_{file_type}.{_EXT_MAP[file_type]}"
        return get_task_path(project.task_id, "frames", filename)

    async def generate_audio(self, project: SceneProject, scene: Scene, index: int) -> Scene:
        """TTS for one scene (mirrors FrameProcessor._step_generate_audio)."""
        config = project.config
        output_path = self.scene_asset_path(project, scene, "audio")

        tts_params = {
            "text": scene.narration,
            "inference_mode": config.tts_inference_mode,
            "output_path": output_path,
            "index": index + 1,  # 1-based index for workflow
        }
        if config.tts_inference_mode == "local":
            if config.voice_id:
                tts_params["voice"] = config.voice_id
            if config.tts_speed is not None:
                tts_params["speed"] = config.tts_speed
        elif config.tts_inference_mode == "vieneu":
            if config.voice_id:
                tts_params["voice"] = config.voice_id
            if config.tts_speed is not None:
                tts_params["speed"] = config.tts_speed
            if config.ref_audio:
                tts_params["ref_audio"] = config.ref_audio
                if config.ref_text:
                    tts_params["ref_text"] = config.ref_text
        else:  # comfyui
            if config.tts_workflow:
                tts_params["workflow"] = config.tts_workflow
            if config.voice_id:
                tts_params["voice"] = config.voice_id
            if config.tts_speed is not None:
                tts_params["speed"] = config.tts_speed
            if config.ref_audio:
                tts_params["ref_audio"] = config.ref_audio

        audio_path = await self.core.tts(**tts_params)
        scene.audio_path = audio_path
        scene.duration = await self._get_audio_duration(audio_path)
        # Audio changed -> previously rendered segment no longer matches
        scene.composed_path = None
        scene.segment_path = None
        # Video clips are length-synced to the audio: a (re)generated narration
        # invalidates any existing clip (the i2v start image stays valid).
        # Imported clips are NOT length-synced (the segment step pads/trims
        # them), so they survive audio regeneration.
        if (project.is_video_workflow and scene.video_path
                and scene.media_origin != "imported"):
            scene.invalidate_video()
        logger.info(f"🎤 Scene {index + 1} audio ready ({scene.duration:.2f}s): {audio_path}")
        return scene

    async def generate_media(self, project: SceneProject, scene: Scene, index: int) -> Scene:
        """Image/video generation for one scene (mirrors FrameProcessor._step_generate_media)."""
        config = project.config
        is_video_workflow = project.is_video_workflow
        media_type = "video" if is_video_workflow else "image"

        media_params = {
            "prompt": scene.prompt,
            "workflow": config.media_workflow,
            "media_type": media_type,
            "width": config.media_width,
            "height": config.media_height,
            "index": index + 1,
        }
        # Video workflows sync the clip length to the narration audio
        if is_video_workflow and scene.duration:
            media_params["duration"] = scene.duration
            logger.info(f"  → Target video duration: {scene.duration:.2f}s (from TTS audio)")

        media_result = await self.core.media(**media_params)
        scene.media_type = media_result.media_type

        if media_result.is_image:
            local_path = await self._fetch_media(
                media_result.url, self.scene_asset_path(project, scene, "image")
            )
            scene.image_path = local_path
            scene.video_path = None
            logger.info(f"🖼️ Scene {index + 1} image ready: {local_path}")
        elif media_result.is_video:
            local_path = await self._fetch_media(
                media_result.url, self.scene_asset_path(project, scene, "video")
            )
            scene.video_path = local_path
            scene.image_path = None
            if media_result.duration:
                scene.duration = media_result.duration
            else:
                scene.duration = await self._get_video_duration(local_path)
            logger.info(f"🎬 Scene {index + 1} video ready ({scene.duration:.2f}s): {local_path}")
        else:
            raise ValueError(f"Unknown media type: {media_result.media_type}")

        # Media changed -> previously rendered segment no longer matches
        scene.media_origin = None   # freshly generated (replaces any import)
        scene.composed_path = None
        scene.segment_path = None
        return scene

    # -------------------- Image → Video (i2v) mode --------------------

    async def generate_start_image(self, project: SceneProject, scene: Scene, index: int) -> Scene:
        """
        i2v step A: generate the still that will seed the video.

        Uses the image workflow chosen for the project (params['i2v_image_workflow'],
        falling back to the configured default image workflow).
        """
        config = project.config
        media_params = {
            "prompt": scene.prompt,
            "workflow": project.params.get("i2v_image_workflow"),  # None -> config default
            "media_type": "image",
            "width": config.media_width,
            "height": config.media_height,
            "index": index + 1,
        }

        media_result = await self.core.media(**media_params)
        if not media_result.is_image:
            raise ValueError(f"Start-image workflow returned {media_result.media_type}, expected image")

        local_path = await self._fetch_media(
            media_result.url, self.scene_asset_path(project, scene, "image")
        )
        scene.image_path = local_path
        scene.media_type = "image"
        scene.media_origin = None   # freshly generated (replaces any import)
        # New start image -> old animation + segment no longer match
        scene.video_path = None
        scene.invalidate_segment()
        logger.info(f"🖼️ Scene {index + 1} start image ready: {local_path}")
        return scene

    async def animate_image(self, project: SceneProject, scene: Scene, index: int) -> Scene:
        """
        i2v step B: animate the start image into a video clip whose length is
        synced to the narration audio.

        Only i2v-capable workflows (workflows/<source>/i2v_*.json) accept a
        start image; executes them directly (mirrors the original i2v pipeline):
        wan2gp descriptors run in-process, ComfyUI ones via ComfyKit.
        """
        import json as _json

        if not scene.image_path:
            raise RuntimeError("Generate the start image first (scene has no image)")
        if not scene.duration:
            raise RuntimeError("Generate the audio first (the clip length is synced to it)")

        workflow_key = project.params.get("i2v_workflow")
        if not workflow_key:
            raise ValueError("No i2v workflow configured for this project")
        source, _, fname = workflow_key.partition("/")
        workflow_path = get_resource_path("workflows", source, fname)

        with open(workflow_path, "r", encoding="utf-8") as f:
            descriptor = _json.load(f)

        config = project.config
        output_path = self.scene_asset_path(project, scene, "video")

        if descriptor.get("source") == "wan2gp":
            # In-process WanGP generation (duration snapped via descriptor fps/frame_quant)
            from Pixelle_video.pixelle_video.services.wan2gp_client import get_wan2gp_client

            client = get_wan2gp_client()
            generated = await client.generate(
                descriptor,
                scene.prompt,
                media_type="video",
                width=config.media_width,
                height=config.media_height,
                duration=scene.duration,
                image_start=scene.image_path,
            )
            local_path = await self._fetch_media(generated, output_path)
        else:
            # ComfyUI workflow (selfhost / runninghub) via ComfyKit
            kit = await self.core._get_or_create_comfykit()
            if descriptor.get("source") == "runninghub" and "workflow_id" in descriptor:
                workflow_input = descriptor["workflow_id"]
            else:
                workflow_input = str(workflow_path)

            workflow_params = {
                "image": scene.image_path,
                "prompt": scene.prompt,
                "duration": scene.duration,
            }
            result = await kit.execute(workflow_input, workflow_params)
            if result.status != "completed":
                raise Exception(f"i2v generation failed: {result.msg or 'Unknown error'}")
            if not result.videos:
                raise Exception("i2v workflow returned no video")
            local_path = await self._fetch_media(result.videos[0], output_path)

        scene.video_path = local_path
        scene.media_type = "video"
        scene.media_origin = None   # freshly generated (replaces any import)
        scene.invalidate_segment()
        # Keep the narration-driven duration unless the clip differs meaningfully
        clip_duration = await self._get_video_duration(local_path)
        if clip_duration > 1.0:
            scene.duration = clip_duration
        logger.info(f"🎬 Scene {index + 1} animated ({scene.duration:.2f}s): {local_path}")
        return scene

    # -------------------- Imported media --------------------

    async def import_media(
        self,
        project: SceneProject,
        scene: Scene,
        index: int,
        src_path: str,
        original_name: Optional[str] = None,
    ) -> Scene:
        """
        Use a user-supplied image/video as this scene's media instead of
        generating it. The file is normalized into the canonical scene asset
        (`<uid>_image.png` / `<uid>_video.mp4`) at the project size — videos
        also re-encoded to the project fps so the final concat (demuxer,
        `-c copy`) stays safe. Imported clips are not length-synced: the
        segment step pads/trims them to the narration audio.
        """
        from Pixelle_video.pixelle_video.utils.imported_media import (
            detect_media_kind,
            normalize_imported_image,
            normalize_imported_video,
            target_media_size,
        )

        kind = detect_media_kind(original_name or src_path)
        if kind is None:
            raise ValueError(f"Unsupported media file: {original_name or src_path}")

        width, height = target_media_size(project.config)

        if kind == "video":
            output_path = self.scene_asset_path(project, scene, "video")
            normalize_imported_video(src_path, output_path,
                                     fps=project.config.video_fps,
                                     width=width, height=height)
            scene.video_path = output_path
            scene.image_path = None
            scene.media_type = "video"
        else:
            output_path = self.scene_asset_path(project, scene, "image")
            normalize_imported_image(src_path, output_path,
                                     width=width, height=height)
            scene.image_path = output_path
            scene.video_path = None
            scene.media_type = "image"

        scene.media_origin = "imported"
        # Media changed -> previously rendered segment no longer matches
        scene.invalidate_segment()
        logger.info(f"📥 Scene {index + 1} imported {kind} ready: {output_path}")
        return scene

    async def render_segment(self, project: SceneProject, scene: Scene, index: int) -> Scene:
        """
        Render the subtitled HTML frame and build the per-scene video segment
        (mirrors FrameProcessor._step_compose_frame + _step_create_video_segment).
        """
        config = project.config

        # --- Compose frame with the HTML template ---
        from Pixelle_video.pixelle_video.services.frame_html import HTMLFrameGenerator

        template_path = resolve_template_path(config.frame_template)
        ext = {"index": index + 1}
        if config.template_params:
            ext.update(config.template_params)

        generator = HTMLFrameGenerator(template_path)
        media_path = scene.video_path if scene.media_type == "video" else scene.image_path
        composed_path = await generator.generate_frame(
            title=project.title,
            text=scene.narration,
            image=media_path,
            ext=ext,
            output_path=self.scene_asset_path(project, scene, "composed"),
        )
        scene.composed_path = composed_path

        # --- Create the video segment ---
        output_path = self.scene_asset_path(project, scene, "segment")
        video_service = VideoService()

        if scene.media_type == "video":
            # Overlay the (transparent) HTML frame on the video, then add narration
            temp_overlay = output_path + "_overlay.mp4"
            video_service.overlay_image_on_video(
                video=scene.video_path,
                overlay_image=scene.composed_path,
                output=temp_overlay,
                scale_mode="contain",
            )
            segment_path = video_service.merge_audio_video(
                video=temp_overlay,
                audio=scene.audio_path,
                output=output_path,
                replace_audio=True,
                audio_volume=1.0,
            )
            if os.path.exists(temp_overlay):
                os.unlink(temp_overlay)
        else:
            # Image or static template: still image + narration audio
            segment_path = video_service.create_video_from_image(
                image=scene.composed_path,
                audio=scene.audio_path,
                output=output_path,
                fps=config.video_fps,
            )

        scene.segment_path = segment_path
        logger.info(f"🎞️ Scene {index + 1} segment ready: {segment_path}")
        return scene

    async def process_scene(
        self,
        project: SceneProject,
        scene: Scene,
        index: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Scene:
        """
        Run all missing steps for one scene:
            audio -> media -> segment                    (image / t2v modes)
            audio -> start image -> animate -> segment   (i2v mode)
        """
        if not scene.audio_path:
            if progress_callback:
                progress_callback("audio")
            await self.generate_audio(project, scene, index)

        if scene.media_origin == "imported" and scene.has_media:
            # User-imported media is used as-is — nothing to generate (the
            # i2v Animate button can still animate an imported still manually)
            pass
        elif project.is_i2v and scene.prompt:
            if not scene.image_path:
                if progress_callback:
                    progress_callback("image_start")
                await self.generate_start_image(project, scene, index)
            if not scene.video_path:
                if progress_callback:
                    progress_callback("animate")
                await self.animate_image(project, scene, index)
        elif project.needs_media and scene.prompt and not scene.has_media:
            if progress_callback:
                progress_callback("media")
            await self.generate_media(project, scene, index)

        if not scene.segment_path:
            if progress_callback:
                progress_callback("segment")
            await self.render_segment(project, scene, index)
        return scene

    # ==================== Final composition ====================

    async def compose_final(
        self,
        project: SceneProject,
        bgm_path: Optional[str] = None,
        bgm_volume: float = 0.2,
        bgm_mode: str = "loop",
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Concatenate all scene segments, add BGM and persist task metadata so
        the result shows up in the History page (mirrors
        StandardPipeline.post_production + finalize + _persist_task_data).
        """
        if not project.all_segments_ready:
            missing = [i + 1 for i, s in enumerate(project.scenes) if not s.segment_path]
            raise RuntimeError(f"Scenes not rendered yet: {missing}")

        segment_paths = [s.segment_path for s in project.scenes]
        final_video_path = get_task_final_video_path(project.task_id)

        video_service = VideoService()
        final_video_path = video_service.concat_videos(
            videos=segment_paths,
            output=final_video_path,
            bgm_path=bgm_path,
            bgm_volume=bgm_volume,
            bgm_mode=bgm_mode,
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_video_path, output_path)
            final_video_path = output_path

        project.final_video_path = final_video_path
        project.total_duration = sum(s.duration for s in project.scenes)
        file_size = Path(final_video_path).stat().st_size

        logger.success(f"🎬 Scene-by-scene video completed: {final_video_path}")

        await self._persist(project, file_size)

        return {
            "video_path": final_video_path,
            "duration": project.total_duration,
            "file_size": file_size,
            "n_scenes": len(project.scenes),
        }

    async def _persist(self, project: SceneProject, file_size: int):
        """Persist metadata + storyboard (compatible with the History page)."""
        try:
            storyboard = self._to_storyboard(project)

            input_params = dict(project.params)
            input_params.setdefault("title", project.title)

            metadata = {
                "task_id": project.task_id,
                "created_at": project.created_at.isoformat(),
                "completed_at": datetime.now().isoformat(),
                "status": "completed",
                "input": input_params,
                "result": {
                    "video_path": project.final_video_path,
                    "duration": project.total_duration,
                    "file_size": file_size,
                    "n_frames": len(project.scenes),
                },
                "config": {
                    "pipeline": "scene_by_scene",
                    "llm_model": self.core.config.get("llm", {}).get("model", "unknown"),
                    "llm_base_url": self.core.config.get("llm", {}).get("base_url", "unknown"),
                    "comfyui_url": self.core.config.get("comfyui", {}).get("comfyui_url", "unknown"),
                    "runninghub_enabled": bool(
                        self.core.config.get("comfyui", {}).get("runninghub_api_key")
                    ),
                },
            }

            await self.core.persistence.save_task_metadata(project.task_id, metadata)
            await self.core.persistence.save_storyboard(project.task_id, storyboard)
            logger.info(f"💾 Saved task metadata + storyboard: {project.task_id}")
        except Exception as e:
            # Persistence failure shouldn't break the generated video
            logger.error(f"Failed to persist scene-by-scene task data: {e}")

    def _to_storyboard(self, project: SceneProject) -> Storyboard:
        """Convert the SceneProject into the core Storyboard model."""
        storyboard = Storyboard(
            title=project.title,
            config=project.config,
            created_at=project.created_at,
        )
        for i, scene in enumerate(project.scenes):
            storyboard.frames.append(
                StoryboardFrame(
                    index=i,
                    narration=scene.narration,
                    image_prompt=scene.prompt,
                    audio_path=scene.audio_path,
                    media_type=scene.media_type,
                    image_path=scene.image_path,
                    video_path=scene.video_path,
                    composed_image_path=scene.composed_path,
                    video_segment_path=scene.segment_path,
                    duration=scene.duration,
                    created_at=scene.created_at,
                )
            )
        storyboard.final_video_path = project.final_video_path
        storyboard.total_duration = project.total_duration
        storyboard.completed_at = datetime.now()
        return storyboard

    # ==================== Helpers ====================

    async def _fetch_media(self, url: str, output_path: str) -> str:
        """Download (or copy a local) media file into the task directory."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not url.startswith(("http://", "https://")):
            # Local file (e.g. generated in-process by Wan2GP) — just copy it
            shutil.copyfile(url, output_path)
            return output_path

        timeout = httpx.Timeout(connect=10.0, read=60, write=60, pool=60)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
        return output_path

    async def _get_audio_duration(self, audio_path: str) -> float:
        """Audio duration in seconds (same fallback logic as FrameProcessor)."""
        try:
            import ffmpeg
            probe = ffmpeg.probe(audio_path)
            return float(probe["format"]["duration"])
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}, using estimate")
            file_size = os.path.getsize(audio_path)
            return max(1.0, file_size / 2000)

    async def _get_video_duration(self, video_path: str) -> float:
        """Video duration in seconds."""
        try:
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            return float(probe["format"]["duration"])
        except Exception as e:
            logger.warning(f"Failed to get video duration: {e}")
            return 1.0
