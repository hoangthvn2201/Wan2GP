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
Flexible Video Engine — LLM-orchestrated generate-or-search per scene.

Extends the scene-by-scene engine (../Pixelle_video_scene_by_scene/sbs, kept
unchanged). Script generation, TTS, AI media generation, segment rendering and
final composition are all inherited; this engine adds the media-sourcing
stages:

    1. generate_scene_plan()      -> per scene: generate vs stock search
    2. search_scene_media()       -> Pexels/Pixabay candidates per scene
    3. rank_candidates()          -> LLM auto-pick (metadata-based)
    4. apply_picked_candidate()   -> download + normalize into canonical assets
       (empty search results fall back to generation when allowed)

Stock media enters the exact same per-scene asset contract the base engine
uses (`<uid>_image.png` / `<uid>_video.mp4`), normalized to the project size
and fps so `compose_final`'s concat-demuxer (`-c copy`) stays safe.
"""

import os
from datetime import datetime
from typing import Callable, List, Optional

from loguru import logger

from Pixelle_video.pixelle_video.utils.content_generators import _parse_json
from Pixelle_video.pixelle_video.utils.prompt_helper import build_image_prompt

from sbs.engine import SceneBySceneEngine
from sbs.models import SceneProject

from flexvid.flex_config import FlexConfig, load_flex_config
from flexvid.models import FlexScene, MediaCandidate
from flexvid.normalize import (
    cleanup_raw,
    normalize_stock_image,
    normalize_stock_video,
    pick_orientation,
    target_media_size,
)
from flexvid.prompts import (
    build_broaden_query_prompt,
    build_candidate_ranking_prompt,
    build_scene_plan_prompt,
)
from flexvid.search import MediaSearchAggregator


class FlexibleVideoEngine(SceneBySceneEngine):
    """Generate-or-search orchestration on top of the scene-by-scene engine."""

    def __init__(self, core, flex_config: Optional[FlexConfig] = None):
        super().__init__(core)
        self.flex_config = flex_config or load_flex_config()
        self.aggregator = MediaSearchAggregator.from_config(self.flex_config)

    # ==================== LLM helper (same pattern as pdfv) ====================

    async def _llm_json(
        self,
        prompt: str,
        *,
        temperature: float,
        base_max_tokens: int,
        max_retries: int = 3,
        label: str = "LLM call",
        validate: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        LLM call that returns parsed JSON, retrying with an ESCALATING token
        budget (base, 2x, 3x, ...) — reasoning models spend hidden
        chain-of-thought tokens against max_tokens and can emit no final
        answer at all on complex prompts; retrying at the same budget just
        fails the same way.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            budget = base_max_tokens * attempt
            try:
                response = await self.core.llm(prompt, temperature=temperature, max_tokens=budget)
                result = _parse_json(response)
                if validate is not None:
                    validate(result)
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{label} attempt {attempt}/{max_retries} (max_tokens={budget}) failed: {e}"
                )
        raise ValueError(f"{label} failed after {max_retries} attempts: {last_error}")

    # ==================== Project creation ====================

    def create_project(self, title, narrations, prompts, params) -> SceneProject:
        """
        Same project scaffolding as the base engine (single source of truth
        for TTS / media-mode handling), with every scene upgraded to a
        FlexScene so the media plan can live on it.
        """
        project = super().create_project(title, narrations, prompts, params)
        project.scenes = [FlexScene.from_scene(s) for s in project.scenes]
        return project

    # ==================== Step: Media plan ====================

    def _media_capability(self, project: SceneProject) -> str:
        """'video' when scenes end up as clips, else 'image'."""
        return "video" if project.is_video_workflow else "image"

    def _generate_only(self, project: SceneProject) -> bool:
        """
        Generate-only mode: stock search is never used at all — every scene is
        AI-generated. Per-project override in params, falling back to
        flex_config. Takes precedence over stock-only.
        """
        flag = project.params.get("generate_only")
        if flag is None:
            flag = self.flex_config.generate_only
        return bool(flag)

    def _search_only(self, project: SceneProject) -> bool:
        """
        Stock-only mode: every scene searches, AI generation is never used
        (so no generation model is ever loaded). Per-project override in
        params, falling back to flex_config; meaningless without providers
        and overridden by generate-only.
        """
        if self._generate_only(project):
            return False
        flag = project.params.get("search_only")
        if flag is None:
            flag = self.flex_config.search_only
        return bool(flag) and self.aggregator.enabled

    def _apply_plan_entry(
        self,
        project: SceneProject,
        scene: FlexScene,
        entry: dict,
        prompt_prefix: str = "",
        search_only: bool = False,
        generate_only: bool = False,
    ):
        """Map one LLM plan entry onto a scene (capability rules enforced)."""
        source = str(entry.get("source") or "generate").lower()
        if source == "search" and (generate_only or not self.aggregator.enabled):
            source = "generate"
        elif search_only and str(entry.get("search_query") or "").strip():
            source = "search"

        media_type = str(entry.get("media_type") or "image").lower()
        if self._media_capability(project) == "image":
            media_type = "image"
        elif media_type not in ("image", "video"):
            media_type = "video"

        scene.source = source
        scene.plan_media_type = media_type
        scene.plan_reason = str(entry.get("reason") or "")
        scene.invalidate_search()
        scene.invalidate_media()

        if source == "search":
            scene.search_query = str(entry.get("search_query") or "").strip() or None
            scene.prompt = None
        else:
            scene.search_query = None
            gen_prompt = str(entry.get("gen_prompt") or "").strip()
            scene.prompt = build_image_prompt(gen_prompt, prompt_prefix or "") if gen_prompt else None

    async def generate_scene_plan(
        self,
        project: SceneProject,
        prompt_prefix: str = "",
        min_words: int = 30,
        max_words: int = 60,
    ) -> List[FlexScene]:
        """
        LLM media plan for ALL scenes: per scene source (generate | search),
        media type, and the matching search query or generation prompt.
        """
        narrations = [s.narration for s in project.scenes]
        search_only = self._search_only(project)
        generate_only = self._generate_only(project)
        prompt = build_scene_plan_prompt(
            narrations,
            title=project.title,
            media_capability=self._media_capability(project),
            # Generate-only reuses the "no providers -> all generate" prompt rule
            providers_enabled=self.aggregator.enabled and not generate_only,
            min_words=min_words,
            max_words=max_words,
            search_only=search_only,
        )

        def _check(result: dict):
            scenes = result.get("scenes") or []
            if len(scenes) != len(narrations):
                raise ValueError(f"Expected {len(narrations)} plan entries, got {len(scenes)}")
            for i, entry in enumerate(scenes):
                source = str(entry.get("source") or "").lower()
                query = str(entry.get("search_query") or "").strip()
                gen_prompt = str(entry.get("gen_prompt") or "").strip()
                if generate_only:
                    if not gen_prompt:
                        raise ValueError(f"Plan entry {i} needs a gen_prompt")
                elif search_only:
                    if not query:
                        raise ValueError(f"Plan entry {i} needs a search_query")
                elif source == "search" and not query:
                    raise ValueError(f"Plan entry {i} is 'search' but has no search_query")
                elif source == "generate" and not gen_prompt:
                    raise ValueError(f"Plan entry {i} is 'generate' but has no gen_prompt")

        result = await self._llm_json(
            prompt, temperature=0.7, base_max_tokens=8192,
            label="Scene media plan", validate=_check,
        )

        for scene, entry in zip(project.scenes, result["scenes"]):
            self._apply_plan_entry(project, scene, entry, prompt_prefix,
                                   search_only=search_only,
                                   generate_only=generate_only)

        n_search = sum(1 for s in project.scenes if s.source == "search")
        logger.info(
            f"🧭 Media plan ready: {n_search} search / "
            f"{len(project.scenes) - n_search} generate scene(s)"
        )
        return project.scenes

    async def regenerate_plan_for(
        self,
        project: SceneProject,
        scene: FlexScene,
        prompt_prefix: str = "",
        min_words: int = 30,
        max_words: int = 60,
    ) -> FlexScene:
        """Re-plan the media sourcing of a single scene."""
        search_only = self._search_only(project)
        generate_only = self._generate_only(project)
        prompt = build_scene_plan_prompt(
            [scene.narration],
            title=project.title,
            media_capability=self._media_capability(project),
            providers_enabled=self.aggregator.enabled and not generate_only,
            min_words=min_words,
            max_words=max_words,
            search_only=search_only,
        )

        def _check(result: dict):
            if len(result.get("scenes") or []) != 1:
                raise ValueError("Expected exactly 1 plan entry")
            entry = result["scenes"][0]
            if generate_only and not str(entry.get("gen_prompt") or "").strip():
                raise ValueError("Plan entry needs a gen_prompt")
            if search_only and not str(entry.get("search_query") or "").strip():
                raise ValueError("Plan entry needs a search_query")

        result = await self._llm_json(
            prompt, temperature=0.8, base_max_tokens=4096,
            label="Scene media re-plan", validate=_check,
        )
        self._apply_plan_entry(project, scene, result["scenes"][0], prompt_prefix,
                               search_only=search_only, generate_only=generate_only)
        return scene

    # ==================== Step: Stock search ====================

    def _project_orientation(self, project: SceneProject) -> Optional[str]:
        width, height = target_media_size(project.config)
        return pick_orientation(width, height)

    async def search_scene_media(
        self,
        project: SceneProject,
        scene: FlexScene,
        index: int,
        prompt_prefix: str = "",
    ) -> FlexScene:
        """
        Query the stock providers for this scene and auto-pick the best
        candidate. Empty results first retry ONCE with an LLM-broadened query;
        if still empty, fall back to generation (when allowed and the project
        is not stock-only) — the scene flips to source=generate and gets a
        generation prompt.
        """
        if not scene.search_query:
            raise ValueError("Scene has no search query — run the media plan first")
        if not self.aggregator.enabled:
            raise RuntimeError("No stock providers configured (see flex_config.yaml)")
        if self._generate_only(project):
            raise RuntimeError("Stock search is disabled for this project (generate-only)")

        search_kwargs = dict(
            media_type=scene.plan_media_type,
            n_total=self.flex_config.candidates_per_scene,
            orientation=self._project_orientation(project),
            min_width=self.flex_config.min_resolution,
        )
        candidates = await self.aggregator.search(scene.search_query, **search_kwargs)

        if not candidates:
            # One broadened retry before considering any fallback
            broadened = await self._broaden_search_query(scene)
            if broadened and broadened != scene.search_query:
                logger.info(f"Scene {index + 1}: retrying with broader query '{broadened}'")
                candidates = await self.aggregator.search(broadened, **search_kwargs)
                if candidates:
                    scene.search_query = broadened   # show what actually worked

        scene.candidates = candidates
        scene.picked_candidate_id = None
        scene.search_attempted = True

        if not candidates:
            logger.warning(f"Scene {index + 1}: no stock results for '{scene.search_query}'")
            if self.flex_config.allow_fallback and not self._search_only(project):
                await self._fall_back_to_generate(scene, prompt_prefix)
            return scene

        await self.rank_candidates(project, scene)
        return scene

    async def _broaden_search_query(self, scene: FlexScene) -> Optional[str]:
        """LLM rewrite of a failed query into broader stock keywords (best effort)."""
        try:
            result = await self._llm_json(
                build_broaden_query_prompt(scene.narration, scene.search_query or ""),
                temperature=0.7, base_max_tokens=1024, max_retries=2,
                label="Broaden search query",
            )
            return str(result.get("search_query") or "").strip() or None
        except ValueError as e:
            logger.warning(f"Query broadening failed: {e}")
            return None

    async def _fall_back_to_generate(self, scene: FlexScene, prompt_prefix: str = ""):
        """Search came up empty -> the scene generates its media instead."""
        scene.fell_back_to_generate = True
        if not (scene.prompt or "").strip():
            scene.prompt = await self.generate_prompt_for(
                scene.narration, prompt_prefix=prompt_prefix
            )
        logger.info(f"↩️ Scene falls back to generation (prompt: {scene.prompt[:60]}...)")

    # ==================== Step: Candidate ranking ====================

    async def rank_candidates(self, project: SceneProject, scene: FlexScene) -> FlexScene:
        """
        LLM-rank the fetched candidates against the narration (metadata only —
        the core LLM has no vision input) and auto-pick the best. The user can
        override the pick in the UI before materializing. LLM failure falls
        back to a heuristic pick instead of blocking the wizard.
        """
        if not scene.candidates:
            return scene

        orientation = self._project_orientation(project)
        prompt = build_candidate_ranking_prompt(
            scene.narration,
            scene.search_query or "",
            scene.candidates,
            orientation=orientation,
        )

        def _check(result: dict):
            best = result.get("best_index")
            if not isinstance(best, int) or not (0 <= best < len(scene.candidates)):
                raise ValueError(f"best_index out of range: {best}")

        try:
            result = await self._llm_json(
                prompt, temperature=0.2, base_max_tokens=4096,
                max_retries=2, label="Candidate ranking", validate=_check,
            )
            best = scene.candidates[result["best_index"]]
        except ValueError as e:
            logger.warning(f"Candidate ranking failed ({e}); picking heuristically")
            best = self._heuristic_pick(scene.candidates, orientation)

        scene.picked_candidate_id = best.id
        logger.info(f"🏆 Auto-picked candidate {best.id} ({best.meta_line()})")
        return scene

    @staticmethod
    def _heuristic_pick(candidates: List[MediaCandidate],
                        orientation: Optional[str]) -> MediaCandidate:
        """No-LLM fallback: orientation match first, then highest resolution."""
        matching = [c for c in candidates if orientation and c.orientation == orientation]
        pool = matching or candidates
        return max(pool, key=lambda c: c.width * c.height)

    # ==================== Step: Materialize the pick ====================

    async def apply_picked_candidate(
        self,
        project: SceneProject,
        scene: FlexScene,
        index: int,
    ) -> FlexScene:
        """
        Download the picked candidate and normalize it into the canonical
        scene asset (`<uid>_image.png` / `<uid>_video.mp4`) at the project
        size — videos also re-encoded to the project fps so the final concat
        (demuxer, `-c copy`) stays safe.
        """
        candidate = scene.picked_candidate
        if candidate is None:
            raise ValueError("No candidate picked for this scene")

        width, height = target_media_size(project.config)

        if candidate.media_type == "video":
            output_path = self.scene_asset_path(project, scene, "video")
            raw_path = output_path + ".raw"
            try:
                await self._fetch_media(candidate.download_url, raw_path)
                normalize_stock_video(raw_path, output_path, fps=project.config.video_fps,
                                      width=width, height=height)
            finally:
                cleanup_raw(raw_path)
            scene.video_path = output_path
            scene.image_path = None
            scene.media_type = "video"
        else:
            output_path = self.scene_asset_path(project, scene, "image")
            raw_path = output_path + ".raw"
            try:
                await self._fetch_media(candidate.download_url, raw_path)
                normalize_stock_image(raw_path, output_path, width=width, height=height)
            finally:
                cleanup_raw(raw_path)
            scene.image_path = output_path
            scene.video_path = None
            scene.media_type = "image"

        scene.attribution = candidate.attribution()
        # The clip is visual-only: the segment step pads/trims it to the
        # narration audio, so scene.duration stays narration-driven.
        scene.invalidate_segment()
        logger.info(f"📥 Scene {index + 1} stock media ready: {output_path} "
                    f"(by {candidate.photographer or 'unknown'} on {candidate.source})")
        return scene

    # ==================== Audio (stock clips are not length-synced) ====================

    async def generate_audio(self, project, scene, index: int):
        """
        Same TTS as the base engine, with two adjustments:

        - The base invalidates video clips on (re)generated audio because
          GENERATED clips are length-synced to the narration. Stock clips
          aren't (the segment step pads/trims them), so a still-existing
          stock clip is restored instead of re-downloaded.
        - The narration is loudness-normalized / amplified afterwards (same
          fix as the PDF engine: TTS output is often noticeably quiet and no
          TTS backend exposes a working volume control through the core).
          Controlled by ``tts_normalize`` / ``tts_volume`` in project.params.
        """
        is_stock_video = (
            isinstance(scene, FlexScene) and scene.is_search
            and scene.attribution is not None and scene.media_type == "video"
        )
        video_path_before = scene.video_path
        scene = await super().generate_audio(project, scene, index)
        if (is_stock_video and not scene.video_path
                and video_path_before and os.path.exists(video_path_before)):
            scene.video_path = video_path_before
            scene.media_type = "video"

        gain = float(project.params.get("tts_volume") or 1.0)
        normalize = bool(project.params.get("tts_normalize", True))
        if normalize or gain != 1.0:
            self._boost_narration_audio(scene.audio_path, gain=gain, normalize=normalize)
            # Re-probe: filtering re-encodes the file (duration drives clip length)
            scene.duration = await self._get_audio_duration(scene.audio_path)
        return scene

    @staticmethod
    def _boost_narration_audio(audio_path: str, gain: float = 1.0, normalize: bool = True):
        """Loudness-normalize / amplify a narration file in place (best effort)."""
        import ffmpeg

        filters = []
        if normalize:
            # Single-pass loudnorm: raises quiet speech to -16 LUFS with a
            # -1.5 dBTP true-peak ceiling (no clipping)
            filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        if gain and gain != 1.0:
            filters.append(f"volume={gain}")
            if gain > 1.0:
                filters.append("alimiter=limit=0.97")  # guard the extra gain against clipping
        if not filters:
            return

        tmp_path = audio_path + ".boost.mp3"
        try:
            (
                ffmpeg.input(audio_path)
                # ar=48000: loudnorm upsamples to 192 kHz internally, which the
                # mp3 encoder can't take — resample back down explicitly
                .output(tmp_path, af=",".join(filters), **{"b:a": "192k", "ar": "48000"})
                .overwrite_output()
                .run(quiet=True)
            )
            os.replace(tmp_path, audio_path)
            logger.info(f"🔊 Narration loudness adjusted (normalize={normalize}, gain={gain}): {audio_path}")
        except Exception as e:
            # A boost failure must never lose the narration itself
            logger.warning(f"Narration loudness boost failed, keeping the original audio: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ==================== Per-scene orchestration ====================

    async def process_scene(
        self,
        project: SceneProject,
        scene: FlexScene,
        index: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> FlexScene:
        """
        Run all missing steps for one scene, branching on its media source:

            search:   audio -> search -> rank -> download+normalize -> segment
            generate: audio -> media (or i2v two-step) -> segment
        """
        def _report(stage: str):
            if progress_callback:
                progress_callback(stage)

        if not scene.audio_path:
            _report("audio")
            await self.generate_audio(project, scene, index)

        prompt_prefix = project.params.get("prompt_prefix") or ""
        if not project.params.get("use_prompt_prefix", True):
            prompt_prefix = ""

        is_flex = isinstance(scene, FlexScene)

        # --- Search branch -------------------------------------------------
        if is_flex and scene.is_search and project.needs_media and not scene.has_media:
            if not scene.search_attempted:
                _report("search")
                await self.search_scene_media(project, scene, index, prompt_prefix)
            if scene.is_search and scene.candidates:   # may have fallen back above
                if not scene.picked_candidate_id:
                    _report("rank")
                    await self.rank_candidates(project, scene)
                _report("download")
                await self.apply_picked_candidate(project, scene, index)
            elif (scene.is_search and self.flex_config.allow_fallback
                  and not self._search_only(project)):
                # attempted earlier but empty and not yet flipped
                await self._fall_back_to_generate(scene, prompt_prefix)
            elif scene.is_search:
                # stock-only (or fallback disabled) and nothing found: fail
                # clearly instead of rendering an empty frame
                raise RuntimeError(
                    f"No stock media found for '{scene.search_query}' and AI "
                    f"generation is disabled for this project — edit the "
                    f"scene's search keywords and retry"
                )

        # --- Generate branch (incl. search fallback) ------------------------
        if project.needs_media and not scene.has_media and not (is_flex and scene.is_search):
            if not (scene.prompt or "").strip():
                _report("media")
                scene.prompt = await self.generate_prompt_for(
                    scene.narration, prompt_prefix=prompt_prefix
                )
            if project.is_i2v:
                if not scene.image_path:
                    _report("image_start")
                    await self.generate_start_image(project, scene, index)
                if not scene.video_path:
                    _report("animate")
                    await self.animate_image(project, scene, index)
            else:
                _report("media")
                await self.generate_media(project, scene, index)

        if not scene.segment_path:
            _report("segment")
            await self.render_segment(project, scene, index)
        return scene

    # ==================== Persistence (history label + attribution) ====================

    async def _persist(self, project: SceneProject, file_size: int):
        """Base persistence, labeled as the flexible pipeline + licensing info."""
        try:
            storyboard = self._to_storyboard(project)

            input_params = dict(project.params)
            input_params.setdefault("title", project.title)
            # Per-scene sourcing + attribution (licensing duty for stock media)
            input_params["scene_sources"] = [
                {
                    "narration": s.narration,
                    "source": getattr(s, "source", "generate"),
                    "search_query": getattr(s, "search_query", None),
                    "fell_back_to_generate": getattr(s, "fell_back_to_generate", False),
                    "attribution": getattr(s, "attribution", None),
                }
                for s in project.scenes
            ]

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
                    "pipeline": "flexible_video",
                    "llm_model": self.core.config.get("llm", {}).get("model", "unknown"),
                    "llm_base_url": self.core.config.get("llm", {}).get("base_url", "unknown"),
                    "comfyui_url": self.core.config.get("comfyui", {}).get("comfyui_url", "unknown"),
                    "runninghub_enabled": bool(
                        self.core.config.get("comfyui", {}).get("runninghub_api_key")
                    ),
                    "stock_providers": self.aggregator.provider_names,
                },
            }

            await self.core.persistence.save_task_metadata(project.task_id, metadata)
            await self.core.persistence.save_storyboard(project.task_id, storyboard)
            logger.info(f"💾 Saved task metadata + storyboard: {project.task_id}")
        except Exception as e:
            # Persistence failure shouldn't break the generated video
            logger.error(f"Failed to persist flexible-video task data: {e}")
