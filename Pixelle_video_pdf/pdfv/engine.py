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
PDF → Video Engine

Extends the scene-by-scene engine (../Pixelle_video_scene_by_scene/sbs, kept
unchanged) with the PDF-specific stages; everything downstream of the visual
prompts — project creation, per-scene audio / media / segment generation,
final composition, history persistence — is inherited as-is:

    1. ingest_pdf()              -> PdfDocument        (extraction + cleaning)
    2. digest_document()         -> DocumentDigest     (map-reduce LLM analysis)
    3. generate_pdf_script()     -> title + narrations (grounded in the digest)
    4. generate_visual_prompts() -> per-scene prompts  (shared visual world)
    5. create_project() + per-scene steps + compose_final()   [inherited]

The map-reduce in step 2 makes the pipeline work for PDFs far larger than an
LLM context window: each chunk is reduced to structured notes (key points +
faithful facts/quotes), then a single reduce pass builds the "video digest"
the scriptwriter consumes.
"""

import json
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from loguru import logger

from Pixelle_video.pixelle_video.models.storyboard import Storyboard
from Pixelle_video.pixelle_video.utils.content_generators import (
    _default_title_length,
    _parse_json,
)
from Pixelle_video.pixelle_video.utils.prompt_helper import build_image_prompt

from sbs.engine import SceneBySceneEngine

from pdfv.models import DocumentDigest, PdfDocument
from pdfv.pdf_ingest import chunk_document, load_pdf
from pdfv.prompts import (
    build_chunk_digest_prompt,
    build_document_digest_prompt,
    build_pdf_script_prompt,
    build_pdf_visual_prompt_prompt,
)


class PdfVideoEngine(SceneBySceneEngine):
    """PDF → video orchestration on top of the scene-by-scene engine."""

    # ==================== Step 1: Ingest ====================

    def ingest_pdf(
        self,
        path: str,
        page_range: Optional[Tuple[int, int]] = None,
        max_pages: Optional[int] = None,
        password: Optional[str] = None,
    ) -> PdfDocument:
        """Extract + clean the PDF text (no LLM involved)."""
        return load_pdf(path, page_range=page_range, max_pages=max_pages, password=password)

    # ==================== Step 2: Digest (map-reduce) ====================

    async def digest_document(
        self,
        doc: PdfDocument,
        focus: Optional[str] = None,
        target_chunk_chars: int = 12000,
        max_chunks: int = 12,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> DocumentDigest:
        """
        Build the "video digest" of the document.

        Small documents (one chunk) are digested in a single LLM call on the
        raw text. Larger ones are map-reduced: per-chunk structured notes
        (key points + faithful facts/quotes), then one reduce pass.

        Args:
            doc: Ingested PdfDocument.
            focus: Optional user instruction (e.g. "focus on the experiments",
                "chỉ tập trung vào chương 3").
            target_chunk_chars: Chunk size for the map step.
            max_chunks: Cap on analyzed chunks for very large documents —
                beyond it, evenly spaced chunks (always keeping the first and
                last) are analyzed and the rest skipped.
            progress_callback: Optional callback(done, total, message).

        Returns:
            DocumentDigest
        """
        if not doc.full_text.strip():
            raise ValueError(
                "No text extracted from this PDF (probably scanned images). "
                "Run OCR first, e.g. `ocrmypdf input.pdf output.pdf`."
            )

        chunks = chunk_document(doc, target_chars=target_chunk_chars)

        # Cap very large documents: keep first + last, sample the middle evenly
        if len(chunks) > max_chunks:
            step = (len(chunks) - 1) / (max_chunks - 1)
            keep_idx = sorted({round(i * step) for i in range(max_chunks)})
            skipped = len(chunks) - len(keep_idx)
            chunks = [chunks[i] for i in keep_idx]
            logger.warning(
                f"Document is large: analyzing {len(chunks)} of {len(chunks) + skipped} chunks "
                f"(evenly sampled; raise max_chunks or set page_range for full coverage)"
            )

        # --- Map: chunk -> structured notes ---------------------------------
        if len(chunks) == 1:
            # Small document: hand the raw text straight to the reduce step
            notes = [{"pages": chunks[0].page_label, "excerpt": chunks[0].text}]
        else:
            notes = []
            for i, chunk in enumerate(chunks):
                if progress_callback:
                    progress_callback(i, len(chunks), f"Analyzing {chunk.page_label}")
                prompt = build_chunk_digest_prompt(
                    chunk_text=chunk.text,
                    page_label=chunk.page_label,
                    chunk_index=i + 1,
                    chunk_total=len(chunks),
                )
                response = await self.core.llm(prompt, temperature=0.3, max_tokens=4096)
                try:
                    summary = _parse_json(response)
                except json.JSONDecodeError:
                    logger.warning(f"Chunk {chunk.page_label}: notes were not valid JSON, keeping raw text")
                    summary = {"key_points": [response.strip()[:2000]]}
                notes.append({"pages": chunk.page_label, **summary})
                logger.info(f"🧩 Notes ready for {chunk.page_label} ({i + 1}/{len(chunks)})")

        if progress_callback:
            progress_callback(len(chunks), len(chunks), "Building the video digest")

        # --- Reduce: notes -> video digest -----------------------------------
        digest_prompt = build_document_digest_prompt(
            doc_info=doc.brief(),
            notes_json=json.dumps(notes, ensure_ascii=False, indent=2),
            focus=focus,
        )
        response = await self.core.llm(digest_prompt, temperature=0.4, max_tokens=4096)
        digest = DocumentDigest.from_llm_dict(_parse_json(response))

        if not digest.title:
            digest.title = doc.title_guess
        if not digest.key_insights:
            raise ValueError("Digest contains no key insights — the LLM response was unusable, try again")

        logger.info(
            f"🧠 Digest ready: '{digest.title}' [{digest.doc_type} · {digest.language}], "
            f"{len(digest.key_insights)} insight(s), visual world: {digest.visual_world[:60]}..."
        )
        return digest

    # ==================== Step 3: Script ====================

    async def generate_pdf_script(
        self,
        digest: DocumentDigest,
        n_scenes: int = 5,
        focus: Optional[str] = None,
        language: Optional[str] = None,
        min_narration_words: int = 5,
        max_narration_words: int = 20,
        max_retries: int = 3,
    ) -> Tuple[str, List[str]]:
        """
        Write the video title + scene narrations from the digest.

        Args:
            digest: DocumentDigest from digest_document().
            n_scenes: Number of scenes.
            focus: Optional extra instruction for the scriptwriter.
            language: Override the narration language (default: the
                document's own language from the digest).
            min_narration_words / max_narration_words: Per-narration length.
            max_retries: Retries when the LLM returns a wrong scene count
                or invalid JSON.

        Returns:
            (title, narrations)
        """
        language_requirement = (language or digest.language or "the document's language").strip()
        title_max_chars = _default_title_length(digest.core_message or digest.title)

        prompt = build_pdf_script_prompt(
            digest_context=digest.to_context(),
            n_storyboard=n_scenes,
            min_words=min_narration_words,
            max_words=max_narration_words,
            title_max_chars=title_max_chars,
            language_requirement=language_requirement,
            focus=focus,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.core.llm(prompt, temperature=0.8, max_tokens=4096)
                result = _parse_json(response)
                narrations = result.get("narrations") or []
                if len(narrations) > n_scenes:
                    logger.warning(f"Got {len(narrations)} narrations, taking first {n_scenes}")
                    narrations = narrations[:n_scenes]
                elif len(narrations) < n_scenes:
                    raise ValueError(f"Expected {n_scenes} narrations, got {len(narrations)}")

                title = str(result.get("title") or digest.title).strip().strip('"').strip("'")
                if len(title) > title_max_chars:
                    truncated = title[:title_max_chars]
                    last_space = truncated.rfind(" ")
                    title = (truncated[:last_space] if last_space > title_max_chars * 0.6
                             else truncated).rstrip(".,!?;:'\"")

                logger.info(f"📝 PDF script ready: title='{title}', {len(narrations)} scene(s)")
                return title, [str(n).strip() for n in narrations]
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning(f"Script attempt {attempt}/{max_retries} failed: {e}")
        raise ValueError(f"Failed to generate the script after {max_retries} attempts: {last_error}")

    # ==================== Step 4: Visual prompts ====================

    async def generate_visual_prompts(
        self,
        narrations: List[str],
        digest: DocumentDigest,
        prompt_prefix: str = "",
        min_words: int = 30,
        max_words: int = 60,
        max_retries: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """
        One digest-aware media prompt per narration: every prompt lives inside
        the digest's `visual_world` so the whole video feels art-directed.
        Returns final prompts with the style prefix already applied
        (the digest-unaware `generate_prompts()` is still inherited).
        """
        prompt = build_pdf_visual_prompt_prompt(
            narrations=narrations,
            visual_world=digest.visual_world,
            tone=digest.tone,
            min_words=min_words,
            max_words=max_words,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.core.llm(prompt, temperature=0.7, max_tokens=8192)
                result = _parse_json(response)
                visual_prompts = result.get("image_prompts") or []
                if len(visual_prompts) != len(narrations):
                    raise ValueError(
                        f"Prompt count mismatch: expected {len(narrations)}, got {len(visual_prompts)}"
                    )
                if progress_callback:
                    progress_callback(len(visual_prompts), len(narrations), "Visual prompts ready")
                logger.info(f"🎨 Generated {len(visual_prompts)} visual prompt(s) in a shared visual world")
                return [build_image_prompt(str(p), prompt_prefix or "") for p in visual_prompts]
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning(f"Visual prompt attempt {attempt}/{max_retries} failed: {e}")
        raise ValueError(f"Failed to generate visual prompts after {max_retries} attempts: {last_error}")

    async def generate_visual_prompt_for(
        self,
        narration: str,
        digest: DocumentDigest,
        prompt_prefix: str = "",
        min_words: int = 30,
        max_words: int = 60,
    ) -> str:
        """Regenerate the visual prompt for a single scene (stays in the visual world)."""
        prompts = await self.generate_visual_prompts(
            [narration],
            digest,
            prompt_prefix=prompt_prefix,
            min_words=min_words,
            max_words=max_words,
        )
        return prompts[0]

    # ==================== Persistence (history label + PDF info) ====================

    async def _persist(self, project, file_size: int):
        """Same persistence as the scene-by-scene engine, labeled as the PDF pipeline."""
        try:
            storyboard: Storyboard = self._to_storyboard(project)

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
                    "pipeline": "pdf_to_video",
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
            logger.error(f"Failed to persist pdf-to-video task data: {e}")
