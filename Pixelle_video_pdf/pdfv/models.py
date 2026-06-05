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
Data models for the PDF → video workflow.

Two new models on top of the scene-by-scene ones (`sbs.models.Scene` /
`SceneProject`, which are reused unchanged for the generation steps):

- `PdfDocument` — the ingested PDF: cleaned per-page text + metadata + TOC.
- `DocumentDigest` — the LLM's structured understanding of the document:
  everything the scriptwriter stage needs (core message, grounded key
  insights, hook ideas, a coherent visual world, tone, language).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ==================== PDF ingestion ====================


@dataclass
class PdfPage:
    """One page of extracted (and cleaned) PDF text."""
    number: int            # 1-based page number in the ORIGINAL file
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass
class PdfChunk:
    """A contiguous slice of the document sized for one LLM analysis call."""
    index: int             # 0-based chunk index
    text: str
    page_start: int        # 1-based, inclusive
    page_end: int          # 1-based, inclusive

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"p.{self.page_start}-{self.page_end}"


@dataclass
class PdfDocument:
    """The ingested PDF document."""
    path: str
    pages: List[PdfPage]
    metadata: Dict[str, str] = field(default_factory=dict)   # title/author/... (only non-empty values)
    toc: List[Tuple[int, str, int]] = field(default_factory=list)   # (level, title, page)
    n_pages_total: int = 0          # page count of the original file (before page_range)
    extractor: str = "pymupdf"      # which backend extracted the text

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def n_chars(self) -> int:
        return sum(p.n_chars for p in self.pages)

    @property
    def page_range(self) -> Tuple[int, int]:
        if not self.pages:
            return (0, 0)
        return (self.pages[0].number, self.pages[-1].number)

    @property
    def title_guess(self) -> str:
        """Best available title without calling the LLM (metadata, else file name)."""
        title = (self.metadata.get("title") or "").strip()
        if title:
            return title
        import os
        return os.path.splitext(os.path.basename(self.path))[0].replace("_", " ").replace("-", " ").strip()

    def brief(self, max_toc_entries: int = 15) -> str:
        """Compact factual summary embedded in downstream LLM prompts."""
        lines = [f"- File name: {self.path.rsplit('/', 1)[-1]}"]
        if self.metadata.get("title"):
            lines.append(f"- PDF metadata title: {self.metadata['title']}")
        if self.metadata.get("author"):
            lines.append(f"- PDF metadata author: {self.metadata['author']}")
        first, last = self.page_range
        lines.append(f"- Pages used: {first}-{last} (of {self.n_pages_total} in the file)")
        if self.toc:
            entries = [t for t in self.toc if t[0] <= 2][:max_toc_entries]
            if entries:
                lines.append("- Table of contents (excerpt):")
                for level, title, page in entries:
                    lines.append(f"  {'  ' * (level - 1)}* {title} (p.{page})")
        return "\n".join(lines)

    def describe(self) -> str:
        """Human-readable summary for printing in notebooks/UIs."""
        first, last = self.page_range
        lines = [
            f"📄 {self.title_guess}",
            f"   {self.path}",
            f"   pages {first}-{last} of {self.n_pages_total} | "
            f"{self.n_chars:,} chars extracted | backend: {self.extractor}",
        ]
        if self.metadata.get("author"):
            lines.append(f"   author: {self.metadata['author']}")
        if self.toc:
            lines.append(f"   TOC: {len(self.toc)} entries")
        return "\n".join(lines)


# ==================== Document digest ====================


@dataclass
class KeyInsight:
    """One video-worthy idea plus the concrete document fact that backs it."""
    insight: str
    grounding: str = ""


@dataclass
class DocumentDigest:
    """
    The LLM's structured understanding of the document — the single source of
    truth for the script and visual-prompt stages. Everything in it must be
    grounded in the document (the prompts enforce this).
    """
    title: str = ""
    language: str = ""              # e.g. "English", "Vietnamese", "Chinese"
    doc_type: str = "other"         # research paper | book / book chapter | report | ...
    audience: str = ""
    core_message: str = ""
    key_insights: List[KeyInsight] = field(default_factory=list)
    hook_ideas: List[str] = field(default_factory=list)
    visual_world: str = ""          # ONE consistent visual setting/motif (English)
    tone: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)   # original LLM dict (for persistence)

    @classmethod
    def from_llm_dict(cls, data: Dict[str, Any]) -> "DocumentDigest":
        insights = []
        for item in data.get("key_insights", []) or []:
            if isinstance(item, dict):
                insights.append(KeyInsight(
                    insight=str(item.get("insight", "")).strip(),
                    grounding=str(item.get("grounding", "")).strip(),
                ))
            elif isinstance(item, str) and item.strip():
                insights.append(KeyInsight(insight=item.strip()))
        return cls(
            title=str(data.get("title", "")).strip(),
            language=str(data.get("language", "")).strip(),
            doc_type=str(data.get("doc_type", "other")).strip() or "other",
            audience=str(data.get("audience", "")).strip(),
            core_message=str(data.get("core_message", "")).strip(),
            key_insights=[i for i in insights if i.insight],
            hook_ideas=[str(h).strip() for h in (data.get("hook_ideas") or []) if str(h).strip()],
            visual_world=str(data.get("visual_world", "")).strip(),
            tone=str(data.get("tone", "")).strip(),
            raw=dict(data),
        )

    def to_context(self) -> str:
        """Compact text block embedded in the script / visual prompts."""
        lines = [
            f"Document title: {self.title}",
            f"Document type: {self.doc_type}",
            f"Document language: {self.language}",
            f"Target audience: {self.audience}",
            f"Core message: {self.core_message}",
            "Key insights (with their grounding in the document):",
        ]
        for i, ki in enumerate(self.key_insights, 1):
            lines.append(f"  {i}. {ki.insight}")
            if ki.grounding:
                lines.append(f"     grounded in: {ki.grounding}")
        if self.hook_ideas:
            lines.append("Possible hooks:")
            for h in self.hook_ideas:
                lines.append(f"  - {h}")
        if self.visual_world:
            lines.append(f"Visual world: {self.visual_world}")
        if self.tone:
            lines.append(f"Tone: {self.tone}")
        return "\n".join(lines)

    def describe(self) -> str:
        """Human-readable summary for printing in notebooks/UIs."""
        lines = [
            f"🧠 {self.title}  [{self.doc_type} · {self.language}]",
            f"   audience: {self.audience}",
            f"   core message: {self.core_message}",
            "   key insights:",
        ]
        for i, ki in enumerate(self.key_insights, 1):
            lines.append(f"     {i}. {ki.insight}")
            if ki.grounding:
                lines.append(f"        ⚓ {ki.grounding}")
        if self.hook_ideas:
            lines.append("   hooks:")
            for h in self.hook_ideas:
                lines.append(f"     - {h}")
        lines.append(f"   visual world: {self.visual_world}")
        lines.append(f"   tone: {self.tone}")
        return "\n".join(lines)
