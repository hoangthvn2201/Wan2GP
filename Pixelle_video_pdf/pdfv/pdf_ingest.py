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
PDF ingestion: extraction, cleaning and chunking.

Extraction uses PyMuPDF (fitz) when available — it is fast and preserves the
table of contents — with a pypdf fallback so the pipeline still works in
environments where only pypdf is installed.

Cleaning is tuned for LLM consumption, not for typography:
- running headers/footers (lines repeated at the top/bottom of most pages,
  with digits normalized so "Page 12" == "Page 13") are dropped,
- hyphenation across line breaks is repaired ("under-\\nstand" → "understand"),
- whitespace is normalized while paragraph breaks are preserved.
"""

import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from loguru import logger

from pdfv.models import PdfChunk, PdfDocument, PdfPage


# ==================== Extraction backends ====================


def _extract_with_pymupdf(path: str, password: Optional[str]) -> Tuple[List[str], Dict, List, int]:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        if doc.needs_pass:
            if not doc.authenticate(password or ""):
                raise ValueError(f"PDF is password-protected and the password did not work: {path}")
        metadata = {k: v.strip() for k, v in (doc.metadata or {}).items()
                    if isinstance(v, str) and v.strip()}
        toc = [(int(lvl), str(title).strip(), int(page)) for lvl, title, page in (doc.get_toc() or [])]
        texts = [page.get_text("text") for page in doc]
        return texts, metadata, toc, doc.page_count
    finally:
        doc.close()


def _extract_with_pypdf(path: str, password: Optional[str]) -> Tuple[List[str], Dict, List, int]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted:
        if not reader.decrypt(password or ""):
            raise ValueError(f"PDF is password-protected and the password did not work: {path}")
    meta = reader.metadata
    metadata = {}
    if meta:
        for key, attr in (("title", "title"), ("author", "author"), ("subject", "subject")):
            value = getattr(meta, attr, None)
            if value and str(value).strip():
                metadata[key] = str(value).strip()
    texts = [(page.extract_text() or "") for page in reader.pages]
    return texts, metadata, [], len(reader.pages)


# ==================== Cleaning ====================

_HF_SCAN_LINES = 2          # lines inspected at the top and bottom of each page
_HF_MIN_PAGES = 3           # a header/footer must appear on at least this many pages
_HF_MIN_RATIO = 0.6         # ... and on at least this fraction of pages


def _normalize_hf_line(line: str) -> str:
    """Normalize a candidate header/footer line so page numbers don't differ."""
    return re.sub(r"\d+", "#", line.strip()).lower()


def _find_repeated_edges(page_lines: List[List[str]]) -> set:
    """Normalized lines that repeat at page edges across most pages."""
    counter: Counter = Counter()
    for lines in page_lines:
        edge = lines[:_HF_SCAN_LINES] + lines[-_HF_SCAN_LINES:]
        for line in {_normalize_hf_line(l) for l in edge if l.strip()}:
            counter[line] += 1
    n_pages = len(page_lines)
    threshold = max(_HF_MIN_PAGES, int(n_pages * _HF_MIN_RATIO))
    if n_pages < _HF_MIN_PAGES:
        return set()
    return {line for line, count in counter.items() if count >= threshold}


def _clean_page_text(text: str, repeated_edges: set) -> str:
    lines = text.splitlines()

    # Drop running headers/footers (only at the page edges, where they live)
    def is_repeated(idx: int) -> bool:
        near_edge = idx < _HF_SCAN_LINES or idx >= len(lines) - _HF_SCAN_LINES
        return near_edge and _normalize_hf_line(lines[idx]) in repeated_edges

    kept = [line for idx, line in enumerate(lines) if not is_repeated(idx)]
    text = "\n".join(kept)

    # Repair hyphenation across line breaks ("under-\nstand" -> "understand")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Normalize whitespace, keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ==================== Public API ====================


def load_pdf(
    path: str,
    page_range: Optional[Tuple[int, int]] = None,
    max_pages: Optional[int] = None,
    password: Optional[str] = None,
) -> PdfDocument:
    """
    Extract and clean the text of a PDF.

    Args:
        path: Path to the PDF file.
        page_range: Optional (first, last) 1-based inclusive page selection.
        max_pages: Optional hard cap on the number of pages (applied after
            page_range), to keep very large documents manageable.
        password: Password for encrypted PDFs.

    Returns:
        PdfDocument with cleaned per-page text, metadata and TOC.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        texts, metadata, toc, n_total = _extract_with_pymupdf(path, password)
        extractor = "pymupdf"
    except ImportError:
        logger.warning("PyMuPDF not installed, falling back to pypdf (no TOC, slower)")
        texts, metadata, toc, n_total = _extract_with_pypdf(path, password)
        extractor = "pypdf"

    # Page selection (1-based inclusive)
    first, last = 1, n_total
    if page_range:
        first = max(1, int(page_range[0]))
        last = min(n_total, int(page_range[1]))
        if first > last:
            raise ValueError(f"Invalid page_range {page_range} for a {n_total}-page PDF")
    numbers = list(range(first, last + 1))
    selected = texts[first - 1: last]
    if max_pages and len(selected) > max_pages:
        logger.warning(f"Capping extraction to the first {max_pages} of {len(selected)} selected pages")
        selected = selected[:max_pages]
        numbers = numbers[:max_pages]

    # Header/footer detection works best on the full selection
    page_lines = [t.splitlines() for t in selected]
    repeated_edges = _find_repeated_edges(page_lines)
    if repeated_edges:
        logger.info(f"Dropping {len(repeated_edges)} repeated header/footer line(s)")

    pages = [
        PdfPage(number=num, text=_clean_page_text(text, repeated_edges))
        for num, text in zip(numbers, selected)
    ]
    non_empty = sum(1 for p in pages if p.text)
    doc = PdfDocument(
        path=os.path.abspath(path),
        pages=pages,
        metadata=metadata,
        toc=toc,
        n_pages_total=n_total,
        extractor=extractor,
    )
    logger.info(
        f"📄 Ingested '{doc.title_guess}': {doc.n_pages} page(s) "
        f"({non_empty} with text, {doc.n_chars:,} chars) via {extractor}"
    )
    if doc.n_chars < 200:
        logger.warning(
            "Very little text extracted — this PDF is probably scanned images. "
            "Run OCR on it first (e.g. `ocrmypdf in.pdf out.pdf`), then ingest again."
        )
    return doc


def chunk_document(
    doc: PdfDocument,
    target_chars: int = 12000,
) -> List[PdfChunk]:
    """
    Split the document into analysis chunks of roughly `target_chars`
    characters, respecting page boundaries (a page never spans two chunks,
    except single pages larger than the target, which are split on
    paragraph boundaries).
    """
    chunks: List[PdfChunk] = []
    buf: List[str] = []
    buf_chars = 0
    buf_start: Optional[int] = None
    buf_end: Optional[int] = None

    def flush():
        nonlocal buf, buf_chars, buf_start, buf_end
        if buf and buf_chars > 0:
            chunks.append(PdfChunk(
                index=len(chunks),
                text="\n\n".join(buf).strip(),
                page_start=buf_start,
                page_end=buf_end,
            ))
        buf, buf_chars, buf_start, buf_end = [], 0, None, None

    def add_piece(text: str, page: int):
        nonlocal buf_chars, buf_start, buf_end
        if buf_chars and buf_chars + len(text) > target_chars:
            flush()
        buf.append(text)
        buf_chars += len(text)
        buf_start = page if buf_start is None else buf_start
        buf_end = page

    for page in doc.pages:
        if not page.text:
            continue
        if len(page.text) <= target_chars:
            add_piece(page.text, page.number)
        else:
            # Oversized page: split on paragraph boundaries
            paragraphs = re.split(r"\n\s*\n", page.text)
            piece, piece_chars = [], 0
            for para in paragraphs:
                if piece_chars and piece_chars + len(para) > target_chars:
                    add_piece("\n\n".join(piece), page.number)
                    piece, piece_chars = [], 0
                piece.append(para)
                piece_chars += len(para)
            if piece:
                add_piece("\n\n".join(piece), page.number)
    flush()

    logger.info(f"Split document into {len(chunks)} chunk(s) (~{target_chars} chars each)")
    return chunks
