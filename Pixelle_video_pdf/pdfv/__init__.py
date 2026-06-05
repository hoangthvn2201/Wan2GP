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

"""PDF → Video pipeline on top of the Pixelle-Video core + scene-by-scene engine."""

from pdfv.engine import PdfVideoEngine
from pdfv.models import DocumentDigest, KeyInsight, PdfChunk, PdfDocument, PdfPage
from pdfv.pdf_ingest import chunk_document, load_pdf

__all__ = [
    "PdfVideoEngine",
    "PdfDocument",
    "PdfPage",
    "PdfChunk",
    "DocumentDigest",
    "KeyInsight",
    "load_pdf",
    "chunk_document",
]
