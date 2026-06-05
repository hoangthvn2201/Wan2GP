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
LLM prompts for the PDF → video pipeline.

The pipeline differs from topic/content generation in two key ways, and the
prompts are built around them:

1. **Grounding.** A PDF is a real document — viewers expect the video to be
   faithful to it. Every stage carries the rule: facts, numbers, names and
   quotes may only come from the document; chunk notes keep the *grounding*
   (the concrete evidence) attached to every insight so the scriptwriter can
   be vivid without inventing anything.

2. **Visual coherence.** A document-explainer video looks much better when
   all scenes live in one consistent "visual world" (setting / motif /
   palette) chosen to match the document, instead of N unrelated images.
   The digest stage picks that world once; the visual stage applies it to
   every scene prompt.

Prompt-style conventions (role / task / requirements / strict JSON output /
reminders, plus the language-consistency emphasis incl. Vietnamese) follow
`Pixelle_video/pixelle_video/prompts/`.
"""

import json
from typing import List, Optional


# ==================== ① Chunk notes (map step) ====================

CHUNK_DIGEST_PROMPT = """# Role Definition
You are a meticulous document analyst. You distill long documents without losing the concrete substance — numbers, names, results, memorable phrasing.

# Core Task
Below is an excerpt of a larger PDF document (pages {page_label}, excerpt {chunk_index} of {chunk_total}). Reduce it to structured notes. A scriptwriter will later use the notes from ALL excerpts to write a short video about the whole document, so keep exactly what a scriptwriter would need.

# Document Excerpt
{chunk_text}

# Output Requirements
- "key_points": the 3~6 most important ideas of this excerpt, one self-contained sentence each, written in the SAME language as the excerpt
- "facts": concrete numbers, dates, names, results, definitions or findings worth quoting in a video (empty list if none) — copy them faithfully from the excerpt, never invent, never round
- "quotes": at most 2 short verbatim quotes that are striking or memorable (empty list if none)
- "language": the main language of the excerpt (e.g. "English", "Vietnamese", "Chinese")
- Ignore boilerplate: references lists, page furniture, copyright notices, acknowledgements

# Output Format
Strictly output in the following JSON format, do not add any additional text explanations:

```json
{{
  "key_points": ["...", "..."],
  "facts": ["...", "..."],
  "quotes": ["..."],
  "language": "..."
}}
```

# Important Reminders
1. Only output JSON format content, do not add any explanations
2. Ensure JSON format is strictly correct and can be directly parsed by the program
3. Everything must come from the excerpt itself — never add outside knowledge
4. key_points must be in the same language as the excerpt

Now, please produce the notes for this excerpt. Only output JSON, no other content.
"""


def build_chunk_digest_prompt(
    chunk_text: str,
    page_label: str,
    chunk_index: int,
    chunk_total: int,
) -> str:
    """Build the per-chunk analysis (map step) prompt."""
    return CHUNK_DIGEST_PROMPT.format(
        chunk_text=chunk_text,
        page_label=page_label,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
    )


# ==================== ② Document digest (reduce step) ====================

DOCUMENT_DIGEST_PROMPT = """# Role Definition
You are a creative director for short videos who specializes in turning dense documents (research papers, reports, books, slide decks) into videos people actually finish watching. You think in hooks, insights and takeaways — but you never bend the truth of the source document.

# Core Task
You receive (1) basic information about a PDF document and (2) ordered source notes extracted from it. Build the "video digest": everything a scriptwriter needs to write ONE compelling short video about this document.
{focus_block}
# Document Info
{doc_info}

# Source Notes (in document order)
{notes_json}

# Output Requirements
- "title": the document's title — prefer the PDF metadata / table of contents when plausible, otherwise infer it from the notes
- "language": the main language of the document (e.g. "English", "Vietnamese", "Chinese")
- "doc_type": exactly one of "research paper" | "book / book chapter" | "report" | "slide deck" | "manual / guide" | "article / essay" | "other"
- "audience": who would watch a short video about this document, one short phrase, in the document's language
- "core_message": the single most important takeaway of the whole document, 1~2 sentences, in the document's language — the thing a viewer should remember a week later
- "key_insights": 5~8 items ordered for a video (strongest first), each an object:
  * "insight": one video-worthy idea, one sentence, in the document's language — concrete and interesting, not a chapter heading
  * "grounding": the specific fact / number / finding / quote from the notes that backs this insight, copied faithfully (this is what makes the video credible)
- "hook_ideas": 2~3 possible opening hooks for scene 1 (a question, a striking fact, a relatable scenario), each one sentence, in the document's language, each grounded in the notes
- "visual_world": ONE consistent visual setting / motif for the whole video, described in English in 15~30 words. Choose imagery that genuinely fits the document — e.g. a neuroscience paper → "glowing neural pathways drifting through deep blue space"; a personal-finance book → "a warm tidy desk where coins grow into small plants"; a climate report → "a miniature Earth on a table, weather shifting above it". Every scene of the video will live inside this world, so pick something that can express many different ideas.
- "tone": the emotional tone for the narration, 2~4 English words (e.g. "curious and encouraging", "urgent but hopeful")

# Strictness Rules (most important)
- Everything must come from the document info and the notes. NEVER invent facts, numbers, names or quotes.
- If notes disagree, prefer the more specific one.
- Skip meta-content (acknowledgements, references, license text) — viewers don't care.

# Output Format
Strictly output in the following JSON format, do not add any additional text explanations:

```json
{{
  "title": "...",
  "language": "...",
  "doc_type": "...",
  "audience": "...",
  "core_message": "...",
  "key_insights": [
    {{"insight": "...", "grounding": "..."}}
  ],
  "hook_ideas": ["...", "..."],
  "visual_world": "...",
  "tone": "..."
}}
```

# Important Reminders
1. Only output JSON format content, do not add any explanations
2. Ensure JSON format is strictly correct and can be directly parsed by the program
3. core_message / key_insights / hook_ideas / audience must be in the document's language; visual_world and tone must be in English
4. Every insight needs real grounding from the notes — an insight without evidence is worthless for this video

Now, please build the video digest. Only output JSON, no other content.
"""


def build_document_digest_prompt(
    doc_info: str,
    notes_json: str,
    focus: Optional[str] = None,
) -> str:
    """Build the digest (reduce step) prompt."""
    focus_block = ""
    if focus and focus.strip():
        focus_block = (
            f'\n# User Focus\nThe user asked the video to focus on: "{focus.strip()}".\n'
            "Bias core_message, key_insights and hook_ideas toward that focus — "
            "while staying strictly faithful to the document.\n"
        )
    return DOCUMENT_DIGEST_PROMPT.format(
        doc_info=doc_info,
        notes_json=notes_json,
        focus_block=focus_block,
    )


# ==================== ③ Script (digest → narrations) ====================

PDF_SCRIPT_PROMPT = """# Role Definition
You are a professional short-video scriptwriter. Your specialty: turning dense documents (papers, reports, books) into short videos that feel like a smart friend telling you about something fascinating they just read — never like a lecture, never like a book report.
Globally, you must strictly write all narrations and the video title in {language_requirement}.

# Core Task
Using the video digest of a PDF document below, write a video title plus exactly {n_storyboard} scene narrations (each will be converted to speech by TTS).
{focus_block}
# Video Digest
{digest_context}

# Output Requirements

## Narration Specifications
- Language: {language_requirement} — natural, conversational, with correct diacritics where applicable
- Purpose: TTS audio for a short video
- Word count limit: strictly {min_words}~{max_words} words per narration (minimum not less than {min_words} words). For languages written with spaces (English, Vietnamese, ...) count space-separated words; for Chinese count characters
- Ending format: no punctuation at the very end of each narration; inside a narration, use punctuation natural for speech to create rhythm and pauses
- Style: like telling a friend about a great read — accessible, sincere, a little excited; avoid academic and stiff expressions, reject formulaic template phrasing
- Prohibitions: no URLs, no emojis, no numeric numbering, no empty talk or clichés

## Grounding Rules (most important)
- Every claim, number, name and quote must come from the digest. NEVER invent or embellish facts — it is better to be vivid about a true fact than precise about an invented one
- When an insight has grounding (a number, a finding, a quote), USE it — concrete evidence is what makes the video worth watching
- Mention the source document naturally ONCE, near the beginning or the end (e.g. "this paper", "the report", "cuốn sách này" — matching the document type and language) so viewers know the video distills a real document; do not repeat it in every scene

## Narrative Arc
- Scene 1 (hook): start from one of the digest's hook_ideas — or a better one, still grounded in the digest — and make the viewer need to know the answer
- Middle scenes: one key insight each, in the digest's order unless a better flow exists; make each concrete using its grounding
- Last scene (takeaway): land the core_message as something the viewer can act on or think about — not a dry summary
- All scenes must flow as one continuous voice with natural transitions, like one person talking

## Opening Diversity Requirements (strict)
- Each narration must open differently; if any word or phrase starts two narrations, the script has failed — rewrite it
- Never fall into hidden patterns ("the Nth sentence always starts with X"); choose each opening naturally from the content itself

# Output Format
Strictly output in the following JSON format, do not add any additional text explanations:

```json
{{
  "title": "Video title here",
  "narrations": [
    "First {min_words}~{max_words} word narration",
    "Second {min_words}~{max_words} word narration"
  ]
}}
```

# Important Reminders
1. Only output JSON format content, do not add any explanations
2. Ensure JSON format is strictly correct and can be directly parsed by the program
3. Must output exactly {n_storyboard} narrations, each strictly {min_words}~{max_words} words
4. "title" is the VIDEO's title (shown on screen): at most {title_max_chars} characters, in {language_requirement}, punchy, no quotes around it — it may differ from the document's own title if that makes a better video
5. Every fact must trace back to the digest — self-check before answering
6. Self-check the openings: no two narrations may start with the same word

Now, please write the title and the {n_storyboard} scene narrations. Only output JSON, no other content.
"""


def build_pdf_script_prompt(
    digest_context: str,
    n_storyboard: int,
    min_words: int,
    max_words: int,
    title_max_chars: int,
    language_requirement: str,
    focus: Optional[str] = None,
) -> str:
    """Build the digest → script prompt."""
    focus_block = ""
    if focus and focus.strip():
        focus_block = (
            f'\n# User Focus\nThe user asked the video to focus on: "{focus.strip()}". '
            "Center the script on that focus while staying faithful to the digest.\n"
        )
    return PDF_SCRIPT_PROMPT.format(
        digest_context=digest_context,
        n_storyboard=n_storyboard,
        min_words=min_words,
        max_words=max_words,
        title_max_chars=title_max_chars,
        language_requirement=language_requirement,
        focus_block=focus_block,
    )


# ==================== ④ Visual prompts (digest-aware) ====================

PDF_VISUAL_PROMPT_PROMPT = """# Role Definition
You are a visual creative director designing the imagery of ONE short video for AI image/video generation models. Your signature: all scenes of a video live inside one coherent visual world, so the video feels art-directed instead of stitched together.

# Core Task
For each narration below, create one corresponding **English** media-generation prompt. The video distills a document, so the imagery must make abstract ideas concrete through the shared visual world.

**Important: The input contains {narrations_count} narrations. You must generate exactly {narrations_count} prompts, one per narration, in the same order.**

# The Visual World (every prompt must live inside it)
{visual_world}
Narration tone: {tone}

# Input Narrations
{narrations_json}

# Output Requirements

## Prompt Specifications
- Language: **English only** (for AI image/video generation models)
- Length: {min_words}~{max_words} words each
- Structure: setting (from the visual world) + subject and action + emotion / atmosphere + one symbolic element expressing the narration's idea
- The prompt describes a single coherent picture or shot — no storyboards, no split frames

## Coherence Rules (most important)
- Every prompt reuses the visual world's setting, palette and motifs, so all scenes clearly belong to the same video
- VARY the composition between scenes: wide establishing shot, close-up, over-the-shoulder, bird's-eye view, silhouette... — same world, different camera
- The symbolic element must express the SPECIFIC idea of its narration (e.g. a path forking for a choice, a chain breaking for freedom, a seed sprouting for growth) — not generic decoration

## Hard Constraints
- NO text, letters, numbers, charts with labels, or signage in the image — AI models render text badly
- No real people's faces or likenesses; figures are anonymous or symbolic
- Do not contradict the narration's emotion

# Output Format
Strictly output in the following JSON format, **prompts must be in English**:

```json
{{
  "image_prompts": [
    "[English media prompt for narration 1]",
    "[English media prompt for narration 2]"
  ]
}}
```

# Important Reminders
1. Only output JSON format content, do not add any explanations
2. Ensure JSON format is strictly correct and can be directly parsed by the program
3. The output image_prompts array must contain exactly {narrations_count} elements, corresponding one-to-one with the input narrations
4. Every prompt embeds the visual world AND varies the composition
5. **All prompts must be in English**

Now, please create the {narrations_count} corresponding **English** prompts. Only output JSON, no other content.
"""


def build_pdf_visual_prompt_prompt(
    narrations: List[str],
    visual_world: str,
    tone: str,
    min_words: int = 30,
    max_words: int = 60,
) -> str:
    """Build the digest-aware visual prompt generation prompt."""
    narrations_json = json.dumps({"narrations": narrations}, ensure_ascii=False, indent=2)
    return PDF_VISUAL_PROMPT_PROMPT.format(
        narrations_json=narrations_json,
        narrations_count=len(narrations),
        visual_world=visual_world.strip() or "a clean minimalist stage with soft neutral light",
        tone=tone.strip() or "warm and curious",
        min_words=min_words,
        max_words=max_words,
    )
