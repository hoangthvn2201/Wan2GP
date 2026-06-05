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
LLM prompts for the flexible (generate-or-search) pipeline.

Two stages are unique to this pipeline:

1. **Scene plan** — the orchestration decision. For each narration the LLM
   chooses the media source: real stock footage (search) when the concept is
   something stock libraries actually have (nature, cities, people doing
   ordinary things, b-roll), or AI generation when the concept is abstract,
   stylized or too specific for stock. Search scenes get short English stock
   keywords; generate scenes get a full media-generation prompt.

2. **Candidate ranking** — given the metadata of the fetched stock candidates
   (alt text / tags, dimensions, duration), pick the one that best matches the
   scene narration. Metadata-only by design: the core LLM service is
   text-only, so thumbnails cannot be sent.

Prompt-style conventions (role / task / requirements / strict JSON output /
reminders) follow `Pixelle_video/pixelle_video/prompts/`.
"""

import json
from typing import List, Optional

from flexvid.models import MediaCandidate


# ==================== ① Scene plan (source orchestration) ====================

SCENE_PLAN_PROMPT = """# Role Definition
You are the media director of ONE short video. For every scene you decide where its visual comes from: real STOCK footage (searched on Pexels/Pixabay) or AI GENERATION — and you write the search keywords or the generation prompt accordingly. Your videos mix both sources so naturally that viewers never notice.

# Core Task
The video below has {n_scenes} scene narrations. For EACH scene, in order, decide the media source and produce the matching query or prompt.

Video title: {title}

# Input Narrations
{narrations_json}

# Decision Rules (most important)
- Choose "search" when the scene's idea is something stock libraries are full of: nature, landscapes, cities, landmarks, weather, animals, food, sports, technology in use, people doing ordinary things (working, walking, cooking, exercising), generic b-roll moods
- Choose "generate" when the scene needs: abstract or symbolic concepts, stylized/artistic looks, fantastical or impossible imagery, very specific compositions, brand- or story-specific elements, or anything a stock library would not plausibly have
- A good mix usually beats all-of-one-kind, but NEVER force it — pick what serves each scene best
{availability_rule}
# Output Requirements
For each scene output an object with:
- "source": "search" or "generate"
- "media_type": "image" or "video" — {media_type_rule}
- "search_query": ONLY when source is "search" — 2~5 short **English** stock-search keywords (e.g. "rainy tokyo street night", "barista pouring latte art"). Concrete and visual; no abstract words, no style words, no quotes
- "gen_prompt": ONLY when source is "generate" — one **English** media-generation prompt of {min_words}~{max_words} words: setting + subject and action + emotion/atmosphere + one symbolic element expressing the narration's idea. No text/letters/numbers in the image; no real people's faces
- "reason": one short sentence (English) explaining the choice

# Output Format
Strictly output in the following JSON format, do not add any additional text explanations:

```json
{{
  "scenes": [
    {{"source": "search", "media_type": "video", "search_query": "...", "reason": "..."}},
    {{"source": "generate", "media_type": "image", "gen_prompt": "...", "reason": "..."}}
  ]
}}
```

# Important Reminders
1. Only output JSON format content, do not add any explanations
2. Ensure JSON format is strictly correct and can be directly parsed by the program
3. The "scenes" array must contain exactly {n_scenes} entries, one per narration, in the same order
4. search_query and gen_prompt must be in English
5. Each entry must have search_query (when source=search) OR gen_prompt (when source=generate) — never both, never neither

Now, please produce the media plan for all {n_scenes} scenes. Only output JSON, no other content.
"""


def build_scene_plan_prompt(
    narrations: List[str],
    title: str,
    media_capability: str,
    providers_enabled: bool,
    min_words: int = 30,
    max_words: int = 60,
) -> str:
    """
    Build the per-scene source-orchestration prompt.

    Args:
        media_capability: "image" (image template — stills only) or
            "video" (video template — clips preferred, stills allowed).
        providers_enabled: False -> every scene must be source=generate.
    """
    if media_capability == "video":
        media_type_rule = (
            'this is a VIDEO project: prefer "video", use "image" only when a '
            "still is clearly stronger or stock clips of the idea are unlikely"
        )
    else:
        media_type_rule = 'this is an IMAGE project: media_type must always be "image"'

    if providers_enabled:
        availability_rule = ""
    else:
        availability_rule = (
            "- NOTE: no stock providers are configured for this project — "
            'every scene MUST use source "generate"\n'
        )

    narrations_json = json.dumps({"narrations": narrations}, ensure_ascii=False, indent=2)
    return SCENE_PLAN_PROMPT.format(
        n_scenes=len(narrations),
        title=title,
        narrations_json=narrations_json,
        availability_rule=availability_rule,
        media_type_rule=media_type_rule,
        min_words=min_words,
        max_words=max_words,
    )


# ==================== ② Candidate ranking ====================

CANDIDATE_RANKING_PROMPT = """# Role Definition
You are a video editor choosing stock footage. You receive a scene narration and the METADATA of stock media candidates (description/tags, dimensions, duration). Pick the candidate whose content best illustrates the narration.

# Scene Narration
{narration}

# Search Query Used
{search_query}

# Candidates (metadata only)
{candidates_block}

# Ranking Criteria
1. Content relevance: the description/tags should match the narration's concrete subject — this matters most
2. Composition fit: prefer candidates whose orientation matches {orientation} and whose resolution is higher
3. For videos: prefer durations of at least {min_duration} seconds (short clips loop awkwardly)
4. Avoid candidates whose description suggests text overlays, logos, watermarks or collages

# Output Format
Strictly output in the following JSON format, do not add any additional text explanations:

```json
{{
  "best_index": 0,
  "scores": [
    {{"index": 0, "score": 8, "reason": "..."}},
    {{"index": 1, "score": 5, "reason": "..."}}
  ]
}}
```

# Important Reminders
1. Only output JSON format content, do not add any explanations
2. Ensure JSON format is strictly correct and can be directly parsed by the program
3. "best_index" must be one of the candidate indices shown above
4. "scores" must contain one entry per candidate, score 0~10

Now, please rank the candidates. Only output JSON, no other content.
"""


def build_candidate_ranking_prompt(
    narration: str,
    search_query: str,
    candidates: List[MediaCandidate],
    orientation: Optional[str] = None,
    min_duration: float = 3.0,
) -> str:
    """Build the metadata-only candidate ranking prompt."""
    lines = [f"[{i}] {c.meta_line()}" for i, c in enumerate(candidates)]
    return CANDIDATE_RANKING_PROMPT.format(
        narration=narration,
        search_query=search_query or "(none)",
        candidates_block="\n".join(lines),
        orientation=orientation or "any",
        min_duration=f"{min_duration:.0f}",
    )
