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
Style Preset Configuration

Predefined prompt prefixes representing different visual styles for
image / video generation. Selected in the Web UI; the chosen prefix is
prepended to every media generation prompt (same mechanism as the
free-text prompt prefix).

Prefixes are written in English since generation models respond best to
English style descriptions.
"""

from typing import Any, Dict, List, Optional

# Default style (must stay in sync with ImageSubConfig/VideoSubConfig
# prompt_prefix defaults in pixelle_video/config/schema.py)
DEFAULT_STYLE_ID = "matchstick"

STYLE_PROMPT_PREFIXES: List[Dict[str, Any]] = [
    {
        # Brand style for the "Chầm chậm mà hiểu" YouTube Shorts channel
        # (Vietnamese psychology / self-understanding). Matches the channel
        # logo & panel in materials/: cream paper background, flowing
        # coral-pink-teal gradient waves, calm flat-vector characters,
        # puzzle / light bulb / heart / sparkle motifs.
        "id": "cham_cham",
        "label_en": "Chầm Chậm Mà Hiểu (Channel Brand)",
        "label_zh": "慢慢才懂（频道品牌风格）",
        "prefix": (
            "Heartwarming flat vector illustration in a gentle psychology-healing style, "
            "soft warm cream paper background, flowing organic gradient waves of coral orange, "
            "rose pink, golden amber and turquoise teal, calm characters with serene "
            "closed-eye expressions drawn in clean rounded shapes with deep navy hair and accents, "
            "colorful puzzle pieces, glowing light bulbs, speech bubbles, small hearts and tiny "
            "four-pointed sparkles floating around as metaphors for thoughts and emotions, "
            "generous negative space, soothing pastel palette with vivid gradient highlights, "
            "cozy, mindful, emotionally comforting atmosphere"
        ),
    },
    {
        "id": "matchstick",
        "label_en": "Matchstick Sketch (Default)",
        "label_zh": "火柴人简笔画（默认）",
        "prefix": "Minimalist black-and-white matchstick figure style illustration, clean lines, simple sketch style",
    },
    {
        "id": "watercolor",
        "label_en": "Watercolor",
        "label_zh": "水彩画",
        "prefix": "Soft watercolor painting style, gentle brush strokes, dreamy pastel colors, textured paper feel",
    },
    {
        "id": "flat_vector",
        "label_en": "Flat Vector Illustration",
        "label_zh": "扁平插画",
        "prefix": "Modern flat vector illustration, bold shapes, minimal details, harmonious pastel color palette, clean composition",
    },
    {
        "id": "cinematic",
        "label_en": "Cinematic Realistic",
        "label_zh": "电影写实",
        "prefix": "Cinematic photorealistic style, dramatic lighting, shallow depth of field, film grain, professional photography",
    },
    {
        "id": "anime",
        "label_en": "Anime",
        "label_zh": "日系动漫",
        "prefix": "Japanese anime illustration style, expressive characters, vibrant colors, detailed backgrounds, soft lighting",
    },
    {
        "id": "cartoon_3d",
        "label_en": "3D Cartoon",
        "label_zh": "3D 卡通",
        "prefix": "Cute 3D cartoon render, soft rounded shapes, vivid colors, playful character design, smooth studio lighting",
    },
    {
        "id": "ink_wash",
        "label_en": "Chinese Ink Wash",
        "label_zh": "国风水墨",
        "prefix": "Traditional Chinese ink wash painting style, elegant brush strokes, monochrome ink gradients, blank space aesthetics",
    },
    {
        "id": "oil_painting",
        "label_en": "Oil Painting",
        "label_zh": "古典油画",
        "prefix": "Classical oil painting style, rich textured brushwork, warm tones, chiaroscuro lighting, museum quality",
    },
    {
        "id": "cyberpunk",
        "label_en": "Cyberpunk Neon",
        "label_zh": "赛博朋克",
        "prefix": "Cyberpunk style, neon lights, futuristic cityscape, dark atmosphere with glowing purple and cyan accents, high contrast",
    },
    {
        "id": "paper_cutout",
        "label_en": "Paper Cutout",
        "label_zh": "剪纸拼贴",
        "prefix": "Layered paper cutout collage style, handcrafted look, soft drop shadows, bright cheerful colors",
    },
]


def get_style_preset(style_id: str) -> Optional[Dict[str, Any]]:
    """Get a style preset by id (None if not found)"""
    return next((s for s in STYLE_PROMPT_PREFIXES if s["id"] == style_id), None)


def get_style_prefix(style_id: str) -> Optional[str]:
    """Get the prompt prefix text for a style id (None if not found)"""
    preset = get_style_preset(style_id)
    return preset["prefix"] if preset else None


def find_style_by_prefix(prefix: str) -> Optional[str]:
    """
    Find the style id whose prefix matches the given text (whitespace-insensitive).

    Used by the UI to preselect the right preset when the configured
    prompt_prefix corresponds to one of the presets.
    """
    normalized = (prefix or "").strip()
    for style in STYLE_PROMPT_PREFIXES:
        if style["prefix"].strip() == normalized:
            return style["id"]
    return None


def get_style_display_name(style: Dict[str, Any], language: str = "en_US") -> str:
    """Get the display label for a style preset in the given UI language"""
    if language.startswith("zh"):
        return style.get("label_zh") or style["label_en"]
    return style["label_en"]
