"""OpenAI-compatible chat client (requests-based, no new hard dependency).

Works against both hosted OpenAI (`https://api.openai.com`) and self-hosted vLLM
(`http://host:port`), since both expose `/v1/chat/completions`.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from .prompts import SYSTEM_PROMPT, build_user_prompt

SCENE_KEYS = ("scene_summary", "image_prompt", "video_prompt", "narration_text", "tts_prompt")


class LLMConfig:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 180):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        self.timeout = timeout

    def chat_url(self) -> str:
        base = self.base_url
        # Accept a base_url that already ends in /v1 (or /v1/).
        if re.search(r"/v\d+$", base):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


def _headers(cfg: LLMConfig) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return headers


def _post_chat(
    cfg: LLMConfig,
    messages: List[Dict[str, str]],
    *,
    json_mode: bool = True,
    temperature: float = 0.7,
) -> str:
    body: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    resp = requests.post(cfg.chat_url(), headers=_headers(cfg), json=body, timeout=cfg.timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def test_connection(cfg: LLMConfig) -> Tuple[bool, str]:
    if not cfg.base_url:
        return False, "Base URL is empty."
    if not cfg.model:
        return False, "Model name is empty."
    try:
        content = _post_chat(
            cfg,
            [{"role": "user", "content": "Reply with the single word: OK"}],
            json_mode=False,
            temperature=0.0,
        )
        return True, f"Connected. Model replied: {content[:80]!r}"
    except requests.HTTPError as e:
        return False, f"HTTP error: {e} - {getattr(e.response, 'text', '')[:200]}"
    except Exception as e:  # noqa: BLE001 - surface any connection problem to the UI
        return False, f"Connection failed: {e}"


def _extract_json(content: str) -> str:
    """Strip markdown fences / leading prose and return the JSON substring."""
    text = (content or "").strip()
    # Remove ```json ... ``` or ``` ... ``` fences.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    # Fall back to first '{' .. last '}'.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_and_validate(content: str, num_scenes: int) -> Dict[str, Any]:
    """Parse model output into the canonical script dict, coercing/padding scenes.

    Adds a non-fatal "_warning" key when the scene count had to be adjusted.
    """
    raw = json.loads(_extract_json(content))
    if not isinstance(raw, dict):
        raise ValueError("LLM did not return a JSON object.")

    scenes_in = raw.get("scenes")
    if not isinstance(scenes_in, list):
        raise ValueError("LLM JSON has no 'scenes' array.")

    scenes: List[Dict[str, str]] = []
    for item in scenes_in:
        item = item if isinstance(item, dict) else {}
        scenes.append({k: str(item.get(k, "") or "") for k in SCENE_KEYS})

    warning = None
    n = int(num_scenes)
    if len(scenes) != n:
        warning = f"LLM returned {len(scenes)} scene(s); expected {n}. Adjusted."
        if len(scenes) > n:
            scenes = scenes[:n]
        else:
            for _ in range(n - len(scenes)):
                scenes.append({k: "" for k in SCENE_KEYS})

    result: Dict[str, Any] = {
        "overall_script": str(raw.get("overall_script", "") or ""),
        "scenes": scenes,
    }
    if warning:
        result["_warning"] = warning
    return result


def generate_script(cfg: LLMConfig, brief: str, num_scenes: int) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(brief, num_scenes)},
    ]
    try:
        content = _post_chat(cfg, messages, json_mode=True)
    except requests.HTTPError:
        # Older / minimal servers may reject response_format -> retry without it.
        content = _post_chat(cfg, messages, json_mode=False)
    return parse_and_validate(content, num_scenes)
