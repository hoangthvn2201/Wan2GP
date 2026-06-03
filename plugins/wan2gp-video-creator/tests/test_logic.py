"""Pure-logic tests for the Video Creator plugin (no GPU / no Gradio).

    python plugins/wan2gp-video-creator/tests/test_logic.py
"""

import importlib
import json
import os
import sys

# Make the plugins/ dir importable so the package resolves with its relative imports.
PLUGINS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PLUGINS_DIR)
PKG = "wan2gp-video-creator"

llm = importlib.import_module(f"{PKG}.llm_client")
sm = importlib.import_module(f"{PKG}.scene_model")
mr = importlib.import_module(f"{PKG}.model_registry")


def test_parse_and_validate():
    clean = json.dumps({"overall_script": "Ov", "scenes": [
        {"scene_summary": "a", "image_prompt": "ip", "video_prompt": "vp", "narration_text": "nt", "tts_prompt": "tp"},
        {"scene_summary": "b", "image_prompt": "ip2", "video_prompt": "vp2", "narration_text": "nt2", "tts_prompt": "tp2"}]})
    assert len(llm.parse_and_validate(clean, 2)["scenes"]) == 2
    assert len(llm.parse_and_validate("```json\n" + clean + "\n```", 2)["scenes"]) == 2
    assert len(llm.parse_and_validate("Here:\n" + clean + "\nThanks", 2)["scenes"]) == 2
    assert "_warning" in llm.parse_and_validate(clean, 4)  # padded
    assert "_warning" in llm.parse_and_validate(clean, 1)  # truncated
    partial = json.dumps({"scenes": [{"image_prompt": "x"}]})
    r = llm.parse_and_validate(partial, 1)
    assert r["scenes"][0]["narration_text"] == "" and r["scenes"][0]["image_prompt"] == "x"


def test_chat_url():
    assert llm.LLMConfig("http://h:8000", "", "m").chat_url() == "http://h:8000/v1/chat/completions"
    assert llm.LLMConfig("https://api.openai.com/v1", "k", "m").chat_url() == "https://api.openai.com/v1/chat/completions"
    assert llm.LLMConfig("http://h:8000/", "", "m").chat_url() == "http://h:8000/v1/chat/completions"


def test_scene_mappers():
    script = {"overall_script": "O", "scenes": [
        {"scene_summary": "s", "image_prompt": "ip", "video_prompt": "vp", "narration_text": "nt", "tts_prompt": "calm"}]}
    pl = sm.scenes_from_llm(script)
    pl["models"] = {"image_model": "qwen_image_20B", "video_model": "ltx2_22B_distilled", "tts_model": "qwen3_tts_base"}
    sc = pl["scenes"][0]
    assert sm.scene_to_image_settings(pl, sc)["model_type"] == "qwen_image_20B"
    sc["image_path"] = "/tmp/x.png"; sc["use_start_image"] = True
    vis = sm.scene_to_video_settings(pl, sc)
    assert vis["image_start"] == "/tmp/x.png" and vis["num_inference_steps"] == 8
    pl["models"]["video_model"] = "ltx2_22B"; sc["use_start_image"] = False
    vis2 = sm.scene_to_video_settings(pl, sc)
    assert "image_start" not in vis2 and vis2["num_inference_steps"] == 30
    tts = sm.scene_to_tts_settings(pl, sc)
    assert tts["prompt"] == "nt" and tts["alt_prompt"] == "calm"


def test_model_registry():
    defs = {"qwen_image_20B": {"image_outputs": True}, "qwen3_tts_base": {"audio_only": True},
            "ltx2_22B": {}, "ltx2_22B_distilled": {}, "wan_t2v": {}}
    gmd = lambda mt: defs.get(mt)
    gname = lambda mt: mt.upper()
    gbase = lambda mt: "ltx2_22B" if mt.startswith("ltx2") else mt
    dmt = list(defs.keys())
    assert mr.classify("qwen_image_20B", gmd) == "image"
    assert mr.classify("qwen3_tts_base", gmd) == "audio"
    assert mr.list_image_models(dmt, gmd, gname) == [("QWEN_IMAGE_20B", "qwen_image_20B")]
    ltx = mr.list_ltx2_video_models(dmt, gmd, gbase, gname)
    assert sorted(v for _, v in ltx) == ["ltx2_22B", "ltx2_22B_distilled"]
    assert mr.resolve_variant_key("Distilled", ltx) == "ltx2_22B_distilled"
    assert mr.resolve_variant_key("Dev", ltx) == "ltx2_22B"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"{fn.__name__}: PASS")
    print(f"\nAll {len(fns)} test(s) passed.")
