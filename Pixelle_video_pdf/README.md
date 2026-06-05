# 📄 Pixelle-Video · PDF → Video

Turns a **PDF document** (research paper, report, book chapter, slide deck,
manual...) into a narrated short video, using the
[Pixelle-Video](../Pixelle_video) core and the
[scene-by-scene](../Pixelle_video_scene_by_scene) engine:

```
① Ingest    PDF → cleaned per-page text + metadata + TOC
            (header/footer removal, de-hyphenation; PyMuPDF, pypdf fallback)
② Digest    map-reduce LLM analysis → "video digest":
            core message · grounded key insights · hook ideas ·
            ONE coherent visual world for all scenes · tone · language
③ Script    digest → video title + scene narrations
            (hook → one insight per scene → takeaway, every fact grounded)
④ Visuals   digest-aware English media prompts — every scene lives inside
            the shared visual world, with varied camera/composition
⑤ Scenes    per scene: 🎤 audio → 🖼️/🎬 image-or-video → 🎞️ segment
⑥ Final     compose all segments (+ optional BGM) → final.mp4
```

Steps ⑤–⑥ (and project creation, per-scene regeneration, history
persistence) are **inherited unchanged** from `SceneBySceneEngine`, so
everything documented in
[`../Pixelle_video_scene_by_scene/README.md`](../Pixelle_video_scene_by_scene/README.md)
— invalidation rules, uid-based asset paths, t2v/i2v modes, BGM — applies
here too.

## What makes the PDF pipeline different

- **Grounding.** A PDF is a real document, so every stage enforces: facts,
  numbers, names and quotes may only come from the document. Chunk notes keep
  the *grounding* (the concrete evidence) attached to each insight, and the
  scriptwriter is instructed to use it — credible beats generic.
- **Scales past the context window.** Long PDFs are chunked (~12k chars,
  page-aligned); each chunk is mapped to structured notes, then one reduce
  pass builds the digest. Very large documents are sampled evenly
  (`max_chunks`, always keeping the first and last chunk).
- **One visual world.** Instead of N unrelated images, the digest picks a
  single visual setting/motif that fits the document (e.g. a neuroscience
  paper → "glowing neural pathways in deep blue space"); every scene prompt
  lives inside it with varied composition, so the video feels art-directed.
- **Language follows the document.** The narration language is auto-detected
  from the PDF (English / Vietnamese / Chinese / ...) and can be overridden
  (e.g. a Vietnamese video about an English paper).

## Relationship with the sibling apps

- `../Pixelle_video` — the **core is reused, unchanged** (LLM / TTS / media /
  video / HTML frame rendering / persistence) and the **data is shared**:
  `config.yaml`, `workflows/`, `templates/`, `bgm/`, `output/` all resolve
  there (via `PIXELLE_VIDEO_ROOT`). Finished videos appear in its History page.
- `../Pixelle_video_scene_by_scene` — `PdfVideoEngine` **subclasses**
  `SceneBySceneEngine` (imported, unchanged) for everything after the visual
  prompts.
- Only this folder contains the new logic:
  - `pdfv/pdf_ingest.py` — PDF extraction, cleaning, chunking
  - `pdfv/models.py` — `PdfDocument`, `DocumentDigest`
  - `pdfv/prompts.py` — the four new LLM prompts (chunk notes, digest,
    grounded script, visual-world prompts)
  - `pdfv/engine.py` — `PdfVideoEngine`

## Usage

Same environment as the original app (Wan2GP conda env, see
[`../Pixelle_video/WAN2GP_BACKEND.md`](../Pixelle_video/WAN2GP_BACKEND.md))
plus the PDF extras:

```bash
pip install -r Pixelle_video_pdf/requirements.txt
```

Then, with `Wan2GP/`, `Pixelle_video/`, `Pixelle_video_scene_by_scene/` and
`Pixelle_video_pdf/` on `sys.path` (see the notebook for the exact setup):

```python
from pixelle_video.service import pixelle_video
from pdfv import PdfVideoEngine

await pixelle_video.initialize()
engine = PdfVideoEngine(pixelle_video)

doc      = engine.ingest_pdf("paper.pdf", page_range=(1, 12))
digest   = await engine.digest_document(doc, focus="the experimental results")
title, narrations = await engine.generate_pdf_script(digest, n_scenes=5)
prompts  = await engine.generate_visual_prompts(narrations, digest, prompt_prefix=STYLE)

project  = engine.create_project(title, narrations, prompts, params)   # inherited
for i, scene in enumerate(project.scenes):                             # inherited
    await engine.process_scene(project, scene, i)
result   = await engine.compose_final(project, bgm_path=None)          # inherited
```

The recommended entry point is the Colab notebook at the repo root:
[`../pixelle_video_pdf_wan2gp.ipynb`](../pixelle_video_pdf_wan2gp.ipynb) —
it installs everything, lets you review/edit the digest, script and prompts
between stages, and previews every asset inline.

## Notes

- **Scanned PDFs** (images, no text layer) extract almost nothing — run OCR
  first (`ocrmypdf input.pdf output.pdf`), then ingest the result.
- Narration edits via `engine.rewrite_narration(...)` and per-scene
  regeneration work exactly as in the scene-by-scene app.
- Tasks are persisted with `pipeline: "pdf_to_video"`, so they are easy to
  tell apart in the History page.
