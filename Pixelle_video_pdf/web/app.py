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
Pixelle-Video PDF → Video Web UI - Main Entry Point

A wizard that turns a PDF document into a narrated short video:
ingest → digest (review/edit) → script → visual prompts → scenes → final.

This app REUSES the Pixelle_video core (`pixelle_video` package) and the
scene-by-scene engine (`sbs` package) untouched; only the web UI and the
PDF-specific pipeline (`pdfv` package) live in this folder.

sys.path layout (order matters):
    1. this folder                       -> `web` (this UI), `pdfv` (PDF engine)
    2. ../Pixelle_video_scene_by_scene   -> `sbs` (per-scene engine, unchanged)
    3. ../Pixelle_video                  -> `pixelle_video` (core, unchanged)
    4. ../ (Wan2GP repo root)            -> `Pixelle_video.*` absolute imports + `shared.api`
"""

import os
import sys
from pathlib import Path

# 1. This project folder (for `web` and `pdfv` packages).
#    MUST be at the very front of sys.path: `python -m streamlit` prepends the
#    cwd (often ../Pixelle_video, which has its own `web` package), and
#    PYTHONPATH entries come after it — so "insert if missing" is not enough.
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
while str(_project_root) in sys.path:
    sys.path.remove(str(_project_root))
sys.path.insert(0, str(_project_root))

# 2/3/4. The scene-by-scene app (sbs engine), the original Pixelle_video
# folder and the Wan2GP repo root
_wan2gp_root = _project_root.parent
_sbs_root = _wan2gp_root / "Pixelle_video_scene_by_scene"
_pixelle_root = _wan2gp_root / "Pixelle_video"
for p in (_sbs_root, _pixelle_root, _wan2gp_root):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# Share data (config.yaml, workflows/, templates/, bgm/, output/) with the
# original Pixelle_video folder so no re-configuration is needed and the
# History page shows tasks from all apps.
os.environ.setdefault("PIXELLE_VIDEO_ROOT", str(_pixelle_root))
if not (Path.cwd() / "config.yaml").exists() and (_pixelle_root / "config.yaml").exists():
    os.chdir(_pixelle_root)

import streamlit as st

# Setup page config (must be first Streamlit command)
st.set_page_config(
    page_title="Pixelle-Video · PDF → Video",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def main():
    """Main entry point with navigation"""
    home_page = st.Page(
        "pages/1_📄_Home.py",
        title="Home",
        icon="📄",
        default=True
    )

    history_page = st.Page(
        "pages/2_📚_History.py",
        title="History",
        icon="📚"
    )

    pg = st.navigation([home_page, history_page])
    pg.run()


if __name__ == "__main__":
    main()
