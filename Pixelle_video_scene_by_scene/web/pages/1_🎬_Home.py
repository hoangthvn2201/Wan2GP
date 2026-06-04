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
Home Page - Scene-by-Scene video generation wizard
"""

import sys
from pathlib import Path

# Add project roots to sys.path (this folder, ../Pixelle_video, Wan2GP root)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_wan2gp_root = _project_root.parent
for _p in (_wan2gp_root / "Pixelle_video", _wan2gp_root):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import streamlit as st

from web.state.session import init_session_state, init_i18n, get_pixelle_video
from web.components.header import render_header
from web.components.settings import render_advanced_settings
from web.components.faq import render_faq_sidebar
from web.components.scene_wizard import render_scene_wizard

# Page config
st.set_page_config(
    page_title="Home - Pixelle-Video Scene by Scene",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def main():
    """Main UI entry point"""
    # Initialize session state and i18n
    init_session_state()
    init_i18n()

    # Render header (title + language selector)
    render_header()

    # Render FAQ in sidebar
    render_faq_sidebar()

    # Initialize Pixelle-Video core (shared with the original app)
    pixelle_video = get_pixelle_video()

    # Render system configuration (LLM + ComfyUI)
    render_advanced_settings()

    # Scene-by-scene wizard
    render_scene_wizard(pixelle_video)


if __name__ == "__main__":
    main()
