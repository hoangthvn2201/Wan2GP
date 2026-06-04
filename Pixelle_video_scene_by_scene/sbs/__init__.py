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
Scene-by-Scene orchestration package.

Re-uses the Pixelle_video core services (LLM / TTS / media / video / html
frame rendering) but exposes the pipeline as individual, user-driven steps
instead of one monolithic `generate_video()` call.
"""

from sbs.models import Scene, SceneProject
from sbs.engine import SceneBySceneEngine

__all__ = ["Scene", "SceneProject", "SceneBySceneEngine"]
