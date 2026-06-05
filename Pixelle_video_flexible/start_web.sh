#!/bin/bash
# Start Pixelle-Video Flexible (generate-or-search) Web UI
#
# Run this from inside the Wan2GP python environment (e.g. `conda activate wan2gp`)
# so that wan2gp/* workflows can load the generation models in-process.
#
# Data (config.yaml, workflows/, templates/, bgm/, output/) is SHARED with the
# original ../Pixelle_video folder — no re-configuration needed. The media
# search API keys live in ./flex_config.yaml (or PEXELS_API_KEY /
# PIXABAY_API_KEY environment variables).

echo "🚀 Starting Pixelle-Video Flexible Video Web UI..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIXELLE_ROOT="$(cd "$SCRIPT_DIR/../Pixelle_video" && pwd)"

# Run from the original Pixelle_video directory so relative resources
# (config.yaml, workflows/, templates/, bgm/, output/) resolve there.
cd "$PIXELLE_ROOT"
export PIXELLE_VIDEO_ROOT="$PIXELLE_ROOT"

# Use a different port than the original app (8501), the scene-by-scene app
# (8502) and the PDF app (8503) so all four can run together.
streamlit run "$SCRIPT_DIR/web/app.py" --server.port "${PORT:-8504}"
