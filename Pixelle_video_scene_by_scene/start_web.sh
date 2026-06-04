#!/bin/bash
# Start Pixelle-Video Scene-by-Scene Web UI
#
# Run this from inside the Wan2GP python environment (e.g. `conda activate wan2gp`)
# so that wan2gp/* workflows can load the generation models in-process.
#
# Data (config.yaml, workflows/, templates/, bgm/, output/) is SHARED with the
# original ../Pixelle_video folder — no re-configuration needed.

echo "🚀 Starting Pixelle-Video Scene-by-Scene Web UI..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIXELLE_ROOT="$(cd "$SCRIPT_DIR/../Pixelle_video" && pwd)"

# Run from the original Pixelle_video directory so relative resources
# (config.yaml, workflows/, templates/, bgm/, output/) resolve there.
cd "$PIXELLE_ROOT"
export PIXELLE_VIDEO_ROOT="$PIXELLE_ROOT"

# Use a different port than the original app (8501) so both can run together.
streamlit run "$SCRIPT_DIR/web/app.py" --server.port "${PORT:-8502}"
