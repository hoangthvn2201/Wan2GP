#!/bin/bash
# Start Pixelle-Video Web UI
#
# Run this from inside the Wan2GP python environment (e.g. `conda activate wan2gp`)
# so that wan2gp/* workflows can load the generation models in-process.

echo "🚀 Starting Pixelle-Video Web UI..."
echo ""

# Always run from the Pixelle_video directory (relative paths: config.yaml,
# workflows/, templates/, output/, ...)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PIXELLE_VIDEO_ROOT="$SCRIPT_DIR"

# Start Streamlit
streamlit run web/app.py
