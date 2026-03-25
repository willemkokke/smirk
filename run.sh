#!/bin/bash
# Run SMIRK unified demo with all options enabled.
# Usage: ./run.sh <input_file>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <input_image_or_video>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run python "$SCRIPT_DIR/demo_unified.py" \
    --input_path "$1" \
    --crop \
    --render_orig \
    --use_smirk_generator \
    --export_scene \
    --device cpu \
    2>/dev/null
