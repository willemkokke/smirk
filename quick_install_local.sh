#!/bin/bash
# Local version of quick_install.sh that uses the local FLAME2020.zip
# instead of downloading it from the FLAME website.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Extract local FLAME2020.zip
if [ ! -f "FLAME2020.zip" ]; then
    echo "Error: FLAME2020.zip not found in the repo root. Please place it there first."
    exit 1
fi

echo -e "\nExtracting local FLAME2020.zip..."
mkdir -p assets/FLAME2020/
unzip -o FLAME2020.zip -d assets/FLAME2020/

# Download Mediapipe Face Mesh model
if [ ! -f "assets/face_landmarker.task" ]; then
    echo -e "\nDownloading Mediapipe Face Mesh model..."
    curl -L -o assets/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
else
    echo -e "\nMediapipe Face Mesh model already exists, skipping download."
fi

# Download SMIRK pretrained model
if [ ! -f "pretrained_models/SMIRK_em1.pt" ]; then
    echo -e "\nDownloading pretrained SMIRK model..."
    mkdir -p pretrained_models/
    uv run gdown --id 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O pretrained_models/
else
    echo -e "\nSMIRK pretrained model already exists, skipping download."
fi

echo -e "\nDone! You can now run the demos:"
echo "  python demo.py --input_path samples/test_image2.png --out_path results/ --checkpoint pretrained_models/SMIRK_em1.pt --crop"
echo "  python demo_video.py --input_path samples/dafoe.mp4 --out_path results/ --checkpoint pretrained_models/SMIRK_em1.pt --crop --render_orig"
