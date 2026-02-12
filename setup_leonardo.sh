#!/bin/bash
# Setup script for Leonardo HPC — run from your LOCAL machine
#
# This script:
#   1. Uploads code + training data to Leonardo
#   2. SSHes in and downloads the model (login node has internet)
#   3. Builds the Singularity container
#
# Prerequisites:
#   - SSH access to Leonardo configured (~/.ssh/config)
#   - HF_TOKEN set if model is gated
#
# Usage:
#   ./setup_leonardo.sh           # full setup
#   ./setup_leonardo.sh upload    # only upload code + data
#   ./setup_leonardo.sh model     # only download model
#   ./setup_leonardo.sh container # only build container

set -euo pipefail

LEONARDO="login.leonardo.cineca.it"
LEONARDO_USER="abueff00"
LEONARDO_HOST="${LEONARDO_USER}@${LEONARDO}"

# Remote paths
REMOTE_HOME="/leonardo/home/userexternal/${LEONARDO_USER}/euroSFT"
REMOTE_WORK="/leonardo_work/EUHPC_E05_119/${LEONARDO_USER}/euroSFT"

# Local paths
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DATA="/home/andreas/PostDoc/euroSFT/training-data"

ACTION="${1:-all}"

# ------------------------------------------------------------------
# 1. Upload code and data
# ------------------------------------------------------------------
upload_files() {
    echo "=== Uploading code and data to Leonardo ==="

    # Create remote directories
    ssh "${LEONARDO_HOST}" "
        mkdir -p ${REMOTE_HOME}/code
        mkdir -p ${REMOTE_WORK}/training-data/sv
        mkdir -p ${REMOTE_WORK}/output
        mkdir -p ${REMOTE_WORK}/models
        mkdir -p ${REMOTE_WORK}/hf_cache
    "

    # Upload code
    echo "Uploading code..."
    scp "${LOCAL_DIR}/code/"*.py "${LOCAL_DIR}/code/"*.yaml "${LEONARDO_HOST}:${REMOTE_HOME}/code/"
    scp "${LOCAL_DIR}/run_leonardo.sh" "${LEONARDO_HOST}:${REMOTE_HOME}/"
    scp "${LOCAL_DIR}/requirements.txt" "${LEONARDO_HOST}:${REMOTE_HOME}/"

    # Upload training data (7.4 MB total — very fast)
    echo "Uploading training data..."
    scp "${LOCAL_DATA}/sv/"*.json "${LEONARDO_HOST}:${REMOTE_WORK}/training-data/sv/"

    echo "Upload complete."
}

# ------------------------------------------------------------------
# 2. Download model on login node (has internet)
# ------------------------------------------------------------------
download_model() {
    echo "=== Downloading Qwen3-1.7B on Leonardo login node ==="

    ssh "${LEONARDO_HOST}" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

MODEL_DIR="/leonardo_work/EUHPC_E05_119/abueff00/euroSFT/models/Qwen3-1.7B"

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "Model already exists at $MODEL_DIR — skipping download."
    exit 0
fi

# Use pip-installed huggingface-hub or python
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download Qwen/Qwen3-1.7B --local-dir "$MODEL_DIR"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B', local_dir='$MODEL_DIR')
" 2>/dev/null || {
    echo "ERROR: huggingface-cli or huggingface_hub Python package not found."
    echo "Install with: pip install --user huggingface-hub"
    exit 1
}
fi

echo "Model downloaded to $MODEL_DIR"
ls -lh "$MODEL_DIR/"
REMOTE_SCRIPT
}

# ------------------------------------------------------------------
# 3. Build Singularity container
# ------------------------------------------------------------------
build_container() {
    echo "=== Building Singularity container on Leonardo ==="

    # Upload the container definition
    cat > /tmp/euroSFT.def <<'CONTAINERDEF'
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:24.12-py3

%post
    pip install --no-cache-dir \
        transformers>=4.46.0 \
        trl>=0.12.0 \
        datasets>=2.16.0 \
        accelerate>=0.25.0 \
        pyyaml>=6.0 \
        flash-attn --no-build-isolation

%labels
    Author abueff00
    Description SFT training for Qwen3-1.7B Swedish
CONTAINERDEF

    scp /tmp/euroSFT.def "${LEONARDO_HOST}:/tmp/euroSFT.def"

    ssh "${LEONARDO_HOST}" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

CONTAINER_DIR="/leonardo_work/EUHPC_E05_119/containers"
CONTAINER_PATH="${CONTAINER_DIR}/euroSFT.sif"
mkdir -p "$CONTAINER_DIR"

if [ -f "$CONTAINER_PATH" ]; then
    echo "Container already exists at $CONTAINER_PATH — skipping build."
    echo "Delete it and re-run to rebuild."
    exit 0
fi

# Build on login node (requires fakeroot or a build allocation)
if command -v singularity &>/dev/null; then
    singularity build --fakeroot "$CONTAINER_PATH" /tmp/euroSFT.def
elif command -v apptainer &>/dev/null; then
    apptainer build --fakeroot "$CONTAINER_PATH" /tmp/euroSFT.def
else
    echo "ERROR: No container runtime found. You may need to:"
    echo "  module load singularity"
    echo "Then re-run this script."
    exit 1
fi

echo "Container built: $CONTAINER_PATH"
ls -lh "$CONTAINER_PATH"
REMOTE_SCRIPT

    rm -f /tmp/euroSFT.def
}

# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------
case "$ACTION" in
    upload)    upload_files ;;
    model)     download_model ;;
    container) build_container ;;
    all)
        upload_files
        download_model
        build_container
        echo ""
        echo "=== Setup complete ==="
        echo "Submit the job with:"
        echo "  ssh ${LEONARDO_HOST} 'cd ${REMOTE_HOME} && sbatch run_leonardo.sh'"
        ;;
    *)
        echo "Usage: $0 {all|upload|model|container}"
        exit 1
        ;;
esac
