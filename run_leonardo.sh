#!/bin/bash
#SBATCH --job-name=sft-swedish
#SBATCH --account=EUHPC_E05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# SFT Baseline: Qwen3-1.7B on EuroEval Swedish — Leonardo HPC
trap 'echo "Error at line $LINENO at $(date)" >&2' ERR

# === PATHS ===
WORK_DIR="/leonardo_work/EUHPC_E05_119/abueff00/euroSFT"
PROJECT_DIR="/leonardo/home/userexternal/abueff00/euroSFT"
MODEL_DIR="${WORK_DIR}/models/Qwen3-1.7B"
DATA_DIR="${PROJECT_DIR}/training-data/sv"
OUTPUT_DIR="${WORK_DIR}/output/qwen3-1.7b-sft-swedish"
CODE_DIR="${PROJECT_DIR}/code"

# Container
USE_CONTAINER=true
CONTAINER="/leonardo_work/EUHPC_E05_119/containers/euroSFT.sif"

# === Environment ===
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${WORK_DIR}/hf_cache"
export HF_DATASETS_CACHE="${WORK_DIR}/hf_cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Offline mode — compute nodes have no internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pass paths to train_sft.py via env vars
export MODEL_PATH="${MODEL_DIR}"
export DATA_DIR="${DATA_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR}"

# Create output dirs
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HF_HOME}"

# === Container detection ===
if [ "$USE_CONTAINER" = true ]; then
    if command -v singularity &>/dev/null; then
        CONTAINER_CMD="singularity"
    elif command -v apptainer &>/dev/null; then
        CONTAINER_CMD="apptainer"
    else
        echo "ERROR: Neither singularity nor apptainer found"
        exit 1
    fi

    # Forward env vars into container
    export SINGULARITYENV_MODEL_PATH="$MODEL_PATH"
    export SINGULARITYENV_DATA_DIR="$DATA_DIR"
    export SINGULARITYENV_OUTPUT_DIR="$OUTPUT_DIR"
    export SINGULARITYENV_HF_HOME="$HF_HOME"
    export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
    export SINGULARITYENV_HF_HUB_OFFLINE=1
    export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
    export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"
    export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
fi

# === Info ===
echo "======================================"
echo "SFT Swedish — Qwen3-1.7B"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "Model:  ${MODEL_PATH}"
echo "Data:   ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi || true
echo ""

# === Run training ===
run_python() {
    if [ "$USE_CONTAINER" = true ]; then
        $CONTAINER_CMD exec \
            --nv \
            --bind "$PROJECT_DIR":"$PROJECT_DIR" \
            --bind "$WORK_DIR":"$WORK_DIR" \
            --pwd "$CODE_DIR" \
            "$CONTAINER" \
            python "$@"
    else
        cd "$CODE_DIR"
        python "$@"
    fi
}

run_python "${CODE_DIR}/train_sft.py"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "SFT complete | Exit: $EXIT_CODE | End: $(date)"
echo "Output: ${OUTPUT_DIR}"
[ $EXIT_CODE -eq 0 ] && echo "Training completed successfully" || echo "Training failed"

exit $EXIT_CODE
