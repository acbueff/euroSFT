#!/bin/bash
#SBATCH --job-name=dpo-swedish
#SBATCH --account=EUHPC_E05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=/leonardo_work/EUHPC_D21_101/abueff00/euroeval/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D21_101/abueff00/euroeval/logs/%x_%j.err

set -euo pipefail

# DPO: Qwen3-1.7B on EuroEval Swedish preference pairs — Leonardo HPC
# Policy:    Qwen3-1.7B + LoRA adapters (trainable)
# Reference: Qwen3-1.7B base weights (frozen, shared via peft_config)
trap 'echo "Error at line $LINENO at $(date)" >&2' ERR

# === PATHS ===
# Model weights (same source as SFT)
MODEL_DIR="/leonardo_work/EUHPC_E05_119/abueff00/euroSFT/models/Qwen3-1.7B"

# Code from $HOME
PROJECT_DIR="/leonardo/home/userexternal/abueff00/euroSFT"
CODE_DIR="${PROJECT_DIR}/code"

# DPO data and all output on D21_101 (storage)
EUROEVAL_DIR="/leonardo_work/EUHPC_D21_101/abueff00/euroeval"
DPO_DATA_DIR="${EUROEVAL_DIR}/dpo-preference-data/qwen3-0.6b-rejected"
OUTPUT_DIR="${EUROEVAL_DIR}/output/qwen3-1.7b-dpo-swedish"
HF_HOME="${EUROEVAL_DIR}/hf_cache"

PIP_PACKAGES="/leonardo_work/EUHPC_E05_119/abueff00/euroSFT/pip_packages"
CONTAINER="/leonardo_work/EUHPC_D21_101/containers/smoLLM_fixed.sif"

# === Environment ===
export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pass paths to train_dpo.py via env vars
export MODEL_PATH="${MODEL_DIR}"
export DPO_DATA_DIR="${DPO_DATA_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${EUROEVAL_DIR}/logs"

# === Container detection ===
if command -v singularity &>/dev/null; then
    CONTAINER_CMD="singularity"
elif command -v apptainer &>/dev/null; then
    CONTAINER_CMD="apptainer"
else
    echo "ERROR: Neither singularity nor apptainer found"
    exit 1
fi

export SINGULARITYENV_PYTHONPATH="${PIP_PACKAGES}:${PYTHONPATH:-}"
export SINGULARITYENV_MODEL_PATH="$MODEL_PATH"
export SINGULARITYENV_DPO_DATA_DIR="$DPO_DATA_DIR"
export SINGULARITYENV_OUTPUT_DIR="$OUTPUT_DIR"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"

# === Info ===
echo "============================================"
echo "DPO Swedish — Qwen3-1.7B"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "Model:    ${MODEL_PATH}"
echo "DPO data: ${DPO_DATA_DIR}"
echo "Output:   ${OUTPUT_DIR}"
nvidia-smi || true
echo ""

# === Run training ===
set +e
$CONTAINER_CMD exec \
    --nv \
    --bind "$PROJECT_DIR":"$PROJECT_DIR" \
    --bind /leonardo_work/EUHPC_E05_119/abueff00/euroSFT:/leonardo_work/EUHPC_E05_119/abueff00/euroSFT \
    --bind "$EUROEVAL_DIR":"$EUROEVAL_DIR" \
    --pwd "$CODE_DIR" \
    "$CONTAINER" \
    python train_dpo.py
EXIT_CODE=$?
set -e

echo ""
echo "============================================"
echo "DPO complete | Exit: $EXIT_CODE | End: $(date)"
echo "Output: ${OUTPUT_DIR}"
[ $EXIT_CODE -eq 0 ] && echo "Training completed successfully" || echo "Training failed"

exit $EXIT_CODE
