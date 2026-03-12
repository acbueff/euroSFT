#!/bin/bash
#SBATCH --job-name=lora-sft-swedish
#SBATCH --account=EUHPC_E05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=/leonardo_work/EUHPC_D21_101/containers/euro-sft/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D21_101/containers/euro-sft/logs/%x_%j.err

set -euo pipefail

# LoRA SFT: Qwen3-1.7B on EuroEval Swedish — Leonardo HPC
trap 'echo "Error at line $LINENO at $(date)" >&2' ERR

# === PATHS ===
# Model weights sourced from E05_119 (already downloaded there)
MODEL_DIR="/leonardo_work/EUHPC_E05_119/abueff00/euroSFT/models/Qwen3-1.7B"

# Code and data from $HOME
PROJECT_DIR="/leonardo/home/userexternal/abueff00/euroSFT"
DATA_DIR="${PROJECT_DIR}/training-data/sv"
CODE_DIR="${PROJECT_DIR}/code"

# All output goes to D21_101
EURO_SFT_DIR="/leonardo_work/EUHPC_D21_101/containers/euro-sft"
OUTPUT_DIR="${EURO_SFT_DIR}/output/qwen3-1.7b-lora-sft-swedish-v2"
HF_HOME="${EURO_SFT_DIR}/hf_cache"

# pip_packages (trl + peft installed here)
PIP_PACKAGES="/leonardo_work/EUHPC_E05_119/abueff00/euroSFT/pip_packages"

# Container
CONTAINER="/leonardo_work/EUHPC_D21_101/containers/smoLLM_fixed.sif"

# === Environment ===
export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Offline mode — compute nodes have no internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pass paths to train_lora_sft.py via env vars
export MODEL_PATH="${MODEL_DIR}"
export DATA_DIR="${DATA_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR}"

# Create output dirs
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${EURO_SFT_DIR}/logs"

# === Container detection ===
if command -v singularity &>/dev/null; then
    CONTAINER_CMD="singularity"
elif command -v apptainer &>/dev/null; then
    CONTAINER_CMD="apptainer"
else
    echo "ERROR: Neither singularity nor apptainer found"
    exit 1
fi

# Forward env vars into container
export SINGULARITYENV_PYTHONPATH="${PIP_PACKAGES}:${PYTHONPATH:-}"
export SINGULARITYENV_MODEL_PATH="$MODEL_PATH"
export SINGULARITYENV_DATA_DIR="$DATA_DIR"
export SINGULARITYENV_OUTPUT_DIR="$OUTPUT_DIR"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"

# === Info ===
echo "======================================"
echo "LoRA SFT Swedish — Qwen3-1.7B"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "Model:  ${MODEL_PATH}"
echo "Data:   ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi || true
echo ""

# === Run training ===
run_python() {
    $CONTAINER_CMD exec \
        --nv \
        --bind "$PROJECT_DIR":"$PROJECT_DIR" \
        --bind /leonardo_work/EUHPC_E05_119/abueff00/euroSFT:/leonardo_work/EUHPC_E05_119/abueff00/euroSFT \
        --bind "$EURO_SFT_DIR":"$EURO_SFT_DIR" \
        --pwd "$CODE_DIR" \
        "$CONTAINER" \
        python "$@"
}

# Disable set -e around training to capture the exit code
set +e
run_python "${CODE_DIR}/train_lora_sft.py"
EXIT_CODE=$?
set -e

echo ""
echo "======================================"
echo "LoRA SFT complete | Exit: $EXIT_CODE | End: $(date)"
echo "Output: ${OUTPUT_DIR}"
[ $EXIT_CODE -eq 0 ] && echo "Training completed successfully" || echo "Training failed"

exit $EXIT_CODE
