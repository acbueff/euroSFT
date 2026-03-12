#!/bin/bash
#SBATCH --job-name=generate-dpo-rejected
#SBATCH --account=EUHPC_E05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=/leonardo_work/EUHPC_D21_101/abueff00/euroeval/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D21_101/abueff00/euroeval/logs/%x_%j.err

set -euo pipefail

# Generate DPO 'rejected' responses using Qwen3-0.6B as weak model.
# Covers scandiqa-sv (reading comprehension) and swedn (summarization).
trap 'echo "Error at line $LINENO at $(date)" >&2' ERR

# === PATHS ===
EUROEVAL_DIR="/leonardo_work/EUHPC_D21_101/abueff00/euroeval"
MODEL_PATH="${EUROEVAL_DIR}/models/Qwen3-0.6B"
DATA_DIR="/leonardo/home/userexternal/abueff00/euroSFT/training-data/sv"
OUTPUT_DIR="${EUROEVAL_DIR}/dpo-preference-data/qwen3-0.6b-rejected"
CODE_DIR="/leonardo/home/userexternal/abueff00/euroSFT/code"
PROJECT_DIR="/leonardo/home/userexternal/abueff00/euroSFT"
PIP_PACKAGES="/leonardo_work/EUHPC_E05_119/abueff00/euroSFT/pip_packages"

CONTAINER="/leonardo_work/EUHPC_D21_101/containers/smoLLM_fixed.sif"

# === Environment ===
export HF_HOME="${EUROEVAL_DIR}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "${OUTPUT_DIR}"
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
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"

# === Info ===
echo "============================================"
echo "DPO Rejected Generation — Qwen3-0.6B"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "Weak model: ${MODEL_PATH}"
echo "Data:       ${DATA_DIR}"
echo "Output:     ${OUTPUT_DIR}"
nvidia-smi || true
echo ""

# === Run generation ===
set +e
$CONTAINER_CMD exec \
    --nv \
    --bind "$PROJECT_DIR":"$PROJECT_DIR" \
    --bind "$EUROEVAL_DIR":"$EUROEVAL_DIR" \
    --bind /leonardo_work/EUHPC_E05_119/abueff00/euroSFT/pip_packages:/leonardo_work/EUHPC_E05_119/abueff00/euroSFT/pip_packages \
    --pwd "$CODE_DIR" \
    "$CONTAINER" \
    python generate_dpo_rejected.py \
        --model-path  "${MODEL_PATH}" \
        --data-dir    "${DATA_DIR}" \
        --output-dir  "${OUTPUT_DIR}" \
        --batch-size  8
EXIT_CODE=$?
set -e

echo ""
echo "============================================"
echo "Generation complete | Exit: $EXIT_CODE | End: $(date)"
echo "Output: ${OUTPUT_DIR}"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Files written:"
    ls -lh "${OUTPUT_DIR}"/*.jsonl 2>/dev/null || echo "  (no JSONL files found)"
else
    echo "Generation failed"
fi

exit $EXIT_CODE
