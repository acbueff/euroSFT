#!/bin/bash
# Quick-reference helper for deploying to Leonardo HPC
#
# Full workflow:
#   1. ./setup_leonardo.sh          # upload code+data, download model, build container
#   2. ssh abueff00@login.leonardo.cineca.it
#   3. cd /leonardo/home/userexternal/abueff00/euroSFT
#   4. sbatch run_leonardo.sh       # submit SLURM job
#   5. squeue -u abueff00           # check job status
#   6. tail -f sft-swedish_<JOBID>.out  # monitor training

set -euo pipefail

echo "=== SFT Swedish Pipeline ==="
echo ""
echo "Step 1: Setup Leonardo (from local machine)"
echo "  ./setup_leonardo.sh"
echo ""
echo "Step 2: Submit job (on Leonardo)"
echo "  ssh abueff00@login.leonardo.cineca.it"
echo "  cd /leonardo/home/userexternal/abueff00/euroSFT"
echo "  sbatch run_leonardo.sh"
echo ""
echo "Step 3: Monitor"
echo "  squeue -u abueff00"
echo "  tail -f sft-swedish_<JOBID>.out"
