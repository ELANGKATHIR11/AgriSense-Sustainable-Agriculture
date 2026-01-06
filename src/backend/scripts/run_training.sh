#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-llm}
EPOCHS=${2:-1}
BATCH=${3:-8}

python ../trainers/runner.py --model "$MODEL" --epochs "$EPOCHS" --batch-size "$BATCH"
