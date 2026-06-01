#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/infer_kda_1to1.sh /path/to/model_dir 'prompt text'"
  exit 1
fi
MODEL_DIR="$1"
PROMPT="$2"
cd "$(dirname "$0")/../code"
python kda_infer_check_1to1.py --model-dir "$MODEL_DIR" --prompt "$PROMPT"
