#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/infer_kda.sh /path/to/model_dir 'prompt text'"
  exit 1
fi
MODEL_DIR="$1"
PROMPT="$2"
cd "$(dirname "$0")/../code"
python kda_infer_check.py --model-dir "$MODEL_DIR" --prompt "$PROMPT"
