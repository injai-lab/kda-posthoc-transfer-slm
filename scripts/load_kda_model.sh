#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/load_kda_model.sh /path/to/model_dir"
  exit 1
fi
MODEL_DIR="$1"
cd "$(dirname "$0")/../code"
python load_kda_stageA.py --model-dir "$MODEL_DIR"
