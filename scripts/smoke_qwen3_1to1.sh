#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/smoke_qwen3_1to1.sh /path/to/output_dir [manifest_dir]"
  exit 1
fi
OUT_DIR="$1"
MANIFEST_DIR="${2:-}"
cd "$(dirname "$0")/../code"
if [ -n "$MANIFEST_DIR" ]; then
  python smoke_checkpoints_1to1.py --out-dir "$OUT_DIR" --manifest-dir "$MANIFEST_DIR"
else
  python smoke_checkpoints_1to1.py --out-dir "$OUT_DIR"
fi
