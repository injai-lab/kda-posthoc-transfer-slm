#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../code"
python kimi_qwen3_kda_train_stageA.py
