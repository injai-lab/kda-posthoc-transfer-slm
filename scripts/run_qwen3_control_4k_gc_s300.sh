#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../code"
python kimi_qwen3_control_train_stageA_4k_gc_s300.py
