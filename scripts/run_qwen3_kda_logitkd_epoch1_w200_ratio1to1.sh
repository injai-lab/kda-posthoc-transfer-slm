#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../code"
python kimi_qwen3_kda_train_stageA_logitkd_epoch1_w200_ratio1to1.py
