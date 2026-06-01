#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../code"
python llama32_control_train_stageA_4k_gc_s300.py
