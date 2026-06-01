#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../code"
python llama32_kda_train_stageA_3to1_ce_sanity100.py
