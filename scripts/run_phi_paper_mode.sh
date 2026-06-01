#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../code"
python kimi_phi_kda_train_paper_mode.py
