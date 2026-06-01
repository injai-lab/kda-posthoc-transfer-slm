[README.md](https://github.com/user-attachments/files/28450102/README.md)
# KDA Post-hoc Transfer Experiments for SLMs

This repository contains the code and release materials for the paper:

**A Study on the Optimization-Generation Gap in Linear Attention Substitution of SLMs: Focusing on Kimi Delta Attention Post-hoc Transfer**  
Korean title: **SLM의 선형 어텐션 치환 시 발생하는 최적화-생성 불일치에 관한 연구**

## Core claim

The experiments show an **optimization-generation gap** in post-hoc KDA substitution for SLMs:

> Teacher-forcing validation loss can decrease, while autoregressive generation remains degraded or collapsed.

## Tested models

- `microsoft/Phi-tiny-MoE-instruct`
- `Qwen/Qwen3-1.7B-Base`
- `meta-llama/Llama-3.2-1B`

## Included

- Original training scripts
- KDA loading and inference scripts
- Smoke-check scripts
- Paper-level result summaries
- Dataset documentation
- Environment files
- Paper PDF
- Audit/manifests generated from the uploaded scripts

## Excluded

- Raw datasets
- Preprocessed `prepared_data/*.jsonl`
- Model checkpoints
- Hugging Face cache
- W&B folders
- Large raw logs

## Layout

```text
.
├── code/                  # Original Python scripts, kept flat to preserve imports
├── configs/               # TrainConfig summaries generated from scripts
├── scripts/               # Safe wrapper commands
├── results/               # Paper-level result tables
├── paper/                 # Paper PDF
├── docs/                  # Reproducibility, audit, release notes
├── logs/                  # Small log summaries only
└── meta/                  # Manifests, code list, checksums, argparse report
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Important

The uploaded training scripts are preserved as original experiment scripts.  
Some paths still point to the original environment, for example:

```text
/root/kda_project/...
/data/pepe_files/kda_project/...
```

Before running training, edit the relevant `TrainConfig` fields inside the Python script:

- `output_dir`
- `cache_dir`
- `prepared_data_dir`
- `init_from_local_dir`

See `docs/AUDIT.md` and `meta/absolute_paths_detected.txt`.

## Example commands

```bash
bash scripts/run_qwen3_kda_stageA.sh
bash scripts/run_qwen3_control_4k_gc_s300.sh
bash scripts/run_llama32_kda_epoch1.sh
bash scripts/smoke_qwen3.sh /path/to/output_dir
bash scripts/infer_kda.sh /path/to/model_dir "Hello my name is"
```

## Results

- `results/main_results.csv`
- `results/failure_examples.csv`

## Citation

See `CITATION.cff`.

## Contributors

