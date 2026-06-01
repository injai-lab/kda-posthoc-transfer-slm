# Reproducibility Notes

1. Install dependencies.
2. Prepare local JSONL files under `prepared_data/`.
3. Edit `TrainConfig` paths inside the selected script.
4. Run a wrapper under `scripts/`.
5. Smoke-check checkpoints with the matching smoke script.

Example:

```bash
bash scripts/run_qwen3_kda_stageA.sh
bash scripts/smoke_qwen3.sh /path/to/output_dir
```
