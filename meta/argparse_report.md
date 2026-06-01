# Argparse Report

CLI arguments detected from uploaded Python files.

## `kda_infer_check.py`
- `--model-dir`: `"--model-dir", required=True`
- `--device`: `"--device", default="cuda"`
- `--dtype`: `"--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]`
- `--max-new-tokens`: `"--max-new-tokens", type=int, default=32`
- `--prompt`: `"--prompt", action="append", required=True`
## `kda_infer_check_1to1.py`
- `--model-dir`: `"--model-dir", required=True`
- `--device`: `"--device", default="cuda"`
- `--dtype`: `"--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]`
- `--max-new-tokens`: `"--max-new-tokens", type=int, default=32`
- `--prompt`: `"--prompt", action="append", required=True`
## `load_kda_stageA.py`
- `--model-dir`: `"--model-dir", required=True, help="Directory containing final checkpoint + kda_reload_manifest.json"`
- `--device`: `"--device", default="cuda"`
- `--dtype`: `"--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]`
- `--strict`: `"--strict", action="store_true", help="Use strict state_dict load."`
## `smoke_checkpoints.py`
- `--out-dir`: `"--out-dir", required=True`
- `--manifest-dir`: `"--manifest-dir", default=None`
- `--device`: `"--device", default="cuda"`
- `--dtype`: `"--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]`
- `--max-new-tokens`: `"--max-new-tokens", type=int, default=16`
- `--prompt`: `"--prompt", action="append"`
## `smoke_checkpoints_1to1.py`
- `--out-dir`: `"--out-dir", required=True`
- `--manifest-dir`: `"--manifest-dir", default=None`
- `--device`: `"--device", default="cuda"`
- `--dtype`: `"--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]`
- `--max-new-tokens`: `"--max-new-tokens", type=int, default=16`
- `--prompt`: `"--prompt", action="append"`
## `smoke_checkpoints_llama.py`
- `--out-dir`: `"--out-dir", required=True`
- `--manifest-dir`: `"--manifest-dir", default=None`
- `--device`: `"--device", default="cuda"`
- `--dtype`: `"--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]`
- `--max-new-tokens`: `"--max-new-tokens", type=int, default=16`
- `--prompt`: `"--prompt", action="append"`
