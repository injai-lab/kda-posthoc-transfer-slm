# Repository Audit

Generated: 2026-06-01T02:57:52.536111+00:00

## Files copied

- Python files copied: `19`
- Missing Python files: `0`
- Paper PDF included: `yes`

## Accuracy improvements in v3

1. All uploaded Python files are copied into a flat `code/` directory to preserve local imports.
2. CLI wrapper scripts use argparse names detected from the actual Python files.
3. Experiment metadata was extracted from each `TrainConfig` into:
   - `meta/experiment_manifest.csv`
   - `configs/*.yaml`
4. Absolute paths detected in the original scripts are listed in:
   - `meta/absolute_paths_detected.txt`
5. Syntax was checked by Python AST parsing and saved in:
   - `meta/syntax_ast_parse_report.txt`

## Known limitations

This is a clean release bundle, not a fully refactored package.

The original training scripts still contain environment-specific absolute paths.  
The Long dataset's raw origin is not identifiable from the uploaded files, so `DATASETS.md` documents it conservatively.
