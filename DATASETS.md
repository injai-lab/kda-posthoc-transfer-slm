# Datasets

This repository documents the dataset interface used by the training scripts.

## Paper-level description

The paper describes a **3-way mixture**:

1. Long
2. General
3. Logic

For the Qwen experiment, the paper reports:

- Training windows: `170,083`
- Validation windows: `1,000`
- Maximum sequence length: `4096`
- Batch size: `1`
- Gradient accumulation: `4`

## Expected local files

```text
prepared_data/
  long.jsonl
  long_train.jsonl
  long_val.jsonl
  general.jsonl
  general_train.jsonl
  general_val.jsonl
  logic.jsonl
  logic_train.jsonl
  logic_val.jsonl
```

## Dataset names found in the code

- `Open-Orca/OpenOrca`
- `meta-math/MetaMathQA`
- `ise-uiuc/Magicoder-OSS-Instruct-75K`

## Long data limitation

The Long branch is referenced through local preprocessed files such as `long_train.jsonl` and `long_val.jsonl`.

The uploaded files do **not** identify the exact raw source of the Long corpus.  
Therefore, this repository documents the local prepared-data interface without inventing a raw source.

## Why data is not included

Raw and preprocessed datasets are excluded because of size, redistribution/license constraints, and repository hygiene.
