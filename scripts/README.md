# Scripts

Training wrappers do not pass extra CLI arguments because the uploaded training scripts use hard-coded `TrainConfig` dataclasses.

Smoke and inference wrappers use the actual argparse names detected in the uploaded scripts.
