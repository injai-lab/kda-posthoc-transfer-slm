#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kimi-style KDA training on Phi-tiny-MoE-instruct.

Goal:
- Keep 3:1 replacement (KDA 3 layers : original attention 1 layer).
- Use KDA kernel path aligned with official Kimi implementation:
  fused_kda_gate + chunk_kda/fused_recurrent_kda + sigmoid-gated RMSNorm output.
- Freeze parent model and train only selected KDA params.
- Robust against prior failure points:
  - rope_scaling='default' crash
  - dataset window shortage crash
  - trainable params = 0 / no gradient flow under gradient checkpointing
  - TrainingArguments API differences across transformers versions
  - save-time tied weight key issues

This script is intended for Colab/A100-style environments.
"""

import gc
import glob
import inspect
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets, load_dataset
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
import transformers.modeling_attn_mask_utils as attn_mask_utils
import transformers.modeling_utils as modeling_utils
import transformers.utils as _tu
import transformers.utils.import_utils as _tiu


# ---------------------------------------------------------------------------
# 0) Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_id: str = "microsoft/Phi-tiny-MoE-instruct"
    model_revision: str = "2fe50e88d0e2a5a132563815686ea0dcc8e252b5"
    init_from_local_dir: str = "/root/kda_project/Kimi_Project/checkpoints_stage1_1000/checkpoint-400"
    output_dir: str = "/root/kda_project/Kimi_Project/checkpoints_stage2_full_kda"
    final_dir_name: str = "final_stage2_full_kda"
    cache_dir: str = "/root/kda_project/huggingface_cache"
    multi_user_safe: bool = False
    run_name: str = ""  # optional fixed run-id. empty => auto "{user}_{timestamp}"

    seed: int = 42
    max_seq_length: int = 4096
    kda_conv_size: int = 4
    kda_mode: str = "chunk"  # chunk | fused_recurrent (training uses chunk only)
    kda_use_unpad: bool = False  # False = safer cross-version path
    # Use native depthwise conv1d to bypass FLA Triton causal_conv1d kernel instability.
    use_native_short_conv: bool = False
    # Use native gated RMSNorm to bypass FLA Triton fused norm instability.
    use_native_rmsnorm_gated: bool = False
    # Force pure PyTorch KDA core (stable, slower) instead of Triton chunk/fused kernels.
    force_python_kda_core: bool = False
    use_fused_kda_gate: bool = False
    # Keep False for full backward through the KDA core.
    detach_kda_core_in_training: bool = False
    # Current KDA path expects q/k/v head counts to match for chunk/fused kernels.
    # Keep True for Phi GQA models unless you also implement grouped-state kernels.
    expand_kv_heads_for_kda: bool = True
    # 0.0 keeps previous behavior; set 0.1~0.5 to start with slower decay gates.
    dt_bias_init: float = 0.2
    # A_log init controls fallback gate decay magnitude: larger -> stronger (more negative) decay.
    a_log_min: float = 1.0
    a_log_max: float = 16.0
    # Only used on python-gate fallback path (or fused gate runtime fallback on illegal memory access).
    # Keep 1.0 for original behavior, lower (e.g. 0.5) if DeltaWatch barely moves.
    gate_scale: float = 1.0
    # Core-kernel parity knobs (FLA chunk/fused path).
    use_qk_l2norm_in_kernel: bool = True
    use_gate_in_kernel: bool = True

    # train mode:
    # g_only     -> KDA gates + o_norm (+optionally convs via enable_conv_in_g_only)
    # g_and_out  -> g_only + o_proj
    # full_kda   -> q/k/v/proj+convs+gates all
    # gqa_lora   -> keep RoPE, train LoRA on original(GQA) q/k projections
    # gqa_lora_nope -> disable RoPE on original(GQA) layers + train LoRA on q/k
    # full_kda_gqa_lora -> full_kda + GQA LoRA (RoPE kept)
    # full_kda_gqa_lora_nope -> full_kda + GQA LoRA + NoPE on original GQA
    train_mode: str = "full_kda"
    enable_conv_in_g_only: bool = True
    gqa_lora_rank: int = 16
    gqa_lora_alpha: int = 32
    gqa_lora_dropout: float = 0.05
    gqa_lora_include_v_proj: bool = False

    batch_size: int = 1
    grad_accum: int = 32
    num_epochs: int = 2
    max_steps: int = 0

    learning_rate: float = 6.0e-6
    warmup_steps: int = 120
    weight_decay: float = 0.01
    max_grad_norm: float = 0.8
    lr_scheduler_type: str = "cosine"

    use_checkpointing: bool = False
    resume_from_last: bool = False
    skip_if_final_exists: bool = False
    force_retrain: bool = True
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    enable_pin_checkpoints: bool = True
    pin_checkpoint_steps: Tuple[int, ...] = (500, 1000)
    group_by_length: bool = False
    dataloader_num_workers: int = 0

    # dataset sizing:
    # stage-1 default for stability/speed. set to 0 for auto sizing by min/max bounds.
    target_train_windows: int = 250_000
    target_val_windows: int = 1_000
    auto_train_windows_min: int = 6_000
    auto_train_windows_max: int = 16_000
    auto_val_windows_min: int = 400
    auto_val_windows_max: int = 1_200
    force_exact_target_windows: bool = False

    # auto mix ratios (renormalized among available sources)
    mix_ratio_long: float = 0.22
    mix_ratio_short: float = 0.58
    mix_ratio_math: float = 0.20

    # Stable data profile (general + logic)
    general_dataset_id: str = "Open-Orca/OpenOrca"
    general_num_raw: int = 80_000
    general_val_rows: int = 2_000
    logic_dataset_ids: Tuple[str, ...] = (
        "meta-math/MetaMathQA",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
    )
    logic_num_raw_per_source: int = 20_000
    logic_val_rows: int = 1_200
    logic_fallback_to_synth_math: bool = True
    prefer_local_prepared_data: bool = True
    require_local_prepared_data: bool = True
    prepared_data_dir: str = "/data/pepe_files/kda_project/Kimi_Project/prepared_data"
    prepared_long_file: str = "long.jsonl"
    prepared_long_train_file: str = "long_train.jsonl"
    prepared_long_val_file: str = "long_val.jsonl"
    prepared_general_file: str = "general.jsonl"
    prepared_general_train_file: str = "general_train.jsonl"
    prepared_general_val_file: str = "general_val.jsonl"
    prepared_logic_file: str = "logic.jsonl"
    prepared_logic_train_file: str = "logic_train.jsonl"
    prepared_logic_val_file: str = "logic_val.jsonl"

    # legacy synthetic-math fallback knobs
    use_math_data: bool = True
    math_num_train_raw: int = 16_000
    math_num_val_raw: int = 800
    math_max_abs_int: int = 2_000
    min_output_chars: int = 1
    max_bad_char_ratio: float = 0.10
    drop_replacement_char: bool = True
    filter_mathy_in_short: bool = True

    # windowing
    long_stride_train: int = 1024
    long_stride_val: int = 2048
    min_sup_tokens_train: int = 16
    min_sup_tokens_val: int = 16
    max_windows_per_long_train: int = 6
    max_windows_per_long_val: int = 2


CFG = TrainConfig()


# ---------------------------------------------------------------------------
# 1) Environment / compatibility patches
# ---------------------------------------------------------------------------
def patch_transformers_compat() -> None:
    if not hasattr(_tiu, "is_torch_fx_available"):
        _tiu.is_torch_fx_available = lambda: False
    if not hasattr(_tu, "is_torch_fx_available"):
        _tu.is_torch_fx_available = _tiu.is_torch_fx_available

    # Clear stale remote-code modules in long-running notebooks
    for key in list(sys.modules.keys()):
        if "transformers_modules" in key and "Phi" in key:
            del sys.modules[key]


def patch_rope_scaling(cfg_obj: AutoConfig) -> AutoConfig:
    rs = getattr(cfg_obj, "rope_scaling", None)
    if isinstance(rs, dict):
        if "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]
        # Avoid "Unknown RoPE scaling type default" error in some envs
        if rs.get("type") == "default":
            cfg_obj.rope_scaling = None
    return cfg_obj


def patch_dynamic_cache() -> None:
    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, pkv):
            if pkv is None:
                return cls()
            cache = cls()
            if isinstance(pkv, tuple):
                for i, layer_cache in enumerate(pkv):
                    k, v = layer_cache
                    cache.update(k, v, i)
            return cache

        DynamicCache.from_legacy_cache = from_legacy_cache


def patch_4d_mask() -> None:
    def _safe_4d_causal_attention_mask(
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        sliding_window=None,
        cache_position=None,
    ):
        bsz, seq_len = input_shape
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        min_val = -10000.0

        causal = torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device)
        causal = torch.triu(causal, diagonal=1)
        causal = causal[None, None, :, :].expand(bsz, 1, seq_len, seq_len).contiguous()

        if attention_mask is None:
            return causal

        if attention_mask.dim() == 4:
            am = attention_mask.to(dtype)
            if am.max() <= 1.0 and am.min() >= 0.0:
                return causal + (1.0 - am) * min_val
            return am

        if attention_mask.dim() == 2:
            pm = attention_mask[:, None, None, :].to(dtype)
            return causal + (1.0 - pm) * min_val

        return causal

    attn_mask_utils._prepare_4d_causal_attention_mask = _safe_4d_causal_attention_mask


def patch_tied_weight_key_reader() -> None:
    def _safe_get_tied_weight_keys(module):
        out = []
        for name, submodule in module.named_modules():
            tied = getattr(submodule, "_tied_weights_keys", {}) or {}
            if isinstance(tied, list):
                keys = tied
            elif hasattr(tied, "keys"):
                keys = list(tied.keys())
            else:
                keys = []
            out.extend([f"{name}.{k}" if name else k for k in keys])
        return out

    if hasattr(modeling_utils, "_get_tied_weight_keys"):
        modeling_utils._get_tied_weight_keys = _safe_get_tied_weight_keys


def normalize_tied_weight_keys(model: nn.Module) -> int:
    fixed = 0
    for _, mod in model.named_modules():
        twk = getattr(mod, "_tied_weights_keys", None)
        if isinstance(twk, list):
            mod._tied_weights_keys = {k: None for k in twk}
            fixed += 1
        elif twk is not None and not hasattr(twk, "keys"):
            mod._tied_weights_keys = {}
            fixed += 1
    return fixed


def has_complete_saved_model(model_dir: str) -> bool:
    single = os.path.join(model_dir, "model.safetensors")
    index_json = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.exists(single):
        return True
    if not os.path.exists(index_json):
        return False

    try:
        with open(index_json, "r", encoding="utf-8") as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
    except Exception:
        return False

    return all(os.path.exists(os.path.join(model_dir, s)) for s in shards)


def write_kda_reload_manifest(cfg: TrainConfig, final_dir: str) -> str:
    """
    Save minimal metadata required to rebuild KDA-augmented architecture before loading weights.
    This enables Option-A reload:
      1) load base Phi model
      2) apply replace_attention_3to1 with same KDA settings
      3) load saved checkpoint weights
    """
    manifest = {
        "schema_version": 1,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "replace_rule": "layer_idx_mod_4_ne_3",
        "kda_config": {
            "kda_conv_size": cfg.kda_conv_size,
            "kda_mode": cfg.kda_mode,
            "kda_use_unpad": cfg.kda_use_unpad,
            "use_native_short_conv": cfg.use_native_short_conv,
            "use_native_rmsnorm_gated": cfg.use_native_rmsnorm_gated,
            "force_python_kda_core": cfg.force_python_kda_core,
            "use_fused_kda_gate": cfg.use_fused_kda_gate,
            "detach_kda_core_in_training": cfg.detach_kda_core_in_training,
            "expand_kv_heads_for_kda": cfg.expand_kv_heads_for_kda,
            "dt_bias_init": cfg.dt_bias_init,
            "a_log_min": cfg.a_log_min,
            "a_log_max": cfg.a_log_max,
            "gate_scale": cfg.gate_scale,
            "use_qk_l2norm_in_kernel": cfg.use_qk_l2norm_in_kernel,
            "use_gate_in_kernel": cfg.use_gate_in_kernel,
        },
    }
    out_path = os.path.join(final_dir, "kda_reload_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out_path


def _sanitize_run_name(name: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name).strip("_")
    return safe or f"run_{int(time.time())}"


def apply_run_isolation(cfg: TrainConfig) -> None:
    if not cfg.multi_user_safe:
        return
    # If already isolated, keep it.
    if "/run_" in cfg.output_dir or cfg.output_dir.endswith("/runs"):
        return

    base_dir = cfg.output_dir
    run_name = (cfg.run_name or "").strip() or os.environ.get("KDA_RUN_NAME", "").strip()
    if not run_name:
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
        run_name = f"{user}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_name = _sanitize_run_name(run_name)
    cfg.output_dir = os.path.join(base_dir, f"run_{run_name}")


def setup_runtime(cfg: TrainConfig) -> None:
    os.environ["HF_HOME"] = cfg.cache_dir
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    try:
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        else:
            torch.backends.cuda.matmul.allow_tf32 = True

        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    set_seed(cfg.seed)


patch_transformers_compat()
patch_dynamic_cache()
patch_4d_mask()
patch_tied_weight_key_reader()
apply_run_isolation(CFG)
setup_runtime(CFG)
print(f"[paths] output_dir={CFG.output_dir}")
print(f"[paths] final_dir={os.path.join(CFG.output_dir, CFG.final_dir_name)}")

final_dir = os.path.join(CFG.output_dir, CFG.final_dir_name)
if CFG.skip_if_final_exists and (not CFG.force_retrain) and has_complete_saved_model(final_dir):
    existing_files = sorted(glob.glob(os.path.join(final_dir, "model*.safetensors")))
    if not existing_files:
        existing_files = sorted(glob.glob(os.path.join(final_dir, "pytorch_model*.bin")))
    print(f"[SKIP] Final model already exists: {final_dir}")
    print("Weight files:", [os.path.basename(x) for x in existing_files])
    raise SystemExit(0)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Hard dependency checks (avoid silent wrong-path runs)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for KDA training, but torch.cuda.is_available() is False.")

try:
    require_version("accelerate>=0.26.0")
except Exception as e:
    raise RuntimeError(
        "accelerate>=0.26.0 is required by Trainer. "
        "Install with: pip install --no-deps 'accelerate>=0.26.0'"
    ) from e

try:
    import flash_attn  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "flash_attn is required by this Phi remote-code model. "
        "Install with: pip install flash-attn --no-build-isolation and restart runtime."
    ) from e

FLA_FusedRMSNormGated = None
chunk_kda = None
fused_recurrent_kda = None

try:
    from fla.modules import ShortConvolution as FLA_ShortConvolution
    try:
        from fla.modules import FusedRMSNormGated as FLA_FusedRMSNormGated
    except Exception:
        FLA_FusedRMSNormGated = None
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    except Exception:
        chunk_kda = None
        fused_recurrent_kda = None
    from fla.ops.kda.gate import fused_kda_gate

    # Keep unpad path optional. On some stacks fla.layers import triggers extra Triton modules.
    try:
        from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
    except Exception:
        def get_unpad_data(*args, **kwargs):
            raise RuntimeError(
                "kda_use_unpad=True requires fla.layers.utils import, "
                "which is unavailable on this stack. Set kda_use_unpad=False."
            )

        def index_first_axis(*args, **kwargs):
            raise RuntimeError(
                "kda_use_unpad=True requires fla.layers.utils import, "
                "which is unavailable on this stack. Set kda_use_unpad=False."
            )

        def pad_input(*args, **kwargs):
            raise RuntimeError(
                "kda_use_unpad=True requires fla.layers.utils import, "
                "which is unavailable on this stack. Set kda_use_unpad=False."
            )
except Exception as e:
    raise RuntimeError(
        "FLA package namespace 'fla' is required for KDA path. "
        "Install with: pip install -U flash-linear-attention (pulls fla-core), "
        "then restart runtime."
    ) from e


# ---------------------------------------------------------------------------
# FLA API compatibility helpers
# ---------------------------------------------------------------------------
class NativeShortConvolution(nn.Module):
    """Drop-in replacement for FLA ShortConvolution using native depthwise conv1d."""

    def __init__(self, hidden_size: int, kernel_size: int, activation: str = "silu"):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.kernel_size = max(1, int(kernel_size))
        self.conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            groups=self.hidden_size,
            bias=True,
        )
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation in ("none", "identity", None):
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported native short-conv activation: {activation}")

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del cache, cu_seqlens  # cache path is unused in this training script
        if x.dim() != 3:
            raise ValueError(f"NativeShortConvolution expects [B, T, C], got {tuple(x.shape)}")

        # Causal depthwise conv over time dimension.
        xt = x.transpose(1, 2)  # [B, C, T]
        if self.kernel_size > 1:
            xt = torch.nn.functional.pad(xt, (self.kernel_size - 1, 0))
        y = self.conv(xt).transpose(1, 2)  # [B, T, C]
        y = self.act(y)

        final_state = None
        if output_final_state:
            # Keep API compatibility; recurrent cache is not used in current train path.
            keep = max(0, self.kernel_size - 1)
            final_state = x[:, -keep:, :] if keep > 0 else x[:, :0, :]
        return y, final_state


class NativeFusedRMSNormGated(nn.Module):
    """Pure PyTorch replacement for fused gated RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, activation: str = "sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)
        self.activation = activation

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        inv_rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = (x32 * inv_rms).to(x.dtype) * self.weight.to(x.dtype)

        if self.activation == "sigmoid":
            g = torch.sigmoid(gate.float()).to(gate.dtype)
        elif self.activation == "silu":
            g = torch.nn.functional.silu(gate.float()).to(gate.dtype)
        else:
            raise ValueError(f"Unsupported gate activation: {self.activation}")
        return y * g


def _safe_python_kda_gate(
    g: torch.Tensor,
    a_log: torch.Tensor,
    head_dim: int,
    dt_bias: torch.Tensor,
    decay_scale: float = 1.0,
) -> torch.Tensor:
    # Fallback if fused_kda_gate is unavailable/unstable in current wheel build.
    # Input g: [B, T, H*D] or [B, T, H, D]
    if g.dim() == 3:
        if g.size(-1) % head_dim != 0:
            raise ValueError(f"Cannot reshape g of last dim {g.size(-1)} with head_dim={head_dim}")
        num_heads = g.size(-1) // head_dim
        g = rearrange(g, "... (h d) -> ... h d", h=num_heads)
    elif g.dim() != 4:
        raise ValueError(f"Unexpected g rank for fallback gate: {g.dim()}")

    # dt > 0 then log-decay-like gate signal
    db = dt_bias.view(1, 1, -1, head_dim).to(g.dtype)
    dt = torch.nn.functional.softplus(g + db)
    decay = (-torch.exp(a_log.float())).to(dt.dtype) * dt * float(decay_scale)
    return decay


def normalize_kda_gate_shape(g: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    # Normalize gate tensor to [B, T, H, D] to satisfy chunk_kda assert.
    if g.dim() == 3:
        if g.size(-1) != num_heads * head_dim:
            raise ValueError(
                f"Unexpected gate hidden size: got {g.size(-1)}, expected {num_heads * head_dim}."
            )
        return rearrange(g, "b t (h d) -> b t h d", h=num_heads)

    if g.dim() == 4:
        if g.size(-2) == num_heads and g.size(-1) == head_dim:
            return g
        if g.size(-1) == num_heads * head_dim:
            return rearrange(g, "... (h d) -> ... h d", h=num_heads)
        raise ValueError(f"Unexpected 4D gate shape: {tuple(g.shape)}")

    raise ValueError(f"Unsupported gate rank: {g.dim()}")


def call_fused_kda_gate(g, a_log, head_dim, dt_bias, decay_scale: float = 1.0):
    # Signature-driven dispatch avoids calling wrong kernel argument order.
    try:
        sig = inspect.signature(fused_kda_gate)
        param_names = set(sig.parameters.keys())
    except Exception:
        param_names = set()

    # Keep gate params in fp32 for numerical/kernel stability.
    a32 = a_log.float()
    d32 = dt_bias.float()

    try:
        if "g_bias" in param_names and "head_dim" in param_names:
            return fused_kda_gate(g, a32, head_dim=head_dim, g_bias=d32)
        if "g_bias" in param_names:
            return fused_kda_gate(g, a32, g_bias=d32)
        if "dt_bias" in param_names and "head_dim" in param_names:
            return fused_kda_gate(g, a32, head_dim=head_dim, dt_bias=d32)
        if "dt_bias" in param_names:
            return fused_kda_gate(g, a32, dt_bias=d32)

        # Fallback positional paths for older wheels.
        try:
            return fused_kda_gate(g, a32, head_dim, d32)
        except Exception:
            return fused_kda_gate(g, a32, d32)
    except RuntimeError as e:
        # Some wheel/driver combos throw triton illegal memory access.
        if "illegal memory access" in str(e).lower():
            return _safe_python_kda_gate(g, a32, head_dim, d32, decay_scale=decay_scale)
        raise


def python_delta_rule_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Pure PyTorch delta-rule KDA core.
    Shapes:
      q/k/v/g: [B, T, H, D]
      beta:    [B, T, H]
      state:   [B, H, D, D]
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or g.dim() != 4:
        raise ValueError("q/k/v/g must be 4D tensors [B,T,H,D].")
    if beta.dim() != 3:
        raise ValueError("beta must be a 3D tensor [B,T,H].")
    if q.shape != k.shape or q.shape != v.shape or q.shape != g.shape:
        raise ValueError("q/k/v/g shape mismatch.")
    if beta.shape != q.shape[:3]:
        raise ValueError(f"beta shape mismatch: got {tuple(beta.shape)}, expected {tuple(q.shape[:3])}.")

    bsz, seq_len, num_heads, head_dim = q.shape
    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

    if initial_state is None:
        state = torch.zeros(
            bsz, num_heads, head_dim, head_dim,
            device=q.device,
            dtype=compute_dtype,
        )
    else:
        expected_state = (bsz, num_heads, head_dim, head_dim)
        if tuple(initial_state.shape) != expected_state:
            raise ValueError(
                f"initial_state shape mismatch: got {tuple(initial_state.shape)}, expected {expected_state}."
            )
        state = initial_state.to(dtype=compute_dtype, device=q.device)

    outs = []
    for t in range(seq_len):
        qt = q[:, t].to(compute_dtype)
        kt = k[:, t].to(compute_dtype)
        vt = v[:, t].to(compute_dtype)
        gt = g[:, t].to(compute_dtype)
        bt = beta[:, t].to(compute_dtype)

        # diag(exp(g_t)) @ S_{t-1}
        decay = torch.exp(gt).unsqueeze(-1)          # [B,H,D,1]
        state = state * decay

        # + beta_t * (k_t \otimes v_t)
        outer = torch.einsum("bhd,bhe->bhde", kt, vt)
        state = state + bt.unsqueeze(-1).unsqueeze(-1) * outer

        # o_t = q_t @ S_t
        ot = torch.einsum("bhd,bhde->bhe", qt, state)
        outs.append(ot.to(q.dtype))

    o = torch.stack(outs, dim=1)  # [B,T,H,D]
    final_state = state.to(q.dtype) if output_final_state else None
    return o, final_state


def call_chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens=None,
    A_log=None,
    dt_bias=None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
):
    if chunk_kda is None:
        raise RuntimeError("FLA chunk_kda kernel is unavailable in this environment.")

    kwargs = dict(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    try:
        param_names = set(inspect.signature(chunk_kda).parameters.keys())
    except Exception:
        param_names = set()

    if "use_qk_l2norm_in_kernel" in param_names:
        kwargs["use_qk_l2norm_in_kernel"] = bool(use_qk_l2norm_in_kernel)
    if "use_gate_in_kernel" in param_names:
        kwargs["use_gate_in_kernel"] = bool(use_gate_in_kernel)
    if use_gate_in_kernel:
        if A_log is not None:
            kwargs["A_log"] = A_log.float().reshape(-1)
        if dt_bias is not None:
            kwargs["dt_bias"] = dt_bias.float().reshape(-1)
    return chunk_kda(**kwargs)


def call_fused_recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens=None,
    A_log=None,
    dt_bias=None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
):
    if fused_recurrent_kda is None:
        raise RuntimeError("FLA fused_recurrent_kda kernel is unavailable in this environment.")
    kwargs = dict(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    try:
        param_names = set(inspect.signature(fused_recurrent_kda).parameters.keys())
    except Exception:
        param_names = set()

    if "use_qk_l2norm_in_kernel" in param_names:
        kwargs["use_qk_l2norm_in_kernel"] = bool(use_qk_l2norm_in_kernel)
    if "use_gate_in_kernel" in param_names:
        kwargs["use_gate_in_kernel"] = bool(use_gate_in_kernel)
    if use_gate_in_kernel:
        if A_log is not None:
            kwargs["A_log"] = A_log.float().reshape(-1)
        if dt_bias is not None:
            kwargs["dt_bias"] = dt_bias.float().reshape(-1)
    return fused_recurrent_kda(**kwargs)


# ---------------------------------------------------------------------------
# 2) KDA module (Phi-compatible, Kimi-kernel path)
# ---------------------------------------------------------------------------
class KimiDeltaAttentionPhi(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        num_kv_heads: int,
        layer_idx: int,
        conv_size: int = 4,
        attn_bias: bool = False,
        rms_eps: float = 1e-5,
        mode: str = "chunk",
        use_unpad: bool = False,
        use_fused_gate: bool = False,
        detach_kda_core_in_training: bool = False,
        expand_kv_heads_for_kda: bool = True,
        dt_bias_init: float = 0.0,
        a_log_min: float = 1.0,
        a_log_max: float = 16.0,
        gate_scale: float = 1.0,
        use_qk_l2norm_in_kernel: bool = True,
        use_gate_in_kernel: bool = True,
    ):
        super().__init__()
        if q_dim % num_heads != 0:
            raise ValueError(f"q_dim({q_dim}) must be divisible by num_heads({num_heads})")
        if kv_dim % num_kv_heads != 0:
            raise ValueError(f"kv_dim({kv_dim}) must be divisible by num_kv_heads({num_kv_heads})")

        self.hidden_size = hidden_size
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = q_dim // num_heads
        self.kv_head_dim = kv_dim // num_kv_heads
        if self.kv_head_dim != self.head_dim:
            raise ValueError(
                f"head_dim mismatch: q_head_dim={self.head_dim}, kv_head_dim={self.kv_head_dim}."
            )

        self.layer_idx = layer_idx
        self.mode = mode
        self.conv_size = conv_size
        self.use_unpad = use_unpad
        self.use_fused_gate = use_fused_gate
        self.detach_kda_core_in_training = detach_kda_core_in_training
        self.expand_kv_heads_for_kda = expand_kv_heads_for_kda
        self.gate_scale = float(gate_scale)
        self.use_qk_l2norm_in_kernel = bool(use_qk_l2norm_in_kernel)
        self.use_gate_in_kernel = bool(use_gate_in_kernel)

        self.q_proj = nn.Linear(hidden_size, q_dim, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=attn_bias)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=attn_bias)

        # Kimi-style short conv before q/k/v
        conv_cls = NativeShortConvolution if CFG.use_native_short_conv else FLA_ShortConvolution
        if CFG.use_native_short_conv and self.layer_idx == 0:
            print("[WARN] Using NativeShortConvolution (nn.Conv1d) to bypass FLA Triton causal_conv1d.")
        if CFG.use_native_short_conv:
            self.q_conv1d = conv_cls(hidden_size=q_dim, kernel_size=conv_size, activation="silu")
            self.k_conv1d = conv_cls(hidden_size=kv_dim, kernel_size=conv_size, activation="silu")
            self.v_conv1d = conv_cls(hidden_size=kv_dim, kernel_size=conv_size, activation="silu")
        else:
            self.q_conv1d = conv_cls(hidden_size=q_dim, kernel_size=conv_size, bias=False, activation="silu")
            self.k_conv1d = conv_cls(hidden_size=kv_dim, kernel_size=conv_size, bias=False, activation="silu")
            self.v_conv1d = conv_cls(hidden_size=kv_dim, kernel_size=conv_size, bias=False, activation="silu")

        # Kimi-style gated delta path
        a_min = max(1e-6, float(min(a_log_min, a_log_max)))
        a_max = max(a_min + 1e-6, float(max(a_log_min, a_log_max)))
        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(a_min, a_max)).view(1, 1, -1, 1)
        )
        self.f_a_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, q_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(q_dim, dtype=torch.float32))
        nn.init.constant_(self.dt_bias, float(dt_bias_init))

        # beta_t per token/head
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # output gate + gated RMSNorm
        self.g_a_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, q_dim, bias=False)
        if CFG.use_native_rmsnorm_gated:
            self.o_norm = NativeFusedRMSNormGated(self.head_dim, eps=rms_eps, activation="sigmoid")
        else:
            if FLA_FusedRMSNormGated is None:
                raise RuntimeError(
                    "use_native_rmsnorm_gated=False requires FLA FusedRMSNormGated import to succeed."
                )
            self.o_norm = FLA_FusedRMSNormGated(self.head_dim, eps=rms_eps, activation="sigmoid")

    @staticmethod
    def _mask4d_to_2d(attention_mask_4d: torch.Tensor) -> torch.Tensor:
        diag = attention_mask_4d[:, 0].diagonal(dim1=-2, dim2=-1)
        return (diag > -5000).to(torch.long)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[DynamicCache]]:
        del position_ids  # not used in this KDA path

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training and mode != "chunk":
            mode = "chunk"

        padding_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                padding_mask = attention_mask.to(torch.long)
            elif attention_mask.dim() == 4:
                padding_mask = self._mask4d_to_2d(attention_mask)
            else:
                maybe_padding = kwargs.get("padding_mask", None)
                if maybe_padding is not None and maybe_padding.dim() == 2:
                    padding_mask = maybe_padding.to(torch.long)

        cu_seqlens = kwargs.get("cu_seqlens", None)
        indices = None
        hs = hidden_states
        if self.use_unpad and padding_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(padding_mask[:, -q_len:])
            hs = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        # No recurrent cache path in training code; keep interface compatibility.
        q, _ = self.q_conv1d(
            x=self.q_proj(hs),
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        k, _ = self.k_conv1d(
            x=self.k_proj(hs),
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        v, _ = self.v_conv1d(
            x=self.v_proj(hs),
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )

        # Delta-rule gate and beta (official Kimi pattern)
        g_raw = self.f_b_proj(self.f_a_proj(hs))
        core_use_gate_in_kernel = bool(self.use_gate_in_kernel and not CFG.force_python_kda_core)
        if core_use_gate_in_kernel:
            # Let chunk/fused kernels compute gate decay from raw projection + (A_log, dt_bias).
            g_core = normalize_kda_gate_shape(g_raw, self.num_heads, self.head_dim)
        else:
            if self.use_fused_gate:
                g_core = call_fused_kda_gate(
                    g_raw,
                    self.A_log,
                    self.head_dim,
                    self.dt_bias,
                    decay_scale=self.gate_scale,
                )
            else:
                g_core = _safe_python_kda_gate(
                    g_raw,
                    self.A_log,
                    self.head_dim,
                    self.dt_bias,
                    decay_scale=self.gate_scale,
                )
            g_core = normalize_kda_gate_shape(g_core, self.num_heads, self.head_dim)
        beta = self.b_proj(hs).float().sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads)
        k = rearrange(k, "... (h d) -> ... h d", h=self.num_kv_heads)
        v = rearrange(v, "... (h d) -> ... h d", h=self.num_kv_heads)

        # Expand KV heads to full heads when using kernels that expect matched head counts.
        if self.num_kv_heads != self.num_heads:
            if not self.expand_kv_heads_for_kda:
                raise ValueError(
                    "num_kv_heads != num_heads requires grouped-state KDA kernel support. "
                    "Set expand_kv_heads_for_kda=True for current chunk/fused path."
                )
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=-2)
            v = v.repeat_interleave(rep, dim=-2)

        if g_core.dtype != q.dtype:
            g_core = g_core.to(q.dtype)

        if CFG.force_python_kda_core:
            o, _ = python_delta_rule_kda(
                q=q, k=k, v=v, g=g_core, beta=beta, initial_state=None, output_final_state=False
            )
        elif mode == "chunk":
            o, _ = call_chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g_core,
                beta=beta,
                cu_seqlens=cu_seqlens,
                A_log=self.A_log if core_use_gate_in_kernel else None,
                dt_bias=self.dt_bias if core_use_gate_in_kernel else None,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                use_gate_in_kernel=core_use_gate_in_kernel,
            )
        else:
            o, _ = call_fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g_core,
                beta=beta,
                cu_seqlens=cu_seqlens,
                A_log=self.A_log if core_use_gate_in_kernel else None,
                dt_bias=self.dt_bias if core_use_gate_in_kernel else None,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                use_gate_in_kernel=core_use_gate_in_kernel,
            )
        if self.training and self.detach_kda_core_in_training:
            o = o.detach()
        o = torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)

        go = self.g_b_proj(self.g_a_proj(hs))
        go = rearrange(go, "... (h d) -> ... h d", h=self.num_heads)
        go = torch.nan_to_num(go, nan=0.0, posinf=0.0, neginf=0.0)
        o = self.o_norm(o, go)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        if self.use_unpad and padding_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        # Keep compatible return tuple for decoder attention module
        return o, None, past_key_value


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


# ---------------------------------------------------------------------------
# 3) Model utilities
# ---------------------------------------------------------------------------
def _get_out_proj(attn_module: nn.Module) -> nn.Module:
    for name in ("o_proj", "dense", "out_proj"):
        if hasattr(attn_module, name):
            return getattr(attn_module, name)
    raise AttributeError("Cannot find output projection in attention module.")


def _sync_phi_attn_dims(module, _inputs=None):
    if isinstance(module, KimiDeltaAttentionPhi):
        return
    if hasattr(module, "q_proj") and hasattr(module, "num_heads"):
        q_dim = module.q_proj.weight.shape[0]
        module.hidden_size = q_dim
        module.head_dim = q_dim // module.num_heads


def apply_phi_attn_dim_fix(model: nn.Module) -> int:
    fixed = 0
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or isinstance(attn, KimiDeltaAttentionPhi):
            continue
        _sync_phi_attn_dims(attn)
        if not hasattr(attn, "_dim_fix_hook"):
            attn._dim_fix_hook = attn.register_forward_pre_hook(_sync_phi_attn_dims)
        fixed += 1
    return fixed


def apply_gqa_lora(model: nn.Module, rank: int, alpha: int, dropout: float, include_v_proj: bool = False) -> int:
    wrapped = 0
    target_proj = ["q_proj", "k_proj"] + (["v_proj"] if include_v_proj else [])
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or isinstance(attn, KimiDeltaAttentionPhi):
            continue

        for proj_name in target_proj:
            proj = getattr(attn, proj_name, None)
            if proj is None or isinstance(proj, LoRALinear):
                continue
            if not isinstance(proj, nn.Linear):
                continue

            lora = LoRALinear(
                base=proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ).to(device=proj.weight.device, dtype=proj.weight.dtype)
            setattr(attn, proj_name, lora)
            wrapped += 1
    return wrapped


def apply_nope_to_original_gqa(model: nn.Module) -> int:
    patched = 0
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or isinstance(attn, KimiDeltaAttentionPhi):
            continue
        if getattr(attn, "_nope_patch_applied", False):
            continue

        orig_forward = attn.forward

        try:
            param_names = list(inspect.signature(orig_forward).parameters.keys())
        except Exception:
            param_names = []

        def _forward_nope(*args, _orig_forward=orig_forward, _param_names=param_names, **kwargs):
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                kwargs["position_ids"] = torch.zeros_like(kwargs["position_ids"])
            elif "position_ids" in _param_names:
                idx = _param_names.index("position_ids")
                candidate_idx = []
                # Bound method calls usually map args directly to params (without self),
                # but some remote-code wrappers can expose signatures including self.
                candidate_idx.append(idx)
                if idx - 1 >= 0:
                    candidate_idx.append(idx - 1)
                for a_idx in candidate_idx:
                    if 0 <= a_idx < len(args) and args[a_idx] is not None:
                        new_args = list(args)
                        new_args[a_idx] = torch.zeros_like(new_args[a_idx])
                        args = tuple(new_args)
                        break
            return _orig_forward(*args, **kwargs)

        attn._orig_forward_nope = orig_forward
        attn.forward = _forward_nope
        attn._nope_patch_applied = True
        patched += 1
    return patched


def replace_attention_3to1(model: nn.Module, cfg: TrainConfig) -> Tuple[int, int]:
    replaced = 0
    total = 0
    for i, layer in enumerate(model.model.layers):
        if not hasattr(layer, "self_attn"):
            continue
        total += 1
        if i % 4 == 3:
            continue

        old_attn = layer.self_attn
        if not all(hasattr(old_attn, n) for n in ("q_proj", "k_proj", "v_proj")):
            continue

        old_q = old_attn.q_proj
        old_k = old_attn.k_proj
        old_v = old_attn.v_proj
        old_o = _get_out_proj(old_attn)

        q_dim = old_q.weight.shape[0]
        kv_dim = old_k.weight.shape[0]
        hidden_size = old_q.weight.shape[1]
        num_heads = getattr(old_attn, "num_heads", getattr(model.config, "num_attention_heads", None))
        if num_heads is None:
            raise ValueError("num_heads not found in attention module/config.")
        num_kv_heads = getattr(
            old_attn,
            "num_key_value_heads",
            getattr(model.config, "num_key_value_heads", max(1, kv_dim // (q_dim // num_heads))),
        )
        if kv_dim % num_kv_heads != 0:
            num_kv_heads = max(1, kv_dim // (q_dim // num_heads))

        attn_bias = old_q.bias is not None

        new_attn = KimiDeltaAttentionPhi(
            hidden_size=hidden_size,
            q_dim=q_dim,
            kv_dim=kv_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            layer_idx=i,
            conv_size=cfg.kda_conv_size,
            attn_bias=attn_bias,
            rms_eps=getattr(model.config, "rms_norm_eps", 1e-5),
            mode=cfg.kda_mode,
            use_unpad=cfg.kda_use_unpad,
            use_fused_gate=cfg.use_fused_kda_gate,
            detach_kda_core_in_training=cfg.detach_kda_core_in_training,
            expand_kv_heads_for_kda=cfg.expand_kv_heads_for_kda,
            dt_bias_init=cfg.dt_bias_init,
            a_log_min=cfg.a_log_min,
            a_log_max=cfg.a_log_max,
            gate_scale=cfg.gate_scale,
            use_qk_l2norm_in_kernel=cfg.use_qk_l2norm_in_kernel,
            use_gate_in_kernel=cfg.use_gate_in_kernel,
        ).to(device=old_q.weight.device, dtype=old_q.weight.dtype)

        with torch.no_grad():
            new_attn.q_proj.weight.copy_(old_q.weight)
            new_attn.k_proj.weight.copy_(old_k.weight)
            new_attn.v_proj.weight.copy_(old_v.weight)
            new_attn.o_proj.weight.copy_(old_o.weight)
            if attn_bias:
                if old_q.bias is not None:
                    new_attn.q_proj.bias.copy_(old_q.bias)
                if old_k.bias is not None:
                    new_attn.k_proj.bias.copy_(old_k.bias)
                if old_v.bias is not None:
                    new_attn.v_proj.bias.copy_(old_v.bias)
                if old_o.bias is not None:
                    new_attn.o_proj.bias.copy_(old_o.bias)

        layer.self_attn = new_attn
        replaced += 1

    gqa_fixed = apply_phi_attn_dim_fix(model)
    return replaced, gqa_fixed


def set_trainable_params(model: nn.Module, mode: str, train_conv_in_g_only: bool = False) -> int:
    valid_modes = {
        "g_only",
        "g_and_out",
        "full_kda",
        "gqa_lora",
        "gqa_lora_nope",
        "full_kda_gqa_lora",
        "full_kda_gqa_lora_nope",
    }
    if mode not in valid_modes:
        raise ValueError(
            "train_mode must be one of: "
            "g_only, g_and_out, full_kda, gqa_lora, gqa_lora_nope, "
            "full_kda_gqa_lora, full_kda_gqa_lora_nope"
        )

    for p in model.parameters():
        p.requires_grad = False

    trainable = 0

    def enable_module(mod: nn.Module):
        nonlocal trainable
        for p in mod.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                trainable += p.numel()

    if mode in {"gqa_lora", "gqa_lora_nope", "full_kda_gqa_lora", "full_kda_gqa_lora_nope"}:
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or isinstance(attn, KimiDeltaAttentionPhi):
                continue
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, proj_name, None)
                if isinstance(proj, LoRALinear):
                    enable_module(proj.lora_A)
                    enable_module(proj.lora_B)
        has_lora = trainable > 0
        if mode in {"gqa_lora", "gqa_lora_nope"} and not has_lora:
            raise RuntimeError("No LoRA parameters found. Did apply_gqa_lora() run?")
        if mode in {"gqa_lora", "gqa_lora_nope"}:
            return trainable

    kda_mode = mode
    if mode in {"full_kda_gqa_lora", "full_kda_gqa_lora_nope"}:
        kda_mode = "full_kda"

    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if not isinstance(attn, KimiDeltaAttentionPhi):
            continue

        # Always train KDA gates/norm parameters
        enable_module(attn.f_a_proj)
        enable_module(attn.f_b_proj)
        enable_module(attn.b_proj)
        enable_module(attn.g_a_proj)
        enable_module(attn.g_b_proj)
        enable_module(attn.o_norm)
        if not attn.A_log.requires_grad:
            attn.A_log.requires_grad = True
            trainable += attn.A_log.numel()
        if not attn.dt_bias.requires_grad:
            attn.dt_bias.requires_grad = True
            trainable += attn.dt_bias.numel()

        if kda_mode in {"g_and_out", "full_kda"}:
            enable_module(attn.o_proj)
        if (kda_mode == "g_only" and train_conv_in_g_only) or kda_mode == "g_and_out":
            # ShortConv is randomly initialized in replaced KDA blocks.
            # Keep it trainable for g_only(opt-in) and g_and_out(always-on).
            enable_module(attn.q_conv1d)
            enable_module(attn.k_conv1d)
            enable_module(attn.v_conv1d)
        if kda_mode == "full_kda":
            enable_module(attn.q_proj)
            enable_module(attn.k_proj)
            enable_module(attn.v_proj)
            enable_module(attn.q_conv1d)
            enable_module(attn.k_conv1d)
            enable_module(attn.v_conv1d)

    if trainable == 0:
        raise RuntimeError("No trainable parameters were enabled.")

    return trainable


def disable_moe_router_noise(model: nn.Module) -> int:
    touched = 0
    if hasattr(model, "config") and hasattr(model.config, "router_jitter_noise"):
        try:
            model.config.router_jitter_noise = 0.0
            touched += 1
        except Exception:
            pass

    for layer in getattr(model, "model", model).layers:
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        for obj in (moe, getattr(moe, "gate", None), getattr(moe, "router", None)):
            if obj is None:
                continue
            for name in ("router_jitter_noise", "jitter_noise", "noise_std"):
                if hasattr(obj, name):
                    try:
                        setattr(obj, name, 0.0)
                        touched += 1
                    except Exception:
                        pass
    return touched


# ---------------------------------------------------------------------------
# 4) Dataset utilities
# ---------------------------------------------------------------------------
def build_basic_math_dataset(n_rows: int, seed: int, max_abs_int: int) -> Dataset:
    rng = random.Random(seed)
    rows = {"instruction": [], "input": [], "output": []}

    for _ in range(n_rows):
        mode = rng.randrange(5)
        a = rng.randint(-max_abs_int, max_abs_int)
        b = rng.randint(-max_abs_int, max_abs_int)
        c = rng.randint(-max_abs_int, max_abs_int)

        if mode == 0:
            q = f"{a} + {b} = ?"
            ans = a + b
        elif mode == 1:
            q = f"{a} - {b} = ?"
            ans = a - b
        elif mode == 2:
            a_mul = rng.randint(-99, 99)
            b_mul = rng.randint(-99, 99)
            q = f"{a_mul} * {b_mul} = ?"
            ans = a_mul * b_mul
        elif mode == 3:
            denom = rng.randint(1, 99)
            numer = denom * rng.randint(-99, 99)
            q = f"{numer} / {denom} = ?"
            ans = numer // denom
        else:
            q = f"({a} + {b}) - {c} = ?"
            ans = (a + b) - c

        rows["instruction"].append("다음 산수 문제를 풀고 숫자만 답하세요.")
        rows["input"].append(q)
        rows["output"].append(str(ans))

    return Dataset.from_dict(rows)


def _as_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _first_non_empty(row: Dict[str, object], keys: Tuple[str, ...]) -> str:
    for key in keys:
        if key in row:
            text = _as_text(row.get(key)).strip()
            if text:
                return text
    return ""


def _extract_user_assistant_from_turns(turns) -> Tuple[str, str]:
    user_text = ""
    assistant_text = ""
    if not isinstance(turns, list):
        return user_text, assistant_text

    for item in turns:
        role = ""
        content = ""
        if isinstance(item, dict):
            role = _as_text(item.get("role", item.get("from", item.get("speaker", "")))).strip().lower()
            content = _as_text(item.get("content", item.get("value", item.get("text", "")))).strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            role = _as_text(item[0]).strip().lower()
            content = _as_text(item[1]).strip()

        if not content:
            continue
        if not user_text and role in {"user", "human", "instruction", "prompt", "question"}:
            user_text = content
            continue
        if not assistant_text and role in {"assistant", "gpt", "bot", "model", "answer", "output"}:
            assistant_text = content
            continue
        if user_text and assistant_text:
            break

    return user_text, assistant_text


def _normalize_row_to_iio(row: Dict[str, object], dataset_name: str) -> Tuple[str, str, str]:
    name = dataset_name.lower()

    if "openorca" in name:
        system_prompt = _first_non_empty(row, ("system_prompt", "system", "context"))
        question = _first_non_empty(row, ("question", "query", "instruction", "prompt"))
        answer = _first_non_empty(row, ("response", "output", "answer", "completion"))
        if system_prompt and question:
            return f"{system_prompt}\n\n{question}", "", answer
        return question, "", answer

    if "metamath" in name:
        question = _first_non_empty(row, ("query", "question", "problem", "instruction", "prompt"))
        answer = _first_non_empty(row, ("response", "output", "answer", "solution"))
        return question, "", answer

    if "magicoder" in name:
        instruction = _first_non_empty(row, ("instruction", "problem", "query", "prompt", "question"))
        in_text = _first_non_empty(row, ("input", "context"))
        output = _first_non_empty(row, ("response", "output", "answer", "solution", "completion"))
        if not instruction or not output:
            turns = row.get("conversations", row.get("messages", None))
            user_text, assistant_text = _extract_user_assistant_from_turns(turns)
            if not instruction:
                instruction = user_text
            if not output:
                output = assistant_text
        return instruction, in_text, output

    # Generic fallback
    instruction = _first_non_empty(
        row,
        ("instruction", "question", "query", "problem", "prompt", "task"),
    )
    in_text = _first_non_empty(row, ("input", "context"))
    output = _first_non_empty(
        row,
        ("output", "response", "answer", "solution", "completion"),
    )
    if not instruction or not output:
        turns = row.get("conversations", row.get("messages", None))
        user_text, assistant_text = _extract_user_assistant_from_turns(turns)
        if not instruction:
            instruction = user_text
        if not output:
            output = assistant_text
    return instruction, in_text, output


def normalize_supervised_dataset(ds: Dataset, dataset_name: str) -> Dataset:
    if set(("instruction", "input", "output")).issubset(set(ds.column_names)):
        return ds

    def _map_fn(examples):
        keys = list(examples.keys())
        if not keys:
            return {"instruction": [], "input": [], "output": []}
        n = len(examples[keys[0]])
        out_inst, out_inp, out_out = [], [], []
        for i in range(n):
            row = {k: examples[k][i] for k in keys}
            inst, inp, out = _normalize_row_to_iio(row, dataset_name)
            out_inst.append(inst)
            out_inp.append(inp)
            out_out.append(out)
        return {"instruction": out_inst, "input": out_inp, "output": out_out}

    normalized = ds.map(
        _map_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc=f"normalize:{dataset_name}",
    )
    return normalized.filter(
        lambda x: len(_as_text(x["instruction"]).strip()) > 0 and len(_as_text(x["output"]).strip()) > 0
    )


def load_local_prepared_split(
    file_path: str,
    dataset_name: str,
    cache_dir: str,
    seed: int,
    val_rows: int,
) -> Tuple[Dataset, Dataset]:
    ds = load_dataset("json", data_files=file_path, split="train", cache_dir=cache_dir)
    ds = normalize_supervised_dataset(ds, dataset_name)
    if len(ds) < 2:
        raise RuntimeError(f"Local dataset too small after normalization: {file_path} rows={len(ds)}")
    v_rows = min(val_rows, max(1, len(ds) // 100))
    v_rows = min(v_rows, len(ds) - 1)
    split = ds.train_test_split(test_size=v_rows, seed=seed)
    return split["train"], split["test"]


def load_local_prepared_train_val(
    train_file_path: str,
    val_file_path: str,
    dataset_name: str,
    cache_dir: str,
) -> Tuple[Dataset, Dataset]:
    tr = load_dataset("json", data_files=train_file_path, split="train", cache_dir=cache_dir)
    va = load_dataset("json", data_files=val_file_path, split="train", cache_dir=cache_dir)
    tr = normalize_supervised_dataset(tr, f"{dataset_name}_train")
    va = normalize_supervised_dataset(va, f"{dataset_name}_val")
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError(
            f"Local train/val dataset is empty after normalization: train={len(tr)} val={len(va)} "
            f"({train_file_path}, {val_file_path})"
        )
    return tr, va


def _bad_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = 0
    total = 0
    for ch in text:
        total += 1
        # Keep common whitespace + printable unicode text
        if ch in ("\n", "\t", "\r"):
            continue
        if ch.isprintable():
            continue
        bad += 1
    return bad / max(1, total)


def _looks_mathy(text: str) -> bool:
    if not text:
        return False
    has_digit = any(ch.isdigit() for ch in text)
    if not has_digit:
        return False
    ops = set("+-*/=%^")
    has_op = any(ch in ops for ch in text)
    math_keywords = (
        "solve",
        "equation",
        "integral",
        "derivative",
        "proof",
        "계산",
        "수학",
        "방정식",
        "미분",
        "적분",
    )
    low = text.lower()
    has_kw = any(k in low for k in math_keywords)
    return has_op or has_kw


def print_window_stats(ds, name: str, sample_n: int = 2000) -> None:
    n = len(ds)
    if n == 0:
        print(f"[stats:{name}] empty")
        return
    sample = ds.select(range(min(sample_n, n)))
    lengths = np.array([len(x) for x in sample["input_ids"]], dtype=np.float32)
    sup = np.array(
        [sum(1 for t in labels if t != -100) for labels in sample["labels"]],
        dtype=np.float32,
    )
    print(
        f"[stats:{name}] n={n} "
        f"seq(mean/p95/max)={lengths.mean():.1f}/{np.percentile(lengths,95):.1f}/{lengths.max():.0f} "
        f"sup(mean/p95/min)={sup.mean():.1f}/{np.percentile(sup,95):.1f}/{sup.min():.0f}"
    )


def take_n(ds, n: int, seed: int = 42):
    if len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def ensure_size(ds, target: int, seed: int = 42):
    if len(ds) == 0:
        raise ValueError("Dataset is empty. Cannot upsample.")
    if len(ds) >= target:
        return ds.shuffle(seed=seed).select(range(target))
    rng = np.random.default_rng(seed)
    idx = rng.integers(low=0, high=len(ds), size=target, endpoint=False).tolist()
    return ds.select(idx)


def resolve_target_windows(available_total: int, configured_target: int, min_windows: int, max_windows: int) -> int:
    if available_total <= 0:
        return 0
    if configured_target > 0:
        return min(configured_target, available_total)
    lower = min(min_windows, available_total)
    upper = min(max_windows, available_total)
    if upper < lower:
        return available_total
    return upper


def allocate_mix_quotas(
    available: Dict[str, int],
    ratios: Dict[str, float],
    target_total: int,
) -> Dict[str, int]:
    active = {k: n for k, n in available.items() if n > 0}
    if target_total <= 0 or not active:
        return {k: 0 for k in available.keys()}

    ratio_sum = sum(max(0.0, ratios.get(k, 0.0)) for k in active.keys())
    if ratio_sum <= 0:
        norm = {k: 1.0 / len(active) for k in active.keys()}
    else:
        norm = {k: max(0.0, ratios.get(k, 0.0)) / ratio_sum for k in active.keys()}

    desired = {k: target_total * norm[k] for k in active.keys()}
    quotas = {k: min(active[k], int(desired[k])) for k in active.keys()}
    assigned = sum(quotas.values())

    order = sorted(active.keys(), key=lambda k: (desired[k] - int(desired[k])), reverse=True)
    for k in order:
        if assigned >= target_total:
            break
        if quotas[k] < active[k]:
            quotas[k] += 1
            assigned += 1

    while assigned < target_total:
        candidates = [k for k in active.keys() if quotas[k] < active[k]]
        if not candidates:
            break
        candidates.sort(key=lambda k: (desired[k] - quotas[k], active[k] - quotas[k]), reverse=True)
        pick = candidates[0]
        quotas[pick] += 1
        assigned += 1

    out = {k: 0 for k in available.keys()}
    out.update(quotas)
    return out


def _get_field(examples, key: str, n: int, default: str = ""):
    return examples[key] if key in examples else [default] * n


def _base_starts(length: int, max_len: int, stride: int) -> List[int]:
    if length <= max_len:
        return [0]
    starts = list(range(0, length - max_len + 1, stride))
    tail = length - max_len
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _make_windows(
    full_ids: List[int],
    full_labels: List[int],
    max_len: int,
    stride: int,
    min_sup_tokens: int,
    max_windows: int,
) -> List[Tuple[List[int], List[int]]]:
    length = len(full_ids)
    sup_pos = [i for i, t in enumerate(full_labels) if t != -100]
    if len(sup_pos) == 0:
        return []

    ans_start, ans_end = sup_pos[0], sup_pos[-1]
    starts = set(_base_starts(length, max_len, stride))

    for a in (ans_start, ans_end, (ans_start + ans_end) // 2):
        s = max(0, min(a - (max_len // 2), max(0, length - max_len)))
        starts.add(s)

    cands = []
    for s in sorted(starts):
        e = min(s + max_len, length)
        ids = full_ids[s:e]
        labels = full_labels[s:e]
        sup = sum(1 for t in labels if t != -100)
        if sup >= min_sup_tokens and len(ids) > 1:
            cands.append((ids, labels, sup))

    if not cands:
        s = max(0, min(ans_start - (max_len // 2), max(0, length - max_len)))
        e = min(s + max_len, length)
        ids = full_ids[s:e]
        labels = full_labels[s:e]
        sup = sum(1 for t in labels if t != -100)
        if sup > 0:
            cands.append((ids, labels, sup))

    cands.sort(key=lambda x: x[2], reverse=True)
    cands = cands[:max_windows]
    return [(ids, labels) for ids, labels, _ in cands]


def build_preprocess_fn(
    tokenizer,
    max_len: int,
    stride: int,
    min_sup_tokens: int,
    max_windows: int,
    min_output_chars: int = 1,
    max_bad_char_ratio: float = 0.10,
    drop_replacement_char: bool = True,
    drop_mathy_prompt: bool = False,
):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is required.")

    has_chat_template = bool(getattr(tokenizer, "chat_template", None))

    def _normalize_ids(x):
        # BatchEncoding / dict path
        if hasattr(x, "input_ids"):
            x = x.input_ids
        elif isinstance(x, dict) and "input_ids" in x:
            x = x["input_ids"]

        # Tensor path
        if torch.is_tensor(x):
            x = x.tolist()

        # Nested list path (batch of one)
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
            x = x[0]

        if not isinstance(x, list):
            raise TypeError(f"Unsupported token id type: {type(x)}")
        return x

    def _make_prompt_ids(user_text: str) -> List[int]:
        if has_chat_template:
            try:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return _normalize_ids(ids)
            except Exception:
                pass
        # fallback template for Phi-style instruct
        enc = tokenizer(
            f"<|user|>\n{user_text}<|end|>\n<|assistant|>\n",
            add_special_tokens=False,
        )
        return _normalize_ids(enc)

    def _fn(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        n = len(examples["output"]) if "output" in examples else len(next(iter(examples.values())))
        inst = _get_field(examples, "instruction", n, "")
        inp = _get_field(examples, "input", n, "")
        out = _get_field(examples, "output", n, "")

        out_ids, out_labels, out_attn = [], [], []

        for i_text, in_text, o_text in zip(inst, inp, out):
            if o_text is None:
                continue
            o_text = str(o_text).strip()
            if len(o_text) < min_output_chars:
                continue
            if drop_replacement_char and "�" in o_text:
                continue
            if _bad_char_ratio(o_text) > max_bad_char_ratio:
                continue

            user_text = (i_text or "") + (("\n" + in_text) if in_text else "")
            if drop_mathy_prompt and _looks_mathy(user_text):
                continue
            p_ids = _make_prompt_ids(user_text)
            a_ids = _normalize_ids(tokenizer(o_text, add_special_tokens=False)) + [eos_id]

            full_ids = p_ids + a_ids
            full_labels = ([-100] * len(p_ids)) + a_ids

            windows = _make_windows(
                full_ids=full_ids,
                full_labels=full_labels,
                max_len=max_len,
                stride=stride,
                min_sup_tokens=min_sup_tokens,
                max_windows=max_windows,
            )

            for ids, labels in windows:
                out_ids.append(ids)
                out_labels.append(labels)
                out_attn.append([1] * len(ids))

        return {"input_ids": out_ids, "labels": out_labels, "attention_mask": out_attn}

    return _fn


def row_ok(x: Dict[str, List[int]]) -> bool:
    return (
        len(x["input_ids"]) == len(x["labels"]) == len(x["attention_mask"])
        and len(x["input_ids"]) > 1
        and any(t != -100 for t in x["labels"])
    )


def causal_lm_collator(features, tokenizer):
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad)
        attention_mask.append(f["attention_mask"] + [0] * pad)
        labels.append(f["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _hf_repo_kwargs(cfg: TrainConfig) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "trust_remote_code": True,
        "cache_dir": cfg.cache_dir,
    }
    rev = (cfg.model_revision or "").strip()
    if rev:
        kwargs["revision"] = rev
    return kwargs


def _fit_matrix_to_shape(src: torch.Tensor, target_shape: torch.Size) -> Optional[torch.Tensor]:
    if src.ndim != 2:
        return None
    if tuple(src.shape) == tuple(target_shape):
        return src
    src_t = src.transpose(0, 1).contiguous()
    if tuple(src_t.shape) == tuple(target_shape):
        return src_t
    return None


def _iter_checkpoint_state_items(local_repo_dir: str):
    from safetensors.torch import load_file as safe_load_file

    st_index = os.path.join(local_repo_dir, "model.safetensors.index.json")
    if os.path.exists(st_index):
        with open(st_index, "r", encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for shard in shards:
            shard_path = os.path.join(local_repo_dir, shard)
            for k, v in safe_load_file(shard_path, device="cpu").items():
                yield k, v
        return

    st_single = os.path.join(local_repo_dir, "model.safetensors")
    if os.path.exists(st_single):
        for k, v in safe_load_file(st_single, device="cpu").items():
            yield k, v
        return

    raise RuntimeError(
        f"No safetensors checkpoint found under: {local_repo_dir}. "
        "Expected model.safetensors or model.safetensors.index.json"
    )


def _load_checkpoint_moe_keysafe(model: nn.Module, cfg: TrainConfig) -> None:
    from huggingface_hub import snapshot_download

    target_sd = model.state_dict()
    mapped_sd: Dict[str, torch.Tensor] = {}
    unmapped_moe_keys: List[str] = []

    re_router = re.compile(r"^(.*)\.mlp\.router\.weight$")
    re_down = re.compile(r"^(.*)\.mlp\.experts\.down_proj(?:\.weight)?$")
    re_gateup = re.compile(r"^(.*)\.mlp\.experts\.gate_up_proj(?:\.weight)?$")

    local_override = (getattr(cfg, "init_from_local_dir", "") or "").strip()
    if local_override:
        local_repo_dir = local_override
        print(f"[load] using local init dir: {local_repo_dir}")
    else:
        repo_kwargs = _hf_repo_kwargs(cfg)
        repo_kwargs.pop("trust_remote_code", None)
        local_repo_dir = snapshot_download(
            repo_id=cfg.model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
            **repo_kwargs,
        )

    for key, tensor in _iter_checkpoint_state_items(local_repo_dir):
        if key in target_sd:
            mapped_sd[key] = tensor.to(dtype=target_sd[key].dtype)
            continue

        m = re_router.match(key)
        if m:
            dst = f"{m.group(1)}.block_sparse_moe.gate.weight"
            if dst in target_sd:
                fit = _fit_matrix_to_shape(tensor, target_sd[dst].shape)
                if fit is not None:
                    mapped_sd[dst] = fit.to(dtype=target_sd[dst].dtype)
                    continue
            unmapped_moe_keys.append(key)
            continue

        m = re_down.match(key)
        if m and tensor.ndim == 3:
            base = m.group(1)
            ok = True
            for idx in range(tensor.shape[0]):
                dst = f"{base}.block_sparse_moe.experts.{idx}.w2.weight"
                if dst not in target_sd:
                    ok = False
                    break
                fit = _fit_matrix_to_shape(tensor[idx], target_sd[dst].shape)
                if fit is None:
                    ok = False
                    break
                mapped_sd[dst] = fit.to(dtype=target_sd[dst].dtype)
            if ok:
                continue
            unmapped_moe_keys.append(key)
            continue

        m = re_gateup.match(key)
        if m and tensor.ndim == 3:
            base = m.group(1)
            ok = True
            for idx in range(tensor.shape[0]):
                dst_w1 = f"{base}.block_sparse_moe.experts.{idx}.w1.weight"
                dst_w3 = f"{base}.block_sparse_moe.experts.{idx}.w3.weight"
                if dst_w1 not in target_sd or dst_w3 not in target_sd:
                    ok = False
                    break

                src_e = tensor[idx]
                fit_pair = None
                for split_dim in (0, 1):
                    if src_e.shape[split_dim] % 2 != 0:
                        continue
                    part_a, part_b = torch.chunk(src_e, 2, dim=split_dim)
                    for first, second in ((part_a, part_b), (part_b, part_a)):
                        fit_w1 = _fit_matrix_to_shape(first, target_sd[dst_w1].shape)
                        fit_w3 = _fit_matrix_to_shape(second, target_sd[dst_w3].shape)
                        if fit_w1 is not None and fit_w3 is not None:
                            fit_pair = (fit_w1, fit_w3)
                            break
                    if fit_pair is not None:
                        break

                if fit_pair is None:
                    ok = False
                    break

                mapped_sd[dst_w1] = fit_pair[0].to(dtype=target_sd[dst_w1].dtype)
                mapped_sd[dst_w3] = fit_pair[1].to(dtype=target_sd[dst_w3].dtype)

            if ok:
                continue
            unmapped_moe_keys.append(key)
            continue

        if ".mlp." in key or "block_sparse_moe" in key:
            unmapped_moe_keys.append(key)

    print(f"[load] remap tensors={len(mapped_sd):,} | unmapped_moe_candidates={len(unmapped_moe_keys)}")
    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    critical_missing = [k for k in missing if ("block_sparse_moe" in k or ".mlp." in k)]
    critical_unexpected = [k for k in unexpected if ("block_sparse_moe" in k or ".mlp." in k)]

    if unmapped_moe_keys or critical_missing or critical_unexpected:
        raise RuntimeError(
            "Checkpoint/model key mismatch (MoE/MLP) after remap.\n"
            f"unmapped_moe_keys={unmapped_moe_keys[:30]}\n"
            f"missing={critical_missing[:30]}\n"
            f"unexpected={critical_unexpected[:30]}"
        )


# ---------------------------------------------------------------------------
# 5) Build model
# ---------------------------------------------------------------------------
print("=" * 88)
print("Kimi-Phi KDA v10.32")
print(f"TrainMode={CFG.train_mode} | Seq={CFG.max_seq_length} | Batch={CFG.batch_size} | Accum={CFG.grad_accum}")
print(f"Epochs={CFG.num_epochs} | MaxSteps={CFG.max_steps}")
if CFG.max_steps and CFG.max_steps > 0:
    print("Note: max_steps is active and will override full-epoch completion.")
if CFG.target_train_windows > 0 or CFG.target_val_windows > 0:
    print(f"Target windows train/val = {CFG.target_train_windows}/{CFG.target_val_windows}")
else:
    print(
        f"Auto windows train[min,max]={CFG.auto_train_windows_min},{CFG.auto_train_windows_max} | "
        f"val[min,max]={CFG.auto_val_windows_min},{CFG.auto_val_windows_max}"
    )
print(
    f"General={CFG.general_dataset_id} | Logic={list(CFG.logic_dataset_ids)} | "
    f"SaveEvery={CFG.save_steps} | KeepCkpt={CFG.save_total_limit}"
)
if CFG.enable_pin_checkpoints and len(CFG.pin_checkpoint_steps) > 0:
    print(f"Pinned checkpoints: {sorted(set(int(x) for x in CFG.pin_checkpoint_steps if int(x) > 0))}")
print(f"group_by_length={CFG.group_by_length}")
print(
    f"force_python_kda_core={CFG.force_python_kda_core} | "
    f"use_native_short_conv={CFG.use_native_short_conv} | "
    f"use_native_rmsnorm_gated={CFG.use_native_rmsnorm_gated}"
)
print(
    f"use_qk_l2norm_in_kernel={CFG.use_qk_l2norm_in_kernel} | "
    f"use_gate_in_kernel={CFG.use_gate_in_kernel} | "
    f"use_fused_kda_gate={CFG.use_fused_kda_gate}"
)
print(
    f"A_log_init=[{CFG.a_log_min}, {CFG.a_log_max}] | "
    f"dt_bias_init={CFG.dt_bias_init} | gate_scale={CFG.gate_scale}"
)
if CFG.train_mode in {"gqa_lora", "gqa_lora_nope"}:
    print(
        f"GQA LoRA r/alpha/dropout = "
        f"{CFG.gqa_lora_rank}/{CFG.gqa_lora_alpha}/{CFG.gqa_lora_dropout}"
    )
if CFG.train_mode in {"gqa_lora", "gqa_lora_nope", "full_kda_gqa_lora", "full_kda_gqa_lora_nope"}:
    print(f"GQA LoRA include v_proj={CFG.gqa_lora_include_v_proj}")
print("=" * 88)

print("\nLoading base model...")
repo_kwargs = _hf_repo_kwargs(CFG)
config = AutoConfig.from_pretrained(CFG.model_id, **repo_kwargs)
config = patch_rope_scaling(config)
config.sliding_window = None
config.use_cache = False

model_from_config_kwargs = {"trust_remote_code": True}
try:
    student_model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation="eager",
        **model_from_config_kwargs,
    )
except TypeError:
    # Some stacks/remote-code configs do not accept attn_implementation here.
    student_model = AutoModelForCausalLM.from_config(config, **model_from_config_kwargs)
_load_checkpoint_moe_keysafe(student_model, CFG)
student_model = student_model.to("cuda", dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(CFG.model_id, **repo_kwargs)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 10**9
student_model.config.pad_token_id = tokenizer.pad_token_id

# Do not clear private hook maps globally; it can break remote-code internals.

replaced, gqa_fixed = replace_attention_3to1(student_model, CFG)
tied_fixed_before = normalize_tied_weight_keys(student_model)
print(f"GKA/KDA replaced={replaced}, GQA dim-fix={gqa_fixed}, tied-key-fixed={tied_fixed_before}")

gqa_lora_wrapped = 0
nope_layers_patched = 0
if CFG.train_mode in {"gqa_lora", "gqa_lora_nope", "full_kda_gqa_lora", "full_kda_gqa_lora_nope"}:
    gqa_lora_wrapped = apply_gqa_lora(
        student_model,
        rank=CFG.gqa_lora_rank,
        alpha=CFG.gqa_lora_alpha,
        dropout=CFG.gqa_lora_dropout,
        include_v_proj=CFG.gqa_lora_include_v_proj,
    )
    if CFG.train_mode in {"gqa_lora_nope", "full_kda_gqa_lora_nope"}:
        nope_layers_patched = apply_nope_to_original_gqa(student_model)
    print(f"GQA LoRA wrapped projections: {gqa_lora_wrapped}")
    if CFG.train_mode in {"gqa_lora_nope", "full_kda_gqa_lora_nope"}:
        print(f"NoPE patched original GQA layers: {nope_layers_patched}")

if CFG.use_checkpointing:
    student_model.config.use_cache = False
    if hasattr(student_model, "gradient_checkpointing_enable"):
        try:
            student_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True}
            )
        except Exception:
            student_model.gradient_checkpointing_enable()
    # Critical when almost everything is frozen + checkpointing
    if hasattr(student_model, "enable_input_require_grads"):
        student_model.enable_input_require_grads()
    else:
        student_model.get_input_embeddings().register_forward_hook(lambda _m, _i, o: o.requires_grad_(True))

trainable_count = set_trainable_params(
    student_model,
    CFG.train_mode,
    train_conv_in_g_only=CFG.enable_conv_in_g_only,
)
bad_trainables = [n for n, p in student_model.named_parameters() if p.requires_grad and ".self_attn." not in n]
noise_touched = disable_moe_router_noise(student_model)
print(f"Trainable params: {trainable_count:,}")
print(f"Non-attn trainables: {len(bad_trainables)}")
print(f"MoE router-noise attrs set to 0: {noise_touched}")
if CFG.train_mode == "g_only":
    print(f"g_only conv trainable: {CFG.enable_conv_in_g_only}")
if CFG.train_mode in {"gqa_lora", "gqa_lora_nope", "full_kda_gqa_lora", "full_kda_gqa_lora_nope"}:
    print(
        "RoPE on original GQA:",
        "disabled (NoPE)" if CFG.train_mode in {"gqa_lora_nope", "full_kda_gqa_lora_nope"} else "enabled",
    )


# ---------------------------------------------------------------------------
# 6) Build dataset (Long + General + Logic mix, robust sizing)
# ---------------------------------------------------------------------------
print("\nLoading datasets...")
prepared_dir = CFG.prepared_data_dir
local_long_path = os.path.join(prepared_dir, CFG.prepared_long_file)
local_long_train_path = os.path.join(prepared_dir, CFG.prepared_long_train_file)
local_long_val_path = os.path.join(prepared_dir, CFG.prepared_long_val_file)
local_general_path = os.path.join(prepared_dir, CFG.prepared_general_file)
local_general_train_path = os.path.join(prepared_dir, CFG.prepared_general_train_file)
local_general_val_path = os.path.join(prepared_dir, CFG.prepared_general_val_file)
local_logic_path = os.path.join(prepared_dir, CFG.prepared_logic_file)
local_logic_train_path = os.path.join(prepared_dir, CFG.prepared_logic_train_file)
local_logic_val_path = os.path.join(prepared_dir, CFG.prepared_logic_val_file)


def _has_pair(train_path: str, val_path: str) -> bool:
    return os.path.exists(train_path) and os.path.exists(val_path)

if CFG.require_local_prepared_data:
    missing = []
    if not (_has_pair(local_long_train_path, local_long_val_path) or os.path.exists(local_long_path)):
        missing.append(f"long pair/single missing under {prepared_dir}")
    if not (_has_pair(local_general_train_path, local_general_val_path) or os.path.exists(local_general_path)):
        missing.append(f"general pair/single missing under {prepared_dir}")
    if not (_has_pair(local_logic_train_path, local_logic_val_path) or os.path.exists(local_logic_path)):
        missing.append(f"logic pair/single missing under {prepared_dir}")
    if missing:
        raise RuntimeError(
            "require_local_prepared_data=True but prepared files are missing: "
            + ", ".join(missing)
        )

long_split = None
if CFG.prefer_local_prepared_data and _has_pair(local_long_train_path, local_long_val_path):
    tr, va = load_local_prepared_train_val(
        train_file_path=local_long_train_path,
        val_file_path=local_long_val_path,
        dataset_name="long_local",
        cache_dir=CFG.cache_dir,
    )
    long_split = {"train": tr, "test": va}
    print(f"Long dataset loaded from local train/val files: {local_long_train_path}, {local_long_val_path}")
elif CFG.prefer_local_prepared_data and os.path.exists(local_long_path):
    tr, va = load_local_prepared_split(
        file_path=local_long_path,
        dataset_name="long_local",
        cache_dir=CFG.cache_dir,
        seed=CFG.seed,
        val_rows=300,
    )
    long_split = {"train": tr, "test": va}
    print(f"Long dataset loaded from local prepared file: {local_long_path} rows={len(tr) + len(va)}")
else:
    if CFG.prefer_local_prepared_data:
        print(f"Local long file(s) not found, fallback to hub under: {prepared_dir}")
    long_url = "https://huggingface.co/datasets/Yukang/LongAlpaca-12k/resolve/main/LongAlpaca-12k.json"
    raw_long = load_dataset("json", data_files=long_url, split="train", cache_dir=CFG.cache_dir)
    long_split = raw_long.train_test_split(test_size=0.01, seed=CFG.seed)

use_general = False
if CFG.prefer_local_prepared_data and _has_pair(local_general_train_path, local_general_val_path):
    try:
        tr, va = load_local_prepared_train_val(
            train_file_path=local_general_train_path,
            val_file_path=local_general_val_path,
            dataset_name="general_local",
            cache_dir=CFG.cache_dir,
        )
        general_split = {"train": tr, "test": va}
        use_general = True
        print(
            f"General dataset loaded from local train/val files: "
            f"{local_general_train_path}, {local_general_val_path}"
        )
    except Exception as e:
        print(f"Local general train/val load failed, fallback to hub: {e}")
elif CFG.prefer_local_prepared_data and os.path.exists(local_general_path):
    try:
        tr, va = load_local_prepared_split(
            file_path=local_general_path,
            dataset_name="general_local",
            cache_dir=CFG.cache_dir,
            seed=CFG.seed,
            val_rows=CFG.general_val_rows,
        )
        general_split = {"train": tr, "test": va}
        use_general = True
        print(f"General dataset loaded from local prepared file: {local_general_path} rows={len(tr) + len(va)}")
    except Exception as e:
        print(f"Local general dataset load failed, fallback to hub: {e}")

if not use_general:
    try:
        raw_general = load_dataset(CFG.general_dataset_id, split="train", cache_dir=CFG.cache_dir).shuffle(seed=CFG.seed)
        raw_general = raw_general.select(range(min(CFG.general_num_raw, len(raw_general))))
        raw_general = normalize_supervised_dataset(raw_general, CFG.general_dataset_id)
        if len(raw_general) < 2:
            raise RuntimeError(f"Too few rows after normalization: {len(raw_general)}")
        general_val_rows = min(CFG.general_val_rows, max(1, len(raw_general) // 100))
        general_val_rows = min(general_val_rows, len(raw_general) - 1)
        general_split = raw_general.train_test_split(test_size=general_val_rows, seed=CFG.seed)
        use_general = True
        print(f"General dataset loaded: {CFG.general_dataset_id} rows={len(raw_general)}")
    except Exception as e:
        print(f"General dataset load failed, fallback to long-only: {e}")

use_logic = False
logic_is_synth = False

if CFG.prefer_local_prepared_data and _has_pair(local_logic_train_path, local_logic_val_path):
    try:
        tr, va = load_local_prepared_train_val(
            train_file_path=local_logic_train_path,
            val_file_path=local_logic_val_path,
            dataset_name="logic_local",
            cache_dir=CFG.cache_dir,
        )
        logic_split = {"train": tr, "test": va}
        use_logic = True
        print(
            f"Logic dataset loaded from local train/val files: "
            f"{local_logic_train_path}, {local_logic_val_path}"
        )
    except Exception as e:
        print(f"Local logic train/val load failed, fallback to hub: {e}")
elif CFG.prefer_local_prepared_data and os.path.exists(local_logic_path):
    try:
        tr, va = load_local_prepared_split(
            file_path=local_logic_path,
            dataset_name="logic_local",
            cache_dir=CFG.cache_dir,
            seed=CFG.seed,
            val_rows=CFG.logic_val_rows,
        )
        logic_split = {"train": tr, "test": va}
        use_logic = True
        print(f"Logic dataset loaded from local prepared file: {local_logic_path} rows={len(tr) + len(va)}")
    except Exception as e:
        print(f"Local logic dataset load failed, fallback to hub: {e}")

if not use_logic:
    logic_parts = []
    for ds_id in CFG.logic_dataset_ids:
        try:
            raw_logic = load_dataset(ds_id, split="train", cache_dir=CFG.cache_dir).shuffle(seed=CFG.seed)
            raw_logic = raw_logic.select(range(min(CFG.logic_num_raw_per_source, len(raw_logic))))
            raw_logic = normalize_supervised_dataset(raw_logic, ds_id)
            if len(raw_logic) > 0:
                logic_parts.append(raw_logic)
                print(f"Logic source loaded: {ds_id} rows={len(raw_logic)}")
        except Exception as e:
            print(f"Logic source skipped ({ds_id}): {e}")

    if len(logic_parts) > 0:
        raw_logic_all = logic_parts[0] if len(logic_parts) == 1 else concatenate_datasets(logic_parts).shuffle(seed=CFG.seed)
        if len(raw_logic_all) < 2:
            raise RuntimeError("Combined logic dataset is too small after normalization.")
        logic_val_rows = min(CFG.logic_val_rows, max(1, len(raw_logic_all) // 100))
        logic_val_rows = min(logic_val_rows, len(raw_logic_all) - 1)
        logic_split = raw_logic_all.train_test_split(test_size=logic_val_rows, seed=CFG.seed)
        use_logic = True

if not use_logic and CFG.logic_fallback_to_synth_math and CFG.use_math_data:
    print("Logic datasets unavailable. Falling back to synthetic math dataset...")
    raw_math = build_basic_math_dataset(
        n_rows=CFG.math_num_train_raw + CFG.math_num_val_raw,
        seed=CFG.seed,
        max_abs_int=CFG.math_max_abs_int,
    )
    logic_split = raw_math.train_test_split(test_size=CFG.math_num_val_raw, seed=CFG.seed)
    use_logic = True
    logic_is_synth = True

print("Preprocessing long...")
train_long = long_split["train"].map(
    build_preprocess_fn(
        tokenizer=tokenizer,
        max_len=CFG.max_seq_length,
        stride=CFG.long_stride_train,
        min_sup_tokens=CFG.min_sup_tokens_train,
        max_windows=CFG.max_windows_per_long_train,
        min_output_chars=CFG.min_output_chars,
        max_bad_char_ratio=CFG.max_bad_char_ratio,
        drop_replacement_char=CFG.drop_replacement_char,
    ),
    batched=True,
    remove_columns=long_split["train"].column_names,
).filter(row_ok)

val_long = long_split["test"].map(
    build_preprocess_fn(
        tokenizer=tokenizer,
        max_len=CFG.max_seq_length,
        stride=CFG.long_stride_val,
        min_sup_tokens=CFG.min_sup_tokens_val,
        max_windows=CFG.max_windows_per_long_val,
        min_output_chars=CFG.min_output_chars,
        max_bad_char_ratio=CFG.max_bad_char_ratio,
        drop_replacement_char=CFG.drop_replacement_char,
    ),
    batched=True,
    remove_columns=long_split["test"].column_names,
).filter(row_ok)
print_window_stats(train_long, "long_train_raw_windows")
print_window_stats(val_long, "long_val_raw_windows")

train_sources = {"long": train_long}
val_sources = {"long": val_long}
mix_ratios = {
    "long": CFG.mix_ratio_long,
    "short": CFG.mix_ratio_short,
    "math": CFG.mix_ratio_math,
}

if use_general:
    print("Preprocessing general (OpenOrca)...")
    train_general = general_split["train"].map(
        build_preprocess_fn(
            tokenizer=tokenizer,
            max_len=CFG.max_seq_length,
            stride=CFG.max_seq_length,
            min_sup_tokens=8,
            max_windows=1,
            min_output_chars=CFG.min_output_chars,
            max_bad_char_ratio=CFG.max_bad_char_ratio,
            drop_replacement_char=CFG.drop_replacement_char,
            drop_mathy_prompt=CFG.filter_mathy_in_short,
        ),
        batched=True,
        remove_columns=general_split["train"].column_names,
    ).filter(row_ok)

    val_general = general_split["test"].map(
        build_preprocess_fn(
            tokenizer=tokenizer,
            max_len=CFG.max_seq_length,
            stride=CFG.max_seq_length,
            min_sup_tokens=8,
            max_windows=1,
            min_output_chars=CFG.min_output_chars,
            max_bad_char_ratio=CFG.max_bad_char_ratio,
            drop_replacement_char=CFG.drop_replacement_char,
            drop_mathy_prompt=CFG.filter_mathy_in_short,
        ),
        batched=True,
        remove_columns=general_split["test"].column_names,
    ).filter(row_ok)
    print_window_stats(train_general, "general_train_raw_windows")
    print_window_stats(val_general, "general_val_raw_windows")
    train_sources["short"] = train_general
    val_sources["short"] = val_general
else:
    print("General dataset unavailable. Auto-mix will renormalize without general.")

if use_logic:
    print("Preprocessing logic...")
    logic_max_len = min(2048, CFG.max_seq_length)
    train_logic = logic_split["train"].map(
        build_preprocess_fn(
            tokenizer=tokenizer,
            max_len=logic_max_len,
            stride=logic_max_len,
            min_sup_tokens=1,
            max_windows=1,
            min_output_chars=1,
            max_bad_char_ratio=0.0,
            drop_replacement_char=True,
        ),
        batched=True,
        remove_columns=logic_split["train"].column_names,
    ).filter(row_ok)

    val_logic = logic_split["test"].map(
        build_preprocess_fn(
            tokenizer=tokenizer,
            max_len=logic_max_len,
            stride=logic_max_len,
            min_sup_tokens=1,
            max_windows=1,
            min_output_chars=1,
            max_bad_char_ratio=0.0,
            drop_replacement_char=True,
        ),
        batched=True,
        remove_columns=logic_split["test"].column_names,
    ).filter(row_ok)
    if logic_is_synth:
        print_window_stats(train_logic, "synth_logic_train_raw_windows")
        print_window_stats(val_logic, "synth_logic_val_raw_windows")
    else:
        print_window_stats(train_logic, "logic_train_raw_windows")
        print_window_stats(val_logic, "logic_val_raw_windows")
    train_sources["math"] = train_logic
    val_sources["math"] = val_logic

avail_train = {k: len(v) for k, v in train_sources.items()}
avail_val = {k: len(v) for k, v in val_sources.items()}

target_train = resolve_target_windows(
    available_total=sum(avail_train.values()),
    configured_target=CFG.target_train_windows,
    min_windows=CFG.auto_train_windows_min,
    max_windows=CFG.auto_train_windows_max,
)
target_val = resolve_target_windows(
    available_total=sum(avail_val.values()),
    configured_target=CFG.target_val_windows,
    min_windows=CFG.auto_val_windows_min,
    max_windows=CFG.auto_val_windows_max,
)

train_quota = allocate_mix_quotas(avail_train, mix_ratios, target_train)
val_quota = allocate_mix_quotas(avail_val, mix_ratios, target_val)

print(f"Available train windows by source: {avail_train}")
print(f"Available val windows by source:   {avail_val}")
print(f"Target train/val windows: {target_train}/{target_val}")
print(f"Allocated train quotas: {train_quota}")
print(f"Allocated val quotas:   {val_quota}")

train_parts = []
for i, key in enumerate(sorted(train_sources.keys())):
    q = train_quota.get(key, 0)
    if q > 0:
        train_parts.append(take_n(train_sources[key], q, seed=CFG.seed + i))

val_parts = []
for i, key in enumerate(sorted(val_sources.keys())):
    q = val_quota.get(key, 0)
    if q > 0:
        val_parts.append(take_n(val_sources[key], q, seed=CFG.seed + 100 + i))

if len(train_parts) == 0 or len(val_parts) == 0:
    raise RuntimeError("No dataset parts selected after allocation. Check preprocessing filters/ratios.")

train_dataset = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts).shuffle(seed=CFG.seed)
val_dataset = val_parts[0] if len(val_parts) == 1 else concatenate_datasets(val_parts).shuffle(seed=CFG.seed)

if CFG.force_exact_target_windows:
    train_dataset = ensure_size(train_dataset, target_train, seed=CFG.seed)
    val_dataset = ensure_size(val_dataset, target_val, seed=CFG.seed + 1)
    print("Force-exact target windows enabled: sampling with replacement may occur.")

print(f"Final train windows: {len(train_dataset)}")
print(f"Final val windows:   {len(val_dataset)}")
print(f"Mix ratios (long/short/math): {CFG.mix_ratio_long:.2f}/{CFG.mix_ratio_short:.2f}/{CFG.mix_ratio_math:.2f}")
print_window_stats(train_dataset, "train_final_mix")
print_window_stats(val_dataset, "val_final_mix")


# ---------------------------------------------------------------------------
# 7) Trainer
# ---------------------------------------------------------------------------
class ConsoleLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)
        msg = [f"[step {step}]"]
        if "loss" in logs:
            msg.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            msg.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            msg.append(f"lr={logs['learning_rate']:.2e}")
        print(" | ".join(msg))


class DeltaWatchCallback(TrainerCallback):
    def __init__(self, model: nn.Module):
        self.watch_name = None
        self.watch_init = None
        for n, p in model.named_parameters():
            if p.requires_grad and "f_b_proj.weight" in n:
                self.watch_name = n
                self.watch_init = p.detach().float().clone()
                break

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.watch_name is None or self.watch_init is None:
            return
        if state.global_step > 0 and state.global_step % 50 == 0:
            model = kwargs.get("model", None)
            if model is None:
                return
            cur = dict(model.named_parameters())[self.watch_name].detach().float()
            delta = (cur - self.watch_init).abs().mean().item()
            print(f"[delta] step={state.global_step} mean_abs={delta:.3e}")


class PinCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str, pin_steps: Tuple[int, ...]):
        self.output_dir = output_dir
        self.pin_steps = {int(s) for s in pin_steps if int(s) > 0}
        self.done = set()
        self.pin_root = os.path.join(output_dir, "pinned_checkpoints")

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step not in self.pin_steps or step in self.done:
            return

        src = os.path.join(args.output_dir, f"checkpoint-{step}")
        dst = os.path.join(self.pin_root, f"checkpoint-{step}")
        if not os.path.isdir(src):
            return

        os.makedirs(self.pin_root, exist_ok=True)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[pin] checkpoint-{step} -> {dst}")
        self.done.add(step)


ta_params = inspect.signature(TrainingArguments.__init__).parameters
eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

args_dict = dict(
    output_dir=CFG.output_dir,
    bf16=True,
    fp16=False,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=max(1, CFG.batch_size // 2),
    gradient_accumulation_steps=CFG.grad_accum,
    gradient_checkpointing=CFG.use_checkpointing,
    num_train_epochs=CFG.num_epochs,
    learning_rate=CFG.learning_rate,
    warmup_steps=CFG.warmup_steps,
    lr_scheduler_type=CFG.lr_scheduler_type,
    weight_decay=CFG.weight_decay,
    max_grad_norm=CFG.max_grad_norm,
    logging_strategy="steps",
    logging_steps=CFG.logging_steps,
    logging_first_step=False,
    save_strategy="steps",
    save_steps=CFG.save_steps,
    eval_steps=CFG.eval_steps,
    save_total_limit=CFG.save_total_limit,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=CFG.dataloader_num_workers,
    dataloader_pin_memory=True,
    group_by_length=CFG.group_by_length,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
if CFG.max_steps and CFG.max_steps > 0:
    args_dict["max_steps"] = CFG.max_steps
args_dict[eval_key] = "steps"

if "gradient_checkpointing_kwargs" in ta_params:
    args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
if "dataloader_persistent_workers" in ta_params and CFG.dataloader_num_workers > 0:
    args_dict["dataloader_persistent_workers"] = True
if "save_safetensors" in ta_params:
    args_dict["save_safetensors"] = True
if "optim" in ta_params:
    # use plain adamw_torch for broad version stability
    args_dict["optim"] = "adamw_torch"
if "logging_nan_inf_filter" in ta_params:
    args_dict["logging_nan_inf_filter"] = False

try:
    training_args = TrainingArguments(**args_dict)
except TypeError as e:
    if "group_by_length" in str(e) and "group_by_length" in args_dict:
        print("[compat] removing unsupported arg: group_by_length")
        args_dict.pop("group_by_length", None)
        training_args = TrainingArguments(**args_dict)
    else:
        raise

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda feats: causal_lm_collator(feats, tokenizer),
)
trainer.add_callback(ConsoleLogCallback())
trainer.add_callback(DeltaWatchCallback(student_model))
if CFG.enable_pin_checkpoints and len(CFG.pin_checkpoint_steps) > 0:
    trainer.add_callback(PinCheckpointCallback(CFG.output_dir, CFG.pin_checkpoint_steps))


# ---------------------------------------------------------------------------
# 8) Train / Save
# ---------------------------------------------------------------------------
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

last_ckpt = get_last_checkpoint(CFG.output_dir) if CFG.resume_from_last else None
launch_msg = f"resume {last_ckpt}" if last_ckpt else "fresh start"
print(f"\nLaunch: {launch_msg}")
try:
    trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)
except Exception as e:
    msg = f"{type(e).__name__}: {e}"
    if CFG.use_checkpointing and ("CheckpointError" in msg or "Recomputed values" in msg):
        print("CheckpointError detected. Retrying with gradient checkpointing disabled...")
        if hasattr(student_model, "gradient_checkpointing_disable"):
            try:
                student_model.gradient_checkpointing_disable()
            except Exception:
                pass
        student_model.config.use_cache = False

        args_no_ckpt = dict(args_dict)
        args_no_ckpt["gradient_checkpointing"] = False
        args_no_ckpt.pop("gradient_checkpointing_kwargs", None)
        training_args_no_ckpt = TrainingArguments(**args_no_ckpt)

        trainer = Trainer(
            model=student_model,
            args=training_args_no_ckpt,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=lambda feats: causal_lm_collator(feats, tokenizer),
        )
        trainer.add_callback(ConsoleLogCallback())
        trainer.add_callback(DeltaWatchCallback(student_model))
        if CFG.enable_pin_checkpoints and len(CFG.pin_checkpoint_steps) > 0:
            trainer.add_callback(PinCheckpointCallback(CFG.output_dir, CFG.pin_checkpoint_steps))
        trainer.train(resume_from_checkpoint=None)
    else:
        raise

final_dir = os.path.join(CFG.output_dir, CFG.final_dir_name)
if os.path.isdir(final_dir):
    shutil.rmtree(final_dir)
os.makedirs(final_dir, exist_ok=True)

# Normalize tied keys once again before save
tied_fixed_after = normalize_tied_weight_keys(student_model)
print(f"Tied-key normalized before save: {tied_fixed_after}")

student_model.save_pretrained(final_dir, safe_serialization=True, max_shard_size="50GB")
tokenizer.save_pretrained(final_dir)
manifest_path = write_kda_reload_manifest(CFG, final_dir)

if not has_complete_saved_model(final_dir):
    raise RuntimeError(f"Final save incomplete or corrupted: {final_dir}")

weight_files = sorted(glob.glob(os.path.join(final_dir, "model*.safetensors")))
if not weight_files:
    weight_files = sorted(glob.glob(os.path.join(final_dir, "pytorch_model*.bin")))

print(f"Saved to: {final_dir}")
print(f"Reload manifest: {manifest_path}")
print("Weight files:", [os.path.basename(x) for x in weight_files])
if len(weight_files) == 0:
    raise RuntimeError("No weight files found after save.")

if torch.cuda.is_available():
    print("peak_alloc_gb   =", round(torch.cuda.max_memory_allocated() / 1e9, 2))
    print("peak_reserved_gb=", round(torch.cuda.max_memory_reserved() / 1e9, 2))
