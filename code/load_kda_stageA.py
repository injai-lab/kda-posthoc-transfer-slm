#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option-A KDA loader:
1) Load base Phi model from manifest model_id/revision.
2) Apply replace_attention_3to1 (runtime architecture swap).
3) Load trained checkpoint weights from final_dir.
"""

import argparse
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

try:
    from fla.modules import ShortConvolution as FLA_ShortConvolution
except Exception:
    FLA_ShortConvolution = None

try:
    from fla.modules import FusedRMSNormGated as FLA_FusedRMSNormGated
except Exception:
    FLA_FusedRMSNormGated = None


@dataclass
class KDAOpts:
    kda_conv_size: int = 4
    kda_mode: str = "chunk"
    kda_use_unpad: bool = False
    use_native_short_conv: bool = True
    use_native_rmsnorm_gated: bool = True
    force_python_kda_core: bool = True
    use_fused_kda_gate: bool = False
    detach_kda_core_in_training: bool = False
    expand_kv_heads_for_kda: bool = True
    dt_bias_init: float = 0.2
    a_log_min: float = 1.0
    a_log_max: float = 16.0
    gate_scale: float = 1.0


def patch_rope_scaling(cfg_obj: AutoConfig) -> AutoConfig:
    rs = getattr(cfg_obj, "rope_scaling", None)
    if isinstance(rs, dict):
        if "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]
        if rs.get("type") == "default":
            cfg_obj.rope_scaling = None
    return cfg_obj


class NativeShortConvolution(nn.Module):
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
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        del cache, cu_seqlens
        xt = x.transpose(1, 2)
        if self.kernel_size > 1:
            xt = torch.nn.functional.pad(xt, (self.kernel_size - 1, 0))
        y = self.conv(xt).transpose(1, 2)
        y = self.act(y)
        final_state = None
        if output_final_state:
            keep = max(0, self.kernel_size - 1)
            final_state = x[:, -keep:, :] if keep > 0 else x[:, :0, :]
        return y, final_state


class NativeFusedRMSNormGated(nn.Module):
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


def _safe_python_kda_gate(g, a_log, head_dim, dt_bias, decay_scale: float = 1.0):
    if g.dim() == 3:
        if g.size(-1) % head_dim != 0:
            raise ValueError(f"Cannot reshape g dim {g.size(-1)} with head_dim={head_dim}")
        g = rearrange(g, "... (h d) -> ... h d", h=g.size(-1) // head_dim)
    db = dt_bias.view(1, 1, -1, head_dim).to(g.dtype)
    dt = torch.nn.functional.softplus(g + db)
    return (-torch.exp(a_log.float())).to(dt.dtype) * dt * float(decay_scale)


def normalize_kda_gate_shape(g: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if g.dim() == 3:
        return rearrange(g, "b t (h d) -> b t h d", h=num_heads)
    return g


def python_delta_rule_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    bsz, seq_len, num_heads, head_dim = q.shape
    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    if initial_state is None:
        state = torch.zeros(bsz, num_heads, head_dim, head_dim, device=q.device, dtype=compute_dtype)
    else:
        state = initial_state.to(device=q.device, dtype=compute_dtype)

    outs = []
    for t in range(seq_len):
        qt = q[:, t].to(compute_dtype)
        kt = k[:, t].to(compute_dtype)
        vt = v[:, t].to(compute_dtype)
        gt = g[:, t].to(compute_dtype)
        bt = beta[:, t].to(compute_dtype)
        state = state * torch.exp(gt).unsqueeze(-1)
        state = state + bt.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhd,bhe->bhde", kt, vt)
        outs.append(torch.einsum("bhd,bhde->bhe", qt, state).to(q.dtype))
    o = torch.stack(outs, dim=1)
    return o, (state.to(q.dtype) if output_final_state else None)


class KimiDeltaAttentionPhi(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        num_kv_heads: int,
        layer_idx: int,
        opts: KDAOpts,
        attn_bias: bool = False,
        rms_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = q_dim // num_heads
        self.layer_idx = layer_idx
        self.mode = opts.kda_mode
        self.use_unpad = opts.kda_use_unpad
        self.use_fused_gate = opts.use_fused_kda_gate
        self.detach_kda_core_in_training = opts.detach_kda_core_in_training
        self.expand_kv_heads_for_kda = opts.expand_kv_heads_for_kda
        self.gate_scale = float(opts.gate_scale)
        self.force_python_kda_core = opts.force_python_kda_core

        self.q_proj = nn.Linear(hidden_size, q_dim, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=attn_bias)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=attn_bias)

        conv_cls = NativeShortConvolution if opts.use_native_short_conv else FLA_ShortConvolution
        if conv_cls is None:
            raise RuntimeError("use_native_short_conv=False but FLA ShortConvolution is unavailable.")
        self.q_conv1d = conv_cls(hidden_size=q_dim, kernel_size=opts.kda_conv_size, activation="silu")
        self.k_conv1d = conv_cls(hidden_size=kv_dim, kernel_size=opts.kda_conv_size, activation="silu")
        self.v_conv1d = conv_cls(hidden_size=kv_dim, kernel_size=opts.kda_conv_size, activation="silu")

        a_min = max(1e-6, float(min(opts.a_log_min, opts.a_log_max)))
        a_max = max(a_min + 1e-6, float(max(opts.a_log_min, opts.a_log_max)))
        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(a_min, a_max)).view(1, 1, -1, 1)
        )
        self.f_a_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, q_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(q_dim, dtype=torch.float32))
        nn.init.constant_(self.dt_bias, float(opts.dt_bias_init))
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_a_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, q_dim, bias=False)

        if opts.use_native_rmsnorm_gated:
            self.o_norm = NativeFusedRMSNormGated(self.head_dim, eps=rms_eps, activation="sigmoid")
        else:
            if FLA_FusedRMSNormGated is None:
                raise RuntimeError("use_native_rmsnorm_gated=False but FLA FusedRMSNormGated is unavailable.")
            self.o_norm = FLA_FusedRMSNormGated(self.head_dim, eps=rms_eps, activation="sigmoid")

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False, **kwargs):
        del attention_mask, position_ids, past_key_value, use_cache, kwargs
        hs = hidden_states
        q, _ = self.q_conv1d(self.q_proj(hs), cache=None, output_final_state=False, cu_seqlens=None)
        k, _ = self.k_conv1d(self.k_proj(hs), cache=None, output_final_state=False, cu_seqlens=None)
        v, _ = self.v_conv1d(self.v_proj(hs), cache=None, output_final_state=False, cu_seqlens=None)

        g = self.f_b_proj(self.f_a_proj(hs))
        g = _safe_python_kda_gate(g, self.A_log, self.head_dim, self.dt_bias, decay_scale=self.gate_scale)
        g = normalize_kda_gate_shape(g, self.num_heads, self.head_dim)
        beta = self.b_proj(hs).float().sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads)
        k = rearrange(k, "... (h d) -> ... h d", h=self.num_kv_heads)
        v = rearrange(v, "... (h d) -> ... h d", h=self.num_kv_heads)

        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=-2)
            v = v.repeat_interleave(rep, dim=-2)

        if g.dtype != q.dtype:
            g = g.to(q.dtype)
        o, _ = python_delta_rule_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=None, output_final_state=False)
        if self.training and self.detach_kda_core_in_training:
            o = o.detach()
        go = self.g_b_proj(self.g_a_proj(hs))
        go = rearrange(go, "... (h d) -> ... h d", h=self.num_heads)
        o = self.o_norm(o, go)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, None, None


def _get_out_proj(attn_module: nn.Module) -> nn.Module:
    for name in ("o_proj", "dense", "out_proj"):
        if hasattr(attn_module, name):
            return getattr(attn_module, name)
    raise AttributeError("Cannot find output projection in attention module.")


def replace_attention_3to1(model: nn.Module, opts: KDAOpts) -> Tuple[int, int]:
    replaced = 0
    for i, layer in enumerate(model.model.layers):
        if i % 4 == 3:
            continue
        old_attn = layer.self_attn
        if not all(hasattr(old_attn, n) for n in ("q_proj", "k_proj", "v_proj")):
            continue
        old_q, old_k, old_v = old_attn.q_proj, old_attn.k_proj, old_attn.v_proj
        old_o = _get_out_proj(old_attn)
        q_dim = old_q.weight.shape[0]
        kv_dim = old_k.weight.shape[0]
        hidden_size = old_q.weight.shape[1]
        num_heads = getattr(old_attn, "num_heads", getattr(model.config, "num_attention_heads", None))
        num_kv_heads = getattr(old_attn, "num_key_value_heads", getattr(model.config, "num_key_value_heads", max(1, kv_dim // (q_dim // num_heads))))
        attn_bias = old_q.bias is not None
        new_attn = KimiDeltaAttentionPhi(
            hidden_size=hidden_size,
            q_dim=q_dim,
            kv_dim=kv_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            layer_idx=i,
            opts=opts,
            attn_bias=attn_bias,
            rms_eps=getattr(model.config, "rms_norm_eps", 1e-5),
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
    return replaced, 0


def _load_shard(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        if safe_load_file is None:
            raise RuntimeError("safetensors is required to load .safetensors checkpoints.")
        return safe_load_file(path, device="cpu")
    return torch.load(path, map_location="cpu")


def load_checkpoint_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    single_safe = os.path.join(model_dir, "model.safetensors")
    safe_index = os.path.join(model_dir, "model.safetensors.index.json")
    single_bin = os.path.join(model_dir, "pytorch_model.bin")
    bin_index = os.path.join(model_dir, "pytorch_model.bin.index.json")

    if os.path.exists(single_safe):
        return _load_shard(single_safe)
    if os.path.exists(single_bin):
        obj = _load_shard(single_bin)
        return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    if os.path.exists(safe_index):
        with open(safe_index, "r", encoding="utf-8") as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        state = {}
        for s in shards:
            state.update(_load_shard(os.path.join(model_dir, s)))
        return state

    if os.path.exists(bin_index):
        with open(bin_index, "r", encoding="utf-8") as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        state = {}
        for s in shards:
            obj = _load_shard(os.path.join(model_dir, s))
            if isinstance(obj, dict) and "state_dict" in obj:
                obj = obj["state_dict"]
            state.update(obj)
        return state

    raise FileNotFoundError(f"No checkpoint file found under: {model_dir}")


def load_manifest(model_dir: str) -> SimpleNamespace:
    path = os.path.join(model_dir, "kda_reload_manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing manifest: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SimpleNamespace(**data)


def parse_dtype(name: str):
    n = name.lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    if n in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Directory containing final checkpoint + kda_reload_manifest.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--strict", action="store_true", help="Use strict state_dict load.")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    manifest = load_manifest(model_dir)
    kda_opts = KDAOpts(**manifest.kda_config)
    dtype = parse_dtype(args.dtype)

    base_cfg = AutoConfig.from_pretrained(
        manifest.model_id,
        trust_remote_code=True,
        revision=manifest.model_revision,
    )
    base_cfg = patch_rope_scaling(base_cfg)
    base_cfg.sliding_window = None
    base_cfg.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        manifest.model_id,
        config=base_cfg,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        revision=manifest.model_revision,
    ).to(args.device)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    replaced, _ = replace_attention_3to1(model, kda_opts)
    print(f"[reload] replaced attention blocks: {replaced}")

    state = load_checkpoint_state_dict(model_dir)
    missing, unexpected = model.load_state_dict(state, strict=args.strict)
    print(f"[reload] missing={len(missing)} unexpected={len(unexpected)} strict={args.strict}")
    if len(missing) > 0:
        print("  sample missing:", missing[:10])
    if len(unexpected) > 0:
        print("  sample unexpected:", unexpected[:10])

    model.eval()
    print("[reload] ready")
    print("  attn type layer0:", type(model.model.layers[0].self_attn).__name__)
    print("  device:", next(model.parameters()).device)
    print("  dtype:", next(model.parameters()).dtype)
    print("  vocab size:", tok.vocab_size)


if __name__ == "__main__":
    main()

