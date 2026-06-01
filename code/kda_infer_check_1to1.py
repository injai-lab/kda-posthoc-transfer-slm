#!/usr/bin/env python3
import argparse
import dataclasses
import inspect
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_attn_mask_utils as attn_mask_utils

from load_kda_stageA import KDAOpts, load_manifest, patch_rope_scaling, load_checkpoint_state_dict

try:
    from fla.modules import ShortConvolution as FLA_ShortConvolution
except Exception:
    FLA_ShortConvolution = None

try:
    from fla.modules import FusedRMSNormGated as FLA_FusedRMSNormGated
except Exception:
    FLA_FusedRMSNormGated = None

try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
except Exception:
    chunk_kda = None
    fused_recurrent_kda = None

try:
    from fla.ops.kda.gate import fused_kda_gate
except Exception:
    fused_kda_gate = None

try:
    from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
except Exception:
    get_unpad_data = None
    index_first_axis = None
    pad_input = None


def patch_4d_mask() -> None:
    def _safe_4d_causal_attention_mask(
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        sliding_window=None,
        cache_position=None,
    ):
        del past_key_values_length, sliding_window, cache_position
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
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_slimmoe") and hasattr(mod, "_prepare_4d_causal_attention_mask"):
            mod._prepare_4d_causal_attention_mask = _safe_4d_causal_attention_mask


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
            raise ValueError(f"Unsupported native short-conv activation: {activation}")

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


def _safe_python_kda_gate(g: torch.Tensor, a_log: torch.Tensor, head_dim: int, dt_bias: torch.Tensor, decay_scale: float = 1.0) -> torch.Tensor:
    if g.dim() == 3:
        if g.size(-1) % head_dim != 0:
            raise ValueError(f"Cannot reshape g of last dim {g.size(-1)} with head_dim={head_dim}")
        num_heads = g.size(-1) // head_dim
        g = rearrange(g, "... (h d) -> ... h d", h=num_heads)
    elif g.dim() != 4:
        raise ValueError(f"Unexpected g rank for fallback gate: {g.dim()}")
    db = dt_bias.view(1, 1, -1, head_dim).to(g.dtype)
    dt = torch.nn.functional.softplus(g + db)
    decay = (-torch.exp(a_log.float())).to(dt.dtype) * dt * float(decay_scale)
    return decay


def normalize_kda_gate_shape(g: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if g.dim() == 3:
        if g.size(-1) != num_heads * head_dim:
            raise ValueError(f"Unexpected gate hidden size: got {g.size(-1)}, expected {num_heads * head_dim}.")
        return rearrange(g, "b t (h d) -> b t h d", h=num_heads)
    if g.dim() == 4:
        if g.size(-2) == num_heads and g.size(-1) == head_dim:
            return g
        if g.size(-1) == num_heads * head_dim:
            return rearrange(g, "... (h d) -> ... h d", h=num_heads)
        raise ValueError(f"Unexpected 4D gate shape: {tuple(g.shape)}")
    raise ValueError(f"Unsupported gate rank: {g.dim()}")


def call_fused_kda_gate(g, a_log, head_dim, dt_bias, decay_scale: float = 1.0):
    if fused_kda_gate is None:
        return _safe_python_kda_gate(g, a_log.float(), head_dim, dt_bias.float(), decay_scale=decay_scale)
    try:
        sig = inspect.signature(fused_kda_gate)
        param_names = set(sig.parameters.keys())
    except Exception:
        param_names = set()
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
        try:
            return fused_kda_gate(g, a32, head_dim, d32)
        except Exception:
            return fused_kda_gate(g, a32, d32)
    except RuntimeError as e:
        if "illegal memory access" in str(e).lower():
            return _safe_python_kda_gate(g, a32, head_dim, d32, decay_scale=decay_scale)
        raise


def python_delta_rule_kda(q, k, v, g, beta, initial_state=None, output_final_state=False):
    bsz, seq_len, num_heads, head_dim = q.shape
    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    if initial_state is None:
        state = torch.zeros(bsz, num_heads, head_dim, head_dim, device=q.device, dtype=compute_dtype)
    else:
        state = initial_state.to(dtype=compute_dtype, device=q.device)
    outs = []
    for t in range(seq_len):
        qt = q[:, t].to(compute_dtype)
        kt = k[:, t].to(compute_dtype)
        vt = v[:, t].to(compute_dtype)
        gt = g[:, t].to(compute_dtype)
        bt = beta[:, t].to(compute_dtype)
        state = state * torch.exp(gt).unsqueeze(-1)
        outer = torch.einsum("bhd,bhe->bhde", kt, vt)
        state = state + bt.unsqueeze(-1).unsqueeze(-1) * outer
        outs.append(torch.einsum("bhd,bhde->bhe", qt, state).to(q.dtype))
    o = torch.stack(outs, dim=1)
    return o, (state.to(q.dtype) if output_final_state else None)


def call_chunk_kda(q, k, v, g, beta, cu_seqlens=None, A_log=None, dt_bias=None, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False):
    if chunk_kda is None:
        return python_delta_rule_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=None, output_final_state=False)
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


def call_fused_recurrent_kda(q, k, v, g, beta, cu_seqlens=None, A_log=None, dt_bias=None, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False):
    if fused_recurrent_kda is None:
        return python_delta_rule_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=None, output_final_state=False)
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


class KimiDeltaAttentionPhi(nn.Module):
    def __init__(self, hidden_size, q_dim, kv_dim, num_heads, num_kv_heads, layer_idx, opts, attn_bias=False, rms_eps=1e-5):
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
            raise ValueError(f"head_dim mismatch: q_head_dim={self.head_dim}, kv_head_dim={self.kv_head_dim}.")
        self.layer_idx = layer_idx
        self.mode = opts.kda_mode
        self.use_unpad = opts.kda_use_unpad
        self.use_fused_gate = opts.use_fused_kda_gate
        self.detach_kda_core_in_training = opts.detach_kda_core_in_training
        self.expand_kv_heads_for_kda = opts.expand_kv_heads_for_kda
        self.gate_scale = float(opts.gate_scale)
        self.force_python_kda_core = bool(opts.force_python_kda_core)
        self.use_qk_l2norm_in_kernel = bool(getattr(opts, "use_qk_l2norm_in_kernel", True))
        self.use_gate_in_kernel = bool(getattr(opts, "use_gate_in_kernel", True))

        self.q_proj = nn.Linear(hidden_size, q_dim, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=attn_bias)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=attn_bias)

        conv_cls = NativeShortConvolution if opts.use_native_short_conv else FLA_ShortConvolution
        if conv_cls is None:
            raise RuntimeError("Requested FLA ShortConvolution but it is unavailable.")
        self.q_conv1d = conv_cls(hidden_size=q_dim, kernel_size=opts.kda_conv_size, activation="silu")
        self.k_conv1d = conv_cls(hidden_size=kv_dim, kernel_size=opts.kda_conv_size, activation="silu")
        self.v_conv1d = conv_cls(hidden_size=kv_dim, kernel_size=opts.kda_conv_size, activation="silu")

        a_min = max(1e-6, float(min(opts.a_log_min, opts.a_log_max)))
        a_max = max(a_min + 1e-6, float(max(opts.a_log_min, opts.a_log_max)))
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(a_min, a_max)).view(1, 1, -1, 1))
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
                raise RuntimeError("Requested FLA FusedRMSNormGated but it is unavailable.")
            self.o_norm = FLA_FusedRMSNormGated(self.head_dim, eps=rms_eps, activation="sigmoid")

    @staticmethod
    def _mask4d_to_2d(attention_mask_4d: torch.Tensor) -> torch.Tensor:
        diag = attention_mask_4d[:, 0].diagonal(dim1=-2, dim2=-1)
        return (diag > -5000).to(torch.long)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False, **kwargs):
        del position_ids
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
            if get_unpad_data is None or index_first_axis is None:
                raise RuntimeError("kda_use_unpad=True but fla.layers.utils is unavailable.")
            indices, cu_seqlens, _ = get_unpad_data(padding_mask[:, -q_len:])
            hs = index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices).unsqueeze(0)

        q, _ = self.q_conv1d(x=self.q_proj(hs), cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(x=self.k_proj(hs), cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(x=self.v_proj(hs), cache=None, output_final_state=False, cu_seqlens=cu_seqlens)

        g_raw = self.f_b_proj(self.f_a_proj(hs))
        core_use_gate_in_kernel = bool(self.use_gate_in_kernel and not self.force_python_kda_core)
        if core_use_gate_in_kernel:
            g_core = normalize_kda_gate_shape(g_raw, self.num_heads, self.head_dim)
        else:
            if self.use_fused_gate:
                g_core = call_fused_kda_gate(g_raw, self.A_log, self.head_dim, self.dt_bias, decay_scale=self.gate_scale)
            else:
                g_core = _safe_python_kda_gate(g_raw, self.A_log, self.head_dim, self.dt_bias, decay_scale=self.gate_scale)
            g_core = normalize_kda_gate_shape(g_core, self.num_heads, self.head_dim)
        beta = self.b_proj(hs).float().sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads)
        k = rearrange(k, "... (h d) -> ... h d", h=self.num_kv_heads)
        v = rearrange(v, "... (h d) -> ... h d", h=self.num_kv_heads)

        if self.num_kv_heads != self.num_heads:
            if not self.expand_kv_heads_for_kda:
                raise ValueError("num_kv_heads != num_heads requires grouped-state KDA kernel support.")
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=-2)
            v = v.repeat_interleave(rep, dim=-2)

        if g_core.dtype != q.dtype:
            g_core = g_core.to(q.dtype)

        if self.force_python_kda_core:
            o, _ = python_delta_rule_kda(q=q, k=k, v=v, g=g_core, beta=beta, initial_state=None, output_final_state=False)
        elif mode == "chunk":
            o, _ = call_chunk_kda(
                q=q, k=k, v=v, g=g_core, beta=beta, cu_seqlens=cu_seqlens,
                A_log=self.A_log if core_use_gate_in_kernel else None,
                dt_bias=self.dt_bias if core_use_gate_in_kernel else None,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                use_gate_in_kernel=core_use_gate_in_kernel,
            )
        else:
            o, _ = call_fused_recurrent_kda(
                q=q, k=k, v=v, g=g_core, beta=beta, cu_seqlens=cu_seqlens,
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
            if pad_input is None:
                raise RuntimeError("kda_use_unpad=True but fla.layers.utils is unavailable.")
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        if getattr(self, "_return_2tuple", False):
            return o, None
        return o, None, past_key_value


def _get_out_proj(attn_module: nn.Module) -> nn.Module:
    for name in ("o_proj", "dense", "out_proj"):
        if hasattr(attn_module, name):
            return getattr(attn_module, name)
    raise AttributeError("Cannot find output projection in attention module.")


def _sync_phi_attn_dims(module, _inputs=None):
    if isinstance(module, KimiDeltaAttentionPhi):
        return
    if not (hasattr(module, "q_proj") and hasattr(module, "num_heads")):
        return
    q_out = int(module.q_proj.weight.shape[0])
    out_w = None
    for name in ("o_proj", "dense", "out_proj"):
        proj = getattr(module, name, None)
        if proj is not None and hasattr(proj, "weight") and proj.weight.ndim == 2:
            out_w = int(proj.weight.shape[1])
            break
    hidden = out_w if (out_w is not None and out_w > 0) else q_out
    module.hidden_size = hidden
    nh = int(module.num_heads)
    if nh > 0 and (hidden % nh == 0):
        module.head_dim = hidden // nh


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


def replace_attention_3to1(model: nn.Module, opts: KDAOpts):
    replaced = 0
    for i, layer in enumerate(model.model.layers):
        if i % 2 == 1:
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
        if getattr(model.config, "model_type", "") == "qwen3":
            new_attn._return_2tuple = True
        layer.self_attn = new_attn
        replaced += 1
    gqa_fixed = apply_phi_attn_dim_fix(model)
    return replaced, gqa_fixed


def parse_dtype(name: str):
    n = name.lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    if n in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_inputs(tok, prompt: str, device: str):
    if getattr(tok, "chat_template", None):
        try:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            if isinstance(enc, dict):
                return {k: v.to(device) for k, v in enc.items()}
        except TypeError:
            pass

        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids, device=device)}

    enc = tok(
        f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", action="append", required=True)
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    manifest = load_manifest(model_dir)
    allowed = {f.name for f in dataclasses.fields(KDAOpts)}
    opts = KDAOpts(**{k: v for k, v in manifest.kda_config.items() if k in allowed})
    dtype = parse_dtype(args.dtype)

    patch_4d_mask()

    cfg = AutoConfig.from_pretrained(
        manifest.model_id,
        trust_remote_code=True,
        revision=manifest.model_revision,
    )
    cfg = patch_rope_scaling(cfg)
    cfg.sliding_window = None
    cfg.use_cache = False

    try:
        model = AutoModelForCausalLM.from_config(
            cfg,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device=args.device, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_config(
            cfg,
            trust_remote_code=True,
        ).to(device=args.device, dtype=dtype)

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    replaced, gqa_fixed = replace_attention_3to1(model, opts)
    state = load_checkpoint_state_dict(model_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[reload] replaced={replaced} gqa_dim_fix={gqa_fixed}")
    print(f"[reload] missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()

    for i, prompt in enumerate(args.prompt, 1):
        model_inputs = build_inputs(tok, prompt, args.device)
        prompt_len = model_inputs["input_ids"].shape[1]

        with torch.no_grad():
            y = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        out = tok.decode(y[0][prompt_len:], skip_special_tokens=True).strip()
        print(f"\n=== Prompt {i} ===")
        print(prompt)
        print("--- Output ---")
        print(out if out else "[EMPTY]")


if __name__ == "__main__":
    main()
