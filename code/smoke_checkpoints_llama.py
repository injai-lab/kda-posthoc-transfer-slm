#!/usr/bin/env python3
import argparse
import dataclasses
import gc
import os
import re
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 기존 helper import
from load_kda_stageA import KDAOpts, load_manifest, patch_rope_scaling, load_checkpoint_state_dict
from kda_infer_check import patch_4d_mask, parse_dtype, build_inputs, replace_attention_3to1


def step_key(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def _normalize_llama_kda_outputs(model):
    """
    Llama decoder layer expects self_attn(...) -> (hidden_states, attn_weights)
    Some KDA helpers return 3 outputs. Normalize them to 2.
    """
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if getattr(attn, "_llama_output_patched", False):
            continue

        orig_forward = attn.forward

        def _wrapped_forward(*args, _orig_forward=orig_forward, **kwargs):
            out = _orig_forward(*args, **kwargs)
            if isinstance(out, tuple):
                if len(out) >= 2:
                    return out[0], out[1]
            return out

        attn.forward = _wrapped_forward
        attn._llama_output_patched = True


def load_model(manifest_dir: str, checkpoint_dir: str, device: str, dtype):
    manifest = load_manifest(manifest_dir)
    allowed = {f.name for f in dataclasses.fields(KDAOpts)}
    opts = KDAOpts(**{k: v for k, v in manifest.kda_config.items() if k in allowed})

    cfg = AutoConfig.from_pretrained(
        manifest.model_id,
        trust_remote_code=True,
        revision=manifest.model_revision or None,
    )
    cfg = patch_rope_scaling(cfg)
    cfg.sliding_window = None
    cfg.use_cache = False

    try:
        model = AutoModelForCausalLM.from_config(
            cfg,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device=device, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_config(
            cfg,
            trust_remote_code=True,
        ).to(device=device, dtype=dtype)

    tok = AutoTokenizer.from_pretrained(manifest_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    replaced, gqa_fixed = replace_attention_3to1(model, opts)
    _normalize_llama_kda_outputs(model)

    state = load_checkpoint_state_dict(checkpoint_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[load] ckpt={os.path.basename(checkpoint_dir)} replaced={replaced} gqa_dim_fix={gqa_fixed}")
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    return model.eval(), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--prompt", action="append")
    args = ap.parse_args()

    patch_4d_mask()
    dtype = parse_dtype(args.dtype)

    out_dir = os.path.abspath(args.out_dir)
    manifest_dir = os.path.abspath(args.manifest_dir or os.path.join(out_dir, "final_stage1"))
    prompts = args.prompt or [
        "3+5=",
        "Q: 3+5는?\nA:",
        "안녕하세요. 저는",
        "Hello, my name is",
        "I am an AI",
    ]

    ckpts = sorted(
        [os.path.join(out_dir, x) for x in os.listdir(out_dir) if re.match(r"checkpoint-\d+$", x)],
        key=step_key,
    )

    if not ckpts:
        raise SystemExit(f"No checkpoints found under {out_dir}")

    for ckpt in ckpts:
        model, tok = load_model(manifest_dir, ckpt, args.device, dtype)
        for i, prompt in enumerate(prompts, 1):
            model_inputs = build_inputs(tok, prompt, args.device)
            prompt_len = model_inputs["input_ids"].shape[1]
            with torch.no_grad():
                y = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            out = tok.decode(y[0][prompt_len:], skip_special_tokens=True).strip()
            print(f"\n[{os.path.basename(ckpt)}] Prompt {i}")
            print(prompt)
            print("->", out if out else "[EMPTY]")
        print("\n" + "=" * 80 + "\n")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
