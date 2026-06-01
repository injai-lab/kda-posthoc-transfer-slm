#!/usr/bin/env python3
import argparse
import dataclasses
import gc
import os
import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from load_kda_stageA import KDAOpts, load_manifest, patch_rope_scaling, load_checkpoint_state_dict
from kda_infer_check_1to1 import patch_4d_mask, parse_dtype, build_inputs, replace_attention_3to1

def _patch_rope_for_qwen(cfg):
    rope_params = getattr(cfg, "rope_parameters", None)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    rope_theta = getattr(cfg, "rope_theta", None) or 1000000.0

    def _norm(d):
        if not isinstance(d, dict):
            return None
        out = dict(d)
        if "rope_type" in out and "type" not in out:
            out["type"] = out["rope_type"]
        if "type" in out and "rope_type" not in out:
            out["rope_type"] = out["type"]
        out["rope_theta"] = out.get("rope_theta", rope_theta)
        return out

    rp = _norm(rope_params)
    rs = _norm(rope_scaling)

    if rp is not None:
        cfg.rope_parameters = rp
        if getattr(cfg, "rope_scaling", None) is None:
            cfg.rope_scaling = dict(rp)
    elif rs is not None:
        cfg.rope_scaling = rs
        cfg.rope_parameters = dict(rs)
    else:
        base = {"rope_type": "default", "type": "default", "rope_theta": rope_theta}
        cfg.rope_parameters = dict(base)
        cfg.rope_scaling = dict(base)

    if getattr(cfg, "rope_theta", None) is None:
        cfg.rope_theta = rope_theta

    return cfg

def step_key(path: str) -> int:
    import os
    import re
    m = re.search(r"checkpoint-(\d+)$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9

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
    cfg = _patch_rope_for_qwen(cfg)
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
        "안녕하세요. 한 줄로 자기소개 해줘.",
        "3+5는 얼마야? 답만 말해.",
        "너는 누구야? 한 문장으로 답해.",
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
                    use_cache=False,
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
