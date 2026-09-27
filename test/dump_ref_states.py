#!/usr/bin/env python3
"""Dump per-module hidden states from exllamav3 for one prompt.

Runs the reference model's own forward on a fixed token sequence and writes the hidden state
entering/leaving each module to a .npz, so the port can be diffed layer by layer instead of
guessed at. Requires the model to fit, so the baseline server must be stopped first.

Usage: python3 dump_ref_states.py <model_dir> <out.npz> [--prompt "text"]
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from exllamav3 import Config, Model, Cache, Tokenizer


def main():
    model_dir = sys.argv[1]
    out_path = sys.argv[2]
    prompt = "The capital of France is"
    for i, a in enumerate(sys.argv):
        if a == "--prompt" and i + 1 < len(sys.argv):
            prompt = sys.argv[i + 1]

    config = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    model.load(progressbar=False)
    cache = Cache(model, max_num_tokens=512)

    ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
    print(f"[dump] prompt={prompt!r} tokens={ids.shape[-1] if torch.is_tensor(ids) else len(ids)}")

    captured = {}

    def hook(name):
        def fn(module, args, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().float().cpu().numpy()
            elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                captured[name] = output[0].detach().float().cpu().numpy()
        return fn

    # exllamav3 modules are not nn.Modules (no register_forward_hook), so wrap each one's forward
    # with a recording closure instead.
    # Walk recursively so sublayer modules (linear_attn, mlp, the hc sites) are captured too.
    all_mods = []
    def walk(mods, prefix):
        for j, m in enumerate(mods):
            nm = getattr(m, "key", None) or f"{prefix}.{j}"
            all_mods.append((nm, m))
            subs = getattr(m, "modules", None)
            if subs:
                walk(subs, nm)
    walk(model.modules, "top")

    saved = []
    for idx, (key, m) in enumerate(all_mods):
        name = f"{idx}:{key}"
        orig = m.forward

        def make(orig_f, nm):
            def wrapped(*a, **kw):
                out = orig_f(*a, **kw)
                if isinstance(out, torch.Tensor):
                    captured[nm] = out.detach().float().cpu().numpy()
                elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                    captured[nm] = out[0].detach().float().cpu().numpy()
                return out
            return wrapped

        try:
            m.forward = make(orig, name)
            saved.append((m, orig))
        except Exception:
            pass

    try:
        with torch.no_grad():
            params = {}
            model.forward(ids, params)
    finally:
        for m, orig in saved:
            m.forward = orig

    # keep the shapes and a few stats rather than every tensor, so the file stays small
    summary = {}
    for k, v in captured.items():
        flat = v.reshape(-1)
        summary[k] = dict(shape=v.shape, rms=float(np.sqrt(np.mean(flat ** 2))),
                          mean=float(flat.mean()), first8=v.reshape(-1)[:8])
    np.savez_compressed(out_path, **{f"v_{i}": captured[k] for i, k in enumerate(captured)})
    with open(out_path + ".txt", "w") as f:
        for k in captured:
            s = summary[k]
            f.write(f"{k}\tshape={tuple(s['shape'])}\trms={s['rms']:.6f}\tmean={s['mean']:.6f}\n")
    print(f"[dump] wrote {len(captured)} module outputs to {out_path}")
    print(f"[dump] summary: {out_path}.txt")


if __name__ == "__main__":
    main()
