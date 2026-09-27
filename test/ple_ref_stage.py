#!/usr/bin/env python3
"""Replicate the reference PLE chain op-by-op on the real module, printing the same RMS values
the engine prints, so the divergent stage is identified exactly.

The reference fuses this into one ext call, so the intermediates are not observable through hooks;
forward_streams_reference() is the module's own op-by-op form and is used here instead.

Usage: python3 ple_ref_stage.py <model_dir> [--ids 1,2,3]
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer


def rms(t):
    t = t.float()
    return float(torch.sqrt(torch.mean(t * t)))


def main():
    model_dir = sys.argv[1]
    ids = [248044, 760, 6511, 314, 9338, 369]
    for i, a in enumerate(sys.argv):
        if a == "--ids" and i + 1 < len(sys.argv):
            ids = [int(x) for x in sys.argv[i + 1].split(",")]

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load(progressbar=False)
    cache = Cache(model, max_num_tokens=512)

    # locate the PLE module and the stream stack entering it
    ple = None
    for m in model.modules:
        subs = getattr(m, "modules", None) or []
        for s in subs:
            if type(s).__name__ == "PLELayer":
                ple = s
        if type(m).__name__ == "PLELayer":
            ple = m
    assert ple is not None, "PLELayer not found"

    emb_mod = model.modules[0]
    dev = getattr(emb_mod, "device", None) or torch.device("cuda:0")
    if not torch.is_tensor(dev) and str(dev).startswith("cpu"):
        dev = torch.device("cuda:0")
    if getattr(emb_mod, "embedding", None) is not None:
        dev = emb_mod.embedding.weight.device
    ids_t = torch.tensor([ids], dtype=torch.long, device=dev)
    emb = model.modules[0].forward(ids_t, {})                      # (1, S, D)
    emb = emb.to(dev)
    streams = emb.float().unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
    print(f"[ref] embed rms={rms(emb):.6f}  streams rms={rms(streams):.6f}")

    params = {}
    # token history: eos-padded context + the ids, as forward_streams expects
    eos = ple.ple_embedding.eos_token_id
    ctx = ple.ple_embedding.context_len
    history = torch.cat((torch.full((1, ctx), eos, dtype=torch.long, device=emb.device), ids_t), dim=1)

    # The PLE and its embedding prefer the CPU (they were designed for host-side hashing), so every
    # submodule that touches the device has to be pointed at it explicitly for this op-by-op replay.
    dev = streams.device
    for mod in (ple, ple.ple_embedding, ple.norm_key, ple.norm_query, ple.norm_conv):
        if hasattr(mod, "device"):
            mod.device = dev
    # forward() needs its disk/RAM staging configured; forward_reference() is the module's own
    # pure-torch pipeline (hashing + codec) and needs none of that, so use it here.
    out_len = history.shape[1] - ple.ple_embedding.context_len
    ngram = ple.ple_embedding.forward_reference(history, params, out_dtype=torch.half)
    print(f"[ref] ngram_embedding rms={rms(ngram):.6f}")

    key = ple.key_proj.forward(ngram, params).view(1, -1, config.hc_mult, config.hidden_size)
    print(f"[ref] key_proj rms={rms(key):.6f}")
    key = ple.norm_key.forward(key, params, out_dtype=torch.float)
    print(f"[ref] norm_key rms={rms(key):.6f}")
    value = ple.value_proj.forward(ngram, params)
    print(f"[ref] value_proj rms={rms(value):.6f}")
    ple.device = streams.device
    query = ple.norm_query.forward(streams, params, out_dtype=torch.float)
    print(f"[ref] norm_query rms={rms(query):.6f}")

    gate = torch.bmm(query.view(-1, 1, config.hidden_size),
                     key.reshape(-1, config.hidden_size, 1)).view(1, -1, config.hc_mult)
    print(f"[ref] gate rms={rms(gate):.6f} scale={ple.gate_scale:.8f}")

    gated = torch.empty((1, gate.shape[1], config.hc_mult, config.hidden_size),
                        dtype=torch.float, device=value.device)
    from exllamav3.ext import exllamav3_ext as ext
    ext.ple_gate(gate, value, gated, ple.gate_scale)
    print(f"[ref] gated rms={rms(gated):.6f}")

    normed = ple.norm_conv.forward(gated, params, out_dtype=torch.half).flatten(-2)
    print(f"[ref] norm_conv rms={rms(normed):.6f}")

    conv_out, _ = ple._short_conv(normed, None)
    print(f"[ref] conv_out rms={rms(conv_out):.6f}")

    delta = gated + conv_out.view(1, gate.shape[1], config.hc_mult, config.hidden_size)
    print(f"[ref] delta rms={rms(delta):.6f}")
    print(f"[ref] streams after rms={rms(streams + delta):.6f}")


if __name__ == "__main__":
    main()
