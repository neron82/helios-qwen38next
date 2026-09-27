#!/usr/bin/env python3
"""Greedy parity check: helios vs the exllamav3 baseline server on fixed prompts.

Runs each prompt through the running baseline server and through the engine binary, then compares
the generated text byte-for-byte. Both sides are driven with temperature 0 and the same max_tokens,
so any difference is a real divergence.

Usage: python3 test/parity.py [--engine ./build/helios] [--port 8080] [--tokens 24]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

PROMPTS = [
    "The capital of France is",
    "1+1=",
    "def fibonacci(n):",
    "The quick brown fox jumps over the lazy",
    "Q: What is the tallest mountain on Earth?\nA:",
]


def server_generate(port, prompt, tokens):
    body = json.dumps({"model": "q", "prompt": prompt, "max_tokens": tokens,
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["text"]


def engine_generate(binary, model_dir, prompt, tokens):
    env = dict(os.environ, HELIOS_MTP="0")
    out = subprocess.run([binary, "gen", model_dir, "--raw", "--prompt", prompt,
                          "--tokens", str(tokens), "--temp", "0"],
                         capture_output=True, text=True, env=env, timeout=1800)
    # the CLI streams decoded text between "[gen] prompt tokens=" and the trailing "[gen] N tokens in"
    m = re.search(r"\[gen\] prompt tokens=\d+\n(.*?)\n\[gen\] ", out.stderr + out.stdout, re.S)
    if not m:
        m = re.search(r"\[gen\] prompt tokens=\d+\n(.*)\Z", out.stdout, re.S)
    return (m.group(1) if m else "").strip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./build/helios")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-exl3"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--capture", metavar="FILE",
                    help="query the baseline server and save its outputs (run with the server up)")
    ap.add_argument("--compare", metavar="FILE",
                    help="run the engine and compare against a captured file (run with the GPUs free)")
    a = ap.parse_args()

    # The baseline and the engine each need both GPUs, so they cannot be resident at once: the
    # reference is captured in one pass and compared in the next.
    if a.capture:
        refs = {p: server_generate(a.port, p, a.tokens) for p in PROMPTS}
        with open(a.capture, "w") as f:
            json.dump({"tokens": a.tokens, "prompts": refs}, f, ensure_ascii=False, indent=1)
        print(f"[capture] {len(refs)} reference completions -> {a.capture}")
        return 0

    if not a.compare:
        ap.error("pass --capture FILE (server up) or --compare FILE (GPUs free)")
    with open(a.compare) as f:
        cap = json.load(f)
    refs = cap["prompts"]

    passed = 0
    for p in PROMPTS:
        ref = refs[p]
        got = engine_generate(a.engine, a.model, p, cap["tokens"])
        ok = ref == got
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {p!r}")
        if not ok:
            print(f"        server: {ref!r}")
            print(f"        helios: {got!r}")
    print(f"\n{passed}/{len(PROMPTS)} prompts token-identical at temperature 0")
    return 0 if passed == len(PROMPTS) else 1


if __name__ == "__main__":
    sys.exit(main())
