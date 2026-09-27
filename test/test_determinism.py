#!/usr/bin/env python3
"""Greedy-decode determinism gate.

Repeated identical greedy generations of the same prompt MUST produce identical
token ids. This is a real gate, not a smoke test: the engine was found to diverge
at ~token 30 while a two-sample md5 comparison said it was reproducible, so
"compare twice" is exactly the check that is too weak to catch this.

Run N independent processes (the engine allocates and loads the model per
process, so this also catches state that leaks across a fresh load) and assert
every run emits the same token sequence. N>=4 by default because the observed
failure mode was bimodal - several runs agreeing and one differing - which a
two-run check passes half the time.

Usage:  test_determinism.py [--runs N] [--tokens N] [--model DIR] [--gen BIN]
Exit 0 on pass, 1 on divergence, 2 on setup failure.
"""

import argparse
import os
import re
import subprocess
import sys
import time

# The engine interleaves streamed text with its [ids] N lines, so "[ids] 1234"
# can appear mid-line as " AI[ids] 11". Match the marker, not the line start.
IDS_RE = re.compile(r"\[ids\]\s+(\d+)")


def run_once(gen: str, model: str, prompt: str, tokens: int) -> list:
    env = dict(os.environ)
    # MTP is a greedy proposer; leave it off so this measures the base decode
    # path. Speculation has its own, separate, non-reproducibility.
    env["HELIOS_MTP"] = "0"
    env["HELIOS_IDS"] = "1"
    t0 = time.time()
    proc = subprocess.run(
        [gen, "gen", model, "--raw", "--prompt", prompt,
         "--tokens", str(tokens), "--temp", "0"],
        capture_output=True, text=True, timeout=1800, env=env,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"engine exited {proc.returncode}\n{out[-2000:]}")
    ids = [int(m) for m in IDS_RE.findall(out)]
    if not ids:
        raise RuntimeError(
            f"no [ids] tokens in output (HELIOS_IDS not honoured?).\n{out[-2000:]}")
    if time.time() - t0 < 1.0:
        raise RuntimeError("run finished implausibly fast; model probably not loaded")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4,
                    help="independent processes; >=4 because the failure is bimodal")
    ap.add_argument("--tokens", type=int, default=96,
                    help="must exceed the observed divergence point (~30)")
    ap.add_argument("--model", default=os.path.expanduser(
        "~/models/Qwen3.8-Flash-Next-exl3"))
    ap.add_argument("--gen", default="./build/helios")
    ap.add_argument("--prompt", default="What's the weather in Paris?")
    a = ap.parse_args()

    if a.runs < 2:
        print("DETERMINISM: need at least 2 runs to compare", file=sys.stderr)
        return 2
    if not os.path.exists(a.model):
        print(f"DETERMINISM: model not found: {a.model}", file=sys.stderr)
        return 2
    if not os.path.exists(a.gen):
        print(f"DETERMINISM: engine not found: {a.gen}", file=sys.stderr)
        return 2

    seqs = []
    for i in range(a.runs):
        try:
            seqs.append(run_once(a.gen, a.model, a.prompt, a.tokens))
        except Exception as e:                      # noqa: BLE001 - report and fail
            print(f"DETERMINISM: run {i+1} failed: {e}", file=sys.stderr)
            return 2
        print(f"  run {i+1}: {len(seqs[-1])} tokens", flush=True)

    uniq = {tuple(s) for s in seqs}
    if len(uniq) == 1:
        print(f"DETERMINISM PASS: {a.runs} runs, {len(seqs[0])} tokens, all identical")
        return 0

    n = min(len(s) for s in seqs)
    first = next((k for k in range(n) if len({s[k] for s in seqs}) > 1), None)
    print(f"DETERMINISM FAIL: {len(uniq)}/{a.runs} distinct token sequences")
    if first is None:
        print("  sequences differ in LENGTH only")
    else:
        print(f"  first divergence at token index {first} (0-based):")
        for i, s in enumerate(seqs):
            ctx = s[max(0, first - 4):first + 4]
            print(f"    run {i+1}: id={s[first]:<7} context={ctx}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
