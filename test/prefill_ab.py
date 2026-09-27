#!/usr/bin/env python3
"""Controlled A/B of helios prefill, best-of-N per cell.

The reference arm is the SAME BINARY with the two mixer stages pinned back to their pre-change
kernels (HELIOS_GR_UP=0 -> gr_up_kernel's generic runtime-trip-count form, HELIOS_GR_DOTS=0 ->
the DBK=128 tiled gr_dots_kernel). Both pinned forms are byte-for-byte the kernels that were
there before, and test_aux asserts the pinned and default arms agree bit for bit, so this is a
true same-binary A/B and not two builds that might differ for unrelated reasons.

Every cell is run --reps times and the BEST is kept: the graph-decode work in this repo showed a
single prefill measurement varying by >20 % run to run, so a one-shot number here is noise.

Usage:
  test/prefill_ab.py --reps 3                     # both arms, 4k/8k/16k
  test/prefill_ab.py --reps 3 --cells 4k 8k       # subset
  test/prefill_ab.py --reps 3 --arm ref           # reference arm only
  test/prefill_ab.py --reps 3 --arm new
"""

import argparse
import os
import re
import subprocess
import sys

PREFILL_RE = re.compile(r"prefill\s+([0-9.]+)\s+tok/s")
TOKENS_RE = re.compile(r"prompt tokens=(\d+)")

CELLS = [("4k", 4000), ("8k", 8000), ("16k", 16000)]
TOK_PER_WORD = 1.18          # same constant test/grid_bench.py uses

# Nondeterministic MTP speculates; the byte-identity reference is measured without it.
BASE_ENV = {"HELIOS_MTP": "0", "HELIOS_PIPELINE": "1"}

ARMS = {
    # name: extra environment
    "ref": {"HELIOS_GR_UP": "0", "HELIOS_GR_DOTS": "0"},
    "new": {},
    # The tensor-core mixer (gr_mix_tc.cu), same binary, same everything else.
    "tc": {"HELIOS_MIXER_TC": "1"},
}


def prompt_for_tokens(tokens: int) -> str:
    return ("word " * int(tokens / TOK_PER_WORD)).strip()


def run_once(gen: str, model: str, prompt: str, env_extra: dict) -> tuple:
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(env_extra)
    proc = subprocess.run(
        [gen, "gen", model, "--raw", "--prompt", prompt, "--tokens", "1", "--temp", "0"],
        capture_output=True, text=True, timeout=3600, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"engine exited {proc.returncode}\n{proc.stdout[-2000:]}")
    m = PREFILL_RE.search(proc.stdout)
    n = TOKENS_RE.search(proc.stdout)
    if not m or not n:
        raise RuntimeError(f"no prefill line in output:\n{proc.stdout[-2000:]}")
    return float(m.group(1)), int(n.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="./build/helios")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-exl3"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cells", nargs="*", default=[c for c, _ in CELLS])
    ap.add_argument("--arm", choices=["ref", "new", "tc", "all"], default="both")
    a = ap.parse_args()

    arms = ["ref", "new"] if a.arm == "both" else (["ref", "new", "tc"] if a.arm == "all" else [a.arm])
    results = {}
    for label, ntok in CELLS:
        if label not in a.cells:
            continue
        prompt = prompt_for_tokens(ntok)
        for arm in arms:
            rates, actual = [], 0
            for _ in range(a.reps):
                rate, actual = run_once(a.gen, a.model, prompt, ARMS[arm])
                rates.append(rate)
            best = max(rates)
            results[(label, arm)] = best
            spread = (max(rates) - min(rates)) / best * 100 if best else 0.0
            print(f"[{label:>3} {arm}] best {best:7.1f} tok/s over {a.reps} runs "
                  f"({', '.join(f'{r:.1f}' for r in rates)}, spread {spread:.1f}%, "
                  f"{actual} prompt tokens)", flush=True)

    if len(arms) == 2:
        print()
        print(f"{'cell':>6} {'ref tok/s':>10} {'new tok/s':>10} {'delta':>9}")
        for label, _ in CELLS:
            if (label, "ref") not in results:
                continue
            r, n = results[(label, "ref")], results[(label, "new")]
            print(f"{label:>6} {r:10.1f} {n:10.1f} {(n / r - 1) * 100:+8.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
