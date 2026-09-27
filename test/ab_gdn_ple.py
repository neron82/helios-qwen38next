#!/usr/bin/env python3
"""End-to-end A/B of the PLE tensor-core GEMM + the tiled GDN conv/transpose.

Two binaries from the same tree:
  ref : the pre-change GDN conv1d + transpose_f32_bf16, run with HELIOS_GEMM_MMA=0 so the PLE
        projection also stays on the scalar kernel - i.e. exactly the engine as it was.
  new : all three changes, defaults.

Arms are interleaved rep by rep and the MEDIAN is reported (this repo has seen single prefill
measurements swing by >20 %, so best-of-N hides a regression and mean-of-N is dominated by one
outlier); min/max are printed too so the spread is visible.

Usage: test/ab_gdn_ple.py --reps 4 [--cells 4k 8k 16k] [--decode]
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-exl3")
REF = "/home/neron/tmpns/ab/helios_ref"
NEW = "/home/neron/projects/new_engine/helios-qwen/build/helios"
CELLS = [("4k", 4000), ("8k", 8000), ("16k", 16000)]
TOK_PER_WORD = 1.18

RATE = re.compile(r"prefill\s+([0-9.]+)\s+tok/s")
DEC = re.compile(r"decode\s+([0-9.]+)\s+tok/s")
PROMPT_TOKENS = re.compile(r"prompt tokens=(\d+)")

CONTINUE = ("\n\nContinue the discussion above at length. Do not repeat previous text; write the "
            "next several paragraphs of original commentary on the same subject, and keep writing.")
WORDS = ("The capital of France is Paris and the river that runs through it is the Seine, which "
         "has supported trade, art and daily life for more than two thousand years. ")


def prompt_for(tokens: int) -> str:
    n = max(1, int(tokens / TOK_PER_WORD) // len(WORDS.split()))
    return (WORDS * n).strip() + CONTINUE


def run(binary: str, env_extra: dict, prompt: str, tokens: int) -> tuple:
    env = dict(os.environ)
    env.update(env_extra)
    out = subprocess.run([binary, "gen", MODEL, "--raw", "--prompt", prompt,
                          "--tokens", str(tokens), "--temp", "0"],
                         capture_output=True, text=True, env=env, timeout=1800)
    m, d, p = RATE.search(out.stdout), DEC.search(out.stdout), PROMPT_TOKENS.search(out.stdout)
    if not (m and d and p):
        raise RuntimeError(f"no rate line from {binary}: {out.stdout[-400:]} {out.stderr[-400:]}")
    return float(m.group(1)), float(d.group(1)), int(p.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--cells", nargs="*", default=[c for c, _ in CELLS])
    ap.add_argument("--decode", action="store_true",
                    help="short-prompt decode cell (256 prompt tokens, 128 generated)")
    a = ap.parse_args()

    envs = {"ref": {"HELIOS_MTP": "0", "HELIOS_PIPELINE": "1", "HELIOS_GEMM_MMA": "0"},
            "new": {"HELIOS_MTP": "0", "HELIOS_PIPELINE": "1"}}
    bins = {"ref": REF, "new": NEW}
    cells = [c for c in CELLS if c[0] in a.cells]
    if a.decode:
        cells = cells + [("decode", 256)]

    print(f"{'cell':>7} {'tokens':>7}  {'ref median':>11} {'new median':>11} {'delta':>9} "
          f"{'ref spread':>18} {'new spread':>18}")
    for name, toks in cells:
        if name == "decode":
            gen = 128
        else:
            gen = 1
        prompt = prompt_for(toks)
        res = {"ref": [], "new": []}
        ptok = 0
        for r in range(a.reps):
            for arm in ("ref", "new", "new", "ref") if r == 0 else ("ref", "new"):
                pre, dec, ptok = run(bins[arm], envs[arm], prompt, gen)
                res[arm].append(dec if name == "decode" else pre)
        rm, nm = statistics.median(res["ref"]), statistics.median(res["new"])
        rs = f"{min(res['ref']):.1f}-{max(res['ref']):.1f}"
        ns = f"{min(res['new']):.1f}-{max(res['new']):.1f}"
        print(f"{name:>7} {ptok:>7}  {rm:>11.1f} {nm:>11.1f} {(nm/rm-1)*100:>8.2f}% "
              f"{rs:>18} {ns:>18}")
        sys.stdout.flush()
        with open("/tmp/ab_results.txt", "a") as f:
            f.write(f"cell {name} prompt_tokens {ptok} n {len(res[chr(39)+chr(39)] if False else res['ref'])} ref {rm:.1f} new {nm:.1f} delta {(nm/rm-1)*100:+.2f}% ref_spread {rs} new_spread {ns}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
