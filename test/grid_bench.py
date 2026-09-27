#!/usr/bin/env python3
"""1:1 head-to-head grid: exllamav3 baseline vs helios, prefill x output.

Reuses bench.py's build_prompt / server_bench / engine_bench verbatim so both engines are measured
with identical methodology (unique per-request nonce to defeat the prefix cache; prefill and decode
timed as separate requests so decode never contains prefill).

Grid: prefill prompt lengths in TOKENS {4k, 8k, 16k} x output tokens {512, 1000, 2000}.

Because both engines need both GPUs, run the two passes sequentially:
  # 1) baseline (server up):   python3 test/grid_bench.py --capture /tmp/grid_ref.json --port 8080
  # 2) helios  (server down):  python3 test/grid_bench.py --compare /tmp/grid_ref.json --engine ./build/helios
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_prompt_for_bench as build_prompt, server_bench, engine_bench  # identical methodology

# prefill target TOKEN counts -> the paragraph runs ~1.18 tok/word, so scale words by /1.18.
PREFILL_TOKENS = [("4k", 4000), ("8k", 8000), ("16k", 16000)]
OUTPUT_TOKENS = [512, 1000, 2000]
TOK_PER_WORD = 1.18


def prompt_for_tokens(tokens):
    return build_prompt(int(tokens / TOK_PER_WORD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./build/helios")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-exl3"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--reps", type=int, default=2, help="repeats per cell; keep the best (steady clocks)")
    ap.add_argument("--capture")
    ap.add_argument("--compare")
    a = ap.parse_args()

    if a.capture:
        res = {}
        for plabel, ptok in PREFILL_TOKENS:
            p = prompt_for_tokens(ptok)
            res[plabel] = {}
            for out in OUTPUT_TOKENS:
                # unique nonce per request, paired t1/tn (same prompt body, different nonce) so the
                # subtraction removes exactly the prefill.
                #
                # completion_tokens is taken ONLY from the server's usage. Defaulting a missing
                # value to the REQUESTED count invents tokens that were never generated and
                # inflates the rate without bound - that is what produced a "2186.8 tok/s" cell.
                # A response without the field is recorded as a failed sample, not a fast one.
                pre_rates, dec_rates, actual_pt, got = [], [], 0, 0
                for i in range(a.reps):
                    t1, u1 = server_bench(a.port, f"nonceA{plabel}{out}A{i} {p}", 1)
                    tn, un = server_bench(a.port, f"nonceB{plabel}{out}B{i} {p}", out)
                    actual_pt = u1.get("prompt_tokens", 0)
                    pre_rates.append(actual_pt / max(t1, 1e-9))
                    if "completion_tokens" not in un:
                        continue                      # no usage -> no trustworthy count
                    got = un["completion_tokens"]
                    if got <= 0:
                        continue
                    dec_rates.append(got / max(tn - t1, 1e-9))
                ok = bool(dec_rates)
                # MEDIAN, not max. Taking the best of N reps is defensible for prefill, where the
                # spread is under 1%, but it is the wrong statistic for decode: a short generation is
                # a short timing sample, so its maximum is mostly a lucky-sample read. It produced
                # cells that could not be true - exllamav3 measuring 66.1 tok/s on a 512-token
                # generation while measuring 60.5 on a 2000-token one, when a longer generation must
                # amortise a fixed cost better. The median of the same samples is the honest figure.
                med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0.0
                res[plabel][out] = {"prefill": med(pre_rates), "decode": med(dec_rates) if ok else 0.0,
                                    "prompt_tokens": actual_pt, "completion_tokens": got, "valid": ok}
                flag = "" if ok else "  INVALID (no usage.completion_tokens in every sample)"
                spread = ""
                if len(dec_rates) > 1:
                    spread = (f"  [reps {min(dec_rates):.1f}-{max(dec_rates):.1f}]")
                print(f"[capture] {plabel}/{out}: prefill {med(pre_rates):.0f} tok/s, "
                      f"decode {(med(dec_rates) if ok else 0.0):.1f} tok/s "
                      f"({actual_pt} tok prompt, {got} generated){spread}{flag}", flush=True)
        json.dump(res, open(a.capture, "w"), indent=1)
        return 0

    if not a.compare:
        ap.error("pass --capture FILE (server up) or --compare FILE (GPUs free)")
    ref = json.load(open(a.compare))
    print(f"\n{'prefill':>7} {'out':>5} {'ptok':>6} {'got':>6} | {'exl3 pre':>9} {'helios pre':>11} "
          f"{'pre %':>6} | {'exl3 dec':>9} {'helios dec':>11} {'dec %':>6}")
    for plabel, ptok in PREFILL_TOKENS:
        p = prompt_for_tokens(ptok)
        for out in OUTPUT_TOKENS:
            # Same statistic on both sides: N reps, median. A median-of-1 for helios against a
            # median-of-3 for the reference would not be a comparison, it would be a comparison of
            # our luck against theirs.
            runs = [engine_bench(a.engine, a.model, p, out) for _ in range(a.reps)]
            med = lambda xs: sorted(xs)[len(xs) // 2]
            gen = med([r[3] for r in runs])
            pre, dec, actual_pt = med([r[0] for r in runs]), med([r[1] for r in runs]), runs[0][2]
            dmin, dmax = min(r[1] for r in runs), max(r[1] for r in runs)
            r = ref[plabel][str(out)]
            # `got` is how many tokens were ACTUALLY generated, on OUR side and on the reference's.
            # When either engine stops early the decode figure is measured over a handful of tokens
            # rather than a sustained generation, and the two are not even measured over the same
            # amount of work. The cell is reported but marked INVALID instead of silently compared:
            # a 1-token "33 tok/s" is not a decode rate, and the reference hit this too - its 16k/512
            # cell generated 1 token and scored 381 tok/s.
            ref_got = r.get("completion_tokens", out)
            bad = (gen < out * 0.5) or (ref_got < out * 0.5) or not r.get("valid", True)
            why = []
            if gen < out * 0.5:
                why.append(f"helios got {gen}")
            if ref_got < out * 0.5:
                why.append(f"exl3 got {ref_got}")
            if not r.get("valid", True):
                why.append("exl3 no usage")
            print(f"{plabel:>7} {out:>5} {actual_pt:>6} {gen:>6}/{ref_got:<6} | {r['prefill']:>9.0f} "
                  f"{pre:>11.0f} {100*pre/r['prefill']:>5.0f}% | {r['decode']:>9.1f} {dec:>11.1f} "
                  f"{100*dec/r['decode']:>5.0f}%  [{dmin:.0f}-{dmax:.0f}]"
                  + (("  INVALID (" + ", ".join(why) + ")") if bad else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
