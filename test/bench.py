#!/usr/bin/env python3
"""Benchmark helios against the exllamav3 baseline on the same prompts.

Both engines need both GPUs, so the baseline numbers are captured in one pass (server up) and the
engine in another (GPUs free). Measures prefill tok/s and greedy decode tok/s at several prompt
lengths, mirroring how the baseline was characterised earlier in the project.

  capture: python3 test/bench.py --capture /tmp/bench_ref.json --port 8080
  measure: python3 test/bench.py --compare /tmp/bench_ref.json --engine ./build/helios
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

# (label, words) - prompts are built by repeating a varied word list so tokenisation is realistic
# target APPROXIMATE token counts (the paragraph below runs ~1.18 tokens per word)
LENGTHS = [("short", 400), ("medium", 4000), ("long", 16000)]
WORDS = ("The capital of France is Paris and the river that runs through it is the Seine, "
         "which has supported trade, art and daily life for more than two thousand years. ")
_PW = len(WORDS.split())


def build_prompt(words):
    copies = max(1, words // _PW)
    return (WORDS * copies).strip()


# The prompt above is a repeated paragraph, and a model asked to continue it very often decides it
# is finished: measured on this build, a 7,426-token version generated 1 token before stopping, and
# 3,697- and 14,884-token versions generated 41. A decode rate measured over 1 token is not a
# throughput number, so every decode cell of the grid was quietly meaningless. Appending an explicit
# instruction makes the model keep going, and grid_bench now also prints the ACTUAL generated count
# and marks a cell INVALID when it falls short, so this cannot silently come back.
_CONTINUE = ("\n\nContinue the discussion above at length. Do not repeat previous text; write the next "
             "several paragraphs of original commentary on the same subject, and keep writing.")


def build_prompt_for_bench(words):
    return build_prompt(words) + _CONTINUE


def server_bench(port, prompt, ntokens):
    body = json.dumps({"model": "q", "prompt": prompt, "max_tokens": ntokens,
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return time.time() - t0, d.get("usage", {})


def engine_bench(binary, model, prompt, ntokens):
    env = dict(os.environ)
    # the prompt is passed through a file: these are long enough to exceed a command line comfortably
    with open("/tmp/helios_bench_prompt.txt", "w") as f:
        f.write(prompt)
    out = subprocess.run([binary, "gen", model, "--raw", "--prompt-file", "/tmp/helios_bench_prompt.txt",
                          "--tokens", str(ntokens), "--temp", "0"],
                         capture_output=True, text=True, env=env, timeout=3600)
    m = re.search(r"prefill ([0-9.]+) tok/s decode ([0-9.]+) tok/s", out.stdout + out.stderr)
    pt = re.search(r"prompt tokens=(\d+)", out.stdout + out.stderr)
    # How many tokens were ACTUALLY generated. A generation that stops early (this model's stop token
    # fires almost immediately on some prompts) still prints a "decode X tok/s" figure, and that
    # figure is then measured over a handful of tokens instead of the requested count - it is not a
    # throughput measurement. The caller needs the count to detect that. The server path already
    # records usage.completion_tokens; this is the engine-side equivalent.
    gn = re.search(r"\[gen\] (\d+) tokens in", out.stdout + out.stderr)
    gen = int(gn.group(1)) if gn else 0
    if not m:
        return (0.0, 0.0, 0, gen)
    return (float(m.group(1)), float(m.group(2)), int(pt.group(1)) if pt else 0, gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./build/helios")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-exl3"))
    ap.add_argument("--port", type=int, default=8080)
    # 320, not 128: a short run never reaches steady-state clocks. On this box a 40-token
    # decode measured 41.2 tok/s and a 320-token one 45.8, so shorter runs understate both
    # engines and the gap depends on which length each side used.
    ap.add_argument("--tokens", type=int, default=320)
    ap.add_argument("--capture")
    ap.add_argument("--compare")
    a = ap.parse_args()

    if a.capture:
        res = {}
        for label, words in LENGTHS:
            p = build_prompt(words)
            # Prefill and decode are timed as two DIFFERENT requests, because a single request's
            # wall time contains both and gen/dt then reports a prefill-contaminated decode.
            #
            # Every request gets a unique prefix. This matters enormously and is easy to get wrong:
            # repeating one prompt gives 33,553 then 35,064 tok/s (the server's prefix cache serving
            # from memory), while a unique nonce gives 2,009 tok/s for the same 17.6k tokens. A
            # harness that reuses the prompt reports a prefill rate ~17x the real one.
            #
            # Decode = N / (t_N - t_1), using the same unique prompt for both so the prefill is the
            # identical work; the subtraction removes exactly the prefill, including its fixed overhead.
            pre_rates, dec_rates = [], []
            for i in range(3):
                # DIFFERENT nonces for the two requests. With the same one, the second is served from
                # the prefix cache the first populated, tn - t1 goes to ~0, and the decode rate divides
                # by float noise (it reported 3.2e11 tok/s). Different nonces make both requests do the
                # same cold prefill work, so the subtraction is meaningful.
                t1, u1 = server_bench(a.port, f"nonce A{i} {p}", 1)
                tn, un = server_bench(a.port, f"nonce B{i} {p}", a.tokens)
                pt = u1.get("prompt_tokens", 0)
                pre_rates.append(pt / max(t1, 1e-9))
                dec_rates.append(un.get("completion_tokens", a.tokens) /
                                 max(tn - t1, 1e-9))
            reps = [(max(pre_rates), max(dec_rates))]
            res[label] = {"prefill": reps[0][0], "decode": reps[0][1],
                          "prompt_tokens": pt}
            print(f"[capture] {label}: prefill {res[label]['prefill']:.0f} tok/s, "
                  f"decode {res[label]['decode']:.1f} tok/s ({res[label]['prompt_tokens']} tok prompt)")
        json.dump(res, open(a.capture, "w"), indent=1)
        return 0

    if not a.compare:
        ap.error("pass --capture FILE (server up) or --compare FILE (GPUs free)")
    ref = json.load(open(a.compare))
    print(f"{'workload':<8} {'prompt':>8} {'baseline pre':>13} {'helios pre':>11} {'baseline dec':>13} "
          f"{'helios dec':>11}")
    for label, words in LENGTHS:
        p = build_prompt(words)
        pre, dec, pt = engine_bench(a.engine, a.model, p, a.tokens)
        r = ref[label]
        print(f"{label:<8} {pt:>8} {r['prefill']:>13.0f} {pre:>11.0f} {r['decode']:>13.1f} {dec:>11.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
