#!/usr/bin/env python3
"""Interleaved sequence slots must answer EXACTLY as each conversation would alone.

The property under test: for every slot s, the tokens slot s produces are byte-identical to the
tokens the same conversation produces on a single-sequence engine. Not "close", not "similar
wording" - the same token ids. That is the only claim strong enough to catch the failure this
feature is exposed to.

Why it has to be a multi-request test in ONE process. The state a slot carries is the KV rows it
wrote and the recurrent state that followed from them, and both live in the process. A fresh
process per request throws that away, so a per-request harness cannot see a slot reading another
slot's rows, a rebind that did not take, or a capture restored onto the wrong conversation - it
passes on an engine that answers every second turn from the wrong thread. This is the same shape of
miss that let a CUDA-graph feature ship a crash a fresh-process benchmark could not see.

What it runs, in one process, in this order:
  * three conversations, three slots, interleaved, with DIFFERENT prompt lengths
  * a long request after short ones (the shape that overruns a short slot's partition)
  * two slots alternating, repeatedly (a rebind that only works once would pass the first pass)
  * the same requests again, in a different order, to catch order-dependent state
and compares every slot's token stream against a single-sequence (HELIOS_SEQUENCES=1) run of the
same prompts in the same order.

Usage:
  test_seq_interleave.py [--model DIR] [--gen BIN] [--slots N] [--tokens N] [--keep]

Exit 0 pass, 1 mismatch, 2 setup failure.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Each conversation is deliberately a different length. A uniform set would let a slot that
# silently ran at the previous conversation's length still produce plausible output.
PROMPTS = {
    "A": "Name the capital of France.",
    "B": ("The history of the printing press begins in the fifteenth century, when Johannes "
          "Gutenberg combined movable type with an adapted screw press. "
          "Summarise what changed after that, in a paragraph."),
    # The trailing sentence is not decoration. A bare instruction makes this model end its turn on
    # the first token, and a request that emits nothing tests nothing - worse, "both runs emitted
    # nothing" is indistinguishable from "both runs agreed", so it passes silently. Every prompt
    # here is asked for a specific length so the comparison has something to compare.
    "C": ("Explain how attention works in a transformer, and then explain why causal masking is "
          "necessary during training. Write at least six sentences and do not stop early. " * 2),
}


def write_prompts(d: str) -> dict:
    out = {}
    for k, v in PROMPTS.items():
        path = os.path.join(d, f"prompt_{k}.txt")
        with open(path, "w") as f:
            f.write(v)
        out[k] = path
    return out


def run(gen: str, model: str, plan: str, env_extra: dict, tokens: int) -> str:
    """Run a plan file through one engine process; return the concatenated token-id stream."""
    env = dict(os.environ)
    # MTP speculates: a batched verify reduces differently from a width-1 forward, so greedy text
    # drifts off the sequential stream after a few dozen tokens. That is a real, documented property
    # of the engine and it is orthogonal to slot routing, but it would drown the signal this test is
    # looking for - a difference at token 40 could be either. Off, so any difference is a routing
    # bug. The speculative path is exercised separately, with MTP on, by test_seq_server.py.
    env["HELIOS_MTP"] = "0"
    env["HELIOS_IDS"] = "1"
    env.update(env_extra)
    # stderr MERGED INTO stdout, and this is load-bearing rather than tidiness. The engine writes
    # its "[gen] request N" markers to stdout and the "[ids]" token stream to stderr, so capturing
    # them separately and concatenating afterwards reorders the whole transcript - every request
    # marker ends up ahead of every token - and the per-request split below then silently returns
    # one request instead of ten. Merging at the source is the only way to keep the pairing the
    # engine actually emitted.
    proc = subprocess.run([gen, "gen", model, "--raw", "--slots-file", plan],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=7200, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"engine exited {proc.returncode}\n{proc.stdout[-3000:]}")
    return proc.stdout


def tokens_by_request(stream: str) -> list:
    """Split the [ids] stream into per-request token lists.

    Marker-driven, not line-driven, and both details are load-bearing:

      * The engine prints generated text with no trailing newline, so a request marker routinely
        lands MID-LINE ("Paris[gen] request 2 slot=1 ..."). Splitting on lines that START with the
        marker misses exactly those boundaries, which silently merges two requests into one and
        makes the comparison meaningless - it looked like a pass because both runs merged the same
        way.
      * An EMPTY request is still a request. Appending only non-empty lists shifts every later
        index by one, so a mismatch would be reported against the wrong conversation.

    One regex over the whole stream, in order, so the split is exactly the order the engine emitted.
    """
    reqs, cur, started = [], [], False
    for m in re.finditer(r"\[gen\] request|\[ids\]\s+(\d+)", stream):
        if m.group(0).startswith("[gen]"):
            # Only a marker that FOLLOWS a previous one starts a new request. The engine prints its
            # load and placement diagnostics before the first request, and treating that preamble as
            # a request inserts a phantom empty entry at index 0 - which shifts every later
            # comparison by one, so request i is compared against request i-1. That is how a run with
            # 17 requests reports "request 0 produced no tokens" when request 0 is the preamble.
            if started:
                reqs.append(cur)
            started = True
            cur = []
        else:
            cur.append(int(m.group(1)))
    if started:
        reqs.append(cur)
    return reqs


def digest(ids) -> str:
    return hashlib.md5(",".join(str(i) for i in ids).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-exl3"))
    ap.add_argument("--gen", default="./build/helios")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--keep", action="store_true", help="keep the temporary plan directory")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if not os.path.exists(args.gen):
        print(f"engine not found: {args.gen}", file=sys.stderr)
        return 2

    tmp = tempfile.mkdtemp(prefix="seqint_")
    try:
        paths = write_prompts(tmp)
        n = args.slots
        keys = list(PROMPTS)[:n]
        if len(keys) < n:
            # More slots than distinct prompts: reuse the short one, which is the interesting case
            # for a long-after-short overflow, and give it its own key anyway.
            keys = (list(PROMPTS) * ((n // len(PROMPTS)) + 1))[:n]

        # The plan: interleaved, mixed lengths, long after short, two slots alternating, and the
        # whole thing again in reverse order. The second pass in a different order is not redundant
        # - state that depends on arrival order passes a single pass and fails this one.
        plan_lines = []
        order = []
        for i, k in enumerate(keys):
            plan_lines.append(f"{i} {paths[k]} {args.tokens}")
            order.append((i, k))
        # Long-after-short: the longest conversation runs after the shortest ones have filled their
        # slots, which is where a slot whose partition is the wrong size would run off the end.
        if len(keys) >= 2:
            longest = max(range(n), key=lambda i: len(PROMPTS[keys[i]]))
            plan_lines.append(f"{longest} {paths[keys[longest]]} {args.tokens}")
            order.append((longest, keys[longest]))
        # Two slots alternating repeatedly. A rebind that takes effect only on the first switch
        # passes a single pass and fails the third.
        if n >= 2:
            for _ in range(2):
                for i in (0, 1):
                    plan_lines.append(f"{i} {paths[keys[i]]} {args.tokens}")
                    order.append((i, keys[i]))
        # Reverse order, so no conversation is always the first or the last one to run.
        for i, k in reversed(order):
            plan_lines.append(f"{i} {paths[k]} {args.tokens}")

        plan = os.path.join(tmp, "plan.txt")
        with open(plan, "w") as f:
            f.write("\n".join(plan_lines) + "\n")

        # 1) Reference: the SAME requests, in the SAME order, on a single-sequence engine. Every
        #    request here is slot 0, so this is exactly "each conversation served alone" - the
        #    property has to hold against this and nothing else.
        ref_plan = os.path.join(tmp, "plan_ref.txt")
        with open(ref_plan, "w") as f:
            f.write("\n".join("0 " + l.split(" ", 1)[1] for l in plan_lines) + "\n")
        print("running reference (1 slot)...")
        ref = tokens_by_request(run(args.gen, args.model, ref_plan, {"HELIOS_SEQUENCES": "1"},
                                    args.tokens))
        print(f"  reference produced {len(ref)} requests")

        # 2) The same plan across N interleaved slots.
        print(f"running interleaved ({n} slots)...")
        got = tokens_by_request(run(args.gen, args.model, plan,
                                    {"HELIOS_SEQUENCES": str(n)}, args.tokens))
        print(f"  interleaved produced {len(got)} requests")

        failures = 0
        if len(got) != len(ref):
            print(f"FAIL request count: interleaved {len(got)} vs reference {len(ref)}")
            failures += 1
        # Every request must have produced tokens, in BOTH runs. An empty one is not a pass: it is
        # either a stop token firing immediately (which makes the request test nothing) or a real
        # failure, and "both runs were empty" is indistinguishable from "both runs agreed".
        for i, (a, b) in enumerate(zip(got, ref)):
            if not a or not b:
                print(f"FAIL request {i}: produced no tokens "
                      f"(interleaved {len(a)}, reference {len(b)}) - this request tests nothing")
                failures += 1
                continue
            if a == b:
                print(f"  request {i:3d}  {len(a):4d} tok  {digest(a)[:12]}  OK")
            else:
                print(f"FAIL request {i}: {len(a)} vs {len(b)} tokens, "
                      f"digest {digest(a)[:12]} vs {digest(b)[:12]}")
                first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print(f"     first divergence at token {first}: {a[first:first+4]} vs {b[first:first+4]}")
                failures += 1

        if failures:
            print(f"\ntest_seq_interleave: {failures} FAILURES")
            return 1
        print(f"\ntest_seq_interleave: PASS - {len(ref)} interleaved requests over {n} slots, "
              f"every one byte-identical to serving that conversation alone")
        return 0
    finally:
        if args.keep:
            print(f"kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
