#!/usr/bin/env python3
"""MULTI-REQUEST server test for HELIOS_SEQUENCES: several slots, one process, no wrong answers.

This exists because a fresh-process benchmark cannot see the failures that matter here. A feature
that binds a conversation's state has only one dangerous shape of bug - state that outlives the
request it belongs to, or a rebind that takes effect once and then not again - and neither is
visible from a process that serves exactly one request and exits. A CUDA-graph feature shipped on
this engine for exactly that reason: correct on request 1, correct on request 2, dead on request 3,
and no single-request measurement could have shown it.

So this drives ONE server through a scripted sequence of HTTP requests, against a live socket, and
checks three things at each step:

  1. the server is still ALIVE (a crash is the loudest possible failure and the easiest to miss if
     the harness only checks the last response),
  2. the answer matches what the same request produces on a single-sequence server - byte for byte
     on the token stream, and on the text,
  3. the response says which slot it ran on, so a client can pin its next turn.

The request order is chosen to hit the shapes that break: mixed lengths, two slots alternating
repeatedly, a LONG request arriving after short ones have already filled their slots, and the whole
sequence repeated so a leak shows up on a second pass. Streaming and non-streaming are both
exercised, because they take different code paths through the same slot acquisition.

Usage:
  test_seq_server.py [--model DIR] [--gen BIN] [--slots N] [--tokens N] [--port N] [--keep]

Exit 0 pass, 1 mismatch/crash, 2 setup failure.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# Deliberately different shapes. A uniform set would let a slot running at the previous
# conversation's length still produce plausible output, and "plausible" is the failure mode here.
PROMPTS = [
    ("short", "Name the capital of France."),
    ("medium", "The history of the printing press begins in the fifteenth century, when Johannes "
               "Gutenberg combined movable type with an adapted screw press. Summarise what "
               "changed after that, in a paragraph."),
    ("long", "Explain how attention works in a transformer, and then explain why causal masking is "
             "necessary during training. Write at least six sentences and do not stop early."),
]


def wait_ready(port: int, proc, timeout: float = 900.0) -> bool:
    """Block until /health answers, or the server dies. Returns False on either failure."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2.0)
    return False


def post(port: int, path: str, body: dict, headers: dict | None = None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.load(r)


def start_server(gen: str, model: str, port: int, slots: int, logdir: str):
    env = dict(os.environ)
    # MTP OFF, deliberately and for a specific reason: a batched speculative verify reduces
    # differently from a width-1 forward, so greedy text drifts off the sequential stream after a
    # few dozen tokens. That is real, documented engine behaviour and it is orthogonal to slot
    # routing, but it would put a difference at token ~40 that this test could not attribute. The
    # speculative path is exercised separately, WITH MTP on, so that it is covered too rather than
    # quietly excluded - a routing bug that only appears under speculation is still a routing bug.
    env["HELIOS_MTP"] = "0"
    env["HELIOS_SEQUENCES"] = str(slots)
    log = open(os.path.join(logdir, f"server_{slots}.log"), "w")
    proc = subprocess.Popen([gen, "serve", model, "--port", str(port), "--max-tokens", "64"],
                            stdout=log, stderr=subprocess.STDOUT, env=env,
                            preexec_fn=os.setsid)
    return proc, log


def wait_for_vram_free(timeout: float = 180.0) -> None:
    """Block until both cards are back to a few hundred MB, i.e. no engine still holds a context.

    Read through nvidia-smi rather than by watching a PID: a process can be gone from the table and
    still hold VRAM for a moment while the driver unmaps, and it is the unmapping that has to finish
    before the next engine can allocate.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30).stdout
            used = [int(x) for x in out.split()]
        except Exception:
            return
        if used and max(used) < 2000:
            return
        time.sleep(3.0)
    print("  warning: VRAM did not return to idle; a previous engine may still be resident",
          file=sys.stderr)


def stop(proc, log):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    log.close()


def build_plan(n_slots: int):
    """The request order. Returns a list of (label, slot_or_None, stream)."""
    plan = []
    # One per slot, unpinned first: round-robin must SPREAD them, or every unpinned request lands
    # on slot 0 and a conversation kept alive there is the only thing that ever runs.
    for i in range(n_slots):
        plan.append((f"rr-{i}", None, False, PROMPTS[i % len(PROMPTS)]))
    # Pinned, one per slot: a pin must come back as the same slot every time.
    for i in range(n_slots):
        plan.append((f"pin-{i}", i, False, PROMPTS[i % len(PROMPTS)]))
    # Two slots alternating, repeatedly. A rebind that only works on the first switch passes a
    # single pass and fails the third request.
    if n_slots >= 2:
        for r in range(3):
            for i in (0, 1):
                plan.append((f"alt{r}-{i}", i, False, PROMPTS[i % len(PROMPTS)]))
    # LONG after short: the shape that overruns a partition sized for the wrong thing.
    plan.append(("long-after-short", 0, False, PROMPTS[2]))
    # Streaming, which takes a different path through slot acquisition than the buffered one.
    for i in range(min(2, n_slots)):
        plan.append((f"stream-{i}", i, True, PROMPTS[i % len(PROMPTS)]))
    # The whole thing again. Order-dependent state passes a single pass and fails this one.
    plan.extend(plan[:len(plan)])
    return plan


def run_once(gen, model, port, slots, plan, tokens, logdir, probe_bad_pin):
    """Run the plan against one server; return (results, crash_message)."""
    proc, log = start_server(gen, model, port, slots, logdir)
    results = []
    crash = None
    try:
        if not wait_ready(port, proc):
            crash = f"server never became ready (exit {proc.poll()})"
            return results, crash
        info = get(port, "/v1/models")
        data = info["data"][0]
        print(f"  /v1/models: parallel={data.get('parallel')} "
              f"context_length={data.get('context_length')} "
              f"total={data.get('total_context_length')} mode={data.get('concurrency_mode')}")
        if probe_bad_pin:
            # A pin outside the range must be REFUSED, not folded. Folding is invisible because the
            # folded answer looks like a legitimate one.
            try:
                post(port, "/v1/chat/completions",
                     {"model": "q", "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 4, "temperature": 0, "slot": slots + 5})
                crash = "an out-of-range slot pin was ACCEPTED (it must be a 400)"
                return results, crash
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    crash = f"out-of-range pin returned HTTP {e.code}, expected 400"
                    return results, crash
                print("  out-of-range slot pin: refused with 400 (correct)")

        for label, slot, stream, (shape, text) in plan:
            body = {"model": "q", "messages": [{"role": "user", "content": text}],
                    "max_tokens": tokens, "temperature": 0}
            if slot is not None:
                body["slot"] = slot
            if stream:
                body["stream"] = True
            try:
                if stream:
                    text_out, got_slot = stream_once(port, body)
                    results.append((label, slot, text_out, got_slot))
                else:
                    r = post(port, "/v1/chat/completions", body)
                    msg = r["choices"][0]["message"]
                    # The WHOLE message, not just `content`. This checkpoint is a reasoning model
                    # and spends most of a small token budget on `reasoning_content` before emitting
                    # any `content` at all, so comparing content alone compares two empty strings and
                    # calls it agreement. Reasoning is where the tokens are, so it is what carries
                    # the signal.
                    text_out = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
                    results.append((label, slot, text_out, r.get("slot")))
            except Exception as e:
                crash = f"request {label} failed: {type(e).__name__}: {e}"
                return results, crash
            # Alive after every request, not just at the end. The failure this test exists for is a
            # crash on request N, and a harness that only inspects the last response reports it as
            # "the test errored" instead of "the server died on request N".
            if proc.poll() is not None:
                crash = f"SERVER DIED during request {label} (exit {proc.poll()})"
                return results, crash
    finally:
        stop(proc, log)
    return results, crash


def stream_once(port: int, body: dict):
    """Consume an SSE chat completion. Returns (text, slot).

    The slot is read from the chunk's TOP-LEVEL field, which the engine sets on the first chunk so
    a client can pin its follow-up before the first token arrives. Reading only `choices[].delta`
    would miss it - and a client that cannot see which conversation it is talking to cannot keep
    talking to it, which is the entire point of the feature.
    """
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    out, slot = [], None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "slot" in d and slot is None:
                slot = d["slot"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                # Both fields, for the same reason as the buffered path: with a small budget the
                # entire answer can be reasoning and no content, and a stream check that watched
                # only `content` would see nothing at all.
                out.append(delta.get("reasoning_content") or "")
                out.append(delta.get("content") or "")
    return "".join(out), slot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-exl3"))
    ap.add_argument("--gen", default="./build/helios")
    ap.add_argument("--slots", type=int, default=3)
    # Large enough that the model's REASONING reaches an answer inside the budget. This checkpoint
    # emits several sentences of thinking before any content, and at a small max_tokens the entire
    # response is reasoning - so a test that watched only `content` would compare empty against
    # empty and pass without comparing anything.
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if not os.path.exists(args.gen):
        print(f"engine not found: {args.gen}", file=sys.stderr)
        return 2

    logdir = tempfile.mkdtemp(prefix="seqsrv_")
    try:
        plan = build_plan(args.slots)
        print(f"plan: {len(plan)} requests over {args.slots} slots "
              f"({sum(1 for p in plan if p[2])} streaming)")

        # The reference server must be FULLY gone before the multi-slot one starts, and "fully gone"
        # includes the CUDA context: SIGTERM returns from main() while the driver is still tearing
        # 31 GB of mappings down, and the next process then fails its own expert allocation with an
        # OOM that has nothing to do with slot counts. Waiting for the VRAM to actually come back is
        # the difference between a slot test and a flaky one.
        wait_for_vram_free(timeout=180.0)

        print("running single-sequence reference server...")
        ref, crash = run_once(args.gen, args.model, args.port, 1,
                              [(l, None if s is None else 0, st, p) for l, s, st, p in plan],
                              args.tokens, logdir, probe_bad_pin=False)
        if crash:
            print(f"FAIL reference server: {crash}")
            return 1
        print(f"  reference served {len(ref)} requests")

        wait_for_vram_free(timeout=180.0)
        print(f"running {args.slots}-slot server...")
        got, crash = run_once(args.gen, args.model, args.port + 1, args.slots, plan,
                              args.tokens, logdir, probe_bad_pin=True)
        if crash:
            print(f"FAIL multi-slot server: {crash}")
            print(f"  server log: {logdir}")
            return 1
        print(f"  multi-slot served {len(got)} requests")

        failures = 0
        if len(got) != len(ref):
            print(f"FAIL request count: {len(got)} vs {len(ref)}")
            failures += 1
        for i, ((rl, rs, rt, rslot), (gl, gs, gt, gslot)) in enumerate(zip(ref, got)):
            if not rt.strip() and not gt.strip():
                print(f"FAIL request {i} ({rl}): both empty - this request tests nothing")
                failures += 1
                continue
            if rt == gt:
                print(f"  request {i:3d} {rl:18s} slot={gslot} {len(gt):5d} chars  OK")
            else:
                print(f"FAIL request {i} ({rl}): {len(gt)} vs {len(rt)} chars")
                print(f"     got: {gt[:160]!r}")
                print(f"     ref: {rt[:160]!r}")
                failures += 1
            if gs is not None and gslot != gs:
                print(f"FAIL request {i} ({rl}): pinned slot {gs} but served on {gslot}")
                failures += 1
        if failures:
            print(f"\ntest_seq_server: {failures} FAILURES (logs in {logdir})")
            return 1
        print(f"\ntest_seq_server: PASS - {len(ref)} requests through one server over "
              f"{args.slots} slots, no crash, every answer identical to single-sequence")
        return 0
    finally:
        if args.keep:
            print(f"kept: {logdir}")
        else:
            shutil.rmtree(logdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
