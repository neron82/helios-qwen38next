#!/usr/bin/env python3
"""Byte-for-byte check of render_chat_qwen against the Qwen3.8 checkpoint's chat_template.jinja.

The template file in the model directory is authoritative: what the model saw in training is
whatever that jinja renders, so any divergence changes the prompt. A wrong-family prompt does not
crash - the engine emits plausible-looking but degenerate text, and rep_num loops - so the bytes are
compared directly rather than judged by output quality.

Keep the case list in sync with src/core/test_chat_qwen.cpp.

    ~/shared-venv-gpu/bin/python test/chat_parity_qwen.py
"""
import json
import os
import subprocess
import sys

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-exl3")
BIN = "./build/helios_chat_qwen_test"

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("transformers not importable - run with ~/shared-venv-gpu/bin/python")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]

C1 = [{"role": "user", "content": "Weather in Paris?"}]
C5 = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    },
    {"role": "tool", "content": "18C and sunny", "tool_call_id": "call_1"},
]
C6 = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello", "reasoning_content": "step 1"},
    {"role": "user", "content": "bye"},
]
C7 = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}]

# name -> (messages, tools, kwargs)
CASES = {
    "QWEN1": (C1, None, {"reasoning_effort": "xhigh"}),
    "QWEN2": (C1, None, {"reasoning_effort": "low"}),
    "QWEN3": (C1, None, {"reasoning_effort": "medium"}),
    "QWEN4": (C1, TOOLS, {"reasoning_effort": "xhigh"}),
    "QWEN5": (C5, TOOLS, {"reasoning_effort": "xhigh"}),
    "QWEN6": (C6, None, {"reasoning_effort": "xhigh"}),
    "QWEN7": (C7, None, {"reasoning_effort": "xhigh"}),
    "QWEN8": (C1, None, {"reasoning_effort": "xhigh", "enable_thinking": False}),
}


def main():
    want = {}
    for name, (msgs, tools, kw) in CASES.items():
        want[name] = tok.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=True, **kw
        )

    out = subprocess.run([BIN], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"{BIN} failed: {out.stderr.strip()}")
    got = {}
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        name, _, payload = line.partition(" ")
        got[name] = json.loads(payload)

    bad = 0
    for name in CASES:
        w, g = want[name], got.get(name)
        if w == g:
            print(f"{name} PASS ({len(w)} bytes)")
            continue
        bad += 1
        print(f"{name} FAIL")
        if g is None:
            print("  engine produced no case")
            continue
        print(f"  want {w!r}")
        print(f"  got  {g!r}")
        for i, (a, b) in enumerate(zip(w, g)):
            if a != b:
                print(f"  first difference at byte {i}: want {a!r} got {b!r}")
                print(f"    want ...{w[max(0, i - 40):i + 40]!r}")
                print(f"    got  ...{g[max(0, i - 40):i + 40]!r}")
                break
        else:
            print(f"  lengths differ: want {len(w)} got {len(g)}")
    print(f"\nCHAT PARITY QWEN: {'ALL PASS' if not bad else f'{bad} FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
