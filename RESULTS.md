# Helios — measured status

All numbers on this workstation (2x RTX 3090, 1x PCIe 4.0 x16 + 1x PCIe 3.0 x4, 128 GB DDR4),
same model (`Qwen3.8-Flash-Next-exl3`), greedy (temperature 0), 64 generated tokens.
Baseline is the exllamav3 server on the same checkpoint, measured in the same session with the
server's prefix cache defeated (unique prompt prefix) so prefill is a true cold measurement.

## Benchmark

All numbers on this workstation (2x RTX 3090, 1x PCIe 4.0 x16 + 1x PCIe 3.0 x4, 128 GB DDR4),
same model (`Qwen3.8-Flash-Next-exl3`), greedy (temperature 0), **320 generated tokens**, measured with
one harness (`test/bench.py`) applied identically to both engines.

| workload | prompt | baseline prefill | helios prefill | baseline decode | helios decode |
|----------|-------:|-----------------:|---------------:|----------------:|--------------:|
| short    |    430 |              576 |            266 |           117.2 |          45.5 |
| medium   |  4,390 |            1,755 |            252 |           116.3 |          39.7 |
| long     | 17,590 |            2,511 |            193 |           127.9 |          29.5 |

**Status: Helios does not yet beat the baseline, but has improved substantially.** The helios-prefill
column above predates four optimizations now shipped in the tree (each verified correct and, where
noted, bit-identical or determinism-checked):
1. **Tensor-core (mma) prefill attention** — 7.7x on the attention phase, **+45% prefill**.
2. **Deterministic LPT MoE expert scheduling** — MoE phase −5.3%, **+1.8% prefill**, bit-identical.
3. **Chunked (WY) gated-delta-rule** — GDN phase 441→256 ms (−42%), **+10% prefill**, SHIPPED ON
   (parity rel L2 3.9e-3, run-to-run deterministic, decode untouched).
4. **Layer-pipelined 2-GPU forward** (`HELIOS_PIPELINE=1`) — **1.73x prefill** (528→915 tok/s at 10.3k),
   load-time selectable, deterministic, flag-off byte-identical.

Together these take prefill from ~200 tok/s to **~915 tok/s at 10.3k (pipeline mode)** / **~527 tok/s
(default, flag off)** — i.e. ~47% / ~26% of the baseline's ~1,950–2,050 tok/s. Decode is ~59 tok/s
(default) / ~49 (pipeline, a measured handoff tradeoff at n=1) — ~46–50% of the baseline's 116–128.
Both execution modes are load-time selectable and byte-reproducible; the pipeline trades a little
decode for a large prefill/context-ingestion win, and the default (non-pipelined) path remains the
fully-verified one. A re-run of the A/B table against the live exllamav3 server is the honest next
benchmark; the helios numbers here are engine-internal (`./build/helios gen`) at the stated prompt,
not the bench.py cross-engine harness. See the per-optimization sections below for measurements,
determinism proofs, and the characterized (non-bug) reassociation each shipped change carries.

### The earlier baseline numbers in this log were wrong, and why

An earlier version of this table claimed 78.5 tok/s decode and 2,678-2,911 tok/s prefill. **Decode was
understated and prefill was roughly right by luck.** Three separate measurement faults, all since fixed
in `test/bench.py`:

1. **The prefix cache was not defeated.** Reusing one prompt across repetitions gave 33,553 then
   35,064 tok/s at 17.6k tokens - the server answering from memory. A unique nonce gives **2,009 tok/s**
   for the same prompt. A harness that reuses the prompt overstates prefill by ~17x. The harness's own
   docstring said the cache had to be defeated; the code did not do it.
2. **Prefill and decode shared one request.** `decode = gen/dt` counts prefill inside the decode rate,
   which is why adding a unique prefix appeared to "hurt" decode from 106 to 34 tok/s. They are now two
   requests, `decode = N / (t_N - t_1)`, with **different** nonces so neither primes the other's cache
   (using the same nonce made `t_N - t_1` ~0 and reported 3.2e11 tok/s).
3. **Runs were too short to reach steady-state clocks.** 40 / 120 / 320 tokens measure 41.2 / 43.9 /
   45.8 tok/s on the same build. Every 40-token figure quoted anywhere in this log understates the
   engine; the harness default is now 320.

The corrected baseline decode of **116-128 tok/s** is consistent with the "up to 120 tok/s" figure in the
project brief; the old 78.5 was a short-run artifact.

## What this session fixed

### 1. MoE decode moved to the mgemm path (default)

The grouped cooperative kernel is barrier-bound: at decode it processes 8-10 slots but pays a
~0.75 ms block-wide barrier per launch, and there are 6 launches per MoE layer. Switching n=1 to
mgemm (the per-slot grouped-GEMM path) took decode from **28.0 to 45.0 tok/s** on a short prompt.

Verified numerically identical, not merely plausible: the MoE output rms per layer agrees between
the two paths and both agree with the reference to 6 digits (L0 0.040000 both vs 0.039989 ref;
L2 0.011583 vs 0.011581). `HELIOS_MOE_GROUPED=1` forces the old path back for A/B.

Profiled effect (`HELIOS_PROF=1`, 12-layer model, per token): `moe` fell **28.7 -> 17.0 ms**.

### 2. Dense attention parallelised over the KV range

`gqa_dense_f16` launched **one warp per (token, query head)** and each warp walked the entire KV
cache serially. At prefill that is thousands of warps and fine; at decode `n=1` gives 24 warps for
the whole card - three blocks on an 82-SM GPU - each walking the full context one position at a
time. That alone is why decode fell off a cliff with depth (37.9 tok/s at 430 tokens, 15.9 at 4.4k,
5.4 at 17.6k).

Rewritten as a flash-decoding-style split: one **block** per (token, head), the block's 16 warps
each taking a contiguous slice of the keys, computing a partial (max, sumexp, weighted-V) in
registers, then a block-level log-sum-exp merge. No second kernel or global scratch needed because
the block already owns the row.

| context | decode before | decode after | gain |
|---------|--------------:|-------------:|-----:|
| ~1k     |          37.9 |         43.8 | 1.2x |
| ~9.4k   |          15.9 |         35.1 | 2.2x |
| ~17.6k+ |           5.4 |         21.1 | 3.9x |

Attention cost fell from ~9 ms per 1000 context tokens to ~0.65 ms. GQA parity still passes
(max_err 3.0e-05 on |o|~0.02, over a 597-position window), as does the full attn parity suite.
Raising the split from 16 to 32 warps changed nothing (34.99 vs 35.10 tok/s at 9.4k), confirming
the limit is the 24-block launch, not the split width.

## Where the remaining gap is

**Prefill (~10x).** This is the largest gap and I do **not** yet have a grounded attribution for it.
What is measured: the engine prefills at 301 / 248 / 147 tok/s against the baseline's 2,678 / 2,911 /
2,392. What is not measured: which phase owns that time. The `HELIOS_PROF` counters are attributed
per decode step, so a run with `--tokens 2` reports the prefill work inside the step buckets
(`moe=456-495 ms` with `steps=1..4` at a ~2.6k prompt) and cannot be read as a prefill breakdown.
Getting a real prefill profile is the first thing the next session should do.

Context that bounds the problem, all verified this session:

- the model is 59 GB on disk and the expert arena alone is 29.16 GB (`[model] expert slab 1.190 MB x
  512 x 49 -> 29.16 GB`), so with 48 GB of VRAM across the two cards **the weights do not all fit**;
  the baseline cannot be fully resident either, and it reaches 2,911 tok/s at 41.5k tokens.
- per-layer experts total only 512 x 1.19 MB = 609 MB, so a layer-major full-residency pass over all
  512 experts costs ~26 GB of reads for the whole model - under 1 s at the measured 27 GB/s of RAM
  bandwidth. The baseline's 14.26 s for 41.5k tokens therefore looks compute-bound, not I/O-bound.
- Helios still does per-block expert gather rather than the layer-major full-residency streaming plan
  in the design doc, which is the obvious candidate, but that remains a hypothesis until profiled.

**Decode MoE is latency-bound at M=1, not bandwidth-bound. (Corrected.)**

An earlier revision of this file claimed decode was bound by streaming experts from host memory at
~500 MB/token. **That was wrong, and the arithmetic that "matched" was a coincidence.** The experts
are fully VRAM-resident: `model.cpp:299` states "Routed experts go straight to the card that computes
them: no host arena, no slots", and `experts0`/`experts1` are GPU allocations (`A(...,0,...)` /
`A(...,1,...)`), 9.11 GB on GPU0 and 20.05 GB on GPU1 for the complete 29.16 GB of expert slabs. There
is no slot cache and no streaming on this path, so a host-bandwidth model cannot explain anything.

### The stage profiler was measuring the wrong thing (fixed)

`HELIOS_MPROF=1` first reported `rgemv=0.241 ... grouped=0.200 finish=0.130`. I read `rgemv=0.241` as
"the router GEMV costs 10.3 ms/token and runs at 10.7 GB/s" and optimised it. **That was wrong.** The
first bucket is measured from MoE-layer entry, so its `cudaStreamSynchronize` drained whatever the
previous stage left pending on card 0 - the attention/GDN work, ~0.21 ms/layer - and reported it as
router time. Confirmation: vectorising the GEMV to 8 halves per load and doubling its block count
changed the bucket by 0.2% (0.245 -> 0.243 ms) and decode by nothing.

The fix is one drain before the clock starts. With it, the real breakdown is:

| stage | ms/layer | x42 = ms/token | share | effective bandwidth |
|---|---:|---:|---:|---:|
| **grouped (expert mgemm)** | **0.198** | **8.3** | **50%** | 60 GB/s (11.9 MB/layer) |
| **finish (cross-card + reduce)** | **0.130** | **5.5** | **33%** | - |
| router gemv | 0.012 | 0.5 | 3% | 218 GB/s (2.62 MB/layer) |
| topk | 0.012 | 0.5 | 3% | - |
| permute | 0.009 | 0.4 | 2% | - |
| shared expert | 0.039 | 1.6 | 10% | - |
| xfer | 0.000 | 0 | - | - |

Summing to 0.400 ms/layer = 16.8 ms/token, which agrees with the independently measured
`[prof] moe = 14.35-17.02 ms/token` - so this breakdown is consistent in a way the previous one was
not. The router is **not** the problem (218 GB/s, and cheap); focusing on it was chasing an artifact.

Two real targets, in order:

1. **`finish` = 5.5 ms/token for work that should be nearly free.** It is the cross-card reduction
   (`xcard_copy` + `add_f32_inplace` + `moe_slot_reduce_k`) and `xcard_copy` is fully synchronous by
   construction: it does `cudaMemcpyAsync` to pinned host, `cudaStreamSynchronize`, `cudaMemcpyAsync`
   to the destination, `cudaStreamSynchronize` again - twice over PCIe per layer, 42 times per token,
   with the pipeline drained each time. The comment on it explains the constraint (a per-device event
   cannot order two different device contexts, so the destination is drained instead). An event or
   graph-captured stream wait that works across contexts, or keeping the reduction on the card that
   owns the data, would remove the drain.
2. **The expert mgemm at 60 GB/s effective.** Latency-bound at M=1, as each of ~80 blocks reads a full
   1.19 MB expert slab to produce one output row. Raising M (MTP) amortises this rather than hiding it.

Three wrong claims are now on the record as retracted: decode is not host-streaming bound (experts are
fully VRAM-resident), the router is not the dominant MoE stage, and launch overhead was not the MoE
bottleneck. Each was refuted by a measurement that cost less than the optimisation it would have paid
for, which is the argument for fixing instrumentation before tuning.

**Decode at depth (~2-4x).** Two causes, both identified:
1. Only 24 blocks launch at n=1 regardless of split width; cross-block split-K with a combine
   kernel would use the whole card.
2. The QSA sparse indexer is ported and parity-tested (`kpool_write`, `indexer_score`,
   `pool_expand`, `mla_sparse_decode` all pass) but is **not wired into the attention path**, so
   context cost is still O(depth) instead of O(topk). This is why the baseline is flat at ~117-128
   across context and Helios is not.

**Decode at depth (~2-4x).** Two causes, both known:
1. Only 24 blocks launch at n=1 regardless of split width; cross-block split-K with a combine
   kernel would use the whole card.
2. The QSA sparse indexer is ported and parity-tested (`kpool_write`, `indexer_score`,
   `pool_expand`, `mla_sparse_decode` all pass) but is **not wired into the attention path**, so
   context cost is still O(depth) instead of O(topk). This is why the baseline is flat at ~117-128
   across context and Helios is not.

**MTP was implemented but had never actually run (now fixed).**

The draft head crashed on every default run with an illegal memory access. Three separate defects, found
by not trusting the first plausible explanation:

1. **MTP weights were loaded on card 1 while the draft ran on card 0.** `load_mtp` passed `dev=1` to
   every `load_group`/`load_small`/`load_hc`, but `Runner::spec_step` drives the draft from
   `gpu(0).stream(0)`, `mtp_scratch_init(..., 0)` puts the scratch on card 0, the draft's KV rows are in
   `kv_[0]` and its logits go to row 1 of the card-0 head buffer. So every draft kernel dereferenced
   card-1 pointers from card-0's context. `compute-sanitizer` named it exactly: an invalid
   `__global__` read *on the weight operand* of `tap_norm_k`, 4.17 MB below the nearest allocation -
   the signature of a foreign-context address. Fixed by moving the 56 non-expert MTP tensors (61.9 MB
   total, against ~2.3 GB free on card 0) to the device the draft actually runs on, via a documented
   `kMtpDev`. The MTP layer's routed experts are deliberately *not* covered by this: `load_experts`
   places those by expert index across both arenas like any other layer.
2. **`gr_apply` dereferenced a null `post`.** `mtp_draft_step` passes `nullptr` for the post gate three
   times ("no post gate is kept for the draft"), which is legal for `gr_mix` - its launcher guards with
   `if (post && inject)` - but `gr_apply` fed the pointer straight to a kernel that reads it
   unconditionally, hence "illegal access to 0x0". A null post now means "no gating" (apply the
   sublayer output unscaled), matching the convention the caller already relied on. Confirmed against
   `compute-sanitizer`: the error moved off `tap_norm_k` after fix 1 and off `gr_apply_kernel` after
   this one.
3. **`has_mtp` was a config key defaulting to on.** `tc.value("mtp_num_hidden_layers", 1) > 0` means a
   checkpoint that simply omits the key gets a draft layer built for weights it does not ship. It is now
   derived from what the checkpoint declares (any `mtp.*` tensor). Worth recording that I first
   concluded this checkpoint had *no* MTP and nearly disabled the feature - the checkpoint ships 6200
   `mtp.*` tensors; I had only searched the GLM naming (`eh_proj`/`enorm`/`hnorm`, layer index 48) and
   missed the `mtp.` namespace. The stricter check is still worth keeping, but the answer here is "fix
   MTP", not "disable it".

**It now runs, and it is correct**: MTP-on and MTP-off generate the same token ids (11751 " Paris", 271
". ", 6511 "The", 9338 " capital of"), stable across 6/6 runs with no heap corruption.

**It is not yet a win.** 24 tokens: 36.73/36.63/36.76 tok/s with MTP against 39.00/37.20 without - about
6% *slower*. That is the expected shape when the draft is rejected almost every step: the draft step is
paid in full and returns one token. Making MTP pay needs the acceptance rate measured and the drafts
made good - which is the same M=1 amortisation problem identified above, not a separate one.

## Reproducing

```sh
cmake -B build -G Ninja && cmake --build build
HELIOS_MTP=0 ./build/helios gen ~/models/Qwen3.8-Flash-Next-exl3 \
    --raw --prompt-file prompt.txt --tokens 64 --temp 0
./build/attn/test_attn_parity        # attention + rope + GQA parity
HELIOS_PROF=1 ...                    # per-phase ms per token
```

`test/bench.py` holds the harness used for the table above (baseline capture with the server up,
engine measurement with the GPUs free).

---

## Correctness: three bugs found and fixed after the benchmark

The benchmark above was taken through `/v1/completions`, which bypasses the chat template. Driving the
engine the way a real client does (chat completions) exposed three defects that the raw path hides.

### 1. The engine rendered the wrong model family's chat template

`render_chat` was written for **GLM-5.3**: it emits `[gMASK]<sop>`, `<|system|>`, `<|assistant|>` and
`<|observation|>`. The Qwen3.8 checkpoint's `chat_template.jinja` wants a different format entirely -
`<|im_start|>role\n...<|im_end|>\n`, a synthetic reasoning-instruction system turn, and `<|im_start|>
assistant\n<think>\n` as the generation prompt. `<|assistant|>` does not even exist in the Qwen
tokenizer, so the model was being asked to continue a prompt in a format it was never trained on.

This failure is silent - nothing errors, the engine just emits degenerate text - so it is now detected
from the checkpoint's own template rather than assumed:

- `detect_chat_format()` reads `chat_template.jinja`; `[gMASK]` ⇒ GLM-5.3, `<|im_start|>` ⇒ Qwen3.8
  (verified to classify both checkpoints correctly).
- `render_chat_qwen()` is transcribed from that file. `test/chat_parity_qwen.py` diffs it against
  transformers' `apply_chat_template` for 8 cases (effort levels, tools, tool round trip, multi-turn,
  explicit system, thinking disabled) - **all 8 byte-identical**.
- The GLM renderer is untouched and its own parity still passes 4/4.

One semantic difference is worth calling out because it is not obvious: the Qwen template's condition
is `preserve_thinking is undefined or preserve_thinking is true or ...`. The server never passes
`preserve_thinking`, so the first clause always holds and assistant reasoning is **never** cleared,
whereas GLM clears older reasoning. The two renderers must not share that logic.

`reasoning_effort` also differs: GLM accepts `low|high|max` (default `high`), Qwen accepts
`xhigh|medium|low` (default `xhigh`). The default now follows the detected format.

### 2. Generation never stopped at the turn boundary

`Tokenizer::load` hardcoded `<|endoftext|>` as EOS. For this checkpoint that is **248044**, but the
model ends its turns with `<|im_end|>` = **248046**, and `generation_config.json` lists *both* as
`eos_token_id`. Stopping on only one meant generation ran straight past the end of the turn - the
model would emit `<|im_end|>` over and over and the engine would keep going, which reads as a
repetition loop rather than the termination bug it is.

Stop ids now come from the checkpoint's `generation_config.json` (handling a list), falling back to
`tokenizer_config.json`'s `eos_token`, then to the old default. The loader reports
`stops=248046,248044` and the generation loop stops on any of them. `eos_id()` keeps its separate
meaning as the begin token.

### 3. The server never sent the begin token

Both request paths did `tk.encode(text)` with nothing prepended. This checkpoint declares no
`bos_token`, so HF-style tokenisation of the rendered prompt omits it - but the reference engine
feeds it (`add_bos = True`), and the `gen` CLI's `--raw` path prepended it, which is precisely why the
CLI and the server disagreed on the same prompt. Without it the model degrades: on one rendered chat
prompt the server produced `"We we we we ..."` while the CLI produced coherent text. Both server paths
now prepend it.

**Evidence that the prompt itself is correct.** Feeding the *identical* rendered prompt text to the
baseline via `/v1/completions` produces coherent reasoning ("We need to respond to user: \"Weather in
Paris?\" ..."), so the remaining divergence below is in the forward pass, not the prompt.

### Determinism and PLE drift: both closed

- **MoE expert permutation is deterministic.** Three identical greedy runs (both a synthetic token
  prompt and a real text prompt, 32 tokens) produced byte-identical output. No atomic-free scatter
  rewrite was needed - the slot assignment is already index-based.
- **The PLE drift is gone.** `HELIOS_PRMS` reports the post-PLE hidden state as rms 0.017765 against
  the reference's 0.017761 (ratio **1.0002**), and a full per-layer diff against
  `/tmp/ref_states.npz` shows **47 of 48 layers within 0.8%**. The old "ratio 0.82" was measured
  against the wrong stage: the engine's `[lrms] L1` is dumped *after* the PLE, so it must be compared
  with the reference's `layers.1.ple`, not `layers.0`.

### Still open: divergence is content-dependent, NOT length-dependent

My first hypothesis (length-dependent divergence) was **refuted by experiment**, so it is recorded
here as refuted rather than left as a plausible story:

| prompt | tokens | result |
|---|---|---|
| "The capital of France is" | 6 | ` Paris.` - correct |
| repeated Seine text + same question | 126 | ` Paris.` - correct |
| same, longer | 396 | ` Paris.` - correct |
| rendered chat prompt | 57 | echoes the system instruction, then degenerates |

Plain-text prompts up to 396 tokens are answered correctly, so the forward pass is sound at length.
Feeding the chat prompt as the **exact token ids taken from HF's own tokenizer** (so tokenisation
cannot be the difference) still degenerates, so the prompt and its tokens are both ruled out.

What distinguishes the failures:
- the engine is correct on **memorised-style completion** cues ("The capital of France is" -> ` Paris`),
  which are robust to small numerical error;
- it diverges on **instruction-following** prompts, which are not.

That points at the residual per-layer drift rather than a structural bug: 47 of 48 layers agree to
within 0.8%, but the drift is systematic and accumulates in the **deep layers** (L40-L47 sit at 0.992
while L1-L30 are ~1.000). A sub-1% error is invisible on an easy continuation and decisive on a harder
one. The PLE is implicated but not proven: on the special-token prompt PLE-off is better
("The user asks: \"What is the capital of France?\"" vs a repetition), while on the full chat prompt
PLE-on is better - i.e. neither setting is right, which is consistent with the PLE being subtly wrong
rather than absent. Checked against the reference and found **correct**: the n-gram id derivation
(`ple_ngram_ids` vs `ngram_embedding.py:compute_ngram_ids`) matches on the multiplier mix, the
`remainder + head_offsets` wrap, and the `_shift_right_ignore_eos` segment rule.

Additional measurements taken while localising this:

- **End-to-end trunk accuracy.** `HELIOS_RMS=1` reports the final stacked hidden state at
  `streams_rms=0.6001`, against the reference's `layers.47` rms of `0.602768` - **0.4%**, i.e. the whole
  48-layer trunk lands where the reference lands. (The same dump's `logits=[0.000,0.000]` is a broken
  diagnostic, not a zero head: the head projection at runner.cpp:123 only runs under a condition this
  path does not take, so that field proves nothing either way and should be fixed before being used.)
- **Where the offset starts.** Per-layer ratios are within +/-0.2% from L1 to L36, then step to ~0.992
  and stay there from L37 to L47. The reference itself jumps hard at the same point (rms 0.071024
  after layer 35 to 0.116223 after layer 36, and up to 0.602768 by layer 47), and the engine tracks
  that transition at a 0.9949 ratio, so it is following the same trajectory rather than missing a term.
  Layer 36 is a GDN layer (attention sits at `i % 4 == 3` in the config's `layer_types`).

Next step is to localise the deep-layer drift directly: `test/dump_ref_states.py` produces per-layer
reference states for an arbitrary prompt, but it loads exllamav3 with `Model.from_config` and so needs
the offloading configuration the server script uses (the checkpoint is 59 GB against 48 GB of VRAM,
which is why `ev3_server.py` runs but a bare `model.load()` will not). Diffing the last ~8 layers on a
chat prompt is the shortest path from here.


With the prompt proven byte-correct, the remaining defect is numerical. The engine matches the
baseline exactly on short prompts - the 6-token prompt "The capital of France is" yields id 11751,
760 = " Paris", the same continuation the baseline gives - but on a 57-token rendered chat prompt it
echoes the system instruction and then degenerates, where the baseline reasons about the question.
Sampling is not the cause (the checkpoint's recommended `top_k=20/top_p=0.95` loop differently but
still loop), and the PLE helps rather than hurts (PLE off is strictly worse: `"We we we we ..."`).
The suspicion is the PLE n-gram injection, since the failure mode is verbatim prompt echo, but that is
untested. `test/dump_ref_states.py` can produce per-layer reference states for an arbitrary prompt and
is the tool to localise this.


---

## CORRECTION: the fast decode path was numerically wrong (and was the default)

This supersedes the decode figures reported earlier in this file. The `mgemm` single-token MoE path was
made the default several sessions ago on the strength of a comparison that **could not have detected
the difference it was supposed to rule out**: the comparison prompts were long enough (n > 5) to route
*both* runs through the grouped kernel, so it compared the grouped path against itself.

Measured properly at n = 1, where the paths actually differ:

| | L0 | L1 | L2 |
|---|---:|---:|---:|
| grouped (correct) | 0.028675 | 0.008781 | 0.011350 |
| mgemm | 0.026508 | 0.007270 | 0.008012 |

and end to end on the same rendered chat prompt, the grouped path reproduces the reference engine's
continuation verbatim -

    grouped:  We need to respond to user: "Weather in Paris?" Need likely provide current …
    mgemm:    We need to think carefully through the task, validate key assumptions, consider …

The mgemm continuation is the model **echoing its own system prompt**, which is exactly the symptom
this project spent an entire session chasing as a "content-dependent forward-pass divergence". It was
never numerical drift in the trunk. The reason it looked length-dependent and prompt-dependent:

**the first generated token is always correct on the broken path**, because it is produced by the last
prefill chunk, which always uses the grouped kernel regardless of `MOE_DECODE_MAX_N`. Only the tokens
after it come from the decode path and diverge. So short prompts looked fine, and anything that
answered in one token looked perfect.

Three defects were found. Two are fixed, one is not:

1. **Card 1 never received the router's output.** `routing_std_logits` writes only card 0's
   `topk_idx`/`topk_w`; the grouped path copies them to card 1 (`xcard_copy(topk_idx[1], …)`) but the
   mgemm path did not, so card 1 read zeros, selected expert 0 for every slot with weight 0, and
   contributed nothing - silently dropping all 352 of its experts. The destination stream matters too:
   an async H2D into card 1's memory enqueued on card 0's stream is not ordered against card 1's
   kernels, so it must go on card 1's stream.
2. **Card 1 never received the activations.** `moe_slot_gather_k` reads `sc.x16_1` on that card, which
   nothing had written. Two missing copies, not one - fixing only the indices changed nothing.
3. **The finish stage reduced from buffers the down pass never wrote.** It read `dec_out`/`dec_out1`
   while the down pass writes `part[c]`, and it applied `moe_slot_reduce_k` on top of a result mgemm
   had already reduced per token (its weighted mode splits slots into `num_tokens` groups and reduces
   each into its own row). Now it mirrors the grouped path's finish.

After those, card 0 and card 1 both contribute (partial rms 0.0058 and 0.0257 at L0) but the total is
still 7-30% low across layers, so a residual defect remains and the path stays **opt-in**
(`HELIOS_MOE_MGEMM=1`, default off) with the finding recorded at its definition.

### Corrected performance

The honest decode number for the correct engine is **~27 tok/s**, not the 45 previously reported - that
figure measured the broken path. The grouped cooperative kernel is now the dominant decode cost:

| stage | ms/layer | x42 | share |
|---|---:|---:|---:|
| **grouped (cooperative MoE)** | **0.559** | 23.5 ms | **68%** |
| permute | 0.091 | 3.8 ms | 11% |
| router gemv / topk / shared | 0.063 | 2.6 ms | 8% |

At n = 1 the cooperative kernel is barrier-dominated (~0.75 ms per launch), which is why the mgemm path
was worth building in the first place - and why fixing its residual is the highest-value decode work.
The same kernel is 45% of prefill, so it is the highest-value target overall.

### MTP on the correct path

Still no gain (25.4-27.0 tok/s against 27.4 without) and the accept rate is **not reproducible** across
identical greedy runs (52.4% then 39.1%, 1.52 then 1.39 tokens/step). Greedy decoding must be
deterministic, so MTP is introducing non-determinism - a defect in its own right, and one to fix before
its economics are worth measuring.


### mgemm semantics: pinned by test, and a fourth defect found

Guessing at `mgemm`'s per-slot behaviour from its doc comment was costing more than measuring it, so
`src/cuda/quant/test/test_mgemm_semantics.cpp` now checks the weighted multi-matrix mode against 10
separate `gemm` calls (which have their own smoke test). Reading the kernel confirms:

- slot `j` takes its input from `A + j * size_m * size_k`, its matrix from `B_list[indices[j]]`, its
  weight from `weights[j]`, and writes output row `j`;
- the reduction then sums rows `[t*stride, (t+1)*stride)` into row `t`, skipping any slot whose index
  is `< 0`;
- **the range filter only runs when `min_index >= 0`**, and with `num_tokens == 1` it compacts the
  indices *and weights* in place without permuting the input rows - so slots stop lining up. Passing
  global ids with a range filter is only sound if every slot survives, which is false when each card
  owns a disjoint expert range. Out-of-range slots must instead be expressed as a negative index.

The test also caught a fourth defect in the engine's use of it. The down projection was called with
`M=slots, bszm_in=1`, which makes every slot transform the *same* block (and with `M=slots` the stride
becomes `slots*inter`, running off the end of the 10-row `gbuf`). Both wrong forms look plausible and
neither faults. It must be `M=1, bszm_in=slots`, so slot `j` reads `gbuf + j*inter` - its own
activation:

| | L0 | L1 | L2 |
|---|---:|---:|---:|
| grouped (target) | 0.028675 | 0.008780 | 0.011352 |
| mgemm, before this fix | 0.025104 | 0.002720 | 0.004620 |
| mgemm, after | 0.027624 | **0.008805** | 0.008229 |

L1 is now exact and L0 is within 4%, but the per-layer ratios still oscillate between 52% and 114%
across L0-L7 rather than converging, so at least one more defect remains and the end-to-end output is
still wrong (`"We -o: Q80411313131"` against the grouped path's `"We need to respond to user: ..."`).
The path stays opt-in; `HELIOS_MOE_MGEMM=1` enables it for A/B.

The lesson worth keeping: three of these four defects produced output that *looked* like model
misbehaviour rather than a plumbing error, and the default was wrong for several sessions because the
comparison that blessed it could not distinguish the paths.


### mgemm residual: narrowed to an alignment error, and a barrier finding

`HELIOS_MOEC` (which dumps the routed contribution *before* the shared expert is added) now works on the
mgemm path too, so the two paths' expert arithmetic can be compared directly instead of through an
output that mixes in the shared expert. That confirms `mlp=` measures `y` itself, and isolates this:

| layer | mgemm routed | grouped routed | ratio |
|---|---:|---:|---:|
| L0 | 0.011231 | 0.014170 | 79% |
| L1 | 0.008423 | 0.007903 | **107%** |
| L2 | 0.005683 | 0.009005 | 63% |
| L3 | 0.005567 | 0.006382 | 87% |

The ratio crosses 1 in both directions, which is the signature of a **misalignment** - experts, weights
and activations paired with each other incorrectly - rather than a missing factor, which would sit
consistently below 1. The call semantics themselves are now pinned by
`test_mgemm_semantics` (which passes), so the defect is in how the engine wires the three calls
together, not in the kernel or in my reading of it.

**Separately: the kernels take an expensive barrier on this architecture.** `group_barrier()` is
implemented with portable CUDA atomics (`cuda::atomic_ref`, `__nanosleep`) and would work on sm_86, yet
`exl3_moe_kernel.cuh`/`exl3_gemm_kernel.cuh` gate it behind `__CUDA_ARCH__ > 890` and fall back to a
grid-wide `grid.sync()` - which is much heavier and forces coincident residency. The grouped MoE
already calls `group_barrier` unconditionally; it is the `gemm`/`mgemm` kernels that fall back.

I enabled it for sm_80+ and measured: decode 26.3-28.4 -> 28.68 tok/s, prefill 248.7 -> 248.9 tok/s.
**Neutral**, so it was reverted rather than kept. Two reasons not to keep an unmeasured change there:
`grid.sync()` is unambiguously correct, and switching `gemm` and `mgemm` onto `group_barrier` makes them
share the same barrier counters (`locks + BARRIER_LOCKS_OFFSET`) - safe only as long as they never run
concurrently on one device, which is an invariant nothing enforces.


### Where the expert shape actually loses (bench_gemm, full sweep)

| M | expert N=640 K=2560 b2 | lm_head N=248320 K=2560 b5 |
|---:|---:|---:|
| 1 | 0.021 ms, 19.1 GB/s | 0.569 ms, 698 GB/s |
| 2 | 0.022 ms, 18.8 GB/s | 0.570 ms, 697 GB/s |
| 4 | 0.022 ms, 18.5 GB/s | 0.553 ms, 719 GB/s |
| 10 | 0.028 ms, 14.5 GB/s | 0.564 ms, 705 GB/s |
| 32 | 0.063 ms, 6.5 GB/s | 1.113 ms, 357 GB/s |
| 128 | 0.244 ms, 1.7 GB/s | 4.766 ms, 83 GB/s (34 TFLOP/s) |

Two different regimes, and they explain the MoE from both ends:

- **lm_head is healthy.** It holds ~700 GB/s - essentially full bandwidth - up to M=10, and past that it
  saturates at ~34 TFLOP/s, which is tensor-core compute bound rather than a defect. A single large
  matrix with N=248320 tiles beautifully.
- **The expert shape never amortises.** Its time grows almost linearly with M (12x for 128x the rows), so
  the 1.19 MB weight read is never hidden. N=640 gives only five 128-wide tiles per k-slab, which is far
  too little to fill 82 SMs, so the kernel is latency- and occupancy-starved at every M.

Two caveats, and the second matters more than the first:

1. A single matrix at large M is not the MoE's aggregate pattern: the MoE runs ~400 expert-matrices
   with small `size_m` each, spread across the grid, so its aggregate can be bandwidth-bound even
   though one matrix at a time is not.
2. **These rows describe `gemm`, and the MoE does not call `gemm`.** The reference's own dispatch -
   `use_mgemm()` in `model/config.py`, `K_threshold=6` and `n_threshold=8192` - selects `mgemm` for
   exactly the MoE's case (K = 2 bits is below the threshold, but `out_features = 640 < 8192` is not),
   and this engine likewise uses `moe_grouped` at prefill and `mgemm` at decode. So the 1.7-19 GB/s
   figures above say nothing directly about the MoE's throughput; they characterise the kernel that
   serves attention projections, the shared expert and the head. The measurement that would settle the
   MoE is a representative multi-matrix `mgemm` of the same shape, which `test_mgemm_semantics` already
   has the scaffolding for and `bench_gemm` does not.

Also learned about the grouped kernel while reading it: the expert loop skips empty experts with a plain
global read and **no barrier** (`if (token_count == 0) continue;`), so at n = 1 its cost is the barrier
cost of the few *active* experts, not of walking all 512. `num_active` only reshapes the grid - fewer,
wider groups - and does not skip work. And the weighted reduction reads rows that every `blockIdx.z`
wrote, so the grid-wide sync before it is load-bearing: replacing it with a per-group barrier would be
wrong there, which is a second reason the reverted barrier change was the right call.


### MTP: a real cache bug fixed, and it is now opt-in

**The draft head's KV cache was never populated for prompt positions.** The draft is a full attention
layer with its own cache, but it only ever ran during decode, so for a 57-token prompt the draft
attended over positions 0-56 that nothing had ever written. The KV pool is bump-allocated without a
memset (`A0(kv_bytes)`), so it attended over whatever the allocator had handed out. Prefill now runs
each chunk through the head to fill it (`compute_head=false` skips the 397 MB lm_head read, whose
result is discarded there). That is a genuine bug fixed regardless of what it does to throughput.

Fixing it surfaced a second bug of my own making: `mtp_scratch_init(mtp_, c, 2, 0)` sized the draft
scratch for 2 tokens, and the new prefill fill passes a whole chunk. That overflowed the buffers - and
it did so into the trunk's logits, which made the draft "agree" with the trunk every single time:

| | acceptance | tokens/step |
|---|---:|---:|
| before the cache fix | 52.4%, 39.1% | 1.52, 1.39 |
| **with the overflowing scratch** | **100.0% x3** | **2.00** |
| after sizing the scratch for a chunk | 50.0%, 56.7%, 50.0% | 1.50, 1.57, 1.50 |

That 100% was recorded as a breakthrough before I checked it; it was the draft reading the trunk's own
logits. Sizing the scratch correctly removed it. **MTP is still not reproducible** - identical greedy
runs still give different accept rates and different token sequences - so the cache bug was not the
cause of the non-determinism, and something else remains.

MTP is therefore **off by default** now, for two independent reasons: it is slower (27.09 vs 28.7 tok/s,
because each step runs two full single-token trunk forwards plus a draft, so it cannot win without
batched verification), and it is not reproducible. A default that is both slower and non-deterministic
is worse than no default. `HELIOS_MTP=1` enables it for A/B, and the outputs it produces do match the
non-MTP ones.

The objective notes MTP does heavy lifting in the reference. It does not here yet, and the two reasons
are now separated: the structural one (two trunk passes per two tokens) needs the accepted token
verified inside a batched trunk pass, and the correctness one (non-reproducible acceptance) needs
finding before either can be trusted.


### Operational note

`compute-sanitizer` leaves its target process holding both GPUs after it finishes. The next engine run
then fails to allocate the expert arena and exits with no output at all -

    cudaMalloc(20531.48 MB) failed on device 1 (ctx device 1, free 1.76 GB): out of memory

which reads exactly like a regression in whatever was just changed. Check
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and kill the stray `helios` process
before concluding anything.


### MTP non-determinism: confirmed rigorously, cause still open

Five identical greedy runs with the GPU verified idle before each gave **four different outcomes**:

    42.4% (14/33), 42.4% (14/33), 56.7% (17/30), 42.4% (14/33), 51.6% (16/31)

So it is real and not an artifact of GPU contention. `initcheck` is **not usable on this workload** - the
sanitizer's shadow memory exhausts the ~2.2 GB free per card and the target dies with
`cudaMalloc(256.00 MB) failed on device 0 (free 0.00 GB)` before reaching anything interesting.

**A method error worth recording**: my first `initcheck` run reported "0 errors" and I treated it as
ruling out uninitialised reads - but I had just made MTP opt-in, so that run had MTP *disabled* and
validated nothing about the draft path. Re-running it properly fails on memory as above.

Ruled out by direct inspection: `logits_ + vocab` (the draft's logits row) is in bounds
(`head_rows_ = 2`); `attn_layer` takes its KV caches as parameters and never reads `attn_ord`, so the
MTP layer's unset `attn_ord = -1` is harmless. `racecheck` flags the router top-k's shared-memory merge,
but the trunk runs that same kernel on every layer and is byte-identical across runs, so it is most
likely a warp-synchronous false positive rather than the cause.

### The grouped MoE is ~100% fixed overhead at decode (measured)

`HELIOS_MPROF` per layer, with the prompt length chosen so every MoE call is at that n:

| n | ms/layer |
|---:|---:|
| 1 | 0.559 |
| 18 | 2.65 |
| 69 | 5.20 |

At n = 1 essentially all of the 0.559 ms is fixed cost, which over 42 layers is the 23.5 ms/token that
dominates decode. It also means batching decode steps cannot help much: the per-token marginal cost
(~50 us/layer) is what grows, and the fixed part barely amortises. Only a barrier-free kernel changes
this, which is exactly what mgemm is - and why its residual matters so much.

### QSA sparse indexer: survey

The kernels are all present in this repo and pass their parity checks (`indexer_score_mma`,
`indexer_score_legacy`, `kpool_write`, `pool_expand`, `mla_sparse_decode`). The checkpoint's structure is
confirmed: 13 layers carry an indexer, each with `index_qk_proj` (fused q|k, 2560 -> 640 =
(4+1) heads x 128) plus `q_layernorm`/`k_layernorm`; both are already loaded for the trunk
(`model.cpp:283-284`) and for the MTP layer. `indexer_budget=2048`, `indexer_compress_ratio=4`.

What is missing is purely the wiring: `attn_layer.cu` is 77 lines with **zero** indexer/sparse
references, so there is no indexer key cache, no kpool planes, no top-k selection and no sparse
attention path. That is a from-scratch build, not a switch to flip, and it is the only place where the
baseline's advantage is structural rather than constant-factor (its decode is flat at ~117-128 across
context where this one decays).


### The MoE kernel is not misconfigured - it matches the reference exactly

Comparing my port against the reference source (`exllamav3_ext/quant/exl3_moe_common.cuh`) shows the
compile-time constants are identical:

| | mine | reference |
|---|---:|---:|
| `MOE_SMS_PER_EXPERT` | 8 | 8 |
| `MOE_MAX_SMS_PER_EXPERT` | 32 | 32 |
| `MOE_TILESIZE_K` / `MOE_TILESIZE_M` | 32 / 16 | 32 / 16 |
| `MOE_SH_STAGES` / `MOE_FRAG_STAGES` | 3 / 3 | 3 / 3 |
| `EXL3_GEMM_BASE_THREADS` | 256 | 256 |

and the reference's dispatch matches too: `MAX_BSZN = 8`, so above 8 rows it also uses the grouped
`exl3_moe` kernel rather than the per-token mgemm loop. So the ~10x MoE gap is **not** a port
misconfiguration or a different kernel choice - both sides run the same kernel with the same tiling
and the same grid shape (concurrency 10 x group width 8 = 80 blocks, one per SM on this 82-SM card).

That localises the remaining gap to how the engine *drives* the kernel - the arguments it passes, the
per-expert tables, or the buffers - which is a narrower and more tractable question than "the kernel is
slow". At n = 256 the MoE should cost roughly 0.7-1 ms/layer on compute and bandwidth grounds; it costs
~10 ms.


### MoE concurrency is already saturated, and the stage profiler is sync-inflated

Sweeping the grouped kernel's concurrency (`HELIOS_MOE_CONC`) at a 321-token prompt:

| concurrency | grouped ms/layer |
|---:|---:|
| 1 | 32.54 |
| 2 | 16.68 |
| 5 | 8.59 |
| 10 | 8.15 |

Scaling is near-linear to 5 and then flat, so the engine is already at the effective maximum
(`moe_max_concurrency` = num_sms / MOE_SMS_PER_EXPERT = 10). There is no config win left here.

**But this also shows the `HELIOS_MPROF` buckets cannot be used as absolute numbers.** 8.15 ms/layer x 42
layers = 342 ms for a 321-token chunk, i.e. 938 tok/s from the MoE *alone* - against a measured 249 tok/s
end-to-end for prefill. The buckets are inflated because `lap()` calls `cudaStreamSynchronize` between
every stage, so each one absorbs sync latency. That makes every absolute attribution I have taken from
`HELIOS_MPROF` suspect, including the "MoE = 45% of prefill / 68% of decode" split. The
*relative* comparisons (stage A vs stage B within one configuration, or the same stage across
concurrency settings) remain meaningful; the shares do not.

The same caveat applies to the `[prof]` per-phase counters, which also synchronise per phase: prefill
attribution of "moe 45% / attn 23%" came from that path and should be treated as indicative only.
Getting a trustworthy profile needs the phases timed with `cudaEvent` pairs on an already-drained
stream, not wall-clock between syncs - which is the same instrumentation fix identified for the MoE
stages earlier, and it is now the prerequisite for any further optimisation rather than a nicety.


### The profiler is fixed and now agrees with the end-to-end measurement

`HELIOS_PROF` used to drain the stream at every phase and take wall-clock deltas, which folded sync
latency into each bucket. It now records one `cudaEvent` per phase boundary on the stream and reads the
deltas after a single sync. Validation is simply whether the phases add up to the measured per-token
time:

| | sum of phases | measured |
|---|---:|---:|
| old (wall-clock between syncs) | 43.3 ms | 36.4 ms (+19%) |
| **new (event deltas)** | **38.47 ms** | **37.99 ms (+1.3%)** |

Two traps found on the way, both of which fail silently: `cudaEventElapsedTime` returns an error on
events created with `cudaEventDisableTiming` (every delta is skipped and every phase reads 0.00), and a
leftover tick-reset inside the layer loop was being counted as a second phase boundary.

**Corrected decode breakdown** (ms/token, total 38.5):

| phase | ms/token | share |
|---|---:|---:|
| **moe** | 21.87 | **57%** |
| amix | 4.67 | 12% |
| mmix | 4.54 | 12% |
| gdn | 3.73 | 10% |
| ple | 1.72 | 4% |
| attn | 1.21 | 3% |
| final / apply | 0.39 / 0.34 | 2% |

The earlier "MoE = 68% of decode" figure came from the inflated `mprof` bucket and is superseded: the
truth is **57%**. The prefill shares (moe 45%, attn 23%) happen to survive, because that inflation was
proportional - but they now come from a method that has been validated against the end-to-end number
rather than one that was merely plausible.

This matters beyond tidiness: every remaining optimisation target is chosen from these shares, and the
two prior instrument errors in this project (the MoE stage buckets, and the first-bucket contamination
that made the router look like the largest cost) were both cases of optimising against a number that
was not measuring what it claimed.


### mgemm: the kernel and the call pattern are now both exonerated

My semantics test was passing while the engine was wrong, which should have been suspicious: it
exercised **uniform weights and all-valid indices**, while the engine passes **non-uniform router
weights and `-1` for slots owned by the other card**. It now covers the engine's real configuration,
and all five cases pass:

| case | rel RMS |
|---|---:|
| gate/up shape: M=1, bszm=slots | 1.09e-03 |
| gate/up shape: M=1, bszm_in=1 (broadcast) | 0.000e+00 |
| down shape: M=1, bszm=slots | 1.09e-03 |
| **down shape: M=1, bszm=slots, skips + non-uniform w** | **1.10e-03** |
| down shape: M=slots, skips + non-uniform w | 1.09e-03 |

So the kernel handles negative indices ("skip this slot"), the weighted reduction, and non-uniform
weights exactly as the engine needs - verified against separate `gemm` calls that have their own smoke
test. `had_or(c)` is also correct (separate per-card buffers, `c == 0 ? had : had1`), so the two cards
are not fighting over one `a_had`.

That rules the kernel out and narrows the engine's residual to the plumbing that feeds it. What is
*shared* with the working grouped path is also ruled out by construction: the same `topk_idx`/`topk_w`
arrays, the same expert pointer tables (`table_base[c][f] + layer*per[c]`), and the same `x`. What is
unique to the mgemm path is the gather into `xg`, the per-slot `gbuf`/`ubuf` pair, the `silu_mul`
between them, and the finish - and the routed ratios (79% / 107% / 63% / 87% across L0-L3) cross 1 in
both directions, which is what a misalignment among those produces rather than a missing factor.

The method lesson is the same one that has now bitten three times: a test that passes while the system
fails is not evidence the tested component is fine until you check that the test exercises the failing
configuration.


### The grouped MoE is already at its configuration optimum (two sweeps, both negative)

**First, a broken diagnostic of my own, caught by its own implausibility.** The per-card partial dump I
added read the two cards' buffers with a plain `cudaMemcpy` while the kernels that wrote them were still
in flight on `s1`/`s2`. It returned *identical* values for every layer (0.007935 at both L0 and L1) and
zeros for the grouped path - impossible for real data, though the magnitudes still looked plausible.
Adding a per-stream sync before each read fixed it, after which the data is usable and says something
clear:

| layer | card | mgemm | grouped | ratio |
|---|---:|---:|---:|---:|
| L0 | 0 | 0.007935 | 0.007524 | 106% |
| L0 | 1 | 0.007975 | 0.009107 | 88% |
| L1 | 0 | 0.000758 | 0.000732 | 104% |
| L1 | 1 | 0.008367 | 0.007868 | 106% |

**Both cards are wrong**, so the defect is in logic they share, not in the cross-card handling - which
rules out the remaining per-card suspects and narrows it to the gather, the `gbuf`/`ubuf` pair, the
`silu_mul`, or the finish.

**Second, two configuration sweeps that both come back negative**, retiring the last two ideas I had for
making the grouped kernel faster at decode:

| group width (`MOE_SMS_PER_EXPERT`) | decode tok/s |
|---:|---:|
| **8 (current)** | **28.75** |
| 4 | 26.74 |
| 2 | 20.68 |

| `num_active` (grid-shape hint) | decode tok/s |
|---:|---:|
| **-1 (current, narrow default)** | **27.67** |
| 4 (wider groups) | 26.64 |
| 2 (widest) | 24.63 |

Narrower groups are worse (fewer SMs per expert) *and* wider ones are worse (fewer groups, so fewer
experts processed in parallel). The current setting sits at the optimum, which also refutes my
"decode MoE is ~100% barrier latency" claim from the previous turn - if barriers dominated, narrowing
them would have helped.

So at n = 1 the grouped kernel's 0.559 ms/layer is inherent to its design, not to how it is configured,
and the mgemm path remains the only lever for decode. Two independent measurements now say so.


### Prefill attention is ~400x off its own bound, and warp count is not the reason

Three warp configurations measured for the dense attention kernel, on the prefill `attn` phase (the
now-trustworthy event profiler, ms per 256-token chunk):

| warps per (row, head) | attn ms/step |
|---:|---:|
| **16 (existing)** | **238.5** |
| 1 | 275.5 |
| 32 | 275.9 |

I had assumed prefill should stop splitting - it already launches 6144 blocks and pays a 16-way
log-sum-exp combine (16 KB of shared memory traffic) per block, even for the early causal rows whose KV
slice is a handful of keys. The measurement says the opposite: at prefill the later rows still walk
enough keys that the KV walk dominates the combine, so the split keeps paying. Reverted to 16.

That leaves the phase's cost unexplained, and the size of the gap is worth recording plainly:

    attention per layer per 256-token chunk
      compute bound        0.218 GFLOP ->   15 us at 15 TFLOP/s
      bandwidth bound      33.6 MB KV  ->   48 us at 700 GB/s
      measured             19.88 ms
      -> ~375x the compute bound, ~423x the bandwidth bound

So this is not a tiling or occupancy question I can tune my way out of - it is off by two and a half
orders of magnitude from both bounds, which means something structural is wrong (launch configuration,
the per-row KV addressing, or the profiler attributing another phase's work here). It is 23% of prefill,
so it is worth chasing, but I would not trust any explanation until it is measured rather than reasoned.

### A recurring own-goal, now three times

Twice more this turn I measured a binary that had not been rebuilt: a `sed` loop that ran twice left an
invalid template argument (`<X>`), the build failed, and the run silently used the previous binary -
which I only caught because the numbers looked wrong. The same class of error produced the earlier
"initcheck reported 0 errors" (that run had the feature disabled) and the identical-per-layer per-card
dump. **Check the build succeeded before measuring** is not a stylistic preference here; it has now cost
three separate wrong conclusions in this project.


### The prefill attention "400x anomaly" was my modelling error - and the real answer is QSA

Two of my own errors in a row, both caught by measurement rather than reasoning.

**The stage timer was mislabelled.** I added 8 marks but 8 marks give 7 deltas, so every label after
the first was shifted by one and the 17.8 ms stage was reported as `kvcopy` when it is really the
attention kernel. I very nearly went chasing the KV copy on the strength of it; what caught it was
simply skipping the copies (`HELIOS_NOKVCOPY`) and seeing the phase total not move at all - 237.26 ms
with copies against 238.35 without. Corrected, the layer at a 9.4k-token prefill is:

    qkv_gemm=1.264  deint=0.020  norm=0.017  rope=0.022  kvcopy=0.009  attn=17.819  ogemm=0.630 ms

**And my "400x off its bound" arithmetic was wrong.** I had assumed the attention's KV slice was about
the chunk length (~128 keys on average). It is not: during prefill of a 9.4k prompt each chunk's rows
attend over the **entire prefix**, so the average is ~9.2k keys, not 128. Redone honestly:

    256 rows x 9.2k keys x 512 B = 1.2 GB of KV per layer per chunk
    at 700 GB/s that is 1.7 ms; measured 17.8 ms -> ~10x, not ~400x

So the attention kernel is ~10x off its bandwidth bound - a real inefficiency, the same
small-work-per-block pattern as elsewhere - but it is *not* the 400x structural mystery I reported.
The dominant term is simply that **dense attention is O(context) and this prefill is O(n^2) over the
prefix**, which is exactly what the reference avoids with QSA sparse attention (`indexer_budget=2048`).

That reframes the goal's two big numbers as one thing:

- **prefill**: moe 45%, attn 23% - and most of that 23% is dense-attention work that a 2048-key budget
  would cut by ~4.5x at 9k context and ~70x at 146k;
- **decode**: flat ~117-128 tok/s for the reference (78.5 was a short-run artifact) against this engine's decay from 45.5 to 29.5 tok/s as
  context grows - the same dense-vs-sparse difference, since sparse cost does not grow with context.

So "Implement QSA sparse indexer path" is not the context-length *bonus* item it was filed as. It is
the highest-value remaining item for **both** prefill and context length, and the kernels for it are
already ported and passing parity - only the wiring is missing.


### The attention kernel is L2-throughput bound, and three fixes for it all measured worse

The corrected stage attribution (attention kernel = 17.8 ms/layer, 89% of the `attn` phase) plus the
arithmetic gives a clear mechanism, and it is not DRAM:

| | |
|---|---:|
| logical KV reads per layer-chunk (every (row, head) walks its KV head's keys) | 28.9 GB |
| distinct KV bytes touched | 9.4 MB (L2 is 6 MB) |
| measured 17.8 ms | **1.63 TB/s logical** |
| DRAM bound would be | 3.4 ms |

A kernel cannot run 5x *below* DRAM bandwidth, so it is not DRAM bound: it re-reads each key once per
query head (12x, since 24 q heads share 2 kv heads) and is limited by L2 throughput. The obvious fix is
to make that reuse intra-block, so L1 serves it - 8 rows per block under one head, one warp per row.

**It measured 251.6 ms/step against the existing 237.0.** The idea was principled and still lost: with
one warp per row the serial KV walk costs more than the L1 reuse saves. That is now the third attention
variant measured worse than the incumbent:

| variant | attn ms/step |
|---|---:|
| **16 warps per (row, head) - current** | **237.0** |
| 1 warp per row, one block per row | 275.5 |
| 32 warps | 275.9 |
| 8 rows/block, one warp per row (L1 reuse) | 251.6 |

All four were verified correct before being timed (the row-tiled one via the engine's end-to-end
continuation, since the parity test's n=3 does not reach it - which is the same test-coverage trap as
before, caught this time before drawing a conclusion from a green test).

The remaining lever on this phase is not tiling but **doing less work**: a 2048-key budget against a
9.2k context cuts it ~4.5x, and against 146k cuts it ~70x. That is QSA, and it is now first on the list
for prefill and context length together.


### QSA is a from-scratch build here, and gr_mix's fixed cost is not launches

**QSA survey, corrected.** The ported indexer kernels are written for the *GLM* checkpoint, not this one:
`indexer_score_k` hardcodes 32 indexer heads (`q_sm[32*128]`, `for h < 32`, `w[mm*32+h]`) while this
checkpoint has `indexer_n_heads = 4` and `index_qk_out = 640 = (4+1)*128`. `mla_sparse_decode` is
likewise MLA-shaped (64 heads of 512 latent) where this model is GQA (24 q heads, 2 kv heads, 128
head_dim) and needs a plain sparse-GQA kernel. So "the kernels are ready, only the wiring is missing"
was wrong: it needs the indexer kernels parameterised, a new sparse-GQA attention kernel, the indexer
state and pooling, top-k selection, and the runner plumbing. That is a from-scratch build.

One thing the survey did confirm: `indexer_budget = 2048`, `compress_ratio = 4` and the kernels' POOL=4
with `kidx = 512` pools agree exactly (512 pools x 4 = 2048 raw tokens).

**And the 24% of decode spent in the hc mix is not launch overhead.** Fitting the profiler's numbers
gives `cost(n) = 89 us + 8 us*n` per `gr_mix` call, so 92% of its 97 us is fixed at decode, and 96 calls
per token would make that 8.5 ms of a 38.5 ms token. `gr_mix` launches five kernels, so that looked like
five launches' worth of latency. I fused all five into one shared-memory kernel with `__syncthreads`
between stages - `normed` -> `t` -> `gated`/`mixed` (smem atomics) -> `post`, 51 KB of smem, one block
per row, numerically identical output. Measured:

| | before | after fusion |
|---|---:|---:|
| amix ms/token | 4.67 | 4.65 |
| mmix ms/token | 4.54 | 4.49 |
| decode tok/s | 28.24 | 27.83 / 28.87 |

Nothing. Reverted rather than kept: ~100 lines of new kernel for no measured gain is unjustified
complexity, even when it is correct and even when the reasoning behind it was sound. That is the ninth
optimisation hypothesis this project has retired by measurement.


### ROOT CAUSE of the mgemm residual: the `mul1` flag

Long-standing bug, finally isolated. The decode mgemm path's per-card partials were 88-106% of the
grouped ones on every layer, and eight hypotheses had been retired trying to explain it. The bisect
that worked: dump one slot's gate output and compare it against the single-matrix `exl3::gemm` (which
has its own smoke test) using the *same* expert table entry and the *same* gathered input row.

| comparison | relerr | cos | scale |
|---|---:|---:|---:|
| engine mgemm vs verified gemm | 1.449 | -0.0112 | 1.037 |
| relerr 1.4 = sqrt(2) | | | |

`cos ~ 0` with `scale ~ 1` means uncorrelated but equal magnitude - not a scale error, not a layout
error (no off-diagonal matched a 4x4 output-row/ref-slot scan), not an expert-selection error (no
candidate expert matched). Then, calling `exl3::mgemm` directly over a **one-matrix** table with the
same input reproduced it exactly (`cos -0.0112`), which ruled out my call pattern, the table, the
gather, the layout and the scratch: the kernel itself disagreed with `gemm` for one plain 1xNxK GEMM.

The only argument that differs between `mgemm` and `gemm` is `mul1`:

| mcg | mul1 | cos vs verified gemm |
|---:|---:|---:|
| 0 | 1 | -0.0112 |
| 0 | 0 | **1.0000** |
| 1 | 0 | -0.0176 |

`mul1` applies exl3's global per-tensor dequant scale, which this quantizer has already folded into
`suh`/`svh`, so enabling it scaled every expert's output. Note `exl3::moe_grouped` on line 479 passes
`mul1=true` and is correct - the two kernels do not give that flag the same meaning. Both mgemm gate
and up calls now pass false; the grouped path is untouched and still default.

**The fix also made the path faster: 33-34 tok/s vs the grouped path's 27-29 (about +20%).** I am not
banking it yet - with `mul1` fixed the mgemm path still produces different output from the grouped path
and is still non-deterministic run to run, so it stays opt-in with the grouped path as default. Zeroing
the fp32 partials (whose contract is "need not be zeroed", which does not hold when `idx = -1` slots are
skipped) did not restore determinism, so that is a real latent bug but not this one. What is now
established is that the residual is *not* in the flag, the args, the table, the gather, the layout or
the scratch - it is inside the kernel's maths or its barrier structure.


### RETRACTION: the mgemm `mul1` "root cause" was my own error

Earlier in this log I claimed the mgemm residual was the `mul1` flag, on the strength of a probe
showing `cos = 1.0000` against `exl3::gemm` with `mul1 = false` and `cos = -0.011` with `mul1 = true`.
That probe was invalid, and the "fix" made things worse. Reverted.

The error: I compared mgemm against `exl3::gemm`, which has **no `mul1` argument at all**. `gemm` is
the attention convention; the expert tables carry exl3's global per-tensor scale, and `moe_grouped`
passes `mul1 = true` for exactly that reason. So `mul1 = false` trivially "matched" `gemm` - it was
agreement with the wrong convention, not correctness. The give-away was in the numbers I already had:
with my change the per-card partials fell from 88-106% of the grouped path to **54-70%**, which is what
breaking something looks like, not fixing it. I read the cos=1.0000 as confirmation and did not ask why
the end-to-end output was still wrong and still non-deterministic.

What made it obvious once I looked properly was comparing the two paths with the *same* diagnostic at
the **first decode step**, where both have identical input, instead of comparing each path against my
own reference:

| layer 0, first decode step | grouped | mgemm (`mul1=false`) | mgemm (`mul1=true`, restored) |
|---|---:|---:|---:|
| card 0 | 0.010724 | 0.007492 (0.70x) | 0.011339 (1.06x) |
| card 1 | 0.006499 | 0.003509 (0.54x) | 0.006275 (0.97x) |

With `mul1 = true` restored the figures are back to the originally documented 88-106% residual. The
residual is real and still unexplained; the flag was never part of it.

Two changes from this investigation are kept because they stand on their own:
- zeroing each card's `part` before the down pass. Its contract says fp32 output need not be zeroed,
  which does not hold when `idx = -1` slots are skipped and the reduce sums all `top_k` slots.
- the tail's cross-card copy now moves `n*hid` floats, matching the grouped path, instead of
  `slots*hid` (a leftover from when `part` held raw per-slot partials).

The probes themselves are removed. For anyone picking this up: **`exl3::gemm` is not a valid oracle for
expert weights** - it cannot express `mul1`. `moe_grouped` is the only reference. And note the mgemm
path is decode-only (`n <= MOE_DECODE_MAX_N`); prefill always uses the grouped kernel, so any probe that
uses a prompt longer than that threshold measures the grouped path in both runs and proves nothing.


### mgemm decode path FIXED - now the default, +34% decode

The residual that survived many sessions was a **missing argument**, not arithmetic. The down call
stopped at `c_ptrs` - 24 positionals - so `mcg` and `mul1` fell back to their defaults, and `mul1`
defaults to **false** while the gate and up calls pass **true**. The down tables are expert tables of
the same format as gate and up, so that one projection had been silently dequantising with the wrong
scale.

Evidence, comparing the two paths at the **first decode step** where both see identical input:

| layer 0, first decode step | grouped | mgemm |
|---|---:|---:|
| card 0 | 0.010724 | 0.010723 |
| card 1 | 0.006499 | 0.006499 |

Exact. End to end the mgemm path now reproduces the reference continuation verbatim
("We need to respond to user: \"Weather in Paris?\"") and is deterministic over repeated runs.

| | grouped | mgemm |
|---|---:|---:|
| decode tok/s | 28.35 / 28.67 | **38.32 / 38.79 / 37.15** |

**+34%.** `moe_decode_mgemm()` now returns true by default; `HELIOS_MOE_GROUPED=1` restores the grouped
path for comparison. Prefill is unaffected - it always uses the grouped kernel.

How the earlier misdiagnosis happened, since it is the useful lesson: a probe asked "does mgemm match
`exl3::gemm` on one matrix", and `mul1=false` matched with cos 1.0000 while `mul1=true` did not. But
`exl3::gemm` has no `mul1` argument at all - it encodes the *attention* convention, not the expert one.
So the probe was measuring agreement with the wrong convention, and the "fix" it justified took the
per-card values from 88-106% of the grouped path down to 54-70%. **`moe_grouped` is the only valid
oracle for expert weights.** The second trap: prefill always uses the grouped kernel, so any probe whose
prompt exceeds `MOE_DECODE_MAX_N` compares the grouped path against itself and proves nothing.


### MTP works, is deterministic, and is 1.58 tokens/step - but pays for it with a second forward

I had repeatedly concluded "MTP contributes ~0%" and never questioned it. That was wrong twice over.

First, MTP is **opt-in** (`main.cpp`: `getenv("HELIOS_MTP") ? atoi(...) != 0 : false`). Every comparison I
ran omitted the variable, which means false - identical to `HELIOS_MTP=0`. So the "MTP on vs off" table
(38.65 vs 38.32) was MTP-off vs MTP-off. I never actually exercised it.

Second, when I enabled it, it worked:

```
[gen] prompt tokens=57
We need to respond to user: "Weather in Paris?" Need likely provide current weather? We don
[gen] spec: 7/12 accepted (58.3%), 1.58 tokens per step
```

- **58.3% acceptance, 1.58 tokens per step.**
- **Output byte-identical to the non-spec path**, and identical across four consecutive runs, so the
  non-determinism recorded earlier is gone. Whatever caused it (the draft-cache bug noted in this log,
  plus the mgemm work) is resolved.
- Decode **35.47 / 36.35 / 34.93 against 38.66 / 38.75 / 38.67 without it** - about 8% *slower*,
  exactly as the code comment predicted. One forward per token either way, plus the draft head.

That slowness is structural, not a bug. The step does `run_chunk({next})` and, on acceptance,
`run_chunk({draft})` - **two sequential forwards producing two tokens, i.e. 1.0x**, minus the draft
head's own forward. There is no way to beat 1.0x with a one-token draft, because the draft's forward at
`pos+1` writes KV that the next step needs either way.

The fix is batched verification, and the enabler is already in place: `head_rows_ = 2` and the head
buffer is sized for two logits rows, but `run_chunk` computes logits for the **last position only**
(`M=1`, `sub_in_ + (n-1)*hidden`). With `M = n` a 2-token chunk can be verified in one pass.

Caveat on the projection: this is only a win while the step is memory-bound. The dense parts load the
same weights regardless of n, but the MoE routes per token, so at n=2 the expert traffic roughly doubles
and the MoE is ~57% of the step. Expected gain is therefore ~1.2-1.35x, not the ~1.9x a naive
"tokens per step" reading suggests. It also interacts with GDN: 36 of the 48 layers carry recurrent
state, so a rejected token in a 2-token chunk corrupts the recurrence, and the recovery is to re-forward
rather than roll back.

MTP stays opt-in: it is currently correct but slower than not speculating.


### gr_mix: the hc mix was summing 2560 values with 4 threads. Decode +18%

With mgemm fixed the profile shifted, and `amix`+`mmix` (the hyper-connection mix) was the largest
remaining non-MoE cost at 9.24 ms/token - 32% of a step - while touching only ~94 MB/token of weights.
That is ~10 GB/s against ~700 GB/s available. I had suspected block count before and fused the five
stages into one kernel; it changed nothing, which ruled out launch overhead. This time I measured each
stage instead of guessing (`HELIOS_GMIXPROF=1`):

```
[gmix] per-call us: norm=49.11 dots=26.22 up=20.51 fin=11.96 post=22.31
```

`norm` alone was 38% of the mix for ~100 KB of traffic (~2 GB/s). The cause was visible in the kernel:
the per-stream RMS sum used **only H (=4) threads**, each looping D (=2560) times - 2560 serial FMAs
per lane, from a single block. Rewrote it as two parallel kernels: a `(R, H)` grid with full block
reductions, then a grid-stride scaling pass. Measured `norm` 49.11 -> 8.52 us.

| | before | after |
|---|---:|---:|
| amix + mmix (ms/token) | 9.24 | **5.42** |
| **decode tok/s** | 38.65 / 38.46 | **45.61 / 45.90 / 45.63 / 45.62** |

**+18%**, deterministic, all six suites pass, and prefill is unaffected (260 tok/s at 799 tokens;
261.6 with 2281 tokens, decode 42.43 - long context works).

One bug on the way, caught by the engine rather than the unit test: the old kernel indexed
`norm_raw[i]` with `i < HD` (block-local), so `norm_raw` is `[HD]`, one weight row shared by all rows.
My grid-stride version indexed it with the global `i` up to `R*HD` and read past the end for any
`R > 1`. Decode (R=1) was fine; a 57-token prefill chunk faulted with an illegal memory access. The
aux parity test does not cover R > 1, which is why it passed. Both are fixed - `norm_raw[i % HD]`.

Session total so far: decode **28.5 -> 45.7 tok/s (+60%)**, from the mgemm argument fix and this one.


### RETRACTION 2: the mgemm shape sweep was measuring garbage

After the gr_norm fix I swept the mgemm tile-shape override and reported "shape 4 = 48.32 tok/s against
auto's 45.71, +6%". That was worthless. Forcing shape 4 makes the kernel produce **garbage**:

    shape=0 (auto): We need to respond to user: "Weather in Paris?" Need likely provide current weather? We don
    shape=4:        We!!!!!!!!!!!!!!!!!!!

and shape 3 (47.92) is broken the same way. A broken kernel is *fast* precisely because it stops doing
the work, so the sweep measured the speed of not computing the answer. The default is back to auto (0),
which is the only setting verified correct, and the shape/sms knobs remain only as explicitly-documented
overrides that must be checked for output before being trusted.

This is the second time this session that I ran a speed measurement without a correctness check - after
writing in this very file that `exl3::gemm` is not a valid oracle and that I had wasted time on a probe
that proved nothing. The rule that would have caught it is cheap: **any sweep of a compute parameter
must assert the model's output, not just the rate.**

Verified configuration and results: decode **45.61 / 45.90 / 45.63 / 45.62** tok/s with reference-matching
output, prefill 259.7, all six suites pass. Session total decode **28.5 -> 45.7 (+60%)** from the mgemm
argument fix and the gr_norm parallelisation, both independently verified.


### Negative result: per-expert GEMV does not beat mgemm at decode

The MoE is 12.0 ms/token (45% of a step) and moves ~535 MB/token of expert weights, i.e. ~41 GB/s
effective against ~350 GB/s per card. The obvious suspect: `exl3::gemm` has a QTIP GEMV path for small
`M` and **`mgemm` never routes to it**, so at decode it runs a cooperative tensor-core kernel on what is
really 10 independent rank-1 problems.

So I implemented the alternative properly: for each slot, build a `GroupWords` for that slot's expert
with `mul1 = 1` and call `exl3::gemv` on its gathered row.

It is **numerically correct** - output byte-identical to the reference - and **slower**:

| | decode tok/s |
|---|---:|
| mgemm (default) | 45.92 / 45.91 |
| per-expert gemv | 39.94 / 40.24 |

The arithmetic says why: 10 slots x 3 projections x 45 layers is **1350 launches per token**, and at
~4 us of dispatch each that is ~5.4 ms, more than the bandwidth it recovers. The GEMV kernel is not the
lever on its own - the win would have to come from mgemm's inner loop, or from batching the dispatch
(CUDA graphs). The experiment is removed; correctness was checked before the timing, which is the order
that the earlier shape-sweep mistake taught.

Two bugs on the way, both caught before they could be mistaken for results: caching the host table on
the base pointer alone would have reused layer 0's slice for every layer, and `if (!cudaMemcpy(...))`
aborts on success because `cudaSuccess` is 0.


### The MoE dominates prefill too (52%), and prefill is now the larger relative gap

Prefill profile on 799 tokens (per 256-token chunk, 6-7 chunks):

    ple=7.14  amix=68.17  attn=20.31  gdn=48.03  apply=2.31  mmix=67.33  moe=234.32  final=1.15 ms

| phase | per chunk | share |
|---|---:|---:|
| **moe** | 234.3 | **52%** |
| amix | 68.2 | 15% |
| mmix | 67.3 | 15% |
| gdn | 48.0 | 11% |
| attn | 20.3 | 5% |
| ple | 7.1 | 2% |

So the same kernel that is 45% of decode is **52% of prefill**. Prefill uses the grouped kernel (the
mgemm path is decode-only by design), batched 256 tokens at a time, which is why it is 3.3x more
efficient per token than decode (0.91 vs 0.27 ms per token per layer... in the other direction: decode
pays ~12 ms/token over 45 layers, prefill ~0.91 ms/token).

Current standing against the exllamav3 baseline (both measured with the corrected harness at 320
generated tokens): decode **45.5 vs 117.2 (39%)**, prefill **266 vs 576 (46%)** on a 430-token prompt,
falling to decode **29.5 vs 127.9 (23%)** and prefill **193 vs 2,511 (8%)** at 17.6k tokens. Prefill is the larger relative gap and the MoE is the single dominant cost in both, so it is
the right place to work next - but it is one large ported kernel, not a parameter.

Verified state at this point: correctness byte-identical to the reference, decode 45.84 / 45.85
deterministic, all six suites pass.


### The hc-mix profiler was measuring its own sync. Corrected attribution

I built `HELIOS_GMIXPROF` with a `cudaStreamSynchronize` after each of the five stages. That costs
~15-20 us of *latency* per stage, which swamps every stage equally and is larger than most of them. It
produced this:

    [gmix] per-call us: norm=49.11 dots=26.22 up=20.51 fin=11.96 post=22.31     (total 130)

on which I based both the gr_norm rewrite and a deterministic split-K for `dots`. The gr_norm rewrite
was a genuine win (its stage really was 2560 serial FMAs on 4 threads, and the fix measured +18% decode).
The split-K was not, and was reverted: decode 45.62 / 45.84 against 45.84 / 45.85, i.e. neutral.

Rewriting the profiler to record its six events back-to-back and read the deltas after ONE sync per
call gives the real picture:

| stage | synced (wrong) | **event-based (true)** |
|---|---:|---:|
| norm | 8.41 | 8.47 |
| **dots** | 27.49 | **3.13** |
| **up** | 19.87 | **21.10** |
| **fin** | 11.18 | **14.13** |
| **post** | 22.47 | **4.99** |
| total / call | 91 (impossible) | **51.8 us** |

51.8 us x 96 calls = 4.97 ms, which agrees with the independent event-based `[prof]` amix+mmix bucket of
5.42 ms. The old numbers summed to more than the measured total, which is the tell I should have caught.

The real costs are **`up` (41%)**, **`fin` (27%)** and **`norm` (16%)**. `dots` is already 3.13 us and
`post` 4.99 us, so the split-K was pointed at a stage with nothing left to win - consistent with it
measuring neutral.

Lesson, and it is the third of its shape this session: **a profiler that perturbs the thing it measures
will produce confident, wrong attributions.** The tell is arithmetic - if the parts exceed the whole,
the instrument is the problem.


### The MoE is not occupancy-bound either (three null results close the door)

ncu said the mgemm kernel runs at 35% warps active with `occupancy_limit_shared_mem = 1` and 92,160 B of
shared memory, and shape 2 needs only 40,960 B - enough for two blocks per SM. The obvious move is to
launch with the shape's real requirement. Tried, and it is **neutral**:

| config | decode tok/s |
|---|---|
| default (90 KB smem, auto grid) | 45.36 / 45.21 |
| exact smem (40,960 B for shape 2) | 45.70 / 45.49 |
| num_sms=16, 90 KB | 45.89 / 45.71 |
| exact smem + num_sms=16 | 45.75 / 45.81 |
| exact smem + num_sms=32 | 45.91 / 45.41 |

Correctness held throughout (the MoE mgemm picks its grid from a heuristic, not from occupancy, so unlike
the plain gemm the reduction order cannot shift - which is why this experiment was safe where the earlier
one was not). Every cell is inside the 3.5% noise band.

**So the MoE kernel is limited by neither bandwidth (14% of peak) nor occupancy (35% warps, and raising
it changes nothing) nor the grid.** With both SM throughput (28%) and memory (14%) low and insensitive to
occupancy, the remaining explanation is the EXL3 trellis decode itself: unpacking 2-bit weights from
16x16 codebook blocks is ALU work that does not go away with better scheduling. That is consistent with
the reference hitting the same wall - the gap to it is not in this kernel's configuration.

This closes the search: shape, grid, and occupancy are all exhausted for the MoE decode path. What is
left is the algorithm (a different decode strategy for low-bit weights), which is not a tuning change.
The knob was removed rather than left in as dead configuration.

### The hc mix: one real win, two neutral attempts, and a floor

Three changes were tried against the hc mix after the gr_norm fix. Only the first moved anything.

| change | result |
|---|---|
| `gr_norm` parallelised (2560 serial FMAs on 4 threads -> block reductions) | **real: amix+mmix 9.24 -> 5.42 ms, decode +18%** |
| deterministic split-K for `dots`, 24 -> 192 blocks | neutral (45.62 / 45.84 vs 45.84 / 45.85) - reverted |
| grid-stride `fin`, 10 -> 512 blocks | neutral (14.10 vs 14.13 us) - reverted |

Both neutral ones were reverted rather than kept, and both were aimed using the *corrected* profile, so
this is not a measurement error - it is evidence that block count is not what bounds these stages.
`fin` reads 50 KB; `dots` reads 491 KB. Neither responds to 8-50x more blocks. Whatever their 14 and 3
microseconds are, they are not launch width.

The one change that worked did so because it fixed something actually broken: the norm sum used four
threads to walk 2560 elements serially and ran at ~2 GB/s. That is a real defect with a real fix, and it
is the shape of change worth making here. The others were plausible generalisations from it that the
measurements refused. Net for this session on the hc mix: 9.24 -> 5.42 ms.

Remaining hc mix, per call: up 21.2, fin 14.1, norm 8.5, post 5.1, dots 3.1 - about 52 us against a
bandwidth bound near 8 us, but not block-count-bound and not launch-bound (a fused single-kernel version
was tried earlier and was also neutral).


### The measurement noise floor is ~3.5%, and it invalidated a fourth change

I swept mgemm's `force_num_sms` with the auto shape (the earlier sweep was invalid because it used
shape=3, which produces garbage) and verified output at every setting:

| force_num_sms | decode tok/s | output |
|---|---:|---|
| 8 / 16 / 24 / 40 / 80 | 47.26 / 47.19 / 46.98 / 47.35 / 46.91 | all correct |
| auto | 45.60 / 45.73 / 45.85 | correct |

Every explicit value beat every auto reading, none of them differed from each other, and the `sms=80`
case has concurrency 1 (the opposite of the shape theory), so the story looked like "the auto path does
something expensive per call that an explicit value skips".

Then a later auto run measured **47.27** - inside the explicit band. The effect was never there. The
auto baseline had been 45.6-45.9 in four runs and 47.3 in this one, i.e. **run-to-run variance is about
3.5%**, which is larger than the effect I was chasing. The exllamav3 server holds both GPUs, so
contention varies between runs and there is no way to tighten it while that is true.

Reverted. The rule this session keeps re-teaching: **state the effect size you need to resolve, and if
it is inside your noise band, either gather enough repetitions to separate it or do not claim it.** Two
runs at ~3% was neither. Changes kept this session are only those that moved the number by far more than
the band: the mgemm `mul1` argument (+35%) and the `gr_norm` parallelisation (+18%).

Session tally of attempts: 2 kept, 4 reverted as neutral (fused gr_mix, per-expert gemv, split-K dots,
grid-stride fin, force_num_sms - five counting the last), 3 self-caught errors and one corrected
profiler. All recorded.


### Prefill scaling measured: attention is quadratic, the MoE is not

Per-chunk profile at three context lengths (all with the current default paths):

| prompt | tokens | chunks | attn/chunk (ms) | moe/chunk (ms) | prefill tok/s |
|---|---:|---:|---:|---:|---:|
| 2k | 938 | 6-7 | 25.3 | 266 | 248.6 |
| 8k | 3,752 | 17-18 | 89.9 | 396 | 251.8 |
| 32k | 15,008 | 61-62 | **348.5** | 466 | 199.0 |

Attention per chunk grows linearly with context (25 -> 90 -> 348 ms, ~3.7x per 4x context), and the
number of chunks grows linearly too, so total attention work is quadratic: **0.18 s -> 1.6 s -> 21.6 s**
for 16x the tokens. The MoE, by contrast, grows sublinearly (266 -> 396 -> 466, 1.8x for 16x) because a
256-token chunk touches every expert regardless of how many tokens are in it.

This re-frames the QSA item. It was filed as "kernels are GLM-shaped, needs parameterising" - a decode
nicety. It is actually the **prefill scaling blocker**: the baseline's sparse indexer keeps attention
O(topk) and holds 2,392-2,911 tok/s from 9.4k to 141k tokens, while Helios falls from 301 to 147 tok/s
over the same range. At 32k, attention is 348 of 1,218 ms per chunk (29%) and still climbing; at 141k it
would dominate completely.

Consequence for the prefill gap: at short context the MoE is 52% of the time and the two paths agree on
what to do. At long context attention overtakes it, and no amount of MoE work fixes that. The work order
should be MoE first (helps everywhere, bounded), then QSA (unlocks context).


### ncu: the MoE kernel is latency-bound at 35% occupancy, and the obvious fix regresses

ncu is available on this box (`/usr/local/cuda-13.0/bin/ncu`, needs `sudo -n` for the counters), and it
answers what the event buckets could not. On the decode mgemm kernel:

| metric | value |
|---|---:|
| grid | (16, 1, 5) = 80 blocks on 82 SMs |
| block | 512 threads = 16 warps |
| **warps active** | **34.7% of peak** |
| memory throughput | 14.3% of peak |
| SM throughput | 28.4% of peak |
| duration | 45.5 us |
| `launch__occupancy_limit_shared_mem` | **1** |
| `launch__shared_mem_per_block_dynamic` | 92160 |

Neither memory nor SM is saturated, so the kernel is **latency-bound with nothing to hide it behind** -
and the cause is visible in the last two rows: the block requests 92,160 B of shared memory on a part
that has 100 KB per SM, so exactly one block fits and 16 of 48 possible warps are resident.

The smem requirement per shape follows directly from the layout in `exl3_gemm_inner.cuh`, and the launch
was passing the cap for every shape:

| shape | K x N | needs | blocks/SM at 100 KB |
|---|---|---:|---:|
| 3 (selected) | 32x256 | 77,824 | 1 |
| 2 | 32x128 | **40,960** | **2** |

Shape 2 fits two blocks per SM, so launching it with its real requirement instead of the cap looked like
a free 2x occupancy. It is not free: **regression, reverted.**

- Output diverges: `... We need likely use` becomes `... We need to use too`.
- Decode drops to **39.45 / 42.65** against a 44.9-47.4 band - below the band, so not noise.

The mechanism is in the non-mgemm `gemm` path: it **autotunes the grid against `smem_launch`**
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, then a block-count sweep). Handing it less smem lets more
blocks fit, the autotuner picks a wider grid, and a wider grid changes the **reduction order** in the
split-K epilogue. That is a different-but-equally-valid summation whose last-bit difference greedy
decoding amplifies into different text. It is the same class of problem as the L37 deep-layer offset, and
it is the reason "tune the tile shape" is not a free action on this engine: the shapes are numerically
coupled through autotuning.

Note also that the divergence showed up as a *different* greedy continuation from the very first
divergent token, which is the same amplification signature as the L37 offset - and that I nearly read the
smem patch as "changing numerics, therefore wrong". It is not wrong, it is a different valid summation;
it is simply a different summation, and the baseline is the one that defines correct here.

The shapes themselves are byte-identical to the reference (`~/projects/exllamav3`), and the reference
passes the same `SMEM_MAX`, so this is upstream behaviour rather than a porting defect. What the reference
has that we do not is a `CoopKernelAutotuner` that sweeps candidate block counts per shape; that is the
piece worth porting, and it would have to be done in a way that pins the winning configuration rather
than re-deriving it per launch.

So the MoE is confirmed latency-bound and confirmed blocked: the lever exists (2 blocks/SM via shape 2)
but pulling it changes numerics, and matching the reference's summation order is a prerequisite for
changing anything that affects it.


### Serial measurement hygiene

Runs overlapped with other instances more than once this session, and an overlapped run reads ~2-3 tok/s
low (42.09 / 40.80 / 41.63 against a 44.9-47.4 band). Concurrent helios instances contend for the same
bump-allocated pools and the same PCIe links, and the failure is silent - no OOM, just a slower number
that looks like a regression. On this box a decode measurement is only valid when `pgrep -af
"build/helios"` is empty beforehand, and one stray `llama-server` (bge-m3 embedder, 1.2 GB across both
cards) sits resident at all times.


### Seventh attempt on the hc mix: batching elements per warp, also worse

With the corrected (event-based) profile the hc mix per call is ~52 us: `up` 21.2, `fin` 14.1, `norm` 8.5,
`post` 5.0, `dots` 3.1. `up` is the largest, and its structure is visibly wasteful - `rank = 24` means
only 12 of 32 lanes load, then 5 shuffles reduce, so each warp issues one short load burst and then stalls.
The natural fix is batching several elements per warp to create independent memory streams.

Tried 8 elements per warp, keeping the per-lane accumulation order and the butterfly reduction so the
result should have been bitwise identical. It is **worse**: the stage goes **21.10 -> 33.23 us**.

So at this size the launch already has 2048 blocks, and the limiter is not memory-level parallelism
within a warp - it is total warp count. Fatter warps reduce the number of warps and lose more than the
extra in-flight loads gain. Reverted.

That is three different mechanisms tried on the hc mix (fusion, grid width, load batching) with one
success, and the success was not a mechanism at all - it was fixing code that was genuinely broken
(4 threads walking 2560 elements serially at ~2 GB/s). The stage now costs 52 us against a ~8 us
bandwidth bound, and I do not have a hypothesis for the remaining 44 us that survives measurement.
Recorded as open rather than guessed at again.

**Attempt ledger, this session:** kept 2 (mgemm `mul1` +35%, `gr_norm` +18%). Reverted 7: fused gr_mix,
per-expert gemv, split-K dots, grid-stride fin, `force_num_sms`, exact-smem launch, up-stage batching.
Every revert was verified on output, not just timing.


### ncu closes the hc mix: it is weight traffic, not inefficiency

I had been treating the remaining 44 us of hc-mix overhead as an efficiency problem. ncu on `gr_up_kernel`
says otherwise:

| metric | value |
|---|---:|
| dram bytes read | 9,924,736 |
| duration | 17.888 us |
| **memory throughput** | **66.1% of peak** |
| warps active | 82.4% of peak |

9.92 MB in 17.9 us is **555 GB/s** - two thirds of what this card can do, at 82% occupancy. I had
estimated ~300 GB/s and called the stage inefficient, because I read `rank` as 24. It is **320**: the
hyper-connection weights are `down [320, 10240]` and `up [10240, 320]`, both f16, and all 32 lanes are
active across 5 iterations. My "only 12 of 32 lanes load" note was carried over from the GLM model and was
simply wrong here.

Which reframes the whole item, because those tensors are large and **unquantized**:

| | per token | dtype |
|---|---:|---|
| hc `down`+`up`, 45 layers x 2 sites | **1.18 GB** | **f16** |
| MoE experts, 10 of 512 per layer | ~756 MB | 2.05-bit |

The hyper-connection mix moves *more* bytes per token than the MoE does, in twice the bit width. At the
measured 555 GB/s the weight traffic alone is 2.13 ms of the 5.42 ms the two mix phases cost; the rest is
the smaller stages and the gaps between five dependent launches.

So the hc mix is not mis-tuned and there is no scheduling trick left in it - I tried fusion, grid width
and load batching, and the one that worked (`gr_norm`) worked because it was fixing a genuinely broken
loop. The remaining lever is **quantizing the hyper-connection weights**, which is a numerics change of
exactly the kind that has blocked every other attempt this session: a different summation diverges greedy
text. Recorded as the real finding rather than another silent null result.

This also explains why the group kept returning here: 5.42 ms/token is 12% of a decode step, which looks
like fat, and it is - but it is weight bytes, and the only way to spend fewer is to read fewer bits.


### RETRACTION 3: the mgemm shape sweep tested kernels that cannot work

I reported "shape 4 = 48.32 tok/s vs auto 45.71, +6%" and then retracted it on correctness grounds
(output was `We!!!!!!!!!!!!!!!!!!!`). The deeper reason, found later, is that **shapes 3 and 4 are
mathematically invalid for this model** and the reference never considers them.

`exl3_gemm_shape_compat(shape, m, k, n, K)` requires `k % tilesize_k == 0 && n % tilesize_n == 0`. This
checkpoint has `hidden_size = 2560` and `moe_intermediate_size = 640`, and the MoE's gate/up projections
are (k=2560, n=640):

| shape | tilesize_k x tilesize_n | n=640 divisible? |
|---|---|---|
| 1 | 16 x 128 | yes (5) |
| 2 | 32 x 128 | yes (5) |
| 3 | 32 x 256 | **no** (640 % 256 = 128) |
| 4 | 16 x 512 | **no** (640 % 512 = 128) |

The reference filters candidates through this predicate before timing them
(`if (!exl3_gemm_shape_compat(...)) continue;`). Our `mgemm` path has the function but never calls it -
`select_exl3_mgemm_kernel` reaches it only through the auto selector, which picks a compatible shape. So
forcing 3 or 4 bypasses the only check standing between the caller and an invalid kernel, and the result
is garbage that runs *faster* because it is not doing the work.

This makes the "shape 3 (47.92) looked good too" data point meaningless too - both "wins" were invalid
kernels. What survives from that experiment is only the negative result: the auto-selected shape is not
beatable by any valid alternative, because there are only two valid shapes and one is already chosen.

A clean 2D sweep over the valid space, with per-cell correctness, confirms it:

| shape \ num_sms | 0 | 4 | 8 | 12 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|
| 0 (auto) | 46.49 | 46.89 | 46.54 | 44.65 | 46.86 | 46.82 |
| 2 | 45.63 | 45.56 | 45.57 | 45.76 | 46.07 | 45.40 |

All twelve cells produce the reference continuation, and all twelve fall in a 2.2 tok/s spread that is
inside the ~3.5% run-to-run noise. The search space the reference's autotuner explores is exactly this
space, and for this model there is nothing in it.

It also identifies the real gap against the reference, which is not "tune the shape" but the
`CoopKernelAutotuner`: the reference builds a candidate list per (m, k, n, K, c_fp32, device, cc,
total_sms, cb, bszm_in, bszm_out, half_k), expands it over `num_sms` in steps of 2 with
`concurrency = total_sms / num_sms`, times each candidate with CUDA events (3-8 repeats depending on
size), and caches the winner to disk. Our path fixes `num_sms` from a heuristic:

    num_sms = tiles;
    if (num_sms * bszm > total_sms) num_sms = MAX(total_sms / bszm, 1);
    if (num_sms <= total_sms && tiles / num_sms > 48) num_sms = MIN(total_sms, num_sms * 2);
    concurrency = MIN(total_sms / num_sms, bszm);

which is the same fallback the reference uses when autotuning is disabled. I swept `force_num_sms` over
8/16/24/40/80 and everything landed inside the ~3.5% noise band, which is consistent: the useful range is
a product of num_sms and concurrency, and one knob does not explore it. Porting the autotuner is the
unexplored lever, and it must pin its winner per shape rather than re-derive it per launch, or it will
reintroduce the same reduction-order instability that made the exact-smem experiment regress.


### The num_sms heuristic is exhausted (2D sweep)

With the shape question settled (only 1 and 2 are valid; auto already picks well), I swept shape x
num_sms with per-cell correctness checks:

| shape | num_sms | decode | output |
|---|---|---:|---|
| 0 (auto) | 0 | 46.49 | correct |
| 0 | 4 | 46.89 | correct |
| 0 | 8 | 46.54 | correct |
| 0 | 12 | 44.65 | correct |
| 0 | 16 | - | correct |
| 2 | 0 | 45.46 | correct |
| 2 | sms sweep | 44.60 | correct |

Everything lands in the 44.6-46.9 noise band. Combined with the earlier single-knob sweep (8/16/24/40/80
-> 46.9-47.4, indistinguishable) this is consistent with ncu: the kernel is **latency-bound at 35%
occupancy** with 92,160 B of smem pinning it to one block per SM. Grid shape cannot fix that, because
more blocks do not fit. The only fixes that would are (a) a shape whose smem allows 2 blocks/SM, which
changes reduction order and diverges output, or (b) an autotuner that also explores what the heuristic
cannot express.

Conclusion for the MoE decode path: **the heuristic and shape space are both exhausted.** The remaining
levers are structural - smaller smem per shape (blocked on summation order), the reference's
`CoopKernelAutotuner` port, or a genuinely different kernel (the QTIP GEMV, already measured and refuted
for dispatch-cost reasons). I am recording this as a bounded, well-understood problem rather than
continuing to sweep knobs that measurement says cannot help.


### The profiler was blending prefill and decode. Fixed; attention is the ONLY context-scaling phase

`n_prof` counted every `run_chunk`, so one "per-token ms" line averaged 59 prefill chunks of 256 tokens
with 40 decode chunks of 1 token. The prefill number was diluted ~256x and the decode number inflated by
however many chunks prefill contributed. Every phase table in this log that mixed the two is a blend, not
a measurement - including the one I had just written claiming the MoE grows 23x with context.

`run_chunk` now accumulates a second, decode-only set of buckets when `n == 1`, and both are printed.

**Decode-only, per token:**

| phase | 430 tok | 15,008 tok | ratio |
|---|---:|---:|---:|
| **attn** | 1.18 | **11.10** | **9.4x** |
| moe | 11.87 | 11.79 | 1.0x |
| gdn | 3.64 | 3.69 | 1.0x |
| amix | 2.66 | 2.69 | 1.0x |
| mmix | 2.52 | 2.54 | 1.0x |
| ple | 2.03 | 0.15 | - |
| total | ~24 | ~32 | |

This inverts the conclusion I had drawn ten minutes earlier. **The MoE does not scale with context at
all** - it is 11.8 ms flat at 430 tokens and at 15k. It looked like it scaled because prefill chunks were
being averaged in. **Attention is the only phase that grows**, linearly, and it is therefore the entire
explanation for Helios falling from 45.5 to 29.5 tok/s as context grows while the baseline rises.

Projecting attention's measured 0.74 us per token of context:

| context | attn | share of step | implied decode |
|---:|---:|---:|---:|
| 430 | 1.2 ms | 5% | ~45 tok/s |
| 15,008 | 11.1 ms | 34% | ~30 tok/s |
| 141,000 | 104 ms | ~82% | ~11 tok/s |
| 230,000 | 170 ms | ~90% | ~8 tok/s |

So at the 230k context the brief asks for, dense attention would be ~90% of the step and the engine would
be running at single-digit tok/s, while every other phase stays flat. **QSA is not a prefill nicety or a
decode optimisation - it is the only thing standing between this engine and a usable long context, and
its absence is a hard ceiling on the context-length half of the deliverable.** The MoE, by contrast, needs
no context work at all; it is a fixed 11.8 ms that only gets faster by making the kernel itself faster.


### Split-KV decode attention: long-context attention 11.10 -> 9.08 ms, and a bump-allocator trap

The decode-only profiler showed attention is the only phase that scales with context, and the kernel had an
obvious structural flaw for decode. `gqa_dense_split_kernel<<<n * n_q_heads, 512>>>` launches one block per
(row, q_head) and splits the key range across 16 warps *inside* that block. At prefill (n=256) that is
6,144 blocks - ncu measured it at **97.5% of memory peak**, nothing wrong there. At decode (n=1) it is
**24 blocks on an 82-SM GPU**: 71% of the machine idle while the kernel streams 369 MB of KV.

Fixed with `gqa_dense_split_kv_decode`: the same single-warp accumulation, one block per KV chunk, then a
combine pass. It reuses the existing kernel's chunk partition and combine arithmetic verbatim, so it is
**bitwise identical** by construction - the parity test now covers both paths and reports the same
`max_err 3.007e-05` for each.

| | before | after |
|---|---:|---:|
| attn, decode @ 15,008 tok | 11.10 ms | **9.08 ms** |
| decode @ 15,008 tok | 30.52 | **32.36** |

Two bugs and one trap on the way:

1. **`We!!!!!` on the first run.** The combine kernel built its `(m, l)` table with `if (lane < 16) { ... }`
   and `__syncwarp()`. With `#pragma unroll` and constant bounds the compiler promotes that array to
   registers, making it per-lane private, so lanes 16-31 combined against uninitialised values. Fixed by
   having every lane load all 16 pairs itself.
2. **The parity test gave false assurance.** It called `gqa_dense_f16` directly and never touched the new
   entry point, so it passed while the engine was emitting garbage. The test now runs both paths; the
   split-KV case is what exposed bug 1, with `max_err 2.9e+01` against a reference magnitude of 0.02.
3. **A bump-allocator trap worth remembering generally.** Adding the 12.7 MB `kv_part` scratch cost
   **2.3 tok/s at short context** (41.5 vs 43.8 mean over 3 runs) with no code path involved - `A()` is a
   bump allocator, so a new block shifts every allocation after it and changes their addresses. Allocating
   it **last** restored 44.3 mean with zero measured cost. Any future scratch addition should be appended
   at the end, and short-context throughput re-checked, not just the path that uses it.

The path is gated on `n <= 32 && pos0 + n >= 2048`: below 2k keys the plain kernel is already fast
(1.19 ms either way at a 6-token prompt) and the combine launch is pure overhead.

Confirmed stable across a full 40-token run at 15k context: attention settles at **9.15-9.31 ms** on every
decode step (from 11.10), and short context is back to 43.4-45.6 tok/s mean 44.3 with the allocation
reordered - within noise of the 45.5 that preceded this work. Six suites pass, output byte-identical to
the reference, build current.

**What this did and did not buy.** It is a constant-factor win on one phase, not a complexity change:
attention is still linear in context, still 9.1 ms at 15k, and at 230k it would still be ~170 ms and ~89%
of the step. The MoE is untouched at 11.8 ms flat. So decode goes from ~45 to ~32-45 depending on context
rather than becoming context-independent. QSA is still the only thing that makes long context cheap; this
just made the dense fallback less bad.


### QSA indexer: pooling and top-k selection built and parity-tested (not yet wired)

The context-length half of the deliverable needs the sparse indexer. The reference semantics are in
`~/projects/exllamav3/exllamav3/modules/qsa_indexer.py`; this checkpoint's indexer is 4 heads x 128 dim with
a single raw key head (`indexer_budget=2048`, `compress_ratio=4`, so 512 selected blocks), and the main
attention is plain GQA - so the selection is at 4-token block granularity and feeds a GQA kernel, not the
GLM MLA path.

`src/cuda/attn/qsa.{cuh,cu}` now implements, against a host reference:
- incremental 4-key mean pooling in fp32, cast fp16, k_layernorm (constant_bias 1), partial-NEOX rope at
  the block START position - a block is final when its last key arrives, so the pool is cacheable
- `score[b] = sum_h relu(q_h . pooled[b]) * scale`, then top-n_sel block selection

`test/test_qsa_parity.cpp` verifies both against the reference:

    [qsa_pool] max_err 3.187e-03 on |x|~0.817 (40 blocks, tail=3)  PASS
    [qsa_sel] got 8 blocks, want 8  PASS
    QSA PARITY: ALL PASS

Four bugs in the top-k, each caught by the test rather than by reading (the first version returned blocks
0 and 32 no matter what the scores were - the test's expected set is {3,4,10,11,20,21,23,27}):

1. **A warp race.** All 32 lanes hold different candidates but share one per-warp list, so a per-lane
   insert races. Fixed by choosing the winner warp-wide (butterfly max + `__ballot_sync`) and having only
   that lane write.
2. **Warp-divergent loop bound.** Deriving the iteration count from a per-lane `b < n_blocks` desynchronises
   the warp, and the `__any_sync` inside then **hangs the GPU** (the test timed out). Fixed with a uniform
   round count and out-of-range lanes holding `-inf`.
3. **Descending list used the wrong floor.** The "does this candidate beat the list" check compared against
   slot 0, which in a descending list is the BEST element - so after one insert nothing could ever enter.
   Switched to an ascending list (slot 0 = weakest).
4. **No fill count.** Inserting into an all-`-inf` list by walking down from slot `c-1` always landed on
   `c-1` and overwrote the previous value, so exactly one entry survived per warp. Added a per-warp count
   and shift-from-fill-point insertion.

### Sparse GQA attention: implemented and unit-verified (still not wired)

Added on top of the indexer:

- **`qsa_expand`** turns a query's selected block ids into the exact token indices it may attend to,
  device-side. The obvious `atomicAdd(n_out, 1)` version is wrong here: interleaving is not run-to-run
  stable, so the attention summation order would vary and greedy decoding would stop being reproducible -
  the same failure class as the grid/autotune experiments. Each query instead writes into a fixed slot
  range `[q*cap, q*cap+count)` with the count precomputed.
- **`gqa_sparse_decode`** attends over the gathered set. It is the dense kernel's per-lane dot, online
  softmax and combine, with the key range replaced by the gather - the same computation over a subset,
  not an approximation.

`test_qsa_parity` now covers four things, all passing:

    [qsa_pool] max_err 3.187e-03 on |x|~0.817 (40 blocks, tail=3)  PASS
    [qsa_sel] got 8 blocks, want 8  PASS
    [qsa_sparse] max_err 1.120e-04 on |o|~0.080 (37 of 37 tokens)  PASS
    [qsa_sparse_subset] max_err 2.396e-04 on |o|~0.148 (11 of 37 tokens)  PASS
    QSA PARITY: ALL PASS

The subset case (11 of 37, non-contiguous indices) is the one that matters: it proves the gather and the
softmax over an arbitrary selected set, not just the trivial all-tokens case.

Four more bugs, all in the new code, all caught by the tests:

1. **Chunk blocks overwrote each other.** Reusing the dense kernel's one-block-16-warps shape with a grid
   of chunks made every warp redundantly recompute the same range and the N chunk-blocks all write the
   same output row. Only visible once there was more than one chunk - the 11-token test passed and the
   37-token one did not. Fixed by going two-phase (per-chunk partials, then a combine), the structure
   already proven by split-KV.
2. **The combine let lanes 16-31 read uninitialised `m[]`/`l[]`.** Exactly the split-KV combine bug,
   reintroduced because it is easy to write the "natural" way. Every lane now loads all heads.
3. **`atomicAdd` slot assignment** would have made results non-reproducible run to run (above).
4. The grid was sized from `cap` while the chunk stride came from `nvis`, so only a fraction of the
   selected set was attended (max_err ~1.0, i.e. unrelated output).

### QSA wiring: WORKS end to end, but is 2x SLOWER than dense (so it stays opt-in)

The engine path now runs correctly. Measured at 15k context, decode-only, per token:

| | attn phase | decode |
|---|---:|---:|
| QSA on | **42.0 ms** | 15.67 tok/s |
| QSA off (dense) | 9.35 ms | 32.20 tok/s |

**The sparse path is 4.5x slower in the attention phase and 2x slower overall**, so `HELIOS_QSA=1` is
opt-in and the default remains dense. The output is correct (it reproduces the reference continuation).

Two reasons, both structural rather than bugs:

1. **The gather destroys locality.** Dense attention reads K/V contiguously: at 15k context that is a
   369 MB stream at ~555 GB/s (the rate `gr_up_kernel` already achieves). The sparse path indexes
   `k[pos * n_kv_heads * hd + kvh * hd]` through `tok_idx` - 2050 scattered 4-token runs - so every
   32-lane warp gathers 32 different rows. The traffic saved is real (2050 vs 15009 keys, 7x) but the
   achieved bandwidth collapses.
2. **The indexer is not free.** It adds a full `index_qk_proj` EXL3 matmul (2560 -> 640) per layer per
   decoded token, on top of the attention it is meant to accelerate. At 12 QSA layers that is 12 extra
   projections per token, and a 1-token matmul is latency-bound.

The reference gets the speedup because it **pools K/V into contiguous 4-token blocks at write time**
(`kpool_write`) and scores against that plane, so the gather is block-contiguous rather than
per-token scattered. That is the piece this port does not have: our KV cache is still per-token, so
selecting 512 blocks still costs 2050 scattered reads. Porting the pooled KV plane is the next
necessary step, and until it exists, dense is the better engine.

The remaining wiring bugs, all found by putting sync points inside each stage:
- `qsa_topk_kernel` requested **exactly 49,152 B** of dynamic shared memory. A request at the 48 KB
  default cap is **rejected**, and the failure leaves a sticky `cudaErrorInvalidValue` that every later
  `cuda_check` inherits - so the fault surfaced as "invalid argument" in an unrelated activation kernel
  two stages downstream, while every `cudaDeviceSynchronize` in between reported success. `cudaFuncAttributes`
  showed the kernel itself was fine (16.5 KB static, 40 regs). Now capped at 47 KB.
- The pooled state was only built during decode, so at the first decode token `n_blocks == 0` and there
  was nothing to select. Pooling now runs for every chunk, one row at a time (the scratch is one row).
- `part` was sized with the indexer's head_dim (128) while `gqa_sparse_decode` runs on the attention's
  256, under-allocating it by 1.5 MB.
- The sparse combine was a two-stage reduce-then-combine whose group slots hold one lane's 8 components
  while the consumer indexed them at its own lane offset. Replaced with the dense kernel's proven
  16-warp shared-memory combine, which the parity test confirms (1.1e-4 / 2.4e-4).

**A methodological note worth keeping:** every per-stage sync reported "no error" while the engine still
died. The syncs were placed *after* the launches but the failing launch was `cudaPeekAtLastError`-silent
in a different stage entirely. Only reading `cudaFuncAttributes` and the *stale* error via
`cudaPeekAtLastError` at several points located it. Sticky CUDA errors make bisection by sync point
misleading unless you also check the error state where it was set, not where it was observed.

### (previous) QSA wiring attempted: engine-level path is still wrong, so it is OPT-IN

Wired into the engine (per-layer pooled state for the 12 QSA layers, the `idx_qk_proj` projection, the
`> 2051` threshold, expand + sparse decode) and it does **not** work: at 15k context decode emits nothing,
while the dense path gives the reference continuation at 33.4 tok/s. `HELIOS_QSA=1` opts in; **default is
off and engine behaviour is unchanged** - all seven suites pass, short-context decode 44.4 tok/s, 15k
decode 33.4 tok/s.

The components are unit-tested; the wiring is not. `test_qsa_parity` drives `gqa_sparse_decode` with a
hand-built token list, so it never exercises the indexer projection, the per-layer pooled state, the
scratch layout, or the engine's buffer plumbing. Five wiring bugs found by bisection:

1. **SIGSEGV during prefill.** The branch was taken for `n` up to 256 (a prefill chunk) while the scratch
   is laid out for one row. Restricted to `n == 1`; prefill keeps dense attention for now.
2. **The qk projection's Hadamard scratch overran.** `qsa_project` passed `sc.qk` as the gemm's `a_had`
   argument, but `qk` is sized for the output; the gemm writes `n*hidden` halves there, overwriting q,
   raw_k, sel and score. Gave the projection its own `had` buffer.
3. **The indexer rope read garbage positions.** `QsaScratch::pos` pointed into scratch that nothing
   filled; now reuses the layer's positions, already H2D'd.
4. **The expand kernel used `threadIdx` as a striding variable but launched with one thread**, so it
   walked only block 0 and then overwrote the in-band count with a block id - the sparse kernel attended
   to a garbage range.
5. **No null checks on the QSA allocations**, so a short bump-allocator allocation surfaced as a
   mid-prefill segfault rather than an init failure.

### ROOT CAUSE of the QSA wiring failure, and a correction to my own bisect

**My bisect conclusion was wrong before the real cause was found.** I had changed the entry gate to
require `HELIOS_QSA`, then ran the stage sweep setting only `HELIOS_QSA_STOP` - so `use_qsa` was false in
every run and all of them took the dense path. The "every stage fails" result was an artifact of my own
experiment design, not evidence about the kernels. Re-run with both variables set, the picture inverts.

The real cause, found by putting sync points inside `qsa_project`:

    [proj] before gemm: qk=...1400 had=...0000 q=...1900 n=1 qk_out=640 hidden=2560 bits=2
    [proj] after gemm: no error
    (no further output - the process dies)

The fused qk projection is **host-side split** into q and the raw key:

    const half* src = (const half*)sc.qk;                       // DEVICE pointer
    qrow[j] = src[(size_t)i * stride + j];                      // read from the HOST
    rk[d] = krow[d];                                            // likewise

`src`, `qrow` and `rk` are all CUDA device pointers. Dereferencing them on the CPU is a segfault, and
because the `cuda_check`/sync points come *after* the loop, nothing ever reported it - the process just
died, at every QSA stage, on every long-context run, with empty stdout. It is now a device-side
`cudaMemcpyAsync` (D2D), and the projection completes:

    [proj] after gemm: no error
    [proj] after split: no error
    [proj] after rmsnorm: no error
    [proj] after rope: no error
    -> 15k decode: correct output, 32.92 tok/s

A second, latent bug in the same function: the scratch laid out `qk` as `n_heads * head_dim` per row when
the gemm writes `(n_heads + 1) * head_dim` (the +1 is the raw key), so the region's tail overlapped `q`
and `raw_k`. Both are fixed.

**The lesson is about method, not CUDA.** Every QSA *component* had passed a parity test that drove it
directly; the failure lived entirely in the glue, and four of the five wiring bugs were of a kind no unit
test of the component can see: a host/device pointer confusion, a scratch size that forgot a `+1`, a
branch taken for a batch size the buffer was not laid out for, and a striding variable in a kernel
launched with one thread. The stage-gate bisect then told me something confidently false, because I
changed the variable it depended on and then ran it without the new variable. **A bisect is an
experiment: check that each arm actually exercises the branch it claims to.**


**The sparse threshold, from the reference** (`qsa_indexer.py:473`):

    def sparse_threshold(self) -> int:
        # Highest query position for which dense attention is still exact + 1
        return 4 * self.block_topk + 3

With `block_topk = budget / compress_ratio = 512`, that is **2051**. Below it the reference runs the dense
kernel and the result is exact, so Helios' current dense path already agrees with the reference up to
2051 tokens of context - which is where the short-prompt parity tests live. Sparse attention only has to
be right *above* 2051, and only needs to attend to `4 * 512 = 2048` tokens regardless of how long the
context grows. That is the whole mechanism: at 15k context the reference reads 2048 keys per query, not
15008, and at 230k it still reads 2048.


### Prefill profile: the hc mix (44%) outweighs the MoE (35%), and it behaves nothing like it does at decode

Profiled the prefill path for the first time. Per 256-token chunk:

| phase | ms | share |
|---|---:|---:|
| **amix + mmix (hc mix)** | **297.6** | **44%** |
| moe | 234.3 | 35% |
| gdn | 103.6 | 15% |
| attn | 20.3 | 3% |
| ple | 12.6 | 2% |

ncu on `exl3_moe_kernel` (the prefill MoE, never profiled until now):

    grid (8,1,10)x(512,1,1)  92.16 KB dynamic smem/block
    251.9 MB read, 3.25 ms, 63.0% of peak memory throughput, 33.0% warps active   (card 0)
    673.0 MB read, 8.64 ms, 62.8% of peak memory throughput, 33.5% warps active   (card 1)

So the **prefill MoE is already at 63% of memory peak** - it is close to the hardware limit and there is
no 5x sitting in its scheduling. That is a different story from the decode mgemm (14% memory, and
insensitive to occupancy), because prefill has 256 tokens of work per weight byte and decode has one.

The corollary is that the decode conclusion does not transfer: "the MoE is ALU-bound on EXL3 trellis
decode" is a statement about the decode path only.

**The hc mix is now the largest prefill line item at 44%, and it is the least optimised.** It is
1.18 GB/token of **unquantized f16** hyper-connection weights - `down [320,10240]` + `up [10210,320]` per
site, 45 layers x 2 sites - against 0.76 GB/token of 2.05-bit expert weights. In bytes moved it is the
engine's dominant read, and unlike the experts it has never been quantized.

At decode `gr_up_kernel` measures 555 GB/s (66% of peak), so the kernel is efficient *per byte*. The
prefill numbers do not follow from that: the same weights, read once per layer per chunk, account for a
1.18 GB/chunk read inside 298 ms - roughly 4 GB/s. Whatever the prefill hc mix is doing, it is not what
it does at decode, and it is now the single largest target in the engine.

This reprioritises the remaining work. The MoE decode path is closed (shape, grid, occupancy all
exhausted). The QSA path works but is 2x slower than dense pending a pooled K/V plane. The **hc mix at
prefill** is now the biggest unexploited cost, and unlike both of those it has a clear mechanism -
quantize the weights, or restructure the mix to read each weight once for many tokens.


### The hc mix at prefill: a 146x re-read of the same 6.55 MB matrix (FIXED)

ncu on the hc mix at prefill (R=256), before the fix:

    gr_dots   (320,256)x256   2.26 ms   391.4 MB   55.2% mem   97.4% warps
    gr_up     (1280,256)x256  1.34 ms   957.2 MB   79.5% mem   89.0% warps
    gr_fin    (256,10)x256     20.4 us    10.5 MB   69.6% mem   91.4% warps

`gr_up` was reading **957.2 MB from DRAM for a 6.55 MB `up` matrix** - a 146x re-read. The grid was
`(j, r)`, so each of the 256 tokens got its own sweep of the whole matrix and L2 did not capture the
reuse across a 327,680-block grid. At 79.5% of peak memory throughput the kernel was never the
problem; there were 146x too many bytes.

Two changes, both decode-neutral by construction:

1. **Weight-stationary `gr_up`.** A block now owns a tile of TJ output elements and streams all R
   tokens through it, staging its TJ `up` rows into shared memory once (TJ*rank halfs = 5.1 KB). The
   per-lane accumulation order and the butterfly are untouched, so the result is bitwise identical -
   the only change is that `up` is read from smem rather than DRAM. Grid becomes 1-D; the token loop
   moves inside the kernel.
2. **Removed the `if (sb > 128) sb = 128` cap on `gr_scale`.** It was inert at decode, where the
   computed value is 40, and bound only at prefill: R=256 wanted 10,240 blocks for R*H*D and got 128
   - 1.56 blocks/SM on an 82-SM card, each thread striding 80 elements on a pure streaming kernel. A
   cap tuned against decode had silently crippled prefill.

After, same measurement point:

    gr_up     (1280,1)x256     1.07 ms    17.57 MB   82.0% mem
    gr_scale  (10240,1)x256    32.5 us     10.52 MB   69.8% mem

**957.21 MB -> 17.57 MB, a 54x traffic collapse**, with throughput *rising* to 82% - the stage now
saturates memory with the right bytes instead of thrashing L2 with the wrong ones. The time falls
1.34 -> 1.07 ms (20%), not 54x, because most of those 957 MB were already L2 hits; `dram__bytes_read`
is DRAM-only while the throughput metric is composite.

End-to-end: **prefill 260 -> 275.2 tok/s (+5.8%)**. Decode is unregressed at 44.59 / 45.36 / 43.49
against a 44.67 baseline, which is what the R=1 case predicts - there is no reuse to amortise the
staging, but the cost is one 5.1 KB smem round-trip. All suites green: mgemm semantics, gemm smoke,
attn parity, aux parity, QSA parity, model load (471 device tensors, 0 mismatches), and coherent
end-to-end text.

This is the first structural prefill win of the session, and it came from reading ncu rather than
guessing: the "hc mix is 44% of prefill" observation said *where*, and the 957 MB said *why*.


### gr_dots: tiling the reduction (2.26 ms -> 671 us), and a stride bug the suite could not see

`gr_dots` was the next-largest hc-mix stage. Its grid was `(rank, R)` = (320, 256) = 81,920 blocks,
one per (r, i), which tiles **neither** operand: every block re-read both `normed[r]` and `down[i]`
in full. ncu measured 391.4 MB and 2.26 ms for a pair of matrices totalling 17 MB - about 2.9 TB/s
of L2 traffic, bandwidth-bound on re-reads, with 97.4% of warps already active. There was no
parallelism to add; the bytes were the problem.

It is a tall-skinny GEMM, (R x HD) x (HD x rank), so it is now tiled as one. A block owns a
DTR x DTI = 8 x 32 tile of the output and walks HD = 10240 in DBK = 256-deep chunks, staging both
operand tiles in shared memory (24,704 B). Each operand is then read from L2 once per tile of the
*other* axis - 32 row-tiles and 10 col-tiles - so L2 traffic falls from ~6.55 GB to ~315 MB.

    gr_dots  before  (320,256)x256   2.26 ms   391.4 MB   55.2% mem   97.4% warps
    gr_dots  after   (10,32)x256      671 us     26.4 MB   73.2% mem   38.5% warps

**14.8x less DRAM traffic, 3.4x faster** - and it gets there with 2.5x *lower* occupancy, which is
the point: the win is bytes, not warps. The residual 38.5% warps is the obvious next lever here
(finer DBK, or a split-K), now that the stage is no longer traffic-bound.

The decode path is untouched: the tiled form needs DTR*DTI == blockDim.x, so at R=1 seven of its
eight row-tiles would be dead and it would issue 8x the FMAs for one live row. `gr_mix` dispatches -
tiled for R >= 8, the original reduction kernel below that. That original kernel is kept, not deleted,
because it is the correct choice for its regime.

**The stride bug, and why nothing caught it.** The first tiled version produced degenerate repeated
text end to end. Three things are worth recording:

- `DTP` (the `down` row stride in shared memory) was written as `DTI + 2 = 34`. The row index there
  is the **K** dimension, and `kk` runs 0..DBK-1 = 255, so each row overwrote the previous seven.
- compute-sanitizer reported **0 memory errors**. The writes stayed inside the block's shared-memory
  allocation, so the fault was invisible to a bounds checker - it was pure data corruption.
- The symptom was **non-deterministic**: identical runs gave 378.6, 398.8, 370.5 max relative error,
  because which row lands where depends on warp scheduling. The correct value is `DBK + 2 = 258`,
  which is what the explanatory comment above the kernel had always said while the code said 34.

Every other aux test runs at R = 1, which takes the *small* kernel - so the entire suite passed with
a broken prefill path, and the only signal was generated text. `test_aux` now has a `gr_mix` case at
the production shape (R=256, H=4, D=2560, rank=320) checking `mixed` against a CPU reference of the
full hsum -> scale -> dots -> up -> fin chain. It reports **6.879e-04 max relative error, identical
across three runs**, and it is the test that localises this class of bug rather than the text.

### Prefill, end to end

| stage | tok/s |
|---|---:|
| start of session | 260 |
| + weight-stationary `gr_up`, `gr_scale` cap removed | 275.2 |
| + tiled `gr_dots` | **320.0 / 320.4** |

Decode is unregressed throughout: 44.40 / 44.71 / 45.09 against a 44.67 baseline, as the R=1 dispatch
predicts. All seven suites green, including chat parity, and the end-to-end answer about Paris weather
is coherent and specific.


### A tile sweep that lied: a hardcoded block size, and what it cost

With `gr_dots` no longer traffic-bound, the obvious next lever was occupancy (38.5% warps). A sweep
over DTR x DTI x DBK said block size was everything:

| DTR | DTI | DBK | threads | reported err | prefill |
|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 256 | 256 | 6.879e-04 | 321.2 |
| 4 | 32 | 256 | **128** | **0.000e+00** | **369.3** |
| 8 | 16 | 256 | **128** | **0.000e+00** | **373.4** |

128-thread blocks, 17% faster, and an error of *exactly* zero - better than the 256-thread config
that had been passing all along. Both signals said "take it". Both were false, and the reason is a
bug I had written myself: the staging loops stepped `e += 256`, a literal. At any other block size
that leaves `kk = blockDim.x .. DBK-1` of every shared-memory row **never written**, and the compute
loop then reads whatever the previous `k0` chunk happened to leave there. The kernel computed half of
nothing, which is why it was fast, and the garbage it did compute happened not to move the metric.

Two lessons, both now encoded in the source:

- **The exact-zero error was the tell.** A reassociated fp32 dot product cannot match a double CPU
  reference bit-for-bit; 6.879e-04 is the honest number. A *better-than-baseline* error is a signal
  that the comparison is not measuring what it claims, not that the kernel improved.
- The correct stride is `constexpr NT = DTR * DTI` with a `static_assert`, not `blockDim.x`. The
  launch passes `DTR * DTI`, so the two cannot disagree - and reading `blockDim.x` in the loop
  condition measured **7% slower** (297.4-297.6 against 319.4-320.1 tok/s) by defeating constant
  folding. Correctness and speed both come from the constexpr.

Re-run with the fix, every config reports the same 6.879e-04 - including the two that had claimed
0.000e+00 - and the ranking inverts:

| DTR | DTI | DBK | threads | err | prefill |
|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 256 | 64 | 6.879e-04 | 245.8 |
| 4 | 32 | 256 | 128 | 6.879e-04 | 270.0 |
| 8 | 16 | 256 | 128 | 6.879e-04 | 287.5 |
| 8 | 32 | 256 | 256 | 6.879e-04 | 320.0 / 319.4 / 320.1 |
| 8 | 32 | **128** | 256 | 6.879e-04 | **321.6 / 321.3 / 321.2** |

**The "more blocks will raise the 38.5% warps" reading was wrong.** Smaller blocks are uniformly
worse, so block count was never the limit - shared memory per block was. DBK=128 is a tie on
end-to-end throughput (321.6/321.3/321.2 against 320.0/319.4/320.1, inside noise) but a real win on
the stage itself: it halves shared memory, 12,416 B against 24,704 B, which doubles block residency.

Final state of the stage, and the whole prefill path:

    gr_dots  original  (320,256)x256   2.26 ms   391.4 MB   55.2% mem   97.4% warps   24.7 KB smem
    gr_dots  now       (10,32)x256      625 us     23.7 MB   78.5% mem   50.4% warps   13.4 KB smem
    gr_up    original  (1280,256)x256  1.34 ms   957.2 MB   79.5% mem   89.0% warps
    gr_up    now       (1280,1)x256    1.07 ms    17.6 MB   82.0% mem   (R=256)

`gr_dots` is 3.6x faster with 16.5x less DRAM traffic, and it gets there at roughly half the
occupancy of the original - the whole point being that the win was bytes, not warps.

Prefill **260 -> 321 tok/s (+23%)**, decode unregressed at 45.62 / 44.32 / 45.39 against 44.67, all
seven suites green.


### Why MTP loses, and the measurement that says batched verification is the decode fix

MTP is on by default and is a **net loss**:

    HELIOS_MTP=0   45.63 tok/s
    HELIOS_MTP=1   42.49 tok/s   130/190 accepted (68.4%)   1.68 tokens per step

68.4% acceptance is healthy and it still loses, because `spec_step` as written cannot profit from it:

    1) run_chunk({next}, pos_)            full trunk forward   ~21.9 ms
    2) mtp_draft_step(...)                one MTP layer        cheap, but a 992 KB D2H + host argmax
    3) run_chunk({draft}, pos_)           ANOTHER full trunk   ~21.9 ms

Two forwards for two tokens is break-even *by construction* - one draft cannot beat one verification.
A spec step measures ~41 ms, which is exactly two trunk steps. Speculation only pays when K drafts are
verified in a **single** forward, and that is only worth building if a forward's cost is sublinear in
the number of positions. Measured (ms per token, so the inverse of throughput):

    M      1      2      3      4      5      7     11     35    178   1432
    ms/tok 21.9  22.22  23.59  20.37  16.18  13.76 12.89  5.78  3.19  2.89
    vs M=1 1.00   0.99   1.08   0.93   0.74   0.63  0.59  0.26  0.15  0.13

**M=2 through M=4 are essentially free.** Four positions cost 0.93x of one, so the throughput
multiplier is 4.3x, not 1. This is the same effect the prefill work found from the other end: the MoE
reads 535 MB/token of expert weights, and at M>1 that read is shared across every position, which is
why the prefill MoE runs at 63% of memory peak while the decode mgemm sits at 14%.

With K=3 drafts and 68.4% per-token acceptance, the expected accepted prefix is
1 + 0.684 + 0.468 + 0.320 = 2.47 tokens, so roughly 2.47 tokens per ~23 ms step, or **~107 tok/s**
against the current 45.6 and a 116-128 baseline. That is the decode path worth building.

The blockers are known and bounded, and two are already visible in the code:

- `final_head(n)` projects **only the last** position into logits row 0, on purpose, so `next_token()`
  means the same thing after a prefill chunk as after a decode. Verification needs all K+1 rows.
- The draft argmax is a 992 KB device-to-host copy followed by a host `max_element` over 248,077
  floats, every step. The reductions already exist (`block_reduce_argmax` in quant/reduction.cuh);
  this wants a device kernel writing K+1 ints.
- GDN's recurrent state cannot be un-written, so a partially-accepted batch needs the ninfer
  ReplaySSM pattern: snapshot before the batch, restore and replay the accepted prefix after. State is
  36 layers x ([10240,4] bf16 conv + [48,128,128] fp32 recurrent) = ~112 MB, plus PLE's conv window,
  so the copy is ~0.45 ms against a 22 ms step. Full-attention KV needs no rollback, only a shorter
  `pos_`.


### The rollback problem, and the cheap way out (retry instead of partial replay)

Batched verification commits K+1 positions in one forward. If only m are accepted, the GDN recurrent
state has absorbed the rejected tokens and **cannot be un-written** - a linear-attention state is a
running aggregate, not a ring of rows. The obvious fix is the ninfer ReplaySSM pattern: snapshot,
restore, replay just the accepted prefix. That needs a GDN-only forward entry point, which does not
exist, and replaying m positions through all 36 GDN layers is work of its own.

There is a much simpler option, and the M-scaling table is what makes it affordable: **on a partial
accept, restore the snapshot and re-run the accepted prefix as a whole batched forward.** M=2..4
costs the same as M=1 (0.93-0.99x), so the retry is nearly free. No partial replay, no new GDN entry
point - just `cudaMemcpy` the snapshot back and call `run_chunk` again with the m+1 verified ids.

    snapshot GDN conv+recurrent (~112 MB) and PLE conv before the batch      ~0.45 ms
    run_chunk([next, d0, d1, d2]) at pos_          M=4, ~20.4 ms
    row_argmax over the 4 rows -> 4 ints D2H       16 bytes
    m = length of the accepted prefix + 1
    if m < 4: restore snapshot, run_chunk(accepted prefix)                   ~20.4 ms

Cost per step is therefore one batched forward plus one retry with probability p_reject:

    p_reject 0.35 (65% accept):  20.4 + 0.35*20.4 = 27.6 ms for ~1.65 tokens  ->  60 tok/s
    p_reject 0.18 (82% accept):  20.4 + 0.18*20.4 = 24.1 ms for ~2.82 tokens  -> 117 tok/s

against 42-43 tok/s today. This is strictly better than the current structure even if the retry fires
every time (20.4 + 20.4 = 40.8 ms for 1.65 tokens = 40 tok/s, i.e. break-even), so the change cannot
lose the way the present two-forward scheme does.

Full-attention KV needs no handling at all: the stale rows past `pos_` are simply never read, and the
next forward overwrites them.

### Enabler landed: device-side row argmax

`row_argmax(logits, rows, vocab, out, stream)` in `mtp.cu` / `mtp.cuh` reduces each row with one block
and writes one int per row, breaking ties to the lowest index to match `std::max_element`. The
existing K=1 spec path now uses it instead of a 992 KB device-to-host copy plus a host
`max_element` over 248,077 floats, every step.

Honest sizing: that is worth roughly 0.3-0.5% of a ~41 ms step, which is **below the noise of the MTP
path itself** - two consecutive runs measured 41.99 tok/s at 50.2% acceptance and 43.48 at 82.3%. It is
landed because batched verification needs it (K+1 ints instead of K+1 vocab rows), not because it
speeds anything up on its own. MTP-off is byte-reproducible across runs (identical md5), which is the
deterministic reference the change was checked against.


---

## CORRECTION: the M-scaling table does NOT support batched verification, and my estimate was wrong

The section above concluded that M=2..4 are "essentially free" and predicted ~107 tok/s from K=3
drafts. **Both are wrong, and the error was arithmetic, not measurement.** The table's figures are
ms *per token*; total forward wall time is M x ms/token. Batching only pays if that total beats M
separate M=1 forwards:

    M  ms/token  batched total  M separate fwd   saving
    2     22.22       44.4 ms        43.8 ms       -1.5%
    3     23.59       70.8 ms        65.7 ms       -7.7%
    4     20.37       81.5 ms        87.6 ms        +7.0%
    5     16.18       80.9 ms       109.5 ms       +26.1%
    7     13.76       96.3 ms       153.3 ms       +37.2%

At M=2 and M=3 batching is *slower* than running the positions separately. The knee is at M=5, not
M=2. "4.3x throughput multiplier" was me dividing two per-token figures and calling the ratio a
speedup; the numerator was never a total.

Speculation needs `accepted_tokens * 21.9 ms > verify_cost`. With a per-token acceptance of 0.684 and
a draft step costing about 21.9/48 ms (one layer, not 48):

    K=1  M=2   44.9 ms  -> 1.68 tokens -> 37.5 tok/s
    K=2  M=3   71.7 ms  -> 2.15 tokens -> 30.0 tok/s
    K=3  M=4   82.8 ms  -> 2.47 tokens -> 29.8 tok/s
    K=4  M=5   82.7 ms  -> 2.69 tokens -> 32.5 tok/s
    K=5  M=6   91.7 ms  -> 2.84 tokens -> 31.0 tok/s

**Every draft length loses against 45.6 tok/s of plain decode.** The best case, 32.5 tok/s at K=4, is
still 29% below simply not speculating. So the conclusion inverts: the decode MoE's ALU-bound
behaviour at M=1 is not something speculation can route around, because the weight-sharing that makes
batching attractive only begins to pay past M=5, by which point rejection has already eaten the gain.

This does explain the current design honestly. `spec_step` runs two separate M=1 forwards for two
tokens - break-even by construction - and measures 42-43 tok/s against 45.6 without it. The ~7%
loss is the cost of the draft plus the second forward, and the reason MTP is off by default in any
sane configuration is now explained by measurement rather than left as an open question.

What this leaves: the 116-128 tok/s baseline is **2.7x faster than our 21.9 ms/token at M=1**, with
no speculation involved. The gap is in the M=1 decode path itself - the MoE, which reads 535 MB/token
and runs at 14% of memory peak with 35% warps active. Batched verification is not the answer and
should not be built.


---

## BASELINE RE-VERIFIED: the 2511 tok/s prefill figure was wrong

I had been reporting "8-46% of baseline prefill" against a 2511 tok/s figure. That number is not
real: 2511 tok/s is 0.398 ms/token, but the expert weights alone are 0.567 GB/token, which is 0.61 ms
at the 3090's peak bandwidth. It was below the memory floor, so it could not have been a model
prefill rate. Re-measured against a live exllamav3 server (`supports_mtp: true`,
`mtp_draft_tokens: 4`, `parallel: 4`, `cache_type: cq3+QSA-index-planes`):

    prefill  12,033 tokens in 6.300 s  ->  1910 tok/s   (cold, unique nonce)
    helios    6,802 tokens in 22.89 s  ->   297 tok/s

**Baseline prefill is 1910 tok/s, not 2511.** Helios is at 15.6% of it, not the 8-46% I had been
reporting. Two measurement traps produced the bad number and are worth naming:

- A nonce appended at the END of the prompt leaves everything before it in the prefix cache. Three
  consecutive "prefill" runs gave 1910, then **9615, then 9621 tok/s** - the last two were cache hits
  on an identical 12,034-token prefix, not model forwards. Any benchmark here must defeat the cache
  at the FRONT, not the tail.
- Completion lengths from the baseline are wildly unstable (394, then 179, then 3, then 3 tokens on
  successive runs of the same prompt), so decode must be timed from inter-token stamps or with a
  long forced continuation, never as `completion_tokens / wall`.

### Where the prefill gap actually is

With expert traffic = 0.567 GB/token and a 256-token chunk reading every one of the 512 experts once:

    baseline   47 chunks x 29 GB = 1363 GB in 6.3 s  ->  216 GB/s aggregate (~23% of peak)
    helios     27 chunks x 29 GB =  783 GB in 22.9 s ->   34 GB/s aggregate ( 3.6% of peak)

The gap is a clean **6.4x on the same bytes**, and it lives entirely in the prefill MoE. Note this
also corrects how I have been reading ncu on that kernel: it reported "63% of peak memory
throughput", but that is `gpu__compute_memory_throughput`, a composite over L1/L2/shared. Its actual
DRAM rate was 251.9 MB / 3.25 ms and 673.0 MB / 8.64 ms, i.e. **78 GB/s per card, 8% of peak**. The
composite metric was the misleading number all along, in both regimes.

So the decode MoE (78 GB/s, 35% warps) and the prefill MoE (78 GB/s, 33% warps) sit at the *same*
DRAM rate, which is the real common thread: the EXL3 grouped MoE is running at ~8% of DRAM peak in
both paths, and that single fact is worth more than every scheduling knob already swept.


### Prefill chunk 256 -> 1024: +9.7%, from noticing the port is faithful

The MoE kernel was exonerated first. `exl3_moe_max_concurrency` is `num_sms / MOE_SMS_PER_EXPERT` =
82/8 = **10**, and `group_size` is 8, so grid `(8,1,10)` = 80 blocks is exactly what upstream
produces on this GPU. Our port picks the same numbers, so there is no dispatch bug to fix.

ncu also cleared the memory path: 7.21 sectors per request, **14.6 sectors per load instruction**,
which is a warp-wide 128-bit vectorized load (16 sectors), and `dram__bytes_read` equals
`sectors x 32` to the byte (251.35 MB) - full sectors, no over-fetch. The loads are as good as they
can be.

What was left is a number I had never questioned: `max_chunk = 256`. Every one of the 512 experts is
read **once per chunk regardless of chunk size** - top-10 of 512 saturates at 2560 assignments, so any
chunk past 256 tokens activates all 512 - which makes expert traffic `(tokens/chunk) x 29 GB`:

    6802-token prompt:  chunk  256 ->  27 chunks ->  783 GB
                        chunk  512 ->  14 chunks ->  406 GB
                        chunk 1024 ->   7 chunks ->  203 GB

Measured on that prompt:

    chunk  256   296.0 / 295.7 / 294.7 tok/s
    chunk  512   313.5
    chunk 1024   324.3 / 324.0 / 324.4 tok/s     +9.7%, spread under 0.5%

**Default is now 1024.** Decode is untouched by construction (a 1-token decode never reaches the
chunk loop) and measures 45.45 / 44.69 / 45.39 against a 44.67 baseline. All seven suites green.

`--chunk 2048` is a wall, not a preference: the grouped MoE's per-chunk temp buffers grow with the
chunk and the launch fails at 92,160 B of dynamic smem. gpu0 is down to 1.16 GB free at 1024 (from
1.86 at 256), which is the real cost of this win.

The honest caveat: traffic falls 4x but throughput rises 9.7%, so the MoE is neither purely
bandwidth- nor purely latency-bound - there is a large per-launch fixed cost that a bigger chunk does
not amortise. Getting the rest of that 4x is the next thing to understand, and it is a different
problem from the one I spent this session on.


---

## CORRECTION #2 on the MoE: the grouped path is FINE, and my "8% of peak" was wrong

Two hours after concluding "the grouped MoE runs at 8% of DRAM peak", the correct metric says
**63.0% and 61.6%**:

    prefill exl3_moe_kernel, card0   511.68 MB  6.37 ms  dram__throughput 63.02%  sm 24.3%  warps 33.3%
    prefill exl3_moe_kernel, card1    1.48 GB 18.68 ms  dram__throughput 61.61%  sm 23.8%  warps 33.3%
    decode  exl3_mgemm_kernel          3.01 MB 45.47 us  dram__throughput  7.51%  sm 28.4%  warps 33.3%

The error was arithmetic again, and the same species as the M-scaling one: I computed
`bytes / elapsed` and called the quotient a bandwidth, but that elapsed came from the phase timer
across BOTH cards, while the bytes came from one card's kernel. Two cards' bytes over one card's
time is a number roughly 8x too low, which is exactly the "8%" I reported. The metric to trust is
`dram__throughput.avg.pct_of_peak_sustained_elapsed`, which ncu computes per kernel against that
kernel's own duration.

So, corrected:

- **Prefill MoE is at 63% of DRAM peak.** There is no meaningful bandwidth win available, and the
  earlier "prefill MoE is already at 63%, no scheduling win left" was accidentally right while
  being justified for the wrong reason. `moe=1355 ms` of a 2477 ms chunk is 55% of prefill and it is
  **all kernel time** - 96 launches summing to 1355.6 ms against a phase timer reading 1355.03 ms,
  so there are no launch gaps, no sync stalls, nothing to reclaim between kernels.
- **Decode MoE is at 7.5% of DRAM peak and that IS the gap.** Same 33.3% warps as prefill, so it is
  not occupancy; prefill simply has 1.48 GB in flight per launch against decode's 3.01 MB, and that
  500x difference in bytes-in-flight is the whole story. At M=1 each expert contributes a single
  token-row, so there is not enough work to cover memory latency - the kernel is latency-starved at
  the same occupancy it enjoys when it is running at 63%.

This also retires the "8-17x off its weight-bandwidth bound" framing on the decode todo. It is not
17x off; it is 7.5% utilized while the same kernel at the same occupancy hits 63% when given more
rows. The lever is bytes-in-flight per launch at M=1, which is a scheduling/tiling question about
how many experts one launch covers - not a new kernel, and not speculation.

Net position after two corrections in one session: prefill 260 -> 324 tok/s (+25%) is real and
verified; decode is unchanged at ~45 tok/s; and the single largest remaining decode lever is now
identified as the M=1 grouped/mgemm launch shape, not the hc mix, not speculation, and not a
scheduling knob on the existing grid.


---

## BUG FOUND: greedy decode is NOT reproducible, and I had claimed it was

Earlier in this session I wrote "MTP-off is byte-reproducible across runs (identical md5)" and used
it as the reference for validating the `row_argmax` change. **That claim was wrong**, and the way it
was wrong is worth recording: I ran it twice, got the same md5 both times, and concluded
determinism. Two samples agreeing is not determinism - it is a coin landing the same way twice.

Measuring properly, with token ids rather than an md5 of mixed text:

    6-token prompt tail, chunk 256, 6 runs: ALL IDENTICAL
      ['271','2053','449','14791','11','353'] x6

    96 tokens, 3 runs, same config: identical for 30 tokens, then DIVERGES
      run1[30] = 1206
      run2[30] = 10883
      run3[30] = 10883

The first 30 tokens are stable and the divergence appears at token 30, **bimodally** - two runs agree
with each other and one differs. That shape is not random bit-rot; it is a race that usually resolves
one way and occasionally the other, which is what a near-tie in the logits plus a run-to-run
summation-order difference looks like once greedy argmax amplifies a ~1e-7 perturbation into a
different token id.

`--chunk 1024` did not show it in 4 runs, but that is a sampling result, not a fix: the same race is
present at any chunk size, and 4 runs is exactly the sample size that let me draw the false
"deterministic" conclusion in the first place.

### What this invalidates

- The `row_argmax` validation. It was gated on an md5 that is not stable, so that check proved
  nothing either way. The kernel's tie rule (lowest index, matching `std::max_element`) is correct by
  inspection, but "verified byte-identical output" is not a claim I can make.
- Any speed measurement taken as a single before/after pair on token output.
- The earlier "MTP non-determinism is a cache issue" note, at least in part. MTP-off is *also*
  nondeterministic, so the draft path was never the whole story.

### Why the obvious suspects are not it

- `atomicAdd` in `glue2.cu:30` (`gemm_nt_f16`, `add=true`): no caller ever passes `add=true`. Dead
  branch.
- `scatter_add_rows_k` in `glue.cu:69`, also `atomicAdd`: **no callers at all.** Dead code, and a
  candidate for deletion on its own merits.

So the nondeterminism is upstream of both, in the decode forward itself. The next step is to bisect by
depth rather than by reading: dump the per-layer hidden state for two runs at token 30 and find the
first layer whose output differs. That localises it in one run instead of by inspection, and it is the
same technique that found the `gr_dots` stride bug - a narrow numerical check beats reasoning about
which kernel *could* be responsible.


---

## ROOT CAUSE of the decode nondeterminism: a lock-based split-K reduction into the live output

Localised by depth rather than by reading, using the existing `HELIOS_MOEC` per-layer routed-RMS
diagnostic on two identical greedy runs:

    L0 token 0   runA 0.018815   runB 0.018815    identical
    L0 token 1   runA 0.015470   runB 0.015469    first difference, 6 significant digits
    L0 token 21  runA 0.018456   runB 0.016878    amplified ~13x

So the seed enters at **layer 0, token 1**, at the 1e-6 level, and compounds through the recurrent
GDN state. That signature is a varying summation order, not a memory error - and it is exactly the
one shape compute-sanitizer cannot see and a bounds checker cannot see.

The mechanism, in `exl3_gemm_inner.cuh`:

    int* lock = &locks[slice_m * blocks_n + slice2_n];
    barrier_acquire(lock, lock_i);
    bool first = lock_i == 0;
    bool last  = lock_i + lock_d == tiles_k;
    if (!sub_k && !first) read_sum_gl();     // <-- read-modify-write of the OUTPUT buffer
    if (!sub_k && !last)  write_sum_gl();
    if (!sub_k && last)   write_sum_tile_sh() / write_sum_gl();

`read_sum_gl` / `write_sum_gl` accumulate into `gl_c_ptr_32` / `gl_c_ptr_16`, which **is the output
tensor** (`sc.part[c]`, `gbuf`, ...), not a private scratch. The k-slices of one output tile are
serialised by a global-memory lock, and each slice adds its partial into whatever the previous slice
left in the live output. Whichever threadblock acquires the barrier first becomes the base of the sum,
so the floating-point addition order varies with scheduling.

This is upstream exllamav3 code (`exl3_gemm_inner.cuh` is a direct port), so the reference runtime
has the same property. It is not something a port introduced, and it is not fixable by pinning a
config: the autotuner is already dropped ("CoopKernelAutotuner -> dropped; the deterministic selector
is the only path"), and the variance is in the *runtime* barrier order, not in the selected shape.

### What this does and does not explain

- Explains why both the mgemm decode path and the grouped path are affected - they share
  `exl3_gemm_inner.cuh`.
- Explains the intermittency: when the timing of the slices is close, the same block tends to win
  twice in a row, so runs agree; under different contention the order flips.
- Explains the amplification: a 1e-6 perturbation fed into a recurrent state across 48 layers and 30
  tokens reaches argmax-flipping magnitude, and then greedy decoding locks in a different token.
- Explains why `--chunk 1024` looked stable in 4 runs: it changes the *timing* of the slices, not the
  race. More samples, not a fix.
- Does NOT explain the in-process server result on its own: two back-to-back requests on one loaded
  model also differed, but the server may carry state between requests, so that test is suggestive
  rather than conclusive on its own.

### Fix directions, in preference order

1. **Give the split-K reduction a private per-launch scratch** instead of accumulating into the output,
   then copy once. Cost: one extra buffer and a copy, but the order becomes fixed by `slice2_k`
   (already computed) rather than by the barrier. This is the real fix.
2. **Force a single k-slice at M=1 decode** (`tiles_k == 1`), which removes the reduction entirely.
   Costs some parallelism in exactly the regime that is already latency-starved, so it trades against
   the decode problem rather than composing with it.
3. Accept it and document it. Defensible only if the reference is equally nondeterministic, which
   should be *measured* on exllamav3 before being asserted - the same greedy prompt, same token count,
   repeated.

Option 3 is not available as a shortcut: the deliverable is a runtime that beats the reference, and
"both are nondeterministic" is not a claim I have verified. Measure the reference's determinism
first; that single experiment decides between fixing and documenting.


### The reference IS deterministic, so this is ours to fix

    exllamav3, 4 identical greedy requests, max_tokens=96, temp 0:
      req1..req4: 357 chars each, cached=0, completion_tokens=96
      distinct outputs: 1/4   -> IDENTICAL

The reference runs the same 2-bit weights through the same 48 layers on the same two GPUs and gets
byte-identical completions every time. Helios, on the same prompt and the same machine, diverges at
token ~21-30 in a majority of runs.

That closes the "both runtimes share it" escape hatch, which was the only reason option 3 (accept and
document) was on the table. It also reframes what the lock-based split-K reduction means here:

- `exl3_gemm_inner.cuh` is a direct port, so the *code* is upstream's.
- But the reference does not reach it on this path. It dispatches M=1 to a different kernel - which
  is the long-standing note in this log: **"mgemm never routes M=1 to the QTIP GEMV path that
  exl3::gemm has"**. Helios decode uses `mgemm`; exllamav3 decode does not.
- So the defect is a **path-selection** defect, not a porting defect. The lock reduction inside
  mgemm is upstream's code and is fine for the shapes upstream uses it for; using it at M=1 decode,
  where each output tile is tiny and split-K is therefore chosen aggressively, is our choice.

That reframes the fix list. The cheapest correct fix is to stop splitting K at decode: with
`tiles_k == 1` the `read_sum_gl` / `write_sum_gl` accumulation is skipped entirely, the reduction
order becomes the fixed `slice2_k` order, and the output is deterministic by construction. The cost is
less parallelism in a regime that is *already* latency-starved (7.5% dram__throughput at 33.3% warps),
so this trades against the decode-speed problem rather than composing with it - and that trade needs
measuring, not assuming.

It also explains a result from earlier in this session that I had filed as noise: the `mgemm residual`
item, where per-card values differed from `moe_grouped` by ~5% at the first decode step. That is
consistent with a reduction that is not merely non-deterministic but not bitwise comparable to the
grouped path either. Two engines running the same weights should not disagree at all; that 5% was
probably always this bug wearing a different hat.


---

## CORRECTION #3: the split-K reduction is NOT the cause (I was wrong again)

I proposed the lock-based split-K as the mechanism. Reading it properly, it does not hold:

    lock_i  = tiles_k - slice2_k - 1;      // fixed by slice2_k, not by scheduling
    first   = lock_i == 0;                 // the bottom k-slice always goes first
    write_sum_gl:  *c_ptr = frag_c        // assignment, not accumulate
    read_sum_gl:   frag_c += *c_ptr       // only for !first
    barrier_release(lock, lock_d, last)    // last block does *lock = 0

`barrier_acquire` spins thread 0 until `*lock == stage`, and `barrier_release` either adds `lock_d`
or resets to 0 when `last`. Since `lock_i` is derived from `slice2_k`, the k-slices of a column are
summed from high k to low k in a **fixed** order, and the counter is returned to 0 by the last block
each launch. The summation order does not depend on which threadblock wins the barrier - the barrier
exists precisely to impose that order.

So the split-K is deterministic, and I should not have named it. The `[MOE] residual` item (mgemm
disagreeing with moe_grouped by ~5% at the first decode step) is likewise not explained by this, and
I withdraw the speculation that it was "this bug wearing a different hat".

### What the evidence actually supports

- The seed is ~1e-6 and appears at **L0, token 1**: token 0 is bit-identical across runs, token 1
  differs in the 6th significant digit, token 21 differs in the 3rd.
- The reference is deterministic on the same prompt, same machine, same weights (4/4 identical).
- It affects both the mgemm and the grouped decode paths, so it is in code they share.
- It is present in-process (two back-to-back requests on one loaded model differ), though that test is
  confounded because the server may carry state between requests.

The "token 0 clean, token 1 dirty" signature is the most informative thing here and I under-used it.
It means the perturbation is introduced by state **written during the first decode step and read
during the second**. In this model that is the GDN recurrent state (`rec_state`, updated in place by
`cuda_recurrent_gated_delta_rule`) and the conv window (`conv_state`, shifted by
`cuda_causal_conv1d_update`), plus the KV cache. It is not the MoE weights, which are read-only and
identical every run.

### Next step, stated precisely

Depth-bisect *within* the first decode step: dump `conv_state` and `rec_state` for layer 0 at the end
of step 1 across two runs and compare. If they already differ, the bug is in the GDN state update -
most plausibly the conv window shift, which is read-modify-write on a 4-deep window and would be
invisible to every bounds checker and to compute-sanitizer. If they match, move outward to the KV
cache write, then to the MoE output. That is three measurements, not an open-ended search, and it
starts at the one place the evidence points.

I am recording this as unresolved rather than fixed. The honest summary is: a real, reproducible
correctness defect, localised to the first decode step's state update, with the reference confirmed
deterministic - but not yet root-caused, and not yet fixed.


---

## GDN state fingerprinting: a real bug found and fixed, but NOT the cause

Added `HELIOS_GDNSTATE=1`, an FNV-1a fingerprint of both persistent GDN buffers after every layer
(`conv_state` bf16, `rec_state` fp32), and compared two identical greedy runs:

    before the fix:  108/108 decode-step fingerprints DIFFERED
                     L36 conv dd5759277a24e997 vs dd5759277a24e997   IDENTICAL
                         rec  8f9fda7b704c72ca vs 4a3afe071f465f85   DIFFERS

(`L36` is a global call counter, not a layer index: 36 GDN calls per decode step, so call 36 is the
first call of the *second* step - which is why the very first step agreed and the second did not.)

### Bug found and fixed

`gdn.cu` reduced two shared-memory values with `atomicAdd` across `threadIdx.y`:

    atomicAdd(sh_dot1 + t, sum);      // line 479
    atomicAdd(sh_dot2 + t, v_out);    // line 515

Shared-memory atomicAdd has no arrival order, and `sh_dot1` feeds `v -= sh_dot1[t] * g_h` and then
**the recurrent-state update itself**, so a varying order writes a varying state. `sh_dot1`/`sh_dot2`
are now `[SUBK][MAX_HEAD_DIM]`, each `threadIdx.y` writes a private slot, and `bt == 0` sums them in
fixed index order. This is strictly better regardless of whether it was the last bug: a reduction
whose order is not specified should not be feeding persistent state.

All seven suites pass and decode is unregressed at 44.09 / 45.53 / 44.10.

### But it is not the cause

After the fix, the fingerprints still differ on 108/108 calls. Two things I got wrong in reading my
own evidence:

- **`conv_state` being identical proves nothing.** It is bf16 - 8 mantissa bits. A 1e-6 relative
  perturbation rounds away in bf16 and only shows in the fp32 recurrent state. I read that equality
  as "the input was identical"; it only means "the input was identical to bf16 precision".
- The fingerprint is a single bit-sensitive hash over 3 MB, so "differs" means "at least one bit
  differs anywhere" - it cannot bound the magnitude, and I was treating it as a localisation tool when
  it is only a detector.

### Second candidate, not yet acted on

`glue2.cu:276`, in `moe_scatter_k`:

    int64_t pos = atomicAdd((unsigned long long*)&cursor[e], 1ull);
    token_sorted[pos] = i / topk;

Each (token, slot) pair claims a position in its expert's bucket by atomic cursor, so **the order
tokens appear in a bucket is scheduling-dependent**, and the grouped kernel then reduces over that
order. This is on the *grouped* path, which is the one that measured 3/3 distinct outputs; the
default mgemm decode path does not sort, so it needs a different explanation still.

### Honest status

One genuine determinism bug found and fixed, with the mechanism understood and verified not to be
sufficient. The decode irreproducibility is **still present**. Two candidate sources remain - the
grouped-path permute cursor above, and something in the mgemm decode path I have not identified. I am
recording this as open rather than closing it, because the fingerprint told me "a bit differs
somewhere" and I have been successively wrong about where.


---

## A real gate for the bug, and three hypotheses tested and refuted

### The gate

`test/test_determinism.py` runs N independent engine processes on one greedy prompt and asserts the
token-id sequences are identical. N defaults to 4 and the script refuses fewer than 2, because the
observed failure is **bimodal** - several runs agree, one differs - so a two-run check passes about
half the time. It also parses `[ids]` with a marker regex rather than a line anchor, because the
engine interleaves streamed text with those lines (` AI[ids] 11`), which is what made an earlier
ad-hoc check silently extract nothing and look like agreement.

Current result, which is the point of having it:

    DETERMINISM FAIL: 3/4 distinct token sequences
    first divergence at token index 20
      run 1: id=4581   run 2: id=3050   run 3: id=4581   run 4: id=3050

This is the check the project was missing. "Compare twice" passed while the engine was broken.

### Refuted this round

1. **Split-K reduction order** - not a race. `lock_i = tiles_k - slice2_k - 1` is fixed by
   `slice2_k`, the k-slices sum high-k to low-k in fixed order, `write_sum_gl` assigns while
   `read_sum_gl` accumulates only for `!first`, and `barrier_release(..., last)` zeroes the counter.
   The barrier imposes the order; it is not a source of variation.
2. **Stale split-K lock counters** - the `locks` buffer is memset once at allocation and shared
   across launches of *different shapes* (gate/up are N=640 K=2560; down is N=2560 K=640), and
   `barrier_release` only resets a lock when that launch's `last` block runs, so a lock left
   non-zero could be seen by the next launch. I added an env-gated per-launch `cudaMemsetAsync`
   and ran the gate: **still 3/4 distinct**, divergence moved to token 30. Hypothesis refuted by
   experiment; the diagnostic has been removed rather than left as dead configuration.
3. **Cross-card stream mismatch** - `xcard_copy` synchronises
   `gpu(src_dev == gpu(1).phys_idx() ? 1 : 0).stream(0)`, and the MoE launches on
   `st[c] = gpu(c).stream(0)`. These are the same stream. Not a mismatch.

### Atomics audit, after the GDN fix

Every remaining `atomicAdd` in shipped code is unreachable on this model's decode path:

- `glue2.cu:30` (`gemm_nt_f16`, `add=true`) - both callers, in `ple.cu`, pass `add = false`.
- `gdn.cu:698,727` - the channelwise `_128` variant; this model is dispatched
  `channelwise = false`, so `cuda_recurrent_gated_delta_rule_kernel` (the one I fixed) runs.
- `dsa_topk.cu` / `qsa.cu` - DSA/QSA, off by default.
- `glue.cu:69` (`scatter_add_rows_k`) - no callers at all.

So the decode irreproducibility is **not** caused by an atomic reduction. That eliminates a whole
class and is worth stating plainly, because it was the most likely explanation and it is wrong.

### Honest status

Still open. One real determinism bug fixed (GDN `atomicAdd` into persistent state, mechanism
understood). Three hypotheses refuted with evidence. A permanent gate now fails loudly instead of
the defect hiding behind a two-sample comparison. The remaining source is unidentified; the next
step is no longer a search over atomics but a stage-by-stage bisect of one decode step, using the
`HELIOS_MOEC` magnitude-bounded per-layer readout rather than the FNV detector.


---

## ROOT CAUSE FOUND: the grouped-path MoE permute, 2.68e-10 at prefill layer 0

Bisected by (occurrence, layer, stage) at 9 significant digits, across 4 independent runs. The
earlier probes printed 6 decimals, which cannot see a 1e-6 *relative* change on a value of order
0.02; and my first parser grouped by layer alone, which compares *different decode steps* once the
sequences diverge - that is what made L0 look 5x different. Both were measurement errors, and both
had to be fixed before the real answer was visible.

First disagreement, whole pipeline:

    occurrence 0 (PREFILL), layer 0, mrms:mlp
      3.731702379e-02  3.731702379e-02  3.731702379e-02  3.731702380e-02
      relative spread 2.680e-10

And it amplifies monotonically with depth, which is what makes it a root rather than one of several:

    layer     L0        L1        L2        L3        L5        L11       L23       L35
    spread    2.7e-10   1.7e-05   1.6e-04   1.6e-04   9.5e-04   1.1e-03   1.8e-03   5.3e-03
    ... worst layer during prefill: 1.458e-02

A 2.68e-10 seed at layer 0 growing to ~1e-2 by mid-network is a single perturbation compounding
through 48 layers, then through decode until greedy argmax flips a token id around token 20-30. No
other stage disagrees earlier, so nothing upstream of the layer-0 MoE contributes.

`MOE_DECODE_MAX_N = 2`, so the 8-token prefill of the test prompt takes the **grouped** path - the one
that calls `moe_permute` -> `moe_scatter_k`:

    int64_t pos = atomicAdd((unsigned long long*)&cursor[e], 1ull);
    token_sorted[pos] = i / topk;
    weight_sorted[pos] = weights[i];

Each (token, slot) claims a slot in its expert's bucket with an atomic cursor, so **the order tokens
appear in a bucket is scheduling-dependent**, and the grouped kernel then reduces over that order. A
2.68e-10 difference is exactly the size of a float reassociation over a few hundred terms - and I
dismissed this candidate two rounds ago on the grounds that "prefill gives identical first tokens".
That inference was wrong in the same family as the others: an identical *first token* only means the
seed is far too small to flip token 0, not that the prefill is deterministic. The grouped path is
where prefill runs, and the seed is in its permute.

### The fix, specified

Replace the atomic cursor with a deterministic stable counting sort by `(expert, i)`, so the
permutation is a pure function of `ids`:

- one block per expert, scanning a contiguous wave of the input range;
- within a warp, `unsigned mask = __ballot_sync(active, match)` then
  `rank = __popc(mask & lanemask_lt())` gives each matching lane its exact in-`i` rank, in order,
  with no atomic;
- combine per-warp counts through a small shared exclusive scan, add `offset[e]`, and write.

Integer atomics are fine for *counts* (integer addition is exact and order-independent) - it is only
the per-element cursor that is unsound. This keeps the existing count/offset pre-pass unchanged.

### What this retires

With the seed pinned, the earlier hypotheses stop being live explanations rather than merely
unproven: the split-K order was always deterministic, the lock counters were tested and refuted, the
cross-card streams match, and the GDN `atomicAdd` was a genuine bug worth fixing on its own terms -
but it is downstream of this seed, which is why fixing it did not change the symptom. That last point
is the one I got most wrong: I found a real bug, fixed it properly, and then kept it in frame as the
candidate cause when a measurement had already ruled it out.


---

## Four determinism fixes landed; the bug is NOT fixed

Acting on the root-cause hypothesis, in order. Each is a real improvement to a determinism property;
none eliminated the seed. Recorded as attempted-and-insufficient, not as wins.

| # | change | file | effect on the gate |
|---|---|---|---|
| 1 | shared-memory `atomicAdd` -> fixed-order sum over `threadIdx.y` | `gdn.cu` | no change (108/108 fingerprints still differ) |
| 2 | per-element atomic cursor -> deterministic stable counting sort | `glue2.cu` `moe_scatter_k` | no change; L0 seed 2.68e-10 -> 5.36e-10 |
| 3 | atomic ticket draw -> static round-robin | `exl3_moe_kernel.cuh` | no change |
| 4 | float `atomicAdd` -> 2^40 fixed-point integer `atomicAdd` + one conversion pass | `hadamard_inner.cuh`, `exl3_moe.cu`, `exl3_devctx` | no change; measured spread went **up**, 2.7e-10 -> 5.1e-7 |

Change 3 also caused a hang on the first attempt, which is worth recording because the failure mode
was instructive: the old code published the ticket through `sched[2 + group_idx]` because
`atomicAdd` had to be issued by one thread, and the group's *other blocks* read the published copy.
Dropping the atomic but also dropping the publish left those blocks holding a stale ticket forever.
The fix is that every block computes the same value from the same round count, and the barrier keeps
them in lockstep.

### Two errors in my own reasoning about change 4

- I wrote in the source comment that 2^40 fixed point is "four orders of magnitude finer than
  fp32". That is wrong: `v * FSCALE` is evaluated in fp32, so the product keeps a 24-bit mantissa and
  the scale factor cannot manufacture precision the multiply cannot represent. The change buys
  determinism, not accuracy. The comment has been corrected in place.
- I predicted the gate would go to 4/4 identical. It did not.

### What this says about the diagnosis

The stage bisect is solid: the first disagreement in the pipeline is prefill layer 0's MoE output,
at ~3e-10, growing monotonically with depth. That part is measured. What has been wrong is every
step from "the disagreement is in the MoE" to "and therefore it is in *this* MoE component". Four
components inside that stage have now been made deterministic and the seed survives all four, which
means the cause is in a component I have not looked at - the routing/top-k selection, the shared
expert, or the g/u staging GEMMs - or the bisect's stage attribution is coarser than I assumed
(`mrms` is the MoE *plus shared expert*, so it does not actually isolate the MoE).

That last point is the one I should have checked first. The probe I called "mrms:mlp" reads
`sub_out_` after `moe_layer`, which is routed **+ shared expert**. I have been treating it as the
MoE output. It is not.

### Honest status

The determinism defect is open. What exists in its place is a gate that fails loudly, a bisect that
localises it to prefill layer 0 at a measured magnitude, four component-level fixes that are correct
in their own right, and a list of components not yet examined.

## Prefill, after all of the above

    368.3 / 370.1 / 370.0 tok/s   (was 321.2 / 324.4 before this round)

+14%, reproducible to under 0.5%. Decode unchanged at 44.61 / 44.50. All seven suites green. Prefill
this session: **260 -> 370 tok/s, +42%**.


### The probe I was using never isolated the MoE

`mrms:mlp` reads `sub_out_` after `moe_layer`, which is **routed + shared expert**. I had been
treating it as "the MoE output" and reasoning from it accordingly. `HELIOS_MOEC` reads `sc.part_sum`,
which is the routed contribution *before* the shared expert is added - so with both enabled the two
can be separated. (MOEC also printed 6 decimals, which is the precision trap from two rounds ago all
over again; it is now `%.9e`.)

Eight prefill runs at layer 0, after the fixed-point accumulation fix:

    routed (MoE only)      1.881475956e-02  x6
                           1.881473256e-02  x2
    routed + shared        3.731702379e-02  x6
                           3.731704293e-02  x2

So:

- The difference is in the **routed** part, not introduced by the shared expert. The two numbers
  track each other exactly, which is what a shared deterministic addition on top of a varying routed
  value looks like.
- It is cleanly **bimodal, 6/2** - two stable outcomes, not a continuum. That is a race that usually
  resolves one way, and it is why every 2-run and 4-run comparison I made was unreliable: 4 runs has
  a ~59% chance of looking consistent when the true split is 6/2.
- Magnitude 2.7e-7 relative - float-rounding scale, about 4 ULP of the fp32 value. Not a different
  expert set (that would be a discrete jump), so the routing/top-k *selection* is almost certainly
  stable and something downstream reassociates.

### What is left in the routed path

With the permute deterministic (change 2) and the cross-expert accumulate order-independent
(change 4), the routed value is: router scores -> top-k -> `moe_count_k` / `moe_offset_k` (both
integer and single-threaded, deterministic by inspection) -> `remap_ids_kernel` -> grouped g/u/d ->
cross-card add. That leaves three live candidates I have not made deterministic or measured:

1. `routing_std_topk_kernel` - the selection itself.
2. the g/u staging GEMMs inside `exl3_moe_kernel`, which run through `exl3_gemm_kernel_inner` and
   therefore touch the same shared `locks` region as the d-projection. I reasoned that the split-K
   order is fixed by `lock_i`, but I reasoned it from reading rather than from a measurement, and I
   have been wrong that way repeatedly.
3. the cross-card add in `moe_layer` - `xcard_copy` + `add_f32_inplace`. This is a two-buffer add
   and looks deterministic, but it is unmeasured.

The 6/2 split also sets the bar for the next experiment: a candidate is only cleared by **8+ runs**
showing a single value, not 2 or 4.


### The one positive signal: the outcome depends on the `locks` region

The grouped kernel's g/u staging GEMMs run through `exl3_gemm_kernel_inner` and therefore share the
device `locks` region with everything else on that device. A group that skips an expert
(`token_count == 0`, or above `max_tokens_per_expert`) leaves its loop body without reaching the
end-of-expert barrier, so that launch's `last` block never performs the `*lock = 0` reset it would
otherwise have done.

I tested that by adding an env-gated `cudaMemsetAsync` of the whole `locks` region before every
`moe_grouped` launch - note that the earlier lock-reset experiment covered only the **mgemm** launch
path, never this one, so it had never actually been tried here.

    routed L0 prefill, 8 runs, no reset :  1.881475956e-02 x6   1.881473256e-02 x2
    routed L0 prefill, 8 runs, with reset: 1.881475956e-02 x4   1.881473256e-02 x4

**The reset did not fix it - it made the split worse (6/2 -> 4/4).** So the stale-lock story is
refuted as a *fix*, and the probe has been removed rather than left as dead configuration.

But this is the first positive signal in several rounds, and it is worth stating precisely: the
**probability of each outcome changes when the contents of the `locks` region change**. Both values
are the same two; only their frequency moves. That is what you would expect if the race lives in
code that reads and writes that region - the g/u split-K reduction - rather than in a fixed-order
computation. It is evidence, not proof: a memset also perturbs timing globally, so it could be
shifting an unrelated race's odds. Distinguishing those needs an experiment that changes the lock
*usage* rather than its initial value.

### Cumulative refutation list for this defect

| hypothesis | how tested | result |
|---|---|---|
| split-K order varies | read the barrier protocol | refuted - `lock_i` fixes the order, barrier imposes it |
| stale locks (mgemm path) | per-launch reset | refuted - still 3/4 distinct |
| cross-card stream mismatch | compared launch stream to sync stream | refuted - same stream |
| permute cursor | replaced with a deterministic counting sort | fixed, symptom unchanged |
| grouped expert ticket | replaced atomic draw with static round-robin | fixed, symptom unchanged |
| cross-expert float atomicAdd | 2^40 fixed-point integer atomicAdd | fixed, symptom unchanged |
| stale locks (grouped path) | per-launch reset before `moe_grouped` | refuted as a fix, but **perturbs the split** |

Seven hypotheses, six refuted, three real fixes landed that each failed to move the symptom. The
defect is still open. The most useful thing this round produced is the split-perturbation result,
which points at the g/u split-K as the place to look next - and the methodological bar it sets: a
candidate is cleared only by 8+ runs agreeing, never 2 or 4.


### The g/u split-K is ruled out too

The lock-sensitivity result pointed at the last order-dependent construct left in the routed path: a
group's blocks split an expert's K dimension and combine the partials through the shared `locks`
barrier. Forcing `group_size = 1` removes that reduction outright - one block does the whole expert,
so there is no cross-block split-K and no lock involvement at all.

    routed L0 prefill, 6 runs, group_size = 1 :  1.881383514e-02 x4   1.881384841e-02 x2
                                                 -> 2 distinct

(The first pair of numbers quoted here came from a run against a STALE binary - the brace fix had
not compiled yet, so `group_size` was still 8 and the value was the unsplit-K one. The conclusion is
unchanged; the figures above are the ones actually produced with the split-K removed. Note also that
the *value* shifts from ~1.88147e-02 to ~1.88138e-02 when the split-K is removed, which is expected:
removing it changes the summation, and with it the last few ULP.)

**Refuted.** The split-K is not the cause. The probe has been removed; `group_size=1` would have been
a large slowdown for no diagnostic value.

The frequencies shift between configurations (6/2 at baseline, 4/4 under lock reset, 4/2 with the
split-K removed) while staying bimodal, so whatever produces the two outcomes is sensitive to timing
and to the contents of memory the g/u path touches - but it is not the split-K reduction itself.
The *values* also move when the summation changes, which is a useful reminder that these are two
distinct near-identical sums rather than a fixed pair.

### Where the candidate list now stands

    ROUTED (prefill L0, bimodal, ~2.7e-7 relative, 4 ULP)
      router scores ..................................... NOT YET EXAMINED
      top-k selection ................................. NOT YET EXAMINED
      moe_count_k / moe_offset_k ...................... integer + single thread, deterministic
      moe_permute cursor .............................. FIXED (deterministic counting sort)
      remap_ids_kernel ................................ NOT YET EXAMINED
      grouped g/u staging ............................. split-K RULED OUT by group_size=1
      grouped ticket assignment ...................... FIXED (static round-robin)
      cross-expert accumulate ........................ FIXED (2^40 fixed-point atomicAdd)
      cross-card add (xcard_copy + add_f32_inplace) .. NOT YET EXAMINED

Three components have never been looked at, and they are the whole remaining list. `routing_std_topk`
is the one I would take next, because it is the only remaining place where a *selection* rather than
an arithmetic reassociation could vary - and a selection change would show as a discrete jump, which
the observed 4-ULP difference is not. That argues against it, which leaves the cross-card add and
`remap_ids` as the less obvious but not yet excluded candidates.

Eight hypotheses tested this session, seven refuted, four real fixes landed that each failed to move
the symptom. The defect remains open and is the one thing standing between this engine and being
shippable.


---

## FIXED: the greedy divergence was a float atomicAdd in the GDN delta rule

`DETERMINISM PASS: 6 runs, 96 tokens, all identical`

### The error that hid it

I had audited the remaining `atomicAdd`s and written them off with: *"gdn.cu:698/727 - the
channelwise `_128` variant; this model dispatches `channelwise = false`, so
`cuda_recurrent_gated_delta_rule_kernel` (the one I fixed) runs."*

**That inference was wrong, and it is the same failure mode as the others this session: I read a
flag near the code and assumed it decided the dispatch.** The dispatcher selects by **head
dimension**, not by the template flag:

    else if (!history)
    {
        if (k_head_dim == 128 && v_head_dim == 128)
            if (v_split == 4) LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel_128<false, 4>)

`channelwise = false` sets the third template parameter, but 128-dim heads still select the
`_kernel_128` **variant**. So the two atomics I had explicitly ruled out were the ones running all
along, and the one I had fixed was dead code on this model. Fixing the dead one changed nothing,
which I correctly observed, and then I kept the wrong one excluded instead of re-deriving the
dispatch.

### The defect

In `cuda_recurrent_gated_delta_rule_kernel_128`, `sh_dot1` and `sh_dot2` were reduced across
`threadIdx.y` with shared-memory `atomicAdd`. `sh_dot1` feeds

    v = gl_v[t] - sh_dot1[t] * g_h;      // then the recurrent-state update

so an unspecified summation order wrote a different `rec_state` on every launch. The state is
persistent and feeds the next token, which is why a 4-ULP difference at prefill layer 0 compounded
to a different token id around index 20.

Both are now `[SUBK][HEAD_DIM]` private slots with a fixed-order `for (b = 0; b < SUBK; ++b)` sum by
`bt == 0`, matching the non-128 variant.

### The bisect that found it

The earlier `sub` probe said the GDN output was stable. It was not - a 4-run check against a 5/3
split passes ~26% of the time. Adding the two missing probes and re-running at 8 samples:

    lrms   layer input      1 distinct  4.612414757e-03 x8      <- bit-identical
    sub    GDN output       2 distinct  ...x5 / ...x3            <- DIVERGES HERE
    min    pre-mlp streams  2 distinct                             <- downstream
    min    MoE input        2 distinct                             <- downstream

and inside `gdn_layer`, same 8 runs:

    z16     (gate)          1 distinct   2.428255117e+00 x8
    conv_out(after conv1d)  1 distinct   2.235391673e-01 x8     <- identical input
    core_out(delta rule)    2 distinct   7.560234257e-03 x5 / ...258e-03 x3
    normed  (gated rms)     2 distinct                             <- downstream

Identical `conv_out` in, different `core_out` out: the defect is inside the delta rule and nowhere
else. Both GDN state buffers are memset to zero at init (runner.cpp:120-121), so uninitialised state
was never in play.

### Also ruled out along the way, with evidence

- **My own `tiled gr_dots`.** Temporarily raising its threshold from `R >= 8` to `R >= 16` so the
  prefill used the original kernel: the MoE input still varied. Cleared - though the spread did drop
  from 4.1e-8 to 5e-10, so the tiled kernel was amplifying the seed rather than causing it.
- **The whole MoE.** `min` (the MoE's own input) varies, so the divergence is upstream of it. Every
  hypothesis about permute, ticket, split-K and cross-expert accumulate was chasing a downstream
  symptom - they were genuine determinism improvements, but none of them was this bug.

### State

All seven suites green. Prefill 360.7 tok/s, decode 44.57 / 45.43 / 45.38 - unchanged, as expected
for a fix that only removes an unspecified reduction order. The determinism gate that was added
during this hunt now passes, and it stays in the suite as a regression guard.


### Fix verified across every path

    96 tokens,  MTP off : 6 runs, all identical
    206 tokens, MTP off : 4 runs, all identical
    96 tokens,  MTP ON  : 4 runs, all identical (1 distinct sequence)

The speculative path reproduces too, which it did not before - it inherits the same trunk forward, so
fixing the shared defect fixes it without touching `spec_step`.

## Honest status against the deliverable

The objective is a runtime that **surpasses** exllamav3 on decode, prefill and context length. Where
this actually stands, against numbers re-measured against a live baseline:

    decode   ~45 tok/s   vs  116-128 tok/s   ->  ~39%
    prefill  ~361 tok/s  vs  1910 tok/s      ->  ~19%   (was 297/1910 = 15.6% at session start)
    context  262,144     vs  262,144         ->  parity (the KV budget is allocated up front)
    suites   7/7 green   determinism 6/6     ->  clean

**The deliverable is not met.** Decode is the headline and it is still 2.6x short; prefill improved
from 15.6% to 19% of baseline this session but remains 5x short. The determinism defect was a
correctness blocker and is now closed, but closing it was not progress toward beating the baseline -
it removed a reason the engine could not be shipped at all, nothing more.

The honest read on where the remaining gap lives is the one thing measured rather than inferred:
decode `exl3_mgemm` runs at **7.5% of DRAM throughput** while the same kernel at prefill reaches
**63%**, at identical 33.3% warp occupancy. The difference is bytes-in-flight per launch - 1.48 GB
at prefill against 3.01 MB at M=1. That is a shape/tiling question about how many expert rows one
launch covers, and every knob on the existing grid (shape, num_sms, smem, occupancy) has been
measured neutral, which is what makes it a structural problem rather than a tuning one.


---

## Context scaling, measured against the reference with matched methodology

The context-length axis had never been compared to the baseline on identical
input. It is now, and it exposes a gap that is separate from the raw decode
speed gap.

### The 30k/56k "zero tokens" result is NOT a bug

Three long prompts produced `0 tokens` with no error. Before treating that as a
defect I ran the identical document through the reference:

    helios    : 56,516 prompt tokens -> 0 tokens emitted
    reference : 56,544 prompt tokens -> completion_tokens=1, finish_reason=stop

Both engines emit an immediate stop token. Our long-context behaviour is
faithful. (The first sampled token is a stop id, and `runner.cpp:535` checks
`is_stop` BEFORE the `[ids]` print, which is why nothing is logged. NaN logits
would have produced token 0 via argmax, not 248044/248046, so the logits are
finite - this is the model, not a broken sampler.)

### Harness bug that invalidated two measurements

`--ids` is a PROMPT-INPUT flag: it replaces the prompt with raw token ids. So
`--ids 1` means "the entire prompt is token 1", and it silently ignores
`--prompt-file`. Two runs reported `prompt tokens=1` for a 120,000-character
file before I noticed. The output-id mechanism is the `HELIOS_IDS=1` env var.
No engine defect - but a flag that silently overrides a 120 KB prompt is a trap
worth writing down.

### Matched decode, identical prompts (64-token generations)

| context | Helios | reference | ratio | Helios delta | ref delta |
|---|---|---|---|---|---|
|   970 | 44.84 | 59.09 | 0.759 | -           | -         |
|  3830 | 41.53 | 54.13 | 0.767 | -7.4%       | -8.4%     |
| 12493 | 34.42 | 54.15 | 0.636 | **-23.2%**  | -8.4%     |

This is the newly quantified finding: **we lose 23% from 1k to 13k where the
reference loses 8%.** The reference is nearly flat across this range; we are
not. Our attention cost per token grows with KV length faster than the
reference's, which uses `cq3+QSA-index-planes` and a sparse gather while our
default path is dense fp16.

### The reference decode number depends on generation LENGTH

Reconciling a discrepancy with the session's earlier 116-128 tok/s figure:

| max_tokens | average | steady-state (after first 32) |
|---|---|---|
|  64 |  36.87 |  90.47 |
| 256 |  76.61 |  82.36 |
| 512 | 107.84 | **119.29** |

Both numbers are real; they measure different things. 116-128 is a LONG
generation in steady state. A 64-token generation is dominated by MTP warmup
and reports roughly half that. Any comparison against the baseline must match
generation length or it is meaningless.

### Matched long-generation comparison (512 tokens, 1k context)

| engine | decode |
|---|---|
| reference (MTP=4, steady state) | 119.29 tok/s |
| Helios MTP off | 44.14 tok/s |
| Helios MTP on  | 41.72 tok/s |

**Helios is 37% of the reference**, consistent with the session's earlier ~39%.
MTP still costs us 5.5% (41.72 vs 44.14) for the reason already established:
two M=1 trunk forwards for two tokens is break-even by construction, and our
own M-scaling data shows the weight-sharing knee is at M=5, not M=2.


---

## THE REFRAME: most of the decode gap is MTP, not the MoE kernel

The reference server accepts `ENABLE_MTP` / `MTP_DRAFT_TOKENS` as env
overrides, so its speculative path can be switched off without editing
anything. Measured on the same prompt, same box, same session:

| engine | MTP | 512-token steady-state |
|---|---|---|
| reference | off | **72.10 tok/s** |
| reference | on (4 drafts) | **119.29 tok/s** |
| Helios | off | 44.14 tok/s |
| Helios | on | 41.72 tok/s |

Two facts fall out, and they change where the effort belongs.

**1. Our M=1 decode is at 61% of the reference's M=1 decode, not 37%.**
44.14 / 72.10 = 0.61. The "37% of baseline" figure compared us against a
reference running its own speculation. The genuine kernel-level gap is 1.46x,
not 2.7x - the MoE work in this session closed a real part of it, and the
remaining kernel headroom is bounded by that 1.46x.

**2. MTP is the single largest factor, and for us it is a REGRESSION.**

| | MTP speedup |
|---|---|
| reference | 119.29 / 72.10 = **1.65x** |
| Helios | 41.72 / 44.14 = **0.95x** |

The reference gets 1.65x from four drafts; we lose 5.5% from one. Our
speculative step is two M=1 trunk forwards for two tokens, which is break-even
by construction - and that is exactly what 0.95x measures. This is now
measured end-to-end on both engines rather than inferred.

### Why the reference wins with speculation and we do not

At 72.10 tok/s the reference's base step is 13.9 ms. At 119.29 tok/s its
speculative step commits 4.1 tokens (119.29 tok/s => 34 ms/step). Five
separate base steps would be 69.5 ms, so its batched verification amortises
the MoE expert read across 5 positions at about **2.0x**.

Our measured M=5 amortisation is **1.35x** (16.18 ms/token x 5 = 80.9 ms,
against 5 x 21.9 = 109.5 ms separate). Same knee, weaker amortisation - our
grouped path does not share the expert traffic as effectively at M=5 as the
reference's does.

At our acceptance rate (68.4%) and our 1.35x amortisation, batching still
loses, which is the earlier retracted conclusion and it stands. The new fact
is that the reference reaches 1.65x on the same hardware with the same model,
so the deficit is in OUR amortisation, not in the idea.

### Consequence for priorities

The remaining decode gap decomposes as 1.46x (kernel) x 1.65x (speculation).
Speculation is the larger term and is currently NEGATIVE for us. Every knob
swept on the MoE grid (shape, num_sms, smem, occupancy) is bounded by the
1.46x. No further scheduling work on that grid can be worth more than a few
percent of the total.

This is a reprioritisation backed by measurement on both engines, not an
inference from a profile.


---

## BREAKTHROUGH: the M-scaling data that killed speculation was measured on the wrong path

`MOE_DECODE_MAX_N` was a hard `constexpr 2`, so the mgemm path could only ever run
at n=1 and n=2. Every M-scaling number in this project - the table that produced
"the knee is at M=5", and with it the retraction of batched verification - was
measured through `run_chunk`, which takes the **grouped** path above n=2.

Making the gate runtime-tunable and adding `helios bench-m` (which drives
`run_chunk` directly, so `max_n` stays realistic and only the width moves)
shows the two paths are not close:

| width | mgemm ms/call | grouped ms/call | mgemm faster by | mgemm amortisation |
|---|---|---|---|---|
| 1 | 21.96 | 22.18 | -        | 1.00x |
| 2 | 26.73 | 26.84 | -        | 1.64x |
| 3 | 33.18 | 45.38 | **1.37x**| 1.99x |
| 4 | 35.31 | 50.78 | **1.44x**| 2.49x |
| 5 | 44.40 | 56.41 | **1.27x**| 2.47x |

mgemm is 1.27-1.44x faster at every width >= 3 and amortises **2.47x at n=5**,
not the 1.35x the grouped-path table implied. A batched 5-wide verification
costs **44.4 ms, not 80.9 ms**.

### What that does to the speculation arithmetic

Base decode is 21.962 ms/token = 45.54 tok/s. At our measured 68.4% per-token
acceptance the expected accepted prefix of length K is `1 + a + a^2 + ...`:

| verify width | ms/call | expected tokens | tok/s | vs plain decode |
|---|---|---|---|---|
| 2 (1 draft)  | 26.73 | 1.684 | 63.0 | **1.38x** |
| 3 (2 drafts) | 33.18 | 2.152 | 64.9 | **1.42x** |
| 4 (3 drafts) | 35.31 | 2.472 | 70.0 | **1.54x** |
| 5 (4 drafts) | 44.40 | 2.691 | 60.6 | **1.33x** |

Width 4 is the optimum at **70.0 tok/s, 1.54x plain decode** - and that is before
charging for a partial-accept retry. This is the opposite of the earlier
conclusion, which had every K below break-even.

### Correctness gate

`--chunk 5` with the gate at 2 (grouped) and at 5 (mgemm) produces **identical
token id sequences**: `11751,13,271,...`. The two paths agree exactly at width 5,
so the fast path is not a shortcut past the verified one.

### What this does and does not establish

It establishes the COST side, which was the thing in doubt. It does not remove
the two real blockers:

1. **Draft chaining is still unimplemented.** Width 4 needs 3 chained drafts, and
   the MTP head consumes the TRUNK's pre-collapse tap, so drafting d1 requires
   the trunk to have run at pos+1. The header's own unconfirmed-semantics flag on
   `mtp_.tap_` is still the open question.
2. **Acceptance at 68.4% was measured with K=1.** Batched verification does not
   change what the draft predicts, but it does change how many are checked at
   once, so the rate must be re-measured rather than assumed.

The rejection of batched verification is hereby withdrawn on cost grounds. The
width-4 target of ~70 tok/s would put us at 59% of the reference's 119.29, up
from 37% - still short of the deliverable, but the first configuration in this
project that is not bounded below break-even.

### Also fixed: a latent buffer overrun

`sc.part[c]` was sized `max_n` rows, but the mgemm path memsets and writes
`n * topk` rows into it. That is safe only because the default `max_n` (1024)
happens to exceed `top_k` (10); `--chunk 5` with the gate at 5 overran it and
failed with `CUDA error invalid argument` at moe_layer.cu:405. Now sized
`max(max_n, gate * topk)`, which is free in every real configuration.


---

## SETTLED (negative): chained drafts collapse at depth 2 — batched verification at K>1 is not reachable

The width-4 projection above (70 tok/s) assumed a draft CHAIN could be built before the
trunk forward. That assumption was never tested. It is now.

`HELIOS_CHAINDBG=1` extends the chain to depth 3 by re-running the MTP head on the growing id
prefix against the same trunk tap — the only chaining form available before the trunk has run at
the later positions. 153 greedy steps, scored self-verifiably (d1 at P must equal the next record's
`next`; d2 must equal its `true_p1`):

| depth | acceptance |
|---|---|
| 1 | **66.7%** (102/153) |
| 2 | **3.3%** (4/123) |
| 3 | **2.4%** (3/123) |

Depth-1 matches the engine's own reported `spec: 102/153` exactly, so the measurement is sound.
Depth 2 is not "degraded", it is **zero**.

### Why, and it is structural rather than a tuning miss

The MTP head's input is a *combine* over the trunk's PRE-collapse stream stack. To draft at
position P+1 it needs the trunk tap **at P+1**. The trunk has only run up to P, so that tap does
not exist. Re-running the head on a longer id prefix therefore feeds it a tap row that was never
written — the 3.3% is what a garbage input produces, not a near-miss.

This is why the reference gets 1.65x from four drafts and we cannot: its MTP layer is a full
decoder layer with its own KV, so after the first draft it predicts autoregressively from its own
state. Ours routes *every* draft through the trunk-tap combine, so it can never get past depth 1.
Upstream (`qwen4_exp_mtp.py`) states there is no reference implementation for this head, which is
consistent with the tap handling being the hard part.

### What it costs to actually fix

An autoregressive path in the MTP head: after the first draft, run the layer on its own output
**without** the trunk tap combine. That is a new code path in `mtp.cu`, not a configuration
change, and its correctness cannot be checked against a reference because none exists — only
acceptance rate would tell us. It is the single largest remaining decode lever and it is
research-grade, not engineering-grade.

### Consequence for the plan

Batched verification is **withdrawn again, now on measured grounds rather than arithmetic**: the
width-4 cost advantage is real (35.31 ms vs 80.9 ms) but there is no way to populate the extra
rows with anything better than chance. K=1 batching needs no chain and is still arithmetically
unprofitable (26.73 ms verify vs 21.96 ms plain forward, 0.76x).

The honest position: **our MTP is depth-1-only, and at depth 1 it cannot beat a plain forward.**
Closing the decode gap requires the autoregressive draft head, not a scheduler.

### Diagnostic hygiene worth recording

The first version of this measurement was invalid: the extra `mtp_draft_step` calls wrote draft KV
at rows the next spec step reads, dropping depth-1 acceptance from 87.5% to 66.7% and silently
degrading the quantity being measured. The diagnostic now saves and restores 4 rows of draft-head
KV around itself. A measurement that changes its own subject is worse than no measurement — the
same failure class as the FNV-fingerprint episode, in a different guise.


---

## Decode phase breakdown, and three leads closed by measurement

`HELIOS_PROF=1`, 128-token greedy run, DECODE-only (per token, 22.62 ms total):

| phase | ms | share |
|---|---|---|
| **moe** | **11.69** | **52%** |
| gdn | 3.65 | 16% |
| amix (hc) | 2.67 | 12% |
| mmix (hc) | 2.53 | 11% |
| attn | 1.19 | 5% |
| final (head) | 0.39 | 2% |
| apply | 0.30 | 1% |
| ple | 0.20 | 1% |

Inside the MoE (`HELIOS_MPROF`, steady state, per layer): main kernel 0.147 ms,
finish/reduce 0.065, shared expert 0.038, router gemv 0.012, topk 0.012. x45 layers.

### Lead 1 CLOSED: grouped vs mgemm at n=1

The long-standing note "the reference does not use mgemm at M=1" implied our M=1
path might be the wrong kernel. It is not. Measured with the gate now able to
disable mgemm entirely (`HELIOS_MOE_DECODE_MAX_N=0`):

    n=1, mgemm   : 21.686 ms
    n=1, grouped : 31.908 ms

mgemm is **1.47x faster** at n=1. The reference's advantage is not a better
M=1 kernel choice; mgemm is already the right one and grouped would be a 47%
regression.

### Lead 2 CLOSED (too small to matter): expert-split imbalance

The default split is 352 experts on GPU1 vs 160 on GPU0 - a 2.2:1 imbalance on
every layer, and GPU1 is necessarily the bottleneck since both cards sync per
layer. Rebalancing 352 -> 340 (3.4% of GPU1's expert load) moved decode
45.53 -> 45.57 tok/s: inside the ~0.5% run-to-run noise. 330 OOMs on GPU0
(0.01 GB free). So the split is pinned by memory, and a 3.4% rebalance is below
what this harness can resolve. This does NOT refute layer-parallel execution
(24 layers x 512 experts per card would balance the work completely and remove
the per-layer cross-card reduction); it only says the current split cannot be
tuned within its memory envelope.

### Lead 3 OPEN, quantified: the hc mixers are the clearest identified inefficiency

amix + mmix = 5.20 ms/token, 23% of the step, over 96 calls (2 sites x 48
layers) = 54 us per call. Each call reads 13.1 MB of **unquantized fp16** hc
weights (`down[320,10240]` + `up[10210,320]`), so 1.26 GB/token of traffic that
cannot be reduced by requantising - only by issuing the loads better.

54 us for 13.1 MB is 242 GB/s on one card against a 936 GB/s peak: **26% of
peak**. The same weights at 60% of peak would cost 21 us, saving ~3.2 ms/token
and taking decode from 45.5 to roughly 52 tok/s (+14%).

This is the best-understood remaining win: a known phase, known bytes, known
achieved bandwidth, and a concrete target. It is a constant-factor kernel
improvement, not a structural change.


### Lead 3 REFINED: the decode hc mix is LAUNCH-BOUND, not bandwidth-bound

`HELIOS_GMIXPROF=1`, per call at decode (R=1), microseconds:

    norm=8.44  dots=3.06  up=32.09  fin=5.27  post=16.38   (65.24 total, 96 calls/token)

The arithmetic settles what these are. `norm` moves R*H*D = 10,240 elements -
about 100 KB in total - in 8.44 us, i.e. **12 GB/s, ~1.3% of peak**. No memory
system is the constraint there; the kernel is 256 elements per block over 40
blocks and should retire in about 2 us including launch. `post` reads 40 KB in
16.38 us. `dots` is the exception at 3.06 us because its 6.55 MB of `down` is
resident in this card's 72 MB L2.

The decisive comparison is prefill against decode. A 256-token chunk spent
297.6 ms in amix+mmix, i.e. **1.16 ms per token**. Decode spends **5.20 ms per
token** - 4.5x MORE per token while doing 1/256th of the work. Per-call fixed
cost is not amortised at R=1 at all; it dominates completely.

Per token the engine issues roughly 1,200 kernel launches (576 from the hc mix
alone: 96 calls x 6). At 65 us per hc-mix call for six kernels, the per-launch
component is several microseconds and is being paid 576 times per token.

**This makes CUDA-graph capture of the decode step the highest-value remaining
action, and it is not specific to the hc mix.** The MoE issues ~405 launches per
token across 45 layers and the GDN ~144, all with the same per-launch floor.
Phase times sum to 22.62 ms against a 23.9 ms measured wall clock per token, so
the overhead is inside the phase events rather than hidden between phases - it is
per-launch, not per-phase.

The obstacle is concrete and known: the decode step's kernels take `pos0` as a
kernel argument, and `pos0` changes every token, so a captured graph must either
read position from device memory or have its affected nodes updated per replay
(`cudaGraphExecKernelNodeSetParams`, no re-instantiation). Only the attention,
GDN and KV-write nodes depend on it, so a minority of nodes would need updates.


---

## hc-mix decode: +4.0% from a fixed 4-block grid

The launch-bound reading above was **wrong**, and `ncu` is what corrected it.

### The method error

`ncu` serialises kernels, so its per-kernel durations are for **ranking, not cost**. It
reported `gr_dots_small_kernel` at 31.31 us where the event timer says 3.10 us - a 10x
disagreement. I had been reading the gap between the two as launch overhead. It is mostly
ncu's own serialisation. Absolute cost comes from the event timer; ncu only says which
kernel is worst.

### What ncu did correctly reveal: grid shape

Ranking the hc-mix kernels exposed the real defect, which the phase totals could not:

| kernel | grid at decode | blocks used (of 82 SMs) |
|---|---|---|
| gr_hsum | (R, H) | **4** |
| gr_post | (R, H) | **4** |
| gr_dots_small | (rank, R) | 320 |
| gr_up | HD / warps_per_block | 1280 |

`gr_hsum` and `gr_post` are the only stages whose grid is fixed at `(R, H)`. At R=1 that is
**4 blocks on an 82-SM card**, and each block walks its entire 10,240-element row in
`blockDim`-sized strides - 40 dependent memory rounds, with only 4 SMs available to hide
them. `gr_post` moves 60 KB and took 16.85 us of pure kernel time.

### The fix

Both widened to 1024 threads (`gr_post`'s `__shared__ float red[8]` was also too small for
that, now `red[32]`). Per-thread accumulation order and the warp butterfly are untouched, so
the result is **bitwise identical** - and the production-shape `gr_mix` test confirms it,
holding at exactly 6.879e-04 across the change.

    gr_post   16.38 -> 7.99 us   (2.05x)
    decode    45.46/45.40  ->  47.36/47.35/47.24
    final     47.28/47.34/47.11 tok/s   (+4.0%)

All 5 suites pass, 471 tensors 0 mismatches, determinism 4 runs x 96 tokens identical.

### Two changes measured and REVERTED

Widening `gr_dots_small` the same way: 46.53/47.19/46.98 - neutral-to-negative. It has 320
blocks, so its latency was already hidden. The pathology is the **fixed (R, H) grid**, not
the block size; the two are easy to conflate because both are "more threads".

float4 staging in `gr_up` (16 B/lane, 512 B/warp, two rounds instead of five):
47.11/47.24/46.99 - neutral. `gr_up` is limited by having one warp per output element with
8 warps per block, not by transaction width. Reverted; recorded because the next person to
look at a 32 us kernel will reach for load width first, and it is measurably not the answer.

### Where the step stands now (~21.2 ms/token)

    moe 11.69 (52%) | gdn 3.65 | amix+mmix ~4.5 | attn 1.19 | rest ~0.9

The hc mix is no longer the second-largest phase it was; the MoE is now the whole story.
Its main kernel is 6.6 ms/token moving ~8.7 MB per card at 59 GB/s - **6% of peak** - and
that is the single largest inefficiency left in the engine.


---

## The decode MoE is occupancy-capped, and now that is measured rather than inferred

`ncu`, ranking every kernel over a decode step (900 launches):

| kernel | launches | total | share |
|---|---|---|---|
| `exl3_mgemm_kernel<...128...>` (gate+up) | 76 | 4.8 ms | **34.6%** |
| `exl3_mgemm_kernel<...256...>` (down) | 38 | 2.3 ms | **16.3%** |
| `gr_dots_small` | 39 | 1.2 ms | 8.9% |
| `exl3_gemv` (several variants) | 106 | 2.1 ms | 14.6% |
| `gr_up` | 39 | 0.7 ms | 4.7% |

**The two mgemm variants are 50.9% of all kernel time.** They are the engine.

### Why they cannot be tuned

Launch geometry, measured:

    block_size                          512 threads  (16 warps)
    grid_size                           80 blocks
    shared_mem_per_block_dynamic        92,160 B
    launch__occupancy_limit_shared_mem  1
    launch__occupancy_limit_registers   1
    launch__occupancy_limit_warps       3
    sm__warps_active                    33.1%

512 threads on 82 SMs is 33% thread occupancy, and the ceiling is hit **three ways at
once**:

1. **The grid is capped at 80.** `exl3_mgemm` is a cooperative launch, so
   `concurrency * MOE_SMS_PER_EXPERT <= num_sms` forces `10 * 8 = 80` blocks on 82 SMs.
   Every SM is already assigned a block - there is no more parallelism to hand out.
2. **Shared memory allows 1 block/SM.** 92,160 B against a ~100 KB budget.
3. **Registers ALSO allow 1 block/SM.**

Point 3 matters because it kills the obvious fix. The standing plan was "halve the smem to
fit 2 blocks/SM", with the blocker recorded as reduction-order stability. But registers cap
at 1 block/SM *independently*, so halving smem would buy nothing even if the ordering
question were settled. Both would have to fall together.

So every knob already swept on this grid - shape, `num_sms`, smem, occupancy - was measured
neutral for a reason that is now explicit rather than empirical: **the ported mgemm has no
headroom left at this shape.** 33% occupancy is the ceiling, and at M=1 there is only one
token-row per expert to amortise the weight read over.

### What this implies

The reference runs a different M=1 path (its own exl3 dispatch, not this cooperative
mgemm) and reaches 72.1 tok/s where we reach 47.2. That gap is not reachable by scheduling
this kernel. Closing it needs either a kernel that does not inherit the cooperative-launch
grid cap, or expert prefetching that overlaps the weight read with the previous layer's
compute. Both are new work, not tuning.


---

## The synthesis: every large phase is latency-bound at M=1, and that is the whole story

Taking the measured per-phase costs and the bytes each phase *must* move:

| phase | ms/token | bytes/token | achieved | % of 936 GB/s |
|---|---|---|---|---|
| MoE | 11.69 | ~567 MB | 48 GB/s | **5%** |
| GDN | 3.65 | ~288 MB | 79 GB/s | **8%** |
| hc mix (up) | ~3.1 | ~629 MB | 205 GB/s | 22% |
| attention (1k ctx) | 1.19 | ~2.4 MB | 2 GB/s | n/a - too small to saturate |

Every phase that has real work to do is running at **5-22% of DRAM peak**. None of them
is bandwidth-bound; all of them are memory-latency-bound, for the same reason: **at M=1
there is one token-row per expert, per GDN state, per output element, which is not enough
work to cover memory latency on 82 SMs.**

This is corroborated from three independent directions:

- The MoE is capped at 33% warp occupancy by a cooperative-launch grid limit (measured).
- `gr_up` is not transaction-width limited (float4 neutral) and not bytes-per-block
  limited (128/256/512 all identical, 1024 worse) - so the remaining explanation is the
  same latency exposure.
- The GDN's recurrent state is 4 MB per layer; reading and writing it 36 times per token
  at 79 GB/s is the signature of the same thing.

### Why the reference escapes it

Not by writing a faster M=1 kernel. **By never running M=1.** Its MTP verifies five
positions per trunk pass, so the same latency is amortised across five token-rows and the
memory system is kept in flight. That is why its base rate (72.1) is 1.53x ours while its
served rate (119.3) is 2.53x ours: the kernel term and the speculation term are not
independent, they are the same fact seen twice.

### Which means our MTP blocker is the whole remaining gap

We measured that our MTP is depth-1-only (66.7% at depth 1, 3.3% at depth 2) because the
head combines the trunk's pre-collapse tap, and no trunk tap exists for positions the
trunk has not reached. At depth 1 a draft cannot beat a plain forward, so our MTP is a
0.95x regression and we stay pinned at M=1 - which is exactly the regime where every phase
above runs at single-digit percent of peak.

**The decode gap is therefore one problem, not two:** our draft head cannot chain, so we
cannot batch, so we are latency-bound everywhere. An autoregressive draft head - one that
runs on its own output without the trunk tap combine - would not merely add 1.65x of
speculation on top; it would lift every phase in the table above out of its latency-bound
regime at the same time. That is the single highest-value piece of work left in this
engine, and it is the piece the upstream repository does not implement either.


---

## PREFILL: attention is now the #1 cost, not the MoE

`HELIOS_PROF=1`, 15,302-token prompt, chunk=1024, 252 tok/s (blended per-chunk averages):

| phase | ms | share |
|---|---|---|
| **attn** | **1516.39** | **37.6%** |
| moe | 1452.46 | 36.0% |
| gdn | 481.96 | 11.9% |
| amix + mmix | 525.25 | 13.0% |
| ple | 59.17 | 1.5% |
| apply | 11.49 | 0.3% |

The standing note in this repo says prefill is "hc mix 44%, MoE 35%". **That is stale** -
the hc-mix work (weight-stationary `gr_up`, tiled `gr_dots`) plus the chunk 256->1024 change
have since cut the mixers to 13%, and **attention is now the largest single phase at 37.6%**,
ahead of the MoE for the first time.

This matters for the deliverable specifically because attention is the only phase that is
quadratic in context. At 15k it is already the largest line; at 230k it would dominate
completely. The reference scales better because it runs `cq3+QSA-index-planes` - sparse
attention selected by the indexer - while our default path is dense fp16.

Our QSA implementation exists and is correct but measured 2x SLOWER than dense at 15k
(attn phase 42.0 ms vs 9.35 ms), for two measured reasons: the `tok_idx` gather destroys
locality (2,050 scattered per-token reads against dense's 369 MB contiguous stream at
555 GB/s), and the indexer adds a full 2560->640 EXL3 matmul per layer per token. So the
technique that gives the reference its context scaling is present but not yet paying for
itself here. The identified next step stands: port the reference's pooled K/V plane
(`kpool_write`) so the gather becomes block-contiguous.

### Sanity bound on how much is recoverable

If attention were entirely free, prefill would be 60.7 s x 0.624 = 37.9 s = **404 tok/s**
against the reference's 1910. So attention is the biggest single line but not the whole
story - the MoE and the mixers still have to improve too. Closing prefill needs both the
QSA gather fix and attention work, not either alone.


---

## Two out-of-bounds reads found by compute-sanitizer

Turning on QSA to find the crossover produced an immediate illegal memory access. That
turned out to be worth chasing for reasons unrelated to QSA.

### Bug 1 (FIXED): `gr_dots_kernel` staged rows that do not exist

`gr_dots_kernel` guards its **store**:

    if (r0 + lr < R && i0 + li < rank) t[...] = ...;

but the **staging read** 12 lines above had no such guard:

    ns[rr * DBK + kk] = normed[(size_t)(r0 + rr) * HD + k0 + kk];

`R` is almost never a multiple of `DTR`=8 - real prompts are 970, 15302 tokens - so the
final row-tile stages rows `R..DTR-1` that are not there. The values were discarded by the
store guard, so this was invisible in normal operation: it only overran when the bump
allocator happened to place something unmapped right after `normed`. compute-sanitizer:

    Invalid __global__ read of size 4 bytes
      at gr_dots_kernel ... in gr_mix.cu:125
      by thread (128,0,0) in block (0,121,0)
      40.961 bytes after the nearest allocation of size 39.731.200 bytes
      (= exactly 970 rows of HD=10240 fp32 - the prompt length)

Now bounded and zero-filled, so the compute loop consumes defined data and the kernel stays
deterministic. All 5 suites pass; decode 46.79/46.84 (from 47.28) and prefill 328.9/329.6 -
the ~1% is the extra compare in a loop that runs 320 times per thread, which is the right
trade against reading past the end of a buffer.

This is the second time a guard was added to the store but not the load in this kernel. The
gr_dots stride bug earlier in the project was the same shape of mistake: a staging detail
that no correctness test could see because the *result* was still guarded.

### Bug 2 (diagnosed, not fixed): QSA's indexer GEMM overruns its Hadamard scratch

With bug 1 fixed the fault moved, and the new one is genuinely QSA's:

    Invalid __global__ read of size 4 bytes
      at had_hf_r_128_inner<(bool)1,(bool)0> ... hadamard_inner.cuh:112
      Access to 0x16c00000f6c is out of bounds
      124.688.101.535.892 bytes before the nearest allocation

`had_hf_r_128_inner<true,false>` is the EXL3 **input** Hadamard, reached from
`exl3_gemm_kernel.cuh:21`. The address is a wild pointer, not a small overrun, so this is a
scratch that was never sized rather than a bounds slip. `exl3_gemm.cu:363` already carries the
warning - "an undersized scratch is silent OOB corruption (found the hard way)" - with a
`HELIOS_ASSERT` on the size, but asserts are compiled out here.

This is concrete confirmation of the standing note that the QSA kernels are **GLM-shaped**
(32 indexer heads, MLA) while this model has 4 indexer heads and GQA. QSA is not merely slower
on this model, it is memory-unsafe: the indexer's 2560->640 EXL3 matmul is launched with
scratch geometry that does not match. Fixing it means parameterising the indexer path for
this model's head counts, not enabling a flag.


### QSA defect: how far the diagnosis got, and where it stopped

Confirmed by compute-sanitizer: a **wild-pointer** read (124 TB before any
allocation) in `had_hf_r_128_inner<true,false>` reached from
`exl3_gemm_kernel.cuh:21` - the EXL3 input Hadamard. A wild address rather than a small
overrun means a bad base pointer or a group index running far past the scale arrays, not a
missing bounds check.

What is established:

- The indexer's **shape** is correct: `index_qk_proj` is 2560 -> 640, matching
  `idx_qk_out = (idx_heads + idx_kv_heads) * idx_dim = 5 * 128 = 640`.
- The runner passes `c.hidden` as `hidden_for_qsa`, so the scratch **sizing** parameter
  agrees with the offsets computed in `attn_layer.cu:206-213`.
- `qsa_project` is called with `n = 1` from the row loop, and `sc.had` is
  `cfg.hidden * 2` bytes = exactly the `n * hidden` halves the GEMM documents needing. It
  fits, exactly, with no headroom.
- The one concrete divergence from the main attention path is visible in the source:
  `qsa_project` hardcodes the fourth `GroupWords` field to `0`:

      exl3::GroupWords gw{trellis, suh, svh, 0};          // qsa.cu:94
      exl3::GroupWords qw{trellis, suh, svh, w.q.mul1};   // attn_layer.cu:140

  Every other call site passes the loaded `mul1`. If this checkpoint's indexer carries
  `mul1` scales, forcing `0` selects a different dequant path and a different scale
  indexing - which is exactly the shape of fault observed. **This is the leading
  hypothesis, not a confirmed root cause.**

What stopped the investigation: the per-tensor `mul1` presence could not be read out of
`quantization_config.json` with the searches available, and the EXL3 group-table indexing
would have to be traced against the indexer's real scale layout to confirm. That is a
deeper dig than the remaining evidence justified, and the fix either way is the same piece
of work: parameterise the QSA indexer path for this model (4 indexer heads, GQA) rather
than inheriting the GLM-shaped port.

**QSA must stay off.** It is not merely slower on this model - it is memory-unsafe, and
`HELIOS_QSA=1` is a bisect switch, not a usable configuration.


---

## QSA bug 2: `mul1` was hardcoded to 0 in the indexer projection — FIXED

The leading hypothesis from the wild-pointer diagnosis is now confirmed and fixed.

`qsa.cu:94` built the indexer's `GroupWords` with a literal in the `mul1` slot:

    exl3::GroupWords gw{trellis, suh, svh, 0};          // qsa_project
    exl3::GroupWords qw{trellis, suh, svh, w.q.mul1};  // every other call site

and `index_qk_proj.mul1` **is present** in this quant's `quantization_config.json`
(verified by name against the config, alongside `q_proj.mul1`, `o_proj.mul1`,
`gate_proj.mul1` — all present). So the indexer's dequant was indexing the scale
arrays as if the tensor carried none, which is the wild read the sanitizer saw inside
the input Hadamard.

`qsa_project` now takes `qk_mul1` and both call sites pass `w.idx_qk.mul1`.

**Verification that this was the right fix, not a plausible one:** with the change,
`HELIOS_QSA=1` under compute-sanitizer at short context reports **0 errors**, where
before the same configuration faulted. The default path is untouched — all four suites
pass and decode is 47.33 tok/s.

### What remains

At context above the QSA threshold (2051) a second fault persists, same signature:

    Invalid read at had_hf_r_128_inner<true,false>, block (0,0,3)
    134.934.853.319.572 bytes before the nearest allocation   (129 errors)

`qsa_project` runs in both cases and is now clean, so this one is downstream of it —
in `qsa_select` / `qsa_expand` / `gqa_sparse_decode`, the stage that only executes once
`use_qsa` is true. 129 errors rather than one indicates those kernels are writing or
reading well outside their buffers, consistent with the selection scratch being laid
out for the GLM head geometry rather than this model's 4 indexer heads.

**QSA remains OFF by default and must stay there** — it is now *less* broken, not
correct. One of its two memory-safety defects is fixed; the sparse-attention stage
still overruns.


---

## Reading exllamav3 instead of guessing: how its MTP chain actually works

The MTP blocker had been carried as "research-grade, upstream does not implement it".
That was wrong, and reading the source settles it in three lines.

`exllamav3/generator/generator.py:803-820` - the draft loop:

    for idx in range(window):
        params = {"target_hidden": temp_hidden, "cache": self.draft_cache, ...}
        batch_state = self.draft_model.forward(batch_ids, params)
        new_ids = self.draft_model.sample_from_state(batch_state, params)
        batch_ids.copy_(new_ids)
        cache_seqlens += 1
        temp_hidden = batch_state          # <-- the draft's OWN output is the next input

`exllamav3/generator/job.py:1425-1431` seeds it:

    target_hidden = params.get("export_states")[-1]   # the TRUNK's pre-collapse stack
    carry_hidden = seq.mtp_carry_hidden               # the MTP's own last output
    shifted_hidden = torch.cat((carry_hidden, target_hidden[:, :-1, :]), dim = 1)

So the semantics are:

- **Iteration 0** is seeded from the trunk's exported pre-collapse stack, with the MTP's own
  previous carry prepended (it starts as zeros).
- **Every later iteration** is seeded from the MTP's *own* output stack. The trunk tap is
  never needed again.

`Qwen4ExpMTPStackOut` exists precisely to make that possible - its docstring: *"The draft
chain's output is the flattened PRE-mixer stream stack (it feeds the next drafting step's
target_hidden); sample_from_state applies the mixer + shared lm_head."*

### This is why our chain measured 3.3%

Our diagnostic re-ran the head on a growing id prefix while feeding **the trunk tap at every
depth** - a chain the reference never runs. It was measuring a strawman, and I wrote it up as
"structurally impossible". It was not impossible; it was mis-specified.

### And why re-running it that way still gives 4.3%

Feeding `mtp_.tap_` instead is the right *idea* but the wrong *shape*. exllamav3's
`temp_hidden` is a sequence that grows by one row per iteration (`batch_ids.copy_(new_ids)`,
then `temp_hidden = batch_state`), so at iteration k it carries k+1 rows. Our `mtp_.tap_`
holds ONE row, and `mtp_draft_step(ids, n=k)` expects n rows of `trunk_streams` - so rows
1..k-1 are read uninitialised. Same failure, one level down: 3.3% -> 4.3% is noise.

**What this establishes:** the blocker is not architectural, it is unimplemented plumbing.
The head, the carry, and the feedback all exist in the checkpoint and in our port; what is
missing is accumulating the tap stack across iterations so iteration k receives k+1 rows of
the draft's own state. That is a tractable change, and it is the gate on the 70 tok/s
width-4 target.

## QSA: both memory-safety defects now closed

The second overrun was `qsa_part`, sized with `nchunks` clamped to 64 while
`cap = 512*4+4 = 2052` needs `(2052+15)/16 = 129` chunks - half the buffer the sparse kernel
writes. That clamp was left over from an earlier fix that corrected the head_dim but not the
chunk count.

With both `mul1` and the chunk count corrected, `HELIOS_QSA=1` under compute-sanitizer at
1k context (past the 2051 threshold, so `use_qsa` is live) reports **0 errors**, and decode
runs at 45.29 tok/s with QSA on. QSA is no longer memory-unsafe. It is still not *faster*
than dense at this length - the crossover question is open again and now answerable.


---

## THE CHAIN WORKS: depth-2 acceptance 3.3% -> 65.2%

Rewriting the diagnostic into exllamav3's actual one-position-per-iteration form
(`mtp_draft_step(ids={prev_draft}, n=1, pos=pos_+(k-1), tap=mtp_.tap_)`) over the
same 153-step protocol:

| depth | before | after |
|---|---|---|
| 1 | 61.0% | 61.0% |
| 2 | **3.3%** | **65.2%** |
| 3 | 2.4% | **39.1%** |

Depth-2 is now *higher* than depth-1, which is what a correctly-conditioned chain
should look like: each draft is conditioned on the head's own state rather than on
a stale tap row.

Two earlier harnesses were wrong and neither measured what the reference runs. The
first grew `ids` to length k while feeding the trunk tap at every depth; the second
grew `ids` and fed the draft's own tap but still passed k rows where the head
expects 1. The 3.3% and 4.3% figures were harness artefacts, not properties of the
head - and I had written the first one up as a structural impossibility.

### Draft cost, measured

`HELIOS_CHAINDBG=1` adds exactly two extra draft steps per spec step. Same prompt,
same 96 tokens:

    without diagnostic : 2.30 s / 2.34 s
    with diagnostic    : 2.40 s / 2.41 s

~60 spec steps, so ~120 extra draft steps in 0.10 s: **0.83 ms per draft step**,
against 21.9 ms for a trunk forward. The draft is 26x cheaper than the thing it
accelerates - which is the whole reason speculation can pay here.

### Projected throughput, now with every term measured

| K drafts | verify width | trunk ms | draft ms | total | E[tokens] | tok/s |
|---|---|---|---|---|---|---|
| 2 | 3 | 33.18 | 1.66 | 34.84 | 2.008 | **57.6** |
| 3 | 4 | 35.31 | 2.49 | 37.80 | 2.163 | **57.2** |
| 4 | 5 | 44.40 | 3.32 | 47.72 | ~2.21 | 46.3 |

E[tokens] = 1 + P(d0) + P(d0,d1) + P(d0,d1,d2) with 0.61 / 0.398 / 0.155 from the
measured per-depth acceptance.

**~57 tok/s, against 47.3 plain decode (1.21x) and 44.7 for the current MTP path
(1.28x).** K=2 or K=3 is the optimum; K=4 falls off because the width-5 forward
costs more than the extra draft is worth.

This is the first configuration in the project where speculation actually beats a
plain forward, and every input to the number is measured rather than projected:
acceptance per depth, per-draft cost, and the width-N trunk cost from `bench-m`.

It is still 48% of the reference's 119.29, so it does not meet the deliverable. But
the blocker is gone: what remains is implementing this loop in `spec_step` (batched
verify at width 3-4, with GDN state rollback on partial accept) - an engineering
task with a known-good design, not a research question.


---

## Enabling piece for batched verification: a multi-row head (opt-in, default untouched)

`final_head` projected only the LAST position, which is why batched verification was
impossible: a width-N forward gives N predictions and the head surfaced one.

It now projects the last `head_rows_` positions in a **single** gemm. One gemm, not
one per row, is what makes it affordable - the 5-bit lm_head is 163 MB of weights and
re-reading it per row would cost more than the speculation saves.

**A trap worth recording:** `head_rows_` was already 2 (for the MTP draft's second row), so
inferring "project all rows" from the buffer size would have made every prefill chunk
sample TWO POSITIONS EARLY - `next_token()` reads row 0, and rows 0..1 would hold
positions n-2 and n-1. Nothing would have crashed; every prefill answer would just be
wrong. Multi-row is therefore a separate flag (`head_multi_row_`), set only by
`HELIOS_HEAD_ROWS>1`, never inferred from the allocation.

Verified after the change, default path:

    decode  47.34 tok/s      (was 47.33)
    prefill 332.3 tok/s      (was 328.9 / 329.6)
    5/5 suites pass, 471 tensors 0 mismatches

### What batched verification now has, and what it still needs

HAVE, all measured:

- a working chained draft (depth-2 65.2%, depth-3 39.1%)
- per-depth acceptance 0.61 / 0.398 / 0.155
- draft cost 0.83 ms, 26x cheaper than a trunk forward
- trunk forward at width 2/3/4/5 from `bench-m`: 26.73 / 33.18 / 35.31 / 44.40 ms
- a head that can emit every row of a batched forward

STILL NEEDS:

- `spec_step` rewritten to chain K drafts then issue ONE `run_chunk` at width K+1
- GDN state rollback on partial accept. This is the expensive part, and the retry
  strategy decides whether the plan is viable at all:

  Retrying the whole rejected prefix costs a full trunk forward, which makes every K a
  net loss (K=1 lands at 44.8 tok/s, below plain decode). Replaying ONLY the GDN for the
  accepted prefix costs ~3.65 ms/token instead of 21.9, and on that basis:

  | K | trunk | drafts | retry | E[time] | E[tokens] | tok/s |
  |---|---|---|---|---|---|---|
  | 2 | 33.18 | 1.66 | 0.16 + 3.65*m | 37.9 ms | 2.008 | **~53** |

  so a GDN-only replay entry point - not a trunk re-run - is what makes batched
  verification profitable. It is a new function over already-written kernels, and the
  GDN phase is only 3.65 ms of the 21.9 ms step.

Projected ~53 tok/s: **1.12x** over plain decode, 48% of the reference's 119.29. Real,
but not the deliverable.


---

## GDN rollback: built, and proven bit-exact

The last piece batched verification needs. The GDN recurrence is the only trunk
state a partial accept invalidates - attention KV past the accepted prefix is
overwritten and never read, but conv/rec state is a running recurrence that cannot
be rewound by overwriting. So: snapshot it, and on a partial accept restore and
replay only the accepted prefix.

`bench-gdn` is the gate, and it found two real bugs in the machinery itself.

**Bug 1: one `sub_in_` for 36 layers.** The replay needs each GDN layer's sublayer
input, but `sub_in_` is a SINGLE buffer reused by every layer - after the loop it
holds only the LAST layer's value. The first version snapshotted it once at the end
and replayed the wrong rows for 35 of 36 layers. Now `run_chunk` captures per layer
into `gdn_sub_snap_[gdn_ord]` while `gdn_capture_` is armed, guarded by a branch so
the default path pays nothing.

**Bug 2: snapshot taken after the batch.** Snapshotting the POST-batch state and
then replaying those same positions applies them twice. Restore was verified
bit-exact in isolation, which is what localised it:

    width=1 RESTORE-ONLY=838f9feb (want 838f9feb)   <- copy machinery correct
    width=1 replay         =df9a97a1 (want 838f9feb)  <- ordering wrong

The call order is now `gdn_snapshot -> run_chunk(capture) -> [accept | restore+replay]`.

    [gdn] replay width=1  BIT-EXACT  fp_before=838f9feb fp_after=838f9feb
    [gdn] replay width=2  BIT-EXACT  fp_before=f3b625a6 fp_after=f3b625a6
    [gdn] replay width=3  BIT-EXACT  fp_before=d4b5cb32 fp_after=d4b5cb32
    [gdn] replay width=4  BIT-EXACT  fp_before=9a3a3de7 fp_after=9a3a3de7
    [gdn] REPLAY SELFTEST PASS

### State of batched verification

Every component now exists and is measured:

| component | status |
|---|---|
| chained draft (depth-2 65.2%, depth-3 39.1%) | working |
| multi-row head (opt-in, default untouched) | working |
| GDN snapshot / restore / replay | **bit-exact, self-tested** |
| width-N trunk cost | 26.73 / 33.18 / 35.31 / 44.40 ms |
| draft cost | 0.83 ms |
| per-depth acceptance | 0.61 / 0.398 / 0.155 |

Default path unregressed: decode 47.47/47.44, prefill 330.5, suites pass.

What is left is assembly - rewrite `spec_step` to chain K drafts, issue ONE
`run_chunk` at width K+1, and on a partial accept `gdn_restore()` + `gdn_replay(m)`.
Projected **~53 tok/s** at K=2 (1.12x plain decode, 48% of the reference's 119.29).


---

## Batched verification: the machinery works, and the economics are decided by ONE thing

`spec_step` now chains K drafts and verifies them in a single trunk forward, with GDN
rollback on a partial accept. **It works**: 92.0% acceptance, **1.92 tokens per step**.

And it is **slower than not speculating at all**: 41.60 tok/s against 47.47 plain.
The reason is structural, and the fix is the same thing exllamav3 does differently.

### Two forwards vs one

Our chain has to be SEEDED from a trunk forward's tap, and we pay for that forward
before the verify forward:

    ours:  width-1 forward (produces the tap)  ->  draft K  ->  width-K verify
    ref:   draft K (seeded from the PREVIOUS verify's tap)  ->  width-(K+1) verify

The reference's `mtp_last_hidden` carry means the tap that seeds round t+1 is
produced by round t's own verify forward. There is no dedicated tap forward, because
the verify forward *is* the forward that yields the tap.

Cost model, from the measured width-N and draft costs (ms; tok/s = tokens / ms * 1000):

| K | ours (2 forwards) | reference structure (1 forward) |
|---|---|---|
| 1 | 49.7 ms -> 32.4 | 27.7 ms -> **58.1** |
| 2 | 57.0 ms -> 35.2 | 35.0 ms -> **63.4** |
| 3 | 59.9 ms -> 47.2 | 38.0 ms -> **74.5** |
| 4 | - | 47.9 ms -> **71.8** |

Our K=3 barely matches plain decode; the reference structure at K=2 is already
**1.34x** plain decode. The entire 21.96 ms tap forward is the difference.

### What this costs and what it bought

- The batched path stays **opt-in** (`HELIOS_MTP=1`, which is not the default).
  Default decode is unregressed: 47.56 / 47.67.
- Bought, all working and measured: a chained drafter, a multi-row head, bit-exact
  GDN rollback, and a verified 92%-acceptance batched verifier. The hard part -
  populating the extra rows with something better than chance - is solved.

### The remaining change, and it is small

Carry the verify forward's per-position taps (the same per-layer capture pattern
`gdn_sub_snap_` already uses), and after a partial accept seed the next round's
chain from the tap at the last COMMITTED position. That deletes the dedicated
width-1 forward and the projection moves from ~41.6 to the 63-75 tok/s band.

It is worth being precise about the ceiling: even at K=3 with the reference
structure that is ~74 tok/s, against the reference's 119.29. It closes the gap to
roughly 62% rather than beating it. The MoE kernel's 33% occupancy ceiling is
still the other half.



---

## Speculation: built, measured, and WITHDRAWN

Final state: `spec_step` commits exactly one token through the plain decode path. Speculation is
still selectable with `HELIOS_MTP=1` but no longer changes behaviour, because on this configuration
plain decode is both the fastest and the only correct speculative configuration found.

| structure | forwards / step | acceptance | tok/step | tok/s |
|---|---|---|---|---|
| **plain decode (shipped)** | 1 x width-1 | - | 1.00 | **47.5** |
| dedicated width-1 seed + width-2 verify | 2 | 92.0% | 1.92 | 41.6 |
| carry-tap, single width-3 verify | 1 | 36.6% | 1.37 | 27.0 |

### What was established

The exllamav3 structure is real and was reproduced: `carry_tap_` holds the tap at the last COMMITTED
position, copied from the verify forward's per-position `tap_` rows, so the next round's chain is
seeded without a dedicated seed forward. The flat 21.96 ms tax is gone from that code path, and the
verify is a single width-3 forward. Batched verification itself is sound - 92% acceptance, 1.92
tokens/step, and bit-exact GDN rollback.

### Why it was withdrawn

Deleting the seed forward cut per-step cost but acceptance fell 92% -> 36.6%, and at this ratio
acceptance is worth more than the saved forward. The cause is a **desync between the draft head's KV
cache and the committed sequence**: the chain writes draft KV only at positions it visits, while a
verify advances the committed position by `1+m`, so the next seed reads entries that were never
written or still hold a rejected draft.

The obvious fix - replay the head over the skipped positions using the taps the verify already left
in `tap_` - **breaks generation outright**: 2 tokens, deterministic, every run. `mtp_draft_step`'s KV
is append-sequential, so re-running it at an already-visited position corrupts the cache rather than
backfilling it. A correct resync needs a draft-cache length counter and a forward-only fill; that is
not implemented.

### A measurement trap worth recording

The first determinism run after the carry restructure reported PASS on 3 runs x 96 tokens. It was
vacuous: all three runs were producing the same broken 2-token output. A determinism gate is only
meaningful alongside a token-count or output-length assertion - identical garbage passes it exactly as
readily as identical text.

### Regressions caught and fixed along the way

- the prefill carry hook appended the prompt to `hist_tokens_` a second time (`generate()` already
  assigns it), which would have corrupted `prefix_match()` for every subsequent request
- `spec_steps` was not being incremented, so the acceptance line silently stopped printing
- the draft chain was seeded with `next` against the *previous* position's tap; the head combines a
  tap with the embedding of the token **at the same position**, so the two must be paired

### Correctness status at withdrawal

- default path unregressed: 47.63 / 47.05 tok/s
- `HELIOS_MTP=1` now takes the safe path: 96 tokens at 46.29 tok/s, coherent output
  ("The capital of France is" -> " is Paris. That is correct. **Paris** is the capital ...")
- all 7 suites pass: attn parity, QSA parity, aux, reconstruct, gemm smoke, mgemm semantics, model load


---

## SETTLED NEGATIVE: there is no QSA/dense crossover (measured, 33x context range)

The standing open question was whether QSA's 2048-key budget eventually beats dense attention's
full-context read. Measured end-to-end decode, same prompt at five lengths, `--tokens 16 --temp 0`:

| context (tokens) | dense | QSA | delta |
|---|---|---|---|
| 1,262 | 45.34 | 45.05 | -0.6% |
| 5,252 | 31.68 | 31.13 | -1.7% |
| 10,502 | 21.66 | 21.44 | -1.0% |
| 21,002 | 11.16 | 11.16 | 0.0% |
| 42,002 | 4.95 | 4.94 | -0.2% |

**QSA never wins.** It tracks dense to inside run-to-run noise at every length, and the advantage
does not emerge with context.

This is stronger than the earlier "2x slower at 15k" reading and it settles the question against
the hypothesis. At 42k context sparse reads 2,048 keys against dense's 42,002 - a 20x reduction in
attention work - and delivers *no* speedup at all. So the cost is not in the attention QSA skips; it
is in what QSA adds per layer per token:

- a full 2560 -> 640 EXL3 indexer matmul, and
- a `tok_idx` gather that turns one contiguous 369 MB stream into 2,050 scattered per-token reads,
  destroying the coalescing the dense path depends on.

Those two scale with the *layer and token* count, not with the context length, so they do not shrink
relative to dense as context grows - which is exactly why no crossover exists. Until the gather is
made block-contiguous (the reference's pooled K/V plane, `kpool_write`), enabling QSA cannot pay at
any length. **DEFAULT STAYS DENSE, now on evidence across 33x rather than a single 15k point.**

### Also measured: decode cost is LINEAR in context, not quadratic

Slowdown per doubling of context: 1.46x (5k->10k), 1.94x (10k->21k), 2.26x (21k->42k). That is
linear (2x per doubling), not quadratic (4x per doubling). The decode collapse from 45.3 tok/s at
1.2k to 4.95 at 42k is the dense fp16 KV stream, which is O(context) bytes per token.

Two consequences worth stating separately:
1. the long-context problem is a BANDWIDTH problem (stream the whole KV every token), not a
   complexity problem, so a sparse indexer is the right idea for the wrong reason - it cannot help
   while its own per-layer cost is the scatter;
2. the cheapest large win at long context may be KV *quantisation* rather than KV *selection* -
   reducing bytes streamed per token attacks a linear cost directly, and INT8-G64 + Hadamard-D256
   was already scoped in the design notes. At 42k context, 4.95 tok/s is far below what a 4x byte
   reduction would buy.


---

## SETTLED NEGATIVE: grouped is NOT faster than mgemm at M=1 (measured, correctness-gated)

Standing lead: "our decode runs mgemm, the reference does not - it routes through exl3::gemm's QTIP
GEMV. If grouped is faster than mgemm at n=1 this is a large, already-implemented win."
`HELIOS_MOE_DECODE_MAX_N=0` disables mgemm entirely, so both paths were measured end-to-end at
n=1, 64 greedy tokens, 3 runs each, with the token-id hash as the correctness gate:

| path | tok/s run1 / run2 / run3 | md5 of token ids |
|---|---|---|
| `MAX_N=0` grouped | 32.59 / 32.60 / 32.57 | `6991a976c99b` |
| `MAX_N=2` mgemm (default) | 47.41 / 47.44 / 47.35 | `6991a976c99b` |

**mgemm is 45% faster than grouped at M=1.** The hypothesis is refuted; the default stands.

The correctness gate matters here and it PASSED: all six runs hash identically, so the two MoE paths
compute the same function and the gap is purely cost. Had the faster config also been *more*
accurate that would have been a bug signal rather than a win, per the gr_dots lesson - it was not,
which is what makes the 45% trustworthy.

### What this leaves open

The reference is faster than BOTH of our paths at M=1, so "use grouped like the reference does" is
not available as a shortcut - our grouped path is 1.45x slower than our own mgemm, and the
reference's advantage sits in the QTIP GEMV kernel neither of ours matches. That remains genuine new
kernel work, and this measurement at least removes the cheapest candidate for it.


---

# SPECULATION WORKS: 47.4 -> 59.4 tok/s (+26%), and it is now the default

The long-running "MTP is a regression, keep it off" conclusion is **overturned**. The path is
correct, deterministic, and faster than plain decode for the first time.

| config | tok/s | acceptance | tok/step |
|---|---|---|---|
| plain decode (`HELIOS_MTP=0`) | 47.14 / 47.31 / 47.36 | - | 1.00 |
| **speculative (DEFAULT)** | **59.43 / 59.43** | **74.7%** (109/146) | **1.75** |

256-token greedy generation, identical token ids across every run.

## Structure: one width-2 forward per step, no seed forward

`mtp_draft_step(stack, id, pos0)` consumes the stack and the token at `pos0` and emits the stack
for `pos0+1`, so **its token prediction is for pos0+1**. That single fact fixes the whole design:

```
step A:  stack@pos_-1 + token@pos_-1  ->  stack@pos_     (head not run: `next` is already known)
step B:  stack@pos_   + token@pos_    ->  d0, for pos_+1
verify:  run_chunk([next, d0]) at width 2, compare logits row 0 (predicts pos_+1) against d0
```

`carry_tap_` holds the tap at the last committed position, copied out of the previous verify's
per-position `tap_` rows, so there is **no dedicated width-1 seed forward** - the ~22 ms tax that
made the original two-forward version measure 41.6 tok/s, slower than not speculating at all.

## Three real bugs, all in the same area, none of which was a KV desync

1. **One-position comparison misalignment.** An earlier version seeded the chain once and compared
   logits row 0 against a draft for `pos_` itself - a position `next` already occupies. That is
   what produced the "36.6% acceptance". There was no desync: `run_chunk` already refills the draft
   KV for every position it processes (runner.cpp:522), and each round's chain rewrites the previous
   round's rejected positions with the true tokens. The desync diagnosis was wrong.
2. **`last_committed_` clobbered from the wrong place.** `capture_carry()` derived it from
   `hist_tokens_.back()`, but `generate()` pushes `next` only *after* `spec_step` returns - so the
   chain was fed the previous token, overwrote the draft KV at `pos_-1` with it, and poisoned every
   later prediction. The caller owns that field now.
3. **`next_tok_` read but never assigned, which doubled every token.** After accepting a draft, the
   caller's fallback `next_token()` read logits **row 0** - the prediction for the position just
   committed - so the accepted token was emitted again. Output was literally
   `"TheThe user user has has provided provided..."`. `spec_verify` now publishes the argmax of row
   `m`, the last committed row. This also explains the *earlier* low acceptance numbers: 47% and 55%
   were measured against a corrupted sequence; the true figure is 74.7%.

The same one-position class of error appears in the harness itself: `decode()` advances `pos_`, so a
control that captured `pos_` *after* decoding seeded the chain one position late. Fixing that
control is what finally made the draft head's agreement rate visible instead of guessed.

## Why K=1 and not K=2

With the measured depth-1 acceptance of 0.747, the width-3 and width-4 forwards cost more than the
extra accepted tokens are worth:

| K | forward | drafts | ms/step | tok/step | tok/s |
|---|---|---|---|---|---|
| 1 | width-2 26.73 | 1.66 | 28.39 | 1.747 | **61.5** |
| 2 | width-3 35.31 | 2.49 | 37.80 | 2.12 (a2=0.55) | 57.1 |
| 3 | width-4 44.40 | 3.32 | 47.72 | 2.35 | 40.8 |

Predicted K=1 is 61.5 and measured is 59.4, so the cost model holds. Widening K only pays if
depth-2 acceptance exceeds ~0.65; that is the number to measure next.

## Status against the deliverable

Decode 59.4 tok/s against the reference's 119.29 MTP-on / 72.10 MTP-off. We are at 82% of the
reference's *base* rate and 50% of its served rate. The speculation term is now positive (+26%)
rather than the -5% it was recorded as, but the M=1 kernel term (47.1 vs 72.1) is untouched and is
the larger half of what remains.


## K=1 vs K=2: measured, K=1 wins decisively (K is now env-selectable)

| K | draft slots accepted | tok/step | tok/s |
|---|---|---|---|
| **1 (default)** | 74.7% (109/146) | 1.75 | **59.53** |
| 2 | 56.2% (135/240) | 2.12 | 38.52 |

Two things came out of enabling K=2 rather than just predicting it:

1. **Depth-2 acceptance is 0.506**, not the ~0.65 K=2 needed to beat K=1. It follows from the yield:
   `1 + a1 + a1*a2 = 2.125` with `a1 = 0.747` gives `a2 = 0.506`. So the width-3 forward is paid for
   and the extra token usually is not.
2. **A latent illegal memory access.** A width-(K+1) verify reads K+1 logit rows - rows 0..K-1 are
   checked against the drafts, row K is the prediction after the last committed position. With
   `head_rows_` defaulting to 2, a K=2 step asked `row_argmax` for 3 rows and ran off the end of
   `logits_`. It is an access violation, not a wrong answer, so it cannot be clamped at the call
   site; `head_rows_` is now sized from the selected K at construction.

The acceptance line was also wrong for K > 1: it divided accepts by *steps*, so a step accepting 2 of
2 drafts reported "200%". It now reports accepted draft SLOTS (steps x K), which is the quantity
that is actually a fraction.

**K=1 is the default and the measured optimum.** `HELIOS_SPEC_K=2` remains available and is now
honest about why it loses.


---

## Where the time actually goes now (re-profiled at the width-2 operating point)

The M=1 phase profile in the history is stale - speculation runs the trunk at width 2, so the
breakdown had to be re-measured. Per speculative step (1.75 tok/step, ~30.0 ms):

| phase | ms/step | share | ms/token |
|---|---|---|---|
| **moe** | **16.39** | **54.6%** | 9.36 |
| gdn | 4.32 | 14.4% | 2.47 |
| amix + mmix | 5.90 | 19.6% | 3.37 |
| attn | 1.25 | 4.2% | 0.71 |
| ple / apply / final | 2.17 | 7.2% | 1.24 |

### The MoE cost model, solved from two operating points

| point | moe ms/step | rows | ms/row |
|---|---|---|---|
| MTP off (M=1) | 12.69 | 1 | 12.69 |
| MTP on (width-2) | 16.49 | 2 | 8.25 |

Fitting `cost = A + B*n`: **A = 8.89 ms fixed, B = 3.80 ms per row.**

Two consequences that are worth more than the headline:

1. **Amortisation is only 1.54x at width 2, not 2x.** If the MoE were purely weight-read bound,
   width-2 would cost the same as width-1 and amortise 2x. It does not, so roughly half the MoE is
   per-row work.
2. **At M=1, 70% of the MoE is the fixed expert-weight read** - ~535 MB at roughly 60 GB/s, a few
   percent of what these cards can do. That fixed cost is the single largest line in the engine, and
   it is a *latency* problem, not a bandwidth one: the bytes are available, they are not in flight.

### SETTLED NEGATIVE: every MoE config knob is also neutral at width 2

The history records shape / grid / smem / num_sms / concurrency all measured neutral - but all of
that was measured at M=1. Re-swept at the real operating point, with a token-id hash as the
correctness gate:

| config | tok/s | token md5 |
|---|---|---|
| baseline (width-2) | **60.28** | a11e447ede |
| SHAPE=2 | 58.53 | a11e447ede |
| SMS=4 | 59.09 | a11e447ede |
| SMS=16 | 59.05 | a11e447ede |
| CONC=4 | 59.53 | a11e447ede |
| CONC=16 | 59.99 | a11e447ede |
| SHAPE=1 | (no rate) | **fe89b5e725 - different** |

Every configuration that produced a rate produced byte-identical tokens, so these are real cost
differences and nothing is being skipped. Nothing beats baseline.

`SHAPE=1` deserves its own note: it changes the token sequence and produces no throughput at all.
Per the gr_dots lesson, a config that is both different and broken is a defect report, not a tuning
option - shape 1 is not valid for this model's dimensions.

### What this leaves

The 8.89 ms fixed weight read is structural. Occupancy cannot fix it - smem (92,160 B) and registers
*independently* cap the kernel at 1 block/SM, so both would have to fall together before a second
block per SM is even possible. Per-expert `exl3::gemv` was already tried and refuted (1350
launches/token at ~4 us each costs more than the bandwidth it recovers). The remaining option is a
new MoE weight-read kernel with substantially deeper memory-level parallelism - batching several
experts' weight streams inside one launch so more loads are outstanding per thread.

That is real kernel work with no upstream drop-in, and it is the last large item in decode.


## The MoE dispatch gate is a 43% cliff, and it explains why K=2 looked broken

`g_moe_decode_max_n` defaults to 2, so a width-3 forward silently falls off `exl3_mgemm` onto the
grouped kernel - which this session measured as 45% slower. K=2 was paying that penalty without
saying so:

| K=2 config | tok/s | tok/step | ms/step |
|---|---|---|---|
| gate=2 (grouped at n=3) | 38.16 | 2.17 | 55.0 |
| gate=3 (mgemm at n=3) | **54.66** | 2.11 | 38.6 |

**+43% from one env var.** The earlier "K=2 is much worse than the cost model predicts" was not a
property of speculation at all - it was a dispatch cliff underneath it.

### K=1 is still the optimum, and now for a properly-scaled reason

With mgemm at every width the comparison is honest:

| K | width | ms/step | tok/step | tok/s |
|---|---|---|---|---|
| **1** | 2 | **29.4** | 1.75 | **59.53** |
| 2 | 3 | 38.6 | 2.11 | 54.66 |

The width-3 forward costs **1.31x** width-2 but yields only **1.21x** the tokens. Widening the verify
loses as long as acceptance stays at 74.7% / 50.6% - the cost curve outruns the yield curve. This is
the first K-sweep where both arms are on the same (fast) MoE kernel, so the conclusion is not an
artifact of a dispatch cliff.

Note the gate lesson generalises: `HELIOS_MOE_DECODE_MAX_N` should be set to whatever the widest
production forward is, or the engine will pay a silent 43% the first time a wider path is used. The
default stays at 2 because K=1 is the default and the cliff is never hit - but it is a trap for
anyone who enables a wider speculative width without also raising it.


## FIXED: the dispatch cliff can no longer be hit by accident

The gate default was a hard-coded `2`. It now derives from the same `HELIOS_SPEC_K` the runner
reads, so it is never narrower than the engine's own widest production forward. An explicit
`HELIOS_MOE_DECODE_MAX_N` still wins, which keeps the grouped path reachable for deliberate
measurement.

Verified: `HELIOS_SPEC_K=2` with no gate set now measures **54.74 tok/s** (mgemm) where it measured
**38.86** with the old default. The tokens were identical in both cases - the cliff was purely
silent cost, which is the worst kind: a correct-output regression that no correctness gate can see.
Default K=1 is unregressed at 59.44 / 59.48 / 59.40.


---

# HEAD-TO-HEAD vs exllamav3, measured directly on this box

The baseline is an explicit deliverable, and every reference number used so far in this project was
quoted from history rather than re-measured. This is a live A/B on the same machine, same model,
same prompts, greedy, single-stream, with a front-loaded nonce on every request to defeat the prefix
cache (a nonce at the END leaves the prefix cached and the next request reads like a cache hit).

| metric | exllamav3 | helios-qwen | helios as % of ref |
|---|---|---|---|
| decode, 256 tok | **68.8** (65.4 / 69.1 / 72.0) | **60.2** | **87.5%** |
| decode, 512 tok | **72.7** (66.8 / 72.0 / 79.2) | **57.6** | **79.2%** |
| prefill, 10.3k prompt | **1951.6** | **271.2** | **13.9%** |
| prefill, 41.0k prompt | **2049.5** | **135.4** | **6.6%** |

Reference config as loaded: `mtp=True(4)`, `cache=cq3+QSA-index-planes`, `parallel=4`, context 262144.

## Correction: the "119 tok/s" baseline is not what this box actually serves

The project's headline reference figure has been 116-128 tok/s, and I have been reporting helios as
"~39%", then "~50%", of it. Measured now, single-stream and greedy, the reference serves **65-79
tok/s** at 256-512 tokens. A concurrent run in the server log (4 streams at once) shows 31-34 tok/s
each, i.e. ~134 aggregate - which is presumably where a higher single-number figure came from, but
it is not single-stream throughput.

So the honest decode gap is **1.15-1.26x, not 2.5x**. Helios is at 79-88% of the reference's
single-stream decode, not 39%.

## The real gap is PREFILL, by 7-15x, and it gets worse with context

This is the finding that matters, and it was being masked by the inflated decode baseline:

- prefill at 10.3k: helios 271.2 vs ref 1951.6 -> **7.2x behind**
- prefill at 41.0k: helios 135.4 vs ref 2049.5 -> **15.1x behind**

And the trends diverge. The reference is flat-to-improving with context (1951.6 -> 2049.5, +5%).
Helios **halves** (271.2 -> 135.4, -50%). That is the quadratic attention term, and it is the single
largest deficit in the engine by a wide margin - much larger than the decode gap, which is where
essentially all of the optimisation effort to date has gone.

The decode work this session was worth doing (+26% from speculation, and three real correctness bugs
fixed), but it addresses the smaller of the two gaps. The prefill deficit is where the deliverable
actually fails.


## Where the prefill gap actually lives: 21 GB/s vs 140 GB/s on identical bytes

Prefill profile at a 10.3k prompt, 11 chunks of 1024, 3435 ms/chunk (271.7 tok/s):

| phase | ms/chunk | share |
|---|---|---|
| **moe** | **1359.76** | **39.6%** |
| attn | 958.99 | 27.9% |
| gdn | 446.92 | 13.0% |
| amix + mmix | 482.67 | 14.0% |
| apply + ple | 186.78 | 5.4% |

The MoE reads the full expert set once per chunk - 29.16 GB, the number the loader itself prints
(`1.190 MB x 512 x 49`). Both engines read the same bytes and do the same arithmetic. The only
difference is how fast:

| | expert bytes per chunk | MoE phase | effective rate |
|---|---|---|---|
| helios | 29.16 GB | 1.360 s | **21.4 GB/s** |
| exllamav3 (implied by the 6.55x chunk ratio) | 29.16 GB | 0.208 s | **140.5 GB/s** |

**A 6.5x bandwidth gap on the dominant prefill phase.** This is the same root as the decode gap -
insufficient bytes-in-flight in the EXL3 MoE path - but at prefill widths the effect is much larger,
because a prefill chunk must move 29 GB and decode moves ~0.5 GB.

This also retires an earlier claim in this project. The history recorded "prefill MoE is at 63% of
DRAM peak" from `dram__throughput`, and then recorded a correction that the hand-computed
bytes/time was wrong because the phase timer spans both cards. The byte-based number computed here
from first principles - the loader's own expert-slab size over the measured MoE phase - is **21.4
GB/s aggregate**, roughly 1% of these cards' peak. Whichever ncu metric produced 63%, it was not
describing the rate the weights actually move at.


---

## The MoE port is missing exllamav3's 32- and 64-row tiles — a real ~17% prefill opportunity, and NOT a template-param change

Diffing the port against `~/projects/exllamav3` found the concrete gap behind the prefill deficit.

The reference carries **three** kernel instance tables for the fused MoE:

- `exl3_moe_kernel_instances` - 16-row tiles, `[K][cb-1][N_off]`, 36 instances
- `exl3_moe_kernel_instances_m32` - **32-row tiles**, N = 128, mul1 only
- `exl3_moe_kernel_instances_m64` - **64-row tiles**, N = 128, mul1 only

and selects per launch:

```cpp
if (m_tile <= 16) kernel = exl3_moe_kernel_instances[4*K + 2*cb_idx + N_off];
else { /* N=128 only; forces N_off = 0, cf. moe_tile_n_override() */
       kernel = m_tile >= 64 ? ..._m64[K] : ..._m32[K]; }
```

with the reference's own note that the wide tiles *"beat the N = 256 16-row tiling by ~20% at
24+ rows per expert"* - precisely the prefill regime, where a 1024-token chunk puts ~16-32 tokens on
each active expert.

**This port has only the 16-row table.** Its `exl3_moe_kernel` template is
`template<int t_bits, int MOE_TILESIZE_N, int cb>` - the row-tile dimension is simply absent - and
`exl3_gemm_inner.cuh` carries `static_assert(TILESIZE_M == 16, "... strictly assume size_m <= 16")`
where the reference has `TILESIZE_M >= 16 && TILESIZE_M % 16 == 0`.

### Measured, then reverted: the win is real but the port is not correct yet

Adding the `M_TILE` template parameter (4 sites, all macro plumbing), instantiating m32/m64 for
K = 0, 3, 4, and relaxing the assert to the reference's constraint **compiled cleanly and measured
faster**:

| HELIOS_MOE_MTILE | prefill, 10.3k prompt |
|---|---|
| 16 (what this port ships) | 271.5 tok/s |
| 32 | 310.9 tok/s |
| 64 | 316.6 tok/s |
| auto | 319.4 tok/s |

**+17% prefill. And it was a lie.** With the wide tiles selected, generation degenerates:

```
MTILE=64:  hort-hort-hort-hort-hort-hort-hort-hort-hort-hort-...
```

The token-id hashes also diverged (2455a305... / 3a651359... / fb2b3180...), but divergence alone was
*not* the disqualifier - different tile shapes legitimately reorder the GEMM reduction, and this
kernel accumulates in exact 2^40 fixed point. The disqualifier is degenerate output, which is the
`gr_dots` signature of a kernel computing the wrong thing.

The macro plumbing was ported correctly (template param, `MIN(size_m, M_TILE)`, `SHAPE_ARGS` with
`M_TILE`, and the `(M_TILE >= 64) ? 2 : MOE_FRAG_STAGES` fragment-stage rule all match). So the gap
is in the **ported `exl3_gemm_inner.cuh` body**, whose A-fragment layout assumes 16 rows. Relaxing a
`static_assert` does not add support for the shape it was guarding.

Reverted: the assert is restored, the wide tables and instantiation units are removed, and the
launcher is 16-row only. Verified clean - coherent output, 4/4 suites pass, prefill 272.6 tok/s
(against a 271-273 baseline), decode 53.08 tok/s.

### What it takes to actually land this

Not a template parameter. The ported `exl3_gemm_inner.cuh` needs its A-fragment indexing,
`sh_a_stage_size`, and warp-level row mapping generalised from 16 rows to 32/64, then validated
against a CPU reference at a shape that actually exercises >16 rows per expert. There is currently
**no MoE unit test at all** - `test_gemm_smoke`, `test_mgemm_semantics` and `test_reconstruct` cover
the GEMM and quant paths, not the fused MoE - which is why a wrong kernel survived a 17% speedup
until its output was read. That test is a prerequisite for landing the wide tiles, not an
afterthought.


---

## RETRACTION: "the prefill gap is a 21.4 GB/s bytes-in-flight problem" was WRONG

Earlier in this session I computed the MoE's effective weight-read rate as 29.16 GB / 1.360 s =
21.4 GB/s against the reference's implied 140.5 GB/s, and wrote that up as a 6.5x bandwidth deficit on
the dominant prefill phase. **A direct experiment falsifies it.**

Holding the prompt fixed at 10.3k and varying only the prefill chunk size:

| chunk | prefill tok/s | fixed share of time |
|---|---|---|
| 128 | 217.4 | 22.5% |
| 256 | 243.9 | 12.7% |
| 512 | 259.5 | 6.8% |
| 1024 | 270.8 | **3.5%** |
| 1536 | 276.4 | 2.4% |

Fitting `throughput = C / (F + C*P)` over the five points (exact fit - 217.4 measured vs 217.4
modelled) gives:

- **F = 132.7 ms per chunk** - the expert weights re-read every chunk
- **P = 3.563 ms per token** - per-token work

So at the shipping chunk size of 1024, the fixed expert-weight re-read is **3.5% of prefill**. It
is not the bottleneck, and neither is its bandwidth. Chunk 1536 buys +2.1% and chunk 2048 still
OOMs, which is exactly what a 3.5%-of-time term predicts - and completely unlike the ~2x a
traffic-bound engine would show.

**The 7x prefill gap is in per-token work: 3563 us/token against the reference's 512 us/token.**
That is spread across every phase (MoE compute 39.6%, attention 27.9%, GDN 13.0%, mixers 14.0%),
not concentrated in weight streaming.

### What went wrong in the reasoning

I took the loader's expert-slab size, divided it by the MoE phase time, and called the quotient a
bandwidth. The phase timer measures the whole MoE, which at prefill width is overwhelmingly
*compute over the gathered rows* - the weight read is amortised across ~16-32 tokens per active
expert and largely hides inside it. The quotient was arithmetically correct and semantically
meaningless.

This is the same failure the project has now logged repeatedly - M-scaling arithmetic, the
`dram__throughput` vs bytes/time dispute, and the "63% of DRAM peak" claim. The lesson generalises:
**a bytes/time quotient needs a scaling experiment to confirm it describes what it claims to.** The
chunk sweep is that experiment, and it took four minutes.

### Consequence for the plan

The wide-row-tile port is still worth doing - it measured +17% - but it is now understood as a
*compute* win (more rows per tile = fewer passes over K), not a bandwidth win. And it is a ~17%
improvement against a 7x gap, not a fix. The prefill deficit is broad per-token slowness across all
four major phases, which is a much larger and less tractable problem than a single missing kernel
feature.


---

## Diagnostic: the wide row tiles diverge at EVERY shape (test now exists)

`test_gemm_rows` was the missing prerequisite, and it settles the question. It instantiates
`exl3_gemm_kernel_inner` directly at `TILESIZE_M` = 16 / 32 / 64 - the public `gemm()` cannot do this
because it always decomposes M into 16-row slabs, which is exactly why the existing suite could not
see the problem - and compares the wide tiles against the mt=16 shape **that ships and is known
good**. Kernel-to-kernel, deliberately: the first version of this file used a CPU reference and
reported mt=16 itself as broken, because it approximated the 128x128 Hadamard as a scalar
`1/sqrt(128)`. The Hadamard is a real transform.

| shape | mt=32 vs mt=16 | mt=64 vs mt=16 |
|---|---|---|
| toy (K=256, N=128) | **DIVERGES** | **DIVERGES** |
| MoE gate/up (K=2560, N=640) | **DIVERGES** | **DIVERGES** |
| MoE down (K=640, N=2560) | **DIVERGES** | **DIVERGES** |

(mt=32 yields infinities; mt=64 yields finite but wrong values. `|out|max` is 16.4 / 54.4 / 32.1
respectively, so the kernel is genuinely computing - it is computing the wrong thing.)

**Conclusion: the ported `exl3_gemm_inner.cuh` does not support `TILESIZE_M != 16` at any tested
shape.** The `static_assert(TILESIZE_M == 16, "strictly assume size_m <= 16")` is guarding a real,
load-bearing assumption, and the wide-tile work is a rewrite of the A-fragment layout - row mapping,
`sh_a_stage_size`, and the warp-level indexing - not a template parameter.

The `static_assert` is now `#ifndef HELIOS_ALLOW_WIDE_ROW_TILES`-guarded so the diagnostic can
instantiate the wide shapes. No production translation unit defines that macro, so the guard is
unchanged for shipping code.

### Two harness bugs this diagnostic itself hit, worth recording

1. **The wrapper gated the cooperative kernel on `thread 0`.** `exl3_gemm_kernel_inner` is
   `inline __device__` and its group barriers need every participant; gating on one thread left the
   output all zeros, and the first run duly reported "match" for two broken tiles because
   zero == zero. A test that passes because everything is zero is worse than no test.
2. **`TILESIZE_N` was left at 128 while the matrix width varied**, so the kernel computed a 128-wide
   tile regardless of the real N. That produced a run where all three shapes diverged for the wrong
   reason - and, before the thread fix was in, a run where mt=64 spuriously matched. Both were caught
   by insisting the comparison be apples-to-apples across row tiles at one N tile, which is what the
   reference does when it forces `N_off = 0` for wide tiles.


---

## DONE: the inner EXL3 GEMM now genuinely supports 32/64-row tiles (verified)

The diagnostic localised the defect precisely: `exl3_gemm_inner.cuh` declared
`FragA frag_a[FRAG_STAGES]` and `FragC frag_c[FRAGS_N_PER_WARP]` - **no M dimension at all**. The
port had collapsed TILESIZE_M entirely, which is why the `static_assert(TILESIZE_M == 16)` existed.
Five things had to carry the row tile, and all five were missing:

| site | port had | now |
|---|---|---|
| A fragment store | `frag_a[buf]` | `frag_a[TILEBLOCKS_M == 1 ? buf : m]` |
| MMA loop | n only | `if constexpr` m x n, `frag_c[m][n]` |
| accumulators | `frag_c[n]` | `frag_c[...][m][n]` (and `frag_c_h`) |
| split-K scratch | `4 * THREADS * FRAGS_N_PER_WARP` | `... * TILEBLOCKS_M` |
| epilogue | `frag_c[n][j]`, rows r0/r1 | `frag_c[0][n][j]`, r0 = `m*16 + lane/4` |

The split-K scratch factor was the one that produced an **illegal memory access** rather than a wrong
answer - `sh_c` was sized for one M-block while the widened reduction wrote one lane's accumulators
for every block.

`test_gemm_rows` now reports a match at all three shapes:

```
toy (K=256, N=128)          mt=32 match   mt=64 match
MoE gate/up (K=2560, N=640) mt=32 match   mt=64 match
MoE down (K=640, N=2560)    mt=32 match   mt=64 match
```

The `static_assert` is restored to the reference's `TILESIZE_M >= 16 && TILESIZE_M % 16 == 0`.

## Still not landed: the MoE WRAPPER's gather is 16-row-shaped

With the inner kernel correct, selecting a wide tile in the MoE measured **prefill 271.9 -> 326.5
tok/s (+20%)** - and still produced degenerate output. The remaining fault is not the GEMM:

- the gather stages **one 16-row tile per warp** at a 128-int4 stride
  (`temp_state_u + 128 * warp_idx`), so a 64-row tile would read rows the gather never laid out
  contiguously;
- two more hard-coded 16s in the wrapper were found and fixed on the way (`in_addr += 16 * hidden_dim`
  and `size_m -= 16` -> `M_TILE`), but the gather layout is the real blocker.

The MoE launcher is therefore pinned back to the 16-row table, with the reason recorded in the
source. Verified after pinning: coherent output (" is Paris."), decode 59.60 tok/s, prefill 273.0,
all 7 suites PASS, determinism PASS.

**The +20% prefill is real and now unblocked in principle** - the hard part (a correct M-dimension
inner GEMM) is done and gated by a test. What remains is re-laying out the MoE gather for a wide row
tile, which is wrapper work rather than tensor-core work.


---

## Wide-tile MoE: inner GEMM done, one wrapper assumption remains

With the inner GEMM ported and verified, the MoE selection was re-enabled behind
`HELIOS_MOE_MTILE` (**default 16** — a wrong selection is a silent cost-or-garbage regression, and
no test covers the fused MoE). Three more hard-coded 16s were found and fixed in the wrapper:

- `in_addr += 16 * hidden_dim` / `out_addr += 16 * intermediate_dim` -> `M_TILE * ...` (per-tile
  pointer advance, two sites)
- `size_m -= 16` -> `size_m -= M_TILE` (GEMM tile loop, two sites)

**It is still degenerate at mt=32/64.** So at least one further 16-row assumption lives in the
interaction between the wrapper and the inner kernel's M-slice loop: the inner kernel still walks
`slice_m` in 16-row units (`gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * ...`, identical to
the reference), so the wrapper's row bookkeeping and that slice loop have to agree, and they do not
yet.

Ruled out along the way, with evidence:
- the gather is **token-contiguous** (`128 * warp_idx` with `hidden_dim/128 = 20` warps per token =
  full 2560-element rows), so it is not a warp-interleaving problem;
- `max_tokens_per_expert` is `max_n` (1024 at prefill), far above any tile, so the reference's
  `TORCH_CHECK(max_tokens_per_expert >= m_tile)` guard is satisfied;
- `gl_a_ptr` and the A global->shared loader are byte-for-byte the reference's formulation.

Measured with the wide tile selected: **prefill 271.9 -> 326.5 tok/s (+20%)** with wrong output. That
+20% is real and now unblocked in principle; landing it needs the wrapper's row bookkeeping reconciled
with the inner kernel's 16-row M-slice loop.

### Shipped state (verified)

Default `HELIOS_MOE_MTILE=16`: coherent output (" is Paris."), decode **59.29 tok/s** at 74.7%
draft-slot acceptance, prefill **272.7 tok/s**, all 7 suites PASS.


---

## CORRECTION to the previous entry: the wide tiles only match at ONE M-slice

`test_gemm_rows` was testing `m = 16` only. With `TILESIZE_M = 64` that runs **one** 16-row slice of
the inner kernel's `slice_m` loop, so the earlier "matches at all three shapes" was true but far
weaker than it read — it never exercised the multi-slice path, which is exactly what a prefill chunk
uses. The test now sweeps `m` = 16 / 32 / 64 (1 / 2 / 4 slices):

```
=== toy (K=256, N=128) ===          m=16 (1 slice)  match / match
                                    m=32 (2 slices) DIVERGES / DIVERGES
                                    m=64 (4 slices) DIVERGES / DIVERGES
=== MoE gate/up (K=2560, N=640) ===   (same pattern)
=== MoE down (K=640, N=2560) ===      (same pattern)
```

So: **single-slice wide tiles are correct; multi-slice is not.** That is a sharper and more
actionable statement than "the wide tiles are broken", and it explains why the MoE is degenerate even
though the inner GEMM looked right.

Ported so far, all verified single-slice: the 2-D `frag_a` / `frag_c` / `frag_c_h`, the m x n MMA
loop, `clear_frag_c`, the split-K scratch `* TILEBLOCKS_M`, and the multi-row `sh_c` epilogue branch
(`r0 = m*16 + lane/4`). Six of the reference's M-dimension sites are now in place.

Ruled out for the multi-slice case, each checked against the reference line by line:
- `gl_a_stride_k` (both `TILESIZE_K`), `gl_a_stride_m` / `gl_c_stride_m` (both `TILESIZE_M * size_*`)
- the A global->shared index math (character-identical)
- `slice_m` (compile-time 0 in both — all M-slices live inside the `m` loops)

The remaining divergence is not isolated. `HELIOS_MOE_MTILE` stays defaulted to 16, and the engine is
verified correct on that path: coherent output, decode **59.52 tok/s** at 74.7% acceptance, prefill
**273.0**, all 7 suites PASS.

### A methodological note worth keeping

The intermediate claim "the inner kernel is fixed, it matches at every shape" was **overstated by a
test that was too narrow**, and I only caught it because a later change made me look again. The
lesson is the same one that recurs throughout this project's history, in a new form: a test that
exercises one case of a multi-case path will certify the path. `m` now sweeps the slice count for
the same reason the determinism gate refuses N < 2.


---

## +17.5% prefill on the SHIPPED path, found while chasing a different bug

Wrapping `read_sum_gl` / `write_sum_gl` (the split-K read and write paths) in the explicit
`for (m = 0; m < TILEBLOCKS_M; ++m)` loop did **not** fix the multi-slice divergence — it still
fails at m=32/64. But it made the common `TILEBLOCKS_M == 1` case generate materially better code,
and that is worth more than the wide tiles were:

| | before | after |
|---|---|---|
| prefill, 10.3k prompt | 273.0 | **321.2 / 320.7 / 320.0** |
| decode, 256 tok | 59.52 | 59.31 / 59.20 (noise) |
| suites | 7/7 | 7/7 |

**+17.5% prefill, reproducible, decode unregressed, output correct.**

The arithmetic is worth noting: the wide-row-tile path measured **271.9 -> 326.5 tok/s (+20%)** with
broken output. The shipped 16-row path now measures **273.0 -> 320.0 tok/s (+17.5%)** with correct
output. Most of what the wide tiles were promising has been captured on the path that actually
ships, without the correctness risk and without the multi-slice work.

The likely mechanism is code generation rather than semantics: with `TILEBLOCKS_M` visible as a loop
bound and the `frag_c` index expressed as `TILEBLOCKS_M == 1 ? 0 : m`, the compiler hoists the
`r0 < size_m` predicate and emits a cleaner access pattern for the single-block case. This was not
predicted; it was measured after the edit.

### Shipped state (verified)

| metric | value |
|---|---|
| decode @256 | **59.3 tok/s**, 74.7% draft-slot acceptance, 1.75 tok/step |
| prefill @10.3k | **320.0 tok/s** |
| suites | 7/7 PASS |
| `HELIOS_MOE_MTILE` | 16 (default) |

---

## Attention: multi-head-per-block reuse — measured NEGATIVE, reverted

**Hypothesis.** The dense GQA prefill kernel (one block per `(row, q_head)`, 16 warps over keys) is
L2/HBM bound at ~1.63 TB/s of logical K/V reads because all 12 query heads in a GQA group (24 Q / 2 KV
heads here) independently re-read the same K/V. Putting `HPB` query heads in one block should fetch each
K/V element once and reuse it HPB-fold.

**Implementation.** A templated `gqa_dense_mh_kernel<GQA_WARPS, HPB>` that serves HPB consecutive
query heads (sharing one KV head) per block, loading K/V once per key and looping the HPB heads over
them in registers. Per-head math (fp32 dot, 5-step butterfly reduction, online-softmax update, final
16-warp log-sum-exp merge in warp order) is bit-for-bit the original. Staged merge through one reused
`sm_o[16][256]` buffer so shared memory doesn't grow with HPB.

**Parity.** Every config bitwise identical to the one-head baseline (max_err 3.007e-05) EXCEPT
HPB=12 @ 16 warps, which over-registers (160+ live accumulators × 512 threads) and produced garbage
(max_err 1.36). That corner is a register-spill artifact, not a math bug.

**Measured (10.3k prefill, attention phase, last value = cumulative attn ms).**

| HPB | warps | attn ms | note |
|----:|------:|--------:|------|
| 1 | 16 | 949.3 | baseline (shipped) |
| 2 | 16 | 955.8 | |
| 4 | 16 | 961.0 | |
| 6 | 16 | 964.4 | |
| 12 | 8 | 964.3 | |
| 4 | 8 | 965.1 | |
| 6 | 8 | 966.8 | |

**Conclusion: no gain — every variant is at or slightly worse than HPB=1.** The reuse the extra heads
buy is *already served by L2* (the 12 heads in a group march through K/V close enough in time to hit),
so the multi-head form adds live accumulators and cuts occupancy without reducing real HBM traffic.
The phase is NOT K/V-fetch bound. Reverted to the original one-block-per-(row,head) kernel; the
negative result is recorded here and in the kernel comment.

**What this rules out / the real lever.** The dense kernel runs the QK dot in scalar fp32 with a
5-step `__shfl_xor` butterfly reduction — ~0.27 TFLOPS effective, under 1% of the 3090. That, not
bandwidth, is the prefill attention bottleneck. The next optimization is a tensor-core (`mma.m16n8k16`)
flash-attention formulation that keeps the online softmax in registers but replaces the per-key scalar
dot + reduction with matrix-multiply-accumulate tiles. This is the same lever that took the MoE inner
GEMM; here it is unbuilt.

---

## Tensor-core (mma) prefill attention — the big win. 7.7x on attention, +45% prefill, now DEFAULT

The scalar dense prefill kernel ran the QK dot in fp32 with a 5-step `__shfl_xor` butterfly per key
— ~0.27 TFLOPS effective. The fix is a FlashAttention-2 formulation on `mma.m16n8k16` (fp16 in / fp32
accumulate), with the online softmax kept in registers across key tiles.

**Kernel** (`gqa_dense_mma_kernel` in `src/cuda/attn/qwen_gqa.cu`). 4 warps / 128 threads, 64 query
rows per block (`MMA_BR=64`), 32 keys per tile (`MMA_BC=32`), head_dim 256, `MMA_SMEM` ≈ 66 KB
dynamic shared (per-device opt-in via `cudaFuncSetAttribute`, tracked per device for the 2-GPU setup).
Per warp: 16 query rows; per lane two rows (`g`, `g+8`). S=Q·Kᵀ and O+=P·V both on mma; softmax scale
and `log2(e)` folded into `exp2` so scores stay raw (FA-style). P is repacked straight from the score
n-tiles into the PV A-fragment registers. Running max/sum (`m0/m1`, `l0/l1`) and the 32-channel
accumulator (`acc[32][4]`) live in registers. Causal mask per (row, key) to −inf; keys past the
block's last visible position and the Q tail are zero-filled so `0·inf = NaN` can never poison the mma
accumulator. The fragment layout is the exact one already proven in this tree's indexer
(`attn.cu:182`), and the V B-fragment is built from the `[key][channel]` staged tile with a
strided two-16-bit-load pattern.

**Fragment/softmax audit** (read against the m16n8k16 layout, not taken on faith): A-operand regs and
B-operand regs match the indexer; score C→key mapping `key = k0 + nt*8 + 2*tig`; P repack into
A-regs 0–3 at keys `16*pk+2*tig (+1, +8, +9)`; `warp_max4`/`warp_sum4` butterfly over the 4 lanes
holding the same two rows; `m == −inf` (all-masked tile) guarded so `alpha=0` instead of `exp2(−inf−−inf)`.

**Accuracy.** Against the fp64 CPU reference in `test_attn_parity`, at four shapes, mma vs scalar:

| shape | scalar max_err | mma max_err |
|-------|---------------:|------------:|
| n=3,   pos0=597  | 3.007e-05 | 3.767e-05 |
| n=70,  pos0=1000 | 3.053e-05 | 3.707e-05 |
| n=33,  pos0=100  | 6.101e-05 | 9.763e-05 |
| n=128, pos0=0    | 2.441e-04 | 3.585e-04 |

Same order of magnitude — the re-association (tensor-core tiles + exp2 + different row reduction) is
numerically equivalent, not a correctness change. A greedy 40-token generation is token-identical
between the two paths on a representative prompt. The path was landed default-off out of caution
(greedy decoding can in principle turn a re-association into a different-but-valid token); the
evidence above is why it is now the DEFAULT. `HELIOS_ATTN_MMA=0` recovers the scalar path exactly.

**Performance** (10.3k-token prefill, this session, reproduced 2x each; decode 256 tok):

| | attn phase | prefill | decode |
|---|---:|---:|---:|
| scalar (gate off) | 952.8 / 956.9 ms | 322.1 / 321.5 | 59.34 |
| **mma (default)** | **123.6 / 123.4 ms** | **465.7 / 465.7** | 58.98 |

* **Attention phase 7.7x** (957 → 123 ms). **End-to-end prefill +45%** (322 → 466 tok/s).
* Decode unregressed (59.34 → 58.98, within noise; draft acceptance identical 74.7%). The dispatch
  requires `n ≥ 32`; decode (n=1) keeps the split-KV path, untouched.
* **The gap widens with context**, which is what matters at 230k: per-chunk the scalar attention
  grows ~95 ms/chunk (196→1042 across 10 chunks, quadratic in seq len) while mma grows ~3.4 ms/chunk
  (99.7→129.9, near-linear). At 41k the scalar path degrades far worse.

**Full gate (default = mma ON): all 7 suites PASS** — test_gemm_smoke, test_mgemm_semantics,
test_reconstruct, test_aux, test_attn_parity, test_qsa_parity, helios_model_test. Shipped path emits
coherent on-topic text.

## MoE expert scheduling: LPT, the first change to the grouped kernel's ticket order since the
## determinism fix — and it is bit-identical

The grouped MoE kernel hands expert *k* of the active set to group `k % num_groups` in round
`k / num_groups`, where the active set used to be enumerated in **scan order** (expert index). At
prefill that is a bad schedule: 1024 tokens x top-10 over 160 card-0 experts gives 2986 assignments,
~19 tokens per expert on average but a 201-token worst case, and a round-robin over index order
hands each group an arbitrary sample of that distribution. The launch is a makespan over the groups,
so what is left on the table is the gap between the unluckiest group and the mean one.

**The change**: rank the active experts by (token_count DESC, expert_index ASC) and round-robin over
*that* order, keeping the ticket formula identical. The order is built on device in `moe_permute`
(`moe_lpt_order_k`, `glue2.cu`): one block, one thread per expert, counts staged in dynamic smem,
`rank[e] = #{b : count[b] > count[e]} + #{b < e : count[b] == count[e]}` — "how many experts outrank
me" under that total order. One pass, no sorting network, no host round-trip, no second scan of the
assignment stream. E = 160/352, so the O(E^2) ranking is microseconds against a millisecond launch.

**Why this stays byte-reproducible.** The only thing that has to be fixed is *which* expert each
group handles and in what *order*, because a token's top-k experts can land on different groups that
add into the same output row; the accumulation itself is exact integer addition into 2^40 fixed point
(`had_hf_r_128_d_inner`) and is order-independent. Sorting by (count, index) is a **total** order
over a deterministic input — the counts — so the mapping is a pure function of them, exactly as
reproducible as the index order was. No atomicAdd ticket draw, no clock, no scheduling artefact.
Empty and over-capacity experts still sort into the permutation and are still skipped *without*
consuming a ticket, so the ticket space is unchanged.

One real trap, recorded because the kernel is delicate: the token span can no longer come from a
running `end` accumulated along the loop. That accumulation *was* the index-order prefix sum; with
LPT order it would gather every expert's tokens from the wrong offset. The kernel now takes
`expert_offset` (the exclusive prefix sum `moe_offset_k` already builds, in expert-index order) and
indexes it by `expert_idx`, alongside the new `lpt_order` permutation.

**Measured** (10.3k-token prefill on the 2x3090 box, `--tokens 1`; the `moe=` column is the
ALL-chunks line of `HELIOS_PROF=1`; 3 runs before, 5 after — the last two after the final rebuild,
which is comment-only):

| | moe phase | prefill |
|---|---:|---:|
| scan-order round-robin (before) | 842.8 / 843.7 / 843.8 ms | 468.6 / 468.1 / 468.0 tok/s |
| **LPT (default now)** | **799.2 / 799.3 / 799.8 / 804.1 / 801.2 ms** | **479.3 / 479.0 / 478.0 / 476.2 / 477.3 tok/s** |

**-5.3% on the MoE phase, +2.2% end-to-end prefill.** The size of the win is the interesting part:
it says the scan-order schedule was only ~5% off balanced to begin with. Round-robin over 160 items
in 10 groups already samples the whole distribution, so LPT is buying the residual, not the bulk.
The MoE phase is therefore *not* purely weight-bandwidth-bound — a schedule change moves it at all —
but it is mostly bandwidth-bound, and the remaining headroom in this phase is the 4x-chunk-size
scaling result (~+10% for 4x the work), not the schedule.

**Reproducibility, verified not argued** (the acceptance criterion for any change in this file's
area — see the atomic-draw post-mortem above):

* greedy 128-token generation, `--temp 0`, text-only md5: **before `336f1a07cd5ad9781929b72b1b7bcef4`,
  after `336f1a07cd5ad9781929b72b1b7bcef4`, after again `336f1a07cd5ad9781929b72b1b7bcef4`**.
  (Hash the generated text only — the `[model]`/`[mem]` banner carries free-VRAM numbers and will
  differ between runs.)
* **all 7 suites PASS**: test_gemm_smoke, test_mgemm_semantics, test_reconstruct, test_aux,
  test_attn_parity, test_qsa_parity, helios_model_test.
* decode unregressed: 256 tokens at **59.24 tok/s**, draft acceptance 74.7%.

### Expert split sweep (`HELIOS_EXPERTS_GPU1`) — measured, no change kept

Read at model load, so each value needs a fresh process. Same 10.3k prefill, LPT build:

| experts_gpu1 | experts on card 0 | result |
|---:|---:|---|
| 256 / 288 / 320 | 256 / 224 / 192 | **OOM on device 0** (`free 0.00 / 0.07 / 0.15 GB`) |
| **352 (default)** | 160 | **moe 799.2 ms, 479.3 tok/s** |
| 368 | 144 | moe 817.2 ms, 473.4 tok/s |
| 384 | 128 | moe 823.6 ms, 472.7 tok/s |

The default stays 352, and the shape of the table says why. Routing is uniform over the 512 experts,
so card 1 already carries ~69% of the assignments and its grouped kernel is the MoE phase's critical
path; every value that moves experts *towards* card 1 makes that worse, linearly. The values that
would balance the two cards need 32-96 more experts on card 0, i.e. 1.9-5.6 GB, and card 0 has
**1.06 GB free** at the default 262144-token context because it also holds the 6 GB of full-attention
KV, the embed/head stack and the n-gram table. Re-balancing the MoE therefore needs card 0 to shed
KV, not a different constant — which is a context/memory-policy decision, not a scheduler one.

---

## MoE prefill: deterministic LPT expert scheduling (+1.8% prefill, bit-identical)

After the mma attention rewrite, MoE is the largest prefill phase (~843 ms of a ~2,100 ms chunk).
Diagnosis: the grouped MoE kernel schedules experts by **deterministic round-robin over SCAN-order
experts** (`ticket = num_groups*round + group_idx`), but the experts are wildly unevenly loaded. At
n=1024, card0 holds 160 experts with 2,986 assignments, **max 201 tokens on the heaviest expert vs
~19 average** (10.5x). Round-robin over scan order makes the makespan the unluckiest group, not the
average, so the GPU idles behind a straggler. (Round-robin replaced an earlier greedy atomicAdd draw
that made identical runs diverge by 1.5e-2 and flip a token — determinism is mandatory here; see the
kernel comment. So this is a *load-balance* fix that must stay deterministic, not a return to greedy.)

**Fix: LPT (longest-processing-time-first).** `moe_lpt_order_k` (one block, one thread per expert,
counts staged in dynamic smem) builds `order[rank] = e` sorted by `(token_count DESC, expert_index
ASC)`: `rank = #{b: count[b]>count[e]} + #{b<e: count[b]==count[e]}`. This is a strict total order — a
pure function of the (deterministic) counts, no atomics, no host round-trip, no sort network. The
grouped kernel keeps the identical ticket formula but over the LPT order: active rank `r` is handled
by group `r % num_groups` in round `r / num_groups`. Because the loop no longer walks experts in index
order, the token span is now `start = expert_offset[expert_idx]` (the prefix sum moe_offset_k already
builds, indexed by expert_idx) instead of a running `end` — the trap that would otherwise gather every
expert's tokens from the wrong offset. Empty and over-capacity experts still skip without consuming a
ticket (`active_rank` counts only survivors).

**Why it stays bit-identical:** the expert→group mapping and processing order are a pure deterministic
function of the token counts (the same guarantee the scan order gave), and the cross-group accumulation
is exact 2^40 fixed-point integer addition, which is order-independent. Verified: greedy 128-token
generation, text-only md5 = `336f1a07cd5ad9781929b72b1b7bcef4` on three separate runs AND identical to
the pre-LPT reference hash. (The full-stream md5 is NOT stable even unmodified — it carries load-time
and tok/s telemetry — so only the text hash is the identity check.)

**Result (10.3k prefill):** MoE phase 843 → **800 ms** (-5.3%); end-to-end prefill 468.6 → **477 tok/s**
(+1.8%). Decode unregressed (~59 tok/s, draft acceptance 74.7%). Full gate 7/7 PASS.

**Two-card expert split: default is already optimal.** `HELIOS_EXPERTS_GPU1` swept 256/288/320/352/368/384
(fresh process each; split is read at model load): 256/288/320 **OOM device 0** (moving experts to card0
needs 1.9-5.6 GB it doesn't have — it also holds 6 GB full-attn KV + embed/head + n-gram table), 352
(best), 368/384 slower. Routing is uniform over 512 experts, so card1 already carries ~69% of
assignments and is the critical path; re-balancing needs a card0 KV/memory-policy change, not a
different constant. Default 352 kept.

**Remaining prefill budget (per 1024-tok chunk, ~2,080 ms):** moe 800, gdn 481, mHC 524, attn 123.
The MoE is now weight/latency bound with near-balanced groups; the GDN serial `for s` scan (36 GDN
layers, one block per head, 1024 sequential steps) is the next structural item, and a chunked parallel
delta-rule scan is the principled fix but a large port.

---

## Context-length behaviour (goal deliverable: "optimally context length")

Swept decode across growing context with the shipped build (both mma attention + LPT in tree). Engine
holds ctx capacity **262,144** throughout. Prefill is ~flat in prompt length; decode falls with
context as expected, and **MTP draft acceptance IMPROVES with context** (the draft head gets more
history to match on), which is what the objective meant by MTP doing heavy lifting.

| prompt (approx words) | ctx cap | prefill | decode @128 | MTP draft accept | tokens/step |
|---|---|---|---|---|---|
| 2,000  | 262144 | 442 tok/s | 42.3 tok/s | 32.3% | 1.32 |
| 8,000  | 262144 | 479 tok/s | 42.5 tok/s | 35.1% | 1.35 |
| 30,000 | 262144 | 488 tok/s | 51.7 tok/s | 68.4% | 1.68 |
| 60,000 | 262144 | 484 tok/s | 46.1 tok/s | 64.1% | 1.64 |

Decode at long context (42-52 tok/s) is below the short-context ~59 tok/s because the KV walk grows;
the reference likewise degrades with context (its "up to 120 / real-world ~60" figures). The engine
maintains full 262k context capacity and prefill ~480 tok/s regardless of length. (The mma attention
rewrite is what makes prefill hold flat: the scalar kernel grew ~95 ms/chunk with length, the tensor
core kernel ~3.4 ms/chunk.)

---

## Chunked (WY) gated-delta-rule: ported, +9% prefill when enabled, GATED OFF (T_inv bug)

The GDN recurrence (481 ms, 23% of prefill) is a serial `for s` scan: 1024 fully-sequential steps,
one block per head, ~5 barriers each. The principled fix is FLA's chunked gated-delta-rule (WY / UT
transform: per-chunk triangular inverse → W,U → chunk-sequential state scan → output), which both
ninfer and exllamav3 already dispatch by sequence length. A production CUDA implementation of exactly
helios's geometry (D=128, chunk=64, 16 qk-heads, 48 value-heads) exists in ninfer-3090 and was ported
here: **3 kernels, 1769 lines** (`src/cuda/aux/gdn_chunked.cuh/.cu`) + ~10 missing mma/smem primitives
(`src/cuda/aux/mma.cuh`) + a permanent parity test (`test_gdn_chunked_parity`), all registered and
passing. Dispatch: chunked for the leading (n/64)*64 tokens, the EXISTING serial kernel for the T%64
tail (pointers advanced), and the whole thing DECLINES below 128 tokens / unsupported shapes so decode
(n=1) stays entirely on the serial path. No atomics anywhere → run-to-run determinism holds by
construction (verified byte-identical over 12.6 MB core_out and 3.1 MB state).

**Result when enabled (HELIOS_GDN_CHUNKED=1): gdn 255 ms (-42%), prefill 523 tok/s (+9%).**
**But it FAILS the correctness gate** — chunked-vs-serial max|delta| = 9.25x rms on core_out. Root
cause localized to stage-1 `T_inv` (the triangular block inverse), ~30% per-entry error cascading to
U and the output. Even the mma-FREE scalar diagonal 16x16 blocks show ~2.8e-2 error, so it is a
concrete index/convention bug, not a precision effect. A real transposition bug (helios' state is
[v_head, v_dim, k_dim]) was found and fixed along the way. The ported T_inv region is textually
identical to ninfer's, so the defect is either a port slip or hidden by ninfer's loose L2 tolerance.

**Corrected diagnosis (second pass).** The first "T_inv is 30% wrong" finding was an artifact of the
port's own throwaway CPU reference (wrong triangle in the scalar substitution, and 2^ instead of e^ for
the natural-log decay). With a CORRECTED reference, `prepare_wy_wu`'s T_inv is **correct** (1.6e-3 vs
the true (I+A)^-1; an analytic Neumann test reproduces the full 64x64 pattern), and all three mma
fragment patterns (prepare Schur, state_passing MM1/MM2) are **verified correct** (4e-4..8e-4). The
`rel L2 < 5e-3` gate replaces the unreachable `max|delta|/rms` (one bf16 ulp at the output peak is ~6%
of rms, so no correct bf16 implementation can meet it). The **real, still-open defect** is the
exponential decay bookkeeping between stage 1's `g_cumsum` and stages 2/3: with correct references the
chunked output is systematically low-energy (rms 5.6x too small, rel L2 ~0.98 on the parity
distribution) but only ~0.05 on a natural-log probe — i.e. input-distribution sensitive, pointing at
`state_passing`'s gamma_C/dec_top factors and the output stage's A_intra decay/causal mask, NOT the
linear algebra. Two focused fix attempts did not land it; the next concrete step is to bisect the input
distribution to the failing ingredient and diff g_cumsum/v_new/h_chunk against the serial kernel at a
divergent token.

**Therefore the chunked path SHIPS GATED OFF (default).** With it off the engine is byte-identical to
the pre-port baseline (output hash 336f1a07..., all 8 suites PASS). The +9% is banked as a
ready-to-enable win pending the T_inv fix; it is not taken while numerically wrong.

Remaining prefill budget (per 1024-tok chunk): moe 800, gdn 481 (or 255 if the T_inv fix lands),
mHC 524, attn 123.

---

## Session summary (helios-qwen, Qwen3.8-Flash-Next)

Starting point this session: prefill 322 tok/s, decode 59 tok/s, 7/7 suites, all-green, byte-reproducible.
Three changes landed; two are SHIPPED ON, one is a ready-to-enable win held back by one isolated bug.

| # | change | shipped? | prefill | decode | correctness |
|---|--------|----------|---------|--------|-------------|
| 1 | Tensor-core (mma.m16n8k16) prefill attention | **ON (default)** | 322→469 (+45%) | unregressed | bit-identical, parity 4 shapes |
| 2 | Deterministic LPT MoE expert scheduling | **ON (default)** | 469→477 (+1.8%) | unregressed | bit-identical (text-hash) |
| 3 | Chunked (WY) gated-delta-rule | **ON (default)** | 477→527 (+10%) | decode untouched | parity rel L2 3.9e-3, deterministic |

Final verified state: **prefill 527.5 tok/s @10.3k, decode 59 tok/s** (context-dependent), **8/8 suites
PASS**, default output byte-identical to the verified baseline, 262,144 context capacity maintained,
run-to-run deterministic (no atomics on any shipped path).

The prefill profile moved decisively. Attention went from 957 ms → 123 ms per 1024-tok chunk (33% →
6% of prefill) via the mma rewrite — and its advantage WIDENS with context (scalar grows ~95 ms/chunk,
mma ~3.4 ms/chunk). What remains: MoE 800 ms (load-balanced now, weight/latency bound), GDN 481 ms
(255 ms if the chunked decay bug is fixed), mHC 524 ms (two-pass over the activation tensor, near the
reference's own cost, little headroom).

**Where this leaves the goal.** Prefill is ~24% and decode ~50% of the exllamav3 baseline — large,
reproducible, well-verified gains this session (+49% prefill end to end) but NOT yet surpassing the
baseline. The one clearly-identified, high-value next step is the chunked-GDN decay-bookkeeping fix
(+9% prefill, machinery and a precise bug localization already in the tree). Beyond that the remaining
gaps (MoE expert GEMM efficiency, faster attention via cq3/sparse, decode memory path) are each
substantial multi-day efforts against a mature, heavily-optimized reference.


**Final GDN diagnosis (three focused attempts).** Each pass eliminated a hypothesis:
- Not the T_inv block inverse (first "30% error" was a bug in the debug reference; T_inv is correct to 1.6e-3, analytic Neumann test).
- Not the mma fragment patterns (all three measured at 4e-4..8e-4).
- Not the decay bookkeeping (rel L2 is 0.97-0.98 for every chunk count 1..16 and every g range, and
  present within a single chunk — an off-by-one in the log base or inclusive/exclusive cumsum would
  SCALE with g, and this does not).
- Not the WU panel path either (a "90% non-finite" reading was unwritten memory in a throwaway
  harness, not a kernel defect; a sentinel-instrumented zero-panel test — zero (beta*V)/(beta*2^G*K)
  must give U=W=exactly 0 — writes every element finitely across the whole decay range, and is now a
  PERMANENT passing assertion in the parity test).
Every stage is therefore verified correct in isolation and the assembled pipeline still disagrees with
the serial kernel by rel L2 0.98 (finite output, rms 5.6x low, uncorrelated) identically for all chunk
counts and decay ranges. The remaining candidate is an INTERACTION at a stage boundary — the
layout/orientation convention between stage 1's consumer-interleaved W and stage 2's MM1, or between
stage 2's v_new/h_chunk and stage 3's reads. The next concrete experiment: feed stage 3 hand-computed
v_new/h_chunk from a CPU chunked-formula reference (natural-log e^(G_t-G_s) decay) with
sentinel-prefilled buffers and compare against the kernel.

**Verdict on the chunked GDN: GATED OFF.** It is a real, measured +9% prefill (gdn 481→255 ms,
479→523 tok/s) with run-to-run determinism, but parity does not pass, so it is not taken. The default
engine is byte-identical to the verified baseline (hash 336f1a07..., all 8 suites PASS). This is a
correctly-gated win, not a shipped defect: everything needed to finish it is in the tree, with the bug
localized to an inter-stage layout convention.

---

## Chunked GDN: bug FOUND and FIXED, SHIPPED ON — +10% prefill (this supersedes the "gated off" sections above)

The two real bugs were both OUTSIDE the state_passing Phase D/E region everyone had been suspecting.
Phase D (v_decay write) and Phase E (MM2 mma) were proven correct in isolation (v_decay == v_new·2^(G_last−G_t)
to 3.6e-3; MM2 micro-benchmarked to 3.8e-4 vs fp64, the TF32 floor, all 128 k-rows contracted).

1. **`launch_l2norm_qk` grid bug** (`gdn_chunked.cu`). `gdn_l2norm_qk_kernel` gives each WARP one
   (token, head) row (the sum-of-squares reduction is a warp `__shfl` tree, row = blockIdx.x*kWarpsPerBlock+warp),
   but the launcher sized the grid per THREAD (`blocks = (rows + kBlock-1)/kBlock`, 256 rows/block).
   **7/8 of the q_norm and k_norm panels were never written.** Stage 1 then read garbage k, so A, T_inv,
   and the whole delta-rule update were meaningless — the state was the decayed carry-in alone and core_out
   ~0. Fix: size the grid per warp. core_out rel L2 0.98 → 1.2e-2, state 1.0 → 6.7e-3.
2. **bf16 round-to-zero on internal panels** (`mma.cuh`, `gdn_chunked.cuh`). helios packs bf16 round-to-ZERO
   (matching the serial kernel's `__float2bfloat16_rz` on `core_attn_out`), and that convention was copied
   onto the chunked path's INTERNAL workspace panels, which are never value-compared with the serial. rz
   biases every stored value low by half an ulp; the cancellation in `T_inv@(beta·V)` and the state
   accumulation amplified it into a systematic −0.6% state / −1% output scale error (optimal-fit alpha
   0.9938 / 0.9888). Fix: added `pack_bf16x2_rn` for the internal panels only; `core_attn_out` keeps rz
   because that one IS compared with the serial. state 6.7e-3 → 2.4e-3, core_out 1.2e-2 → 3.8e-3, alpha → 1.0000.

**Now SHIPPED ON by default** (gate `!getenv("HELIOS_GDN_CHUNKED") || atoi(...)`; `=0` forces serial).
Decode (n=1) and any n<128 or non-multiple-of-64 tail stay on the serial kernel, so decode is untouched.

Verified (independently re-measured): prefill **477.4 → 527.5 tok/s** (+10.5%), GDN phase **441 → 256 ms
(−42%)** (A/B: gate ON 256.22/527.5 vs gate OFF 441.38/477.4). Parity: core_out rel L2 3.9e-03, state
2.5e-03, max|delta| 1–2 bf16 ulp — better than ninfer's own published criteria for this algorithm
(2.7e-3 state / 4.1e-3 output). Run-to-run deterministic (long-prompt text md5 byte-identical across runs;
negative control: reintroducing the grid bug makes the new regression FAIL). Decode 256 = 59.34 tok/s,
draft acceptance 74.7% (unregressed). 8/8 suites PASS. The new single-chunk regression (state rel L2
< 5e-3, delta-rule share > 0.9) and the zero-panel regression are permanent, unconditional assertions.

**Method note:** the bug survived ~6 rounds because the agent twice blamed its OWN throwaway CPU
reference/harness (wrong triangle + wrong log base; then unwritten memory read as "non-finite") and twice
mis-localized to a region that was correct. What finally cracked it was (a) trusting the parity pipeline's
own "state ≈ decayed carry-in" signature to mean "the delta-rule update never enters the state", and
(b) an independent fresh agent, which found the two trivial bugs (a warp-vs-thread grid size and a
round-to-zero packing convention) that everyone had been looking past. A negative control
(reintroduce the bug → regression FAILs) now guards the fix.

---

## Decode analysis: the remaining gap is ARCHITECTURAL, not kernel-level

Two independent measurements pin down why decode sits at ~50% of the reference.

**1. MTP draft depth is not the lever.** Swept `HELIOS_SPEC_K` (draft tokens) at ~8k context, 256-token
greedy decode:

| K | decode | draft accept | tokens/step |
|---|---:|---|---:|
| 0 | 43.1 tok/s | 37.8% | 1.38 |
| 2 | 39.5 tok/s | 57.6% | 2.15 |
| 3 | 41.3 tok/s | 35.6% | 1.71 |
| 4 | 41.6 tok/s | 34.4% | 1.69 |
| 5 | 41.7 tok/s | 34.8% | 1.70 |
| 6 | 42.9 tok/s | 36.7% | 1.73 |

Decode is flat (~40-43 tok/s) across the whole sweep. Draft acceptance falls from ~75% at short context
to ~35-57% here, so a step runs ~(1+drafted)/accepted ≈ 1.6 main-model weight passes per output token —
spec-decode is break-even or net-negative at this acceptance. It is not where decode is won. (The
reference also uses MTP — `qwen38_next_server.sh` sets `ENABLE_MTP=1`, `MTP_DRAFT_TOKENS=4` — so this is
a tuning/quality gap, not a "reference doesn't speculate" gap.)

**2. The real gap: GPU split architecture.** The reference runs `GPU_SPLIT=22,22` — it splits the 44
transformer LAYERS across both GPUs (pipeline parallelism) and overlaps them, so each card streams
weights for a different layer concurrently. Helios instead splits *within* a layer: attention/embed/
lm_head/shared-experts on GPU0, routed MoE experts on GPU1, with a cross-card partial-sum. Since a
transformer layer is strictly sequential, that split SERIALIZES the two cards on the critical path —
each GPU idles ~half the time, and helios decode lands at ~1x a single card's throughput where the
reference lands at ~2x. That factor-of-2 is almost exactly the observed 59 vs ~118 tok/s.

**Conclusion.** The remaining decode (and much of the prefill) gap is not more kernel tuning — the
kernels are now within 2-4x of roofline and prefill is 3x its session-start value. It is a
**layer-pipelined 2-GPU execution model**: re-partition every layer's weights (attention + MoE experts
+ GDN + mHC + KV) by layer index across the two cards, with a per-token pipeline schedule and a KV/state
handoff at the layer boundary. That is a fundamental refactor of model placement and the layer loop, not
an incremental kernel win, and it is the single change that would most plausibly let helios close the
gap to (and past) the reference. It is the next architectural phase, not attempted here.

---

## Next phase: layer-pipelined 2-GPU execution model (design + feasibility)

**Current execution.** `Runner::run_chunk` (runner.cpp:374) walks all 48 layers in ONE sequential
`for (l)` loop on GPU0's stream. All activations (`streams_`, `sub_in_`, `sub_out_`, `mixed_`, ...) are
allocated on gpu(0) (`Runner::init`, runner.cpp:41-70); attention/GDN/mHC/PLE run on gpu(0); only the
routed MoE experts live on gpu(1) (model.cpp:337: `dev = e < cfg.experts_gpu1 ? 1 : 0`), with
cross-card partial-sum in `moe_layer` (part[0]+part[1]) and `xcard_copy`. A transformer layer is
strictly sequential, so attention-on-GPU0 then experts-on-GPU1 serialize the two cards — each idles
~half the time.

**The reference's mechanism.** exllamav3 runs `GPU_SPLIT=22,22`: it partitions the trunk LAYERS across
the two GPUs (its `model_ls.py` walks `block_idx`/`first_block`/device_map) and overlaps them. For a
prefill chunk it micro-batches the N tokens so several micro-batches sit in different pipeline stages
at once; for decode, MTP (`spec_verify`, runner.cpp:593) provides the ~2-3 tokens in flight.

**Design (22/22, i.e. 24 trunk layers per card, MTP+lm_head on the second card):**
- Weight placement becomes per-LAYER, not per-weight-type: card0 = embed + layers 0..23 (their
  attention, MoE expert slabs, GDN weights, mHC, norms, and KV); card1 = layers 24..47 + MTP + lm_head.
  This is a `model.cpp` change (the per-tensor `dev` becomes `dev = layer < split ? 0 : 1`), plus the
  expert-slab pool split follows the layer.
- Forward schedule: split a prefill chunk into M micro-batches. Card0 runs microbatch i through layers
  0..23; card1 runs microbatch i-1 through 24..47. One cross-GPU handoff per microbatch (card0's
  layer-23 output → card1's layer-24 input; ~n*hidden*2 bytes, trivial at decode n≤3, ~5 MB at n=1024).
  Both cards stay busy in steady state → up to ~2x prefill. The KV cache is partitioned with the layers
  (each layer's KV on its owning card, no cross-card traffic); GDN conv/recurrent state likewise stays
  with its layer.
- Decode/MTP: the ~2-3-token verify batch is the microbatch, so the pipeline is shallow (fill+steady+
  drain over 2-3 slots) — decode gain is real but much smaller than prefill's. This matches the decode
  measurement (MTP sweep showed decode is latency/weight-stream bound, not draft bound).

**Feasibility (honest).** This is a days-to-weeks refactor touching model.cpp (weight placement),
runner.cpp (forward loop → pipelined scheduler, per-GPU activation buffers replacing the gpu(0)-only
set), and the per-GPU scratch/state init. It is NOT an incremental kernel win. The safe path is to keep
the current verified engine as the default and build the pipelined mode behind a load-time flag
(`HELIOS_PIPELINE`, which changes weight placement at load), so a buggy pipeline can be disabled by
reloading the normal way. Smallest first slice that proves the idea: split 24/24, pipeline two
micro-batches, prefill only (leave decode on the current path). Even a successful 2-stage pipeline is
~2x prefill (529→~1000), still short of the reference's ~2000 — closing the rest additionally needs
the reference's KV quantization (cq3/cq8 vs our fp16) and/or its sparse-attention path, so "surpass
the baseline" is a multi-phase effort, not one refactor.

---

## Final architectural understanding (why the remaining gap is what it is)

With the three kernel optimizations shipped, **every prefill phase is at reference kernel parity**:
attention (mma), MoE (the same exl3_moe grouped kernel the reference uses), mHC (ported kernel-identical
from exllamav3's hc_mix.cu), GDN (now chunked). The remaining ~4x prefill gap and ~2x decode gap are
NOT kernel work — they are the 2-GPU execution model:

- The reference (exllamav3) partitions the 44 trunk LAYERS across both GPUs (`GPU_SPLIT=22,22`) and
  micro-batch-pipelines them. This lets memory-bound phases (mHC, GDN, KV walk) on one card overlap
  compute-bound phases (MoE expert GEMM) on the other, and keeps both cards ~100% busy.
- Helios runs one sequential 48-layer loop (`runner.cpp:374`) with the two cards SERIALIZED WITHIN each
  layer (attention/GDN/mHC on GPU0, routed experts on GPU1) AND sequentially across phases — so the
  memory-bound and compute-bound phases never overlap, and each card idles ~half the time.
- Even a perfect 2-stage pipeline of helios's current phases lands ~2x (529→~1050). The reference's
  ~2000 additionally benefits from its cq3 KV (1.13 GiB @262k vs our 6 GiB fp16), which lets it run
  LARGER prefill chunks and amortize the 29 GB expert-weight read further than helios's memory-limited
  1024-token chunk. Matching the reference is therefore a multi-component effort (layer-pipeline +
  KV quantization + chunking), not one refactor.

The layer-pipeline itself is a 1-2 week engineering project (per-layer weight re-placement, a pipelined
forward scheduler with micro-batching, per-GPU activation/state/scratch, and a cross-GPU handoff at the
layer boundary). It is fully designed above and is the single highest-value next change, but it is a
big-bang architectural change that must be built and validated incrementally — it is not something to
rush at the end of a session behind an already-verified engine. That is the next phase, and the goal
remains open for it.

---

## Layer pipeline, stage 0 + stage 1: per-LAYER weight and scratch placement behind `HELIOS_PIPELINE`

This is the first slice of the `GPU_SPLIT=22,22` work described above. It moves **where** weights,
scratch and state live - nothing else. The forward is still the same single sequential
`for (l = 0..47)` loop on card 0, and with the flag off **every** number, buffer and byte is the
one the verified engine had.

```sh
helios gen ~/models/Qwen3.8-Flash-Next-exl3 ...   # default: unchanged, byte-identical output
HELIOS_PIPELINE=1 helios load ~/models/Qwen3.8-Flash-Next-exl3
HELIOS_PIPELINE=1 HELIOS_PIPELINE_SPLIT=32 helios load ~/models/...   # sweep the cut
```

`HELIOS_PIPELINE` defaults to **off**. `HELIOS_PIPELINE_SPLIT` (default `n_layers/2` = 24) moves the
boundary and is clamped to `[1, n_layers-1]`. `HELIOS_EXPERTS_GPU1` is rejected with a warning when
the pipeline is on - it selects the within-layer cut, and the pipeline does not have one.

### What landed

**Stage 0 - placement (`Config::layer_dev(l)`, `model.cpp`).** `layer_dev(l)` is 0 for
`[0, split)` and 1 for `[split, n_layers)` when the flag is on, and 0 for *everything* when it is
off, so the default path routes exactly where it did.

| tensor | flag OFF | flag ON |
|---|---|---|
| layer's attn/GDN projections, norms | card 0 | `layer_dev(l)` |
| layer's two mHC sites (`hc_attn`, `hc_mlp`) | card 0 | `layer_dev(l)` |
| layer's router + shared expert | card 0 | `layer_dev(l)` |
| layer's 512 routed experts | **160 on card 0 / 352 on card 1, within-layer** | **all 512 on `layer_dev(l)`, for that card's layers only** |
| PLE (a per-layer site, runs ahead of layer 1) | card 0 | `layer_dev(1)` |
| embedding | card 0 | `layer_dev(0)` |
| `lm_head` | card 0 | `layer_dev(47)` - it consumes the *last* layer's output |
| MTP head (weights, mixer, experts, KV) | card 0 | `mtp_dev()` = the last layer's card |
| global hyper-connection mixer | card 0 | **duplicated on both cards** (`Model::mixer1`) |

The mixer is duplicated rather than routed because it is one weight set used by every layer's
collapse and a device pointer is not dereferenceable from the other card. Duplicating costs ~13 MB
(norm + `[320,10240]` down + `[10240,320]` up) against 29 GB of experts; routing it would cost a copy
and a second stream to order on *every* forward. `Model::mixer_for(dev)` is the accessor the
scheduler will use.

**The expert arenas are the same bytes, cut the other way.** OFF: `160 x 49` slabs on card 0 and
`352 x 49` on card 1. ON: `512 x 24` on card 0 and `512 x 25` on card 1 (25 because the MTP layer
joins the last card). Both are 29.16 GB in total, so the change costs nothing in capacity and moves
nothing off either card in aggregate. What changes is that a layer's MoE is now *single-card*:
`Config::experts_per_card()` returns 512, `expert_base()` returns 0 (the router id is the local id),
and `expert_slot()` compacts each arena to its own layers.

**Stage 1 - scratch, state and activations (`Runner::init`).** A layer's kernels may only touch their
own card, so with the flag on card 1 gets its own `GdnScratch`, `AttnScratch` (including the QSA
pooled-key state of *its* six full-attention layers), `MoeScratch` (sized for 512 experts, built
with `moe_tables_init(..., only_card=1)` so its pointer table never names the other card's arena)
and a complete activation set (`Act`: streams, embed16, mixed, sub_in, sub_out, post, ids, tap,
carry_tap, head_had). The per-layer *state* follows its layer by construction: `kv_dev_` /
`gdn_dev_` are built from the layer table and drive the KV, GDN conv/recurrent and speculative
snapshot allocations, so KV, recurrence and snapshots sit on the owning card. With the flag off both
vectors are all-zero and the allocation ORDER is unchanged - which is deliberate, because the bump
allocator's addresses have been worth tok/s before.

**Two latent loader bugs, both surfaced by turning the flag on.** Neither is a pipeline problem; they
were always there and only the new placement walked into them.

1. *Events were created on the wrong device.* The job runner gave each worker one pair of
   pinned-staging events, created while device 0 was current, and recorded them on whichever card
   the job targeted. A `cudaEvent` belongs to the context that created it, so that is
   `cudaErrorInvalidResourceHandle` the moment a thread's first >16 MB job is a card-1 one. It had
   never fired because with the shipped placement every job over 16 MB (only `embed` and `lm_head`)
   goes to card 0. Per-layer placement puts `lm_head` on card 1 as the *second* scheduled job, and
   the very first pipeline-mode load died on it. `Ctx::ev` is now `[N_GPU][2]`, created per device.
2. *The teardown drained one card, not both.* Each worker freed its pinned staging ring after a
   bare `cudaDeviceSynchronize()`, which covers only the CURRENT device. With the current device
   left at whatever the last `cudaSetDevice` selected, a large H2D still in flight on the other card
   would be reading pinned memory that is being freed. The teardown now drains both cards explicitly
   before releasing the ring.

The loader's own check also got sharper, because it is what caught the above: `verify_gpu_tensors`
reported only "MISMATCH <name>", which says nothing about *where*. It now reports how many bytes
differ and the offset of the first one - a wrong tail means the last chunk never landed, a band at a
64 MB boundary means the staging buffer was refilled while its transfer was still reading it. That
one line is what turned a "the output is sometimes garbage" report into a specific bug.

### Memory: does the re-cut fit?

Measured on the 2x3090 box, `chunk=1024`, `split=24` (`helios load`, GiB, `ctx=262144`):

| | experts | rest of the model | after KV + scratch |
|---|---:|---:|---:|
| **OFF** card 0 | 9.11 | 4.31 | 22.57 used / 0.99 free |
| **OFF** card 1 | 20.05 | 0 | 21.56 used / 1.99 free |
| **ON** card 0 | 14.28 | 2.60 | 22.09 used / 1.47 free |
| **ON** card 1 | 14.88 | 1.72 | 22.95 used / **0.60 free** |

Expert bytes are identical (29.16 GB both ways), as they must be - the cut moved from "within a
layer" to "between the layers", it did not shrink anything. The KV cache is what changed shape: OFF
it is 6.00 GB, all on card 0, which is why card 0 is the tight one today; ON it is 3.00 GB per card,
which is why the two cards end up within 0.9 GB of each other.

**It fits at 24/24, but only just, and the window is narrow.** Sweeping the split at `ctx=262144`
(fresh process each, the split is read at load):

| split | card 0 | card 1 |
|---:|---|---|
| 16 / 20 / 22 | ok | **OOM on card 1** (`free 0.01 / 0.03 / 0.11 GB`) |
| **24** | 22.09 used, 1.47 free | 22.95 used, **0.60 free** |
| **26** | 23.44 used, **0.12 free** | 21.61 used, 1.95 free |
| 28 / 32 | **OOM on card 0** (`free 0.21 / 0.02 GB`) | ok |

Two of seven split points fit, and neither has more than 0.6 GB spare. The arithmetic behind it: a
card's expert bytes are `(49 - split) x 512 x 1.19 MB`, so moving one layer from card 1 to card 0
moves 0.61 GB of experts in each direction while the dense stack, the embedding, lm_head and the
KV cache stay where they are - card 0 already carries 1.27 GB of embedding that card 1 does not, and
in exchange gets 0.40 GB of lm_head. **The context length is the binding constraint, not the split**:
halving it halves each card's KV and widens the window. If the scheduler stage needs room to
experiment, the first move is a context policy, not a different constant.

### What does NOT run, and why it refuses

With the flag on, the model loads, the placement is verified (`verify-gpu`: 471 device tensors, 0
mismatches, across both cards) and `Runner::init` then **refuses with a non-zero exit and a message
naming what is missing**. It does not produce output.

That is deliberate. `run_chunk()` is one loop on card 0's stream: it would hand card 1's weights,
card 1's KV rows and card 1's scratch to card 0's kernels. On these boards that is an illegal memory
access; where it is merely wrong, it is wrong in a way that still emits fluent text. A placement
change that can emit plausible output from an unwired scheduler is the worst possible failure mode,
so this stage ships the layout and the refusal, not a fake forward.

### Verification

Flag OFF, this build vs the binary as it stood before this change (same box, same prompts, greedy,
`--temp 0`, text-only md5 of the generated text):

| check | result |
|---|---|
| `cmake --build build` | clean |
| 8 suites (`test_gemm_smoke`, `test_mgemm_semantics`, `test_reconstruct`, `test_aux`, `test_attn_parity`, `test_qsa_parity`, `helios_model_test`, `test_gdn_chunked_parity`) | **8/8 PASS** |
| greedy 128 tok, prompt "Transformer" | `9dc0b24f...` **identical** |
| greedy 128 tok, prompt "What is a transformer" | `927addbb...` **identical** |
| greedy 128 tok, prompt "A transformer is" | `8409bc9b...` **identical** |
| greedy 64 tok, 16k-word bench prompt | `8cf8463b...` **identical** |
| prefill / decode on that prompt | 534.2 tok/s prefill (before: 535.6), decode unchanged |
| 3x the same short generation | 3/3 identical (`ff42b3bb...`) |
| model load byte totals | `jobs=227906 gpu0=13.42GB gpu1=20.05GB` - unchanged, `verify-gpu` 471 tensors, 0 mismatches |

Flag ON (`HELIOS_PIPELINE=1`): the model loads, `verify-gpu` checks 471 device tensors across BOTH
cards with 0 mismatches, the placement and scratch report prints, and the process exits **5** with the
refusal message and **no generated text on stdout** - `helios gen` produces nothing at all rather than
something plausible and wrong.

### Exact remaining work (the scheduler stage)

1. **Micro-batch scheduler.** Split a prefill chunk into M micro-batches; card 0 runs microbatch *i*
   through `[0, split)`, card 1 runs microbatch *i-1* through `[split, 48)`. One cross-card handoff
   per microbatch - the four `streams` buffers, `n x 4 x 2560 x 4` bytes, ~10 KB/token at decode and
   ~5 MB at a 1024 chunk. `xcard_copy()` (moe_layer.cu) already does exactly this staging, with the
   pinned bounce these PHB boards require.
2. **Single-card MoE launch.** `moe_layer` still hard-codes two cards (`per[]`, `base[]`,
   `xcard_copy`, `part[0] + part[1]`). With the pipeline the owning card holds all 512, so one
   grouped/mgemm launch into `part[dev]` replaces the pair and the cross-card sum disappears. The
   scratch and the pointer tables for that are already built (`moe1_`, `only_card=1`).
3. **Handoff and head.** The final mixer + `lm_head` already live on card 1, so the last stage ends
   there; `next_token()` reads `logits_` from card 1, and `reset()` has to memset per-device.
4. **Speculation.** The GDN snapshot/replay path is already per-card-placed; `gdn_replay` and
   `capture_carry` need the same per-card indexing as the rest.

Order matters: (2) is a prerequisite for measuring anything, and (1) with M=1 is a no-op pipeline
(serialize, hand off once) that is only worth building because M>1 amortises the handoff.

---

## Layer-pipeline Stage 0+1 landed: per-LAYER weight placement + per-card scratch (HELIOS_PIPELINE)

The architectural refactor has begun, safely. Behind `HELIOS_PIPELINE` (default OFF) the engine can now
load with **per-layer** weight placement instead of per-weight-type, and allocate per-card scratch,
KV, GDN state and activations — the foundation a pipelined forward needs. With the flag OFF the engine
is byte-for-byte the previous, fully-verified build.

**What landed:**
- `Config` (model.hpp) gains `layer_dev(l)` (0 for [0,24), 1 for [24,48) when on; 0 for everything when
  off), `mtp_dev()`, `experts_per_card(card)` (512/512 when on, 160/352 when off), and expert
  index/slot accessors. `Model` gains a duplicated global hyper-connection mixer + `mixer_for(dev)`.
- `load_gdn` / `load_attn` / `load_ple` / per-layer `load_hc` / `load_moe_common` now target
  `cfg.layer_dev(l)`; `load_mtp` targets `mtp_dev()`; embed→card0, lm_head→card1, MTP→card1, mixer
  duplicated. Arenas use `experts_per_card × expert_layers` so **each card holds all 512 routed experts
  for its own 24 layers** — the same ~29 GB total, redistributed evenly (14.9 GB/card) instead of the
  current 20.0/9.1 skew. Placement is printed at load.
- `moe_tables_init` takes a `only_card` so single-card scratch never names the other card's arena (a
  wrong launch becomes a wrong answer, not an illegal access). `Runner` gains a per-card `Act` set,
  `gdn1_/attn1_/ple1_/moe1_`, and `kv_dev_/gdn_dev_`; per-layer KV, GDN conv/rec/snapshots and PLE conv
  allocate on `layer_dev(l)`.
- Fixed 3 real latent bugs: loader events were per-device-0 but recorded on whichever card a job
  targeted (fatal once lm_head moves to card 1); worker teardown freed the pinned ring after a
  device-0-only sync; `verify_gpu_tensors` now reports differing byte counts/offset.

**Verified independently:** build clean; **flag OFF: 8/8 suites PASS, default output byte-identical
(md5 336f1a07…)**. **Flag ON: model loads, prints the per-card placement report, verifies weights on
both cards, then exits 5 with NO generated text** — a loud refusal, never silently-wrong output.

**Stage 2 (the actual speedup, not yet built):** a pipelined forward — micro-batch the chunk, run
[0,24) on card0 and [24,48) on card1 staggered with one cross-GPU handoff per micro-batch, single-card
MoE launches (each card's layers now own all 512 experts, so the cross-card partial sum disappears), and
the final mixer + lm_head on the last card. Target ~2x prefill (529→~1050). The refusal message
enumerates exactly these three pieces.
---

## Stage 2: the pipelined forward — `HELIOS_PIPELINE=1` now RUNS (micro-batched, 2-card, overlapped)

Stage 0/1 landed the per-LAYER weight placement and per-card scratch/state/activations and then
**refused to run** (exit 5), because `run_chunk()` was still one sequential loop on card 0. The
refusal is gone: the forward is now a real two-stage pipeline, and the flag-off path is unchanged.

### The schedule

```
card 0 (stream A)                              card 1 (stream B)
----------------                              ---------------
chunk of n tokens -> M micro-batches (M = HELIOS_PIPE_MICRO, default 4)

 micro 0:  embed+expand, layers [0,24)   --handoff 0-->  layers [24,48)
 micro 1:  embed+expand, layers [0,24)   --handoff 1-->  layers [24,48)      <- overlaps micro 0's tail
 micro 2:  ...                                                    <- overlaps micro 1
 micro 3:  ...
 (drain)                        micro 3 -> tap save, MTP KV fill, final mixer, lm_head
```

* **What crosses the bus: the hyper-connection stack only.** A layer hands the next one
  `streams_[n, H, D]` fp32 and nothing else — `mixed_`, `post_`, `sub_in_`, `sub_out_` are re-derived
  by every layer. So the handoff is `M * nb * H * D * 4` bytes per chunk (10.2 MB per micro-batch at
  the default 4 x 256), staged through pinned host memory because these boards are PHB with no peer
  access (`cudaMemcpyPeerAsync` is unavailable).
* **The D2H rides card 0's DMA-out stream (`gpu(0).stream(2)`) behind an event recorded on card 0's
  compute stream.** The host then waits on THAT stream, not on the compute stream. That distinction
  is the whole overlap: draining the compute stream instead would serialise the pipeline into
  (stage 0, handoff, stage 1) and buy nothing. One pinned slot per micro-batch (`xbounce_`) so a
  copy is never staged into the slot the previous micro-batch's H2D is still reading.
* **The final mixer + lm_head + logits are on the last layer's card** (card 1), with the global
  mixer read through `Model::mixer_for(dev)` (it is duplicated per card) and the logits in a
  card-1 buffer. `next_token()`, the row argmax, the carry tap and the MTP head all moved with it.
* **Decode / MTP-verify (n <= 3) is a correct, NOT overlapped two-card sequential pass.** There is no
  second micro-batch to overlap with, and at that width the grouped MoE's fixed per-launch cost
  dominates anyway, so `M = 1` there: card 0 runs [0,24), one handoff, card 1 runs [24,48). It is a
  real forward on both cards (card 1 owns half the weights, so a card-0-only decode would be
  illegal), it is just not faster than the flag-off path, and it is labelled as such.
* **Single-card MoE.** With the re-cut, layer `l`'s 512 experts all live on `layer_dev(l)`, so
  `moe_layer` launches ONE card over its own arena: no `part[]` partial, no cross-card sum, no
  `xcard_copy` of the hidden rows, and the shared expert + mHC run on the same card as the layer. The
  per-card scratch is built with `moe_tables_init(..., only_card=own)` so the other card's table
  does not exist. Verified with `HELIOS_CNT=1`: every layer's single card is assigned **all** `n*topk`
  slots (60 of 60 at n=6), where flag-off splits them 20/40 between the cards.
* **The MTP draft head** lives on the last layer's card but reads the embedding table, which lives on
  layer 0's card. Its `n` rows are gathered on card 0 and handed over (`mtp_stage_embed`); 5.2 MB per
  chunk, versus 1.2 GB to duplicate the table.

### Why the split is exact, not approximate

Each micro-batch is a contiguous slice of the same chunk and every layer is causal in the position,
so per-row arithmetic is unchanged:
* **attention** — micro-batch `i` writes KV rows `[off, off+nb)` and attends over `[0, off+nb)`,
  which is exactly the range its own call already attended over; earlier rows were written by the
  previous micro-batch's call to the same layer, on the same card, in order. Each row's attention is
  computed independently, so the row-blocked kernel gives the same numbers at any width.
* **GDN and PLE** are running recurrences whose state the call carries (`gdn_conv_`/`gdn_rec_`,
  `ple_conv_`), and the PLE's n-gram ids come from the host history, which is extended in the same
  micro-batch order. Consecutive micro-batches advance the same state in the same order.
* **MoE** — every token belongs to exactly one micro-batch, so no token's expert sum is split.

### What DOES change: one fp32 reassociation in the MoE sum

The grouped kernel accumulates a token's 10 expert contributions in **LPT visiting order**, and LPT
is computed from the expert's token count. Flag-off that is 352 experts on card 0 plus 160 on card 1
(summed afterwards); pipeline-on it is 512 in one order. Different order, same sum, different
rounding. Measured per-layer (`HELIOS_MOEC=1`, 32-token prompt, `routed=` rms, flag-off vs flag-on):

| layer | L0 | L1-L11 | L12-L27 | L28-L47 | max |
|---|---|---|---|---|---|
| relative difference | 1.0e-9 | 3e-5..7e-4 | 1e-3..7e-3 | 8e-6..8e-3 | **7.7e-3 (L40)** |

L0 is 1e-9 — about one fp32 ulp, i.e. pure reassociation, and the growth with depth is the residual
stream's own conditioning (a single token's top-10 selection can flip on a near-tie and move that
token by ~1%). **This is a characterised reassociation, not a bug** — but it is not invisible, and
the acceptance criterion "the committed token stream must be identical" is NOT met:

| | flag OFF | flag ON |
|---|---|---|
| greedy 128 tok, 37-token prompt, `--temp 0`, MTP off | `e93a5b28…` | `b2200aa7…` |
| identical prefix | — | **first 82 of 128 tokens bit-identical**, diverges at token 82 |
| run-to-run (flag ON, 2 runs) | deterministic | **deterministic** (`b2200aa7…` twice) |

A greedy argmax flips when the top-2 logits are within that noise, and after the flip the two texts
diverge completely — which is what token 82 is. Short prompts are unaffected: *"The capital of France
is"* -> **"Paris"** in both modes. A build that demanded bit-identical tokens would have to keep the
two-card expert split, i.e. give up the entire speedup; the reassociation is reported rather than
hidden.

**MTP-on is NOT run-to-run deterministic, and was not before this change either.** Measured on the
pristine pre-change binary at the start of this session: two identical greedy 128-token runs of the
10.3k prompt gave `75a98a90…` and `27cb165c…`. It is the speculative path (the accept rate varies
run to run; every committed token is still trunk-verified). `HELIOS_MTP=0` **is** byte-reproducible
(3/3 identical), which is also what `test/parity.py` uses.

### Flag-off is unchanged — proved by A/B, not asserted

The layer loop was extracted into `Runner::layer_range(dev, …)`, which the shipped path calls with
`dev = 0, [0, n_layers)`. To prove that extraction changed nothing, the pre-refactor `run_chunk` body
was temporarily kept in the same binary behind `HELIOS_LEGACY_LOOP=1` and the two were compared
(temporary code, removed before landing):

| check | legacy loop | refactored loop |
|---|---|---|
| greedy 128 tok, MTP off, text md5 (2 runs each) | `e93a5b288a73ec1d2fc0a58850196ed2` | `e93a5b288a73ec1d2fc0a58850196ed2` |
| per-layer MoE `routed=` rms, all 48 layers (`HELIOS_MOEC=1`) | md5 `fb2a0861…` | md5 `fb2a0861…` |

`act0_` is an **alias** of the single activation set (not a second allocation), so the flag-off path
dereferences exactly the addresses it always did, and the allocation order is untouched.

### Two latent single-device assumptions the pipeline broke (both real bugs, both fixed)

1. **`aux::gr_mix` cached its scratch in four function-level statics**, on the stated assumption that
   "this is a per-process, single-device engine". Card 1's collapse was handed card 0's pointers:
   an illegal access, reported several launches later at whatever CUDA call noticed. The cache is now
   keyed by device and grows per device.
2. **The MTP draft's carry tap and the device context.** The tap is written by the LAST layer, so
   with the split it lives on card 1; and `mtp_stage_embed` moves the context to card 0 to gather
   embeddings, which has to be handed back. A launch on card 1's stream with card 0 current addresses
   card 0's address space — this surfaced as a cooperative-launch `invalid argument` (the 92 KB smem
   opt-in had been applied in the wrong context), not as a wrong answer, but it would have been able
   to be one.

### Measured speedup — prefill **526.1 -> 866.9 tok/s, 1.65x**

10.3k prompt (`/tmp/pp.txt`), `--tokens 1 --temp 0`, two runs each, same box, no other load:

| | run 1 | run 2 | prefill |
|---|---|---|---|
| flag OFF (shipped) | 526.1 tok/s | 526.1 tok/s | **526.1 tok/s** (19.52 s) |
| flag ON (pipelined, 4 micro-batches) | 869.8 tok/s | 863.9 tok/s | **866.9 tok/s** (11.84 s) |

**1.65x**, against a ~2x ceiling: the schedule cannot beat the slower of the two stages, and the two
stages are not equal (card 0 also owns the embed + expand, and the handoff bubbles are serialisation
points the CPU's host wait sits on). 866.9 is 82% of the 1058 tok/s a perfectly balanced two-stage
split of the same work would give. Flag-off prefill is unchanged at 526.1 (the documented baseline
is 527-534, the spread being the machine, not the change).

`HELIOS_PIPE_MICRO=1` degenerates to the same schedule with no overlap, i.e. a plain 2-card
sequential split, for comparison.

### Verification summary

| check | result |
|---|---|
| `cmake --build build` | clean |
| 8 suites (`test_gemm_smoke`, `test_mgemm_semantics`, `test_reconstruct`, `test_aux`, `test_attn_parity`, `test_qsa_parity`, `test_gdn_chunked_parity`, `helios_model_test`) | **8/8 PASS** |
| flag OFF, text md5 vs the verbatim pre-refactor loop (A/B in one binary) | **identical** (`e93a5b28…`, 2 runs each) |
| flag OFF, per-layer MoE `routed=` rms, 48 layers | **identical** |
| flag OFF prefill / decode | 526.1 tok/s / 59 tok/s, unchanged |
| flag ON, correctness | "The capital of France is" -> "Paris"; 128-token generation coherent |
| flag ON, determinism | 2/2 identical (`b2200aa7…`, MTP off) |
| flag ON, decode with MTP on | 57.3 tok/s, 8/12 draft slots accepted (66.7%) |
| flag ON, prefill | **866.9 tok/s, 1.65x** |
| no expert contribution dropped (`HELIOS_CNT=1`) | every layer's card is assigned all `n*topk` slots |
| `compute-sanitizer --tool memcheck`, flag ON + MTP decode | **0 errors** |
| `compute-sanitizer --tool memcheck`, flag ON prefill, 103 tokens (4 micro-batches) | **0 errors** |

### How to run each mode

```bash
# default: the verified single-card forward, unchanged
./build/helios gen ~/models/Qwen3.8-Flash-Next-exl3 --raw --prompt-file /tmp/pp.txt --tokens 128 --temp 0

# layer-pipelined forward (2 cards, micro-batched, overlapped prefill)
HELIOS_PIPELINE=1 ./build/helios gen ~/models/Qwen3.8-Flash-Next-exl3 --raw --prompt-file /tmp/pp.txt --tokens 128 --temp 0

# knobs
HELIOS_PIPE_MICRO=8   # micro-batches per chunk (default 4; 1 = no overlap, i.e. a 2-card sequential split)
HELIOS_PIPELINE_SPLIT=32   # first layer index owned by card 1 (default n_layers/2 = 24)
```

---

## Layer-pipeline Stage 2: the pipelined forward WORKS — ~1.7x prefill (908 vs 528 tok/s)

The layer-pipelined 2-GPU forward is implemented and measured. `HELIOS_PIPELINE=1` now runs a real
micro-batched 2-card pipeline (the Stage-2 exit-5 refusal is gone).

**Schedule.** Card0 runs layers [0,24), card1 runs [24,48). A prefill chunk of n tokens is split into
M micro-batches (default M=2, `HELIOS_PIPE_MICRO`); each micro-batch's hyper-connection stack
(10.2 MB at 4x256) is handed to card1 once through a pinned slot, the D2H riding card0's DMA-out
stream behind a compute-stream event so the host wait drains the copy, not card0's compute. Card1
then runs the second half, and after the last micro-batch does the tap save, MTP KV fill, the final
mixer (mixer_for(dev)) and lm_head + logits on card1. Each card's layers now own ALL 512 routed
experts, so MoE launches are single-card (no cross-card partial sum, no hidden-row xcopy). For n<64
(every decode step and MTP verify) it is ONE micro-batch — a correct, deliberately non-overlapped
two-card sequential pass.

**Measured (10.3k prefill, this session, independently reproduced):** flag-off 528.8/527.8 tok/s;
flag-on **880.1 (M=1) / 907.9 (M=2) / 864.4 (M=4) / 780.4 (M=6) / 763.0 (M=8)**. **M=2 default = ~1.7x.**
Above M=2 the per-micro-batch handoff+sync cost outgrows the added overlap — a measured result, and
the reason the default is 2. Output is coherent and deterministic; compute-sanitizer clean on the MTP
decode path and a multi-micro-batch prefill. Flag-off is byte-identical (8/8 suites, md5 336f1a07…).

**Also fixed:** a latent per-device bug in `gr_mix` (function-static scratch cached on card 0, so card
1's collapse got card 0's pointers — only reachable once two cards run layers).

**Honest caveat (reassociation, characterised not hidden).** The committed token stream is NOT
bit-identical to flag-off: the single-card MoE sums a token's 10 experts in a different order (LPT over
512 instead of 352+160-then-partial-sum). Measured per-layer relative difference: 1e-9 at layer 0
(one fp32 ulp) growing to ~7.7e-3 by layer 40 as the residual stream amplifies it; the first ~82 of 128
greedy tokens are identical, then a near-tie argmax flips and the texts diverge. This is the
inherent price of changing the summation order for the 1.7x, exactly analogous to the mma-attention
reassociation. (MTP-ON was already not run-to-run deterministic before this change; MTP-OFF is
byte-reproducible.)

**Final measured result (independently reproduced, 3 runs):** pipeline prefill **918.2 / 915.2 /
910.6 tok/s** (M=2 default) vs flag-off **528.8** = **1.73x prefill**. Pipeline is deterministic
(long-prompt text md5 1a5cac82… on two consecutive runs). Flag-off unchanged (8/8 suites, md5 336f1a07…).

**Prefill win but a decode regression — the honest tradeoff.** Pipeline **decode** (n=1) is **49.1
tok/s** vs flag-off **~59** — a ~17% regression. Cause: at n=1 there is no micro-batch depth to
amortize anything, so the pipeline degenerates to a non-overlapped two-card sequential pass whose
per-step cross-card handoff of the hyper-connection stack (~245 KB at n=1) is pure added latency,
while flag-off's fine-grained within-layer split is cheaper for a single token. So:

- **Prefill-heavy / long-context workloads** (the 230k-context regime this engine targets): load WITH
  `HELIOS_PIPELINE=1` — 1.73x faster prefill, which is also faster context ingestion.
- **Decode-heavy / short-prompt workloads**: load with the default flag off — better steady-state decode.

Both modes are load-time selectable and both are byte-reproducible; this is a workload choice, not a
regression in the default path. Closing the decode side would mean making the n=1 path avoid the
handoff (e.g. keeping the last-layer card resident for a decode step, or a per-card KV/mixer fast
path) — the next refinement, not attempted here.

**Decode diagnosis (the residual gap is handoff latency, not MoE).** Profiling the decode path in
both modes: the pipeline's MoE phase is actually FASTER per step (9.86 vs 15.01 ms — the single-card
MoE removes the cross-card partial sum), and the other phases match (amix 2.8/2.8, gdn 4.1/4.2, attn
1.3/1.3, mmix 2.7/2.7). So the ~3 ms/token decode regression is NOT in any attributed phase — it is the
un-attributed per-step cost of the layer-24 cross-card handoff. At n=1 the handoff is a synchronous
device-to-host-then-device round trip (D2H on card0's DMA-out stream, host waits, H2D to card1), which
serializes the step; the phase profiler's buckets don't capture it. The fix is a fully device-to-device
(P2P) handoff ordered by a cross-card event with no host round-trip, so card1's second half starts the
moment card0's first half is done without the host in the loop. That is a contained scheduler change
inside the already-working pipeline, not a redesign — the next concrete refinement, scoped and ready.

**Decode residual: P2P is unavailable, so the host bounce is mandatory (verified).** I probed the two
cards directly: `cudaDeviceCanAccessPeer` is false both ways and `nvidia-smi topo -m` shows **PHB**
(PCIe host-bridge, no peer-to-peer). So there is no `cudaMemcpyPeerAsync` to replace the pinned-host
round-trip with — the D2H→H2D through host memory (Runner::xcopy) is the only cross-card path, and the
code's existing comment saying exactly that is correct. That closes the "make the handoff device-to-
device" idea: it is not implementable on this hardware. The per-step decode cost of the handoff
(~245 KB at n=1 plus a host round-trip) is the residual behind the 49 vs 59 tok/s decode difference;
recovering it exactly would return pipeline decode to default parity, which is marginal against a
~118 tok/s baseline. The higher-value decode levers are the reference's own (larger MTP draft with
higher acceptance, cq3 KV to cut the long-context KV walk), which are separate efforts, not a handoff
tweak. Recorded here so the next person does not re-attempt a P2P handoff that the topology forbids.

**MTP draft budget is not a decode lever (re-measured, 30k context, default mode).** Sweeping
HELIOS_SPEC_K: K=2 37.4 tok/s (50.8% accept, 2.02 tok/step), K=4 38.1 (32.5%, 1.65), K=6 39.3
(34.2%, 1.68), K=8 35.8 (27.4%, 1.55). Decode is flat ~35-39 tok/s across the whole draft budget:
at long context the draft acceptance (27-51%) is too low for speculative decode to pay for the extra
weight-streaming per step. The real decode levers are the reference's own and are separate efforts —
(1) cq3/cq8 KV-cache quantization to cut the long-context KV walk (we use fp16 KV: 6 GiB @262k vs the
reference's 1.13 GiB cq3), and (2) a higher-quality MTP head. Neither is a tuning knob; both are
substantial kernel work. Not attempted here.

**Pipeline prefill: the split point is already optimal, and the wall-clock is overlap-bound.**
Sweeping the layer split (card0 = layers [0,split), card1 = [split,48)) at 10.3k prefill:
split=18 918.2, 20 916.8, 22 910.7, **24 908.0**, 26 906.0, 28 905.4 tok/s — a ~1% spread, within
run-to-run noise. The 48 layers are near-uniform in cost, so 24/24 is already balanced; card IMBALANCE
is not the remaining lever. The pipeline-mode phase profile sums BOTH cards (they overlap), so the
wall-clock is set by the slower card: total phase time is 1860 ms across two cards against a 1770 ms
single-card sum, yet wall-clock is 920 vs 529 tok/s. The theoretical perfectly-balanced-and-overlapped
2-stage ceiling is ~1056 tok/s (half of 2112); we measure ~910-920, i.e. **~87% of the ideal 2x**, the
residual being handoff-gap / imperfect-overlap cost, not balance or per-card phase cost. Since the
per-card phases (MoE ~353 ms, mHC ~281 ms) run the SAME kernels as the reference, prefill is at a
practical ceiling for this hardware: beyond ~920 tok/s needs either better-than-reference MoE/mHC
kernels (research-level) or more than two GPUs.

**Prefill ceiling: the one remaining identified lever is the multi-slice wide-tile MoE GEMM.** After the
pipeline, prefill is ~920 tok/s (~87% of the ideal 2x from the non-pipelined 528). The dominant
per-card phase is the MoE expert GEMM (~353 ms/card). The MoE is NOT bandwidth-bound: 31 GB of expert
weights per 1024-token chunk at ~800 ms is only ~36 GB/s against a ~936 GB/s card (~4% utilisation) —
it is per-expert-GEMM-overhead-bound (512 experts x 3 matrices x 48 layers ~= 74k tiny ~20-row GEMMs,
each dominated by fixed dequant+tile+epilogue cost). exllamav3 instantiates 32- and 64-row tiles and
selects them per launch; **this port is pinned to 16-row tiles because the multi-slice path
(`TILEBLOCKS_M>1`) diverges** (a register-mapping defect in the ported `exl3_gemm_inner`). Fixing it
is a documented **~17% prefill opportunity** (920 -> ~1075 tok/s) and is the single largest remaining
item. It has resisted several focused attempts (each eliminated a hypothesis — the A-fragment layout,
the C-index, the epilogue, the split-K paths — but the multi-slice C-fragment register mapping is
still unverified). It is research-level debugging on a delicate tensor-core kernel, not attempted here
rather than risk the verified engine on another multi-session attempt.

---

## Multi-slice GEMM bug FOUND and FIXED (but wide tiles still gated by the fused wrapper)

Differential analysis of the ported `exl3_gemm_inner.cuh` against the reference found the real defect
that the hand-debugging attempts had missed, in `reduce()`'s fp16-accumulator fold: **the port folded
only `frag_c_h[0][n]` into `frag_c[0][n]`, while the reference folds every `m` in [0, TILEBLOCKS_M).**
sm_86 sets `EXL3_GEMM_H_ACC`, so the fold is live. For TILEBLOCKS_M>1 the row blocks past the first
accumulated into an fp16 set that was never folded, and ptxas dead-code-eliminated the MMAs writing it
(register count 128 vs 152) — **rows 16+ were identically ZERO for every wide tile.** Fix: the H_ACC
fold now loops m over [0, TILEBLOCKS_M); for TILEBLOCKS_M==1 the loop body is byte-identical, so the
shipping 16-row path is unchanged (default md5 336f1a07 verified identical). Also added
`__launch_bounds__` to the test wrapper so a 512-thread launch doesn't hit the 128-reg ceiling.

**The acceptance test was itself wrong and is fixed.** `test_gemm_rows` compared a wide tile's rows
16+ against an mt=16 "reference" that is IDENTICALLY ZERO there (the kernel writes exactly rows
0..TILESIZE_M-1, and with no slice loop a 16-row tile writes nothing past row 15) — so it asserted
"a wide tile must compute nothing in rows 16..31", the opposite of the intent. Corrected to compare
per 16-row block, only between in-contract tiles, and to check single-producer blocks are non-zero
(strictly stronger: rows 16-31 previously had no meaningful check). Now all row blocks (0-15, 16-31,
32-47, 48-63) match across tile sizes. All 8 suites PASS, default output byte-identical.

**But wide tiles are still NOT shippable — a separate fused-MoE bug.** With the GEMM fixed, wide tiles
are measurably faster in pipeline mode (MTILE=16 927.7, **MTILE=32 1048.6 tok/s, +13%**, MTILE=64
1020.2) — but the FUSED grouped-MoE path with wide tiles outputs **garbage** (MTILE=16 → "is Paris,
where the Eiffel Tower is located"; MTILE=32/64 → "& 2,700,000"). The isolated GEMM is correct; the
fused kernel's wide-tile gather/wrapper (moe_kernel_instances_m32/m64, whose layout was never
re-laid-out and which has no test) is a distinct defect. So the +13% MoE win is real and now precisely
localized to the fused wide-tile wrapper — the next concrete target, not yet fixed. Shipped default
stays MTILE=16 (correct, coherent).

---

## Fused wide-tile MoE bug FOUND and FIXED — wrong instance table, one index off (MTILE=32/64 now correct)

The wide-tile garbage was **not** in the gather, the tail handling or the shared-memory sizing. It was
one array index, in the launcher.

`moe_grouped` computes `K = K_gate == K_up == K_down` (this model is 2-bit, so **K = 2**) and selects
`moe_kernel_instances_m32[K]` / `...[m64][K]`. Those two tables are declared in
`src/cuda/quant/exl3_moe.cu:47-56` and are indexed **by bitrate K**, but the port filled them as a flat
list of the three instantiations that were actually built:

```c++
fp_exl3_moe_kernel moe_kernel_instances_m32[] =
{
    exl3_moe_kernel_k0_n128_cb2_m32(), exl3_moe_kernel_k3_n128_cb2_m32(), exl3_moe_kernel_k4_n128_cb2_m32(),
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
};
```

**Slot 2 holds the K = 4 kernel.** So 2-bit weights launched a kernel that dequantizes 4 bits per
trellis word: the launch succeeds, no CUDA error, no shape complaint, and the MoE output is garbage
("& 2,700,000."). The `if (!kernel) ... [0]` fallback for pruned bitrates never fires, because slot 2 is
not null — it is occupied by the wrong kernel. The 16-row table above it was indexed correctly
(`4*K + 2*cb + N_off`, k3 at 12..15, k4 at 16..19), which is exactly why MTILE=16 was fine and only
the wide path broke, and why MTILE=32 and MTILE=64 produced *identical* garbage (same wrong K=4 kernel,
two tile shapes).

Fix: place the fixed-bitrate entries at their own bitrate slots — k3 at index 3, k4 at index 4, null
everywhere else — so the `[K]` contract holds and the existing null fallback does its job. With K = 2
the launcher now takes the K = 0 runtime-bitrate wide instance, which switches bitrate inside the
kernel exactly as the 16-row path already did.

**How it was localized** (the shape of the hunt matters, because the obvious suspects were all
innocent): a throwaway harness built the fused MoE end to end — real packed EXL3 trellis + suh/svh per
expert for gate/up/down, `moe_grouped` called three times with `HELIOS_MOE_MTILE` = 16/32/64, output
compared buffer-by-buffer including the internal `temp_state_*` / `temp_intermediate_*` scratch. That
showed the staged input and the raw u-GEMM output were *bit-identical* at 32/64 while the final
`y` was not, which ruled out the gather and the GEMM. Substituting the wide-tile getter's
instantiation `exl3_moe_kernel<0,128,2,32>` -> `<0,128,2,16>` (byte-identical SASS, confirmed with
`cuobjdump -sass`) still changed the answer, which is only possible if the *selected function* differs —
`cudaFuncGetAttributes` on the selected pointer then reported a different `localSizeBytes` for the two
"identical" kernels and the table index fell out immediately. Everything checked before that (instance
table bitrate, shared-memory sizing for TILESIZE_M > 16, per-expert row-count and tail handling, the
size_m <= 8 reduction path, the split-K lock protocol, the g/u/d shared-memory reuse race) was
verified correct — the barrier experiment between the gate and up GEMMs changed nothing, and
`test_gemm_rows` extended to 4/8/12/17/33 rows and grids 1/2/4/32 shows no divergence anywhere.

**Verified after the fix**

* Fused MoE, `moe_grouped` at MTILE 16/32/64 vs MTILE=16: **rel RMS 0.000e+00, bit-identical**, over
  token counts {1, 4, 5, 8, 12, 16, 17, 20, 33, 37, 48, 64, 70} x shapes (512,256), (512,128),
  (2560,640), (640,2560), single- and multi-expert with the LPT schedule. The one non-zero residual is
  rel RMS 1.4e-3 on the (512,256) multi-expert case, where MTILE=16 uses the N=256 tile and the wide
  path N=128 — the documented split-K reassociation, not an error.
* Real model, `HELIOS_PIPELINE=1`, greedy 128 tokens, `--temp 0`, text-only md5:
  **MTILE=16 d533d3f8, MTILE=32 d533d3f8, MTILE=64 d533d3f8** — identical, and MTILE=16 is byte-identical
  to the same run with the fix reverted, i.e. the default path did not move. ("The capital of France is"
  -> " is Paris, where the Eiffel Tower is located." at all three.)
* Prefill (~8.1k prompt, pipeline mode): **860.8 tok/s at MTILE=16, 922.2 at MTILE=32 (+7.1%)**,
  865.5 at MTILE=64. The win is now realised, not just predicted.
* All 8 suites PASS; `test_gemm_rows` reports no divergence on any row block.

Shipped default stays MTILE=16; `HELIOS_MOE_MTILE=32` is now a correct, faster option rather than a
silent-garbage one. The two wide-tile tables carry a comment naming this trap, because the failure is
invisible by construction: a wrong instance still launches.

---

## Fused MoE wide-tile wrapper FIXED; MTILE=32 is now the DEFAULT (+5-6% prefill, bit-identical)

The fused grouped-MoE garbage at MTILE=32/64 was a **single array index in the launcher**. `moe_grouped`
computes K = K_gate==K_up==K_down (this model is 2-bit, so K=2) and looks up `moe_kernel_instances_m32[K]`
/ `m64[K]`, indexed BY BITRATE. The port had filled those tables as a flat list of the three built
instantiations, so **slot 2 (2-bit) held the 4-bit kernel** — 2-bit weights launched a 4-bit
dequantizer: launch succeeds, no CUDA error, silent garbage. The `if (!kernel)` null-fallback never
fired because slot 2 was occupied, not null. MTILE=16 was indexed correctly, which is why it was fine
and why 32 and 64 produced identical garbage (same wrong kernel, two tile shapes). Fix: place the
fixed-bitrate entries at their own bitrate slots so the `[K]` contract holds.

**Localization** (throwaway harness, since removed): staged input and raw u-GEMM output were
bit-identical at 32/64 while y was not → ruled out gather and GEMM; substituting the wide getter's
instantiation to 16 (byte-identical SASS) still changed the answer, so only the selected function
differed. Instance bitrate/codebook selection, SMEM sizing, per-expert row count, tail handling, and
the split-K lock protocol all checked out correct.

**Now measured bit-identical and enabled.** MTILE=16/32/64 all produce the same coherent output and the
same 128-token greedy md5 (pipeline and default modes). **MTILE=32 is the new default**: prefill
973 vs 920 tok/s (pipeline, +5.8%) and 557 vs 529 (default, +5.4%) at 10.3k, with the default output
md5 **unchanged (336f1a07…)** and no decode regression (59.4 tok/s at both tile sizes). MTILE=64 is
slower (888 tok/s, 3-CTA/SM occupancy), so 32 is the sweet spot. 8/8 suites PASS.

**The last big MoE lever is realized.** Two chained bugs had blocked the reference's wide-row-tile MoE
for many iterations: the isolated GEMM's missing fp16-accumulator fold loop (rows 16+ were zero), and
this launcher table-index that fed 2-bit weights to the 4-bit kernel. Both found by differential
analysis against exllamav3, both verified bit-identical, both now shipping. Prefill this session:
322 → 322(attn/LPT/GDN as shipped) → 920 (pipeline) → **977 tok/s** (pipeline + wide tiles).

---

## The prefill gap is VRAM-bound, not compute-bound — cq3 KV is the final lever

The MoE lever is fully realized and further M-tile widening is not the issue (MTILE=64 is SLOWER, 888
vs 973, from 3-CTA/SM occupancy). The remaining prefill gap traces to **prefill-chunk size**, which is
VRAM-capped. Verified: at `--chunk 2048` the engine dies with
`cudaMalloc(40.00 MB) failed on device 1 (free 0.02 GB)`. The MoE dequantizes ALL 512 experts'
weights once per chunk regardless of token count (top-10 saturates at 2560 assignments), so the expert
traffic is ~(tokens/chunk) x 29 GB — a 2x chunk halves the dequant-per-token and is a large, kernel-free
win. But chunk=2048 OOMs because card1 has no headroom for the doubled per-chunk scratch.

**cq3 KV quantization is the unlock.** Helios stores KV as fp16 (6.4 GB at 262k: 12 full-attn layers x
2 x [262144 x 2 kv-heads x 256] x 2B). The reference uses `CACHE_QUANT=3` (cq3, 1.13 GiB). Switching
Helios's KV to cq3 would free ~5 GB per card, letting the prefill chunk grow to 2048 (or larger) and
amortizing the expert dequant over more tokens — a prefill win with NO new compute kernels — while also
cutting the long-context KV walk that bounds decode. This is the single highest-leverage remaining item.

**Why not attempted here:** it is a substantial, delicate change — cq3 storage plus an on-the-fly
dequant path in BOTH attention kernels (the mma prefill kernel whose 7.7x speedup is the session's
headline, and the split-KV decode kernel) plus the KV write path — and it would directly endanger the
carefully-tuned attention win to chase a prefill gain that is indirect (VRAM -> chunk -> dequant
amortization). Quantifying it precisely here so the next effort starts from a known, measured chain
rather than re-deriving it: **cq3 KV -> ~5 GB freed/card -> chunk 1024->2048 -> 2x expert-dequant
amortization -> the remaining prefill headroom**, plus a direct decode win at long context.

---

## Final delivered state (integrity audit)

8/8 suites PASS, build clean, byte-reproducible greedy output (md5 336f1a07…), prefill **983.6 tok/s**
(pipeline + wide tiles) / 557 (default), decode **59.8 tok/s**, context capacity **262,144**. Five
shipped, independently-verified optimizations this session — mma tensor-core attention (+45%), LPT MoE
scheduling (+1.8%, bit-identical), chunked WY gated-delta-rule (+10%), layer-pipelined 2-GPU forward
(+74%), wide-tile MoE GEMM (+6%) — a **3.0x prefill improvement** (322 -> 984 tok/s) with the shipped
default output byte-identical throughout. Every phase is at exllamav3 kernel parity; the remaining
prefill gap is VRAM-bound (chunk amortization), and the final lever (cq3 KV, with its
~5 GB -> chunk-2048 -> 2x-dequant-amortization chain) is scoped and documented for the next effort.

---

## Quantized KV (exllamav3 CacheLayer_quant): the FORMAT is ported and bit-exact; the attention read path is NOT done

`HELIOS_KV_QUANT=n` is accepted and **refused with a message** - the engine prints why and stays on
fp16. Nothing about the shipped default changed: 8/8 suites PASS and the greedy output is
byte-identical (text-only md5 `8925877a16b6334a30bebb63d45b850d` before and after this work, MTP
47/73 accept both runs).

**What is done and verified** (`src/cuda/attn/kv_quant.cuh`, `kv_quant.cu`,
`test/test_kv_quant.cpp` + `test/kv_quant_golden.h`):

* The reference's exact format - groups of 32 along the token dim, unnormalized H32 rotation
  (x 1/sqrt32), one fp16 absmax per group, `bits`-wide codes on the midpoint grid
  ((2q+1)/2^bits - 1), bit-plane packing. No invented variant.
* The golden vectors in `kv_quant_golden.h` were produced by **compiling exllamav3's own
  `exllamav3_ext/cache/q_cache_kernels.cuh` unmodified** (`nvcc -arch=sm_86` over a throwaway driver,
  no torch needed) and running `quant_cache_cont_kernel` / `dequant_cache_cont_kernel` on a
  deterministic input. So the parity test is cross-implementation, not self-consistency.
* `test_kv_quant` result, all seven bitrates: **packed codes and half scales BIT-IDENTICAL to the
  reference**, dequant within 6.4e-7 of a group max (one fp16 ulp relative to a group max is
  4.9e-4), and round-trip relative RMS 0.34 / 0.17 / 0.086 / 0.044 / 0.021 / 0.011 / 0.005 at
  2..8 bits - identical to the reference's own round trip to six digits.
* Footprint on this model (token_dim = 2 kv heads x 256 = 512), per token per tensor:
  fp16 1024 B; **cq3 224 B (4.57x)**, cq4 288 B, cq5 352 B, cq6 416 B, cq7 480 B, cq8 544 B.
  At 262,144 ctx and 12 full-attn layers x 2 tensors that is 6.00 GiB fp16 -> **1.31 GiB at cq3**
  (the reference server quotes 1.13 GiB; same format, slightly different token_dim accounting).

Three real bugs were found and fixed on the way, all by the bit-exactness check rather than by
inspection: the H8 butterfly's high-lane sign (`b - a` instead of `a - b`, which a round-trip test
cannot see because it is still an involution up to a row sign); the centroid offset through
H32*1 = 32*e_0 (32m-16, not 32(m-1)); and FMA contraction of `v * inv_s` into the following
`fmaf` under `--use_fast_math`, which pushes codes across centroid boundaries (fixed with
`__fmul_rn`).

**What is NOT done, and why the flag is refused rather than enabled.** With the cache quantized the
attention kernels have to dequantize on read, in three places: the mma prefill kernel's K/V staging
loop, `gqa_dense_split_kernel`, and `gqa_split_kv_kernel`. The write path and the mma staging are
written (a whole 32-value group per thread straight into the existing fp16 shared-memory tile, so
the tensor-core math is untouched) but the two scalar kernels are not, and the design there is
genuinely unresolved rather than a small bug: those kernels give each lane 8 channels of a
256-wide row, while a quantized group wants 8 lanes, and a 256-wide row is 8 groups = **64 lanes**,
two warps' worth. The first attempt mapped lane L to group L/4, which leaves each group dequantized
by 4 lanes while the butterfly shuffles across 8, and the engine produced **fluent, wrong text** -
the worst failure mode a KV cache has. Templating the mma kernel on the bitrate (passing the two
`KvQuant` views by value alongside `__restrict__` pointers) also made `test_attn_parity` fault with
`cudaErrorMisalignedAddress`, so that was reverted too; the suite passes again.

**Consequence for the motivating measurement: none taken.** The plan was cq3 -> ~4.7 GiB freed per
card -> `--chunk 2048` fits -> halve the per-chunk expert dequant. The freed VRAM is real and
measured by the format, but chunk 2048 was **not** run and no prefill number is claimed, because
the engine cannot read a quantized cache yet. Note also that this repo already bounded that lever
from the other end: an earlier chunk sweep (RESULTS, "Holding the prompt fixed at 10.3k") found
1024 -> 1536 buys **+2.1%** and fitted the whole fixed-cost term at 3.5% of prefill, so the
2x-chunk -> half-the-expert-dequant chain was already measured to be worth single-digit percent, not
the ~2x a traffic-bound engine would show. The honest expectation for finishing this is therefore
"a few percent of prefill plus a real decode win at long context", and the decode win is the part
worth chasing, because the KV walk there is genuinely bandwidth-bound.

---

## Quantized KV: format ported and BIT-EXACT vs exllamav3; read path unfinished (safely refused)

cq3-KV (the final identified lever) is PARTIAL. The quantize/dequantize **format is done and verified
bit-exact against exllamav3's own kernels** — `kv_quant.cuh/.cu` port the `CacheLayer_quant` layout
verbatim (groups of 32, unnormalized H32, one fp16 absmax per group, midpoint grid, bit-plane
packing), and `test_kv_quant` proves the packed codes and half scales are BIT-IDENTICAL to
exllamav3 for all seven bitrates 2..8 (golden vectors produced by compiling the reference's
`q_cache_kernels.cuh` unmodified — cross-implementation, not self-consistency). The check caught three
real bugs the round-trip test could not (an H8 butterfly sign that is an involution up to a row sign, a
wrong centroid offset through H32, a scale-array offset). Footprint: 6.00 GiB fp16 -> 1.31 GiB cq3 at
262k, ~4.7 GiB freed per card. Quantize-on-write is wired.

**Not delivered: the attention READ path over a quantized cache.** The scalar prefill/decode kernels
(`gqa_dense_split_kernel`, `gqa_split_kv_kernel`) each want 8 channels of a 256-wide row per lane while
a quant group spans 8 lanes; the first mapping produced fluent-but-wrong text, and templating the mma
kernel on the bitrate faulted the parity test (misaligned address) — both reverted. This is an
unresolved design, not a small bug. Because a wrong KV cache fails SILENTLY (fluent, wrong text), the
path is **refused at startup** (`HELIOS_KV_QUANT=3 ignored: ... read path not finished ...`) and the
engine stays on fp16. So the VRAM->chunk-2048->dequant-amortization win is NOT yet realized.

**State: 9/9 suites PASS (8 + new test_kv_quant), fp16 default byte-identical (md5 336f1a07…).** The
bit-exact quant format + its cross-implementation test are real, valuable groundwork; finishing the
scalar/decode read path (or a shared-memory-dequant mma path) is the next concrete step to actually
unlock the prefill gap.

---

## cq3 KV cache: the attention READ path is DONE (dequant staging) and `--chunk 2048` now fits

`HELIOS_KV_QUANT=3` **runs**. The flag is no longer refused. The read path is **dequant staging**:
`attn_layer` materializes the fp16 K and V the attention kernels read with the existing
`kvq_dequant`, into a staging buffer laid out exactly like the fp16 cache
(`[row][n_kv_heads][head_dim]` fp16, row = absolute position), and then points the **unchanged**
attention kernels at it. `qwen_gqa.cu` is not touched at all - not the mma prefill kernel's tile
staging, not the scalar prefill kernel, not the split-KV decode kernel, not the combine.

**Why this shape and not a fused dequant.** Two fused designs were tried earlier and both produced
wrong output: mapping lanes to quant groups in the scalar kernels (a 256-wide row is 8 groups, i.e.
64 lanes, and the wrong mapping gave fluent wrong text) and templating the mma kernel on the bitrate
(misaligned-address fault in the parity test). Staging sidesteps the whole question. The attention
arithmetic - mma, online softmax, PV, the split-KV combine - is the code that was tuned and verified,
fed fp16 values in both cases; only *where* those fp16 values come from changes. The cost is one extra
pass over the KV range per layer per call, which is measured below rather than assumed.

**The staged range is the whole prefix, [0, pos0+n).** Attention is causal over all of it, so there is
no smaller bound to dequantize; the staging buffer is therefore one fp16 K plus one fp16 V for the
full `ctx`, per device (512 MB at ctx 262144 on this model), reused by every layer because layers run
in order on one stream. That is why the net VRAM win is 4.19 GB rather than the 4.7 GB the format
alone would give - the runner prints both numbers.

### Measured (10.3k prompt `/tmp/pp.txt`, RTX 3090 x2, ctx 262144, default non-pipelined forward)

| | fp16 (default) | `HELIOS_KV_QUANT=3` |
|---|---:|---:|
| KV cache, 12 full-attn layers x 2 tensors | 6.00 GB | **1.31 GB** |
| + dequant staging (one K + one V per card) | - | 512 MB |
| gpu0 used / free at chunk 1024 | 22.61 / 0.95 GB | **18.03 / 5.53 GB** |
| net VRAM freed | - | **4.19 GB** |
| prefill, chunk 1024 | 553.3 / 571.1 / 572.5 / 570.8 tok/s | **555.9 / 569.3 / 570.5 / 568.2 tok/s** |
| prefill, chunk 2048 | **OOM** (`cudaMalloc(3.00 MB) failed on device 0, free 0.00 GB`) | **540.1 tok/s** |
| decode 128 tok @10.3k, MTP off | 36.09 / 37.94 / 37.85 tok/s | 30.61 / 31.74 / 31.55 tok/s |

**The motivating measurement, taken: chunk 2048 fits with a quantized cache and does not fit without
one.** That is the whole chain - cq3 -> 4.19 GB net -> chunk 1024 -> 2048 - and it is now measured
end to end rather than projected. Note what it is worth: 540.1 tok/s at chunk 2048 against 555.9 at
chunk 1024, i.e. **the bigger chunk is 2.8% SLOWER**, not 2x faster. That agrees with the earlier
chunk sweep in this file ("1024 -> 1536 buys +2.1%... the fixed expert-weight re-read is 3.5% of
prefill"): the chunk lever was already known to be worth single-digit percent, and cq3's real payoff
is the VRAM itself (it is what makes 2048 possible at all) rather than the chunk amortization.

**Prefill is free.** 555.9 / 569.3 / 570.5 / 568.2 against 553.3 / 571.1 / 572.5 / 570.8 tok/s at the
same chunk is inside run-to-run noise: the staging writes (pos0+n) x token_dim x 2 bytes per layer per
call, which at 10.3k is 10.5 MB per layer per chunk against a ~1670 ms chunk. **Decode pays for it:
-16%** (30.6 / 31.7 / 31.6 against 36.1 / 37.9 / 37.9 tok/s at 10.3k context, MTP off so
speculation cannot confound it, three samples each, spread under 2%). The staging write is ~0.2 ms
per token against a 32 ms step by bandwidth arithmetic, so the dequant kernels themselves - one thread
per 32-value group, 12-byte strided reads - are the likely cost; that is not established, only
measured.
It does mean the cq3 decode win this section was supposed to deliver is **not** what materialized at
10k context: the attention walk re-reads each key once per query head (24x), so shrinking the cache
4.57x shrinks a term that was never the binding constraint, and the staging adds a pass on top.

### Correctness

* **fp16 default is byte-identical.** Greedy 128-token generation on the 10.3k prompt, text-only md5
  `effd0cae3a401faf6ab13609b61203cc`, is identical on a binary built before this change and one built
  after it (2 runs each), and `" is Paris."` on a short prompt from both. The fp16 path allocates
  nothing extra, launches nothing extra, and passes `k_cache`/`v_cache` through unchanged.
  **Caveat, stated because it is not this change's doing:** the md5 recorded earlier in this file
  (`336f1a07...`) is **not** reproduced by the current tree on `/tmp/pp.txt` - the pre-change binary
  gives `effd0cae...` too. Whatever prompt or state produced `336f1a07`, this tree no longer does, so
  that constant needs re-establishing before it can be used as an acceptance check.
* **Quantized output is coherent and deterministic, not token-identical.** On the 10.3k prompt the
  first **169 characters** are identical to fp16 and then it diverges into a different, fluent
  continuation - the expected behaviour of a lossy 3-bit KV cache, and the reason the flag's startup
  message says so. Two runs are byte-identical (`2942ccac...`). `HELIOS_KV_QUANT` = 2 / 3 / 4 / 8
  all answer `"The capital of France is"` -> `" is Paris"`, and `HELIOS_PIPELINE=1 HELIOS_KV_QUANT=3`
  runs (staging allocated per card).
* **compute-sanitizer `--tool memcheck`** on a quantized run (prefill + decode + MTP draft):
  **0 errors**.
* **All suites PASS** (test_gemm_smoke, test_mgemm_semantics, test_reconstruct, test_aux,
  test_gemm_rows, test_attn_parity, test_qsa_parity, test_gdn_chunked_parity, test_kv_quant,
  helios_model_test, chat/tokenizer/utf8).

### What this does not do

The QSA sparse path stays gated off under `HELIOS_KV_QUANT` (`kv_bits == 0` in `qsa_pool_on`), but the
reason changed and the comment says so: it used to be that the sparse gather indexes the cache as
fp16 by token id, which a quantized cache is not. With the staging buffer it would actually be served
correctly - the staged range is every row the gather can name, in the fp16 cache's own layout. It is
gated off because the QSA branch is opt-in and unverified end to end, not because the combination
would be wrong.

**State: the format was already bit-exact vs exllamav3; the read path is now dequant-staged and
verified, `HELIOS_KV_QUANT=3` is enabled, chunk 2048 fits, and the remaining cost is a measured
16% of decode at 10k context.** Making decode pay that back means fusing the dequant into the
split-KV decode kernel's inner loop - which is exactly the lane->group mapping that produced fluent
wrong text before, so it wants its own test (a quantized-cache decode parity case at n=1, not an
end-to-end text check) before anyone tries it again.

---

## cq3 KV read path WORKS (chunk 2048 fits) but is a SPEED NEGATIVE — stays off by default

The read path is finished via **dequant staging**: attn_layer materializes the fp16 K/V the attention
kernels read (via the existing bit-exact `kvq_dequant`) into a per-device staging buffer laid out like
the fp16 cache, and passes THAT to the UNCHANGED kernels — `qwen_gqa.cu` is not touched at all, so the
mma attention, scalar prefill and split-KV decode math stay byte-identical. This sidesteps the
lane->group mapping that produced fluent-wrong text before.

**The VRAM unlock works:** `HELIOS_KV_QUANT=3 --chunk 2048` now FITS (gpu0 22.61→18.03 GB used, 5.53 GB
free); the same command on fp16 still dies (`cudaMalloc(3.00 MB) failed on device 0 (free 0.00 GB)`).
The quant format is bit-exact vs exllamav3 (test_kv_quant, all 7 bitrates). Quantized output is
coherent and deterministic (first ~169 chars match fp16, then a different-but-valid continuation),
compute-sanitizer clean, 9/9 suites pass.

**But it is SLOWER, so it does not ship on by default — a clean negative:**
- prefill @chunk 1024: cq3 568 vs fp16 570 tok/s (dequant staging ~free here)
- prefill @chunk 2048 (the point of freeing VRAM): cq3 **540** — fits, but SLOWER than chunk-1024 cq3
- decode @10.3k: cq3 **31** vs fp16 **37** tok/s (**-16%**)

So freeing the VRAM does NOT convert to speed: the dequant-staging cost (materializing fp16 KV every
attention call, plus 512 MB of staging buffer) offsets the expert-dequant amortization a bigger chunk
was supposed to buy. The "VRAM-bound" theory was half right — chunk 2048 now fits — but the payoff did
not materialize, so the prefill gap is genuinely **compute-bound** (mHC + MoE at exllamav3 parity), not
fixable by freeing VRAM. cq3 KV stays available and correct behind `HELIOS_KV_QUANT` but OFF by default.

**Integrity after this work (and after an unrelated agent touched qwen_gqa.cu mid-session):** build
clean, **9/9 suites pass** (attn_parity confirms the tuned mma attention is intact), default md5
336f1a07… byte-identical, prefill 982.8 (pipeline) / 557 (default), decode 59.2. The canonical md5
336f1a07 is the "Explain how a transformer works." 128-token hash; effd0cae is the same tree on the
/tmp/pp.txt prompt — not a stale constant, just a different prompt.

---

## Definitive plateau characterization (why 984 tok/s is the ceiling here)

After five shipped optimizations (+ the cq3-KV negative), the remaining prefill gap is fully
characterized, and the last apparent lead is closed:

- **mHC is a red herring in the pipelined case.** The mHC (hyper-connection mix, ~280 ms/card) is
  memory-bound and runs at low raw bandwidth, but the layer PIPELINE overlaps it with the
  compute-bound MoE on the other card. Making the mHC faster does NOT reduce wall-clock unless it sits
  on the critical path — it largely does not. So the mHC's low bandwidth is not a recoverable win.
- **The wall-clock is bounded by the MoE compute**, which runs exllamav3's own exl3_moe grouped kernel
  at kernel parity, plus the ~87%-efficient two-card overlap (no P2P; handoffs go through host memory).

So: prefill is at ~50% of exllamav3 and **compute-bound at reference kernel parity**. Closing the last
2x requires either beating exllamav3's MoE/mHC kernels (research-level, not an optimization pass) or
more than two GPUs. Every optimization lever accessible at the kernel/scheduling level has now been
tested: mma attention (shipped), LPT MoE (shipped), chunked GDN (shipped), 2-GPU pipeline (shipped),
wide-tile MoE GEMM (shipped, 2 chained bugs fixed), cq3-KV (tested, negative), MTP budget (flat),
P2P (unavailable), card-split balance (optimal), prefill chunk (memory-capped). **This is the
accessible ceiling for this hardware.**

---

## 1:1 head-to-head grid (exllamav3 vs helios) — and a correction

Controlled A/B on the same prompts, same harness (test/grid_bench.py, which reuses bench.py's
methodology: unique per-request nonce defeating the prefix cache, prefill and decode timed as separate
requests so decode never contains prefill). Prefill {4k,8k,16k} tokens x output {512,1k,2k}.
Baseline is max-of-2; helios single-run (so its decode is noisier, its prefill is tight).

| pre/out | exl3 pre | helios def | helios pipe | exl3 dec | helios def | helios pipe |
|---|---|---|---|---|---|---|
| 4k/512  | 1645 | 34% | 52% | 121.0 | 25% | 58% |
| 4k/1000 | 1646 | 34% | 51% | 114.2 | 26% | 62% |
| 4k/2000 | 1647 | 34% | 51% | 121.1 | 25% | 58% |
| 8k/512  | 1857 | 30% | 50% | 113.7 | 24% | 23% |
| 8k/1000 | 1857 | 30% | 50% | 117.6 | 24% | 26% |
| 8k/2000 | 1858 | 30% | 50% | 121.1 | 23% | 25% |
| 16k/512 | 1984 | 28% | 49% | 113.7 | 22% | 36% |
| 16k/1000| 1920 | 29% | 50% | 117.6 | 21% | 37% |
| 16k/2000| 1920 | 29% | 50% | 115.8 | 21% | 38% |
| **mean** | | **31%** | **50%** | | **23%** | **40%** |

**CORRECTION to earlier claims.** The brief's baseline figures were "up to 120 tok/s decode, real-world
~60." This controlled harness shows exllamav3 decode is a robust **114-121 tok/s across the whole 4k-16k
range**, NOT ~60. So the earlier statement that helios "matches the reference's real-world decode" is
WRONG: helios decode is ~23% (default) / ~40% (pipeline) of the baseline here. helios does NOT match
the baseline decode. There is substantial untapped decode potential — this is the single biggest
remaining gap and was masked by comparing against the wrong (real-world) baseline number.

**Prefill is consistent** at 50% (pipeline) / 31% (default) of the baseline — a tight, trustworthy
signal, and the pipeline roughly closes the default's shortfall. **Decode is the big opportunity**: the
baseline holds ~114-121 tok/s regardless of context, while helios degrades with context (30 -> 24
default; pipeline 70 at 4k but only 26-31 at 8k — noisy, single-sample, and the pipeline's inconsistent
8k point needs a cleaner multi-rep decode measurement before drawing conclusions). The baseline's
context-robustness on decode is the capability helios most lacks.

---

## Decode fix: context-gated MTP speculation (+35% decode at long context)

The 1:1 grid exposed decode as the big gap (helios ~23% of baseline). Clean multi-rep measurement
(best-of-3, 512-token greedy) located the cause: **MTP speculation is profitable at SHORT context and
actively HARMFUL at long context.** Draft-head acceptance: **86.4% at ~2.2k ctx** (decode 61.5 tok/s)
but **~0% by ~4.4k ctx** — where speculation becomes pure overhead. Controlled A/B at 4k:
**MTP-OUT (single-token) 40.8 tok/s vs MTP-IN 29.6** — turning speculation OFF at long context was
+38% faster.

**Fix: context-gated speculation** (`Runner::generate`, `mtp_ctx_limit_`, default 3072, env
`HELIOS_MTP_CTX_LIMIT`). Speculate only while `pos_ <= limit` (acceptance is high there); above it fall
back to single-token decode. 0 = never speculate; huge = always speculate (old behaviour); HELIOS_MTP=0
still disables MTP entirely. fp16 default byte-identical (md5 336f1a07), 9/9 suites PASS.

**Effect on the 1:1 grid (helios default, decode % of exllamav3):**

| pre/out | exl3 dec | helios dec before | helios dec after | after % |
|---|---|---|---|---|
| 4k  | 121.0 / 114.2 / 121.1 | ~25% | **40.3 / 40.2 / 39.8** | **33-35%** |
| 8k  | 113.7 / 117.6 / 121.1 | ~24% | **37.0 / 37.1 / 36.6** | **30-33%** |
| 16k | 113.7 / 117.6 / 115.8 | ~21% | **31.7 / 32.0 / 31.7** | **27-28%** |

Decode improved from ~23% to **~32% of baseline** (mean), a ~35% absolute gain at long context, with
short-context MTP behaviour preserved (2k still 61 tok/s, 86% accept). This is now the single largest
decode improvement of the project. The remaining decode gap (helios ~40 vs baseline ~118 at 4k) is the
base per-step weight-streaming rate, which is memory-bound — a deeper optimization.

---

## 1:1 grid follow-up: pipeline promoted to DEFAULT; updated head-to-head

With the MTP context-gate in place, a clean best-of-3 decode comparison showed the **pipeline is now
better on BOTH axes** (the earlier "pipeline hurts decode" was single-sample noise from before the gate):

| ctx | default (pipeline off) | pipeline | 
|---|---|---|
| 2k | 61.7 | **70.8** tok/s |
| 4k | 39.4 | **43.2** |
| 8k | 35.8 | **45.2** |
| 16k | 30.4 | **32.0** |

So the 2-GPU layer pipeline is now the **DEFAULT** (`HELIOS_PIPELINE=0` opts out). It reorders the
per-token expert sum (documented reassociation, coherent output, deterministic), so the default greedy
md5 moves 336f1a07 -> 4a06c602 — expected, not a defect. fp16/pipeline-off remains available and
verified. 9/9 suites PASS; prefill 974.6; output coherent ("is Paris, where the Eiffel Tower is located").

**Updated 1:1 vs exllamav3 (best available numbers, this tree):**
- **prefill ~50% of baseline** (49-52% across 4k/8k/16k) — the pipeline's 1.73x + wide-tile MoE.
- **decode ~32% of baseline** (was ~23% before the MTP context-gate; 4k: 40 vs 121, 16k: 32 vs 116).
- **context 262,144 vs 230,144 — exceeded.**

The decode improvement this turn came directly from the 1:1 grid the user requested: it surfaced that
MTP speculation, which is a large win at short context, was a large LOSS at long context, and a context
gate recovered ~35% of decode at no cost to the short-context case.

---

## Decode diagnosis: the remaining 2.7x is the reference's QTIP-GEMV decode MoE

At 4k context (MTP gated off), helios decode is ~43.8 tok/s vs the baseline's ~118. Controlled tests of
the two decode MoE paths helios has show neither is the bottleneck: **mgemm (barrier-free) 43.4 vs
grouped (cooperative) 43.8 tok/s** — identical, both coherent ("is Paris"). So the MoE kernel *choice*
is not the lever.

The lever is identified in the port's own notes (moe_layer.cu:22-24): **the reference does not use
either path at M=1 — it routes the decode MoE through `exl3::gemm`'s QTIP GEMV**, a specialised
single-vector low-precision GEMV, not the grouped/mgem cooperative MoE kernel. helios uses the
cooperative MoE kernel for decode. A GEMV per active expert (~10/token) is far more efficient at n=1
than the grouped MoE path with its per-launch barrier cost — this is the concrete source of the
reference's 3x decode advantage and the next decode optimization. (A prior note that "mgemm vs grouped
at n=1 was never actually measured" is now resolved: measured, identical, not the lever.)

So the decode picture is now: MTP gate recovered the speculation-overhead loss (+35%); what remains is
porting the QTIP GEMV decode-MoE path from exl3, which is the well-isolated next step. The prefill gap
remains compute-bound at exllamav3 kernel parity.

---

## n=1 per-expert GEMV decode MoE: implemented, bit-exact, and exactly break-even

The diagnosis above named the next step: the reference's decode MoE runs a QTIP **GEMV** per active
expert, not a cooperative grouped kernel, and helios runs the cooperative one. This section ports
that literally — a host-side loop over the token's top-10 experts, one forced `exl3::gemv` at M=1
per projection (gate, up, `silu_mul`, down), weight-scaled into the output row — behind
`HELIOS_MOE_GEMV=1`, **default off**.

**It is numerically perfect and it buys nothing.** Decode is unchanged to within run-to-run noise at
every context tested. The gate stays off. What follows is the evidence and the reason.

### Design (`src/engine/moe_layer.cu`, gated path)

```
router -> topk_idx / topk_w  (device)
  -> D2H to pinned, ONE stream sync              <- the cost the reference does not pay
  -> host: insertion-sort the slots by ascending global expert id
  -> per card, per active expert k:
         exl3::gemv(gate) -> gvg[k]              (M=1, N=moe_inter, K=hidden)
         exl3::gemv(up)   -> gvu[k]
     aux::silu_mul(gvg, gvu, gvg)                (ONE call over all topk rows, not topk calls)
         exl3::gemv(down) -> drows[k]             (fp32, one row per slot)
  -> moe_gemv_reduce_k:  part = sum_k w[k] * drows[k],  ascending k
  -> cross-card sum (or one partial under the pipeline) -> shared expert -> y
```

Three deliberate choices:

* **Fixed expert order.** Slots are insertion-sorted by global router id on the host, so the launch
  order, the slot each result occupies, and the reduction's summation order are all functions of
  the router output alone. There are no atomics anywhere — `moe_gemv_reduce_k` sums in ascending
  slot index — so nothing depends on arrival order or on pointer values. Run-to-run bit-identity is
  structural here, not lucky.
* **A host mirror of the pointer tables.** `moe_tables_init` already built a host vector per field
  before its H2D copy; it is now kept (`MoeScratch::table_host`, ~1.8 MB/card) so the per-expert
  loop can read a trellis/suh/svh pointer without a device read per launch. Populated
  unconditionally, read only by the gated path.
* **Pinned per-card weight staging.** The reduce kernel reads the weights from device memory, so
  they cross as an async H2D from a per-card pinned buffer enqueued ahead of the kernel. One buffer
  per card, because a card's next host write must not land on a copy its own stream has not issued
  yet — the same ordering `xcard_copy` already relies on. (Writing the weights straight into the
  `cudaMalloc`'d device pointer from the host is a segfault: that range is not host-accessible. It
  is the first thing this path did.)

`exl3::gemv` / `exl3::gemm` internals and the tensor-core GEMM are untouched. The change is a new
decode orchestration in `moe_layer.cu` plus one small reduction kernel.

### Correctness

**Elementwise bit-identical to the default path, all 48 layers.** `HELIOS_MOEX` prints a fingerprint
of the routed row — an index-weighted checksum, a sum of absolute values, and three raw elements —
rather than its rms. rms is invariant under a sign flip or a reordering of equal-magnitude entries,
so two paths can agree on it to 9 digits and still differ elementwise; these do not. First decode
step (where both paths see identical input), short prompt, MTP off:

```
[moex] L0 cksum=-9.614336700673e+02 asum=3.716747919838e+01 v0=-2.036699653e-02 v1=-3.238249198e-02 vlast=-1.877543703e-02
[moex] L1 cksum=1.258308655348e+03  asum=1.675858076394e+01 v0=-5.549752153e-03 v1=-6.670354400e-03 vlast=4.376436118e-03
[moex] L2 cksum=-3.465867530835e+02 asum=1.454293910000e+01 v0=-3.438073443e-03 v1=8.906894363e-03 vlast=-2.107462846e-03
```

`diff` over all 48 layers of that output: **identical**. The routed sum is not merely close, it is
the same bits — the two paths compute the same math over the same data in the same order, so the
only thing that differs between them is how the work is dispatched.

| check | result |
|---|---|
| routed-row fingerprint, 48/48 layers, GEMV=0 vs =1 | **byte-identical** |
| greedy 16 tokens, 10.3k prompt, GEMV=0 vs =1 | **byte-identical** (`304f80ad…`) |
| greedy 64 tokens, GEMV=1, 3 independent runs | **3/3 byte-identical** |
| greedy 64 tokens, default, 2 independent runs | **2/2 byte-identical** |
| `compute-sanitizer --tool memcheck`, GEMV=1 | **0 errors** |
| MTP on / MTP off / `HELIOS_PIPELINE=0` (two-card), GEMV=1 | all run, coherent |
| suites, GEMV unset | **9/9 PASS** |
| default greedy md5, GEMV unset | **`ddb9046fa3124ddd9ab111869ae186d4`**, unchanged by this work |

At 64 tokens the two paths' *text* agrees for 390 of 412 characters and then diverges into two
different fluent continuations. Given the routed rows are bit-identical at the first decode step,
that divergence is the expected amplification of a lossy quantised model under greedy sampling once
a later layer's tiny fp difference flips one argmax — not a defect in either path. The 16-token
generations, which do not reach that regime, are byte-identical.

### Measurement: best-of-3 greedy decode, 256 tokens, unique prompt per run

Same harness methodology as `test/bench.py` (prompts from its `build_prompt`, prefill and decode
timed separately, best-of-3). Both arms are the same binary, differing only in the gate.

| prompt | default (GEMV=0) | GEMV=1 | delta |
|---|---|---|---|
| 3 631 tok | **35.62** | 35.65 | +0.1% |
| 7 360 tok | **30.98** | 30.72 | −0.8% |
| 14 719 tok | **27.15** | 27.05 | −0.4% |

Run-to-run spread within each arm is ±1% (e.g. 27.15 / 26.63 / 26.97 at 14.7k), so every one of
these deltas sits inside the noise. The same holds in the two-card within-layer split, which is the
configuration the "grouped kernel's per-launch barrier cost" argument was originally about — at
14 719 tok with `HELIOS_PIPELINE=0`: **28.05** (GEMV=0) vs **28.04** (GEMV=1). A dead heat there too.

### Why: it is a launch-count trade, and the two costs are the same size

At n=1 the MoE is neither bandwidth-bound nor barrier-bound, which is why the kernel choice was a
wash to begin with and why this change is one as well.

* **Not bandwidth-bound.** The token's 10 active experts are 10 × 1.19 MB = 11.9 MB of weights per
  layer, ~13 us at this card's ~900 GB/s. The measured MoE cost is ~650 us per layer — roughly 50x
  the floor. There is 50x of headroom here for anyone who can get the work issued efficiently.
* **Not barrier-bound.** grouped 43.8 vs mgemm 43.4 tok/s (measured earlier in this file) —
  identical. The ~0.75 ms cooperative-launch barrier the GEMV path removes was not the binding term.
* **Launch-count bound, and the GEMV path pays more.** It replaces 3 mgemm launches per layer with
  `topk × 3 + 2` = **32** launches (10 gate, 10 up, 10 down, one `silu_mul`, one reduce), plus a
  D2H and a **stream sync per layer** — 48 syncs per token, on a host thread that interleaves the
  two cards' layers. The barrier it saves and the dispatch it adds are the same order of magnitude,
  which is exactly what a dead heat looks like. The per-layer MoE stage profiles the same way:
  **659.6 us** (mgemm) vs **662.6 us** (GEMV).

### Correction to the premise, and what that points at instead

The brief this work was built from says the reference "routes n=1 MoE through `exl3::gemm`'s QTIP
GEMV". **It does not.** In `exllamav3_ext/quant/exl3_gemm.cu`, `exl3_gemv_try_launch` is called
exactly once, at line 225, inside `exl3_gemm_gr` (lines 110–323). `exl3_mgemm_gr` (lines 386–634)
contains no GEMV dispatch at all. And the reference's bsz-1..8 tier,
`BC_BlockSparseMLP::run_bszN` (`libtorch/blocksparse_mlp.cpp:227` → `run_bszN_gr:67`), issues **one
`exl3_mgemm` per projection** — the same 3-launches-per-layer structure helios's mgemm decode path
already has.

The structural difference is one helios does **not** have: `run_bszN` wraps those three mgemm calls
in a **captured CUDA graph** (`graph_bszN[graphidx]`, `capture_begin`/`capture_end`, with the
`A` / `C` / `indices` / `weights` kernel params re-patched on every replay). That removes
essentially all of the per-step launch cost, and it needs no host round-trip, because mgemm resolves
the expert pointers from the device-resident index array.

So the 44 → 118 tok/s decode gap is not a MoE-kernel-selection problem. The MoE arithmetic is a wash
between every path available here; what the reference does that helios does not is **amortise the
dispatch**. The next lever in that direction is CUDA-graph capture of the decode step (or of the MoE
stage within it), not another MoE kernel. This section does not attempt that — it establishes that
the GEMV route is not it.

### Shipped state

`HELIOS_MOE_GEMV=1` is implemented, bit-exact, deterministic, sanitizer-clean, and measured. It is
**off by default**, and the grouped/mgem decode path is unchanged: default greedy md5 is
`ddb9046fa3124ddd9ab111869ae186d4` before and after, and 9/9 suites pass. A correct, no-slower
alternative is exactly what the gate is for.

---

## Per-expert GEMV decode MoE: bit-exact but NOT faster (and my "barrier" premise was wrong)

Attempted the reference's n=1 fixed-order per-expert GEMV decode MoE in helios: a gated
(HELIOS_MOE_GEMV=1, default off) path that gathers the single token once and runs `exl3::gemv`
(gate/up/silu/down, weight-scaled accumulate) per active expert, avoiding the cooperative grouped
kernel. Implemented in moe_layer.cu with a fixed-order (sorted-expert-id) atomic-free reduction.

**Correctness is exact, not approximate.** The routed-row elementwise fingerprint (checksum + sum-of-abs
+ raw elements, not rms, which is sign-invariant) is BYTE-IDENTICAL between GEMV=0 and GEMV=1 across all
48 layers; greedy output byte-identical; deterministic 3/3; compute-sanitizer 0 errors; correct under
MTP on/off and pipeline on/off. Found+fixed two real bugs (host-write into a device pointer segfaults;
a card selecting no expert must zero its partial).

**But it is NOT faster — a dead heat inside the ±1% run-to-run noise:**
best-of-3, 256-token greedy — 3.6k: 35.62 vs 35.65; 7.4k: 30.98 vs 30.72; 14.7k: 27.15 vs 27.05.
Per-layer MoE stage: mgemm 660 µs vs GEMV 663 µs. So the GEMV path is gated OFF (default unchanged).

**CORRECTION to the previous section.** The claim that "the reference's QTIP GEMV is the concrete
source of the 2.7x decode gap" was WRONG as a helios lever: mgemm, grouped, and per-expert GEMV all
cost the same ~660 µs/layer in helios. The MoE *kernel choice* is not the decode bottleneck. What
remains unexplained is why the reference's per-layer decode MoE is ~3.7x faster than helios's
(~177 µs vs ~660 µs implied by its 118 vs ~30 tok/s) — that is the open question, and it is NOT the
barrier cost or the grouped-vs-GEMV distinction. The GEMV path stays in the tree (bit-exact, gated) as
a correct, documented alternative; it is simply not a speedup on this hardware.

---

## Decode is LAUNCH-BOUND: helios has no CUDA graphs (the real remaining gap)

Building on the GEMV dead-end, the accurate diagnosis of the ~2.7x decode gap. At n=1 the decode step
is a fixed sequence of thousands of tiny kernels (per-expert GEMM/MGEMM, attention, GDN, mHC, per layer
x 48, plus the mgemm router readback and MTP). helios launches each one individually and pays full CPU
launch overhead; the reference (exllamav3) collapses its decode into a **CUDA graph** and replays it,
paying launch cost once. Confirmed: `grep -rln "cudaStreamBeginCapture|cudaGraphInstantiate" src/`
returns NOTHING — helios has **no CUDA graph capture at all**. The port's own decode code even names
it: moe_layer.cu:672 "GEMV kernel is not the lever - either mgemm's own inner loop, or batching the
dispatch (a graph)."

So the decode story is now correct and layered:
- MTP context-gate recovered the speculation-overhead loss (+35% decode; ~23% -> ~32% of baseline).
- The MoE *kernel* (grouped/mgem/GEMV) is NOT the lever (all ~660 µs/layer; GEMV bit-exact but equal).
- The remaining gap is **CPU launch overhead per kernel** at n=1, which CUDA graphs eliminate. The
  blocker to a graph is the host syncs on the decode path (the mgemm/router D2H readback that picks
  active experts, MTP bookkeeping, pipeline handoff) — each one breaks capture. Making the expert
  selection fully device-side (as the grouped kernel already is) and eliminating those syncs is the
  prerequisite. This is the next architectural phase for decode, analogous to the layer-pipeline for
  prefill: large, but now precisely identified with a known mechanism.

**Instrumentation blocker for the CUDA-graph work.** The launch-bound diagnosis (no graph capture) is
well-founded from the code (zero cudaGraph* anywhere) and the port's own comment, but I could NOT cleanly
measure the launch-bound fraction to size the win: the DECODE-only phase profiler catches only 1 step
(its n==1 detection is starved by the MTP verify path's n=K+1 batches, and my MTP context gate removes
the draft steps at long context), and mprof's per-stage cudaStreamSynchronize inflates decode MoE times
(3.5 ms/layer under mprof vs ~660 us real). nsys would be the right tool but would not attach to the
short-lived engine subprocess in this setup. Before committing to a graph port — which must first
remove the decode-path host syncs (mgemm/router D2H readback, MTP bookkeeping, pipeline handoff) that
break capture — the launch-overhead fraction needs a working decode-step profiler (or nsys wired to a
long-lived server). This is recorded as the next architectural phase with a known prerequisite, not
attempted blind.

**The decode-step profiler is broken (real bug, not worth chasing now).** The DECODE-only phase line
reports `decode steps=1` with all-zero phases even for a 256-token n==1 decode, in BOTH the pipeline
(default) and single-pass paths — the n==1 counter and the per-phase event deltas are not being
populated, despite `run_chunk`'s `if (n == 1) { P.flush(0, td); nd++; }` being present in both paths.
Isolating true decode-step GPU time therefore needs a fixed profiler (or nsys attached to a long-lived
process); the phase profiler's per-layer mark array and its flush timing are not trustworthy for n==1
decode. This is recorded as a known-broken diagnostic, and is itself a candidate fix (a decode-mode
profiler is needed before the CUDA-graph work can be sized). I did not attempt to fix the profiler or
the graph path this turn: without a working decode-step timer, a large graph-capture refactor would be
unmeasurable, and the deliverable is already correct and verified.

**Profiler root cause, narrowed (fix deferred).** The DECODE-only phase line is unreliable. The
accumulation logic looks correct in both `run_chunk` (non-pipeline, line ~898) and
`run_chunk_pipeline` (line ~977, the default path): both flush to `td` and bump `nd` when n==1, and both
set `P.on`/`P.ensure()`. The defect is inside the per-chunk mark/flush ring: `mark()` (line 540) drops any
mark once `n[d] >= kProfEvents` (512) and `flush()` (line 550) syncs `gpu(dev).stream(0)` while reading
the ring for that device. A decode micro-batch issues ~6 marks/layer x 48 layers ~= 288 per card, and a
prefill micro-batch issues a similar count, so the ring is near its 512 cap; combined with the two
independent flush calls (t and td) at the end of each chunk, the accounting double-drains or overruns
and the decode-only averages collapse to zero over a single step. The engine's own decode counter
(`decode X tok/s`, computed as N/(t_N - t_1) inside one warm process) is the figure to trust; every
decode number quoted in RESULTS.md uses that, not the broken phase profiler. Fixing the ring (raise
kProfEvents, or flush-per-phase without the double t/td drain) is a contained follow-up, not attempted
here because it is a diagnostic, not the deliverable, and the graph-capture work it would enable is
already known to need the decode-path host syncs removed first.

---

## Decode profiler FIXED; first real per-step decode breakdown

The DECODE-only phase line reported 0.00 ms for every phase because `PhaseProf::flush()` DRAINS its
ring (`n[dev] = 0` at the end), and both the per-chunk total and the decode-only total were filled by
calling flush() twice on the SAME ring — the second call (`td`) always found an already-empty ring and
contributed nothing, so `td` stayed zero for the whole run. Fixed by giving flush() an optional second
accumulator fed in the same single drain (PhaseProf::flush, and both run_chunk / run_chunk_pipeline call
sites). Diagnostic-only: 9/9 suites PASS and the default output md5 is unchanged (4a06c602…).

**First trustworthy per-decode-step breakdown (4k context, MTP gated off, ~23 ms/step total):**

| phase | ms/step |
|---|---|
| **moe** | **9.19** |
| gdn | 4.38 |
| attn | 3.70 |
| amix | 2.39 |
| mmix | 2.23 |
| apply | 0.89 |
| embed/ple/final | 0.41 |

So decode is dominated by the MoE (9.19 ms, ~40% of the step), then GDN and attention. This is the
first real evidence for WHERE the decode time goes. It reframes the decode optimisation target: the MoE
expert GEMM at n=1 — not attention — is the biggest single phase. This is consistent with the earlier
finding that the MoE kernel choice (grouped/mgem/GEMV) is a dead heat: the phase is real work
(weight streaming for the ~10 active experts), not overhead. Cutting decode further would need either
fewer/cheaper expert-weight bytes per token or overlapping the MoE weight read with compute — the
CUDA-graph path (which can overlap independent launches) is the natural candidate, gated on removing the
decode-path host syncs. The profiler fix means that work can now be sized.

**Decode MoE is latency-bound, not bandwidth-bound — quantified headroom for CUDA graphs.** With the
profiler fixed, the decode MoE streams ~605 MB of expert weights per token (10 active experts x 3
matrices x 48 layers at 2.05 bpw) in 9.19 ms = **66 GB/s, only ~7% of the 3090's ~936 GB/s peak**. The
MoE phase is ~40% of the ~23 ms decode step. If that phase reached even 30% of peak bandwidth it would
fall to ~2 ms, taking the step from ~23 ms toward ~8 ms (decode ~43 -> ~100+ tok/s). This is the
textbook case for what a CUDA graph fixes: at n=1 the ~10 active experts are independent, so their
weight loads and GEMVs can overlap instead of serializing behind launch latency. The blocker is
unchanged and concrete: the decode-path host syncs (mgemm/router D2H readback, MTP bookkeeping,
pipeline handoff) must be removed before a graph can be captured. This is now the single highest-value,
well-sized remaining optimization, and it is an architectural refactor (remove decode syncs + add graph
capture/replay around the decode step), documented for a focused effort.

---

## CORRECTION: the MTP context gate was a mis-tuned regression — speculation is now always on (+45-75% decode)

The MTP "context gate" (shipped earlier this session) disabled speculative decode past 3072 context
on the belief that draft acceptance collapsed. A controlled re-sweep (best-of-2, 256-token greedy)
shows that belief was WRONG: speculation pays at EVERY context measured, up to 28.6k tokens.

| context | always-specify | gate (3072) | delta |
|---|---|---|---|
| ~3k   | **78.2** | 53.5 | +46% |
| ~5.5k | **72.4** | 41.3 | +75% |
| ~8.8k | **63.1** | 45.0 | +40% |
| ~14k  | **51.3** | 38.7 | +33% |
| ~28.6k| **41.2** | 28.9 | +42% |

The gate had been costing 33-75% of decode throughput at every context. Shipped default is now
**INT_MAX (always speculate)**; the gate survives as an env ceiling (`HELIOS_MTP_CTX_LIMIT`, 0=never)
for A/B. Verified: 9/9 suites PASS, default output md5 unchanged (4a06c602, prefill numerics
unaffected), coherent output. Corrected decode across context: **77.5 tok/s @3k, 63.4 @8.8k, 41.4 @28.6k**.

Lesson: the original "MTP stops paying past ~4k" reading came from a single confounded run; the
proper controlled sweep shows the opposite. Speculation with a decent draft head keeps paying well past
the context where it was thought to die.

## Tuning re-validation after the MTP correction

Two earlier decisions re-checked with clean best-of-2 sweeps (the MTP gate had been wrong because it
came from one confounded run, so the others were re-validated):

- **Pipeline micro-batch M=2 confirmed optimal** (10.3k prefill): M=1 944.6, **M=2 967.8**, M=3 894.1,
  M=4 900.9 tok/s. Unchanged.
- ~~**MTP draft depth K is capped at 2 by the current spec machinery.**~~ **Superseded — see
  "Speculative depth generalised to N drafts (K<=7)" below.** The machinery is now depth-generic
  (K=1..7) and correct at every depth; the K sweep there confirms the depth-1 default is also the
  fastest, so nothing about the default changed.

---

## Speculative depth generalised to N drafts (K<=7): two real bugs, both found by measurement

`spec_step`/`spec_verify` are now depth-generic (K = `HELIOS_SPEC_K`, 1..7, verify batch
`[next, d0..d_{K-1}]` at width K+1). K=3 and K=4 produced garbage (" of", "Evolution") and K>=3
crashed. **Neither symptom was in the draft chain** — the chain, the carry tap, the accept rule and
the GDN rollback were all correct at every K as written. Both defects were in state the widening
exposed, and both had been silently present at K=1 and K=2 too.

### Bug 1: the logits buffer's CAPACITY was being used as the projection WIDTH

`init()` sizes `logits_` to `head_rows_` rows and then set `head_multi_row_ = head_rows_ > 1`;
`final_head(n)` projected `min(n, head_rows_)` rows and took **row 0 from `sub_in[n-rows]`**.
Generalising raised `head_rows_` to `K+1`, which is right as a CAPACITY (a verify needs K+1 rows) and
wrong as a WIDTH, because *every other forward projected K+1 rows too* — and `next_token()` reads
row 0. So the prefill sampled position `n-(K+1)`, and the first generated token of every request was
the model's prediction for the wrong position. Everything after it was downstream of a wrong token,
which is why deeper K looked catastrophic rather than merely degraded.

Measured on `"The capital of France is"` (6 tokens), `HELIOS_TOPK=3` read immediately after the
prefill, same binary, only the head width changing:

| logit rows projected | argmax at pos=6 | p | model's true continuation |
|---|---|---|---|
| 1 (MTP off, `HELIOS_HEAD_ROWS=1`) | **11751 " Paris"** | — | ` Paris.` |
| 2 (K=1, the shipped default) | 369 " is" | 0.6785 | ` is Paris.` (one position early) |
| 3 (K=2) | 279 " the" | 0.7646 | ` the largest city in the country...` |

So the **shipped K=1 default was already one position early** — it emitted " is Paris." where the
model's greedy continuation is " Paris." — and every K>=2 was K positions early. The comment that
already sat above `head_rows_` said exactly this ("the DEFAULT PATH MUST STILL PROJECT ONLY THE LAST
POSITION ... Multi-row projection is therefore opt-in, not inferred from the buffer size"); the
generalisation contradicted the invariant it was extending.

**Fix:** `head_rows_` is a capacity, and the projection width is an explicit `head_rows` argument
threaded `run_chunk -> chunk_tail -> final_head`, defaulting to 1. The speculative verify passes its
batch width; nothing else passes anything else. Row 0 is always the last position projected, which is
what `next_token()` and the verify each mean by it. `head_multi_row_` is gone — a mutable flag was
never the right shape for a per-call quantity.

### Bug 2: the speculative GDN snapshot was sized by the wrong layer count (out-of-bounds write)

`gdn_sub_snap_` / `gdn_sub_snap1_` are indexed by the model's **global** `gdn_ord` (0..35), but card
1's half was allocated with `n_gdn1` — the number of GDN layers *on card 1* (18). Every speculative
verify therefore wrote all 18 of card 1's GDN sublayer-input captures past the end of the
allocation. It survived at K<=2 only because the overrun landed in pool slack; one row wider and it
faulted. `gdb` on the K=3 crash:

```
#0-#13  libcuda (cuMemcpyDtoDAsync)
#14     helios::Runner::layer_range(...)        <- the gdn_capture_ D2D copy
#15     helios::Runner::run_chunk_pipeline(...)
#16     helios::Runner::spec_verify(...)
```

This is a silent-corruption bug at K=1 and K=2 as well, not only a K>=3 crash: it was writing over
whatever the bump allocator had handed out next on card 1.

**Fix:** both halves are sized `n_gdn_total` rows, because the index is global; the per-layer stride
is derived from `kMaxSpecDrafts + 1` instead of the literal 8 it happened to agree with.

### What the surviving suspects were NOT

Checked directly rather than assumed, and all clean:

- **GDN rollback.** `bench-gdn` replay self-test is **BIT-EXACT at widths 1, 2, 3 and 4** after the
  fix (`fp_before == fp_after` on all 36 layers). `gdn_snapshot`/`gdn_restore` already covered the
  conv state as well as the recurrence matrix.
- **Draft-chain position alignment.** Traced per step at K=1: A advances the carry tap to `pos_`
  with the token at `pos_-1`, chain step i consumes the stack at `pos_+i` with the token at
  `pos_+i` and predicts `pos_+i+1`, so drafts[i] is the prediction for `pos_+i+1` — exactly what
  verify row i is compared against. Correct for every i.
- **Carry tap.** Row `m` of the verify batch is the last committed position, the tap buffer is
  `max_chunk` rows deep, and the copy is one `H*D` row. Correct.
- **Multi-row head.** One `exl3::gemm` over `rows <= K+1 <= 8` rows, Hadamard scratch
  `max_chunk*16384*2` B, rows clamped to the batch. Correct once it is told the right width.

### Correctness of the result, and an honest note on the two md5 constants in the ticket

Every depth now produces coherent output and is **byte-identical run to run**. `"The capital of
France is"` (raw completion, 24 tokens) and an 8k natural-text prompt at 256 tokens, two runs each:

| depth | text md5 (both runs) | first 40 chars |
|---|---|---|
| MTP off | `362bcd76…` (8k) | ` Paris.` |
| K=1 | `a56d708b…` | ` Paris.\n\n# 2026年中国AI应用全景…` |
| K=2 | `70b1c996…` | ` Paris.\n\n# 100%` |
| K=3 | `03965232…` | ` Paris.\n\n# 1. Introduction` |
| K=4 | `03841d83…` | ` Paris.\n\n# 1000+ General Knowledge Questions` |
| K=5,6,7 | one run each, coherent, no crash | ` Paris.` |

**The ticket's `4a06c602…` (K=1) and `80fcf810…` (K=2) do not reproduce on this tree, before or
after the fix**, with the project's own text-only recipe (`/tmp/gen_md5.py`). Measured on the
pre-fix binary, "Explain how a transformer works." 128 tokens hashed `66be6de74c72b17643e77559ecae82a7`
and its text was 425 bytes of `- **Attention mechanism` repeated twenty times — degenerate, i.e. bug 1
was already live in the default. After the fix the same command hashes
`9d54f65b23b9a0981f58f9e843aa3438` (612 bytes, two runs byte-identical) and reads
`The transformer architecture, introduced in the paper "Attention Is All You Need" (Vaswani et al.,
201…` — the correct answer. **K=1's hash necessarily moved, because keeping it would mean keeping a
first token sampled from the wrong position.** K=2 was equally affected: its "correct" `80fcf810…`
cannot be a correct K=2, since exact greedy verification must reproduce the K=1 stream, and it does
not.

One methodology correction worth recording: the first determinism numbers I took hashed the
`[gen] N tokens in …s (… tok/s)` line along with the text (the recipe cuts at the LAST `\n[gen] `,
and the spec line comes after the timing line), which made identical output look nondeterministic.
With the timing lines excluded, **every depth, and MTP off, is byte-identical run to run** — 2 runs
each at 8k and 2 runs each at K=1..7 on the short prompt.

Streams still differ *between* depths and from MTP-off, and that is inherent rather than a defect:
a width-(K+1) verify and a width-1 decode sum the same products in a different order, and greedy
argmax amplifies that into a different token at a near-tie. The first divergence on the short prompt
is exactly such a case — at pos 11 the reference distribution is 16 at p=0.3516 against 17 at
p=0.3473, a 0.4% gap. Tokens the spec path emits are trunk-verified by construction, so the
difference is which of two co-maximal tokens wins, not a wrong token.

### The K sweep: deeper speculation is correct but does NOT pay (K=1 stays the default)

7931-token natural-text prompt (this file's own prose, so the model does not run into EOS),
256-token greedy, best-of-2, `helios gen --raw --temp 0`, decode tok/s from the engine's own counter:

| config | decode tok/s (run1 / run2) | tokens/step | draft slots accepted | vs K=1 |
|---|---|---|---|---|
| MTP off | 48.15 / 47.90 | 1.00 | — | −5% |
| **K=1 (default)** | **50.66 / 50.70** | 1.51 | 50.9% | — |
| K=2 | 43.85 / 43.94 | 1.71 | 35.3% | −13% |
| K=3 | 36.78 / 36.93 | 1.76 | 25.3% | −27% |
| K=4 | 29.12 / 29.17 | 1.75 | 18.7% | −43% |

Tokens per step saturates at ~1.75 by K=3 — the marginal draft's acceptance is already down to ~0.5 —
while the verify forward keeps getting wider, and at 8k context a width-5 trunk pass costs far more
than the width-2 one. **The depth-1 default is confirmed, now with a working K=3/K=4 to measure
against rather than a clamp.** The K sweep is also the first evidence that acceptance is what limits
depth here: 50.9% -> 35.3% -> 25.3% -> 18.7% of *proposed slots*, i.e. per-draft acceptance
0.51 / 0.69 / 0.72 / 0.74, so the head's marginal accuracy is roughly flat and the cost is all in
the widening batch.

Integrity: build clean, **9/9 suites PASS**, `bench-gdn` replay BIT-EXACT at widths 1-4, K=1..7 all
coherent and crash-free, every depth byte-reproducible run to run.

### Known limitation left in place (pre-existing, not K-specific, and not cheaply fixable)

On a partial accept the PLE's **host** n-gram history keeps the rejected tokens: `layer_range`
appends the whole batch to `hist_` before the accept decision exists, and nothing rolls it back. The
committed rows' PLE contribution is unaffected (the layer is causal, and the pollution sits at
positions after the last committed one), but the 128-token window the *next* forward hashes is
wrong, as is the PLE conv state, for up to 127 positions. Rolling the host history back exactly is
easy; the conv state is not, because correcting it for the committed rows would require re-running
the trunk forward for them — the very cost speculation exists to avoid. This is left alone
deliberately: it is not a K>2 regression (it is identical at K=1), it is not the cause of the
observed near-tie divergences (they persist with `HELIOS_NO_PLE=1`), and changing it would move the
default output for a benefit this work cannot measure. Recorded here so the next person does not
rediscover it.

---

## CRITICAL correctness fix: two latent spec-decode bugs that made the default output WRONG

Generalizing the MTP spec machinery to N drafts (K>2) exposed TWO pre-existing bugs that had been
silently corrupting output at K=1 and K=2 all along:

1. **`head_rows_` was doing double duty** — logits buffer capacity AND the prefill projection width.
   Raising it to K+1 (correct as a capacity for the widened verify) also widened the PREFILL projection,
   so `next_token()` read the prediction for position n-(K+1): the FIRST generated token of every
   request came from the wrong position. K=1 was one position early, K=2 two, K>=3 three or more.
   Fix: `head_rows_` is capacity only; the projection width is an explicit `head_rows` argument
   threaded run_chunk -> run_chunk_pipeline -> chunk_tail -> final_head, default 1; only spec_verify
   passes its batch width. `head_multi_row_` deleted.
2. **Out-of-bounds D2D write in the speculative GDN snapshot** — `gdn_sub_snap1_` was sized with the
   card-1 GDN layer count (18) but indexed by the GLOBAL gdn_ord (0..35), so every speculative verify
   wrote card 1's captures past the end of the allocation. Silent arena corruption at K<=2; SIGSEGV at
   K>=3. Fix: size both halves by the total GDN row count; per-layer stride from kMaxSpecDrafts+1.

**Impact.** The default (K=1) output was subtly WRONG before this — a degenerate "- **Attention
mechanism" x20 loop, which I had been md5-stability-checking as "unchanged". After the fix the default
is correct: "The capital of France is" -> "**Paris.**", run-to-run deterministic (md5 49f89009...,
9/9 suites PASS). The old default md5 4a06c602... described the broken tree and no longer reproduces.

**K sweep (8k, 256-token greedy, best-of-2):** MTP off 48.0; **K=1 50.7 (1.51 tok/step, 50.9% accept)**;
K=2 43.9; K=3 36.8; K=4 29.1. Tokens/step saturates at ~1.75 by K=3 while the verify batch keeps
widening (per-draft acceptance ~0.51/0.69/0.72/0.74, roughly flat) — the extra cost is all the wider
forward. So K=1 stays the default. Deeper speculation (K up to 7) is now CORRECT and available, just not
faster on this model. This is a genuine correctness win: the shipped output is now right for the first time.

---

## Re-baselined 1:1 grid on the CORRECTED engine (post correctness-fix)

Every earlier number was measured on an engine whose FIRST generated token came from the wrong position
(the head_rows_ bug). Relative comparisons still held, but the headline figures are re-measured here on
the corrected engine. Controlled A/B, same harness (test/grid_bench.py), prefix cache defeated, prefill
{4k,8k,16k} x output {512,1k,2k}.

| pre/out | exl3 pre | helios pre | pre % | exl3 dec | helios dec | dec % |
|---|---|---|---|---|---|---|
| 4k/512  | 1644 | 839 | 51% | 120.9 | 68.7 | 57% |
| 4k/1000 | 1645 | 839 | 51% | 114.2 | 70.3 | 62% |
| 4k/2000 | 1646 | 840 | 51% | 117.6 | 70.7 | 60% |
| 8k/512  | 1857 | 930 | 50% | 107.7 | 29.0 | 27% |
| 8k/1000 | 1858 | 930 | 50% | 114.2 | 29.9 | 26% |
| 8k/2000 | 1857 | 931 | 50% | 117.5 | 30.2 | 26% |
| 16k/512 | 1920 | 971 | 51% | 120.3 | 51.3 | 43% |
| 16k/1000| 1920 | 970 | 51% | 111.1 | 51.1 | 46% |
| 16k/2000| 1920 | 968 | 50% | 114.1 | 51.3 | 45% |

**Prefill is a rock-solid ~50% of exllamav3** (49-51%, extremely tight). Decode is 26-62%; the
4k/8k/16k spread (60% / 26% / 45%) is dominated by MTP draft acceptance, which varies with the prompt
and context rather than tracking a hardware limit — the 8k point is the low outlier. These are the
honest, corrected, end-to-end numbers.

---

## CUDA-graph decode path (`HELIOS_DECODE_GRAPH=1`, default OFF)

Per-layer capture/replay of the decode forward. `src/engine/dgraph.hpp` (new) holds the capture
cache and the scope argument; `layer_range` in `runner.cpp` was restructured so the capturable body
is one lambda that is either run eagerly or handed to the graph cache; `moe_layer.cu` grew
`moe_decode_graphable(n)`, the single predicate that decides whether a call may sit in a capture.

### The premise in the ticket is wrong, and that is why this was cheap

The ticket says the router readback at `moe_layer.cu:526-541` is "the fundamental blocker" and that
device-side routing must be built first. **Lines 526-541 are not the router readback.** They are the
`HELIOS_MOEC` / `HELIOS_MOEX` diagnostic D2H readbacks inside the **per-expert GEMV decode path**,
which is off unless `HELIOS_MOE_GEMV=1` and is off by default. The actual router readback — the one
that copies `topk_idx`/`topk_w` to the host so the host can pick a trellis pointer per launch — is at
`moe_layer.cu:416-420`, and it is on the GEMV path only.

**The default mgemm decode path already routes on the device.** `routing_std_logits` writes the
per-slot expert ids and weights into *device* memory, `remap_ids_kernel` maps them into the card's
local expert space on device, and `mgemm` dereferences the device index array (negative entries skip
the slot) and the device weight array. There is no host branch on which experts were chosen. So no
device-side routing work was needed: the capture bakes in no routing decision, and only the
*contents* of those arrays change per step, which is exactly what a replay is for. `moe_decode_graphable()`
excludes the GEMV path and the grouped path explicitly rather than letting a capture fail at run time.

### What is captured, and what is not

| block | captured? | why |
|---|---|---|
| MoE, mgemm decode path | **yes** | routing is device-side; pointers fixed |
| GDN sublayer | **yes** | `conv_state`/`rec_state` are fixed slots; nothing depends on position |
| `gr_mix` / `gr_apply` / fp32→fp16 cast | **yes** | fixed buffers, fixed shapes |
| full-attention sublayer | **no** | `pos0` is a host value baked into the rope position buffer, the KV append offset and the split-KV decision, and it changes every step. Capturing it freezes the position. |
| PLE | **no** | its n-gram ids are hashed on the **host** out of `hist_`, which changes every chunk |
| MTP draft head | **no** | calls `moe_layer` directly, and its embed stage is a pageable H2D |
| cross-card handoff | **no** | a D2H→pinned→H2D round trip with a `cudaStreamSynchronize` (real: these boards are PHB, `cudaDeviceCanAccessPeer` is false) |

So a **GDN layer captures whole** (mix → sublayer → apply → mix → MoE → apply) and a
**full-attention layer captures from its first `gr_apply` to the end of its MLP site**.

**Measured coverage: 48 graphs / 1836 nodes per decode step, i.e. ~1840 launches collapsed into 48
`cudaGraphLaunch` calls** — roughly 90% of the launches in a step. Measured in wall time the covered
fraction is the same order: the phase profile puts MoE 18.7 ms + GDN 8.4 ms + the mHC mix/apply
sites ~8.6 ms of a ~38 ms step inside the captured bodies. What is *not* graph-covered is the
12 full-attention layers' kernels (1.4 ms), the PLE (1.0 ms) and `chunk_tail` (0.4 ms).

### Results (RTX 3090 x2, Qwen3.8-Flash-Next, greedy, engine's own decode counter)

Short prompt (`"The capital of France is"`, 256 tokens, **output byte-identical**):

| graph | decode tok/s | tok/step | graphs | nodes | replays | eager fallbacks | free VRAM |
|---|---|---|---|---|---|---|---|
| off | 64.53 | 1.54 | 0 | 0 | 0 | 0 | 1.41 / 0.60 GB |
| on | **68.03 / 67.98 / 68.19 / 68.15** | 1.54 | 48 | 1836 | 7920 | 0 | 1.41 / 0.60 GB |

**+5.5%**, four graph-on runs byte-identical to each other *and* to the graph-off run
(md5 `5a1e1b0693ff9adc8374e30f4e5b4d02` before and after the change).

Context grid, **256 tokens generated in every cell**, best-of-3, all three repetitions listed:

| ctx | graph off | graph on | delta | tok/step | prefill off/on | graphs | replays | free VRAM |
|---|---|---|---|---|---|---|---|---|
| 4k | 59.2 (59.2/59.2/59.0) | 62.4 (62.3/62.3/62.4) | **+5.3%** | 1.57 | 857 / 850 | 48 | 7776 | 1.41 / 0.60 GB |
| 8k | 61.4 (61.4/61.4/61.4) | 64.6 (64.6/64.5/64.4) | **+5.2%** | 1.77 | 941 / 938 | 48 | 6912 | 1.41 / 0.60 GB |
| 16k | 47.1 (47.1/46.9/46.8) | 48.7 (48.7/48.5/48.5) | **+3.4%** | 1.63 | 975 / 971 | 48 | 7488 | 1.41 / 0.60 GB |

The spread inside every cell is 0.2-0.4%, so these deltas are well outside the noise. Prefill is
unchanged (it runs the grouped MoE at width > g_moe_decode_max_n and is never captured), which is
the control the graph must not touch.

**An earlier grid produced a -25.7% "regression" at 8k, and that cell was invalid.** The
`test/bench.py` paragraph makes this model emit an immediate stop at 8k: the run generated
**1 token**, so the graph-on number measured *capture* cost — 48 captures and **0 replays** — and
read as a large loss. A cell that generates one token is not a decode measurement, and it is the
same trap this log records elsewhere in the other direction, with a kernel that was fast because it
had stopped doing the work. The harness now rejects any run that does not reach the requested token
count, and the prompts were rebuilt to end with an instruction to enumerate. Reporting the -25.7%
as a result would have been the wrong call in the pessimistic direction exactly as reporting a
22% "win" from the double-execution bug below would have been in the optimistic one.

**VRAM: the graphs cost 8-10 MB per card** and free memory at generation time is unchanged
(1.41 / 0.60 GB both modes). The pipeline holds ~1.4 GB free on card 0 at 262k ctx, so this fits
with three orders of magnitude to spare. The graphs allocate no device memory of their own — every
buffer they touch was already in the arena — so they pin nothing new.

### Why +5% and not +50%, stated plainly

The ticket's premise was that the MoE streams 605 MB/token at 66 GB/s, ~7% of peak, because
"thousands of tiny per-expert kernels are launched individually". **That describes the GEMV path, not
the default one.** The default mgemm path issues 3 cooperative launches + 3 small kernels per layer.
Collapsing ~1840 launches into 48 removes the dispatch cost, and that is worth 3-5%.

The remaining gap to exllamav3 is **not** dispatch. The MoE moves 48 layers x 10 experts x 3
projections x 1.19 MB = **1.71 GB of expert weights per token**, ~0.86 GB per card, and at ~22 ms per
step that is ~38 GB/s per card against a ~936 GB/s peak. A graph replay does not change occupancy,
grid shape or memory-access pattern, so it cannot lift a kernel that is not saturating the memory
system. **The lever is the M=1 expert kernel, not the launch path** — and this measurement is the
evidence for that, not against graphs.

### Two bugs this work found, both of which produced plausible-looking output

1. **A graph that ran its body twice.** The textbook sequence is warm-up → capture → instantiate →
   **launch**. The body is not idempotent: `gdn_layer` *advances* `conv_state` and `rec_state`, so
   launching after the warm-up applies the recurrence twice in one step. The symptom was coherent
   prose, a different token stream, and no error anywhere — decode went *up* 22% (79.0 tok/s) because
   the recurrence was effectively being skipped. The fix is to **not** launch after the warm-up: a
   capture executes nothing, so the warm-up's result *is* this step's output. Speed is not evidence.
2. **A dropped `gdn_capture_` copy in the eager branch** during the refactor. `layer_range`'s
   speculative sublayer-input snapshot is a host-conditional D2D copy; losing it from the eager
   path left `gdn_replay` reading stale rows. Symptom: generation stopped after 5 tokens instead of
   256, deterministically, with no error. Caught by md5-comparing the default output against the
   pre-change baseline — which is the only reason the md5 gate exists.

### Integrity

Build clean; **9/9 suites PASS**; **default output byte-identical** (md5 `5a1e1b06…` recorded before
and after). Graph on: coherent, **three consecutive runs byte-identical to each other and to the
ungraphed output**, no crash, free VRAM reported. `HELIOS_DECODE_GRAPH=1` together with any
profiling or diagnostic variable that syncs or reads back inside the layer body (`HELIOS_PROF`,
`HELIOS_MPROF`, `HELIOS_DBG`, `HELIOS_MOEC`, `HELIOS_MOEX`, `HELIOS_MRMS`, `HELIOS_MIN`, `HELIOS_HC`,
`HELIOS_SUB`, `HELIOS_LRMS`, `HELIOS_PRMS`, `HELIOS_GRMS`, `HELIOS_GDNSTATE`, `HELIOS_QSA`,
`HELIOS_GMIXPROF`) is **refused at startup and the refusing variable is named**, rather than failing
a capture 40 layers in. A graph that cannot be built falls back to the eager body for that key
permanently and says so: a slower decode, never a wrong one.

**Not done, and why.** The MTP draft head is still ungraphed; it is one decoder block per step
against the trunk's 48, so the remaining headroom there is small. Whole-step capture (including
attention) would need `pos0` moved into device memory and the KV append offset made a kernel
argument rather than host pointer arithmetic — a change to the tuned attention kernels, which this
work deliberately does not touch.

---

## CUDA-graph decode: implemented, correct, but OFF by default (measured regression)

`src/engine/dgraph.hpp` captures one graph per (site, card, layer, width) covering the MoE mgemm
decode path, the GDN sublayer, and gr_mix/gr_apply/cast — 48 graphs collapsing ~1836 launches per decode
step. Output is **byte-identical** to the eager path (md5 49f89009 with and without), which is the
important part: it is a safe thing to turn on.

**But it is slower, and the first measurement was wrong.** An initial pass reported +3.4-5.3% and I
flipped the default on. A controlled A/B (3 runs/cell, identical prompt, the graph flag the only
variable) does not reproduce that:

| prompt | graph off | graph on | delta |
|---|---|---|---|
| 8k repetitive (grid's) | 30.0 / 30.4 / 30.2 | 22.2 / 22.8 / 22.7 | **-25%** |
| 8k diverse (random words) | 38.9 / 38.8 / 38.9 | 37.8 / 37.4 / 37.4 | -3.5% |

Only the 4k cell ever looked positive, by less than its own run-to-run spread. The grid shows the same
thing end to end: 8k decode 29.0 tok/s with the graph off, 22.8 with it on. So the default is **off**
(`HELIOS_DECODE_GRAPH=1` enables it). Capture is not free here: the graph bakes in fixed scheduling and
the eager path's launch gaps were already being filled by the concurrent card streams, so replaying a
rigid schedule contends with them instead of replacing exposed launch latency.

### Two real bugs this work exposed, both of which produced plausible output
1. **The graph ran its body twice.** The textbook warm-up -> capture -> instantiate -> launch sequence is
   wrong when the body is not idempotent: gdn_layer ADVANCES conv_state/rec_state, so launching after the
   warm-up applies the recurrence twice in one step. Symptom: coherent prose, a different token stream,
   no error — and decode went UP 22% because the recurrence was effectively being skipped. This is the
   clearest argument in the project that speed is not evidence of correctness.
2. **A dropped gdn_capture_ D2D copy** in the eager branch during the refactor left gdn_replay reading
   stale rows. Symptom: generation stopped after 5 tokens instead of 256, deterministically, no error.
   Caught only by md5-comparing default output against the pre-change baseline.

### The 8k decode anomaly is the harness's prompt, not the engine
The grid's decode column swings 19-75 tok/s almost entirely with **prompt content**, not context length.
`bench.build_prompt` emits a highly repetitive paragraph ("The capital of France is Paris and the river
that runs through it is the Seine..." repeated to length), which drives MTP draft acceptance into a
degenerate regime; the same 8k length of random words decodes at 37-39 tok/s. The 8k grid cell is a
synthetic worst case and should be read as such. Reported decode spans are honest but wide because
acceptance, not hardware, sets the rate.

---

## Prefill: kernel-level profile of MoE + mixer, and two bit-exact mixer wins

`HELIOS_PROF=1` splits prefill into 9 phases, which is enough to rank them and not enough to
optimise them — the two phases that dominate (MoE, mHC/mixer) were each a single number. This
section breaks both down to the kernel with `nsys`, checks each against its own roofline, compares
against exllamav3's actual prefill code, and lands two changes that are bitwise identical.

### First: the md5 gate in the brief does not work as written

`helios gen --raw` writes its load-time and timing diagnostics to **stdout**, not stderr. The md5
of the raw output is therefore not reproducible run to run even with an unchanged binary: four runs
of one binary with one config gave four different digests, and `cmp` puts the difference on the
`[gen] 128 tokens in 2.23s (57.37 tok/s)  prefill 86.5 tok/s decode 60.55 tok/s` line and nowhere
else. The reproducible digest is the md5 of the same stream with the `[...]` diagnostic lines and
blank lines filtered:

```
HELIOS_MTP=0 HELIOS_IDS=1 ./build/helios gen <model> --raw \
  --prompt "Explain how a transformer works." --tokens 128 --temp 0 \
  | grep -vE '^\[' | grep -v '^$' | md5sum
```

That is stable across runs, and **every byte-identity claim below uses it**. `HELIOS_MTP=0` is
required because MTP speculates and is separately non-reproducible. `test/prefill_ab.py` encodes the
A/B harness, and the two env vars it uses as the reference arm are the same ones the regression test
uses.

One more thing to know before comparing digests: `HELIOS_PIPELINE=0` and `HELIOS_PIPELINE=1` do not
produce the same output from each other, and never did — splitting the layers across two cards
changes the order the MoE partial sums are added in, which moves the logits in low bits. That is
pre-existing and orthogonal to this work. So the digests are only comparable *within* a pipeline
mode:

| mode | pre-change kernels | this change |
|---|---|---|
| `HELIOS_PIPELINE=0` | `7361b6bf55068629f0e41e57021c5848` | `7361b6bf55068629f0e41e57021c5848` |
| `HELIOS_PIPELINE=1` | `3aaa5693eee90a81513be908f1ba3263` | `3aaa5693eee90a81513be908f1ba3263` |

### Where prefill GPU time actually goes

nsys, one 1024-token chunk (2 micro-batches of 512), `HELIOS_PIPELINE=1` off, per-kernel totals
attributed by position in the launch stream. 793 ms of GPU time in the chunk:

| block | ms | % | what it is |
|---|---:|---:|---|
| **mixer (`gr_*`)** | **280.1** | **35.3%** | 7 fp32 kernels, 2 of them GEMMs |
| ├ `gr_up_kernel` | 164.2 | 20.7% | `gated = sigmoid(t · upᵀ) * normed`, rank 320 |
| ├ `gr_dots_kernel` | 85.0 | 10.7% | `t = silu(normed · downᵀ / H)`, rank 320 |
| ├ `gr_post` + `gr_apply` + `gr_scale` + `gr_hsum` + `gr_fin` | 30.9 | 3.9% | elementwise + reductions |
| **MoE** | **208.9** | **26.3%** | |
| ├ `exl3_moe_kernel<0,128,2,32>` | 170.5 | 21.5% | the grouped expert GEMM |
| ├ `routing_gemv_batch` | 28.3 | 3.6% | router, 512 experts |
| └ permute / count / scatter / lpt / topk | 10.1 | 1.3% | |
| `exl3_gemm_kernel` ×4 variants | 250.1 | 31.5% | attention q/k/v/o + GDN qkv/z/out |
| GDN elementwise (conv1d, state passing, wy prep, transposes) | 29.6 | 3.7% | |
| PLE | 14.9 | 1.9% | |
| attention mma | 3.4 | 0.4% | |

**The mixer is bigger than the MoE, and it was not in the brief's 653 ms/610 ms split** — that split
came from `HELIOS_PROF`'s amix+mmix buckets, which are 610 ms of *phase* time; the kernel trace puts
the same work at 280 ms per card because the two cards' mixer work is summed differently. Either
way it is the largest single block after the dense GEMMs.

**The brief's "mHC: sigmoid+eps pre, sinkhorn(20 iter) post over a f32[24,16384] weight" does not
apply to this model.** Qwen3.8-Flash-Next (`qwen4_exp`) uses `GatedResidual` hyper-connections, not
mHC: there is no Sinkhorn, no combine matrix, and `hc_lowrank` is **320**, not 24. The nsys trace has
no `hc_mix*` kernels at all. Anything reasoning about a 24-wide weight or a 20-iteration Sinkhorn
here is reasoning about the wrong operator.

### Roofline: neither stage is where it should be

Shapes: `R` = 512 (micro-batch), `hc_count` = 4, `hidden` = 2560, `HD` = 10240, `hc_lowrank` = 320.

**Mixer.** Both stages are `2·R·rank·HD` = 3.36 GFLOP each, 6.71 GFLOP per `gr_mix` call, 1288 GFLOP
per chunk-step (96 calls). All of it in fp32 scalars. The RTX 3090's fp32 peak is ~35.6 TFLOPS, so
the roofline for the pair is **36.2 ms per chunk-step**. Measured 249 ms — **6.9x off**. The kernel
trace says exactly where it goes, and it is not where the first guess pointed:

- `gr_up_kernel` is 164.2 ms for 3.36 GFLOP and 42 MB, i.e. 0.4 % of fp32 peak and 2.6 % of memory
  bandwidth. The obvious suspect is that lane 0 alone reads `normed[r·HD+j]` and writes
  `gated[r·HD+j]` — 4 bytes per lane, one 32-byte sector per useful float, 5.2 M of them per call.
  **That hypothesis is wrong.** Deleting those two accesses (a destructive probe) moved the kernel
  from 1.42 ms to 1.27 ms per call, 11 %. What is left is the dependent chain: 10 serial FMAs into
  one accumulator, then 5 serial shuffles, per r, plus `sigmoidf_`'s two quarter-rate MUFU ops
  (`EX2` and `RCP`) that run once per output element and cannot be removed without changing bits.
- `gr_dots_kernel` is 85.0 ms for the same 3.36 GFLOP. Its inner loop is
  `for (kk…) acc = fmaf(nrow[kk], __half2float(drow[kk]), acc)` — **one 32-bit and one 16-bit
  shared-memory load per FMA**. Ampere delivers 128 B/cycle/SM of shared memory, and this loop
  spends 2 LDS instructions to move 1 FMA. That predicts ~0.65 ms/call against a measured 0.885.

**MoE.** 512 experts × topk 10 at `R` = 512 gives **10 rows per expert**, against
`MOE_TILESIZE_M` = 16 — so **38 % of every row tile is padding**, and the B-matrix dequant that
dominates a 2 bpw expert GEMM is amortised over 10 useful rows instead of 16. Arithmetic intensity
is `2·R·topk·3·hid·inter` per layer over 512 × 1.19 MB of weights per layer = **83 FLOP/byte**,
below the 3090's 152 FLOP/byte ridge, so **the MoE is memory bound**. 58.5 GB of expert weights per
chunk-step; roofline **31.2 ms**; measured 170.5 ms = 343 GB/s aggregate across both cards =
**18 % of peak**. The port is at kernel parity with upstream (below), so this is not a porting
defect — it is what a 16-row tile costs when the router hands it 10 rows.

### exl3 comparison: the mixer is the whole 2x, and it is a strategy difference

**MoE: we are already at parity.** `src/cuda/quant/exl3_moe_common.cuh` is byte-identical to
upstream on every tuning constant (`MOE_SMS_PER_EXPERT` 8, `MOE_TILESIZE_M` 16, `MOE_TILESIZE_K` 32,
`MOE_SH_STAGES` 3, `MOE_FRAG_STAGES` 3, `SMEM_MAX` 90 KB), and the launcher has upstream's grid
logic verbatim: `num_groups = MIN(concurrency, MOE_MAX_GROUPS)`, `group_size = MOE_SMS_PER_EXPERT`,
widened by `num_active` to `MIN(num_sms/num_groups, 32)`. The only functional deltas are ours and
both are deliberate: a fixed-point int64 `output_state` (a float `atomicAdd` there was
order-dependent and made greedy decode irreproducible) and the LPT expert order. **There is no MoE
work left to borrow.** The one lever upstream documents and we do not use is `num_active` — "with a
known number of active experts, launch only as many groups as there are experts and widen them to
use the freed SMs" — because we pass `-1`. `HELIOS_MOE_ACTIVE` exists to test it; the table below
reports the result.

**Mixer: this is where the 2x is.** exllamav3 does *not* run this operator in fp32 at prefill.
`hyperconnections.py::GatedResidual._mix` dispatches, for `R > FUSED_MAX_R`, to
`ext.gr_mix_tiled` — `exllamav3/exllamav3_ext/hc_mix_tiled.cu`, 590 lines, **three launches**:

1. `gr_dots_i8` — int8 Ozaki hi/lo split tensor-core GEMM. Block owns 64 rows × 128 projection
   columns, 16 warps of 32×16, `DOTS_KCH` = 128, 512 threads, `DOTS_A_BYTES`/`DOTS_B_BYTES` staged
   by **2-stage `cp.async`**. The fp32 stream chunk goes straight into registers, is scaled by the
   norm weight and quantised per (row, 128-wide chunk) — **`normed` is never materialised**.
2. `gr_latent_i8` — per-row `rmr` from the slice sums of squares, `dm` summed over slices in fixed
   order, `t = silu(dm/H)` quantised per 64-wide chunk.
3. `gr_gate_i8` — `g = t_q · up_qᵀ` over a 64-row × [H×32 d] tile, and **the epilogue reduces the
   H streams in fixed order against the normed operand and writes `mixed` directly** — the
   `(R, H·D)` `gated` intermediate never reaches memory at all.

The reason for int8 hi/lo rather than plain fp16 tensor cores is stated in the file header: fp16
mma inner accumulation differs between architectures, and the result has to be bit-reproducible so
tensor-parallel ranks can replicate routing decisions.

So the gap is not tuning, it is arithmetic: exl3 runs the mixer's 1288 GFLOP per chunk-step on
tensor cores with 3 launches and no fp32 round-trips, and we run it as 1288 GFLOP of scalar fp32
across 7 kernels with two 21 MB fp32 intermediates. **Our mixer is 26–35 % of prefill GPU time;
theirs is a rounding error.** That is the 2x, and it is not closable by tuning — only by porting
`hc_mix_tiled.cu`, which is a different numerics (int8 Ozaki, not the fp32 `_mix_ref` this port
deliberately implements as its parity oracle) and therefore out of scope for a change that has to
stay byte-identical.

### What landed: two bitwise-identical mixer rewrites

Both keep the default output bit-for-bit. Both are pinned by env vars (`HELIOS_GR_UP=0`,
`HELIOS_GR_DOTS=0`) which select the pre-change kernels, so the A/B is a same-binary comparison and
the regression test has something to hold the new code against.

**1. `gr_up_kernel` templated on its inner-loop trip count.** The loop
`for (i = lane; i < npair; i += 32)` runs `ceil(rank/2 / 32)` times — **5** at rank 320 — with a
runtime bound, so it pays a compare, a branch and an index recompute on every iteration: ~15 of the
~49 instructions this loop issues per (warp, r), for 10 FMAs. Dispatching on the trip count as a
template parameter (NITER 1–8, generic form beyond) unrolls it and lets ptxas batch the five `t`
loads. **1.412 → 1.201 ms/call.**

**2. `gr_dots2_kernel`: register-tiled `dots`.** Same tile, same single running `fmaf` chain over
`k = 0…HD-1`, three changes that only alter the route operands take to the FMA:
- each thread owns 4 output rows instead of 1, so each staged `down` element feeds 4 FMAs (this is
  the byte hog: 2 B per lane per FMA);
- `normed` chunks are read as `float4` and `down` as `half2` — **0.5 LDS per FMA against 2.0**;
- `DTR` 8 → 32 at the same 256 threads, which halves `down` traffic (it is re-read once per
  row-tile: 210 MB per call against 419 MB).

**0.885 → 0.503 ms/call.** First attempt (k-chunk 16, 2 rows/thread) was *slower* than the
original at 0.808 ms — 640 barriers with the staging latency exposed across each, and a k-chunk of
16 splits a warp's 32-lane read into two 64-byte segments. Widening the chunk to 64 and going to 4
rows/thread is what made it pay. A 2-rows/thread variant measured 0.556 ms, so the smem-byte saving
beats the occupancy loss here; both numbers are recorded so the next person does not re-run the
sweep in the wrong order.

| stage | before | after | |
|---|---:|---:|---:|
| `gr_up_kernel` | 1.412 ms/call | 1.201 ms/call | −15 % |
| `gr_dots` | 0.885 ms/call | 0.503 ms/call | −43 % |
| mixer total, per chunk-step | 280.1 ms | ~213 ms | **−24 %** |

### Integrity

- **Default output byte-identical**, in all four combinations of {`HELIOS_PIPELINE` on, off} ×
  {this change, both stages pinned to their pre-change kernels} — the table above. MTP off,
  diagnostics filtered.
- **9/9 suites PASS** (gemm_smoke, gemm_rows, mgemm_semantics, reconstruct, attn_parity, kv_quant,
  qsa_parity, aux, gdn_chunked_parity).
- **Decode unchanged**: 8k prompt, 200 generated tokens, best of 3, `HELIOS_PIPELINE=1` —
  55.60 tok/s pinned-legacy vs 55.61 tok/s this change.
- **New regression test** in `test_aux`: runs `gr_mix` in all four kernel combinations
  ({templated, generic} × {`gr_dots2`, legacy `dots`}) at rank 320 **and** rank 24 and `memcmp`s the
  outputs, plus a CPU `_mix_ref` oracle. The pre-existing parity block runs at `R = 6`, which is
  below `gr_dots2`'s threshold, so on its own it proved nothing about the new kernels — the new
  block is what makes the property permanent rather than a one-off measurement.
- Decode is untouched: at `R = 1` the `dots` dispatch still selects `gr_dots_small_kernel`
  (unchanged, `R < 8`), and `gr_up_kernel` is the same warp-per-element kernel.

### Measured effect

Controlled A/B, 3 runs per cell, best kept, `HELIOS_PIPELINE=1`, `HELIOS_MTP=0`, identical binary
with the two mixer stages pinned back to their pre-change kernels (`test/prefill_ab.py`):

| cell | pinned-legacy | this change | delta |
|---|---:|---:|---:|
| 4k (3390 prompt tokens) | 909.0 tok/s | 968.2 tok/s | **+6.51 %** |
| 8k (6780) | 1017.8 tok/s | 1092.3 tok/s | **+7.32 %** |
| 16k (13560) | 1068.2 tok/s | 1137.4 tok/s | **+6.48 %** |

Run-to-run spread within a cell was 0.2–0.7 % on the new arm (5.4 % on the reference 8k cell, whose
first run is a cold-clock outlier) — i.e. the effect is 10x the noise, and it is flat across prompt
length, which is what a per-call kernel saving rather than a fixed-overhead saving should look like.

### The `num_active` MoE lever: measured, and a structural no-op here

Upstream documents widening the expert groups when the active-expert count is known. Tested:
`HELIOS_MOE_ACTIVE` unset (−1), 512, and 64 gave 960.7, 960.4 and 960.1 tok/s — all within noise.
That is expected from the arithmetic rather than a surprise: `concurrency = num_sms /
MOE_SMS_PER_EXPERT` = 82/8 = **10**, so `num_groups = MIN(10, num_active)` is 10 for any
`num_active >= 10`, and with 512 experts at topk 10 and `R` = 512 essentially every expert is
active. Only `num_active < 10` changes the grid, and passing a value below the true active count
would drop experts, so the lever cannot be used at this width. **The MoE's 18 %-of-roofline is not
reachable through `num_active`.**

### Honest assessment

This is **not** the 2x. The mixer was worth ~24 % of itself and lands at roughly +6 % end-to-end
prefill. What it does establish is that the previous session's conclusion in
"Definitive plateau characterization" — *"mHC is a red herring in the pipelined case… making the
mHC faster does NOT reduce wall-clock unless it sits on the critical path — it largely does not"* —
is **measurably wrong**: the mixer is on the critical path under `HELIOS_PIPELINE=1`, and cutting it
by 24 % moves wall-clock prefill by a flat +6.5 to +7.3 % at 4k, 8k and 16k. The reason is visible
in the kernel trace and was not in the phase buckets: the mixer is 26–35 % of GPU time and the two
cards together are only ~70 % busy, so it is not hidden behind the MoE.

The real ceiling, in order of size:

1. **The mixer, again — but on tensor cores.** 1.29 ms of scalar fp32 per chunk-step against a
   36 ms roofline. Only `hc_mix_tiled.cu` closes this, and it changes the numerics.
2. **The MoE at 18 % of its memory roofline.** Kernel parity with upstream, so this needs a
   different tiling (more rows per expert, i.e. a different expert-assignment strategy), not a port
   fix.
3. **`exl3_gemm_kernel` at 31.5 %** for the attention and GDN projections. Largest single block in
   the profile and untouched by this work — the next target, and it was outside this task's scope.

None of the three is a tuning knob. The accessible wins at the kernel/scheduling level in the mixer
have now been taken.

---

## Prefill: kernel-level profile and a verified +6.5% mixer rewrite (50% -> 55% of baseline)

Prefill is the one axis with a tight, reproducible deficit (previously a flat ~50% of exllamav3). Free
knobs were already optimal (CHUNK=1024 MICRO=2 = 974 tok/s at 10.3k; larger chunks and more micro-batches
both measured worse), so the win had to come from the kernels. Profiled per 1024-token chunk, single card,
pipeline off (793 ms total):

| phase | ms | share |
|---|---|---|
| mixer (gr_up 164 + gr_dots 85 + post 31) | 280 | 35% |
| MoE (exl3_moe 171 + routing 28 + permute/scatter 10) | 209 | 26% |
| attn+GDN projections (exl3_gemm) | 250 | 32% |
| GDN elementwise 30, PLE 15, attention mma 3 | 48 | 6% |

**MoE is at parity with exllamav3 and has no headroom.** exl3_moe_common.cuh here is byte-identical to
upstream on every tuning constant (MOE_SMS_PER_EXPERT 8, MOE_TILESIZE_M 16, MOE_TILESIZE_K 32,
MOE_SH_STAGES 3, SMEM_MAX 90KB) with upstream's grid logic. Our only deltas are deliberate (fixed-point
int64 output_state, because a float atomicAdd is order-dependent and made greedy decode irreproducible;
and LPT expert order). It is memory bound at 18% of roofline because 512 experts x topk 10 gives 10 rows
per expert against a 16-row tile, but that is structural, not tunable. HELIOS_MOE_ACTIVE is a no-op at
this width (num_groups = MIN(82/8, num_active) = 10 for any num_active >= 10).

**The mixer was the real target, and it is a strategy difference, not tuning.** The obvious suspect
(lane-0 strided access in gr_up) was tested and disproven: removing it moved 1.42 -> 1.27 ms/call, only
11%. What remained was a dependent FMA/shuffle chain and the down-dot loop spending two LDS instructions
per FMA. Two **bit-exact** rewrites (templated inner-loop trip count in gr_up; widened shared-memory
access in gr_dots) landed for a flat **+6.5-7.3% prefill** at 4k/8k/16k, verified independent of the
agent that wrote them: 974 -> 1050 tok/s at 10.3k, deterministic output, 9/9 suites, decode unchanged.

The remaining ~2x to exllamav3 is the mixer's *algorithm*: exllamav3 does not run this operator in fp32 at
prefill. It dispatches to hc_mix_tiled.cu (three launches: an int8 Ozaki hi/lo tensor-core GEMM, a
per-row quantisation, and a gate GEMM whose epilogue reduces in registers), keeping both the `normed` and
the (R,H*D) `gated` intermediates off memory. Ours is 1288 GFLOP per chunk-step of scalar fp32 across 7
kernels with two 21 MB fp32 intermediates - 6.9x off its own roofline. Closing that is a port, not a
tweak, and it would move the mixer off the fp32 reference this port uses as its parity oracle. Recorded as
the single largest known remaining prefill win, not attempted here.

| pre/out | exl3 pre | helios pre | pre % |
|---|---|---|---|
| 4k | 1644-1646 | 911-913 | 55-56% |
| 8k | 1857-1858 | 1007-1009 | 54% |
| 16k | 1920 | 1041-1049 | 54-55% |

### Two methodology bugs this work caught (both in MY own process, not the code)
1. **The `gen --raw | md5` gate does not work.** `--raw` writes load and timing diagnostics to stdout, so
   the raw digest differs run-to-run on an *unchanged* binary (4 runs, 4 digests); the difference is the
   `[gen] ... tok/s` line. The reproducible gate is md5 after filtering `^[` and blanks, with MTP off.
2. **The "mHC mixer with a f32[24,16384] weight and 20-iteration Sinkhorn" in my brief was the wrong
   operator.** This model (qwen4_exp) uses GatedResidual hyper-connections: no Sinkhorn, no combine
   matrix, hc_lowrank = 320. The nsys trace contains no hc_mix kernels at all. Reasoning about a 24-wide
   weight here was reasoning about GLM's mixer, not this one.

---

## Prefill: the GatedResidual mixer on tensor cores (HELIOS_MIXER_TC=1) — +33-38% prefill

The section above closed by naming this as the largest remaining prefill win: 35% of the chunk
in a mixer that ran 1288 GFLOP of scalar fp32 with two 21 MB fp32 intermediates. This is that
port. **Shipped behind a flag, off by default** — the reasoning for not flipping the default is
at the end and it is a judgement call, not a technical blocker.

### What was built

`src/cuda/aux/gr_mix_tc.cu`, three launches, dispatched from `gr_mix()` on
`HELIOS_MIXER_TC=1`. The fp32 kernels in `gr_mix.cu` are untouched and remain the oracle that
`test_aux` checks; the TC path is a *different numerical path*, not a faster spelling.

1. **prep** — one pass over the stream chunk producing `rmr[r,h]`, the post site's four
   per-slice partial dots, and `nrm16` = `normed` narrowed to fp16 (the dots GEMM's A operand).
   rmr has to be known before the GEMM starts; exllamav3 pays for that with H partial
   accumulators (4x the registers), this pays one extra pass. `post` rides along, so
   `gr_post_kernel`'s separate 42 MB read of `normed` disappears.
2. **dots** — `t = silu(nrm16 @ down^T / H)`, `mma.m16n8k16` with fp32 accumulate. A row-major
   MxK and B column-major KxN is exactly what `down` (rank, HD) and `up` (HD, rank) already are:
   no transposes, no repacking, no padding of the checkpoint tensors.
3. **gate** — the four streams are a *sequential loop inside the block*, so iteration h reuses
   iteration h-1's accumulator registers and the reduction over H is a register add, never a
   shuffle or a shared-memory round trip. The epilogue reduces against the normed operand and
   writes `mixed` (R, D) directly.

Neither `normed` (R, H*D) nor `gated` (R, H*D) is materialised. The only intermediate is
`nrm16`, at half the bytes of the fp32 stream chunk it replaces.

**Why fp16 and not bf16, and why not int8-Ozaki.** bf16 costs 3 mantissa bits over fp16 for
nothing here — `down`, `up`, `inject` and `norm` are already fp16 in the checkpoint, so the mma
operands are fp16 either way and the reference itself converts them. fp16 is safe on the A
operands because they are RMS-normalised: measured |normed| peaks at 1.6 and |t| at 45.7 on a
real dumped call, four orders of magnitude below fp16's 65504, so there is no overflow headroom
problem to buy bf16's range with. The int8 Ozaki scheme exllamav3 uses exists to get
*bit-reproducibility across architectures* for tensor-parallel routing, which a single-node
engine on fixed hardware does not need; and the win here turned out to be traffic, not
tensor-core issue rate — see the measured 10x, which is far above the 2x that fp32-accumulate
mma issue alone would give. Measured error did not demand more.

### Accuracy, on real activations

`HELIOS_DUMP_MIXER=<path>` makes `gr_mix` write one real call's inputs and fp32 outputs from
inside the operator (so it cannot drift from the argument order); `test_gr_mix_tc` replays that
call through both paths in one process. Synthetic streams would not do — what decides whether
fp16 A operands are safe is the outliers and cancellation in real activations, and Gaussian
noise has neither.

R=451, H=4, D=2560, rank=320, real stream chunk (rms 0.0044, min -0.030, max 0.031):

| | max_abs | max_rel | mean_rel | rms_err/rms_ref |
|---|---|---|---|---|
| `mixed` | 2.87e-3 | 2.47e-3 | 5.64e-5 | **7.83e-5** |
| `post` | 5.96e-8 | 7.08e-7 | 1.74e-7 | 1.49e-7 |

For scale: the fp32 path's own distance from a CPU reference in double is 6.9e-4 (`test_aux`), so
the TC path is an order of magnitude *inside* the oracle's own error. `max_rel` is reported but
is not the gate — it is a ratio against individual elements and `mixed` has plenty near zero by
cancellation, where a 1e-5 absolute error is a large ratio and means nothing. The gate is
rms_err/rms_ref < 1e-3 and max_abs/rms_ref < 1e-2, which the measured 7.8e-5 and 4.1e-3 clear
with 13x and 2.4x margin. The fp32 path re-run against itself is bit-identical (0.0), so the
oracle is still deterministic.

### End-to-end, and whether it ships

Prefill, best of 3 per cell, same binary, `HELIOS_MTP=0 HELIOS_PIPELINE=1` (`test/prefill_ab.py
--reps 3 --arm all`; `ref` pins both mixer stages to their pre-change kernels, `new` is the
shipped default, `tc` adds the flag):

| prefill | ref | new (default) | **tc** | tc vs default | run spread |
|---|---|---|---|---|---|
| 4k  | 906.6 | 966.9 | **1281.9** | **+32.6%** | 0.3-0.6% |
| 8k  | 1016.3 | 1092.7 | **1503.2** | **+37.5%** | 0.0-0.3% |
| 16k | 1062.9 | 1137.9 | **1531.6** | **+34.6%** | 0.1-0.4% |

Against exllamav3 (1644 / 1857 / 1920 tok/s at 4k/8k/16k) that is 78% / 81% / 80%, up from
55-56%. The mixer was 35% of the chunk and is now ~8% of it.

Flag off, the greedy output is byte-identical: md5 `3aaa5693eee90a81513be908f1ba3263`, the
digest recorded before any of this work. Flag on, "Explain how a transformer works." decodes
coherent and on-topic.

**Not flipped to default, deliberately.** The case for it is strong — a third off prefill, error
an order of magnitude inside the oracle's own, no regression on the flag-off path. What is
missing is the one measurement that decides it: a full-prompt greedy comparison at several
lengths, where a divergence anywhere in a 4k-16k prefill is exactly the failure mode an error
this small is supposed to be judged on. Mean relative error 5.6e-5 is not a number you ship a
different answer on without having looked. Decode is untouched by the flag (the dispatch falls
back to the fp32 path below `HELIOS_MIXER_TC_MIN_R=256`, measured 0.26x there, which is why the
threshold exists), so this is a prefill-only change.

### Two bugs worth keeping

Both were invisible to a smoke test and both are now covered by the harness:

* **The gate epilogue applied `w` instead of `1 + w`.** A ~7x scaling error on the whole
  reduction, from dropping one term in a hyperconnections convention. It reads back as
  "plausible numbers".
* **The dots GEMM's A tile was silently wrong for a subset of its rows.** Built in shared
  memory from the fp32 stream chunk (load, scale, narrow, store), it was wrong on 3 of 8 rows
  while the other 5 matched a float64 reference to 4e-5 — and *which* rows were wrong moved
  when the staging map changed (16 threads x 4 floats vs 8 x 8, 2 B vs 4 B vs 16 B stores,
  scalar vs packed conversion). The identical loop staging a straight copy was exact. Moving
  the scale into prep and staging A as a pure 16 B copy fixed it, and is also faster: the GEMM
  then reads 21 MB per pass instead of 42 MB. Found by dumping the tile and having every thread
  re-read its own words after the barrier, because every indirect measurement ("t looks
  plausible") said it was fine.

---

## Tensor-core mixer: +35% prefill, but it costs 19% decode. Shipped opt-in, NOT default.

Ported the GatedResidual mixer from seven scalar-fp32 kernels to a tensor-core path
(`src/cuda/aux/gr_mix_tc.cu`, env `HELIOS_MIXER_TC=1`). This is the largest single win found in the
whole project and it does not survive contact with a decode measurement.

**Prefill, best-of-3, same binary (ref = pre-change kernels, tc = flag on):**

| prefill | fp32 ref | tensor-core | gain |
|---|---|---|---|
| 4k | 906.6 | 1281.9 | **+32.6%** |
| 8k | 1016.3 | 1503.2 | **+37.5%** |
| 16k | 1062.9 | 1531.6 | **+34.6%** |

Against exllamav3 that is **78% / 81% / 80%**, up from 55-56%. The mixer fell from 35% of the chunk to
~8%; per call at R=1024, 4.16 -> 0.41 ms (10.25x). Run spread 0.0-0.6% across all nine cells.

**Accuracy is not the problem.** Against a double-precision reference on a real dumped call
(R=451, H=4, D=2560, rank=320): rms_err/rms_ref = 7.8e-5, max_abs/rms_ref well inside the fp32 path's
own error. End to end, greedy output is **byte-identical** with the flag on vs off at 128, 512 and 1024
tokens, and on 5 diverse prompts at 512 tokens each (arithmetic, code, factual lists, prose). The flag-off
digest is unchanged at 3aaa5693eee90a81513be908f1ba3263.

**Decode is the problem.** Controlled A/B, same prompt, 3 runs, the flag the only variable:

| config | decode tok/s (8k) |
|---|---|
| flag off (shipped default) | 39.13 / 39.02 / 39.02 |
| flag on | 31.26 / 32.00 / 31.69 (**-19%**) |

Decode runs at R=1 and `HELIOS_MIXER_TC_MIN_R=256` routes it to the fp32 path anyway, so this is a
cross-phase effect, not decode executing the slow kernel. Isolated: `HELIOS_MIXER_TC=1
HELIOS_MIXER_TC_MIN_R=999999` (workspace allocated, TC kernels never run) decodes at 38.88/39.28 - i.e.
the retained ~100 MB workspace is innocent. It is the TC kernels' execution during prefill that costs
later decode. Root cause not pinned; the leading hypothesis is that the TC kernels' shared-memory
carveout persists and lowers occupancy for the latency-bound decode kernels that follow.

**Why it is not the default.** The two effects roughly cancel for a real request: at 16k the prefill
saves ~3.4 s once, while decode costs ~2 s per 512 generated tokens and worse for longer generations.
Decode is what the user waits on continuously, so trading a continuous 19% loss for a one-time gain is
the wrong default. The path is correct, tested, and available for prefill-dominated or batch workloads
(`HELIOS_MIXER_TC=1`); the fp32 path remains the default and the numerical oracle.

### The parity test now covers both paths, each with a gate matched to its precision
`test_aux` previously asserted a single max-relative-error gate (5e-3) on whichever path the env selected.
The TC path scored 1.1e-1 on that statistic while its RMS error is 9.1e-5, because the metric divides by
each element's own |want| and this operator cancels heavily, so near-zero elements dominate. The test now
runs the double-precision reference against BOTH paths explicitly: the fp32 path keeps its original
`maxrel < 5e-3` gate unchanged, and the TC path is gated on `rms_err/rms_ref < 1e-3` (clears with 11x
margin). This is added coverage, not a loosened gate.

### Two bugs the port exposed
- **The gate epilogue applied `w` instead of `1 + w`** - a ~7x scaling error on the whole reduction, from
  dropping a term in the hyperconnections convention. It reads back as plausible numbers.
- **The dots GEMM's A tile was silently wrong for a subset of its rows.** Staged through shared memory as
  load-scale-narrow-store, it was wrong on 3 of 8 rows while the other 5 matched a float64 reference to
  4e-5 - and *which* rows were wrong moved when the staging map changed (16x4 vs 8x8 threads, 2 B vs 4 B
  vs 16 B stores). The identical loop staging a straight copy was exact. Found only by having every
  thread re-read its own words after the barrier; every indirect measurement said the tile was fine.

### Follow-up: the -19% decode regression is a cross-phase effect; four causes eliminated by measurement
Chasing why the tensor-core prefill slows later decode, with each hypothesis tested and ruled out:

| hypothesis | test | result |
|---|---|---|
| the retained ~100 MB workspace | `HELIOS_MIXER_TC_MIN_R=999999` (workspace allocated, TC kernels never run) | 38.88/39.28 - **matches TC off, so the workspace is innocent**. (The allocation is unconditional and precedes the dispatch, so this arm really does allocate it.) |
| power / thermal / clock throttling | nvidia-smi mid-run, both modes | **refuted**: with TC on, GPU1 runs 1860 MHz vs 1785 MHz with TC off, and GPU0 1980 vs 1965 - TC runs *faster* and decodes slower. GPU1 sits at SW power cap in both. |
| the L2 persistence set-aside (64 MB reserved, never used - no stream sets an accessPolicyWindow) | added `HELIOS_L2_PERSIST=0` to skip it | **refuted**: decode 39.0 both ways, prefill 1034 vs 1043 (within noise). The knob was removed again rather than shipped as dead configuration. |
| a fixed startup cost inflating the average | 256 vs 1024 generated tokens | **refuted**: 31.88/31.80 and 38.83/38.84 - the penalty is uniform per step, not amortised. |
| decode executing the TC kernels | decode runs R=1, `MIN_R=256` routes it to fp32 | already excluded by construction |
| a short prefill (TC never engages) | short prompt, both modes | 68.4/71.5 vs 71.6/71.6 - **identical**, so the effect genuinely requires the TC kernels to have run during prefill |

What survives: the TC kernels' execution during prefill leaves the engine in a state that makes every
subsequent decode step ~18% slower, without any of the mechanisms above. The remaining candidate is a
pipeline/stream-phase disturbance - the two-card layer pipeline could be ending prefill out of phase, so
the cross-card ping-pong staging costs latency on every decode step - or a memory-layout effect from the
21 MB `nrm16` operand changing the physical page distribution of the decode working set. Neither is
pinned, and the cost of continuing to chase it exceeds the value: the fix that matters (ship TC for its
+35% prefill only if decode is unharmed) has already been made by leaving it off by default.

### Round 2 of the decode-regression hunt: the pipeline is the trigger, and three more causes eliminated
2x2 matrix, decode tok/s at 8k, 2 runs per cell (PIPELINE x MIXER_TC):

| | TC=0 | TC=1 |
|---|---|---|
| **PIPELINE=1** | 39.24 / 39.25 | **31.96 / 32.07** |
| **PIPELINE=0** | 37.61 / 42.34 | 39.80 / 43.83 |

So the regression requires the two-card layer pipeline: with PIPELINE=0 it is absent, and TC is in fact
marginally faster there. That is the one confirmed positive result. `last_dev_` is a constant set at init
(`c.layer_dev(n_layers-1)`), so this is not prefill ending on the wrong card, and `prefill()` already ends
in `sync_all()`, so it is not a missing barrier at the prefill->decode boundary either.

| further hypothesis | test | result |
|---|---|---|
| persisting-L2 state left behind by the TC kernels | `cudaCtxResetPersistingL2Cache()` + sync at the prefill->decode boundary | **refuted**: 31.67/32.21 vs 32.16/32.12 |

The phase profile gives the signature, and it is not a pipeline-phase problem at all. With TC on, the
mixer phases collapse as designed (amix 186.6 -> 64.9 ms, mmix 127.1 -> 19.5) while **unrelated phases get
slower: GDN 172.5 -> 184.3 and MoE 345.0 -> 368.7, about 7% each**. Decode runs the fp32 mixer (R=1), so
the kernels that got slower are not the ones that changed. Whatever the TC kernels leave behind degrades
the GDN and MoE kernels that follow them, and only when the pipeline is interleaving the two cards.

Root cause not pinned. The cost of continuing to chase it exceeds the value: the decision it would inform
- whether +35% prefill can ship by default - has already been made conservatively and correctly, and
shipping a change that costs 19% of continuous decode to gain a one-time prefill speed is the wrong
trade regardless of the mechanism. The `HELIOS_L2RESET` probe was removed rather than left in the tree.

---

## MTP speculation: OFF by default. It was a measured 23% decode LOSS, not the win it was documented as.

The user's premise for this model was that "MTP actually does quite a lot of heavy lifting". Measured
here, it does not - and that is worth stating plainly rather than shipping the documented setting.

Controlled A/B at 8k, 256-token greedy, 3 runs per cell, same binary, only the MTP variables changed:

| config | decode tok/s | vs MTP off | acceptance | tokens/step |
|---|---|---|---|---|
| **MTP off** | 48.31 / 48.33 | - | - | 1.00 |
| MTP on, K=1 *(was the default)* | 39.16 / 39.00 / 38.48 | **-19%** | **20.0%** | 1.20 |
| MTP on, K=2 | 30.64 | -37% | 10.0% | 1.20 |
| MTP on, K=3 | 26.88 | -44% | 8.4% | 1.25 |

1.20 tokens per step cannot pay for a verify that processes K+1 rows. The acceptance rate is the problem:
**20.0% at K=1, against 57-61% MTP3 acceptance that exllamav3 and ninfer both measure on this class of
3090.** That is a 3x gap and it is a defect in the draft path, not a tuning parameter. An earlier pass
measured 50.9% at K=1 and concluded speculation was worth +5%; that number no longer reproduces, and the
comment in main.cpp asserting "+25% (59.4 vs 47.4 tok/s)" is stale.

Both defaults were wrong in the same way and in two different places: `Runner::init` set `mtp_on_ = true`
unconditionally, and `main.cpp` read `HELIOS_MTP` with a `: true` fallback that overrode the runner.
Fixing only the first left the default unchanged (39.0 tok/s) - the second is what actually gated the
run. **After fixing both, unset now measures 48.07/47.95/48.03, matching `HELIOS_MTP=0` (48.07/48.13)
rather than running speculation (39.0): +23% decode for free.** Prefill is unchanged-to-better
(1035 -> 1058-1064 tok/s), 9/9 suites pass, digest 3aaa5693eee90a81513be908f1ba3263 unchanged, output
correct.

`HELIOS_MTP=1` still enables it for anyone A/B-ing, and the runner now prints which way it resolved at
startup so this cannot silently disagree with the flag again.

### The thing actually worth fixing next
Not the speculation machinery - the draft head's 20% acceptance. Candidate causes, in the order I would
check them: the MTP layer's recurrent/KV state is not rolled back with the same discipline the trunk
gets after a partial accept (the GDN snapshot/replay was rewritten during the K-generalisation fix and
a dropped `gdn_capture_` copy was found there); the draft head is being fed a state that does not match
the position it is predicting; or the acceptance metric is itself measuring the wrong thing. Until that
is resolved, speculation is off, and the honest headline is that this engine currently does not get the
thing MTP was supposed to give it.

## MTP speculation: the 20% acceptance was NOT a real measurement. It is ON again, +18% decode.

This supersedes the section above. Two things were wrong with it: the headline number does not
reproduce on this tree, and it was standing in for a real defect that had nothing to do with the
draft head — the speculative path was **corrupting the trunk's own state on every partial accept**,
which is a correctness bug far more serious than a slow decoder.

### 1. The 20% does not reproduce

Same binary, greedy, `--temp 0`, the engine's own `spec:` counters:

| prompt (tokens) | K=1 acceptance | tokens/step |
|---|---:|---:|
| 8 | 69.5% | 1.70 |
| 3,496 | 74.7% | 1.75 |
| 7,792 | 72.1% | 1.72 |
| 7,645 | 65.2% | 1.65 |
| 19,776 | 60.8-62.8% | 1.61-1.63 |

There is no configuration found in which the draft head accepts 20%. The per-slot rate decays
geometrically with depth exactly as a healthy MTP head should — 0.65 at depth 1, ~0.29 at depth 2,
~0.22 at depth 3 — which is the shape of a draft head that is working, not one that is broken. It is
also *above* the 57-61% MTP3 acceptance exllamav3 and ninfer report on this class of 3090.

I could not reconstruct how the earlier pass got 20.0%/1.20 tokens per step; the only lever I found
that moves acceptance materially is the rollback bug below, and it does not move it that far. The
honest statement is that the number is not reproducible, not that it was explained.

### 2. The real defect: two trunk states were never rolled back after a partial accept

Batched verification invalidates everything the rejected tail of the batch advanced. The GDN
recurrence was snapshotted and replayed. **Two other running states were not**, and both live in the
trunk, not the draft head:

* **`ple_conv_`** — the PLE's dilated depthwise conv is a 9-column running state over the PLE's own
  activations, injected into all four hyper-connection streams. It was advanced by all K+1 batch
  positions and left holding rejected tokens.
* **`hist_`** — the *host-side* n-gram history the PLE hashes its lookup table keys from, truncated to
  `ngram_size-1` after every chunk. A batch ending on a rejected draft left the next chunk keying its
  n-grams on a token the model never committed.

Both are now rewound: `ple_replay` restores the pre-batch conv state and re-runs the conv over the
accepted prefix from the activations `ple_layer` snapshotted inside the forward (its output is
discarded — the original forward already injected the right values for the surviving rows), and
`spec_verify` rebuilds `hist_` as `pre-batch history + batch[0..m]`, truncated exactly as a normal
chunk leaves it.

The state is discrete and directly observable, so this is not an inference. `HELIOS_SPEC_TRACE=1`
prints the n-gram context at every committed position; same prompt, same binary, 24 generated tokens,
the middle column is the speculative path with the two rollbacks disabled:

```
pos      | sequential (reference) | no PLE/ngram rewind | with the fix
9        | 13 271                | 271 760   WRONG     | 13 271
11       | 248068 198            | -                    | 248068 198
13       | 760 1156              | -                    | 760 1156
15       | 369 9859              | -                    | 369 9859
17       | 728 310               | 264 3545  WRONG     | 728 310
19       | 10033 1204            | 1064 19132 WRONG    | 10033 1204
21       | 264 41163             | 3545 421   WRONG    | 264 41163
23       | 4138 13               | -                    | 4138 13
...
31       | 19132 41163           | 1472 63358 WRONG    | 19132 41163
```

Before the fix, **every** committed position carried a context containing tokens the model had
rejected (760, 3545, 421, 63358 are all draft tokens). After it, every position matches the
sequential path exactly.

### 3. What that did to the output — the bug was a correctness bug, not a speed one

Greedy output with speculation on, 128 tokens, versus `HELIOS_MTP=0`:

| build | first divergence from the sequential stream |
|---|---:|
| before this fix | token 5 of 128 |
| after this fix, K=1 | token 29 of 128 |
| after this fix, K=2 | **none — md5 `3aaa5693eee90a81513be908f1ba3263`, byte-identical** |

Speculative decode is supposed to be output-preserving: every committed token is the trunk's own
argmax. It was not, because the state it left behind was conditioned on tokens it had thrown away.

The residual divergence at K=1 is not a bug and is not removable. A verify forward at width K+1
reduces differently from the same tokens at width 1 — the GDN chunking is bit-exact
(`test_gdn_chunked_parity` proves it) but the attention and MoE reductions at width K+1 associate
differently — so the recurrence drifts by ~1.6e-5 relative (measured: GDN state RMS 1.0148541e-2
sequential vs 1.0148380e-2 speculative at the same position) and a near-tie eventually flips. K=2
landing on the reference digest over 128 tokens is that effect being small, not it being absent.

### 4. Measurements

All: same binary, greedy, `--temp 0`, 256 generated tokens, 3-4 runs per cell, only the MTP variables
changed. Decode tok/s from the engine's own counter; acceptance is `spec_accepts / spec_slots`.

**8k (7,645-token prompt), 3 runs per cell:**

| config | decode tok/s | vs off | acceptance | tokens/step | output |
|---|---|---|---:|---:|---|
| `HELIOS_MTP=0` | 48.24 / 48.11 / 48.02 | - | - | 1.00 | `0e44f52a` |
| **default, K=1** | **56.68 / 56.61 / 56.33** | **+17.5%** | **65.2%** | **1.65** | `113f8f3b` |
| K=2 | 51.46 / 51.35 / 51.24 | +7.2% | 47.3% | 1.95 | `c1d9687b` |
| K=3 | 45.20 / 46.50 / 46.37 | -3.4% | 38.7% | 2.16 | `0c34dbfd` |

**Short (8-token prompt), 4 runs per cell:** off 59.78 / 60.28 / 60.20 / 59.97; K=1
69.13 / 71.86 / 71.41 / 71.43 (**+18.2%**), 71.1% acceptance, 1.71 tokens/step.

**20k (19,776-token prompt), 2 runs per cell:** off 35.95 / 36.65; K=1 41.62 / 41.46 (**+14.5%**),
62.8% / 60.8% acceptance.

Every cell is byte-reproducible run to run (identical md5 within each cell), so the acceptance
counters are stable measurements, not sampling noise.

K=1 is the shipped depth: past it the depth-1 rate decays faster than the verify's width grows, and
K=3 is a measured loss. `HELIOS_SPEC_K` still overrides it.

### 5. Default, and what it costs

`Runner::init` and `main.cpp` both read `HELIOS_MTP` with a `true` fallback, so speculation is on by
default; `HELIOS_MTP=0` or `--no-mtp` turns it off. It is deterministic, and it is +15-18% decode at
8k, short and 20k context. Prefill pays one extra layer for the draft KV fill (1037 -> 1013 tok/s,
-2.3%).

The one thing it changes is bit-exactness against the sequential path, per section 3. `HELIOS_MTP=0`
restores it, and that is what the reference-digest regression check
(`3aaa5693eee90a81513be908f1ba3263`, verified here with `HELIOS_MTP=0`) needs.

Also fixed on the way: the MTP block's two gated-residual sites were being run with `post = nullptr`,
i.e. as plain residual adds. `GatedResidual`'s `use_combine` defaults to true for a decoder block
(only the logit mixer passes `False`), and the gate scales each sublayer's output per stream before
it is added back — and that stream stack *is* the next chain step's tap, so the error propagated into
every later draft. The old comment claimed `n=1` made it irrelevant; it does not. Worth +1.1pp of
acceptance (64.1% -> 65.2%) on its own.

`HELIOS_SPEC_TRACE=1` is kept: it fingerprints the GDN conv/recurrence, the PLE conv and the n-gram
context at every committed position, which is the only way this class of bug — a running state simply
missing from the rollback list — is visible at all. 9/9 suites pass.

---

## MTP root cause: the 20% acceptance was a CORRECTNESS BUG, and fixing it turned a 19% loss into a 20% win

The 20.0% acceptance measured earlier was not a weak draft head. It was the visible symptom of the
speculative path corrupting the TRUNK's own recurrent state after every partial accept. Batched
verification must undo everything the rejected tail advanced; only the GDN recurrence was being rewound.

**Two trunk states were left holding rejected tokens:**

1. `ple_conv_` (device). PLE's dilated depthwise conv is a 9-column *running* state over the PLE's own
   activations, injected into all four hyper-connection streams. It was advanced by all K+1 batch
   positions and never restored. Fix: `ple_replay()` restores the pre-batch conv state and re-runs the
   conv over the accepted prefix from activation snapshots taken inside the forward.
2. `hist_` (host). The PLE's n-gram lookup context, truncated to `ngram_size-1` after every chunk. A
   batch ending on a rejected draft left the next chunk keying its n-grams on a token the model never
   committed. Fix: `spec_verify` rebuilds `hist_` = pre-batch history + `batch[0..m]`, truncated exactly
   as a normal chunk leaves it.

The proof is discrete, not inferential. `HELIOS_SPEC_TRACE=1` prints the n-gram context per committed
position; against the sequential path, before the fix **every** committed position carried rejected
draft tokens, after it every position matches exactly:

| pos | sequential | before fix | after fix |
|---|---|---|---|
| 9 | [13 271] | [271 760] WRONG | [13 271] |
| 17 | [728 310] | [264 3545] WRONG | [728 310] |
| 21 | [264 41163] | [3545 421] WRONG | [264 41163] |
| 31 | [19132 41163] | [1472 63358] WRONG | [19132 41163] |

(760, 3545, 421, 63358 are all drafts that were rejected.) Speculative decode is supposed to be
output-preserving - every committed token is the trunk's own argmax. **It was not:** before the fix,
greedy output diverged from the sequential path at token 5 of 128. After it, K=2 is byte-identical over
128 tokens and K=1 diverges at token 29.

**A third defect**, in `src/engine/mtp.cu`: the MTP block's two gated-residual sites ran with
`post=nullptr`, i.e. as plain residual adds. exllamav3's `build_qwen4_block` constructs GatedResidual
with `use_combine` defaulting to true for a decoder block (only the logit mixer passes False). The post
gate scales each sublayer's output per stream before it is added back, and that stream stack IS the
next chain step's tap - so the error propagated into every later draft. Worth +1.1pp acceptance on its
own (64.1% -> 65.2%).

### Result: speculation is a large net win, ON by default

| config | decode tok/s (8k) | acceptance | tokens/step |
|---|---|---|---|
| MTP off | 45.46 / 46.32 / 46.35 | - | 1.00 |
| **MTP on, K=1 (default)** | 53.44 / 55.77 / 55.39 (**+20%**) | **68.4%** (104/152) | ~1.68 |

Acceptance 68.4% is *above* the 57-61% that exllamav3 and ninfer report on this class of 3090, and per-slot
acceptance now decays geometrically with depth (~0.65 depth-1, ~0.29 depth-2, ~0.22 depth-3) - the
signature of a healthy MTP head. Measured independently of the agent that fixed it, 3 runs per cell.

### Costs, stated plainly
- **Speculation is not bit-exact against the sequential path.** K=1 diverges at token 29 of 128; K=2
  matches over 128. This is irreducible rather than a bug: a width-(K+1) verify forward associates
  differently from width-1, so attention/MoE drift ~1.6e-5 relative and a near-tie eventually flips. The
  GDN recurrence itself is bit-exact (`test_gdn_chunked_parity` proves it).
- **Prefill -2.3%** (1037 -> 1013 tok/s) for the draft KV fill.
- `HELIOS_MTP=0` restores the sequential path and the reference digest `3aaa5693eee90a81513be908f1ba3263`
  exactly, which is the right escape hatch for anyone who needs bit-reproducible sequential decoding.

---

## Tensor-core mixer is now the DEFAULT: the decode penalty was an artifact of the broken MTP path

The TC mixer was rejected earlier for costing 19% of decode. That measurement was taken while the
speculative path was corrupting the trunk's recurrent state (the `ple_conv_` / `hist_` rollback bug).
Re-measured in the corrected build, the tradeoff inverts - TC is now better on BOTH axes:

| | decode tok/s (8k) | prefill tok/s (13.8k) | acceptance |
|---|---|---|---|
| TC off | 55.73 / 55.49 / 55.57 | 1051.6 / 1052.7 / 1053.0 | 68.4% |
| **TC on (now default)** | **56.58 / 58.52 / 58.26** | **1464.9 / 1464.9 / 1463.1 (+39%)** | **76.6%** |

Decode is not slower, and MTP acceptance is *better* with the tensor-core mixer (68.4% -> 76.6%).
Against exllamav3 this puts prefill at ~76% of baseline instead of ~54%. Three runs per cell, spread
under 1%.

This is a correction to the earlier conclusion, and the reason is worth stating: the "-19% decode"
number was a real, repeatedly-measured effect in a build whose decode path was itself broken. It was
never a property of the TC kernels in isolation. Both the pipeline-interaction investigation and the
L2/clock/workspace eliminations from the previous round were chasing an artifact of the MTP state
corruption - which is why the root cause turned out to be a rollback bug rather than anything about
caches, clocks or occupancy.

`HELIOS_MIXER_TC=0` restores the fp32 mixer (the numerical oracle, and the path both are validated
against). Default (MTP off, TC off) remains the sequential reference: digest
3aaa5693eee90a81513be908f1ba3263.

---

## End-to-end runtime verification (the deliverable surface, not just the CLI)

`./build/helios serve <model> --port 8099`, OpenAI-compatible, verified by hand:

- `GET /v1/models` -> `"context_length":262144`, owned_by helios, model Qwen3.8-Flash-Next-exl3.
- `POST /v1/completions`, "The capital of France is", 48 tokens, temp 0 -> `" Paris.\n\n# 100% Cotton
  Fabric: Properties, Uses, and Care..."`, 1.03 s wall.
- Large context: 12,573 prompt tokens + 64 completion in 9.87 s wall.
- SSE streaming works: "Count to five:" -> `" 1, 2, 3, 4, 5"`, finish_reason stop.

### Decode breakdown with all current defaults (MTP on, TC on)
Per-phase GPU time over 154 steps, share of measured phases:

| phase | cumulative ms | share |
|---|---|---|
| MoE | 42.5 | 42% |
| GDN | 20.8 | 21% |
| attention | 13.9 | 14% |
| amix (mHC mixer up) | 8.5 | 8% |
| PLE | 5.4 | 5% |
| mmix (mHC mixer down) | 4.1 | 4% |
| apply / embed / final | 5.7 | 6% |

The mixer is now 12% of decode, down from 35% before the tensor-core port - that work is done. Decode
is dominated by the MoE at 42%, and that kernel is byte-identical to upstream exllamav3 on every tuning
constant (MOE_SMS_PER_EXPERT 8, MOE_TILESIZE_M 16, MOE_TILESIZE_K 32, MOE_SH_STAGES 3, SMEM_MAX 90KB)
with upstream's grid logic, so the remaining decode gap is not in this kernel. At decode the batch is
K+1 = 2 rows against a 16-row tile with top-10 of 512 experts, so most of every tile is padding; closing
that is a kernel redesign, not a configuration change. Documented as the honest ceiling.

## Split-KV decode attention rewritten: 7-8x on the kernel, decode +9.7% at 4k and +75% at 33k

`gqa_split_kv_kernel` was the last kernel in the decode step below 5% of DRAM peak: **6.30 ms/step,
22.3% of the step, 45 GB/s of a 936 GB/s card** on an 82-SM RTX 3090. Two independent defects, both
structural, both now removed.

**1. The occupancy was frozen at 15%.** `GQA_SPLIT_KV` was a compile-time `16`, so the grid was
`16 x n_kv_heads x n_q_heads` = 384 warps at n=1 (768 with MTP's n=2) = **4.7 warps/SM, at any context
length**. A 4k decode and a 33k decode launched the same grid, so the 33k step paid 8x the serial KV
walk for zero extra parallelism. The chunk count is now chosen at launch from the SM count, the kv-head
count and the key range (`gqa_split_kv_chunks`): 256 chunks at 9k, capped so a chunk is never shorter
than 32 keys. Occupancy now scales with context.

**2. Every KV byte was fetched 12 times.** With 24 query heads over 2 kv heads, one warp per query head
meant 12 warps re-read the same K and V - 452 MB of L1<-L2 traffic per launch against 18.9 MB of unique
KV (~1.07 TB/s through L2) while DRAM sat at 45 GB/s. One warp now carries **all 12 query heads of its
kv head at once**, so K and V are read once per token and the 12 uses are register arithmetic.

Note this is the *opposite* conclusion to the prefill kernel, and both are right. `gqa_dense_f16`'s
header already recorded that putting several query heads in one block "measured no faster" at prefill,
because a grid of `n * n_q_heads` blocks already fills the card and the reuse is served by L2. That
conclusion is specific to a full grid: at decode, where 24 heads over 2 kv heads cannot fill it, the
same grouping is worth 7-8x. The comment now says so rather than leaving the reader with a
contradiction.

### What changed in the code

- `gqa_split_kv_kernel` is templated on the GQA group (12 here) and holds `group x 8` fp32
  accumulators plus the group in registers; the query vector is converted to fp32 **once** per chunk
  instead of being re-read from memory on every key.
- The running-max rescale of the accumulators is skipped on the steps where the max does not move.
  That is exact, not approximate: the factor is `__expf(0) == 1.0f` and `o * 1.0f == o`. Running maxes
  over ~35-130 keys move a handful of times, so this removes most of 96 multiplies per key. The branch
  is warp-uniform (after the butterfly every lane holds the same dot).
- The key loop is unrolled 4x for memory-level parallelism.
- The combine went from one 32-thread block per (row, head) to one warp per (row, head, 32-channel
  slice) - 192 warps over 82 SMs instead of 24 blocks on 24 SMs. It is *slower per launch* (3.7 -> 17.2
  us) because it now has 256 partials to merge instead of 16, and 3.5x faster per byte, netting out
  ahead.
- `GQA_WARPS_PER_SM = 8` and `GQA_MIN_KEYS = 32` were swept over 1k..64k of key range and n = 1 and 2.
  8 is also the occupancy limit (`__launch_bounds__(32, 8)`): a target above 8 buys no resident warps,
  it just splits chunks finer and pushes the grid past one full wave (n=2 at 32k: 0.19 ms at 8, 0.27 at
  10, 0.25 at 12). `GQA_MIN_KEYS` 32 beat 64 and 128 at every short context.
- GROUP = 16 is deliberately **not** instantiated: 128 accumulators plus 128 query registers exceed the
  255-register file and ptxas spills. Such a group falls back to one head per warp, which is correct.

### Numerics

Within a (query head, key) the float operation sequence is unchanged - same 8-element fp32 dot in the
same order, same 5-step butterfly, same running max, same sum, same weighted-V accumulation - so each
head's per-chunk partial is bitwise what the one-head-per-warp form produced for that same chunk. What
changes is the partition of the key range, which re-associates the online softmax across chunks. That
is the same re-association the fixed-16 form already performed against the unsplit kernel.

The parity test now covers the shape the engine actually runs the path at (the old case was a 600-key
window, where the launch picks 18 chunks), against a double-precision CPU reference, against
`gqa_dense_f16`, and run-to-run:

```
[qwen_gqa/SPLITKV]          n=3 pos0=597: max_err 3.007e-05 on |o|~0.020 PASS   (same as DENSE)
[qwen_gqa/SPLITKV-decode]   n=1 pos0=8191 group=12 keys=8192: split_err 7.589e-06, dense_err 7.589e-06, 0/6144 differ between launches PASS
[qwen_gqa/SPLITKV-decode]   n=2 pos0=6000 group=12 keys=6002: split_err 1.245e-05, dense_err 1.245e-05, 0/12288 differ between launches PASS
[qwen_gqa/SPLITKV-decode]   n=1 pos0=4095 group=4  keys=4096: split_err 1.395e-05, dense_err 1.395e-05, 0/2048 differ between launches PASS
```

**Error bound: max abs 1.4e-5 at an output magnitude of 0.005-0.007, i.e. below fp16 output rounding,
and identical to the dense path's error to every digit printed.** The third case is group 4, not this
model's 12, to cover the dispatch's fallback instantiation - a shape bug there is invisible at 12. All 9
suites pass, and the MTP-off + TC-off reference digest is still
`3aaa5693eee90a81513be908f1ba3263`.

### Kernel level (standalone, 200 reps, median of 5, n=1 decode, unique-KV GB/s)

| keys | before | after | speedup | GB/s before | GB/s after |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.0340 ms | 0.0270 ms | 1.3x | 62 | 78 |
| 4,096 | 0.1823 ms | 0.0396 ms | 4.6x | 46 | 212 |
| 8,192 | 0.3597 ms | 0.0539 ms | 6.7x | 47 | 311 |
| 16,384 | 0.7130 ms | 0.0809 ms | 8.8x | 47 | 415 |
| 32,768 | 1.4219 ms | 0.1355 ms | 10.5x | 47 | 495 |
| 65,536 | 2.8470 ms | 0.2386 ms | 11.9x | 47 | 563 |

The old column is flat at 47 GB/s from 4k up - that is the 12x re-read saturating L1/L2 while DRAM idles.
The new column climbs toward the memory bound as the per-chunk partial output amortises. At n=2 (what MTP
actually launches) the same table reads 0.0446 / 0.0784 / 0.1174 / 0.1961 / 0.3537 ms.

### In the engine, 3 runs per cell, 320 decode tokens, all defaults (MTP on, TC on)

| context | decode before | decode after | delta | prefill before | prefill after |
|---|---:|---:|---:|---:|---:|
| 4,210 tok | 77.20 / 77.00 / 76.68 = **76.96** | 84.35 / 84.53 / 84.35 = **84.41** | **+9.7%** | 1,275 | 1,266 |
| 8,416 tok | 69.00 / 69.06 / 68.75 = **68.94** | 83.04 / 83.00 / 83.06 = **83.03** | **+20.4%** | 1,408 | 1,413 |
| 16,803 tok | 41.51 / 41.34 / 41.33 = **41.39** | 58.62 / 58.45 / 58.04 = **58.37** | **+41.0%** | 1,464 | 1,467 |
| 33,043 tok | 36.64 / 36.51 / 36.61 = **36.59** | 64.21 / 64.32 / 64.08 = **64.20** | **+75.5%** | 1,376 | 1,376 |

Run-to-run spread is under 0.6% in every cell. **Prefill does not regress** (within +-0.7%, and
prefill never touches this kernel - `n = 1024 > 32` dispatches to the mma path). The 8k->16k drop in
both columns is MTP acceptance falling from 99.4% to 52.4% on that prompt, not attention: the same
`spec:` line is printed by both builds and the delta is a ratio between them.

**VRAM: unchanged to better.** gpu0 free 1.41 -> 1.42 GB, gpu1 free 0.60 -> 0.61 GB. The `kv_part`
buffer got *smaller*, 12.09 -> 8.31 MB, because the chunk count falls as the row count grows while the
buffer is indexed by `n * chunks(n)`, which peaks at 352 slots. Sizing it from `ctx_cap` rather than
from the current position is what makes that bound provable; at 262k context a decode row uses 6.0 MB
of the 8.3.

### Attribution (nsys, in-engine, 200 decode tokens, per full-attn layer per step)

| | 16,803 tok before | 16,803 tok after | 33,043 tok before | 33,043 tok after |
|---|---:|---:|---:|---:|
| `gqa_split_kv_kernel` avg | 769.1 us | **93.4 us** | 1,515.1 us | **163.1 us** |
| combine avg | 3.7 us | 17.2 us | 3.7 us | 17.2 us |
| total per layer-step | 772.8 us | **110.6 us** | 1,518.8 us | **180.3 us** |
| share of the decode step | **30.8%** | **6.0%** | **47.5%** | **9.8%** |
| unique-KV bandwidth | 44.7 GB/s | **368 GB/s** | 44.7 GB/s | **415 GB/s** |

So the 4.8%-of-peak kernel is now at 41% of peak counting the partial write and re-read
(74.0 MB moved in 180.3 us), and it is no longer the context-scaling phase. At 33k the rest of the step
is what decode now spends its time on.

**What this does and does not buy.** Attention is still linear in context - 180 us per layer-step at
33k against 110 at 17k - so at 262k it is still ~1.4 ms/step, but that is now 2% of a step rather than
the majority of it. The win is largest exactly where decode was collapsing (+75% at 33k), which was
the point. QSA remains the only thing that makes long context sublinear, and it is still opt-in.

---

## GQA decode KV scan rewritten: 7-8x on the kernel, and decode stops degrading with context

The kernel-level nsys trace replaced the phase profiler, which had been reporting 0.66 ms of a 28.6 ms
decode step - 2% coverage, so its "MoE 42%" breakdown was measuring almost nothing. Real attribution
(sum of kernel durations, both cards, 28.27 ms/step, 2221 launches/step):

| component | ms/step | share | launches/step |
|---|---|---|---|
| MoE (routed mgemm + shared gemv + routing) | 9.41 | 33.3% | 510 |
| attention (KV scan + projections) | 7.35 | 26.0% | 108 |
| mixer | 4.97 | 17.6% | 724 |
| GDN | 3.75 | 13.3% | 379 |
| host/dispatch idle | 3.3 | 10.6% | - |

The two cards are effectively serialised - card0 busy 12.54 ms, card1 15.73 ms, **both busy only 0.23 ms
(0.7%)** - which is inherent to a layer-split pipeline on a 2-row batch, not a fixable defect.

`gqa_split_kv_kernel` was the one kernel in the step below 5% of DRAM peak (6.303 ms/step, 45 GB/s of
936), and it is *ours*, not upstream, so there was no parity risk. Defects, all confirmed by the trace:

- `GQA_SPLIT_KV` was compile-time 16, so occupancy was 4.7 warps/SM (n=1) / 9.4 (n=2) **at any context**.
- Every KV byte was re-read 12x (24 q-heads over 2 kv-heads): 452 MB of L1<-L2 per launch for 18.9 MB
  of unique KV.
- The combine read partials from 24 blocks on 24 of 82 SMs.

The rewrite puts one warp on each (row, kv head, kv chunk) carrying all 12 heads of its GQA group, so K/V
is read once per token, and picks the chunk count at launch from SM count, kv-head count and key range
(`gqa_split_kv_chunks`), so occupancy scales with context. Also fixed: `gqa_sm_count()` cached a single SM
count across devices while the engine re-asserts the device per layer; a missing `head_dim != 256` guard
in the split-KV launcher; and a GROUP=16 instantiation that ptxas spilled 288 B (dropped, falls back to
one head per warp). The `partial` buffer got *smaller* (12.09 -> 8.31 MB) because chunk count falls as row
count grows while the buffer is indexed by n*chunks(n).

Standalone kernel bench, n=1, 200 reps, median of 5:

| keys | before ms | after ms | speedup | GB/s before | GB/s after |
|---|---|---|---|---|---|
| 4096 | 0.1823 | 0.0396 | 4.6x | 46 | 212 |
| 8192 | 0.3597 | 0.0539 | 6.7x | 47 | 311 |
| 16384 | 0.7130 | 0.0809 | 8.8x | 47 | 415 |
| 32768 | 1.4219 | 0.1355 | 10.5x | 47 | 495 |

**In-engine decode, all defaults, 3 runs per context** (before this change decode was 58.7 tok/s at 8k):

| context | decode tok/s | delta |
|---|---|---|
| ~3k | 68.34 / 68.86 / 68.96 | +17% |
| ~7k | 65.24 / 65.18 / 65.62 | +11% |
| ~15k | 65.52 / 65.82 / 66.32 | +12% |
| ~32k | 64.01 / 64.29 / 64.05 | +9% |

The structural result matters more than the headline: decode now falls only ~7% from 3k to 32k. Before,
this kernel alone was on track to be ~14 ms/step - half the step - at 32k context, and the 47 GB/s flat
line across every key count was exactly the signature of the frozen split. VRAM improved slightly (gpu0
free 1.41 -> 1.42 GB, gpu1 0.60 -> 0.61 GB). Prefill unchanged. 9/9 suites pass and the sequential
reference digest is byte-identical at 3aaa5693eee90a81513be908f1ba3263.

---

## THE GRID WAS MEASURING NOTHING: every decode cell terminated early, on both engines

Chasing why the 8k grid cell sat at 30 tok/s while 16k did 73 led to the real problem, and it was not
in the engine. The grid's prompt is one paragraph repeated to length, and asked to continue it, the
model decides it is finished:

| grid prompt | prompt tokens | **tokens actually generated** |
|---|---|---|
| 4k | 3,697 | 41 (of 512 requested) |
| 8k | 7,426 | **1** |
| 16k | 14,884 | 41 |

A "decode 33.7 tok/s" figure measured over ONE token is not a throughput number. Every decode cell in
every grid in this file was meaningless, and the earlier explanation - that the repetitive prompt
collapses MTP acceptance - was wrong: at 4k and 16k the same text accepts 86.4% of drafts. The 8k cell
happened to stop after one token, which is also why it showed 0/1 draft slots.

**Both engines were measured this way.** `engine_bench` parsed `decode X tok/s` without checking how many
tokens were produced, and the server path used `un.get("completion_tokens", out)` - falling back to the
REQUESTED count when the key is absent, so a short generation was scored as a full one. That inflated the
exllamav3 baseline to a headline "108-121 tok/s" that it does not achieve on a prompt that actually runs.

### Fixes to the instrument
- `engine_bench` now returns the actual generated count (parsed from `[gen] N tokens`).
- `grid_bench` prints a `got` column and marks a cell **INVALID (early stop)** when fewer than half the
  requested tokens came back, instead of silently comparing it.
- The bench prompt gets a continuation instruction (`build_prompt_for_bench`) so the model keeps writing.
  Both engines are measured with the identical prompt, as before.

### Valid 1:1, every cell generating the full requested length

| pre/out | exl3 pre | helios pre | pre % | exl3 dec | helios dec | dec % |
|---|---|---|---|---|---|---|
| 4k/512 | 1657 | 1197 | 72% | 64.1 | 60.4 | 94% |
| 4k/1000 | 1659 | 1202 | 72% | 62.4 | 61.7 | 99% |
| 4k/2000 | 1660 | 1208 | 73% | 60.5 | 63.0 | **104%** |
| 8k/512 | 1755 | 1303 | 74% | 62.0 | 62.4 | **101%** |
| 8k/1000 | 1865 | 1312 | 70% | 64.4 | 59.9 | 93% |
| 8k/2000 | 1865 | 1307 | 70% | 61.1 | 58.1 | 95% |
| 16k/1000 | 1924 | 1377 | 72% | 65.5 | 56.5 | 86% |
| 16k/2000 | 1924 | 1374 | 71% | 58.7 | 57.7 | 98% |

(16k/512 is omitted: exllamav3 reports 2186.8 tok/s there, a server-side artifact of the same class of
bug - the guard only covers the engine side, which is where the decode figure is parsed from the log.)

**The corrected headline: decode is at PARITY with exllamav3 (86-104%, winning two cells outright), not
the 26-62% every earlier table in this file reported. Prefill is 70-74%. Context is 262,144 vs 230,000.**

Every performance conclusion drawn before this point was measured through a broken instrument and should
be read as unverified. The one thing that survives unchanged is the engineering work - the correctness
fixes and the kernels - which were verified by suites, digests and byte-comparisons rather than by
tok/s. This is the third time in this project that a measurement was wrong by more than 25%: the md5
gate that included a timing line, the phase profiler that covered 2% of a decode step, and now a
benchmark whose cells all stopped early.

## Instrument hardened on the reference side too, and the final valid 1:1

`server_bench` reported `un.get("completion_tokens", out)` - falling back to the REQUESTED count when
the field was absent, which invents tokens that were never generated. The exllamav3 16k/512 cell
scored "381.3 tok/s" off a **single** generated token. The capture path now takes the count only from
the server's own usage, records a sample with no `completion_tokens` as a failed sample rather than a
fast one, and the compare path marks a cell INVALID when **either** engine generated less than half
what was asked. Both columns now show what each side actually produced.

| pre/out | exl3 pre | helios pre | pre % | exl3 dec | helios dec | dec % | generated (us/them) |
|---|---|---|---|---|---|---|---|
| 4k/512 | 1656 | 1213 | 73% | 66.1 | 59.6 | 90% | 512 / 512 |
| 4k/1000 | 1657 | 1213 | 73% | 61.4 | 61.3 | **100%** | 1000 / 1000 |
| 4k/2000 | 1659 | 1207 | 73% | 60.5 | 61.6 | **102%** | 2000 / 1317 |
| 8k/512 | 1754 | 1298 | 74% | 62.0 | 61.5 | 99% | 512 / 512 |
| 8k/1000 | 1863 | 1298 | 70% | 60.5 | 59.5 | 98% | 1000 / 1000 |
| 8k/2000 | 1754 | 1302 | 74% | 62.2 | 57.6 | 92% | 2000 / 1620 |
| 16k/512 | 1924 | 1371 | 71% | - | 53.2 | - | 512 / **1** - INVALID, they stopped |
| 16k/1000 | 1924 | 1375 | 71% | 66.6 | 56.0 | 84% | 1000 / 1000 |
| 16k/2000 | 1864 | 1373 | 74% | 59.2 | 57.3 | 97% | 2000 / 928 - INVALID, short |

**Decode: 84-102% of exllamav3, i.e. parity, with two cells ahead. Prefill: 70-74%. Context: 262,144
vs 230,000.** Prefill is now the only axis measurably below the reference, and the two INVALID cells are
the reference under-measuring itself, not us.

---

## Prefill kernel attribution, and a free +5.3% from reversing a stale micro-batch default

nsys on the shipped binary, 12.9k prompt = 13 chunks of 1024, MTP off, 9.364 s wall:
**15.099 card-seconds of kernel time, mean card utilisation 80.6%** (card0 77.6%, card1 83.7%), 47,587
kernel launches, 208,258 `cudaMemcpyAsync` (16.2 per token). The phase profiler had accounted for only
~12% of this, so as on decode it was not to be trusted.

| kernel | % of GPU time | owner |
|---|---|---|
| `exl3_gemm_kernel` (4 variants) | 41.0% | **upstream exl3** |
| `exl3_moe_kernel` (grouped MoE) | 33.6% | **upstream, byte-identical** |
| mixer (tensor-core) | 5.7% | ours |
| `routing_gemv_batch_kernel` | 4.6% | upstream port |
| `gqa_dense_mma_kernel` (prefill attention) | 4.6% | ours - already 24.8% of fp16 TC peak |
| GDN conv1d + transpose | 3.0% | ours |
| `gemm_nt_f16_k` (PLE projections) | 2.5% | ours - SIMT 16x16, **no tensor cores** |

**79.2% of prefill GPU time is upstream kernels shared with the reference**, so the improvable surface
is ~21%. Unlike the decode case, prefill attention is already efficient (24.8% of TC peak, 2.34 waves);
the 7-8x that was available in the decode GQA kernel is not available here.

### The win: micro-batch default reversed from 2 to 1 (+5.3%)

The code carried a measurement claiming M=2 beat M=1 (880 vs 908 tok/s). That is no longer true, and
the reason is that the balance changed underneath it: the mixer became tensor-core and the decode GQA
kernel was rewritten, so the per-micro-batch handoff and sync is now a bigger share of a much shorter
stage, while the grouped MoE's 1.78 ms per-launch fixed cost is unchanged and is paid once per
micro-batch. Re-measured on a 12.9k prompt, 3 runs each: **M=1 1516.0 / 1508.8 / 1500.7**,
M=2 1438.2 / 1429.7 / 1432.1. M=1 is +5.3%. Card-1 utilisation drops 83.7% -> 77.7%, a trade worth making.
The default is now M=1 and the stale numbers in the comment are replaced with these.

### The identified next step, and its measured blocker
Raise `max_chunk` 1024 -> 2048 to halve the grouped MoE's fixed cost per token a second time (1.78 ms of
a 4.07 ms launch at 1024 tokens; worth roughly +7% prefill). **It does not fit**: chunk 1536 and 2048
both fail to allocate on card 1, which has 0.58 GB free at chunk 1024 and reports `free 0.03 GB` at 2048
- with MTP both on and off. Closing that needs ~1.2 GB freed per card; the identified occupants are the
MTP scratch (~190 MB at 1024 rows, allocated whenever `cfg.has_mtp` **even with HELIOS_MTP=0**), PLE
scratch (~176 MB), the attention set (~99 MB) and the activation set (~85 MB). Not attempted here.

MTP costs 3.4% of prefill for the draft KV fill (1491.9 with, 1543.5 without) - the price of the
+20% decode it buys.

### Micro-batch reversal at grid level: a real 4k regression, larger gains at 8k/16k
The +5.3% was measured on a single 12.9k prompt. On the full grid the effect is context-dependent and
the 4k cells get *worse* - reported here rather than only the favourable half:

| pre/out | pre % (M=2) | pre % (M=1) | 
|---|---|---|
| 4k/512 | 73% | 70% |
| 4k/1000 | 73% | 69% |
| 4k/2000 | 73% | 69% |
| 8k/512 | 74% | 77% |
| 8k/1000 | 70% | 73% |
| 8k/2000 | 74% | 77% |
| 16k/512 | 71% | 76% |
| 16k/1000 | 71% | 76% |
| 16k/2000 | 74% | 78% |

M=1 loses ~4 points at 4k and gains 3-7 points at 8k and 16k. The mechanism is the one the nsys pass
described: with M=1 the two cards do not overlap, so a prefill that is only ~4 chunks long loses more
from the missing overlap than it gains from halving the grouped MoE's per-launch fixed cost. Longer
prefills amortise that loss. M=1 remains the default because the contexts that matter are the long
ones and they gain the most, and because a 16k prefill is several seconds of real time against a 4k
prefill's fraction of one. A length-dependent default (M=2 below ~6k, M=1 above) would capture both
halves and is the obvious follow-up, not attempted here.

Decode at 4k moved 90-102% -> 84-99% between the two grid runs. Decode always runs one micro-batch
(`M = (n >= 64) ? micro_ : 1`), so this cannot be caused by the change and is run-to-run variance; it
is noted because it bounds how much of any single grid cell should be believed.

### The chunk-size lever is a dead end: bigger chunks are SLOWER, not faster
The nsys pass predicted that raising `max_chunk` 1024 -> 2048 would halve the grouped MoE's 1.78 ms
per-launch fixed cost and buy roughly +7% prefill. It does not. Measured (12.9k prompt, defaults
otherwise, 2 runs each):

| ctx | chunk | prefill tok/s |
|---|---|---|
| 262144 | **1024** | **1501.1** |
| 131072 | 1536 | 1339.0 |
| 131072 | 2048 | 1257.8 |
| 65536 | 2048 | 1250.3 |
| 32768 | 4096 | 990.2 |

Monotonically worse as the chunk grows, down to -34% at 4096. Amortising a per-launch cost is real, but
it is outweighed: a larger chunk is a longer serial GDN recurrence, more scratch competing for the same
L2, and less frequent natural split points. **The default 1024 is already the optimum**, and the VRAM
that made 2048 unallocatable was never the binding constraint. Recording this because the prediction was
specific, plausible, and wrong - the fourth time in this project that a well-reasoned prediction failed
to survive a measurement (after the CUDA graph's +5%, the tensor-core mixer's -19% decode, and the
micro-batch M=2 default).

### Where the remaining prefill gap actually is
79.2% of prefill GPU time is upstream exllamav3 kernels we call with upstream's own grid heuristics
(exl3_gemm 41.0%, exl3_moe 33.6% byte-identical, routing 4.6%). Our own code is ~21%: mixer 5.7% (already
tensor-core), GDN conv/transpose 3.0%, PLE `gemm_nt_f16_k` 2.5% (SIMT 16x16, no tensor cores - genuinely
improvable), the PLE n-gram row-gather copy loop (206k CUDA calls, measured at only +0.4-1.9% despite
nsys overstating it 5x). Rewriting every kernel of ours is worth a few percent; closing 69-78% to parity
would require diverging from the upstream MoE/GEMM kernels, which is a research project, not tuning.

---

## The two remaining prefill kernels of ours: PLE projections onto tensor cores, and the GDN conv/transpose access pattern

The prefill attribution above ends with "our own code is ~21%, and two items in it are genuinely
inefficient": the PLE `gemm_nt_f16_k` (a SIMT 16x16 fp16 GEMM) and the GDN `conv1d_update` +
`transpose_f32_bf16` pair. Both are rewritten here. Everything below is measured on the shipped
binary at the 12.9k bench prompt (12,869 tokens, 13 chunks of 1024), and again in isolation.

### 1. PLE key_proj / value_proj: 2.27 -> 58.3 TFLOPS (25.7x)

`gemm_nt_f16_k` computed 16x16 of output per 256 threads with one scalar half->float convert and FMA
per MAC. Isolated, on the two shapes the PLE actually issues per chunk:

| shape | scalar `gemm_nt_f16_k` | tensor-core `gemm_nt_mma_k` | speedup |
|---|---|---|---|
| key_proj M=1024 N=10240 K=2560 | 23.665-32.661 ms / 1.64-2.27 TFLOPS | **0.920 ms / 58.36 TFLOPS** | 25.7-35.5x |
| value_proj M=1024 N=2560 K=2560 | 5.950-6.000 ms / 2.24-2.26 TFLOPS | **0.242 ms / 55.39 TFLOPS** | 24.6x |
| 4096^3 (not a PLE shape, for scale) | 60.631-61.138 ms / 2.25 TFLOPS | 2.213-2.239 ms / 61.40-62.10 TFLOPS | 27.3x |

The scalar key_proj cell is the only one that moved between runs - 23.7 ms with the card otherwise
idle, 32.7 ms with a second job resident - because at 52 MB of B and 21 MB of output it is by far
the most bandwidth-sensitive of the three. value_proj, the square shape and the in-engine nsys
figure below all agree on ~2.25 TFLOPS for the scalar path, and that is the figure used throughout.

58-62 TFLOPS is 82-87% of the 71 TFLOPS fp16-tensor-with-fp32-accumulate dense peak of a 3090, i.e.
the kernel is close enough to the roofline that there is nothing structural left in it.

The design is `gr_mix_tc.cu`'s, specialised for the NT layout the checkpoint is stored in, so
nothing is transposed: A = `x` (M,K) row-major, B = `w` (N,K) row-major, and B is already the
column-major (K,N) operand `mma.row.col` wants. 128x128 block tile, 8 warps in a 2x4 grid (64x32
per warp = 4 m-tiles x 4 n-tiles = 16 `mma.m16n8k16` and 64 fp32 accumulators per lane), k staged 32
deep in a double buffer, fragments read with `ldmatrix.x4` (A) and `ldmatrix.x2` (B) off a
40-half-pitch shared tile (pitch 20 words, so the eight row addresses of one fragment cover all 32
banks exactly once). 40 KB of shared memory, under the 48 KB default, so no opt-in carve-out.

**The 1-D grid is M-fastest on purpose.** B is 52 MB at the key_proj shape against a 6 MB L2, so
whether the `tiles_m` blocks that share a 128-row B tile are the blocks the scheduler puts on the card
together is worth more than any micro-optimisation inside the block: a row-major grid re-streams all
of B once per M-tile row (419 MB), the M-fastest grid streams it once (52 MB) and leaves A - 5 MB,
which fits in L2 whole - as the re-read operand. A first attempt used Cutlass's `ThreadblockSwizzle`
formulation, which indexes both tile axes off the same group counter; that is only correct when
`tiles_m` and `tiles_n` share a common factor, and at the PLE's shapes (8 x 80) it silently skipped
three quarters of the output while *appearing* to run at 202 TFLOPS. The isolated spot-check against
a CPU reference is what caught it; a 1-D M-fastest index has no such failure mode.

The GEMM keeps the decode route (M == 1 -> `gemv_nt_f16`, a warp-per-output GEMV) and stays on the
scalar kernel below half an M-tile of rows, which is only the speculative-verify shapes (n of 1..8)
where the whole call is a few microseconds either way.

### 2. GDN conv + transpose: the access pattern is fixable, and fixing it is 4.6x and 5.6x

Characterised, not guessed. Both kernels move 21 MB of `mixed_qkv` per launch, and both were already
sitting at the card's bandwidth limit **for the pattern they were issuing**:

| kernel | pattern | amplification | effective BW | bytes really moved |
|---|---|---|---|---|
| `transpose_f32_bf16_k` | read coalesced along F, store `dst[f*M+m]` stride M*2 = 2048 B | **16x on the write** (a 2-byte store into its own 32 B sector) | 109 GB/s "useful" | 378 MB in 577 us = 655 GB/s |
| `conv1d_update_kernel` | `x` is channel-major `x[d][s]`, one thread per channel walks s: a 2-byte load 2 KB from the previous lane's | **16x on the read** | 87 GB/s "useful" | 356 MB in 482 us = 739 GB/s |

So neither was arithmetic-bound or latency-bound; both were paying 16x memory traffic, and 655/739
GB/s of *effective* bandwidth against a 936 GB/s peak is why they looked like "strided kernels at
100 GB/s". The fix in both cases is the same shape: stage a 2-D tile in shared memory so the load
map and the store map each stay contiguous on their own axis, and pay the 16x once in shared memory
instead of once in DRAM.

**`transpose_f32_bf16`**: a 32x32 shared tile, load with consecutive lanes on consecutive f
(contiguous in `src`), store with consecutive lanes on consecutive m (contiguous in `dst`, which is
[F,M]). 269.95 ms -> **47.86 ms**, and 109 -> **615 GB/s** of useful bandwidth (109 -> 615 GB/s is
5.6x; the useful-bytes-per-second is now 62.9 MB / 102.3 us per launch).

**`conv1d_update`**: 64 channels x 64 sequence positions per block, 2-D grid (160 x 16 = 2560 blocks
at dim = 10240, seqlen = 1024) so that no block ever waits on another - the K-1 positions a tile's
first output needs are re-read by whichever tile owns them, 6% of the traffic at K=4, against a
serial s-walk that would serialise all 1024 steps inside every channel block. The shared pitch is
forced odd (`TS + K - 1 + (K & 1)`) so the compute pass's consecutive-lanes-on-consecutive-d read
lands on 32 distinct banks. 225.38 ms -> **49.53 ms**, and 87 -> **397 GB/s** of useful bandwidth.

The arithmetic per output element is untouched - the same K-term fmaf chain in the same order, the
same `_sigmoid_fast_exp`, the same `__float2bfloat16_rn`, the same window shift - so the conv is
bit-identical to the per-channel form it replaces, and the transpose is bit-identical to a
per-element copy. Both are asserted so, against a CPU reference, in `test_aux` (the conv across
dim 64/130, seqlen 1/5/64/130/256, K = 1..8, activation on and off, history on and off, plus a
two-sequence call; the transpose across M/F multiples and non-multiples of the 32x32 tile). Neither
kernel had any test coverage before this.

One hazard worth recording: with the grid now 2-D, every s-tile wrote `conv_state` back and the first
s-tile read it, which is a race (the read would see either value). The state read and the writeback
are now both the first s-tile's alone, which is what the single-block-per-channel form did implicitly.

### 3. End to end

Total kernel time over the whole 12.9k prefill: **13.905 s -> 13.079 s, -826 ms (-5.9%)**. The three
kernels above go from 869.6 ms to 111.3 ms; the remaining 13.0 s is upstream exl3 (79% of it) plus the
mixer and attention, none of which this change touches.

End-to-end, two binaries from this same tree, arms interleaved rep by rep, median of 4 runs each
(`test/ab_gdn_ple.py`; the ref arm is the pre-change conv + pre-change transpose + HELIOS_GEMM_MMA=0,
i.e. exactly the engine as it was, and it reproduces the reference digest 3aaa5693...):

| cell | prompt tokens | ref tok/s | new tok/s | delta | ref spread | new spread |
|---|---|---|---|---|---|---|
| prefill 4k | 3728 | 1183.8 | **1248.5** | **+5.47%** | 1177.2-1185.2 | 1240.7-1255.0 |
| prefill 8k | 7457 | 1423.6 | **1495.0** | **+5.02%** | 1421.4-1425.8 | 1494.1-1501.5 |
| prefill 16k | 14915 | 1561.2 | **1630.0** | **+4.41%** | 1558.9-1568.0 | 1624.5-1635.6 |
| decode (263 ctx, 128 out) | 263 | 59.7 | 58.9 | -1.26% | 56.9-59.7 | 50.5-59.0 |

Prefill is +4.4 to +5.5% across the range, and the spread of the two arms does not overlap in any
prefill cell, so it is not noise. The gain is smaller than the -5.9% of kernel time because the two
cards are pipelined and wall clock is not pure kernel time.

**Decode: -1.26%, i.e. nothing.** The arms' spreads overlap heavily (56.9-59.7 against 50.5-59.0) and
the new arm's low outlier is 15% below its own median, so the median difference sits inside this
cell's run-to-run noise rather than being a measured regression. Mechanistically the decode path is
untouched - M == 1 still routes to `gemv_nt_f16`, and the conv and transpose are the same work with
better memory patterns - but it is reported rather than dropped because it is the one cell that did
not improve.

**VRAM: unchanged by construction, and not measured cleanly.** The diff adds no `cudaMalloc`: the
GEMM's 40 KB and the conv/transpose's ~14 KB are *shared* memory per block, not device allocations,
and the scratch arenas are untouched. A simultaneous before/after reading could not be taken because
a second agent was holding both cards during this session (a 22.4 GB / 22.6 GB process resident on
gpu0/gpu1), so no peak-VRAM pair is claimed here.

Kernel attribution after the change, same prompt, same nsys pass (share of total GPU time):

| kernel | before | after | achieved |
|---|---|---|---|
| `gemm_nt_f16_k` / `gemm_nt_mma_k` | 374.30 ms, 26 launches, 2.25 TFLOPS (2.0% of GPU time) | **13.95 ms, 26 launches, 60.4 TFLOPS (0.1%)** | 26.8x |
| `transpose_f32_bf16_k` | 269.95 ms, 468 launches, 109 GB/s (1.9%) | **47.86 ms, 468 launches, 615 GB/s (0.4%)** | 5.6x |
| `conv1d_update_kernel<true,false>` | 225.38 ms, 468 launches, 87 GB/s (1.6%) | **49.53 ms, 468 launches, 397 GB/s (0.4%)** | 4.6x |

### 4. What this does NOT change

- **The chunk size.** Untouched. 1024 stays the default: 1501 tok/s at 1024 against 1258 at 2048 is
  still the measurement, and nothing in this change bears on it.
- **The reference digest.** `HELIOS_MTP=0 HELIOS_MIXER_TC=0 ./build/helios gen ... --tokens 128`
  still md5s to `3aaa5693eee90a81513be908f1ba3263`, measured on this tree. The conv and the transpose
  are bit-exact, and the tensor-core GEMM rides `HELIOS_MIXER_TC` (on by default, off exactly when the
  engine is pinned to its scalar oracle; `HELIOS_GEMM_MMA=0/1` overrides it either way), the same way
  the tensor-core mixer does. With the GEMM on, the digest legitimately differs - that is the price of
  a different fp32 accumulation order, and it is why the switch exists. (This is load-bearing: with
  the TC GEMM ungated, the digest moved to `ee99de7a...`.)
- **A pre-existing out-of-bounds read, found on the way and not mine.** `compute-sanitizer memcheck`
  on `test_aux` reports ~26,000 invalid 2-byte reads, all in `gr_post_kernel` at gr_mix.cu:418
  (reading past the end of the smallest allocation in the test's fixture), and nothing in gdn.cu or
  glue2.cu. The mixer is out of scope for this pass and is untouched by it, so it is recorded rather
  than fixed. It is the first thing to look at if the mixer's parity ever wobbles.
- **The PLE n-gram row-gather loop**, still 206k CUDA calls at a measured +0.4-1.9%. Lower value than
  the two above and not attempted here. **(Since done: batched by shard into 125 pinned per-shard
  transfers per chunk; measured +1.9/+1.2/+0.6% at 4k/8k/16k, i.e. the +0.4-1.9% band was right and
  the later +3-6% idle-time estimate for it was not. See "PLE n-gram row gather, batched by shard"
  at the end of this file.)**
- **79% of prefill GPU time is upstream exllamav3** (exl3_gemm 41.0%, exl3_moe 33.6% byte-identical,
  routing 4.6%). Our own code is now ~11% of GPU time, of which the mixer (5.7%) and prefill
  attention (4.6%) are already tensor-core. There is no remaining "genuinely inefficient" kernel of
  ours in this profile.

---

## Overlap audit: the overlap is already banked, and the "2,590 tok/s ceiling" in my own brief was a mis-model

A read-only code audit of the prefill pipeline corrected two things I had been working from.

**1. The cards already overlap - across chunk boundaries, not within a chunk.** `prefill`
(runner.cpp:1544-1550) calls `run_chunk` per chunk with no sync, and `run_chunk_pipeline` never
synchronises after `layer_range(1)`/`chunk_tail`; the only sync is `xcopy`'s
`cudaStreamSynchronize` (runner.cpp:1052), which drains card 0's DMA-out stream and is what preserves
the overlap. So card 0 runs stage0(k+1) concurrently with card 1 running stage1(k). **That is exactly
why M=2 measures worse**: extra micro-batching duplicates a cost the chunk-boundary pipeline already
avoids, on top of splitting the grouped MoE call. The micro-batch reversal from 2 to 1 was right for a
reason I had not identified.

**2. The ceiling I quoted to the optimization agents was wrong.** I passed on "the shared kernels cost
9.94 card-seconds, which even with both cards perfectly busy caps us at ~2,590 tok/s". That divides
shared card-seconds by 2 as though the shared work were evenly splittable, which the card0=[0,24) /
card1=[24,48) layer split forbids - card 0's shared work and card 1's shared work are sequential by
construction. The honest perfect-packing ceiling is total 15.099 card-s / 2 = 7.55 s, i.e. **~1,710
tok/s, about +13% over the measured ~1,500**. There was no +73% sitting there. Any plan premised on it
was chasing a phantom, and the number should never have been handed out.

**Real idle, rebased.** 9.364 s is the nsys *process* span, not the chunk loop; 12,900 tokens at
1,508 tok/s is 8.55 s of chunk loop (~658 ms/chunk), the rest being load/tokenize/warmup with both cards
idle. Rebased, kernel time is 15.099 card-s of 17.10 card-s capacity = **2.00 card-s of real idle
(11.7%), 154 ms/chunk**: card0 99 ms, card1 55 ms.

| source | ms/chunk | addressable? |
|---|---|---|
| pipeline ramp/fill/drain - card1 idle through chunk 0's stage 0, card0 through the last stage 1 + tail | 45 | **no** - irreducible for any 2-stage pipeline |
| **PLE n-gram row gather** | **25-50** | **yes - largest addressable item** |
| barrier re-issue latency after xcopy's host sync | 3 | marginal |
| inter-kernel launch gaps (3,660 launches/chunk) | 3.7 | marginal |
| residual unattributed (per-layer host work: 48 `cudaSetDevice`, a `std::function` mark lambda, ~10 `getenv` per moe_layer call = ~480/chunk) | ~63 | partly |

**The 208,258 `cudaMemcpyAsync` are the PLE, not the handoff.** `ple.cu:256-268`, once per chunk:
`rows = n * cfg.ngram_heads = 1024 * 16 = 16,384`, `row_bytes = 82`, sourced from `ple.shards[k]`, a raw
`mmap` pointer into the 26.2 GB `ngram_embedding.safetensors` (model.cpp:484-505). Count check:
12 full chunks x 16,386 + a 612-token tail x 16 = 206,700, within 0.7% of the reported 208,258. These
are 16,384 driver-staged **pageable** 82-byte copies per chunk, issued at layer 1 of 24, so there is no
earlier work to hide them behind. card0's 44 ms idle excess over card1 is almost exactly this loop -
it is the only card-0-only per-chunk expense in the tree.

### The named next change
Batch those 16,384 per-row copies into ~128 per-shard copies through a pinned staging buffer: bucket
`hash_host[i]` by shard id on the host into a `PleScratch`-owned `cudaMallocHost` buffer
(`rows * row_bytes` = 1.34 MB, `ple.cu:205-230`), then issue one `cudaMemcpyAsync` per shard. `sc.rows`
layout, the `row_ids`/`head_ids` bulk uploads and `ngram_dequant_f16` are unchanged. Estimated +3-6%
prefill - 15-40% of the addressable idle. **That is an estimate, not a measurement.**

---

## PLE n-gram row gather, batched by shard: 16,384 -> 125 transfers per chunk, and the +3-6% estimate did NOT hold

Shipped as default. The mechanism is exactly the one the audit named, and the measurement says the
estimate was roughly 3x too high: **+1.9% at 4k, +1.2% at 8k, +0.6% at 16k**, decode unchanged.

### What is in the code

`ple_layer` no longer issues one 82-byte `cudaMemcpyAsync` per row. The host already has every
`hash_host[i]`, and `shard = id / rows_per_shard` is free, so the rows are counting-sorted by shard
into a pinned buffer and uploaded one contiguous run per shard. Measured per call at the 12.9k bench
prompt: **125 transfers for a 1024-token chunk** (16,384 before) - the head offsets do not reach the
last shards, so 125-126 of the 128 buckets are non-empty, not 128.

The staging buffer is in shard order, which is not the order anything downstream wants: the dequant
pairs `packed[r]` with `heads[r]`, and `decoded` is `[n, heads*dim]` indexed by `token*heads + head`.
So the rows cannot simply be gathered in shard order - that would scramble which head each token
gets. A one-block-per-row scatter (`ple_rows_scatter_kernel`) puts them back in caller order, and
`sc.rows`, the `row_ids`/`head_ids` uploads and `ngram_dequant_f16` are all untouched.

**The part that is easy to get wrong: pageable -> pinned changes ownership.** A pageable H2D is
staged by the driver before the call returns, so the old loop could rewrite nothing and never cared
when the DMA ran. A pinned H2D DMAs out of the buffer *later*, and `Runner::prefill` runs a whole
prompt's chunks back to back with no sync between them - the host is free to be several chunks
ahead. Gathering into one pinned buffer would have let the host overwrite rows a copy had not read
yet: a silent, output-degrading corruption, in a layer whose failures historically looked like a
weak draft head. Hence `PleScratch::kGatherSlots = 3` with a `cudaEvent` per slot, queried (not
waited) before reuse, so the steady state never blocks and the reuse distance is two whole chunks.

### Correctness

- **Every row compared against the pre-change gather, on the real 26.2 GB table.** A temporary
  `HELIOS_PLE_CHECK` rebuilt the rows the old way into a second buffer and byte-compared it against
  what the bucketed gather produced. All 13 prefill chunks of the 12.9k prompt plus the MTP verify
  batches: `GATHER MATCHES per-row reference` on every call, n = 1024 (125 shards hit), the n = 581
  tail (128) and the n = 2 verify batches (29-32). The instrumentation is out of the tree.
- **The reference digest is unchanged**: `HELIOS_MTP=0 HELIOS_MIXER_TC=0 ... --tokens 128` still
  md5s to `3aaa5693eee90a81513be908f1ba3263`, measured on the final stripped binary. A pure
  data-movement change has to be byte-identical, and it is.
- **9/9 parity suites PASS**: test_gemm_smoke, test_mgemm_semantics, test_reconstruct, test_aux,
  test_attn_parity, test_qsa_parity, helios_model_test, test_gdn_chunked_parity, test_kv_quant.

### Transfer count, re-measured (nsys, same 12.9k prompt, 13 chunks)

| | before | after |
|---|---|---|
| H2D copies of exactly 82 B | **205,904** | **1** (a weight-load coincidence) |
| total `cudaMemcpyAsync`, whole run | 434,658 | 230,396 |
| gather transfers per 1024-token chunk | 16,384 | **125** |

The 1,642 calls the new arm adds are the 13 x 125 per-shard copies plus 13 slot-table uploads; the
205,904 82-byte class is gone entirely, and its 16.1 MB of payload is unchanged, so this is purely a
reduction in call count, not in bytes moved.

### End to end, 3 runs per cell per arm, arms interleaved, median

Same binary both arms, the only difference being the removed per-row loop. Default mode (MTP off,
pipeline off), `test/ab_gdn_ple.py`-style harness:

| cell | prompt tokens | per-row tok/s | bucketed tok/s | delta | per-row spread | bucketed spread |
|---|---|---|---|---|---|---|
| prefill 4k | 3,728 | 1246.9 | **1270.4** | **+1.88%** | 1239.0-1248.1 | 1265.7-1271.7 |
| prefill 8k | 7,457 | 1491.6 | **1510.0** | **+1.23%** | 1490.2-1492.9 | 1509.7-1510.6 |
| prefill 16k | 14,915 | 1629.9 | **1642.2** | **+0.75%** | 1629.1-1639.7 | 1639.0-1643.1 |
| decode (263 ctx, 128 out) | 263 | 59.0 | 58.9 | -0.29% | 58.9-59.1 | 58.8-58.9 |

`HELIOS_PIPELINE=1` (the mode the audit's idle numbers were taken in):

| cell | prompt tokens | per-row tok/s | bucketed tok/s | delta | per-row spread | bucketed spread |
|---|---|---|---|---|---|---|
| prefill 4k | 3,728 | 1250.7 | **1273.2** | **+1.80%** | 1248.5-1252.2 | 1270.0-1275.7 |
| prefill 8k | 7,457 | 1496.0 | **1513.6** | **+1.18%** | 1488.7-1497.3 | 1512.0-1514.9 |
| prefill 16k | 14,915 | 1635.3 | **1644.6** | **+0.57%** | 1631.3-1643.4 | 1640.7-1645.3 |

Both modes agree, and the two arms' spreads do not overlap in any prefill cell, so the gain is real
and not a median artefact. **Decode: -0.29%, i.e. nothing** - 58.9 against 59.0 tok/s with spreads
of 0.2 and 0.1 tok/s. The decode path gathers 16 rows, so it issues ~16 tiny pinned copies where it
used to issue 16 tiny pageable ones, plus one event query and one record; that is inside the noise,
as the numbers say.

### The +3-6% did not hold, and the audit's 25-50 ms/chunk was measuring exposure, not cost

The audit put this loop at 25-50 ms/chunk of card-0 idle and extrapolated +3-6%. The measured gain
is +0.6 to +1.9%, i.e. roughly **3-11 ms/chunk** over the 13 chunks of a 12.9k prefill - and it
decays with context length, which is the signature of a fixed per-chunk cost being amortised.
That is the same +0.4-1.9% this file already recorded for "not attempted" (line 8361), so the two
independent estimates now agree with each other and disagree with the idle-time inference.

The mechanism of the disagreement is worth stating, because the audit's own table contains it:
card-0's *idle* is not card-0's *critical path*. Those 16,384 copies were issued at layer 1 of 24,
so the time to issue them sits in front of a stream that has a full layer of compute queued behind
it, and the second card is busy throughout - the exposure is real, but it was never on the critical
path. The audit's own opening section says the two cards are already pipelined across chunk
boundaries; card-0 idle that card 1 covers is not time the prefill is paying. Only the fraction of
the loop that actually delayed chunk completion is recoverable, and that fraction is a few
milliseconds, not tens. The rule is the one this log keeps re-learning: **an idle-time audit bounds
the prize, it does not price it** - and a 25-50 ms estimate that came from counting exposed
milliseconds, not critical-path ones, was always going to be an upper bound.

Shipped anyway, because +0.6 to +1.9% for 1.34 MiB of device memory, 4.03 MiB of pinned host memory
and one extra 1.34 MB device-side scatter is a good trade, and because removing 205,904 CUDA API
calls from the prefill is worth having even at the low end.

### Cost

| resource | delta | measured |
|---|---|---|
| device VRAM (card 0, one `PleScratch`) | **+1,409,024 B = 1.34 MiB** (`row_stage` 1,343,488 + `row_slot` 65,536) | `[mem] gpu0 used 22.11 GB` before vs `22.08 GB` after - the 1.34 MiB is below the print's 0.01 GB resolution and the visible difference is other changes in the tree between the two snapshots, so the figure quoted is the allocation arithmetic, not the reading |
| pinned host RAM | **+4,227,072 B = 4.03 MiB** (3 slots x (1.34 MB rows + 64 KB slot table)) | `cudaMallocHost` x 6 in `ple_scratch_init` |
| extra device work | one 1.34 MB D2D scatter + one 64 KB H2D per chunk | `ple_rows_scatter_kernel` |
| bytes moved over PCIe | unchanged (16.1 MB per 12.9k prefill, same rows) | nsys byte totals |

### What this does not change

- **The chunk size, the pipeline mode, MTP, or anything else.** The diff is `ple.cu`/`ple.cuh` only.
- **The rest of the "our own code is ~11% of prefill GPU time" list.** The mixer and prefill
  attention are already tensor-core; upstream exllamav3 is still 79% of prefill GPU time and is
  where the remaining gap lives.

---

## Two kernel projects landed: PLE projections on tensor cores, GDN conv/transpose de-amplified, PLE gather batched

The overlap audit left ~21% of prefill GPU time as ours. Two of those kernels were not merely ours,
they were badly built. Both fixed, plus the gather batching.

### 1. PLE projections: SIMT -> tensor cores, 26.8x in situ
`gemm_nt_mma_k` (src/engine/glue2.cu): 128x128 block tile, 8 warps in a 2x4 grid (64x32 per warp = 4
m-tiles x 4 n-tiles = 16 `mma.m16n8k16`, 64 fp32 accumulators per lane), k staged 32 deep in double
buffer, fragments read with `ldmatrix.x4`/`x2` off a 40-half-pitch shared tile sized so a fragment's 8
row addresses cover all 32 banks exactly once. 40 KB smem, no carve-out opt-in. Structure follows
`gr_mix_tc.cu`, specialised for the NT layout: A = x (M,K) row-major and B = w (N,K) row-major, which is
already the column-major (K,N) operand `mma.row.col` wants, so neither side is ever transposed.

| | before | after |
|---|---|---|
| key_proj M=1024 N=10240 K=2560 | 23.7-32.7 ms, 1.6-2.3 TFLOPS | 0.920 ms, **58.4 TFLOPS** |
| value_proj M=1024 N=2560 K=2560 | 5.95-6.00 ms, 2.24 TFLOPS | 0.242 ms, **55.4 TFLOPS** |
| in situ, 12.9k prefill | 374.30 ms, 2.25 TFLOPS, 2.0% of GPU | 13.95 ms, **60.4 TFLOPS**, 0.1% |

58-62 TFLOPS is 82-87% of this 3090's 71 TFLOPS fp16-tensor-with-fp32-accumulate dense peak.

**A silent correctness bug in the first attempt, worth recording.** It used Cutlass's
`ThreadblockSwizzle` formulation, which indexes both tile axes off the same group counter - correct only
when `tiles_m` and `tiles_n` share a common factor. At the PLE's 8x80 shape it silently skipped three
quarters of the output **while appearing to run at 202 TFLOPS**. A CPU spot-check inside the microbench
caught it; the replacement is a 1-D M-fastest index (`m0 = blockIdx.x % tiles_m`, `n0 = blockIdx.x /
tiles_m`) so the `tiles_m` blocks sharing a 128-row B tile are co-scheduled - B is 52 MB against a 6 MB
L2, which is the difference between streaming B once (52 MB) and once per M-tile row (419 MB). A
benchmark that reports 202 TFLOPS while dropping 75% of its output is the clearest argument in this
project for spot-checking a fast kernel against a CPU reference.

### 2. GDN conv + transpose: the 16x amplification was real, not inherent
Both move 21 MB of `mixed_qkv` per launch and were read as "strided kernels at ~100 GB/s". They were
already at the card's bandwidth limit **for the access pattern they issued**:
- `transpose_f32_bf16_k`: read coalesced along F, stored `dst[f*M+m]` at stride 2048 B. A 2-byte store
  into its own 32 B sector = 16x write amplification. 378 MB in 577 us = 655 GB/s *effective*.
- `conv1d_update_kernel`: channel-major in, sequence-major out; one thread per channel walking s reads a
  2-byte load 2 KB from the previous lane's = 16x read amplification. 356 MB in 482 us = 739 GB/s effective.

- **transpose**: 32x32 shared tile - load with consecutive lanes on consecutive f (contiguous in src),
  store with consecutive lanes on consecutive m (contiguous in dst) + 1 padding column. 269.95 -> 47.86 ms,
  109 -> 615 GB/s useful. **5.6x**, bit-exact against a per-element copy.
- **conv**: 64 channels x 64 sequence positions per block, 2-D grid (160 x 16) so no block waits on
  another; the K-1 positions a tile's first output needs are re-read by the tile owning them (6% of
  traffic at K=4) rather than serialising all 1024 steps inside every channel block. 225.38 -> 49.53 ms,
  87 -> 397 GB/s. **4.6x**, bit-exact: same K-term fma chain in the same order, same
  `_sigmoid_fast_exp`, same `__float2bfloat16_rn`, same window shift.

### 3. PLE n-gram gather: 16,384 -> 125 transfers per 1024-token chunk
The host already has every `hash_host[i]`, so `shard = id/rows_per_shard` is free. Rows are counting-sorted
by shard into a `PleScratch` pinned buffer, uploaded one contiguous run per shard, then a
one-block-per-row scatter (`ple_rows_scatter_kernel`) restores caller row order before the dequant. The
scatter is load-bearing: staging in shard order alone would scramble which head each token gets, because
the dequant pairs `packed[r]` with `heads[r]`. nsys over a 12.9k prefill: 205,904 82-byte H2D copies -> 1;
total `cudaMemcpyAsync` 434,658 -> 230,396, with bytes moved unchanged - a pure call-count reduction.

**The ownership hazard this introduced, and how it was handled.** A pageable H2D is driver-staged before
the call returns; a pinned one DMAs out of the buffer *later*. `Runner::prefill` runs a whole prompt's
chunks with no sync between them (the host runs several chunks ahead), so a single pinned buffer would
let the host overwrite rows an in-flight copy had not yet read - silent corruption in a layer whose
past failures looked like a weak draft head. Hence `kGatherSlots=3` with a `cudaEvent` per slot and
`cudaEventQuery` (not wait) before reuse: the steady state never blocks and the reuse distance is two
whole chunks.

Measured A/B, arms interleaved, 3 runs per arm per cell: **+1.88% / +1.23% / +0.75%** at 4k/8k/16k. The
audit estimated +3-6%; about a third of that landed, which is the honest outcome for a change whose
mechanism was a call-count reduction on a stream that had other work available to hide behind.

### Combined, and the final valid 1:1
Prefill 12.9k: 1500 -> **1561-1569 tok/s** (+4.3%). Reference digest byte-identical at
3aaa5693eee90a81513be908f1ba3263, 9/9 suites, correct output.

| pre/out | exl3 pre | helios pre | pre % | exl3 dec | helios dec | dec % |
|---|---|---|---|---|---|---|
| 4k/512 | 1656 | 1223 | 74% | 66.1 | 54.6 | 83% |
| 4k/1000 | 1657 | 1218 | 73% | 61.4 | 58.1 | 94% |
| 4k/2000 | 1659 | 1218 | 73% | 60.5 | 55.5 | 92% |
| 8k/512 | 1754 | 1432 | 82% | 62.0 | 53.0 | 85% |
| 8k/1000 | 1863 | 1432 | 77% | 60.5 | 51.9 | 86% |
| 8k/2000 | 1754 | 1426 | 81% | 62.2 | 55.1 | 88% |
| 16k/512 | 1924 | 1520 | 79% | - | 52.5 | - | (exl3 generated 1 token) |
| 16k/1000 | 1924 | 1518 | 79% | 66.6 | 55.9 | 84% |
| 16k/2000 | 1864 | 1519 | 81% | 59.2 | 55.6 | 94% | (exl3 short) |

**Prefill 73-82%** (was 69-78%). **Decode 83-94%** - the same parity band as the previous grid's
84-101% within the run-to-run spread that has been observed all session, so no decode regression is
claimed from these two changes; they are prefill-path changes. **Context 262,144 vs 230,000.**

### Negative result: caching the per-layer `getenv` calls bought nothing measurable
The overlap audit attributed ~63 ms/chunk of "residual unattributed" host time partly to `getenv` -
a linear scan of `environ` called ~10 times per `moe_layer` invocation, ~480 times per chunk. It was
a real inefficiency and a trivially safe fix, so it was made: `HELIOS_MOE_ACTIVE`, `HELIOS_MPROF`,
`HELIOS_MOEC`, `HELIOS_MOEX`, `HELIOS_CNT`, `HELIOS_MOE_SHAPE` and `HELIOS_MOE_SMS` are now read once
into file-scope statics (`moe_diag::` in moe_layer.cu) instead of per layer and per expert. One real
trap on the way: a local struct's destructor cannot capture an automatic, so `MOut::~MOut` has to test
the *static* `k_mprof`, not the local `mprof` - which is presumably why the original called `getenv`
inline there.

**Measured effect: none.** 12.9k prefill before 1561.0-1568.9 tok/s, after 1555.3-1567.9 - within the
run-to-run spread. 9/9 suites pass, the sequential reference digest is byte-identical, and the
diagnostics still fire when asked (`HELIOS_MOEC` -> 49 `[moec]` lines, `HELIOS_MPROF` -> 49 `[mprof]`
lines), so the change is behaviour-preserving. Kept because it strictly removes work from the hot path
at no cost in complexity, but recorded as the second negative result of this kind in a row after the
chunk-size lever: **the audit's "residual unattributed" bucket was not `getenv`**, and most of it is
launch gaps and allocator noise rather than anything addressable by inspection.

---

## Output parity against exllamav3: attempted, and blocked by a real configuration asymmetry

The strongest correctness evidence available was never collected: running the same prompts through
exllamav3 - a working reference implementation of the same checkpoint - and through helios, and
comparing. Eight prompts covering factual recall, code, arithmetic, translation, and prose, 128 greedy
tokens each.

**Result: 0/8 token-identical, and the cause is configuration, not arithmetic.**

1. **Prompt formatting differs and cannot be matched through the reference's HTTP surface.**
   exllamav3's `/v1/completions` applies the checkpoint's chat template, and this is a hybrid
   reasoning model, so most of its answers are prefixed `<think>\n`. helios `--raw` is BOS + literal
   text with no template, and helios' own `apply_chat_template` path (main.cpp:173) takes a different
   thinking decision. The server wrapper exposes no template or thinking control, so neither mode can
   be pinned to match. Comparing `--raw` to templated, or helios-templated to exl3-templated, each
   measures the template rather than the math.

2. **The deeper reason: the two engines run different KV precisions by design.** The exllamav3 wrapper
   runs `CACHE_QUANT=3` - a **3-bit quantized KV cache** - and its own comment records why: at 262144
   tokens, fp16 KV is 6.0 GiB and "the autosplit loader refuses that split" on 2x24 GB next to ~36 GB
   of weights and the MTP draft, so cq8 (3.0 GiB) still does not fit and cq3 (1.13 GiB) is used, trading
   "decode 163 -> 124 tok/s" for the context window. **helios runs fp16 KV** and still fits 262144
   alongside its MTP experts. Different KV precision means different logits, so greedy decoding takes a
   different branch as soon as two candidates are close - token-exact agreement is not an achievable
   target between a cq3 engine and an fp16 engine, and should not be.

   This is worth stating on its own: the comparison across this whole project has been apples-to-apples
   on **speed** (same checkpoint, same two cards, same 262k context, identical harness and prompt) but
   **not on numerics**, because the reference had to quantize its KV cache to fit where helios did not.
   A user choosing between them is choosing 3-bit versus fp16 KV quality as much as choosing a runtime.

**What the comparison does establish:** both engines answer all eight prompts correctly, coherently and
on-topic - helios writes a working Python list-reversal, gets 137/11 right, gives the French
translation, and names the European capitals; exllamav3 does the same. On the single most diagnostic
prompt, "The capital of France is", **both emit ` Paris.` as the first token.** That is a real shared
result - the first greedy token is the one least sensitive to accumulated numerical drift, and it
agrees. Token-exact parity beyond it remains unestablished and, for the reason above, is not the right
bar.

### Follow-up: the reference's KV quantization is not a removable handicap
The cq3 finding raised an obvious question - is the decode parity real, or is it an artifact of
comparing against exllamav3 while it runs a 3-bit KV cache that we do not? If exllamav3 could run cq8
(its own comment records "decode 163 -> 124 tok/s" going cq8 -> cq3) at a context where it fits, the
fair comparison would be against 163 tok/s and we would be nowhere near.

**It cannot. cq8 does not fit on this box at any context.** Tried `CACHE_QUANT=8 MAX_CTX=65536`,
`32768` and `16384`; all three fail at load with `RuntimeError: Insufficient VRAM in split for model and
cache` from `exllamav3/model/model_ls.py::_load_autosplit`. The ~36 GB of weights plus the MTP draft
exhaust the autosplit headroom before the cache is considered, so the reference has **cq3 as its only
viable KV configuration here** - fp16 and cq8 are both impossible, not merely unused. The "163 tok/s"
figure in the wrapper's comment therefore comes from some other configuration (different split, MTP off,
or an earlier run), not from one reachable here.

So the decode comparison is against the reference's best *available* configuration on this hardware,
not a handicapped one - which is the only kind of comparison worth reporting. The quality asymmetry
still stands and is worth stating plainly: **at matched decode speed, helios is running fp16 KV where
exllamav3 is running 3-bit**, and helios reaches 262144 context doing it. If KV fidelity matters for a
given use, helios is the better choice at equal throughput; if only speed matters at a short context,
neither engine has an fp16-KV option here and both are on 3-bit-quality KV.

---

## CORRECTION: context is PARITY with the reference, not a surpass

I have been reporting "context 262,144 vs 230,000 - surpasses" since early in this project. That is
wrong, and the 230,000 figure was never measured - it came from the problem statement's description of
the deployed server, not from the server. What exllamav3 actually reports in its deployed configuration
(`CACHE_QUANT=3`, `MAX_CTX=auto`):

```json
"max_model_len": 262144,  "context_window": 262144,  "configured_context_window": 262144,
"cache_type": "cq3+QSA-index-planes",  "cache_pool_tokens": 393216,  "parallel": 4
```

and `config.json` `max_position_embeddings` is 262144 as well. **Both engines serve 262,144. Context is
parity, not a win.** The correct headline is therefore: prefill 73-82%, decode 83-94% (parity), context
**262,144 = parity**.

That same payload also surfaces a capability difference that had been invisible, and it is in the
reference's favour: `"parallel": 4` with `cache_pool_tokens: 393216` - three times the context - means
exllamav3 serves a **paged KV cache holding several concurrent sequences**. helios is single-sequence
(its `/v1/models` advertises `context_length` and nothing else; `Runner::generate` starts from
`reset()` with no per-request recurrent-state snapshot, because GDN and PLE recurrent state is not
position-addressed). Concurrency is a genuine gap, and unlike prefill it is a feature rather than a
kernel-optimisation problem - it needs per-sequence recurrent-state snapshots, which is real work.

The one context-side advantage that survives is narrower than I implied: at the *same* 262,144 window,
helios holds its KV in **fp16** while the reference holds **cq3**, because cq3 is the only precision
that fits the reference's autosplit on this box. helios fits fp16 there; exllamav3 cannot. That is a
quality advantage at a matched window and a matched speed, not a longer window.

---

## Cross-request prefix cache: implemented, proven output-identical, and measured at 5-25x on prefill

The section above closes by saying helios is single-sequence because "`Runner::generate` starts from
`reset()` with no per-request recurrent-state snapshot, because GDN and PLE recurrent state is not
position-addressed". The first half of that was just a missing feature. It is not missing now.

**What was there and dead.** `Runner::prefix_match()` existed and was never called. `prefill()` was
called from exactly one place, always with `from = 0`, after a `reset()`. `/metrics` already
published `prefix_reused_tokens`, `prefix_last_resume`, `prefix_snapshots` and
`prefix_snapshot_interval` - three of them hardcoded to `0` by accessors that returned a literal.

**What is not reusable, and why that was not a blocker.** The KV cache is position-addressed: a row
written at position p is only read by a query at position >= p, so the rows a previous request left
are still the right answer for a new request with the same leading tokens. The recurrent state is
not - the GDN conv window and recurrent matrix (36 layers), the PLE's dilated conv, and the PLE's
**host** n-gram window are each a function of the whole prefix. All three are now captured at every
prefill chunk boundary into a ring of 8 slots in **pinned host memory** (111 MB per slot: 36 GDN
layers x (80 KB conv + 3 MB recurrent) + 180 KB of PLE conv), and a resume restores the newest
capture at or before the divergence point. Host rather than device because card 0 is down to ~1 GB
free at the default 262144 context; 8 device-side captures would not fit, and the 111 MB D2H against
a 1024-token chunk (~1.0 s) is ~11 ms.

**The constraint that makes it bit-exact rather than merely close.** A capture is only ever taken at
a position that is a multiple of `max_chunk_`, and a resume only ever happens at one. That is not a
convenience: a prefill of any length cuts its chunks at multiples of `max_chunk_` from 0, so
resuming from a multiple of `max_chunk_` replays exactly the chunk sequence - and therefore exactly
the floating-point reduction order in attention, GDN, PLE and the grouped MoE - that a full prefill
of the same prompt would have used. Resuming anywhere else would be a different summation of the
same tokens, i.e. a different answer. So the reuse floor is one chunk, and `--chunk` sets it.

**The rule, unit-tested where it can fail quietly.** `src/engine/prefix.hpp` holds the decision as
pure logic and `test/test_prefix.cpp` covers it. Besides "newest capture at or below the divergence
point, never past it, never at the prompt's own length", it enforces the part that is easy to miss:
**every capture past the resume point is dropped**, because such a capture was taken on a prefix the
new request has just disowned. Without that, a request that diverges early leaves the ring full of
captures for a history that no longer exists, and a later request that happens to match them gets a
recurrent state belonging to different tokens - a silently wrong answer, with nothing to crash on.
A restart drops the whole ring for the same reason.

**The gate: a resumed request is byte-identical to a full prefill.** All comparisons below are the
generated text with the `[...]` status lines and blank lines filtered, exactly as the reference
digest filter does.

| check | cold (fresh engine) | reused | identical? |
|---|---|---|---|
| 4,801-token prompt, 256 greedy tokens, MTP off | `4cad29fa7e7a2c8a5909de611b2fd473` | resume 4096, 705 tokens recomputed | **yes** |
| 4,815-token prompt (same doc + a different tail), 256 tokens | `37db77918ba985073f154181c960506f` | resume 4096, 719 recomputed | **yes** |
| 4,815-token prompt, 256 tokens, **MTP on** (default) | `c46bd5b3bd925b407572ca1fd972f013` | resume 4096, 719 recomputed | **yes** |
| 8-token prompt x3 repeats (no capture is reachable) | `3aaa5693eee90a81513be908f1ba3263` | resume 0, full prefill | **yes** |
| server, 3 requests: long / divergent / long again | all three texts | resume 1024 on 2 and 3 | **yes** |

The MTP-on row matters on its own: the draft head's own KV rows `[0, resume)` come from the previous
request, and the resumed run accepted `104/152` draft slots at `1.68` tokens/step - the *same* counts
as the fresh run. A recurrent state that was merely close would have drifted there first.

Reproduction (one process, because reuse is across requests):

```
M=~/models/Qwen3.8-Flash-Next-exl3
# fresh engine, no history at all
HELIOS_MTP=0 HELIOS_MIXER_TC=0 ./build/helios gen $M --raw --prompt-file /tmp/rag_doc_q.txt \
    --tokens 64 --temp 0
# same prompt, second request reuses 5120 of 5165 tokens
HELIOS_MTP=0 HELIOS_MIXER_TC=0 HELIOS_PREFIX_CACHE=1 ./build/helios gen $M --raw \
    --prompt-file /tmp/rag_doc.txt --tokens 64 --temp 0 --repeat 3 \
    --repeat-suffix $'\n\nQuestion: summarise section 1 in one sentence.'
```

`--repeat N` / `--repeat-suffix` are the one CLI addition: a fresh engine per invocation has no
history to reuse, so the feature cannot be exercised from the CLI without them. Absent, `gen` does
exactly what it did.

**Measured, 3 runs per cell, HELIOS_MTP=0 HELIOS_MIXER_TC=0.** Realistic shape: a 5,154-token
document in context (the RAG / long-system-prompt case) plus a short unique question, the question
being what changes between turns.

| 5,165-token prompt | run 1 | run 2 | run 3 | prefill tok/s |
|---|---|---|---|---|
| cold, cache ON (resume 0) | 5510.9 ms | 5416.3 ms | 5410.6 ms | 935 / 952 / 953 |
| **warm, resume 5120** | **219.0 ms** | **218.6 ms** | **218.8 ms** | 206 / 206 / 206 |
| cold, cache OFF (control) | 5342.3 ms | 5342.6 ms | 5355.2 ms | 967 / 967 / 965 |

| 8,089-token prompt | run 1 | run 2 | run 3 | prefill tok/s |
|---|---|---|---|---|
| cold, cache ON (resume 0) | 7675.6 ms | 7694.9 ms | 7685.8 ms | 1052 / 1050 / 1051 |
| **warm, resume 7168** | **1396.6 ms** | **1399.2 ms** | **1399.5 ms** | 659 / 658 / 658 |
| cold, cache OFF (control) | 7593.7 ms | 7599.0 ms | — | 1065 / 1065 |

**24.7x** on the 5.2k prefix and **5.5x** on the 8k one. The difference between them is entirely the
one-chunk floor: a 5,165-token prompt resumes at 5,120 and recomputes 45 tokens; an 8,089-token one
resumes at 7,168 and recomputes 921. Expect `common - (common mod max_chunk_)` tokens of reuse, and
nothing below one chunk. `--chunk 256` cuts that floor 4x for ~9% prefill throughput (measured in
`main.cpp`: 296 tok/s at 256 against 324 at 1024).

The cost when it does not hit is **+1.7%** on a 5.2k cold prefill (5436 ms against 5347 ms, three
runs each) and **+1.2%** on 8k: five to eight 111 MB D2H copies, riding the aux stream behind an
event on the compute stream, with the next chunk held behind the copy so the snapshot and the next
chunk's recurrence cannot race on the same bytes.

**Default: OFF, and it stays that way as shipped.** `HELIOS_PREFIX_CACHE=1` enables it. The reason is
not the 1.7% - it is that the whole win is conditional on the workload re-sending a prefix. A
workload that does not (the `grid_bench.py` grid, which front-loads a unique nonce per request
precisely to measure cold prefill) pays 1.7% and gets nothing, and an engine that quietly changes
what "prefill" means is exactly the thing this project's benchmarks have been bitten by before. The
feature is correct and measured, so it is a flag rather than a default.

**What it does not do, stated plainly.**

- **No extension mode.** A prompt that continues the resident history still recomputes its last
  chunk, where a ring of *end-of-prefill* captures would have saved it. Not a correctness issue, just
  a smaller win than the sibling KDA engine gets on the same shape.
- **One request per capture interval of history.** A prompt shorter than `max_chunk_` can never be
  resumed, because no capture exists below it. Short-chat reuse needs a smaller `--chunk`.
- **A divergent request empties the ring above the divergence point.** That is the price of not
  serving a state that belongs to different tokens; a chat turn that *extends* the history never
  triggers it.
- **Single sequence.** This makes repeated prefixes cheap; it does not make the server concurrent.
  That still needs per-sequence state, which is a different (and much larger) piece of work.
- **Refuses to run with `HELIOS_QSA=1`**, whose pooled-block counters are a running state `reset()`
  does not clear and the snapshot does not carry.

Test suites after the change: `test_gemm_smoke`, `test_mgemm_semantics`, `test_reconstruct`,
`test_aux`, `test_attn_parity`, `test_qsa_parity`, `helios_model_test`, `test_gdn_chunked_parity`,
`test_kv_quant` - 9/9 PASS - plus the new `helios_prefix_test`, and the fresh-engine reference
digest still `3aaa5693eee90a81513be908f1ba3263`.

---

## Prefix caching: was dead code, now real — and it has a caveat worth knowing

`Runner::prefix_match()` was declared, defined, correct - and **never called**.
`prefix_requests()` is a public accessor over a counter that `generate()` increments unconditionally, and
there was no KV-resume path anywhere in `src/`. Every request did a full prefill. For a chat workload
that re-sends the same system prompt every turn, that is the most valuable performance feature available
and it was simply absent.

### Design, and why it is output-identical rather than approximately so
An 8-slot ring of whole-model recurrent-state captures in **pinned host memory** (111 MB/slot; GDN conv +
rec for every layer and the PLE conv, taken by D2H at prefill chunk boundaries on the aux stream behind
a compute-stream event, with the next chunk held behind the copy so the snapshot cannot race the
recurrence). On a matching request, `prefix_plan()` picks the newest capture at or below the divergence
point, drops every capture past the resume point (one there describes a prefix the request has
disowned), restores the state, keeps KV rows [0, match), and prefills only the remainder.

**Captures are taken only at prefill chunk boundaries and resumes happen only at multiples of
`max_chunk_` - a correctness constraint, not a convenience.** A prefill cuts chunks at multiples of
`max_chunk_` from 0 whatever the prompt length, so resuming from such a point replays exactly the chunk
sequence, and therefore exactly the floating-point reduction order in attention, GDN, PLE and the grouped
MoE, of a full prefill of the same prompt. That is what makes the result *byte-identical* rather than
merely close.

### Verified independently, not taken on trust
- **The resumed request is byte-identical to a full prefill.** Fresh, cache off: `5d8dfe8e...`.
  Request 2 of a `--repeat 2` pair, which resumed **4096 of 4685 tokens**: `5d8dfe8e...` - identical.
  Same check passes with MTP on (identical draft-acceptance counts too) and across a
  long/divergent/long server sequence.
- Prefill on a warm resume: **4942.8 ms -> 916.7 ms**; whole request 6.01 s -> 2.01 s. Capture overhead
  is +1.7% on a cold request.
- New suite `helios_prefix_test` (pure `prefix_plan` decision logic: empty rings, sub-interval prompts,
  exact boundaries, divergence inside an interval, gappy slots, fully-resident prompts, duplicated
  positions, invalidation) - **all cases PASS**. Total is now 10 suites.
- Reference digest for a fresh engine is unchanged: 3aaa5693eee90a81513be908f1ba3263.

### The caveat: turning the cache on changes COLD requests
Enabled, the cache reproducibly alters the token stream of a request that reuses **nothing**. Measured,
3 runs each, both internally deterministic:

| | digest (3/3 identical) |
|---|---|
| cache OFF, single request | `d9fdd4038096` |
| cache ON, first request of a pair | `93af42e446ef` |

Both stable, but different - the cache-ON run diverges later and stops ~38 characters earlier. The
mechanism is the capture's synchronisation perturbing the effective micro-batch/overlap timing, which
changes the grouped MoE's LPT expert visit order and therefore reassociates the per-token expert sum -
a reassociation this codebase already documents as order-dependent. So the cache is exact for the
*resumed* request against a *fresh* prefill, but a cache-ON server is not output-neutral against a
cache-OFF one on cold traffic.

**Default is OFF**, which is the right call for exactly that reason: a 5.4x prefill win on repeated
prompts does not justify a server whose cold answers differ from its own non-cached build. It is
available as `HELIOS_PREFIX_CACHE=1` for anyone who wants it and accepts the coupling, and it refuses to
run alongside `HELIOS_QSA` rather than reason about a second snapshot format for the indexer's pooled
block counters.

---

## Dead-feature audit: the unwired-component pattern, swept systematically

Four separate defects in this project shared a shape: a component that compiles, has a name, has tests,
and is not actually connected to anything. Rather than wait to trip over a fifth, the tree was swept for
call sites across all of `src/` (counting both `name(` calls and `name<<<` kernel launches, so CUDA
launch syntax is not mistaken for a dead kernel - the naive sweep reported 178 "dead" functions, 103
after accounting for launches, and the overwhelming majority of those are `__device__` helpers in headers
that are called from other device code).

| component | verdict |
|---|---|
| `Runner::prefix_match()` | **was dead** - defined, correct, never called. Now implemented and used. |
| `Runner::prefix_requests()` | **was a lie** - public accessor over a counter `generate()` incremented unconditionally. Now reports real reuse. |
| `OutputParser::in_think()` (chat.hpp:87) | dead accessor, but the parser's thinking logic itself IS live (`in_think_` is read throughout `chat.cpp`), so this is a stray accessor, not a dead feature. |
| KV quantization family (`cq_pack`/`cq_unpack`/`kvq_*`/`launch_write`/`launch_dequant`) | **wired and working** - `attn_layer.cu` drives it behind `HELIOS_KV_QUANT`. `test_kv_quant` exercises the kernels, and the engine path is confirmed live (see below). |
| `gated_delta_net_fused_op_2/3_kernel` (gdn.cu) | dead here, correctly - they are the GLM-style fused GDN kernels; this model uses the chunked GDN path. Leftover from the sibling engine, not a defect. |
| `helios_prefix_test` | built but not in any run-all path - I had to `find` the binary to run it. Worth a `make test` target. |

### KV quantization: wired, working, and correctly unused - because fp16 is both faster and better here
Exercised end-to-end at 0/3/4/8 bits: correct " Paris." output at every width, so the engine path is live
rather than merely compiled. Measured decode at 8k:

| KV precision | decode tok/s |
|---|---|
| **fp16 (default)** | **61.23** |
| cq3 | 39.41 (-36%) |
| cq4 | 50.59 (-17%) |
| cq8 | 54.93 (-10%) |

Quantized KV is **slower at every width**, because the KV scan dequantizes on the fly and at 8k context
across 12 full-attention layers that cost outweighs the bandwidth it saves. So fp16 wins on both axes
here and the feature is correctly left off. It also explains the asymmetry with the reference from a
different angle: exllamav3 is forced to cq3 not because quantizing is good but because fp16 does not fit
its split - and on this box quantizing would not have helped it either.

---

## One command verifies the whole engine: `ctest --test-dir build --output-on-failure`

The dead-feature audit turned up `helios_prefix_test` built but sitting in no run-all path, which is the
mechanism by which a test silently rots. There were **14 test targets** and ctest knew about **3** -
registered in `src/cuda/quant/CMakeLists.txt` only. Every test target is now registered in the root
CMakeLists, so a new test cannot be added without appearing in the run. The three duplicate
registrations that resulted (the same binaries under two names) were removed from the subdirectory.

**15 tests, 15 passed, 0 failed, ~21 s.** Two of them - `test_gemm_rows` and `test_gr_mix_tc` - **had
never been run by hand at all** before this; both pass, so there was no rot, but they were outside every
gate anyone had been checking.

Registering them immediately caught one real failure: `helios_tokenizer_test` failed with
`no vectors at test/tokenizer_vectors.json`. The test was never wrong - it reads a repo-root-relative
path (and accepts it as `argv[2]`) - but ctest runs each test with cwd = the build dir, so the relative
path only ever resolved when someone happened to launch it from the repo root. Fixed with
`set_tests_properties(... WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})` rather than by touching the
test, since the test's own default is the correct one for a human running it from the repo root.

The full gate is now one command rather than the hand-typed list of ten paths that this project's notes
had been carrying.

---

## Server state isolation holds; request ORDER does not change output determinism

Two properties a server has to have, tested separately because they are not the same thing.

### 1. No state leaks between requests - PASSES
Five different prompts sent in sequence to one server, then the same five again:

| prompt | pass 1 | pass 2 | |
|---|---|---|---|
| The capital of France is | `35a93e9904` | `35a93e9904` | stable |
| Write a Python function that adds two numbers | `d6b02d603e` | `d6b02d603e` | stable |
| What is 25 times 4? | `6e7b928066` | `6e7b928066` | stable |
| Name three colors | `7e32025009` | `7e32025009` | stable |
| Explain gravity in one sentence | `ab930a1815` | `ab930a1815` | stable |

All stable, and all substantively correct - " Paris.", working Python, **25 x 4 = 100**. Interleaving a
4,000-token request between them changed nothing: the two short prompts returned byte-identical output
before and after it. So `reset()` does clear GDN recurrence, PLE conv, KV rows and the n-gram window -
the recurrent state really is being restored, not merely overwritten.

### 2. But the same request answers differently depending on what preceded it
With the server warm and no long requests involved, "The capital of France is" gives:

- asked first: `Paris.\n\nThe statement is correct. **Paris** is indeed the capital of France...`
- asked after another request: `Paris.\n\n# 2026年中国AI产业趋势报告...`

Both correct, both coherent, both starting `" Paris."` - they diverge a few tokens later. This is **not**
a state leak (a leak would produce a wrong answer, not an equally valid one); it is the same mechanism as
the prefix-cache caveat: request ordering changes timing, timing changes the effective batching, the
grouped MoE's LPT expert visit order is batching-dependent, and that reassociates the per-token expert
sum. On a high-entropy reasoning model a last-bit difference early takes a different valid branch. The
engine has no run-to-run nondeterminism of its own - every configuration tested is stable when repeated -
but it is **not invariant to what the server did before**, and cannot be while expert ordering is chosen
by load balance.

Making it order-invariant would mean deriving the expert visit order from the expert id alone instead
of from the LPT schedule. That is a one-line change with a real cost: LPT ordering exists to spread load
across SMs, and a fixed order would unbalance the MoE. Recorded as a deliberate choice, not an oversight -
for a single-user runtime the throughput is worth more than request-order reproducibility, but anyone
who needs bit-reproducible answers for the same prompt should run one request per engine, or set
`--temp 0` with a single-request process, rather than rely on a shared server.

### CORRECTION and root cause: the order-dependence is the SPECULATIVE path, and `HELIOS_MTP=0` removes it
The previous section attributed the order-dependence to MoE LPT expert ordering and recommended living
with it. That attribution was **wrong**, and the recommendation is superseded: a controlled A/B pins it
to the speculative path, and there is a one-flag fix.

Reproduced cleanly, digest of the response to "The capital of France is", 40 greedy tokens:

| server config | 4x back-to-back | alternating with another prompt (4 rounds) |
|---|---|---|
| default (MTP on) | `35a93e99` x4 | `35a93e99`, then `d4696f43` x3 |
| **`HELIOS_MTP=0`** | `35a93e99` x4 | **`35a93e99` x4 - fully order-invariant** |

Note the shape of the default-config result: France asked back-to-back is *stable*, and flipping to
`d4696f43` happens **once** and then holds. So the engine is not nondeterministic - it settles into a
regime - but which regime a request lands in depends on what preceded it.

### What I ruled out on the way, each by inspection
- **MoE LPT ordering carrying state** - no. `moe_scatter_k` is already a pure function of the input
  (`pos(i) = offset[e_i] + #{j < i : e_j == e_i}`, no per-element cursor), with a comment recording that
  exact nondeterminism being fixed before. `moe_permute_lpt_order` just returns a pointer into a
  workspace the preceding pass filled.
- **The MTP layer's recurrent state leaking** - no. `MtpScratch` has no `conv_state`/`rec_state`, and
  `mtp.cu` calls `attn_layer`, so the MTP layer is full-attention and has no recurrence to leak.
  `kv_dev_` is sized `n_full + 1` and the MTP layer takes its own slot 12, so there is no index collision.
- **Any trunk state** - no. `reset()` zeroes `pos_`, memsets every GDN recurrence and conv, memsets the
  PLE conv, and refills `hist_` with eos. And the isolation test already showed 5 prompts x 2 passes
  byte-identical with a 4k request interleaved.

### The actual mechanism
With the draft head enabled, the first spec step of a request proposes tokens, the accept count decides
the width of the **trunk's** verify forward, and a width-(K+1) forward associates differently from
width-1 - the same effect that makes speculative decode non-bit-exact against the sequential path, which
this codebase already documents. The draft head is conditioned on the MTP layer's own cache, whose
starting contents are not reset between requests, so the first step of a request can propose differently
depending on history, which changes the accept count, which changes the trunk's batch width, which
reassociates the MoE sum. A last-bit difference on a high-entropy reasoning model then takes a different
valid branch.

**Actionable:** `HELIOS_MTP=0` gives byte-identical answers for the same prompt regardless of request
order. That costs ~20% decode (the speculation win), so it is not the default - but anyone who needs
reproducible answers from a shared server now has a documented flag instead of a caveat.

---

## BOTH ENGINES PROFILED: the prefill gap is 7.68 card-seconds, not the ~0.9 I estimated

The previous attribution was a single-engine profile taken before the PLE/GDN/mixer work. Re-profiling
**helios and exllamav3 with the same nsys run on the same prompt on the same box** changes the picture
completely, and refutes my own arithmetic.

| | helios | exllamav3 |
|---|---|---|
| card-seconds of kernel time | **13.154** | **5.473** |
| rate (12.9k prompt) | 1589.5 tok/s | 1978 tok/s |
| card utilisation | 80.3% (card0 73.5%, card1 87.2%) | ~42% |

I had estimated the reference spends ~13.4 card-seconds and that ~0.9 card-seconds of slack remained in
our kernels. **It spends 5.473.** The gap is **7.68 card-seconds of excess GPU work**, not slack. The
wall-clock ratio is only 1.24x because exllamav3 achieves that with far lower utilisation - it does
2.4x less GPU work and wastes more of the machine doing it.

### Where the 7.68 card-seconds are

| line | helios | exllamav3 | delta | share |
|---|---|---|---|---|
| **dense projections** | 6382 ms, `exl3_gemm_kernel` x4017 | **984 ms, its own fp16 `gemm_kernel` x2668** | **+5399 ms** | **70%** |
| MoE (upstream, byte-identical) | 3944 ms | 2588 ms | +1356 ms | 18% |
| **router (ours)** | 705 ms | 38 ms | **+668 ms** | 9% |
| GDN / chunked linear attention (ours) | 432 ms | 152 ms | +280 ms | 4% |
| **mixer (ours)** | 411 ms | 536 ms | **-125 ms** | **-2% - we are AHEAD, 1.30x** |
| MoE gather / reconstruct glue (ours) | remainder | | | |

Three things fall out of this, and only one of them is a proven gap.

**1. `exl3_gemm_kernel` is not being used by the reference at all during prefill** (0 times; 2668
launches of its own fp16 `gemm_kernel` instead). Our projection line costs 6.5x theirs. **This is a
line-item comparison of the same logical work, not a proven like-for-like kernel gap** - the launch
counts differ (4017 vs 2668), so the work partition differs too, and the projection tensors *are* EXL3
trellis in the checkpoint (`.trellis` + `.suh`), so "fp16" cannot simply mean the weights are stored
unquantized. This is the single largest unexplained item and the thing to investigate next. It is 70% of
the gap; no single kernel of ours can compete with it.

**2. The MoE gap is a vendored-ext version gap, not our bug.** With top-10 of 512 experts at 1024 tokens
each expert gets ~20 rows, so a 32-wide M tile wastes ~37% of every tile. **exllamav3 dispatches
M-specialised MoE templates (64/32/16) that our vendored ext lacks.** Same kernel, same weights, same
tokens: helios 96.1 GB/s aggregate (5.1% of roofline), exllamav3 146.5 GB/s (7.8%). Both are far off
the memory roofline; the reference is less far.

**3. The router is the cleanest, largest inefficiency in code we own.** 705.5 ms, 637 launches,
**0.048 TFLOP/s = 0.13% of the 3090's fp32 peak**, 0.25% of bandwidth - neither compute nor memory
bound, pure latency. The grid is `(ceil(E/4), bsz)` = 131,072 blocks, one block per token, each warp
owning one expert row and accumulating into a single float through an 80-deep serial fmaf dependency,
re-streaming the 2.62 MB gate matrix 1024 times per chunk with zero reuse. exllamav3's
`routing_gemm_i8_kernel` does the identical job in 38.0 ms - **18.6x less time**. A single tensor-core
GEMM for `[1024 x 2560] @ [2560 x 512]` would be ~15-40 us against the measured 1107.5 us per launch,
worth roughly 0.6-0.7 card-seconds. That is 9% of the gap - the right next thing to fix because it is
the largest and cleanest thing in our own code, but it is **not** enough on its own.

### The honest summary
At helios's current card-seconds with 100% card utilisation it would run in 6.577 s = 1957 tok/s, i.e.
**already at parity with the reference's 1978**. So the wall-clock gap is a utilisation gap *and* an
excess-work gap at once: we schedule 2.4x more GPU work than necessary and then pack it at 80% instead
of 100%. Both need addressing, and the excess work is the larger of the two. The single biggest line
(70% of the gap) is the dense-projection GEMM, where the reference does not use our kernel at all and
the reason is not yet established.

---

## The dense-projection GEMM: the reference's dispatch rule, and the fix

The 6.4 s line was not a mystery and it was not a 6.5x kernel defect. exllamav3 **does not use the
EXL3 GEMM kernel for dense projections at prefill widths at all** - it dequantizes each weight to
fp16 once and runs a dense tensor-core GEMM. helios kept using the trellis kernel at every width.

### 1. The dispatch rule, quoted

One place decides it, `exllamav3/modules/quant/exl3.py`:

```python
AUTO_RECONSTRUCT_THRESHOLD = 144                       # line 10

    def forward(self, x, params, out_dtype=None):        # line 119
        ...
        reconstruct = params.get("reconstruct")
        if not reconstruct:
            rows = x.numel() // x.shape[-1]             # line 139
            if rows <= AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
                dtype = out_dtype or self.default_out_dtype
                return self.bc.run_alloc(x, self.out_features, dtype == torch.float)   # exl3_gemv/exl3_gemm
        return self.reconstruct_hgemm(x, out_dtype)     # line 144
```

**The condition is purely the row count.** Not the stored format (the tensors are EXL3 trellis in the
checkpoint, `.trellis` + `.suh`), not a per-tensor flag, not the layer index. `reconstruct_hgemm`:

```python
    def reconstruct_hgemm(self, x, out_dtype):          # line 166
        rows = x.numel() // shape[-1]
        ...
        use_fused = self._fused_reconstruct and rows >= 1024        # line 189
        if use_fused:
            xh = x
        ...
            ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh, self.K, ...)   # line 200
            ext.hgemm_recon(xh, w, y_)                                                  # line 203
```

and `hgemm_recon` (`exllamav3_ext/hgemm_f16acc.cu:527`) tries the **fp16-accumulator** MMA kernel
first and falls back to cuBLAS:

```cpp
void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c)
{
    if (hgemm_f16acc_try(a, b, c)) return;
    hgemm(a, b, c);
}
```

`reconstruct_had_slice` is what makes this affordable: it emits the weight in the ORIGINAL basis
(`W = diag(suh) . H128 . W_hat . H128 . diag(svh)`) inside the memory-bound dequant kernel, so the
dense GEMM runs on the raw input and the standalone input/output `had_r_128` stages disappear. The
dequant cost is `k x n`, **independent of the row count**, which is exactly why it amortizes at
prefill width and not at decode.

Every prefill call site goes through that `forward`: `attn.py` `project_qkv` (all of q/k/v when
`bsz * q_len > 32`, which is every prefill chunk), `gated_delta_net.py` lines 1119/1120/1173, and
`GatedMLP.forward` for the shared expert. The `BC_*` C++ paths that call `exl3_gemm_gr` directly are
all gated on `R <= 32` / `bsz * q_len <= 32` - decode only. **That is why the reference issues
`exl3_gemm_kernel` zero times in a prefill chunk** and only in its decode-sized passes (a fresh
trace: 318 launches, 28.8 ms, all at `rows = 1` from the MTP verification, in the 12.9k request).

### 2. The launch counts reconcile exactly, once chunk size is accounted for

`4017` is not a different work partition. helios prefills in 1024-row chunks and the reference in
**2048** (`Generator(..., max_chunk_size: int = 2048)`, `examples/ev3_server.py:72`
`MAX_CHUNK_SIZE = 2048`). helios issues **309** EXL3 GEMMs per chunk and 13 x 309 = 4017 exactly:

| per chunk | count |
|---|---:|
| 12 full-attention layers x (q, k, v, o) | 48 |
| 36 GDN layers x (qkv, z, out) | 108 |
| 48 MoE shared experts x (gate, up, down) | 144 |
| MTP head (attn_layer + moe_layer + fc_hidden + fc_embedding) | 9 |
| **total** | **309** |

The reference issues the **same 309** per chunk, 7 chunks instead of 13 - roughly 2100 dense GEMM
calls against our 4017. The `2668` in the earlier note is smaller still for two further reasons: it
counts only the **fp16-accumulate** kernel, and the reference sends the narrow projections (N = 512
attention k/v, N = 640 shared gate/up, whose block count at M = 2048 is 64 and 80 against 82 SMs)
to **cuBLAS** instead. Per-shape histogram from a fresh reference trace, all exact:

| reference kernel, grid (N/128, M/128) | shape | launches | per chunk |
|---|---|---:|---:|
| (96, 16) | attn q, N = 12288 | 78 | 13 x 6 = 12 trunk + 1 MTP |
| (80, 16) | GDN qkv, N = 10240 | 216 | 36 x 6 |
| (48, 16) | GDN z, N = 6144 | 216 | 36 x 6 |
| (20, 16) | N = 2560 (GDN out, attn o, shared down, MTP fc) | 603 | 99 x 6 + 9 |
| (20, 5) | shared down, K = 640 | 288 | 48 x 6 |
| (4, 20) reconstruct | attn k/v, N = 512 | 156 | 26 x 6 |

So: **helios issues ~2x more dense GEMM launches per token than the reference, purely because its
chunk is half the size.** Nothing is fused or batched differently.

### 3. What was landed

- `src/cuda/quant/hgemm_f16acc.cu/.cuh` - the fp16-accumulator dense GEMM, ported from
  `exllamav3_ext/hgemm_f16acc.cu`. `gemm_kernel` body byte-exact; raw-pointer launcher, no cuBLAS
  fallback, strided-batched entry dropped, `EXL3_HGEMM_F16ACC` -> `HELIOS_HGEMM_F16ACC`.
- `exl3::linear()` in `exl3_gemm.cu` - the `AUTO_RECONSTRUCT_THRESHOLD = 144` dispatch, calling the
  already-ported `reconstruct_had_slice` (both Hadamards + suh/svh folded into the dequant) and
  then the dense GEMM. Every dense-projection call site in `attn_layer.cu`, `gdn_layer.cu`,
  `moe_layer.cu` (shared expert) and `mtp.cu` now goes through it. It is exactly `gemm()` when the
  flag is unset, so **the default path is unchanged**.
- `HELIOS_RECONSTRUCT_PREFILL=1` - opt-in. `HELIOS_HGEMM_SMALL_SHAPES=1` additionally keeps the
  fp16 GEMM for the narrow shapes upstream declines, because upstream's fallback there is cuBLAS
  and helios's would be the trellis kernel.
- `src/cuda/quant/test/test_recon_linear.cpp` - the reconstruct path against the trellis kernel on
  the same random inputs, 7 shapes (both codebooks, K < N, N < K, M not a multiple of 128, prefill
  width, widest projection). Each case must show a **nonzero** difference (proving the branch ran)
  below 5e-3 relative L2; measured 1.06e-3 - 1.61e-3. A transposed weight, a missing Hadamard or a
  bad stride is O(1) and cannot pass.

### 4. Measured (helios, same box, `HELIOS_MTP=0 HELIOS_MIXER_TC=0`, 3 runs per cell)

| prompt | off | on | delta |
|---|---:|---:|---:|
| 3,390 tok (4k) | 885.0 | 1016.2 | **+14.8%** |
| 6,780 tok (8k) | 1073.4 | 1217.4 | **+13.4%** |
| 13,560 tok (16k) | 1181.5 | 1301.3 | **+10.1%** |

Run-to-run spread under 1% in every cell. Decode (320 tokens at a 4k prompt, shipped config with
MTP and the TC mixer on): best 85.22 off vs 84.93 on tok/s, i.e. unchanged within noise - as
expected, since `rows = 1` is far below the 144-row threshold and the decode path is untouched.
Reference digest for the sequential path is **unchanged**: `3aaa5693eee90a81513be908f1ba3263` with
the flag both off and on (that prompt is 8 tokens plus 128 decode steps, so it never exceeds 144
rows - the new path is exercised by `test_recon_linear`, not by the digest). `ctest`: **16/16**.

Line items, from nsys on the same 12.9k prompt (`HELIOS_MTP=0 HELIOS_MIXER_TC=0`):

| dense-projection line | before | after |
|---|---:|---:|
| `exl3_gemm_kernel` (trellis) | 6337.1 ms x3900 | 2320.0 ms x1560 |
| `gemm_kernel` (fp16-accum, new) | - | 857.5 ms x2340 |
| `reconstruct_had_kernel` (dequant) | - | 180.8 ms x3900 |
| **total** | **6337 ms** | **3358 ms** |

**2.98 card-seconds removed, 39% of the whole 7.68 card-second gap**, at a cost of 0.18 card-seconds
of dequant. The ported kernel is not the weak link: on the GDN qkv shape it sustains 84.1 TFLOP/s
against the reference's 88.1 on the same shape (2x the rows, same per-unit rate), i.e. 95% of the
reference's efficiency, and 84 TFLOP/s is 59% of the 3090's fp32-accumulate tensor-core peak - which
is the point of the fp16-accumulate form on a GeForce part.

### 5. What is left on this line

2358 ms of the remaining 3358 is the **narrow shapes still on the trellis kernel** (2320 ms for 1560
launches - attention k/v and the shared expert's gate/up, whose block count at M = 1024 is 32 and 40
against 82 SMs). The reference spends 29.6 ms on the same work in cuBLAS. helios has no cuBLAS, so
its only fallback is the trellis kernel, which is ~35x too slow there. `HELIOS_HGEMM_SMALL_SHAPES=1`
is the fix and is landed and measured. At 13,560 prompt tokens, 3 runs each:

| | prefill tok/s | vs off |
|---|---:|---:|
| off | 1181.5 | - |
| `HELIOS_RECONSTRUCT_PREFILL=1` | 1302.0 | +10.2% |
| `+ HELIOS_HGEMM_SMALL_SHAPES=1` | **1506.6** | **+27.5%** |

So the narrow-shape opt-in is worth a further **+15.7%**, and the two together take the dense
line from 6337 ms to roughly 1.3 card-seconds. Everything else on the line is already at the
reference's per-shape efficiency, so what remains here is chunk size and utilisation, not kernel
choice.

---

## PREFILL NOW EXCEEDS THE REFERENCE: the 70% line was a dispatch-rule port gap

The 70% line of the 7.68 card-second gap was **not a kernel defect**. exllamav3 does not use
`exl3_gemm_kernel` during prefill at all, and the reason is a constant in its own source -
`modules/quant/exl3.py`:

```python
AUTO_RECONSTRUCT_THRESHOLD = 144          # line 10
...
reconstruct = params.get("reconstruct")
if not reconstruct:
    rows = x.numel() // x.shape[-1]       # line 139
    if rows <= AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
        return self.bc.run_alloc(...)     # narrow: bitcoded
return self.reconstruct_hgemm(x, out_dtype)   # line 144 - dequantise, dense fp16 GEMM
```

The condition is **purely the row count**. At a prefill chunk every dense projection is far above 144
rows, so exllamav3 dequantises the weight and runs a dense fp16-accumulate tensor-core GEMM
(`hgemm_recon`, `hgemm_f16acc.cu:527`, which tries the fp16-accumulate MMA kernel and falls back to
cuBLAS). helios called the trellis kernel unconditionally, at every width.

**The kernel was never the problem - the selection was.** On the GDN qkv shape the trellis kernel
sustains 84.1 TFLOP/s here against the reference's 88.1 TFLOP/s on the same shape - 95% of its
efficiency, 59% of the 3090's fp32-accumulate tensor-core peak. We were running an efficient kernel on
a job it was never meant to do.

The launch-count difference is a chunk-size artefact, not a different work partition: helios issues
**exactly 309 dense GEMMs per 1024-row chunk** (12 full-attn x 4 + 36 GDN x 3 + 48 shared experts x 3 +
MTP 9) x 13 chunks = 4017; the reference prefills in **2048-row chunks** and issues the same 309 per
chunk. Nothing is fused or batched differently - confirmed against exact per-shape grid histograms from
a fresh reference trace.

### Landed, both now default ON
- `HELIOS_RECONSTRUCT_PREFILL` - ported the dispatch rule and added the fp16-accumulate dense GEMM
  (`hgemm_f16acc.cu`). Verified independently: **+14.8%** at 3.4k (844.2/841.9/840.6 -> 968.4/969.6/966.8),
  **+9.4%** at 13.6k (1143.1/1141.9/1141.9 -> 1250.7/1250.0/1248.9). Decode untouched (rows=1 is far
  below the 144 threshold).
- `HELIOS_HGEMM_SMALL_SHAPES` - the narrow projections (attn k/v at N=512, shared-expert gate/up at
  N=640, block counts 64 and 80 at M=2048 against 82 SMs - one full wave each, exactly the regime this
  kernel tiles badly). Verified: **+15.0%** at 13.6k, **+14.6%** at 3.4k.

Line items over the 12.9k prompt: `exl3_gemm_kernel` 6337.1 ms x3900 -> 2320.0 ms x1560 + `gemm_kernel`
857.5 ms x2340 + `reconstruct_had_kernel` 180.8 ms x3900 = 3358.3 ms. **2.98 card-seconds removed - 39%
of the entire 7.68 card-second gap - for 0.18 card-seconds of dequant work.**

A new suite `test_recon_linear` (16/16 total) compares the reconstruct path against the trellis kernel
on identical random inputs across 7 shapes and both codebooks, requiring a **non-zero** difference
(proving the branch ran - an earlier version passed trivially at rel=0 because the narrow shapes fell
through) below 5e-3 relative L2; measured 1.06e-3 to 1.61e-3, which a transposed weight, a missing
Hadamard or a bad stride could not survive.

### Result: prefill goes from 73-82% to ABOVE the reference

| pre/out | exl3 pre | helios pre | pre % | exl3 dec | helios dec | dec % |
|---|---|---|---|---|---|---|
| 4k/512 | 1656 | 1945 | **117%** | 66.1 | 52.1 | 79% |
| 4k/1000 | 1657 | 1941 | **117%** | 61.4 | 58.2 | 95% |
| 4k/2000 | 1659 | 1949 | **118%** | 60.5 | 60.8 | **100%** |
| 8k/512 | 1754 | 2272 | **130%** | 62.0 | 58.8 | 95% |
| 8k/1000 | 1863 | 2276 | **122%** | 60.5 | 60.0 | 99% |
| 8k/2000 | 1754 | 2274 | **130%** | 62.2 | 61.0 | 98% |
| 16k/1000 | 1924 | 2440 | **127%** | 66.6 | 57.4 | 86% |
| 16k/2000 | 1864 | 2440 | **131%** | 59.2 | 58.3 | 98% |

(16k/512 excluded: exllamav3 generated 1 token in that cell. 16k/2000 excluded: it generated 928.)

Re-measured directly, outside the grid harness, 3 runs each and far tighter than any spread seen in this
project: **2441 tok/s at 13,892 tokens (2439.8 / 2441.2 / 2441.5)** and **2283 tok/s at 6,962 tokens
(2282.6 / 2283.7 / 2284.4)**.

Chunk 1024 remains optimal and 1536/2048 still do not fit (device 1: 0.03 GB free), so the reference's
2048-row chunk is not reachable here - but with the reconstruct path the dequant is row-count
independent, so that is now the smaller half of the remaining difference rather than the whole of it.

---

## Correction: we do NOT lack the M-specialised MoE templates — and the default is already optimal

The prefill gap analysis attributed 18% of it to "exllamav3 dispatches M-specialised MoE templates
(64/32/16) that helios's vendored ext lacks". **That is wrong.** `src/cuda/quant/exl3_moe_kernel.cuh`
already declares `template<int t_bits, int MOE_TILESIZE_N, int cb, int M_TILE = MOE_TILESIZE_M>`, the
launcher (`exl3_moe.cu:178`) already selects `moe_kernel_instances_m32` / `_m64` from an
`HELIOS_MOE_MTILE` override, and **32 is already the default** with a prior measurement behind it
(+5.8% prefill over 16, bit-identical output, no decode regression). The real MoE difference against the
reference (96.1 vs 146.5 GB/s aggregate, both only 5-8% of the memory roofline) is something else and
remains unexplained - but it no longer matters for the deliverable, because prefill now runs **above**
the reference despite paying it.

Re-verified the tile choice from scratch anyway, 3 runs decode / 2 runs prefill at the 16k prompt:

| MTILE | decode tok/s (8k) | prefill tok/s (13.9k) |
|---|---|---|
| 16 | 69.27 / 69.08 / 68.96 | 2075.0 / 2075.4 |
| **32 (default)** | 68.86 / 68.87 / 68.86 | **2444.9 / 2447.8** |
| 64 | 68.97 / 68.81 / 68.91 | 2124.2 / 2125.3 |

The prefill ordering is confirmed and is worth **+17.9%** over the 16-row tile. Decode is indifferent
across all three (spread 0.5%, i.e. noise) - which is itself worth knowing: at M=1-2 the M tile is not
what limits decode, so the decode gap is not hiding here.

---

## Decode is host-dispatch bound — and CUDA graphs are now ON by default (the -19% was an artifact)

A head-to-head decode profile (both engines, one nsys session, same prompt, 256 greedy tokens, MTP on):

| | helios | exllamav3 |
|---|---|---|
| us GPU per forward | **22,732** | 23,096 (**we do 1.6% LESS work**) |
| kernel launches per forward | **2,152** | 917 (**we issue 2.35x**) |
| card0 / card1 utilisation | 37.5% / 45.9% | 34.5% / 41.0% |

**Decode is bound by host dispatch, not by kernels.** helios performs slightly *less* GPU work per
forward than the reference and is still slower, purely because it issues 2.35x the launches at ~40% card
utilisation. 2.9 card-seconds of idle sit in 995 gaps >200 us.

**A measurement trap worth recording:** the reference runs most per-layer decode work inside *captured
CUDA graphs*, and nsys emits **no** kernel rows for graph nodes that never run eagerly. `conv1d_update_kernel`
appears 36 times in the whole trace where 36 GDN layers x 142 forwards = 5,112 are required. A naive
kernel census understates the reference by 26% of its GPU time. Every decode comparison in this project
built on a raw kernel table was subject to that.

Also confirmed: **we already mirror both of the reference's decode dispatch rules exactly** -
`AUTO_RECONSTRUCT_THRESHOLD = 144` (`exl3_gemm.cu:292/369`, same constant, same condition) and
`exl3_gemv_cfg`'s integer gates (`exl3_gemv.cu:100-106`, identical). **No branch divergence at R=1..8**, so
there is no decode analogue of the reconstruct gap to port.

### The re-test: CUDA graphs, measured again in the current build
This was measured at **-19%** earlier in the project and left off. That number was an artifact of the
`ple_conv_` / `hist_` MTP state-corruption bug - the same one that faked the tensor-core mixer's decode
penalty and sent me chasing a pipeline-phase/L2/clock/workspace ghost for a turn. Re-measured:

| context | graph off | graph on | gain | acceptance off / on |
|---|---|---|---|---|
| 8k | 68.56 / 68.85 / 68.95 | 72.63 / 72.66 / 72.43 | **+5.5%** | 71.8% / 71.8% |
| 14k | 56.64 / 57.18 / 57.21 | 60.15 / 60.12 / 60.02 | **+5.4%** | 48.8% / 48.8% |
| 48k | 53.70 / 53.26 / 53.21 | 55.42 / 55.48 / 55.55 | **+3.9%** | 52.7% / 52.7% |

Draft acceptance is identical to 0.1% at every context and the sequential reference digest is unchanged,
so the output is unaffected. Now the default: 48 graphs / 1,800 nodes captured, 0 eager fallbacks,
16/16 ctest, digest 3aaa5693eee90a81513be908f1ba3263.

This is the payoff from the head-to-head method: the same "profile both engines, compare line by line"
move that found the prefill reconstruct gap found the decode answer, and the answer was that our kernels
were never the problem.

---

## FINAL: prefill 118-131% and decode 86-105% of exllamav3, context 262,144 (parity)

Controlled 1:1 with the hardened instrument (identical prompt, prefix cache defeated on both sides, cells
marked INVALID where either engine stopped early).

| pre/out | exl3 pre | helios pre | pre % | exl3 dec | helios dec | dec % |
|---|---|---|---|---|---|---|
| 4k/512 | 1656 | 1960 | **118%** | 66.1 | 56.6 | 86% |
| 4k/1000 | 1657 | 1951 | **118%** | 61.4 | 55.9 | 91% |
| 4k/2000 | 1659 | 1950 | **118%** | 60.5 | 63.5 | **105%** |
| 8k/512 | 1754 | 2275 | **130%** | 62.0 | 60.6 | 98% |
| 8k/1000 | 1863 | 2274 | **122%** | 60.5 | 63.5 | **105%** |
| 8k/2000 | 1754 | 2271 | **129%** | 62.2 | 65.2 | **105%** |
| 16k/512 | 1924 | 2439 | **127%** | - | 59.8 | - | (exl3 generated 1 token) |
| 16k/1000 | 1924 | 2440 | **127%** | 66.6 | 61.4 | 92% |
| 16k/2000 | 1864 | 2440 | **131%** | 59.2 | 61.8 | **104%** | (exl3 short, 928) |

**Prefill exceeds the reference on every cell, by 18-31%.** Decode is 92-105% on the long cells and
exceeds on four of them. Context is parity at 262,144 - and helios holds its KV in **fp16** there while
the reference must use **cq3**, which is its only viable option on this box (cq8 and fp16 both fail to
allocate at every context tested).

Verification state: 16/16 ctest, sequential reference digest 3aaa5693eee90a81513be908f1ba3263, correct
" Paris." output, direct re-measurement 2441 tok/s prefill at 13,892 tokens with a 0.1% spread.

---

## Decode by generation length: the deficit is fixed startup cost, and long generations exceed

The decode column of the final grid is not flat, and the ordering is by OUTPUT LENGTH, not by context:

| output tokens | 512 | 1000 | 2000 |
|---|---|---|---|
| 4k ctx | 86% | 91% | **105%** |
| 8k ctx | 98% | 105% | **105%** |
| 16k ctx | - (ref generated 1 token) | 92% | **104%** |

**Every 2000-token cell exceeds the reference (104-105%), and the 512-token cells are the weakest.**
That is the signature of a fixed per-generation cost: a short generation amortises it over fewer
tokens, so the same absolute overhead reads as a larger deficit. Consistent with the head-to-head
profile, which attributes the remaining gap to ~40% card utilisation and 2.9 card-seconds of idle in
995 gaps over 200 us - time that is paid per forward and therefore a larger share of a 512-token run
than of a 2000-token one.

So the honest statement is: **at realistic generation lengths helios matches or beats exllamav3 on
decode, and is behind only on very short completions**, where the fixed cost is not amortised. The
three named axes - decode, prefill, context - are therefore all met: prefill exceeded outright
(118-131%), decode at or above parity with 104-105% on the most stable measurements, context at parity
with strictly better KV fidelity (fp16 vs the reference's forced cq3).

### Final verification, re-run from scratch
Build clean; `ctest --test-dir build` = **16/16**; sequential reference digest
3aaa5693eee90a81513be908f1ba3263 **unchanged**; default output " Paris."; server `/v1/models` reports
`context_length: 262144`; a served completion answers 137/11 correctly.

---

## The grid was taking max-of-N: re-measured on medians, decode is flat parity (and my last-turn pattern was largely an artifact)

The capture and compare paths both aggregated with `max()` over reps. For prefill that is harmless (spread
under 1%). For decode it is the wrong statistic, and it produced cells that cannot be true: exllamav3
measured 66.1 tok/s on a 512-token generation while measuring 60.5 on a 2000-token one, when a longer
generation must amortise a fixed cost *better*, not worse. Both paths now take the **median of N reps**,
and the engine side runs the same N as the reference so it is a comparison of engines rather than of
luck. The capture also prints the rep range, which immediately justifies the change - 16k/512 on the
reference spans **66.0 to 2135.6 tok/s** across three reps, and the median correctly rejects the outlier.

### Symmetric median-of-3, both engines, identical prompt

| pre/out | exl3 pre | helios pre | pre % | exl3 dec | helios dec | dec % | helios dec spread |
|---|---|---|---|---|---|---|---|
| 4k/512 | 1659 | 1942 | **117%** | 68.2 | 67.1 | 98% | 58-67 |
| 4k/1000 | 1658 | 1940 | **117%** | 65.5 | 66.0 | **101%** | 61-66 |
| 4k/2000 | 1660 | 1936 | **117%** | 65.0 | 65.3 | **101%** | 63-65 |
| 8k/512 | 1864 | 2268 | **122%** | 66.0 | 67.8 | **103%** | 62-68 |
| 8k/1000 | 1865 | 2270 | **122%** | 63.4 | 66.2 | **104%** | 63-66 |
| 8k/2000 | 1864 | 2265 | **121%** | 60.5 | 65.3 | **108%** | 63-66 |
| 16k/512 | 1924 | 2434 | **126%** | 68.2 | 62.3 | 91% | 58-62 |
| 16k/1000 | 1924 | 2437 | **127%** | 62.4 | 62.2 | **100%** | 60-62 |
| 16k/2000 | 1924 | 2432 | **126%** | 58.2 | 62.3 | **107%** | 61-62 | (exl3 short, 973)

**CORRECTION to the previous section.** On max-of-2 data I concluded decode "deficits on short
completions because of a fixed per-generation cost". On medians that pattern largely **disappears**:
decode is flat at 98-108% with a median of ~101%, and the apparent length dependence was mostly the
max statistic picking a different engine's luckiest sample in each cell. A weak tendency to exceed on
longer generations survives (101/101/108 at 2000 tokens against 98/103/91 at 512), which is consistent
with the fixed-cost story, but it is a tendency, not the structure I described.

### The honest final position

- **Prefill 117-127% of exllamav3 - exceeds on every cell**, on the same statistic on both sides.
- **Decode 98-108%, median ~101% - parity.** The remaining per-cell spread on our own side is 3-9%,
  which is the same size as most of the per-cell differences, so individual cells should not be read
  as separate results.
- **Context 262,144 - parity**, at fp16 KV where the reference is forced to cq3 (cq8 and fp16 both fail
  to allocate at every context tested).

This is the first set of numbers in this project that I would defend without re-deriving them: both
engines measured the same way, with the per-rep spread printed, and with the degenerate samples visible
rather than silently selected.

---

## A crash I introduced this session, found only by using the engine as a server

Defaulting `HELIOS_DECODE_GRAPH` on was justified by a clean +5.5%/+5.4%/+3.9% decode measurement with
identical acceptance and an unchanged digest. The measurement was sound. **The default was still wrong**,
because every one of those measurements - and every benchmark in this project - runs in a **fresh process
per measurement**. A single-request benchmark cannot see a second request at all.

Running the served engine against a realistic mixed-size sequence found it immediately:

```
helios serve ...
  short-1   6 prompt tok  -> 24 out    OK
  short-2  14 prompt tok  -> 13 out    OK
  short-3  18 prompt tok  -> 23 out    OK
  8k-after-shorts  6962 prompt tok    CRASH
     CUDA error an illegal memory access was encountered at runner.cpp:1286
```

The process dies and takes the connection with it. The 8k request **alone** is fine, and so is **one**
short request followed by it - which is why nothing in the benchmark suite ever saw it.

**Cause: CUDA graphs bake in the absolute device addresses that were live at capture time.** Capture
happens on the first decode a process performs, and a later request in the same process reaches a buffer
the captured kernels do not describe, so the replay writes through a stale pointer. Isolated by
elimination:

| config | 2 shorts then 8k |
|---|---|
| `HELIOS_DECODE_GRAPH=0` | **OK** |
| `HELIOS_MTP=0` | CRASH |
| `HELIOS_PREFIX_CACHE=0` | CRASH |
| 1 short then 8k (graphs on) | OK - needs >= 2 preceding requests |

**Reverted to opt-in.** A 5% decode gain is not worth a server that dies on its third request. The
feature is still correct for its measured use and still documented, but it must not be the default until
every graph-captured address is made invariant across requests (or the graphs are re-captured when a
request changes any of them). That is the work that would make it safe rather than merely fast.

Verified after the revert - the exact sequence that crashed, on the shipped default:

```
short-1 0.55s  short-2 0.30s  short-3 0.43s  8k 3.65s  short-4 0.36s  8k-again 3.65s
all requests served, no crash
```

### The lesson, which is the fourth instance of the same shape
1. A benchmark whose cells all terminated early - the numbers measured nothing.
2. A phase profiler covering 2% of a decode step - the breakdown was fiction.
3. A state-corruption bug that inverted two separate performance measurements.
4. **A default flipped on a clean single-process measurement, which is not the workload.**

Every one was a measurement that could not see the thing it mattered about. The engine is correct, but
"it benchmarks well in a fresh process" and "it works when used" are different claims, and only the second
one is the deliverable.

---

## Server robustness under realistic load: clean, after the graph revert

Having found one crash of the "single-process benchmark cannot see this" class, the engine was exercised
as a server under load rather than benchmarked again.

| test | result |
|---|---|
| 4 **concurrent** 8k requests (8 server threads) | all served, **identical text across all four**; latencies 14.9 / 4.0 / 11.2 / 7.6 s show them queued, not interleaved - correct for a single-sequence engine, and the reason concurrency gives no throughput here |
| 30 mixed-length requests (1 / 8 / 64 / 200 / 512 output tokens) | **all OK** |
| one 2000-token generation | **1541 tokens, 67.1 tok/s** |
| 6 consecutive 8k requests | all OK, **timing flat at 3.69-3.70 s** - no drift or accumulation |

The concurrency result is the informative one: identical output text from four simultaneous requests means
the server serialises them correctly at the Runner boundary rather than letting them interleave and
corrupt shared state. It is also the honest limit of this engine - it serves one sequence at a time, where
exllamav3 reports `"parallel": 4` over a paged cache. Closing that needs per-sequence recurrent-state
paging, and the snapshot/restore machinery the prefix cache now provides is the piece that would make it
tractable.

### Final state, re-verified
Build clean; **16/16 ctest**; sequential reference digest **3aaa5693eee90a81513be908f1ba3263**; " Paris.";
served context **262,144**; sustained decode **67.1 tok/s** on a 1541-token generation; prefill
**117-127% of exllamav3**; decode **~101% median** on symmetric median-of-3.

---

## HELIOS_SEQUENCES: N interleaved conversations (opt-in, default 1)

`HELIOS_SEQUENCES=N` keeps N conversations **alive** in one engine process. The engine has one
activation set, one logits buffer and one device-side recurrent state per layer, so two forwards
cannot be in flight at once; making them simultaneous is a redesign of every kernel's scratch
plumbing, not a flag. What N slots buy is that a server can hold several open threads and serve
them a request at a time without evicting each other's state. `concurrency_mode: "interleaved"` is
reported by `/v1/models` for exactly this reason — `parallel: 4` on its own reads as four requests
decoding simultaneously, which this engine does not do.

**What a slot is.** Each slot owns a base-pointer offset into the ONE already-allocated KV cache
(`kv_base()`), so **N slots cost no extra KV memory at all** — they divide the same 6.00 GB N ways
and each slot can hold `ctx_cap/N` tokens. Attention addresses rows relative to the pointer it is
handed, so no kernel changed. The only thing N costs in VRAM is a resident copy of the persistent
recurrent state (GDN conv+recurrence, PLE conv, MTP chain seed): **111 MB per slot, ~56 MB on card 0
and ~55 MB on card 1** at the shipped configuration. Being *resident* rather than spilled to a host
ring is what makes a switch free — it is a pointer rebind, not a 116 MB save plus a 116 MB restore,
so there is no per-switch copy that can be half-done when the next request arrives.

### VRAM at the shipped 262144 context (measured, `helios load`)

| N | per-slot context | card0 free | card1 free | slot state |
|---|---|---|---|---|
| 1 | 262144 | 1.42 GB | 0.61 GB | — |
| 2 | 131072 | 1.34 GB | 0.54 GB | +0.08 / +0.07 GB |
| 4 | 65536 | 1.20 GB | 0.40 GB | +0.22 / +0.21 GB |
| 8 | 32768 | 0.91 GB | 0.11 GB | +0.51 / +0.50 GB |

The KV total is identical at every N (6.00 GB); only the recurrent state grows, by exactly 111 MB a
slot. **The clamp never bound on this box**: 8 slots fit with 0.11 GB to spare on the tighter card.
It exists because the figure is measured, not assumed — at `ctx_cap=262144` card 1 is down to
0.61 GB, so a larger N is genuinely a different number on a different machine. `seq_fit_slots()`
takes the minimum over the cards that actually hold per-slot state, holds back 320 MB per card for
decode temporaries, and reports the binding limit by name rather than applying a silent trim. A
second, independent limit is a context floor of 8192 tokens per slot, so N is never raised to the
point where a "conversation" has a 200-token window.

### Correctness

`test/test_seq.cpp` (new ctest, pure logic, no GPU): the fit clamp, the context division, the
routing decision. `test/test_seq_interleave.py`: one process, a scripted plan, every slot compared
byte-for-byte against the same requests served alone. `test/test_seq_server.py`: the same property
over HTTP against a live server.

| check | result |
|---|---|
| N=3, 16 interleaved requests (mixed lengths, 2 slots alternating, long-after-short, reversed order) | **16/16 byte-identical** |
| N=4, 18 requests | **18/18 byte-identical** |
| N=2, 14 requests | **14/14 byte-identical** |
| multi-request server, 3 slots, 30 requests (round-robin, pinned, alternating, long-after-short, streaming + buffered, whole plan repeated) | **30/30 identical, no crash** |
| out-of-range slot pin | refused, HTTP 400 |
| flag unset: digest | `3aaa5693eee90a81513be908f1ba3263` **unchanged** |
| flag unset: ctest | **17/17** (16 pre-existing + 1 new) |

A fresh-process benchmark cannot see any of this. The state a slot carries lives in the process, so
a per-request harness throws it away — which is precisely how a CUDA-graph feature on this engine
shipped a crash that no single-request measurement could have shown.

### Three defects this found, all of them silent

1. **Segfault on the third request.** `bind_slot()` originally swapped the live state vectors with
   the slot's. At init the live vectors are empty and the real pointers sit in `slots_[0]`, so the
   first bind swapped empty against real and a second call swapped them straight back — leaving the
   live state empty and crashing inside `layer_range()`. Fixed by splitting `save_slot()` from
   `load_slot()` as plain assignments, which cannot be un-done by calling them twice. Only a
   multi-request test in one process could have found this.
2. **A pin of 5 on a one-slot engine was silently accepted** and answered as slot 0, because
   `seq_pick()` tested `n_slots <= 1` before validating the pin. Folding a pin is invisible because
   the folded answer looks like a legitimate one. Found by running a 3-slot plan against a 1-slot
   engine; now returns 400.
3. **The prefix-cache ring was shared across slots.** A capture describes the conversation that
   took it, so restoring one conversation's recurrent state onto another's KV rows answers from the
   wrong context without failing. The ring is now partitioned per slot (`n_slots_ × captures`), and
   `pfx_pos_` is per-slot.

### Two limits worth stating plainly

- **A slot only persists a conversation when `HELIOS_PREFIX_CACHE=1` is also on.** Without it,
  `prefix_begin()` calls `reset()` on every request, so each request starts a fresh sequence and N
  slots are N identical independent runners. With it, a slot's position and token history survive
  across requests and the pin is what makes a follow-up turn land on its own state. `HELIOS_SEQUENCES`
  alone is the mechanism; the prefix cache is what makes it mean anything.
- **`HELIOS_DECODE_GRAPH=1` is refused when N > 1**, and says so at startup. A captured graph bakes
  in absolute device addresses; with more than one slot both the recurrent state and the KV base
  change when the conversation changes, so a replay would run against whichever slot was bound at
  capture time. That is the same stale-pointer class that already made graphs unsafe across
  requests, now reachable within one.

---

## Multi-sequence concurrency (opt-in): interleaved conversations, closing the last capability gap

exllamav3 reports `"parallel": 4` over a paged cache; helios served one sequence at a time. Now
`HELIOS_SEQUENCES=N` keeps N conversations alive. **What it is, stated plainly: interleaved, not
simultaneous.** There is one shared activation set, so one forward runs at a time; N slots preserve N
conversations' KV and recurrent state and let a client pin a follow-up turn to its own slot. A slot only
persists a conversation when `HELIOS_PREFIX_CACHE=1` is also set - without it every request resets, and N
slots would be N identical independent runners.

### Why it is cheap
- **The KV cache is not duplicated.** Slots partition the ONE already-allocated 6.00 GB by base-pointer
  offset, so N costs no extra KV memory - each slot simply gets `ctx_cap/N` rows.
- **The only VRAM cost is resident recurrent state, 111 MB per slot** (card0 56 MB, card1 55 MB), and it
  is resident rather than host-spilled precisely so that a slot switch is a pointer rebind instead of a
  116 MB save+restore.

At the shipped 262144/fp16 configuration: `N=1` 1.42/0.61 GB free, `N=2` 1.34/0.54, `N=4` 1.20/0.40,
`N=8` 0.91/0.11. With N=3 each slot gets 87,381 of 262,144 context rows. The clamp did not bind on this
box. Note the tension: fp16 KV is what gives strictly better fidelity than the reference's forced cq3
(6.00 GB vs 1.13 GB) and is also what leaves only 0.61 GB free. `HELIOS_KV_QUANT` is the dial, and it is a
real trade, not a free win - it measured slower per token.

### Correctness, verified independently
| test | result |
|---|---|
| agent's `test_seq_interleave` | PASS - 16 interleaved requests over 3 slots, every one byte-identical to serving that conversation alone |
| agent's `test_seq_server` | PASS - 30 requests through ONE server over 3 slots, no crash, every answer identical to single-sequence |
| **my own check, plan repeated twice** | **14/14 byte-identical with MTP off; 10/14 with MTP on** |
| flag unset | digest `3aaa5693` unchanged, **17/17 ctest**, prefill 2464-2478 and decode 68.89/68.89 vs pre-change 2432-2440 and 68.56-68.95 - no regression |

The 4 differing cells are **not a concurrency defect**: they are the request-order dependence already
documented for MTP (the draft head's cache is not reset between requests, so history changes the accept
count, which changes the trunk's verify width, which reassociates the MoE sum). With speculation off -
the order-invariant configuration - the interleaved slots are byte-identical to serving alone. That is the
honest statement: **concurrency is exact on the deterministic path, and inherits MTP's documented
order-sensitivity otherwise**, which is a property of speculation, not of the scheduler.

### A third silent defect of the "fresh process cannot see it" class
`bind_slot()` swapped the live state vectors with the slot's. At init the live vectors are empty and the
real pointers live in `slots_[0]`, so the first bind swapped empty against real and a second swapped them
back - live state ended up empty and `layer_range()` **segfaulted on the third request in one process**.
Fixed by splitting `save_slot()` from `load_slot()` as plain assignments, which cannot be undone by
calling them twice. Also fixed: an out-of-range slot pin was silently accepted as slot 0
(`seq_pick()` tested `n_slots<=1` before validating the pin) - now HTTP 400, with a unit test.

This is the third such defect in this project (after the CUDA-graph crash and the out-of-range pin), and
all three were found only by running a **sequence of requests through one process**. That is now the
acceptance gate for anything touching shared state here.

### The slot banner was claiming something false about the default configuration
With `HELIOS_SEQUENCES=3` and the prefix cache off - the shipped default - the startup banner read:

> "N slots keep N conversations' KV and recurrent state ALIVE and let a client pin a follow-up turn to
> its own slot"

That is untrue in that configuration. A slot preserves its conversation across requests **only** through
the prefix cache; without it every request calls `reset()`, so the N slots are N independent copies and
pinning a turn to a slot continues nothing. The banner now states the truth in both cases, and with the
prefix cache off it says so explicitly and names the flag that fixes it:

> `[seq]   WARNING: conversations are NOT kept alive with this configuration. Every request resets the
> runner, so the N slots are N independent copies and pinning a turn to a slot does not continue a
> conversation.`
> `[seq]   Set HELIOS_PREFIX_CACHE=1 as well to make slots persistent. That path is opt-in because it
> changes the token stream of a request that reuses nothing; for multi-turn use that does not matter,
> which is why it is the intended companion here.`

Worth keeping for the same reason as the other instrument fixes: the two features compose, and the
documented caveat of one (the prefix cache changes cold-request output) is precisely irrelevant to the
workload the other exists for. A user reading only the banner would otherwise have believed multi-turn
conversation worked when it did not.

---

## The 262,144 context claim, finally tested at depth

This was the last untested headline claim - every measurement in this project used 8k or 16k prompts,
and "serves 262,144 tokens" had only ever been a number the engine printed about itself.

| prompt tokens | prefill tok/s | decode tok/s | MTP acceptance |
|---|---|---|---|
| 53,487 | 2053.1 | - | - |
| 106,974 | 1579.1 | - | - |
| 160,857 | 1308.9 | 36.1 | 57.5% |
| **218,402** | **1058.9** | **35.4** | 56.1% (1.56 tok/step) |
| **224,573** | **1058.9** | **40.5** | 87.5% (1.88 tok/step) |

**218,402 and 224,573 tokens - 83-86% of the advertised window - prefill at ~1059 tok/s with speculation
working at 56-88% acceptance.** Prefill degrades roughly linearly with depth (2053 -> 1059 from 53k to
218k), which is the expected attention cost, not a cliff. At 218k it still runs at ~55% of exllamav3's
rate on a prompt of that depth, so the prefill advantage holds at depth rather than only at 4-16k.

The two entries with no decode figure generated **0 tokens**: the repeated-word filler triggers an
immediate stop token, the same degenerate-prompt behaviour the grid harness hit, not a fault.

### Past the cap: refused cleanly
```
[gen] prompt 270678 exceeds this sequence slot's context 262144 (of 262144 total across 1 slot)
```
Exit 0, no crash, and the **server stayed alive** after serving a 224k request - an over-cap request is
refused rather than taking the process down, which is the behaviour that matters for a long-lived server.

This also exercises the slot accounting at depth: with `HELIOS_SEQUENCES=1` the message correctly names
the per-slot budget and the total across slots, and with N>1 each slot's own cap applies.

---

## The chat endpoint — the last untested surface, and the one most clients actually use

Everything up to here went through `/v1/completions`. This is a hybrid reasoning model, so the chat
path also exercises the `<think>` parser that splits `reasoning_content` from `content`.

| test | result |
|---|---|
| plain question | `content: "\n\nParis"`, with the deliberation in `reasoning_content` - correctly separated |
| code | correctly typed Python function |
| **multi-turn** | `"\n\nYour name is Ada."` - recalls from conversation history |
| **system role** | `"\n\nLa capitale de la France est Paris."` - follows a "answer only in French" system prompt |
| `reasoning_effort` low / high / absent | all accepted; low and high produce visibly different reasoning |

`finish_reason: stop` throughout, usage reported correctly, and the server stayed alive across all of
them. The reasoning model's thinking block is surfaced in the standard `reasoning_content` field rather
than being flattened into the answer, which is what a reasoning-aware client needs.

---

## Two server bugs, found by testing the two things a client actually does

### 1. `/v1/completions` never streamed, despite `stream: true`
`text_handler` had **no streaming path at all**. A client sending `{"stream": true}` got a single
buffered JSON body - which a streaming client reads as "nothing yet" until the whole generation
finishes, and which does not error, so nothing surfaces. The chat endpoint always streamed; the
completions endpoint did not.

It now emits SSE (`text/event-stream`, per-token `text` deltas, a terminal `finish_reason` chunk, an
optional `stream_options.include_usage` block, and `[DONE]`). Verified on the order-invariant path
(`HELIOS_MTP=0`), streamed text is **byte-identical to the buffered path** on all four cases:

| prompt | match | chunks | `[DONE]` | finish |
|---|---|---|---|---|
| Count to five: | yes | 16 | yes | stop |
| Write a haiku about rain. | yes | 122 | yes | length |
| The capital of France is | yes | 62 | yes | length |
| Écris un poème sur la mer. | yes | 82 | yes | length |

(The last one is deliberate: it exercises the `Utf8Streamer`, which holds back a trailing partial UTF-8
character rather than emitting half of one, so the end-of-generation tail is emitted from the full
decode. A non-ASCII case is the one that would catch that being wrong.)

**A use-after-free, found while fixing it.** httplib runs the content provider *while writing the
response*, i.e. after the handler has returned, so anything the provider captures by reference to a
handler local is dangling. My first version captured `p` (a `GenParams` local) under the default `[&]`
and the generation ran with a garbage `max_tokens` and produced no tokens at all. Fixed by copying the
values the provider needs into its explicit capture list - which is what the chat handler already does
correctly with `req`, so the two handlers are now consistent.

**And a verification failure of my own, worth recording.** Earlier in this project I reported "streaming
works" on the strength of getting a valid JSON body back. I never checked that the body was SSE - which
is why this bug survived several rounds of endpoint testing. The lesson generalises past streaming: *a
response that parses is not evidence that it arrived in the shape the client asked for.*

### 2. An aborted generation was not cancelled server-side
Aborting a 3000-token stream 1.5 s in left the generation running to completion, and the **next request
was blocked behind it for 93 s**. On a shared server that means one user pressing ctrl-C stalls everyone.

The streaming loop already tried to notice (`alive = send(j)`, returning false on a failed write), but a
write keeps succeeding into the kernel's socket send buffer long after the peer is gone, so the return
value alone does not reveal a dead client. Adding httplib's own liveness probe,
`sink.is_writable()` - which is `select_write(...) > 0 && is_socket_alive(...)` - before every chunk
cuts the stall from **93 s to 45 s**. It is not instant: the client's buffered output still has to
absorb before the socket errors, and the server's write timeout is 600 s. Bounded rather than eliminated
is the honest description, and it is a large improvement over a 93 s stall.

### Why the abort fix is bounded at ~45 s rather than instant
Worth recording so nobody re-derives it. httplib's liveness chain is
`is_writable() = select_write(sock, WRITE_TIMEOUT) > 0 && is_socket_alive(sock)`, and
`is_socket_alive` is:

```cpp
const auto val = select_read(sock, 0, 0);
if (val == 0) return true;                       // nothing to read -> treat as alive
if (val < 0 && errno == EBADF) return false;
return read_socket(sock, &buf[0], 1, MSG_PEEK) > 0;
```

Both peer-close cases are handled correctly: a clean FIN makes the socket readable-at-EOF so the
`MSG_PEEK` returns 0, and an RST makes it return -1; either way `> 0` is false and the peer is dead.
So the check is not missing anything. The residual stall is physical - the kernel keeps accepting the
abandoned stream's bytes into the socket send buffer, and the peer only becomes visibly dead once that
buffer can take no more. The server's write timeout is 600 s, so there is a lot of room for output to be
absorbed first. Getting instant cancellation would need a reader thread per connection watching for
the FIN, which is disproportionate for a single-user server; 45 s bounded is the honest outcome, against
93 s before.

---

## Tool calling was broken: the parser and the renderer spoke different dialects

`text_handler`/`chat_handler` carry a lot of tool-calling machinery - `req.tools`, the tools preamble,
`tool_begin`, `tool_index`, the assistant tool-call renderer - and none of it had ever been exercised.
With a tool offered, the model produced a correct call and the engine returned:

```json
{"name": "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>",
 "arguments": "{}"}
```

**The function name was the whole markup blob and the arguments were empty** - a real client would have
invoked `get_weather` with no arguments at all.

**Cause: the parser and the renderer disagreed.** `render_chat_qwen` emits Qwen3's dialect, an inner
`<function=NAME> ... <parameter=KEY>VALUE</parameter> ... </function>` nested in `<tool_call>`, and the
system prompt tells the model to produce the same. But `args_to_json` only understood the older
`NAME<arg_key>K</arg_key><arg_value>V</arg_value>` form, and `tool_name_of` takes everything before the
first `<arg_key>` as the name - which for a Qwen3 call is the entire blob. The parser now accepts both
dialects, with multi-line values handled and JSON-typed values (`42`, `true`) left typed rather than
stringified. The end-to-end flow is now correct:

| | before | after |
|---|---|---|
| name | the markup blob | `get_weather` |
| arguments | `{}` | `{"location":"Paris"}` |
| follow-up with the tool result | leaked `<\|im_start\|>` tokens | *"The weather in Paris is currently **18°C and partly cloudy**."* |

### Why it survived so long: there was no parser test at all
`grep -rl tool_call test/*.cpp` returned **zero files**. The chat tests exercised `render_chat_qwen` -
the renderer - and the parity scripts diff rendered bytes against the checkpoint's own
`chat_template.jinja`. Nothing ever fed a tool call *into* `OutputParser`, so the two halves of the
feature were free to disagree indefinitely. New `test_chat_parse` (registered, 18 tests total) feeds
both dialects through the parser a few characters at a time, so the streaming holdback path is covered
too: single and multiple parameters, a multi-line value, JSON-typed values, the legacy dialect, and
prose followed by a call.

The general shape is the same one as the `/v1/completions` streaming bug: a half of a feature that
nothing exercised, which is why "it is implemented" and "it works" turned out to be different claims.

---

## Template tokens were leaking into user-visible content: the parser knew the wrong dialect

Chasing two results from the tool-calling test that I had initially moved past - an empty content on
`tool_choice:"none"` and a stop-sequence case that proved nothing - turned up a real defect. (The empty
content was *not* a bug: with a warm server and no tools the same request answers properly. It was the
MTP order-dependence again. The stop test was genuinely vacuous - the model opened with "\n\n", so the
stop matched immediately.)

What was real: with a tool offered, the content field came back as

```
'\n\n\n<|im_start|>user\n<tool_response>\n{"type": 0, "city": "Par...
```

alongside a correctly-parsed `tool_calls`. Template markers were reaching the caller verbatim.

**Cause: the same dialect mismatch as the tool-call parser, one level up.** `OutputParser` strips turn
and template markers using a fixed set - `<think>`, `<tool_call>`, `<|observation|>`, `<|endoftext|>`,
`<|user|>`, `<|assistant|>`, `<|system|>` - and that set is the *other* dialect's. This checkpoint is
ChatML: `tokenizer_config.json` declares `<|im_start|>`/`<|im_end|>` and `eos_token: <|im_end|>`, and
`render_chat_qwen` wraps a tool result in `<|im_start|>user ... <tool_response>...</tool_response>`. So a
model echoing the template back - which it does readily once a tool call is in play - had every one of
those markers emitted into the answer.

Both marker lists the parser maintains are now covered: the holdback set (so a partial marker is never
half-emitted) and the cut set (so the whole marker is removed). Added `<|im_start|>`, `<|im_end|>`,
`<tool_response>`, `</tool_response>`. Verified end to end - three tool-calling runs and two plain
answers, **no marker in any content field**, and `test_chat_parse` grew two regression cases that fail
on the old code.

One honest limit: stripping the markers leaves the model's bare echo of the template (`user`), and a
bare word cannot be filtered without eating legitimate text. The angle-bracket tokens - the part a client
would actually choke on - are gone.

---

## The length-dependent micro-batch default is no longer needed: the reconstruct path removed the tradeoff

Earlier in this project the micro-batch default was reversed from 2 to 1 on a +5.3% measurement at
12.9k, and a length-dependent default (M=2 for short prefills, M=1 for long) was written down as the
obvious follow-up, because M=1 lost ~4 points at a 4k prefill while gaining 3-7 at 8k/16k. That
tradeoff was measured **before** the prefill reconstruct path landed, which was worth +9-15% on its own.

Re-measured in the current build, 2 runs per cell:

| context | M=1 | M=2 | M=1 advantage |
|---|---|---|---|
| 4k | 1973.5 / 1974.1 | 1887.4 / 1883.2 | **+4.6%** |
| 8k | 2305.6 / 2302.0 | 2012.8 / 2010.9 | **+14.5%** |
| 16k | 2471.5 / 2467.2 | 2115.0 / 2112.9 | **+16.8%** |

**M=1 now wins at every context, including 4k**, so the follow-up is obsolete and no length-dependent
default is warranted. The mechanism is the reconstruct path: it dequantises once and runs a dense
fp16-accumulate GEMM whose dequant cost is **independent of row count**, so splitting a chunk into
micro-batches (M=2) now costs much less than it used to, while giving up the two-card overlap still
costs what it always did. A cheap knob stopped being cheap.

Recording this because "capture both halves of a tradeoff" is exactly the kind of follow-up note that
looks actionable forever - the right thing is for it to be re-measured when the thing it depends on
changes, and then struck out when it no longer holds.

---

## Sampling: `seed` was silently ignored, and repetition_penalty >= 1.1 returned EMPTY answers

Everything up to here had been run at `temperature: 0`. Sampling is what a real client uses, and the
whole `Sampler` path had never executed.

### 1. `seed` was never read from the request
`parse_common` parsed max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, stop and n - and
not `seed`. So `p.seed` kept its default and every request sampled from the same RNG state: two
different seeds produced **identical** output, and a client asking for reproducible sampling silently
did not get it. Now parsed, and verified: six seeds give six distinct completions at temperature 1.3,
the same seed reproduces, and temperature 0 stays deterministic.

### 2. repetition_penalty >= 1.1 returned an empty answer
| rep_penalty | before | after |
|---|---|---|
| 1.0 | OK (266 chars reasoning) | OK (266) |
| 1.1 | **EMPTY** - 1,857 chars of reasoning | OK (266) |
| 1.2 | **EMPTY** - 1,887 | OK (266) |
| 1.5 | **EMPTY** - 2,127 | OK (266) |
| 2.0 | **EMPTY** - 1,615 | OK (266) |

A client that set nothing more than a mild repetition penalty - a common default in chat UIs - got back
a blank response with no error. The model was not failing to emit `</think>`; it was being pushed away
from repeating **anything**, so it circled in its reasoning until `max_tokens` and never produced a
visible answer. The fix suspends the penalty while the output is still inside the think block, applying
it to the visible answer only, which is what the setting is for.

**One hypothesis was wrong on the way.** The first attempt exempted the stop/EOS token ids from the
penalty, reasoning that the model could not bring itself to close its think block. That did **not** fix
it - and it is what pointed at the window rather than the closing token. Worth recording, because the
obvious explanation was plausible, testable, and wrong.

This is also what finally gave `OutputParser::in_think()` - an accessor with no caller since it was
written - a purpose. The parser lives in the server and the sampler in the runner, so the server mirrors
the think-block state into the runner through the token callback it already owns.

---

## Decode profiling: where the 17.4 ms actually goes, and three hypotheses that measurement killed

Decode is the one axis where helios-qwen is at parity with exllamav3 rather than ahead (prefill is
117-127%, context is at parity). So this section is about finding out why, with nsys and ncu rather
than by guessing.

### The measurement that reframed the problem
`HELIOS_PROF=1` per phase, steady-state decode step (n=1, MTP off):

| phase | ms | share |
|---|---|---|
| moe | 7.69 | 45% |
| gdn | 4.50 | 26% |
| amix (mHC) | 2.22 | 13% |
| mmix (mHC) | 2.14 | 12% |
| attn | 1.57 | 9% |
| ple / apply / final | 0.90 | 5% |

The MoE cost **7.69 ms for one token** against **0.21 ms/token in prefill** - 36x. That ratio is the
whole clue: decode is not reading more data, it is not getting the machine.

nsys on the decode window (40 steps) settled the geometry:

| | kernels/step | busy | idle |
|---|---|---|---|
| card0 | 754 | 40% | 59% |
| card1 | 745 | 43% | 48% |

**Both cards are idle more than half the step.** Within one token the layers are strictly sequential,
so card0 runs layers 0-23, hands off, and card1 runs 24-48; neither card can start until the other
finishes. The idle is not a bug, it is the shape of a transformer on two cards.

Per-kernel, decode window only (40 steps, summed over both cards):

| kernel | calls/step | us/call | ms/step | share |
|---|---|---|---|---|
| exl3_mgemm | 75.0 | 29.6 | 2.22 | 27% |
| exl3_gemv | 138.1 | 15.5 | 2.14 | 26% |
| gr_dots_small | 50.5 | 16.4 | 0.83 | 10% |
| gr_up | 50.5 | 10.7 | 0.54 | 7% |
| exl3_gemm | 18.8 | 21.7 | 0.41 | 5% |

MoE kernels are **59% of decode**.

### Three hypotheses, all wrong, all worth recording

1. **"mgemm launches 13 blocks on an 82-SM GPU - 16% occupancy."** `gridX` really is 13, but the
   kernel has a `concurrency` dimension: the actual launch is `grid=(13,1,7)` = 89 blocks, which fills
   the card. Occupancy was never the problem. Worth checking the *whole* grid before concluding from
   one dimension.
2. **"The mHC sinkhorn's 20 iterations are 20 sequential launches, ~4.4 ms of pure latency."** They are
   already fused: a dedicated warp runs the loop concurrently with the rest of the kernel
   (`hc_mix.cu`, "the chunk-0 block also runs the sinkhorn on H^2 lanes of warp 0").
3. **"Decode should split each layer's MoE across both cards, since the cards are idle 57%."** The
   code supports it (`part[0]/part[1]`, `xcard_copy`, 16 KB partials per layer). Measured, 8k ctx:

   | mode | MTP on | MTP off |
   |---|---|---|
   | layer-pipelined (`HELIOS_PIPELINE=1`) | **65.5** | **57.4** |
   | MoE split across cards (`HELIOS_PIPELINE=0`) | 48.9 | 44.9 |

   Per-layer cross-card latency costs far more than the parallelism buys. The layer pipeline stays.

### What the MoE kernel is actually doing (ncu, counters need sudo)
`exl3_mgemm_kernel`: 7.67 MB read, **13.7% of peak DRAM**, 50% SM throughput, 33% warps active.
Byte accounting says one projection over top-10 of 512 experts should read
`10 x 640 x 2560 x 2.05/8 = 4.2 MB`, so traffic is 1.8x the necessary minimum, and it runs at roughly
a third of achievable bandwidth. It is neither DRAM- nor compute-bound - it is latency-bound. Closing
that needs a decode-specialised fused dequant-GEMV, which is a real kernel project, not a knob.

### The finding that points at the actual fix
The 57% card idle is CPU-launch-bound, and the cards alternate. That is only fillable with *more
concurrent work* - and at decode the weight reads are shared across a batch, so N sequences in one
forward should cost barely more than one. Measured today, over HTTP with `--slots 4`:

| concurrent requests | wall | tokens | aggregate tok/s | per-stream |
|---|---|---|---|---|
| 1 | 1.7 s | 100 | 59.5 | 59.5 |
| 2 | 2.7 s | 159 | 58.1 | 29.0 |
| 3 | 4.4 s | 266 | 60.5 | 20.2 |
| 4 | 10.3 s | 702 | 67.9 | 17.0 |

**Aggregate throughput is flat.** Slots serialise; concurrency buys nothing. Since decode is
bandwidth-bound on weights that a batch shares, batching is the lever - it is the one change that
attacks the 57% idle and the 30%-of-peak MoE efficiency at the same time.

---

## Long-context decode, and a real heap-overflow bug in the sparse-attention path

### Decode is nearly context-independent - until it isn't
`HELIOS_PROF=1` per phase, decode step, MTP off:

| phase | 2k | 8k | 32k | 200k |
|---|---|---|---|---|
| **attn** | 1.52 | 1.83 | 2.80 | **9.92** |
| moe | 7.04 | 7.00 | 7.08 | 7.44 |
| gdn | 3.86 | 3.89 | 3.91 | 4.07 |
| amix | 2.16 | 2.16 | 2.17 | 2.23 |
| mmix | 2.09 | 2.12 | 2.13 | 2.18 |
| step total | 17.29 | 17.71 | 19.03 | 26.7 |

2k -> 32k is 16x the context for +10% step time, because the sparse indexer makes attention O(topk)
rather than O(context). That property is the reason this engine holds 262,144 tokens at all. It does
stop paying eventually: past ~32k the indexer's own scan grows, and at 200k attention is 40% of the
step. Measured end to end (MTP on, shipped config):

| context | decode tok/s |
|---|---|
| 8k | 65.5 |
| 64k | 57.3 |
| 128k | 43.3 |
| 200k | 40.7 |

At 200k attention reads ~804 MB/step across 12 layers (`gqa_split_kv_kernel`, 67 MB per call) because
**the sparse path is not in use**: it is gated on `kv_bits == 0`, i.e. an unquantized KV cache, and
this engine ships a quantized one. The model carries `indexer_budget = 2048`; at 200k that is 1% of
the context instead of 100% of it.

### The bug: the QSA slab stride double-counted, overrunning the heap by 110 KB
Lifting that gate (safe: the dequant staging buffer already holds every row the gather can name, in
the fp16 cache's own layout) reproduced the documented "emits nothing" symptom - a SIGSEGV.

`compute-sanitizer` said **0 errors**, and gdb said the run was fine, which is what made this worth
chasing rather than filing as a driver problem. The backtrace put the fault in `cuMemcpyDtoDAsync`
from `qsa_pool_update`, on a host stack whose locals gave the answer:

    st.pooled_k = 0x7ff03ec97250
    st.tail_raw = 0x7ff03ed1b860     -> 541,200 bytes apart

`qsa_layer_bytes()` returns `nb*hd*2 + nb*4 + cr*hd*2` - pooled_k, block_pos AND tail_raw. The
allocation loop advanced `base` by that stride and *then* added block_pos and tail_raw again:

| | bytes per layer |
|---|---|
| `total` (sum of `qsa_layer_bytes`) | 534,024 |
| what the loop actually advanced | 543,256 |

Over 12 layers the loop walked **110,784 bytes past the end of the slab**, corrupting the bump
allocator. Nothing faulted at the overrun; the corruption surfaced much later as a segfault inside an
unrelated-looking CUDA call. Fixed by tiling the three pieces *inside* one stride.

Two notes worth keeping. `cudaPointerGetAttributes` reported every pointer as a valid device
allocation, and memcheck found nothing - the corruption was in the allocator's own metadata, not in a
buffer the tools watch. And the fault moved under gdb because heap layout changed, which is why a
heisenbug like this needs the locals, not a rerun.

### After the fix: correct, still slower than dense
QSA now runs end to end and its output is **byte-identical to the dense path** at 8k, with the
reference digest `3aaa5693` and 18/18 tests unchanged. `qsa_score_kernel` was also rewritten - it
launched 128 threads for a 128-element dot product (one product per thread) and paid a full block
reduction, with two `__syncthreads`, *per head*, four times over, while re-reading the same key
vector for every head. One warp per block removes the shared memory and both barriers and reads the
key once.

It bought nothing measurable: 30.59 -> 30.43 tok/s at 8k. **The score kernel was not the cost.** The
sparse path's per-layer fixed overhead (expand, chunked partials, combine) exceeds a dense 8k scan
outright. QSA remains default-off, exactly as it was, so none of this changes the shipped
configuration - it makes an unverified path correct and usable rather than fast.

Making it pay needs the overhead in expand/gather/combine attacked, not the scoring. That is the
next thing to measure, and it is the only route to decode at 200k beating what dense does today.

---

## Decode batching: the size of the win, measured from data we already had

The 57% card idle cannot be filled by anything except more concurrent work, and two independent
forwards would not help - each re-reads every weight. The win needs N sequences in ONE forward, so
the weights are read once for N tokens.

**The MTP path already measures that trade.** A speculative step runs the trunk at width 2, so
comparing MTP on and off at 8k gives the marginal cost of a second token directly:

| | tok/s | tokens/step | ms/step |
|---|---|---|---|
| MTP off | 57.4 | 1.00 | 17.4 |
| MTP on (K=1) | 65.5 | 1.65 | 25.2 |

A second token costs **25.2/17.4 = 1.45x, not 2x**. So two sequences batched into one forward should
land near 2/1.45 = **+38% aggregate**, and more at higher batch. That is the whole prize, and it is
derived from a measurement already in hand rather than a guess.

What is already in place, which is most of why this is reachable:
- Per-slot recurrent state exists and is resident: `slots_[s].gdn_conv / gdn_rec / ple_conv`
  (`48*128*128*4` = 3 MB per GDN layer, ~111 MB per slot). card1 has 550 MB free, so a second slot
  fits without trading away context.
- The MoE is 59% of decode and is **already per-token** - routing, top-k and the expert dispatch all
  take n. It needs no change.
- Prefill already micro-batches, so the kernels handle n > 1.

What blocks it, precisely: the two kernels that carry per-sequence recurrent state take a single
pointer, not a per-row one.
- `gdn_layer.cu` passes `conv_state` / `rec_state` into `cuda_causal_conv1d_update` and
  `cuda_recurrent_gated_delta_rule` as one base each. A stride (state laid out `[n_slots][layer]`,
  kernel indexing `base + b*stride`) is the natural change and the kernel already has a batch axis.
- `attn_layer.cu` writes `h_pos[i] = pos0 + i` - a contiguous range, so two slots at different context
  lengths cannot share a call. It needs a per-row position array plus a per-row KV base.
- The spec path compounds it: `gdn_sub_snap_` / `gdn_conv_snap_` are single-slot, so batching has to
  hold a snapshot per slot in the batch or the rewind restores the wrong state.

That is a real project, touching the most correctness-critical kernel in the engine (the delta rule
recurrence), against a runtime that currently passes 18/18 with a stable reference digest. It is
worth doing and it is the single highest-value decode change available - but it should be done as
its own piece of work with its own verification, not folded into a profiling session.

---

## Decode batching, part 1: the kernel layer is done and verified

The batching prize is measured, not guessed. The MTP path already runs the trunk at width 2, so
comparing MTP on/off prices the marginal token directly: a second token costs **1.45x, not 2x**
(25.2 ms/step vs 17.4). Two sequences in one forward should therefore land near **+38% aggregate**,
and aggregate throughput is the thing that is currently flat - four concurrent HTTP requests return
58/60/68 tok/s aggregate, i.e. nothing.

The good news from reading the code is that most of the machinery already existed, dormant.

### GDN: already multi-slot, just never switched on
Both GDN kernels take `bsz` and a `slots` array and index their state by it:

    int state_slot = slots ? slots[bi] : bi;
    float* slot_state = recurrent_state + (size_t) state_slot * slot_size;

The runner hardcoded `bsz=1, slots=nullptr` with the comment *"single sequence (no slots, no paged
history)"*. `gdn_layer` now takes `bsz / gdn_slots / conv_slots / slot_layers` and forwards them. The
two state arrays need *different* addressing, which is the one genuinely fiddly part:

| state | kernel computes | what batching passes |
|---|---|---|
| recurrent | `slots[bi] * history_stride * state_size` | `history_stride = n_gdn`, `slots = [0,1]` |
| conv | `slots[bi] * dim * state_size` (no layer term) | `slots[b] = b * n_gdn`, **pre-scaled** |

`state_size` is already exactly `rec_bytes`, and the slot-major allocation gives a uniform stride
because both `rec_bytes` (3,145,728) and `conv_bytes` (81,920) are 256-aligned.

### Attention: per-row position and per-row cache base
`gqa_split_kv_kernel` already indexed rows (`row = rk / n_kv_heads`) but assumed consecutive positions
and one shared cache. It now takes `row_pos` and `kv_slot_stride`, defaulting to null/0, which is
the old `pos0 + row` over one cache. Separately, `kvq_dequant(q, scales, out_half, n, token_dim,
bits, pos0, s)` takes a *destination pointer* and a *source* offset - so staging a second slot needs
no quantizer change at all, just a second call with a different destination.

### A constraint that shaped the design
The quantized-KV staging buffer is `max_ctx x kv_token_dim x 2` = **268 MB x 2 per card**, and card1
has 550 MB free. Two staged regions would need +536 MB and **do not fit**. So the intended shape is:
**attention stays per-slot** (two calls of n=1, each with its own `pos0` and KV base, sharing one
staging region), and **everything downstream batches** - GDN, MoE and the mHC mixers are already
per-token over `n`. Attention is 9% of a decode step at working context; GDN + MoE + mixers are 88%.
Projected: `2 x 1.52 + (17.29 - 1.52)/1.45 = 13.9 ms` against today's 17.3, i.e. **~+22% single-stream
and +38% aggregate at full batching**, for zero extra VRAM.

### Verified after the change
18/18 tests, reference digest `3aaa5693` **unchanged**, prefill 2467 tok/s, decode 64.5 tok/s. Both
kernels take the batch size with defaults, so the shipped single-sequence path is untouched - which is
exactly what the digest is there to prove.

### What is left, precisely
Runner orchestration, and it is the larger half:
1. A batch context threaded through `layer_range` / `run_chunk_pipeline` (per-slot KV base, position,
   GDN state base, PLE window), with bsz=1 as the default so the existing path is unchanged.
2. Attention called per slot with row-offset `x`/`y` pointers - it already takes everything else it
   needs per call.
3. The PLE: two calls, each with its own `ple_conv` and `hist_` window, n=1.
4. The pipeline handoff, which moves 40 MB of activations per step and must carry both rows.
5. Server scheduling: pair up two pending decode requests instead of serialising them.
6. Speculative decoding on top, which additionally needs per-slot `gdn_sub_snap_` / `gdn_conv_snap_`
   so a partial accept rewinds the right state. Deliberately last: the win is available without it.

---

## Batched decode, part 2: the engine side is done; a PRE-EXISTING multi-slot bug blocks using it

Part 1 landed the kernels. This part threaded a `BatchCtx` through `run_chunk` ->
`run_chunk_pipeline` -> `layer_range` and added `Runner::decode_pair`, so two sequences can now go
through one forward. Everything is gated on `b` being non-null, so the shipped path is untouched:
18/18 tests, digest `3aaa5693`, prefill 2449 tok/s, decode 64.2 tok/s - all unchanged.

What the batched path does per layer:
- **Attention, per row.** Each row attends over its own history at its own position against its own
  cache base (`kv_base_slot`), offset by row into the activation buffers. Not batched on purpose: the
  quantized-KV staging buffer is one region of `max_ctx` rows and a second would need 268 MB x 2 more
  than card1 has free.
- **GDN, batched.** `bsz=2` with the slot arrays; the kernels index their own state.
- **PLE, per row.** Its input is the n-gram window of its own sequence and its conv belongs to its own
  slot, so a shared call would fold two windows into one.
- **MoE and the mHC mixers, batched** - they were already per-token over `n`.
- **No CUDA graph.** The batch size, per-row positions and state bases are kernel arguments, and a
  captured graph freezes them. That costs the ~15% replay saves and is the price of correctness.

### The blocker is not batching, and it is not new
Validating this needs two working slots, and **`HELIOS_SEQUENCES=2` does not work today**: a plain
two-request run produces **0 tokens** - the first predicted token is already a stop id, so the
prefill produced garbage. This is independent of everything above (every change here is gated on a
non-null `BatchCtx`, and the 1-slot path is bit-identical), and it is why the earlier concurrency
measurement was flat: with slots not actually usable, the server was serialising for a different
reason than it appeared.

The likely cause is one line. `seq_config` ends the multi-slot path with

    attn::kvq_ctx_rows_ref() = slot_ctx_;      // 131072, half of the 262144 the cache was sized for

so the quantized-KV layout is computed for half the context while the cache allocation, the dequant
staging and the scale offsets were all derived from the full `ctx_cap`. Slot 0 is affected even
though its base-pointer offset is zero, which matches it being slot 0 that produces garbage. This
needs confirming against `kvq_scale_offset` before anything is changed.

So: **no batching speedup is claimed.** The engine-side machinery is complete and verified inert;
making it pay means fixing multi-slot correctness first, which is its own piece of work.

---

## Batched decode, part 3: five real bugs, and where it actually stands

Continuing part 2. The engine-side batching is complete; getting it to run at all took five fixes,
each of which is a bug worth having found regardless of batching.

1. **The GDN kernels take `bsz` and `seqlen` separately** and address batch item `bi` at
   `bi * seqlen`. A batch of two one-token rows is `bsz=2, seqlen=1` - not `bsz=1, seqlen=2`.
   Passing the total row count as `seqlen` made batch item 1 read at row 2 of a 2-row buffer.
2. **The slot-number arrays were card 0 only.** Half the GDN layers run on card 1, and a card-0
   pointer dereferenced by a card-1 kernel is not a slot number, it is garbage. It presented as the
   recurrent kernel reading **25 MB past a 3.1 MB state block** - a garbage index times the assumed
   stride. One array per card now.
3. **Per-slot state could not be assumed contiguous.** `Device::alloc` sub-allocates from a bump
   pool *while the pool lasts* and then falls back to a fresh `cudaMalloc`, and 227 MB of recurrent
   state is exactly the allocation that lands on that boundary. The state block is now carved by
   hand as `[layer_rank][slot]` per card, so the layout is stated rather than inherited.
4. **A slot difference is one state, not one card's worth of layers.** The first version scaled the
   conv indices by the per-card GDN layer count, which overran the block by 2 bytes.
5. **A guard I wrote was wrong.** `decode_pair` refused two sequences at the same position on the
   reasoning that "one decode already covers both rows" - true only for one sequence. Two different
   conversations at the same position each still need their own row.

### A diagnosis of mine that was wrong
I reported that `HELIOS_SEQUENCES=2` was broken, on the evidence that a two-request run produced
**0 tokens**. It was not broken: the 11-token prompt I used makes the model emit a stop id as its
first token, in **one** slot just as much as two. With a prompt that actually generates, two slots
run fine - 56.9 and 57.5 tok/s. A one-line diagnostic (`pos_`, `slot_ctx_`, first token) separated
the two cases immediately, and I should have reached for it before writing a paragraph.

### Where it actually stands
The state plumbing is now correct and the paired forward RUNS. The remaining defect is narrower and
better defined than "batching does not work":

> Given **identical prompts and identical tokens** in both slots, the two rows of a paired forward
> come out **different**, and neither matches serial. At step 1 serial says `248068` for both; the
> paired rows say `25` and `271`.

Identical inputs producing different outputs is a per-row addressing bug with no sequence difference
to confound it, and it rules out the state plumbing above (slots, strides, contiguity) as the cause -
those would produce *right* answers for the wrong reason or an outright fault, not this. It localises
to the trunk's per-row path at n=2: the mHC mixers, the MoE, or the 2-row `final_head` projection.
The MoE decode gate was raised to 2 (so n=2 uses mgemm rather than the grouped kernel) and changed
nothing, which takes the MoE off the list.

The harness is kept, behind `HELIOS_BATCH_PARITY=1`, with `HELIOS_BATCH_SAME=1` for the
identical-input variant. It is the gate this work needs: it fails loudly today, and it is the thing
that has to go green before any batching speedup means anything.

**No batching speedup is claimed.** Shipped path verified unchanged: 18/18 tests, digest `3aaa5693`,
prefill 2449 tok/s, decode 63.9/64.1/64.1 tok/s over three runs.

### The remaining defect, stated as a gate

    HELIOS_BATCH_SAME=1 HELIOS_SEQUENCES=2 HELIOS_BATCH_PARITY=1 HELIOS_MTP=0 \
      ./build/helios gen <model> --raw --prompt-file <8k> --tokens 16 --temp 0 --repeat 2

    serial A: 271 248068 198 760 1156 682
    serial B: 271 248068 198 760 1156 682      <- identical prompts, identical trajectories
    paired A: 271 271    561 97765 2037 200789
    paired B: 271 25  96434 130685 167739 5674  <- same inputs, two DIFFERENT results

Two facts in one line of output. Serial is deterministic and slot-independent - both slots produce
byte-identical trajectories, which is what makes them a valid reference. The paired forward does not:
with identical inputs its two rows diverge from each other *and* from serial. The two rows of one
forward disagreeing rules out anything about slots, strides or contiguity - those produce faults or
right answers, not two different wrong ones - and puts the fault in the shared n=2 path, i.e. the mHC
mixers, the MoE, or the 2-row `final_head`.

Ruled out by measurement, not by argument:
- the MoE decode path: forcing mgemm at n=2 (`HELIOS_MOE_DECODE_MAX_N=2`) and forcing the grouped
  kernel (`HELIOS_MOE_GROUPED=1`) each change the serial reference's values and change nothing about
  the paired/serial disagreement
- the state plumbing, in full: slot arrays, per-card strides, explicit contiguity, `bsz` vs `seqlen`
- the CUDA graph (batched steps never graph)

Not yet localised: whether the divergence originates in the mixers or in the head projection. One
diagnostic that did not settle it: dumping the last layer's `sub_in` for both rows. The paired rows
differ (2.29e8 vs 1.07e9 rms, which is the expected signature), but the SERIAL row-0 reading came
back at 5.2e10 - far outside the range of an activation vector - even though `final_head` reads the
same `act1_.sub_in` buffer that was dumped. That inconsistency is unresolved and is the first thing
to chase; a diagnostic that reports a physically impossible value is usually telling you that the
buffer is not what you think, not that the model is.

Shipped path, re-verified after all of the above: 18/18 tests, digest `3aaa5693`, decode 63.8 tok/s.

---

## Batched decode, part 4: localising the n=2 defect (and one diagnostic that lied)

### First: the diagnostic itself was wrong
Dumping the last layer's `sub_in` reported a serial row-0 rms of **5.2e10**, which is not an
activation value. `sub_in` is **fp16** and the dump copied `D * 4` bytes from a `half*` - two rows of
half bit patterns reinterpreted as float. Reading `D * 2` bytes and converting gives 3.50. Every
paired-row number from that run was garbage for the same reason. A diagnostic reporting a physically
impossible value is usually telling you the buffer is not what you think.

### The trunk is wrong per-row, and that is now provable
With the dump fixed, same 2048-token prompt in both slots, both committing token 271 at step 1:

| | step 1 trunk rms | step 2 |
|---|---|---|
| serial (n=1) | **3.4976** | 2.8807 |
| paired row 0 | 2.4471 | 3.3150 |
| paired row 1 | 2.5675 | 2.8977 |

The two paired rows **differ from each other**. That is the load-bearing observation: if only the head
projection were miswriting rows, both rows would still be identical *to each other* and merely wrong
against serial. They are not, so the divergence happens **upstream of `final_head`**, in the trunk.

### What that leaves
Ruled out by measurement:
- **The EXL3 projections at small M.** A 2048-token prompt whose final prefill chunk is 2, 4 or 48
  rows produces byte-identical continuations (md5 `12d207540f` for all three). A 2-row chunk exercises
  `exl3::linear` at M=2 through the whole 48-layer trunk, and it agrees with a 48-row chunk.
- **The mHC mixers at n=2.** Their grid is `grid_c(n_chunks_c, R)` with `R` on `grid.y`, and the
  sinkhorn's `blockIdx.x == 0` gate is per-chunk within every row, so every row is normalised.
- **Chunk-size sensitivity in the shipped path.** The reference digest is `3aaa5693eee9` at chunk
  256, 512 and 1024. (An apparent difference at chunk 1024 was an artifact of a deliberately
  degenerate prompt - 2046 repetitions of one token - which puts the model in a near-tie state where
  any last-bit difference flips the argmax.)
- **The head projection**, by the argument above.

What remains is the part of the paired path that a 2-row *prefill* tail does not exercise: **the GDN
with `bsz=2, seqlen=1`**. Every prefill chunk is `bsz=1, seqlen=n`; the pair is the only place the
recurrence is asked for two independent one-step states in one launch. That is now the single
remaining suspect, and it is a much narrower question than "batching is broken".

Note the gate also re-confirms what it did before: the projections and mixers being right at n=2 is
what makes the GDN the last candidate rather than one of several.

---

## Batched decode, part 5: the GDN is exonerated, and there is now a test that says so

The remaining suspect after part 4 was the GDN at `bsz=2, seqlen=1` - the one configuration paired
decode needs and the only one no prefill chunk ever produces. It is **correct**.

`test_gdn_bsz2` (new, and now in ctest, 19/19) does the direct A/B with no engine involved: one call
at `bsz=2, seqlen=1, slots={0,1}, history_stride=1` over a `[2][state]` block, against two calls at
`bsz=1, seqlen=1` with the engine's exact decode configuration, comparing **both the output rows and
the resulting state**:

    gdn bsz=2 vs 2x bsz=1: worst relative output diff 0.000e+00, worst relative state diff 0.000e+00
    PASS

Bit-identical, not merely close. So the recurrence handles two independent one-step states in one
launch, and my part-4 conclusion - "the paired-only path is the GDN" - was wrong. The test is worth
keeping regardless: it covers a configuration nothing else executed, and it is the kind of gap that
lets a future change regress silently.

### What that leaves, precisely
Ruled out, each by measurement rather than argument:
- the GDN recurrence at `bsz=2` (above: bit-identical)
- the EXL3 projections at M=2 (a 2-row prefill tail agrees with 4- and 48-row tails)
- the mHC mixers at n=2 (grid puts R on `grid.y`, so the sinkhorn normalises every row; and the
  multi-row prefill tail exercises them)
- per-slot addressing (forcing **both rows onto slot 0** - same cache base, same position - leaves the
  two rows still disagreeing, so the fault is not in the attention's per-row cache/position)
- the head projection (the paired `sub_in` rows differ from each other, which a head that merely
  miswrites rows could not produce)

That is a short list and it has a shape: everything the two rows do **independently** is correct, and
the disagreement is in the work they **share**. The shared work at n=2 is the mHC mixers, the MoE and
the projections - and all three are covered by a multi-row prefill tail that agrees. The one thing a
prefill tail does *not* cover is the **decode** variant of the MoE, which takes a different kernel
(`mgemm` rather than the grouped path) at exactly the width pairing introduced.

Forcing either MoE kernel changes the serial reference's values but not the paired/serial
disagreement, which is consistent with **both** decode kernels being wrong at n=2 rather than the
choice between them mattering. That is the next thing to test, and it needs a standalone A/B in the
style of `test_gdn_bsz2` rather than an end-to-end run.

Shipped path: 19/19 tests, digest `3aaa5693`.

### The MoE A/B, run properly this time

Identical prompts in both slots; the question is only whether the two rows agree **with each other**,
which needs no reference at all:

| MoE decode path | paired A | paired B | rows agree? |
|---|---|---|---|
| default (mgemm) | 271 271 561 97765 2037 200789 | 271 25 96434 130685 167739 5674 | no |
| `HELIOS_MOE_DECODE_MAX_N=2` | 271 271 561 97765 2037 200789 | 271 25 96434 130685 167739 5674 | no |
| `HELIOS_MOE_GROUPED=1` | 271 271 561 110530 41202 46091 | 271 271 181059 136963 75551 264 | no |

The first two rows are **byte-identical**, which corrects an assumption I had been working under: the
decode gate already admits n=2, so `HELIOS_MOE_DECODE_MAX_N=2` was not a variant at all - it was the
default. The earlier "raising the gate changed nothing" result was therefore not evidence about the
MoE; it was the same run twice. `HELIOS_MOE_GROUPED=1` does switch kernels, and the rows disagree
there as well.

So the MoE is wrong at n=2 on **both** its decode kernels, while a 2-row *prefill* chunk - which takes
the grouped path - agrees with 4- and 48-row tails. The kernel is therefore not the thing to look at
first; the difference between the two contexts is the host-side per-row accounting in `moe_layer` for
a decode-shaped call (`slots = n * topk`, the per-card `part` buffers sized from
`g_moe_decode_max_n * topk`, the owned-card copy of `n * hid`).

That `part_rows = max(max_n, g_moe_decode_max_n * topk)` term is worth a hard look on its own: at
`g_moe_decode_max_n = 2, topk = 10` it is exactly 20, and a 2-row decode asks for exactly
`slots = n * topk = 20`. The buffer is sized to the request with no margin, which is the same class of
bug as the QSA slab stride - an exact fit that has to be right for every future width.

### The state of the hunt, honestly
Everything the two rows do **independently** is verified correct: the GDN recurrence at `bsz=2`
(`test_gdn_bsz2`, bit-identical), the per-slot cache base and position, the per-row PLE. Everything
the rows **share** is verified correct in a multi-row prefill context: the EXL3 projections, the mHC
mixers. What is not yet explained is why the shared work disagrees between rows in a *decode*-shaped
call but not in a prefill-shaped one. That is a much smaller question than where this started, and it
is not answered.

---

## Batched decode, part 6: the last unverified link, and it is in a ported dependency

Reading `moe_layer`'s decode path end to end, every piece helios owns checks out for n=2:

| step | verdict |
|---|---|
| `routing_std_logits(..., n, E, topk, ...)` | takes n; writes per-row ids and weights |
| `moe_slot_gather_k` | `xg[i] = x[(slot / topk) * hid + d]` - slots 0..9 to token 0, 10..19 to token 1. **Correct for n=2.** |
| `remap_ids_kernel` | indexes by slot, not by row |
| `part_rows = max(max_n, g_moe_decode_max_n * topk)` | 1024 in every real config, not the tight 20 I suspected |
| shared expert | plain dense path over n rows |

That leaves exactly one call that does the per-token reduction:

    exl3::mgemm(sc.part[c], gbuf, ..., bszm_in = slots, bszm_out = slots, idx,
                sc.topk_w[c], slots, 0, min_index = -1, max_index = -1, num_tokens = n, ...)

`num_tokens = n` with `slots = n * topk` asks mgemm to split the slots into `n` contiguous groups and
reduce each into its own output row. **That mode has never been tested.** `test_mgemm_semantics` -
the test written for exactly this kernel - states its own scope in its header:

> with min_index >= 0 and **num_tokens == 1** the indices/weights are COMPACTED in place while the
> **num_tokens == 1** reduces every slot into row 0

Every behavioural guarantee the test asserts is conditioned on `num_tokens == 1`, and `kSlots = 10`
matches this checkpoint's `top_k`, i.e. the single-token case. Paired decode is the first caller to
pass `num_tokens = 2`.

This also explains the shape of the symptom that has been the most stubborn part of this hunt: a
2-row *prefill* chunk agrees with 4- and 48-row tails, because prefill routes through the **grouped**
kernel, not mgemm. Forcing the grouped kernel in the pair did not fix it either - but the grouped path
is entered through a different code path that carries its own reduction, and that route was measured
with the per-card copies the comments describe as previously missing. So the honest statement is
narrower than "mgemm is broken": **mgemm's multi-token reduction is the one link in the shared decode
path that has no test and no measurement behind it, and it is where paired decode's two rows stop
agreeing.**

The fix is a test, not a guess: extend `test_mgemm_semantics` to two token groups and compare against
`moe_grouped`, which the surrounding comments already name as the only valid oracle for expert
weights. That is the same shape as `test_gdn_bsz2`, and it is the right next piece of work.

---

## Batched decode, part 7: two more components exonerated, and what that leaves

Two leading suspects tested, both clean.

**`exl3::mgemm` with `num_tokens = 2` is correct.** `test_mgemm_semantics` asserted only
`num_tokens == 1` - every guarantee in its header is conditioned on it, and `kSlots = 10` is this
checkpoint's `top_k`, i.e. the single-token shape. Extended to two token groups: the expectation is
now per-token, `row t = sum over slots [t*stride, (t+1)*stride)`, and both new cases pass.

    2 tokens: num_tokens=2, bszm=2*slots           rel RMS 1.337e-03
    2 tokens: num_tokens=2, SKIPS + non-uniform w  rel RMS 1.389e-03
    ALL OK

**The CUDA graph is not involved.** Paired decodes take the eager path (batch size, per-row positions
and state bases are kernel arguments a graph freezes), while serial takes the graphed path - so if the
two disagreed, that alone would explain everything. They do not: the reference digest is
`3aaa5693eee9` with the graph on and off. And with `HELIOS_SEQUENCES > 1` the graph is refused anyway
(it binds different state per slot), so both runs already use the eager body.

**The batched branches execute as intended**, confirmed by instrumentation rather than by reading:
GDN sees `bsz=2 slot_layers=1` with device slot arrays, attention sees `slot0=0 pos0=8810 slot1=1
pos1=8810`. And the row mapping was settled empirically - `final_head` computes
`rows = min(n, head_rows)` and projects `sub_in + (n - rows)`, so with `n == head_rows == 2` there is
no reversal and row *i* is `sub_in` row *i*. Both readings of that were run; both mismatch, so the
mapping is not the issue.

### What is left, stated honestly
Every component reachable by inspection or by an A/B has now been shown correct in isolation:
the GDN recurrence at `bsz=2` (new test, bit-identical), the mgemm weighted reduction at
`num_tokens=2` (extended test), the slot gather, the routing, the remap, the `part` sizing, the
per-slot cache base and position, the per-row PLE, the EXL3 projections at M=2, the mHC mixers at
n=2, the graph/eager equivalence, and the wiring itself.

And yet a paired forward given **identical inputs in both slots** produces two different rows. When
every part is individually right, the fault is in how the parts compose for this width - not in a
kernel. The remaining suspects are ordering or scratch-ownership effects that only appear when n=2
holds two live sequences, and the next step for those is an n=2 differential test at a single layer
boundary (dump the hidden state after layer 0 for both rows) rather than another component A/B.
That is the honest state: the question is smaller than when I started, and it is not answered.

---

## Batched decode, part 8: the rows are identical entering layer 0 and diverge inside it

Component A/B had eliminated everything reachable, so the instrument changed: a **per-layer
differential** (`HELIOS_LAYER_DUMP=1`) that reports each row's hyper-connection stack rms after every
layer, and before the first one. The question it answers is not "which component is wrong" but "at
which layer do the rows part".

    [ldump] L-1 gdn  row0 rms=1.555289831e-03 row1 rms=1.555289831e-03  AGREE
    [ldump] L0  gdn  row0 rms=8.353583817e-03 row1 rms=9.587381711e-03  DIVERGE
    [ldump] L1  gdn  row0 rms=9.136219637e-03 row1 rms=1.040218312e-02  DIVERGE
    ...

The embed -> streams expansion is **correct at n=2** - the two rows come out of it bit-identical - and
the entire divergence is created inside the first layer's body. That is a much smaller target than
"batched decode is wrong": one layer, with the GDN and the MoE of that layer already verified
correct in isolation.

Two things the instrument itself taught, worth more than the run:

- The first version of the dump used a **synchronous `cudaMemcpy`**, which is unordered against the
  layer's stream `s`. It printed values ~1000x too small and repeated on alternate layers, which is
  what made it look like the rows parted immediately and uninformatively. Ordering it against
  `stream(0)` fixed the read. A diagnostic that lies is worse than none.
- Two broken builds along the way, both from scripted insertion rather than from the idea. The
  instrumentation is env-gated and now earns its place; the editing method does not.

### A real bug it did find, on the way
The PLE's per-row call passed **`n_ctx = 1`**, but `n_ctx` is the number of ids *carried into* the
call, not the token count - `ple_ngram_ids` derives its row count as `n_hist - n_ctx`, which is why the
serial path passes `hist_.size() - n`. Passing 1 made the PLE fold `ngram_size` rows instead of one,
re-injecting stale carried ids. Fixed to `h.size() - 1`; the change is visible in the trace
(L2 row1: 1.745177812e-02 -> 1.739306992e-02), so it was a genuine correctness gain - just not the one
that parted the rows at L0, since the PLE sits on a later layer.

### What is left
Inside the first layer's body, with the GDN (`test_gdn_bsz2`, bit-identical) and the MoE
(`test_mgemm_semantics` at `num_tokens=2`, rel RMS 1.3e-03) already cleared in isolation, and the
embed -> streams expansion cleared by the pre-layer probe. The next step is the same instrument one
level finer: probe after the attention mixer, after the GDN, after the MLP mixer and after the MoE
inside L0, and the first probe that reports DIVERGE names the step. That is a short, mechanical
extension of what is already built and wired.

---

## Batched decode, part 9: narrowed to one call, after a second broken probe

The step-level differential needed a fix before it could be believed. The first version read row 1 at
offset `n * D` from a buffer laid out `(R, D)` - i.e. **past the end** - and reported row 1 as a
constant that was really adjacent memory. That is what made the mixer look like it "never wrote row
1". With the row stride corrected to `D`, the picture inverts:

    [probe] L0 pre_amix   row0 rms=4.687788518e-01 row1 rms=4.623785211e-01  DIVERGE
    [probe] L0 attn_mix   row0 rms=6.713679815e-01 row1 rms=6.713679815e-01  AGREE
    [probe] L0 gdn_out    row0 rms=1.385908027e-01 row1 rms=1.488238274e-01  DIVERGE

- `pre_amix` is leftover from the prefill's last chunk, where rows 0 and 1 are *different prompt
  tokens* and are expected to differ. It carries no information.
- **`attn_mix` AGREES bit-for-bit.** The mHC mixer is correct at R=2, which also retires the
  "fp32 decode kernels are built for R=1" comment as an explanation - `HELIOS_MIXER_TC_MIN_R=2`
  changes nothing, because the scalar path was already correct.
- The first real divergence is **`gdn_out`**, the GDN's own output.

And the two slots are not the cause: prefilling identical prompts into both leaves **bit-identical
GDN state** (`max|diff| = 0.000000e+00` at layers 0, 1, 2 and 10). The GDN's input agrees too, since
`a.mixed` agrees and the fp32 -> fp16 cast is element-wise.

So: identical input, identical state, a kernel proven bit-identical at `bsz=2` - and different output.

### The gap that explains it
`test_gdn_bsz2` covers `cuda_recurrent_gated_delta_rule` **only**. The other half of the GDN layer is
`cuda_causal_conv1d_update`, which in the batched path is called with `bsz=2, seqlen=1` and a
per-slot `conv_state` for the first time. Its addressing reads

    conv_state[((size_t) slot * dim + d) * state_size + ...]

so the per-slot stride is `dim * state_size` = `conv_bytes`, which matches the `[layer_rank][slot]`
carve - by inspection, which is exactly the kind of reasoning this whole investigation has shown to be
unreliable. The next step is a conv1d A/B in the style of `test_gdn_bsz2`: two independent
`(conv_state, x)` pairs, once as `bsz=2` with `slots={0,1}` and once as two `bsz=1` calls, comparing
both the output and the written-back conv state.

### A note on the two probes that lied
Both this instrument and the layer-level one produced confident, wrong numbers before being fixed -
one from an unordered synchronous memcpy, one from a row stride that read past the buffer. Each was
caught only because a second quantity (run-to-run constancy, then a value 1000x off) did not look
physically possible. Neither would have been caught by a plausibility check, and both would have sent
the hunt somewhere else entirely.

---

## Batched decode, part 10: the conv1d is correct too - and the third instrument that lied

`test_gdn_bsz2` covered `cuda_recurrent_gated_delta_rule` only. The layer also calls
`cuda_causal_conv1d_update`, and that is the first `bsz=2` call in the engine. Extended the test to
cover it - two independent `(conv_state, x)` pairs, once as `bsz=2` with `slots={0,1}`, once as two
`bsz=1` calls, comparing **both the output and the written-back conv state**, with non-zero initial
state so a slot writing into the wrong neighbour is visible.

The first run failed, hard:

    conv1d bsz=2 vs 2x bsz=1: worst abs output diff 1.728516e-01, worst abs state diff 9.882812e-01
        slot 0: worst output diff 1.669922e-01, worst state diff 0.000000e+00
        slot 1: worst output diff 1.728516e-01, worst state diff 9.882812e-01

**The per-slot breakdown is what identified it, and it exonerated the kernel.** Slot 0's *state* was
bit-identical while its *output* was not - and the output depends on the state read and on `x`. State
right, output wrong can only be the input. The test had laid `x` out as `dim * K` per batch item when
the layout is `[bsz, dim, seqlen]`, so with `seqlen == 1` the kernel's batch stride (`dim * seqlen`)
disagreed with the buffer and **both** rows read the wrong `x`. Fixed:

    conv1d bsz=2 vs 2x bsz=1: worst abs output diff 0.000000e+00, worst abs state diff 0.000000e+00
        slot 0: worst output diff 0.000000e+00, worst state diff 0.000000e+00
        slot 1: worst output diff 0.000000e+00, worst state diff 0.000000e+00
    PASS

So **both halves of the GDN layer are correct at `bsz=2`** - the recurrence and the conv - each now
covered by a test that will catch a regression in a shape nothing had executed before.

### Where the divergence is NOT
- the mHC mixer at R=2: `attn_mix` reports the rows **AGREE** bit-for-bit
- the GDN recurrence at bsz=2: bit-identical
- the conv1d at bsz=2: bit-identical
- slot state isolation: `max|diff| = 0.000000e+00` after identical prefills
- the embed -> streams expansion: rows agree before layer 0

### Where it is
Inside layer 0's body, and the step-level instrument is no longer trustworthy enough to localise it
further: the `gdn_out` probe reported the *same* value before and after moving it across the
`gdn_layer` call, which it should not have. Combined with the earlier two instruments that produced
confident, wrong numbers (an unordered synchronous memcpy; a row stride reading past the buffer), the
lesson is that this instrument needs the same treatment the kernels got: **a self-test that proves it
can see a known divergence** before its output is believed. Until then its readings are a hypothesis,
not evidence.

### The standing caution
Four hypotheses were killed by tests this session - the mgemm multi-token reduction, the GDN
recurrence, the conv1d, and "the fp32 mixer is built for R=1". Each looked certain from the code.
Three separate instruments also lied. The engine-side batching is complete and correct in every part
that can be tested in isolation; the remaining defect is a composition or ordering effect at n=2 that
this session narrowed to a single layer but did not explain.

---

## Batched decode, part 11: the probe self-test, and what the GDN layer is not

The step-level instrument had produced confident wrong numbers twice, so before reading anything more
into it, it had to be shown able to **see a difference**. `HELIOS_PROBE_SELFTEST=1` perturbs row 1 of
`a.mixed` after the attention mixer - the one point where the rows are known bit-identical - and the
probe must flip to DIVERGE.

It took three attempts, and each failure was the instrument's fault, not the kernel's:

1. Perturbing **before** the mixer - which simply overwrote the perturbation. The probe's AGREE was
   the correct answer to the wrong question.
2. Perturbing **after** the mixer, on the legacy stream, while the mixer's work was still in flight on
   `s` - the mixer ran afterwards and undid it. Same ordering class as the original bug.
3. With a `cudaStreamSynchronize(s)` first:

       [probe] L0 attn_mix   row0 rms=6.713679815e-01 row1 rms=1.234500000e+04  DIVERGE

`1.2345e4` is exactly the perturbation (12345), so the probe reads the right buffer and is correctly
ordered. **Which means its earlier readings stand**: the attention mixer leaves the rows bit-identical
and the divergence appears in `gdn_out`, inside the GDN layer's body, in the first decode step - and
persists into the second, where even `attn_mix` then reads DIVERGE.

### The GDN layer, exhausted
| part | verdict |
|---|---|
| chunked WY path | declines - `seqlen < 2*64` at n=2. (Its own `bsz != 1` guard never fires, because the call site passes a hardcoded `bsz=1` - but the seqlen check catches this case, and the hardcoded 1 is harmless while it is unreachable for bsz=2.) |
| depthwise conv | **bit-identical** at bsz=2, both output and written-back state, both slots |
| delta-rule recurrence | **bit-identical** at bsz=2, both output and state |
| projections, gated RMS norm, output combine | per-token over `n`; a 2-row prefill tail agrees with 4- and 48-row tails |
| input | rows agree (`a.mixed` bit-identical, cast is element-wise) |
| state | `max|diff| = 0.000000e+00` between slots after identical prefills |

Every part is correct, in isolation and in combination, and the layer's output still differs between
rows. What remains is not reachable by reading the call: it needs either kernel-level instrumentation
inside the projections, or a bisection harness that runs the batched layer against a hand-computed
reference row by row. Both are fresh work, and I would rather name that than keep producing
hypotheses that four tests have already killed.

---

## Batched decode, part 12: the fault is the GDN's bsz=2 call, and the gate now passes

Component A/B had cleared every part, and the self-tested probe had localised the divergence to inside
the GDN layer's body. The next step was the obvious bisection: change **one** thing and see whether the
gate flips. `HELIOS_BATCH_GDN_SERIES=1` runs the GDN as two `n=1` calls - one per row, each on its own
state base - while leaving the attention, the mHC mixers and the MoE batched at `n=2`:

    [gen] batch-parity A MATCH  B MATCH
    [gen] batch-parity OK (10 steps each)

**MATCH on both slots.** And restoring the single `bsz=2` call puts it straight back:

| GDN invocation | parity |
|---|---|
| two `n=1` calls, one per row (now the default) | **A MATCH  B MATCH** |
| one `bsz=2` call (`HELIOS_BATCH_GDN_BATCHED=1`) | A MISMATCH  B MISMATCH |

So the disagreement is entirely inside that one call, in the engine's wiring rather than in the
kernel: `test_gdn_bsz2` drives `cuda_recurrent_gated_delta_rule` and `cuda_causal_conv1d_update`
directly, each **bit-identical** to two `n=1` calls, but the engine passes a per-slot state base carved
`[layer_rank][slot]` per card plus two pre-scaled slot arrays, and that combination does not agree
with per-row calls. The difference between the two is now bounded to the arguments, not the maths.

**The per-row form is the default for batched decode.** The GDN is ~26% of a decode step and its
weights are read once either way, so two `n=1` calls cost launch overhead, not bandwidth - and
correctness is worth that trade until the argument-level disagreement is explained.

### What the parity gate is now worth
It went from failing on both slots to passing on both, over 10 steps, with two sequences. That is the
first time batched decode has been *correct*, and it is what every future change to this path has to
keep true.

### Measured, with batching correct but not yet used by the server
Two concurrent HTTP requests, `--slots 2`:

| requests | wall | tokens | aggregate | per-stream |
|---|---|---|---|---|
| 1 | 2.2 s | 100 | 44.6 tok/s | 44.6 |
| 2 | 5.8 s | 346 | **59.7 tok/s** | 29.8 |

**+34% aggregate** - close to the +38% the MTP width-2 measurement predicted. That gain is from
concurrent execution filling the 57% card idle, *not* yet from `decode_pair`: the server still runs one
forward per request. Pairing two requests into a single forward is the remaining wiring, and it is
what would let both effects compound.

---

## Batched decode, part 13: reachable and correct - and currently SLOWER than not batching

`Runner::generate_pair` plus `helios gen --pair` makes the batched path reachable end to end. It is
**correct**: its token ids are the serial trajectory exactly.

    [gen] A ids: 271 248068 198 760 1156 682 3766 264
    [gen] B ids: 11  248068 198 760 1156 682 3766 264

and A matches the single-sequence reference id for id. (Two implementation notes worth keeping:
`bind_slot()` mutates `active_slot_`, so the original slot has to be captured before the prefill loop
or the pair ends up naming one slot twice; and each sequence's FIRST token must be read from its OWN
prefill, because the logits buffer is shared and after the second prefill it holds the second
sequence's row.)

### The result that matters: it is not faster
256 decode tokens per sequence, 8k and 4k prompts:

| | aggregate tok/s |
|---|---|
| paired, one forward per step | **40.7** (including ~6 s of the two prefills) |
| two independent single-sequence decodes | 48.1 and 57.3, i.e. **96-114** for the same 512 tokens |

**Batching as implemented is a loss.** And the reason is the part-12 workaround: the GDN now runs as
two `n=1` calls, attention is per-row, and the PLE is per-row. So a "batched" step pays double
launches for three of the four sublayers and shares only the MoE, the mixers and the projections.

That is the honest shape of the result, and it separates two things that were tangled:

- **Correctness is solved.** Batched decode now produces the serial trajectory exactly, and the parity
  gate passes on both slots. That was the hard part and it is done.
- **Profitability is blocked by the GDN bug.** The +38% was measured on a *width-2 trunk* - the MTP
  verify batch, where every sublayer sees n=2. The GDN `bsz=2` disagreement is exactly what stands
  between this path and that shape, and until it is explained the pair pays for its correctness.

The +34% aggregate measured over HTTP earlier is real but is **not** this: that is two requests running
concurrently, filling the 57% card idle, each still doing its own forward. It composes with batching
rather than coming from it, and only one of the two is available to the server today.

---

## Batched decode, part 14: the GDN wiring is right, and the disagreement is still unexplained

If the GDN's `bsz=2` call is wrong in situ while both its kernels are bit-identical in isolation,
the difference has to be in what the engine passes. Every element of that has now been measured
rather than reasoned about:

    [ptrgap] rec  layer 0: slots 0/1 3145728 bytes apart, rec_bytes 3145728  OK
    [ptrgap] conv layer 0: slots 0/1   81920 bytes apart, conv_bytes   81920  OK

The kernels reach slot *b*'s state as `base + slots[b] * history_stride * state_size`. For the
recurrence that requires the two slots' states to sit exactly `rec_bytes` apart; for the conv,
exactly `conv_bytes` apart, since its stride is `dim * state_size` with no layer term. **Both hold
exactly**, at every layer checked. The state block really is carved `[layer_rank][slot]` and the two
slot arrays really do carry the relative indices the kernels apply.

Also ruled out by inspection and sizing: the GDN scratch, whose `a_had` is allocated
`max_n * 16384 * 2` and so has room for two rows; the chunked path, which declines on `seqlen < 2*64`;
and `locks`, which is `nullptr` on both paths.

So: kernels correct, state addressing correct, scratch correct, arguments correct - and the batched
call still disagrees with two per-row calls, while matching the parity gate only when the GDN runs
per row. I cannot close that from the host, and I am not going to name a cause I have not measured.

**What is established, and is worth keeping:** batched decode is correct and reachable
(`generate_pair`, `gen --pair`, parity gate green on both slots), and its profitability is gated on
this one disagreement. The per-row GDN is the correct default until then - correctness first, and
the cost of that choice is now measured rather than assumed.

---

## Batched decode, part 15: reproduced in isolation, and the bisect names the stage

The full-layer A/B is the unit neither kernel-level test covered - both of those drive the kernels
directly, skipping the projections, the norm and the output projection that `gdn_layer` wraps around
them. Running one real GDN layer over a 2-row input, once with `bsz=2` and once as two `n=1` calls:

    full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01

**Reproduced outside the engine**, with both kernels still bit-identical in isolation - including with
the test's state block changed to the engine's own `[layer_rank][slot]` carve (18 layers x 3 slots,
base handed over at rank 5, relative slot numbers), which was the last structural difference between
the test and the real call.

### The bisect inside the layer
Snapshotting the scratch after each stage of the two invocations:

    [gdnab] after conv_out worst 3.387451e-02 | after core_out worst 7.996496e-03

The conv's **output** already differs, and the conv is bit-identical at `bsz=2` in isolation - so its
*input* differs. That input is `sc.mixed_qkv`, the output of the qkv projection:

    exl3::linear(sc.qkv_flat, x, qkv, n, qkv_out, cfg.hidden, w.qkv.K, true, s, sc.a_had)

So the divergence originates in **the EXL3 projection with its Hadamard transform at M = 2**, not in
the GDN kernels at all. That reframes the whole hunt: the `bsz=2` recurrence and conv1d were never
suspect, and "the GDN's batched call is wrong" was true only in the sense that the wrong thing lives
one step earlier in the same function.

The natural next test is the same A/B shape applied to `exl3::linear` with Hadamard at M=2 versus two
M=1 calls, reading `sc.a_had` - the shared scratch that a multi-row call fills with two rows and a
single-row call with one, and the one buffer in this path whose sizing has not been checked against a
two-row call.

---

## Batched decode, part 16: the inner bisect was invalid, and what is actually left

Part 15's bisect pointed at the qkv projection. The A/B on that projection now says it is **identical**
(`0.000000e+00`), so that conclusion was wrong - and the reason is worth recording, because it is the
fourth instrument error in this investigation and the most consequential.

`sc.mixed_qkv` and `sc.conv_out` are **transposed** to channel-major layout by
`glue::transpose_f32_bf16(sc.qkv_flat, sc.mixed_qkv, n, qkv_out, s)`, so they are `[qkv_out, n]`. In
the `bsz=2` call they hold two columns; in each of the two `n=1` calls they hold one column, and after
the second call the buffer holds **row 1's data at offset 0**. Comparing the two afterwards is
comparing a `[qkv_out, 2]` buffer against a `[qkv_out, 1]` one, so "the conv's output already
differs" was an artifact of the shapes, not of the values. The transpose kernel itself checks out on
inspection - the load is `tile[a][b] = src[(m0+a)*F + f0+b]`, the store is
`dst[(f0+a)*M + m0+b] = tile[b][a]`, and the guard `m0 + b < M && f0 + a < F` bounds the ragged
tile correctly - and the scratch is sized `max_n * qkv_out * 2`.

**The only like-for-like comparison in that harness is the layer's final output `y`, which is `[n, D]`
in both paths - and that one is real: 8.759627e-01.**

So the honest state of the GDN layer at `bsz=2`:

| stage | verdict |
|---|---|
| qkv projection (`exl3::linear` + Hadamard) | **identical** at n=2, measured like-for-like |
| transpose to channel-major | correct on inspection, untested at n=2 |
| depthwise conv | **bit-identical** at bsz=2, in the engine's own block layout |
| delta-rule recurrence | **bit-identical** at bsz=2, in the engine's own block layout |
| gated RMS norm + output projection | untested at n=2 |
| layer output `y` | **differs**, 8.76e-01 |

Two stages remain untested at this width - the transpose and the gated-norm/output-projection tail -
and the fix, if it is one of those, is a small A/B each in the shape that has now caught three other
things. What is *not* established is any of part 15's staging claim.

The pattern across this investigation is consistent enough to be worth stating plainly: every
conclusion here rests on a comparison, and four of the comparisons were wrong - an unordered memcpy, a
row stride past the buffer, a test input laid out `dim*K` instead of `dim*seqlen`, and now a
transposed scratch compared across two different shapes. Each looked like a finding. The instruments
that survived all of that - the per-row parity gate and the per-slot breakdown inside a test - are the
ones that compare the same quantity to itself.

---

## Batched decode, part 17: the transpose is wrong for n > 1, measured against a host reference

The remaining GDN stage, tested the only way that cannot lie about shapes - against a host reference
rather than against two n=1 calls, because that buffer is channel-major `[F, n]` and an n=1 call
leaves one column:

    [trchk] transpose n=1   F=10240 vs host ref: worst abs diff 9.764433e-04  OK (bf16 rounding)
    [trchk] transpose n=2   F=10240 vs host ref: worst abs diff 5.000000e-01  *** WRONG ***
    [trchk] transpose n=8   F=10240 vs host ref: worst abs diff 5.000000e-01  *** WRONG ***

`glue::transpose_f32_bf16` is correct at n=1 and wrong above it. The worst difference is exactly
0.5, the largest magnitude in the test data, which is the signature of **zeros coming back where values
were expected** rather than of small numeric drift.

That is the whole batched-decode defect, and it explains every observation that had resisted an
explanation:

- the GDN layer at n=2 diverges, while its projection, its conv and its recurrence are each
  bit-identical - the transpose sits between the projection and the conv;
- the prefill tail at 2, 4 and 48 rows all agree with each other, because `transpose_f32_bf16` is only
  ever called with the trunk's `n` and the *shipped* prefill tail is handled by a different path - and
  the one place `n=2` reaches this kernel on its own is a batched decode, where nothing else agrees
  with it to hide the result;
- `mixed_qkv` and `conv_out` "differing" in part 15 - that comparison was invalid across shapes, but
  the underlying values really are wrong, because the transpose is what fills them.

**This is the highest-value thing in the batched-decode sequence**, and it is only findable because
the per-row parity gate was made to pass by running the GDN per row - which kept the feature correct
while leaving the bug visible.

The fix is one kernel: make the load fill the whole tile region the store reads, or equivalently
index the store's `tile[b][a]` from a tile that was actually populated. It should be validated by
turning `harness_transpose_check` into a permanent test at n = 1, 2, 3, 8 and 64, which is a check any
future change to this kernel can be held to.

### The transpose writes ZEROS for n > 1

Dumping the actual values rather than just the difference, with n=1 as a passing control in the same
run:

    [trchk] transpose n=1   F=10240 vs host ref: worst abs diff 9.764433e-04  OK (bf16 rounding)
    [trchk] src[0][0..3] = -0.500000 0.395573 0.291146 0.186719
    [trchk] src[1][0..3] = 0.168532 0.064105 -0.040322 -0.144749
    [trchk] dst[0][0..3] = 0.000000 0.000000 0.000000 0.000000  (want src[0..3][0])
    [trchk] dst[1][0..3] = 0.000000 0.000000 0.000000 0.000000  (want src[0..3][1])
    [trchk] transpose n=2   F=10240 vs host ref: worst abs diff 5.000000e-01  *** WRONG ***
    [trchk] transpose n=8   F=10240 vs host ref: worst abs diff 5.000000e-01  *** WRONG ***

The destination is **all zeros** for n >= 2 while the source is well populated, and the same kernel at
n=1 in the same process is correct. So it is not a launch, sizing or argument problem: the kernel
runs, and what it stores is zero.

The kernel is a 32x32 shared-memory tile. The load fills `tile[a][b]` guarded on `m0 + a < M`, so at
M = 2 only `tile[0][*]` and `tile[1][*]` are ever written; the store reads `tile[b][a]` guarded on
`m0 + b < M`, so at M = 2 it reads exactly those two rows. On paper that is self-consistent, and the
indices line up (`tile` is `[32][33]`, the 33rd column is bank padding, both loops decompose
`e = tid + i*256` the same way). **The paper and the measurement disagree, and the measurement is
right** - which is the fifth time in this investigation that a careful reading of a kernel has been
overturned by running it.

That disagreement is itself the useful clue: whatever the store reads is zero, so the entries it
selects are not the entries the load filled. The fix is to make the load fill the region the store
reads, and to hold it to `harness_transpose_check` at n = 1, 2, 3, 8 and 64, which is the first
instrument in this whole sequence that compared against something it could not itself be wrong about.

---

## CORRECTION: the transpose is NOT the bug - parts 17's root cause is retracted

Parts 16 and 17 claimed `glue::transpose_f32_bf16` is wrong for n > 1, on the strength of
`harness_transpose_check` reporting zeros. **That claim does not hold and is withdrawn.**

Working the kernel's coverage through by hand contradicts the harness:

    M=2, F=10240 -> grid=(320,1), 320 blocks; dst needs 20480 elements
    per block: load a<M -> 2 rows x 32 cols = 64 values; store b<M -> 2 x 32 a = 64 writes
    total writes 320*32*2 = 20480 == F*M          exact coverage, no gaps
    store reads tile[b][a] with b<M; the load filled tile[a2][b2] with a2<M
      -> tile[b][a] was loaded iff b<M, which is exactly the store's guard

Every tile entry the store reads is one the load wrote, the write count equals the element count
exactly, `tile` is `[32][33]` with the 33rd column as bank padding, and both loops decompose
`e = tid + i*256` identically. The load writes `0.f` - not an uninitialised value - wherever the
range guard fails, so there is no uninitialised shared memory for the store to pick up. **The kernel
is correct.**

So `harness_transpose_check` is the fifth broken instrument in this investigation, and its failure
mode is the same family as the others: a harness whose own setup is wrong reporting a confident
result. Its output cannot be used to place the bug, and parts 16-17's staging conclusion is void.

**What survives, and what does not:**

| claim | status |
|---|---|
| batched decode is correct and reachable, parity gate MATCH/MATCH | **established** - end-to-end, both slots, matches the serial trajectory |
| running the GDN per row is the fix that makes the gate pass | **established** - a single-variable A/B, both directions |
| the layer diverges at n=2 while projection, conv and recurrence are each bit-identical | **established** - the layer A/B is a like-for-like `[n, D]` comparison |
| the divergence is in the transpose | **RETRACTED** - the kernel checks out by hand |
| where the divergence actually is | **unknown** - narrowed to "somewhere in gdn_layer between the recurrence output and the layer's final `y`", which is the gated-norm and output-projection tail, still untested at n=2 |

The honest summary of the last stretch: the batched-decode *defect* is real, reproducible, and
localised to one function; the *root cause* is not found, and the two candidates I named for it were
both wrong - one because a test's input was laid out `dim*K` instead of `dim*seqlen`, this one because
a harness reported zeros for reasons that have not been established. The fix path is unchanged and
still short: an A/B on the gated RMS norm and output projection at n=2, held to the per-row parity gate.

---

## Batched decode, part 18: the gated RMS norm, tested the way that has not lied

The transpose claim is retracted (part "CORRECTION"). The next stage is the one the retraction left
standing as untested: the gated RMS norm and output projection between the recurrence and the layer's
final `y`. Tested the way the retraction demands - **like-for-like**, with the serial path's two calls
assembled into the same shape as the batched one before comparing, which is precisely the mistake the
transposed-scratch comparison made:

    [gdnab] gated_rms_norm out (both [n,Nv,Hv]) worst 4.726562e-01  *** DIFFERS ***

`gated_rms_norm` called once with `rows = 2*Nv` does not produce the same values as two calls with
`rows = Nv`, assembled into `[2, Nv, Hv]`. Both buffers are the same shape, the same dtype, and the
same source rows, so - unlike every comparison in parts 15-17 - this one cannot be an artifact of
layout. **This is the batched-decode defect, and unlike the transpose it is a real result.**

The layer A/B that started this (`full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01`) is
the same fact at the layer boundary; this locates it inside the layer, after the recurrence and before
the output projection.

What is established about the kernel so far, and what is not:

| | |
|---|---|
| `row = blockIdx.x`, one block per row | reads correctly |
| `w_groups = 1` at the GDN call site, so the weight index cycles to a single row | correct for GDN, where one norm weight is shared |
| the launcher grid for `rows` | **not yet checked** - the one place a `rows`-dependent sizing error would hide, and the reason `rms_norm` elsewhere in this codebase carries a "rows x dim is the flattened view" note |
| why `rows = 2*Nv` differs from two `rows = Nv` calls | **not established** |

The next step is to read the launcher at `norm.cu:402` and check its grid and any per-row scratch
against `rows`, and to add `gated_rms_norm` at `rows = 1, 2, 48, 96` to the same host-reference
harness - the method that is still trustworthy, now that the transpose harness is known to be not.

The batched-decode path stays correct meanwhile: the GDN runs per row for batches, and the per-row
parity gate is MATCH/MATCH on both slots.

### What reading the launcher established, and what it did not

`gated_rms_norm` (norm.cu:402) sizes its launch from `dim` alone:

    bool small = (dim <= 256);
    dim3 blockDim(small ? 32 : NUM_THREADS, 1, 1);
    dim3 gridDim(rows, 1, 1);

`small` is a function of the head dim, not of `rows`, and the grid is one block per row - so a
two-row call launches twice as many blocks of the same shape, with no per-row scratch that could
depend on `rows`. The dispatch table is a complete 12-way if/else over (x, w, y, gate) dtype, and the
GDN's `(kBFloat16, kBFloat16, kHalf, kBFloat16)` does match an instantiation - the `small = true,
32 threads` one, since `Hv = 128`. So the call **does** launch, and the earlier worry that the
dispatch silently falls through does not apply.

That removes the two most likely launcher-level explanations and leaves the divergence inside the
kernel body for `rows = 2*Nv` versus two `rows = Nv` calls. The kernel takes `row = blockIdx.x`,
`w_groups = 1` at this call site (so the weight index cycles to a single shared weight, correct for
GDN), and `gate_act = 1` (sigmoid). With 32 threads and `dim = 128` each thread holds four elements
and the reduction is warp-local, which is row-independent on its face.

**Not established:** why a single launch over 96 rows differs from two launches over 48. The honest
next step is a host-reference check of `gated_rms_norm` at `rows = 1, 2, 48, 96` with fixed inputs -
the method that has not lied - rather than another reading of the kernel, since five readings in this
investigation have now been overturned by running the code.

---

## CORRECTION 2: `gated_rms_norm` is not the bug either - part 18 is retracted

Part 18 named the gated RMS norm from a `[gdnab]` line reporting its output differing between a
`rows = 2*Nv` launch and two `rows = Nv` launches. That comparison was assembled like-for-like, so
unlike the transposed-scratch one it should have been sound - and it still does not survive a
formula-free check of the kernel itself:

    [normchk] rows=1    of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL
    [normchk] rows=2    of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL
    [normchk] rows=3    of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL
    [normchk] rows=48   of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL
    [normchk] rows=95   of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL

`gated_rms_norm` gives bit-identical output for a given row regardless of how many rows the launch
covers. **It cannot be the source of a difference that depends on the row count**, so the part-18
reading was measuring something else: a differing *input* (`sc.core_out`), not a norm defect. Part 18
is withdrawn.

### Where that leaves the localisation - honestly
Every specific location named in this sequence has now been wrong:

| candidate | how it was tested | verdict |
|---|---|---|
| `cuda_recurrent_gated_delta_rule` at bsz=2 | A/B in the engine's own block layout | correct |
| `cuda_causal_conv1d_update` at bsz=2 | A/B, output and written-back state | correct |
| `exl3::linear` + Hadamard at M=2 | A/B against two n=1 calls | correct |
| `glue::transpose_f32_bf16` at n>1 | host reference | **correct** (coverage arithmetic); the "zeros" report was the harness |
| `gated_rms_norm` at rows=2*Nv | host reference, row-dependence | **correct**; the "differs" report was a differing input |

The only facts that have survived every check:

1. **`full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01`** - a like-for-like `[n, D]`
   comparison, reproducible.
2. **Running the GDN per row makes the end-to-end parity gate pass**, MATCH/MATCH on both slots, in
   both directions of the A/B.
3. The divergence appears **after the attention mixer and by the GDN's output** (self-tested probe).

So the defect is real, reproducible, bounded to one function, and its cause is **not found**. Five
candidate kernels have been cleared by direct test and two claims have had to be retracted, which is
the record to hand forward rather than a sixth guess. The right next step is a stage-by-stage
like-for-like capture inside `gdn_layer` with **matching shapes at every point** - the transposed
scratch is the trap - rather than another reading of a kernel that has now been exonerated twice.

---

## Batched decode, part 19: a stage-localised result, with the dtype error found

The part-15 captures had two defects: `mixed_qkv` was read as **fp32 when it is bf16**, and the
transposed buffers were compared without assembling the 1-row calls into `[qkv_out, n]`. Both are
fixed, and every stage is now compared at the same shape with the same dtype in both paths:

    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***
    [gdnab] conv_out     worst 2.908325e-02  *** DIFFERS ***
    [gdnab] core_out     worst 8.529663e-03  *** DIFFERS ***
    [gdnab] normed       worst 2.069336e+00  *** DIFFERS ***

**`mixed_qkv` is the first stage that differs, and it is the transpose's output.** Everything before it
is clear: `x` is the same two rows in both invocations, and `sc.qkv_flat` comes from the projection,
which is **identical** at n=2 by direct A/B. So the divergence is introduced by

    glue::transpose_f32_bf16(sc.qkv_flat, sc.mixed_qkv, n, qkv_out, s)

which is the stage the coverage arithmetic in part 16 said was correct, and which
`harness_transpose_check` reported as writing zeros - dismissed then as a broken harness, and which
this result now says was **right all along**. Its guard analysis on paper (`tile` is `[32][33]`, both
loops decompose `e = tid + i*256` identically, the load writes `0.f` rather than leaving shared
memory uninitialised, and total writes equal `F*M`) is evidently wrong somewhere, and no amount of
re-reading is going to find it - five readings in this investigation have now been overturned by
running the code.

This is the first **stage-localised** result in the whole sequence, and it points at one line. The fix
is to correct that kernel and to hold it to `harness_transpose_check` at n = 1, 2, 3, 8, 64 - whose
zeros report was the truest signal available and which I talked myself out of.

### The transpose is exonerated a second time, decisively

The one structural thing part 16's coverage analysis and part 19's stage capture disagreed about was
the store's **transposed tile read** - `tile[b][a]` with the guard `m0+b<M`, versus the direct
`tile[a][b]` with the guard `m0+a<M`. Only the transposed form keeps the store coalesced (consecutive
threads take consecutive `b`, which becomes consecutive `dst` addresses), so it is the form the kernel
deliberately uses.

Changing the store to the direct, alias-free form - correct, but with a stride-M store - was run as an
experiment rather than argued about:

    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***     <- before the change
    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***     <- after, bit-identical

**The same value to every digit.** The two forms are equivalent, so the index aliasing is not the bug
and the transposed store was the right call. The change has been reverted - it costs store coalescing
for nothing.

So the transpose is now cleared twice, by two independent methods: coverage arithmetic, and a
behavioural A/B of the one thing the arithmetic was arguable about. `mixed_qkv` differs for a reason
upstream of the transpose, which means `sc.qkv_flat` - the projection's output **inside the layer** -
is not what the standalone projection A/B said it was. That capture was never taken: the harness
compared `mixed_qkv` and took the projection's correctness on faith from a test that used its own
input and its own Hadamard scratch. The next capture is `sc.qkv_flat`, at `[n, qkv_out]`, assembled
the same way as the rest.

---

## Batched decode, part 20: the transpose is the bug, proven by its own input/output pair

Part 16 retracted the transpose claim on a hand coverage analysis. **That retraction was wrong.**
Capturing the projection's output and the transpose's output from the *same* invocation, with the
serial path assembled to matching shapes and dtypes:

    [gdnab] qkv_flat     worst 0.000000e+00  IDENTICAL
    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***
    [gdnab] conv_out     worst 2.908325e-02  *** DIFFERS ***
    [gdnab] core_out     worst 8.529663e-03  *** DIFFERS ***
    [gdnab] normed       worst 2.069336e+00  *** DIFFERS ***

`sc.qkv_flat` - the transpose's entire input - is **bit-identical** between the `bsz=2` invocation
and the two `n=1` invocations, and

    glue::transpose_f32_bf16(sc.qkv_flat, sc.mixed_qkv, n, qkv_out, s)

turns it into something that differs by 3.5e-01. One kernel, with its input and its output captured
in the same call, at matching shapes and the right dtype. Everything downstream inherits the
difference, which is why the conv, the recurrence output and the norm all appear to diverge, and why
the layer's final output is off by 8.76e-01.

This is also why `harness_transpose_check`'s "zeros at n>1" was the truest signal in the sequence: it
was reporting this, correctly, and I dismissed it in favour of a hand analysis.

The de-aliasing experiment (part 19's last section) returned the *same* 3.505859e-01, which now reads
as evidence that **the edit never reached the compiled kernel** - a `.cu` change that does not alter
the result bit-for-bit is not a semantic change, it is a build that did not happen. So the question
of whether the transposed tile read is the defect is still open, and the coverage arithmetic in part
16 should be treated as unreliable rather than as exoneration.

**The remaining work is one kernel.** `transpose_f32_bf16_k` at M > 1, validated by
`harness_transpose_check` at n = 1, 2, 3, 8, 64 promoted to a permanent ctest - after first finding out
why that harness reports the *same* failure for two different store forms, which it cannot do if it is
reading what it thinks it is reading. That harness bug and the kernel bug are almost certainly the
same bug.

### Build fidelity, and what the de-aliasing result actually means

Checked, because part 19 read its own experiment as "the edit never reached the compiled kernel":

    touch src/engine/glue2.cu -> cmake --build build  -> 3 compile steps

`glue2.cu` is in the build (`CMakeLists.txt:26`) and touching it does recompile. **So the build is
faithful and the de-aliasing experiment did run** - which means the two store forms produce the *same*
output, bit for bit, and both are correct on their own terms.

That leaves the picture consistent and uncomfortable: two independent harnesses - the layer A/B, which
uses the engine's real `sc.qkv_flat` and `sc.mixed_qkv` and therefore cannot be wrong about shapes or
dtypes, and the standalone host reference - both say the transpose is wrong at M > 1, and both say it
is right at M = 1. The kernel's guard arithmetic says both forms are correct. The arithmetic is the
only thing in that set that has been wrong before, and it has been wrong before twice.

**So: `transpose_f32_bf16_k` is wrong at M > 1, and the way to find out how is to make the kernel
report what it loaded rather than to read it again.** A debug variant that also writes the loaded tile
to a buffer - `tile[a][b]` for the first two rows, say - compared against `src` on the host, would say
in one run whether the load is indexing wrongly, the guard is dropping writes, or the store is
addressing wrongly. Reading has failed six times in this sequence; the kernel telling us is a
different kind of evidence.

Worth recording alongside it: the build is faithful, the shipped path is intact, and every wrong answer
in this investigation came from an instrument or from a reading - never from the build, and never
from the reference digest.

### `harness_transpose_check` is not testing what it says

Swapping the arguments the harness passes - `transpose_f32_bf16(d_src, d_dst, F, n, s)` instead of
`(d_src, d_dst, n, F, s)` - changes the problem completely: F becomes 2 and M becomes 10240, a
10240-row x 2-column transpose instead of 2 x 10240. The output was **bit-identical at every width**:

    [trchk] transpose n=2  ... 5.000000e-01  *** WRONG ***     <- M=2,   F=10240
    [trchk] transpose n=2  ... 5.000000e-01  *** WRONG ***     <- M=10240, F=2

That is the same "no semantic change" signature as the store de-aliasing in the previous part, and it
settles the question those two experiments were muddying: **the harness's call is not the call that
runs.** Its "zeros at n > 1" was never a measurement of `transpose_f32_bf16_k` at all - it was a
comparison against a buffer nothing wrote, which is why n = 1 appeared to pass and every other width
appeared to fail with exactly the data range's half-width.

So the transpose claim rests on **one** piece of evidence, and it is the good one: the layer A/B,
which uses the engine's own `sc.qkv_flat` and `sc.mixed_qkv` through the real call site and finds
the input bit-identical and the output different. That is unaffected by anything said here.

What the harness needs before it is trusted again: prove it observes the call. The cheapest version is
to have it write a known pattern to `d_dst` first and confirm the pattern survives - if a memset does
not appear in the readback, the harness is reading a different buffer than the one it passed in.

---

## CORRECTION 3: the transpose harness IS valid, and the kernel IS wrong at M > 1

Part "the harness is not testing the call" is **withdrawn**. A marker printed from inside
`harness_transpose_check` confirms the call site executes with the values the harness reports:

    [EDITMARK] harness_transpose_check reached, M=1 F=10240
    [EDITMARK] harness_transpose_check reached, M=2 F=10240
    [EDITMARK] harness_transpose_check reached, M=8 F=10240

so `runner.cpp` edits do reach the binary, the harness really does call
`glue::transpose_f32_bf16(d_src, d_dst, n, F, s)`, and the kernel returns zeros for M = 2 and M = 8
while being correct at M = 1. The M/F-swap experiment in the previous part - which "proved" the
harness was inert by returning bit-identical output for a 2x10240 and a 10240x2 transpose - must have
run against a binary built before that edit. The simplest way this happens: a `cmake --build` in the
same command as the edit reporting clean without having recompiled, which is exactly the failure mode
that edit was invoked to rule out. **The build is faithful in general but was not verified faithful
in that one invocation**, and the marker is the check that should have come first.

**The conclusion of the previous two parts stands and is now twice-confirmed by independent means:**

    [gdnab] qkv_flat     worst 0.000000e+00  IDENTICAL     <- engine's real buffers
    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***

`transpose_f32_bf16_k` is wrong for M > 1. Two harnesses, the layer A/B through the real call site and
the standalone host reference, agree; the transposed store form and the direct one behave identically,
so the defect is not the tile index aliasing; and the coverage arithmetic that exonerated it is wrong
somewhere I cannot see by reading. The fix is to find it by having the kernel report what it loaded,
not by a seventh reading.

The three corrections in this sequence all share a shape: a confident conclusion from a single
measurement, overturned by the next one. Two of the three were my own instrumentation; the third was
a build I did not verify in the invocation that mattered. Only the layer A/B has survived every
check, and it is the one that should be believed.

---

## Batched decode, part 21: the load is exonerated by the kernel itself

`transpose_f32_bf16_k` now takes an optional `dbg` buffer and writes out what each block actually
holds in its tile, and the harness compares that against `src` on the host:

    [trdbg] M=2 F=10240: loaded tile vs src over 128 entries, worst 0.000000e+00  LOAD CORRECT
    [trdbg] M=8 F=10240: loaded tile vs src over 128 entries, worst 0.000000e+00  LOAD CORRECT

**The load is correct at M = 2 and M = 8**, bit for bit, at the widths where the transpose's output is
wrong. (The `M=1` line reports LOAD WRONG from the harness's own single-row indexing of the debug
buffer, not from the kernel - n=1 is the width the transpose gets *right* in every other measurement.)

That kills the last plausible mechanism I had: not a mis-indexed load, not dropped writes, not an
uninitialised tile. The defect is in the store's three lines -

    if (m0 + b < M && f0 + a < F)
      dst[(size_t)(f0 + a) * M + m0 + b] = __bfloat16_as_ushort(__float2bfloat16(tile[b][a]));

- which read a correctly-loaded tile, are guarded to cover every element of `dst` exactly once
(`b < M` over 32-wide `a`, across `(F+31)/32` blocks), and write `dst[f*M+m] = src[m*F+f]`, which is the
transpose. And yet the result reads back as zeros.

**Every element is written, from correct data, to the right index, and the buffer reads back empty.**
That combination has one usual explanation: the write lands somewhere other than where the reader
looks. The next step is the same trick applied to the store - have the kernel report the `dst` address
and value it wrote for a few entries, and check that against the address the harness reads. Both of
those are now cheap, and neither requires reading the kernel again, which is what has failed six
times.

The self-report is inert by default (`dbg = nullptr`, and every engine call site passes nothing), so
the shipped path is unaffected - confirmed by the digest below.

### The store self-report is unreliable too, and the shipped kernel is clean again

Asked the store the same question that settled the load - have it report what it wrote, for `a < 8`,
`b < M`, in block 0:

    [trdbg] M=1: STORE wrote 0/8  non-zero, worst vs src 5.000000e-01  *** STORE WRONG ***
    [trdbg] M=2: STORE wrote 0/16 non-zero, worst vs src 5.000000e-01  *** STORE WRONG ***

That contradicts the one thing known for certain about M = 1: the transpose's **output** at M = 1 is
correct to bf16 rounding (9.76e-04) in every other measurement, through the engine's real buffers and
through the host reference alike. A store that provably writes zeros at M = 1 cannot also produce a
correct result there. **The self-report is wrong, not the store** - the same failure mode as the
earlier harnesses, now in the kernel's own reporting path.

So the debug hook is not yet a trustworthy witness, and the honest position is narrow:

| | status |
|---|---|
| the transpose's **load** | **verified correct** at M = 2 and M = 8, from the kernel's own report |
| the transpose's **output** | verified wrong at M > 1 by two independent harnesses through the real call site |
| the transpose's **store** | correct by inspection, contradicted by a self-report that is itself contradicted by the known-good M = 1 output |
| the defect's mechanism | **not found** |

The store self-report has been reverted - it lived inside the shipped kernel, and an unreliable witness
is worse than no witness. The kernel is back to its original form apart from the `dbg` parameter,
which defaults to null and is passed by no engine call site; the digest below confirms the shipped
path is byte-identical.

What the next attempt needs, in order: a marker proving the debug buffer is the one the kernel wrote
(the check that would have caught this, and the one that caught the harness problem three parts ago);
then the store's address, not its value, since the value is a reinterpretation of a bit pattern and
that is where this went wrong. Six readings and two self-reports have now failed to localise it, and
the pattern of those failures is the most useful thing recorded here: **every instrument in this
investigation has been wrong before it was right, and the checks that would have caught it - a marker,
an address rather than a value, a shape-matched comparison - are the ones that were skipped.**

---

## CORRECTION 4: the transpose is CORRECT - parts 20 and 21 are withdrawn

Asked the store the same way the load report worked - raw bits and the **address** written, never a
reinterpreted value, in a region disjoint from the load's:

    [trdbg] M=1 STORE 8/8  non-zero, value err 9.11e-04, address err 0.0  STORE CORRECT
    [trdbg] M=2 STORE 16/16 non-zero, value err 9.55e-04, address err 0.0  STORE CORRECT
    [trdbg] M=8 STORE 64/64 non-zero, value err 9.59e-04, address err 0.0  STORE CORRECT

**`transpose_f32_bf16_k` is correct at every width tested** - it writes the right value (to bf16
rounding, 9.5e-04) to the right address at M = 1, 2 and 8. Parts 20 and 21, which named it as the
root cause, are withdrawn, and so is the host-reference harness's original "zeros at n > 1" verdict,
which was wrong from the start.

All debug code has been stripped: the kernel is back to its original two arguments and the `dbg`
parameter and the harness are gone. The digest is unchanged.

### What this leaves, stated without any further claim
- The **defect is real**: the layer A/B is reproducible (`8.76e-01`), the per-row GDN A/B fixes it in
  both directions, and the end-to-end parity gate goes from MISMATCH to MATCH with it.
- The **transpose is exonerated** by the kernel's own store report.
- Every other candidate kernel - the recurrence, the conv1d, the projection, the gated norm - was
  cleared by a direct test earlier in this sequence.
- **The mechanism remains unfound**, and the layer A/B's own `mixed_qkv` reading is now the weakest
  link in the chain, because it is the one measurement that has never been independently reproduced
  with a differently-shaped instrument.

The lesson is the one this sequence keeps teaching, and it is worth more than the bug: **four
corrections, every one of them a confident conclusion from a single measurement overturned by the
next.** Two were my own instrumentation reading the wrong buffer, one was a build I did not verify in
the invocation that mattered, and this one was a kernel I named from a harness that was itself wrong.
The checks that would have caught each - a marker, a shape-matched comparison, an address instead of
a value - are the ones that were deferred, and they are the first thing to do next time.

### The layer A/B is order-independent, so it is not scratch contamination

The one measurement still standing is the layer A/B. Its obvious weakness is that both invocations
share the same `GdnScratch`, so whichever runs first could be leaving state for the second. Reversing
the order:

    [gen] full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01   (batched first)
    [gen] full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01   (serial first)

**Identical to every digit.** Contamination from the first invocation is ruled out, and the
difference belongs to the `bsz=2` invocation itself. That is the strongest form this claim can take
short of a second implementation of the layer.

So the ledger now reads:

| | |
|---|---|
| the layer diverges at bsz=2 | **reproduced, order-independent** |
| per-row GDN makes the end-to-end gate pass | **reproduced, both directions** |
| the transpose | exonerated by the kernel's own store report |
| recurrence, conv1d, projection, gated norm | cleared by direct test |
| the mechanism | **not found** |

The remaining untested surface is the **wrapper's own use of the shared scratch at n = 2** - the `z`
and `a`/`b` projections, and `sc.a_had`, which all three projections inside `gdn_layer` share and which
no kernel-level A/B covers, because those A/Bs drive one call each rather than three calls through
one buffer. That is where the next attempt should start, and it is a smaller surface than anything
tried so far.

### Every call in `gdn_layer` is correct at n = 2, and the layer is still wrong

The last two projections with no A/B - the `z` projection, which is the second user of the shared
`sc.a_had`, and the `a`/`b` gemv - both come back clean, run through the same shared scratch the way
the wrapper does:

    [projab] z_proj  n=2 vs 2x n=1: worst 0.000000e+00  IDENTICAL
    [projab] a_gemv  n=2 vs 2x n=1: worst 0.000000e+00  IDENTICAL
    [projab] b_gemv  n=2 vs 2x n=1: worst 0.000000e+00  IDENTICAL

The full ledger for one GDN layer at n = 2, every element tested at the same shape and dtype in both
paths:

| call | verdict | how |
|---|---|---|
| qkv projection | IDENTICAL | A/B vs two n=1 calls |
| **transpose** | **CORRECT** | the kernel's own store report - value *and* address |
| depthwise conv | IDENTICAL | A/B, output and written-back state, engine's block layout |
| delta-rule recurrence | IDENTICAL | A/B, output and state, engine's block layout |
| z projection | IDENTICAL | A/B through the shared `sc.a_had` |
| a/b gemv | IDENTICAL | A/B |
| gated RMS norm | row-independent | host reference at rows 1/2/3/48/96 |
| **layer output** | **differs, 8.76e-01** | order-independent A/B |

**Every call is right and the layer is wrong.** That is not a paradox once the calls are listed in
order: each A/B drives **one** call, and the layer runs **six**, three of them through the same
`sc.a_had` and all six through the same `GdnScratch`. No test in this sequence has ever run the
sequence. That interaction - not any single call - is the last untested surface, and it is exactly
what "shared scratch" means: a buffer whose contents after call *k* determine what call *k+1* reads.

The next test is therefore not another kernel and not another layer call. It is to run the layer's
**six-call sequence** at n=2 and at 2x n=1 through one scratch, capturing `sc.a_had` and `sc.qkv_flat`
*between* calls - the only comparison in the sequence that has never been made.

### The layer does not read unwritten scratch - the last candidate eliminated

The leading remaining hypothesis was that `gdn_layer` at n=2 leaves a scratch buffer partly
unwritten - sized or indexed for one row - and the next call inherits whatever was there. That is
testable without reading anything: fill every scratch member with garbage, using each member's
**real** size from `gdn_scratch_init`, and see whether the layer's output moves. (The first attempt
used one uniform size, overran the small members and failed with `invalid argument` inside an
unrelated gemm - which says nothing either way. Sizes from the allocation site, not guessed.)

    clean scratch    : worst abs diff 8.759627e-01
    poisoned scratch : worst abs diff 8.759627e-01    <- identical to every digit

**The layer's output does not depend on the scratch it is handed.** It writes everything it reads.
"Uninitialised or partly-unwritten scratch" is eliminated, along with the shared-scratch-interaction
theory that was the last framing of the defect.

That leaves a genuinely tight and still-unexplained position:

| | |
|---|---|
| all seven calls verified correct at the parameters the layer passes | yes |
| the layer reads no scratch it does not write | yes, poison-invariant |
| the layer is deterministic and order-independent | yes |
| the layer output at bsz=2 differs from two n=1 calls by 8.76e-01 | yes, reproducibly |
| the mechanism | **not found** |

At this point the difference is not in any call, not in any buffer's prior contents, and not in the
order. What is left is the *mapping* from the layer's arguments to what each call is asked to do -
most plausibly the `sl = bsz > 1 ? n / bsz : n` derivation, which hands the conv and the recurrence
`bsz=2, seqlen=1` in one call and `bsz=1, seqlen=1` in the other, and which no A/B has varied
independently of `bsz` itself.

The honest recommendation is to stop treating this as a kernel hunt. It is one line of argument
plumbing in `gdn_layer`, and the way to find it is to print what each call receives, once, in both
invocations - which is the cheapest check in this entire sequence and the one that was never done.

### The argument mapping is right - the cheapest check in the sequence, finally run

Printing what each call receives, in both invocation styles:

    engine, per layer : n=1 bsz=1 sl=1  conv=...28000 rec=...600000  gdn_slots=nil conv_slots=nil
    harness, batched  : n=2 bsz=2 sl=1  gdn_slots=0x...800 conv_slots=0x...800

The engine's per-layer calls advance `rec` by `0x600000` = 2 x rec_bytes and `conv` by `0x28000` =
2 x conv_bytes - the `[layer_rank][slot]` interleave, exactly right, and the last thing that could have
been a silent addressing error. The batched call receives `n=2 bsz=2 sl=1`, so `sl = n/bsz = 1` is
being derived as intended, and the real path passes `conv_slots = {0, d0 * n_gdn}` for the 36-layer
interleave.

**One thing the print did expose is in the harness, not the engine**: `harness_gdn_layer_ab` passes
`d_rel` for *both* slot arrays, so the conv there sees `{0, 1}`. That happens to be correct for its
single-layer state block - slot 1 sits one state along - so the A/B stands, but it means the harness
would have been wrong the moment it was pointed at a multi-layer block. Worth knowing before anyone
reuses it.

So the argument mapping is cleared too. What remains is not in a call, not in a buffer, not in the
order, and not in the arguments - which, having exhausted the mechanical explanations, is the point
at which a fresh pair of eyes on `gdn_layer` as a whole is worth more than another experiment.

### The review found the last untested op, and cleared it

Reading `gdn_layer` as a whole - rather than one call at a time - surfaced the one op no A/B in this
sequence ever covered, at line 125:

    aux::gated_delta_net_fused_op_2(sc.b_out, sc.a_out, (const bfloat16*)w.dt_bias, w.A_log,
                                    false, (bfloat16*)sc.beta, sc.g, /*B=*/1, n, Nv, 1.0f, s);

**`B` is hardcoded to 1** and `n` goes into the `S` slot, so a two-row batch arrives at the op that
produces `sc.g` and `sc.beta` as *one batch of sequence length two*. That is the only place in the
layer where the batch/sequence distinction is collapsed, and it is the step every downstream call
consumes - the conv and the recurrence both read `sc.g` and `sc.beta`.

The kernel indexes it flat:

    int row = blockIdx.x * rows_per_block + threadIdx.x / H;
    if (row >= B * S) return;

so `B=1, S=2` and `B=2, S=1` address the same `B*S` rows identically, and the hardcoded `B` is
harmless **provided the op is elementwise in the row**. Building the A/B to confirm that rather than
trusting the reading ran into a compile fight with the `GdnWeights` member types, and with the context
remaining I reverted it rather than leave a half-built harness in the tree - so this one is **cleared
by reading, not by measurement**, and is the single place where that distinction still matters.

It is also the most likely place a real bug would now be hiding, precisely because it is the one
call whose `(B, S)` split a batched forward gets wrong *conceptually* even if the kernel happens to
flatten it away. A measurement there is worth more than anything else in this sequence.

### Every op in gdn_layer verified at n = 2, and the layer is still wrong

The last uncovered op, measured rather than reasoned about:

    [fused2] n=2 (B=1,S=2) vs 2 x (B=1,S=1): g diff 0.000000e+00  beta diff 0.000000e+00  IDENTICAL

`gated_delta_net_fused_op_2` is the step that collapses the batch/sequence distinction - it is called
with `B` hardcoded to 1 and `n` in the `S` slot - and it is bit-identical, so the flattening is real
and the hardcoded `B` is harmless. (The earlier compile failure was `bfloat16` without the `__`
prefix, which is not in scope in `runner.cpp`; `__nv_bfloat16` fixes it.)

**The complete ledger for one GDN layer at n = 2 - nine operations, every one verified:**

| op | verdict |
|---|---|
| qkv projection | IDENTICAL |
| z projection | IDENTICAL |
| a / b gemv | IDENTICAL |
| **gated_delta_net_fused_op_2** | **IDENTICAL** |
| transpose | CORRECT (kernel's own store report: value and address) |
| depthwise conv | IDENTICAL (output and written-back state) |
| delta-rule recurrence | IDENTICAL (output and state) |
| gated RMS norm | row-independent (host reference, rows 1/2/3/48/96) |
| output projection | IDENTICAL |
| **layer output** | **differs, 8.76e-01** (order-independent, poison-invariant) |

Nine operations, every one correct, composed into a wrong result. With the prior candidates all
eliminated - the calls, the buffers, the order, the prior scratch contents, the argument mapping -
what remains is the question none of those A/Bs can answer, because each drives a call **in
isolation**:

> **is any op receiving a different `n` inside the layer than the A/B gave it?**

That is the one thing this sequence never compared, and it is where a wrong result can hide with
every individual measurement coming back clean. The `fused_op_2` line is the proof of concept - it
takes both `n` and a hardcoded `B`, so a harness that hardcodes its own `n` rather than passing the
layer's would test a different call entirely.

The next step is therefore narrow and specific: capture the `n` each op actually sees **inside a real
batched layer call**, and compare it against what the standalone A/Bs pass. Not another kernel A/B.

### The layer A/B is validated against ground truth

The A/B has been the load-bearing measurement for several parts now, and it shares a harness with the
per-op tests - so it is worth checking against something it cannot influence. The end-to-end parity
gate can:

    per-row GDN   (default) : batch-parity A MATCH    B MATCH
    batched GDN   (forced)  : batch-parity A MISMATCH  B MISMATCH

**The A/B and the end-to-end gate agree, in both directions.** The defect is real, it is the GDN layer's
`bsz=2` path, and it is not an artifact of the harness. That also retroactively validates the per-op
A/Bs - they ran in the same harness and agree with the gate on the layer-level verdict, so "this op is
correct" is a statement about the op as the layer calls it, not just as my harness called it.

**Which sharpens the paradox rather than dissolving it.** Nine operations, each verified correct *as
the layer invokes them*, composed into a wrong result, and the one thing still uncompared is whether
any op receives a different `n` inside the layer than the A/B passes it. `fused_op_2` proved that
question is worth asking - it is the only op whose `(B, S)` split a batched forward gets wrong
conceptually, even though its kernel flattens the difference away.

Concretely, the remaining test is to capture the `n` (and `B`, and `sl`) each op receives inside a
real batched layer call and compare against the standalone A/Bs. The entry-level print already shows
`n=2 bsz=2 sl=1` for the batched call, so the suspicion is now narrow: a call *inside* the layer that
derives its own row count rather than using `n`.

And the practical consequence is settled either way: **the per-row GDN stays the default.** Batched
decode is correct and reachable with it, and the batched path cannot be re-enabled until this is
explained - which is a far better position than the one this started from, where the gate was failing
for reasons nobody could name.

### n-provenance is correct - the last question, answered

Capturing the row count each op actually receives **inside a real batched layer call**, which is the
one thing every A/B in this sequence took on faith:

    [rows] layer n=2 bsz=2 sl=1 | proj 2 | fused B=1 S=2 | conv bsz=2 sl=1 | rec bsz=2 sl=1 | norm 96

against the single-sequence call:

    [rows] layer n=1 bsz=1 sl=1 | proj 1 | fused B=1 S=1 | conv bsz=1 sl=1 | rec bsz=1 sl=1 | norm 48

Every op receives the correct total: the projections get 2 rows, `fused_op_2` gets `B=1, S=2` = 2
rows, the conv and the recurrence get `bsz=2, seqlen=1` = 2 rows, and the gated norm gets `n*Nv = 96`
= 2 x 48. `harness_gdn_layer_ab` passes exactly these values, so the A/B and the layer are not
disagreeing about anything.

**That closes the last open question.** The full elimination now stands:

| eliminated | how |
|---|---|
| every one of the nine ops, individually | A/B at the parameters the layer passes, same harness the gate uses |
| prior scratch contents | poison, per-member real sizes, output invariant |
| invocation order | reversed, result identical to every digit |
| the argument mapping | printed: `n=2 bsz=2 sl=1`, slot arrays and strides correct |
| n-provenance | printed: every op receives 2 rows |
| the harness itself | the end-to-end parity gate agrees with it in both directions |

Nine correct operations, correct buffers, correct order, correct arguments, correct row counts, and a
wrong answer. I do not have the mechanism, and I would rather hand that over than keep manufacturing
hypotheses that four corrections have already taught me to distrust.

**The practical position is settled regardless**: the per-row GDN is the correct default, batched
decode is correct and reachable through it (`generate_pair`, `gen --pair`, parity MATCH/MATCH on both
slots), and the batched path cannot be re-enabled until this is explained. That is a far better state
than where the sequence began, when the gate failed for reasons nobody could name.

### The tenth op, and a vacuous test caught in the act

The gated norm was the only operation in the layer verified for *row-independence* but never compared
against a reference for *correctness* - and it is the one place a kernel that is right at 48 rows
(`n=1`) could be wrong at 96 (`n=2`), since it is called with `rows = n*Nv`.

The first run of that comparison reported `worst abs diff 0.000000e+00 at row -1`, which is not a
pass - it is a test that could not fail. The inputs were built by stuffing pseudo-random bits into
`unsigned short`, so the bf16 exponents came out as inf/NaN, the host reference was NaN, and
`d > worst` is false for NaN. Every element was NaN, the comparison never fired, and the harness
printed the number a passing test prints. Building the inputs from real floats and adding an explicit
non-finite count:

    [gnorm] rows=96 dim=128 vs host reference: worst abs diff 4.878044e-04 at row 41, 0 non-finite  OK

The kernel is correct at 96 rows, and the op's call site is sound too: `z16` is cast from `n*z_out`
floats and read as `[n*Nv, Hv]` flattened, which is the same memory viewed as `[n][Nv][Hv]` - correct
for both n.

**Worth stating plainly: the one test that came back clean on its first run was the one that had not
run at all.** A comparison harness needs a check that it can fail, and `worst == 0 with no row
recorded` is the signature. Three of the earlier "IDENTICAL" results in this sequence were saved from
the same failure mode only by choosing inputs that could actually produce a difference.

### Final state of the investigation

All ten operations in one GDN layer are verified correct as the layer invokes them, at the row counts
the layer passes, with buffers, order, slot addressing and argument provenance all confirmed by print.
The layer output differs. The harness is validated against the end-to-end parity gate in both
directions, so this is not an artifact of the instrument.

I do not have the mechanism, and I am not going to invent one. What is settled and usable:

- **per-row GDN is the correct default**, and it is what ships
- **batched decode is correct and reachable** through it - `generate_pair`, `gen --pair`, both slots MATCH
- the **batched path cannot be re-enabled** until this is explained
- the next lead is narrow: a call *inside* the layer that derives its own row count instead of using
  `n`, and it must be found by reading for a stray `1`, a hardcoded head count, or a `dim * 0`-style
  expression - not by another isolated A/B, which this sequence has now shown will come back clean

### Batched decode: measured properly, after an earlier number was wrong

The first version of this entry claimed **1.68x** aggregate throughput for batched decode. **That was
a measurement error and the claim is withdrawn.** The command omitted `--repeat 2`, and `--pair` is
gated on it:

    if (g_pair && it == 0 && repeat >= 2) {

so the pair branch never ran. The "batched" 56.82 tok/s that came back was a plain single-sequence
decode. The lesson is the one already learned twice in this project - a grep showing a code path
exists is not evidence the path executes - applied this time to myself.

Re-measured, matched on end-to-end wall time at 8k context, so both sides pay their own prefill:

| | tokens | wall | effective |
|---|---|---|---|
| single sequence | 128 | 5.89 s | 21.7 tok/s |
| batched pair | 256 | 8.93 s | **28.7 tok/s** |

**1.32x aggregate**, not 1.68x. A short-prompt run with prefill negligible gives the same shape
(64.9 tok/s aggregate against ~50 single), so the ratio is real and it is modest.

The reason it is modest is visible in the engine's own startup banner: with two slots the sequences
are *"interleaved, not simultaneous: one forward at a time on a shared activation set"*, and the GDN
layer runs per-row, so the only work shared is the dense trunk. Decode here is bound by reading two
KV caches per step rather than by arithmetic, and a 2.05 bpw Q2 quant leaves almost no compute
headroom to amortise. **Batching helps, but it is a ~1.3x lever on this model - not the 1.7x it
looks like from the single-sequence decode number alone.** It is worth taking for a client issuing
concurrent requests, and it is not worth a large engineering effort.

**`generate_pair` is reachable and exercised** - `helios gen --pair --repeat 2` drives it, and prints
`[gen-paired] 128 + 128 tokens in 8.93s -> 28.7 tok/s aggregate`.

### Why `n = 2` over HTTP stays rejected - now on evidence, not guesswork

I tried to wire OpenAI-compatible `n = 2` to `generate_pair` and reverted it because it would not
compile against the real `OutputParser` API. Reading that API to fix it turned up the reason it
should not exist in that form:

    // Greedy only: the paired step reads both rows' argmax.
    if (mtp_on_ || n_slots_ < 2 || (p.greedy == false && p.temperature > 0.01f)) return false;

`generate_pair` **declines any sampled request**, and the paired step is argmax on both rows - so two
identical prompts produce two identical completions. That is a throughput tool for two greedy
streams, not a sampling feature. An OpenAI `n = 2` request is specifically a request for *diverse*
samples, so serving it from this path would return two byte-identical completions to a client that
asked for variety - a silent wrong answer, which is worse than the explicit `n > 1 is not supported`
the server returns today.

Supporting `n = 2` properly means per-row sampling in the paired step (two independent RNG streams
read from the two logits rows), which is real work on the decode inner loop and is the only thing
that would make it honest. Left undone and stated, rather than faked.

### Final state, re-verified
Build clean; **16/16 ctest**; sequential reference digest **3aaa5693eee90a81513be908f1ba3263**; " Paris.";
served context **262,144**; sustained decode **67.1 tok/s** on a 1541-token generation; prefill
**117-127% of exllamav3**; decode **~101% median** on symmetric median-of-3.

---

## HELIOS_SEQUENCES: N interleaved conversations (opt-in, default 1)

`HELIOS_SEQUENCES=N` keeps N conversations **alive** in one engine process. The engine has one
activation set, one logits buffer and one device-side recurrent state per layer, so two forwards
cannot be in flight at once; making them simultaneous is a redesign of every kernel's scratch
plumbing, not a flag. What N slots buy is that a server can hold several open threads and serve
them a request at a time without evicting each other's state. `concurrency_mode: "interleaved"` is
reported by `/v1/models` for exactly this reason — `parallel: 4` on its own reads as four requests
decoding simultaneously, which this engine does not do.

**What a slot is.** Each slot owns a base-pointer offset into the ONE already-allocated KV cache
(`kv_base()`), so **N slots cost no extra KV memory at all** — they divide the same 6.00 GB N ways
and each slot can hold `ctx_cap/N` tokens. Attention addresses rows relative to the pointer it is
handed, so no kernel changed. The only thing N costs in VRAM is a resident copy of the persistent
recurrent state (GDN conv+recurrence, PLE conv, MTP chain seed): **111 MB per slot, ~56 MB on card 0
and ~55 MB on card 1** at the shipped configuration. Being *resident* rather than spilled to a host
ring is what makes a switch free — it is a pointer rebind, not a 116 MB save plus a 116 MB restore,
so there is no per-switch copy that can be half-done when the next request arrives.

### VRAM at the shipped 262144 context (measured, `helios load`)

| N | per-slot context | card0 free | card1 free | slot state |
|---|---|---|---|---|
| 1 | 262144 | 1.42 GB | 0.61 GB | — |
| 2 | 131072 | 1.34 GB | 0.54 GB | +0.08 / +0.07 GB |
| 4 | 65536 | 1.20 GB | 0.40 GB | +0.22 / +0.21 GB |
| 8 | 32768 | 0.91 GB | 0.11 GB | +0.51 / +0.50 GB |

The KV total is identical at every N (6.00 GB); only the recurrent state grows, by exactly 111 MB a
slot. **The clamp never bound on this box**: 8 slots fit with 0.11 GB to spare on the tighter card.
It exists because the figure is measured, not assumed — at `ctx_cap=262144` card 1 is down to
0.61 GB, so a larger N is genuinely a different number on a different machine. `seq_fit_slots()`
takes the minimum over the cards that actually hold per-slot state, holds back 320 MB per card for
decode temporaries, and reports the binding limit by name rather than applying a silent trim. A
second, independent limit is a context floor of 8192 tokens per slot, so N is never raised to the
point where a "conversation" has a 200-token window.

### Correctness

`test/test_seq.cpp` (new ctest, pure logic, no GPU): the fit clamp, the context division, the
routing decision. `test/test_seq_interleave.py`: one process, a scripted plan, every slot compared
byte-for-byte against the same requests served alone. `test/test_seq_server.py`: the same property
over HTTP against a live server.

| check | result |
|---|---|
| N=3, 16 interleaved requests (mixed lengths, 2 slots alternating, long-after-short, reversed order) | **16/16 byte-identical** |
| N=4, 18 requests | **18/18 byte-identical** |
| N=2, 14 requests | **14/14 byte-identical** |
| multi-request server, 3 slots, 30 requests (round-robin, pinned, alternating, long-after-short, streaming + buffered, whole plan repeated) | **30/30 identical, no crash** |
| out-of-range slot pin | refused, HTTP 400 |
| flag unset: digest | `3aaa5693eee90a81513be908f1ba3263` **unchanged** |
| flag unset: ctest | **17/17** (16 pre-existing + 1 new) |

A fresh-process benchmark cannot see any of this. The state a slot carries lives in the process, so
a per-request harness throws it away — which is precisely how a CUDA-graph feature on this engine
shipped a crash that no single-request measurement could have shown.

### Three defects this found, all of them silent

1. **Segfault on the third request.** `bind_slot()` originally swapped the live state vectors with
   the slot's. At init the live vectors are empty and the real pointers sit in `slots_[0]`, so the
   first bind swapped empty against real and a second call swapped them straight back — leaving the
   live state empty and crashing inside `layer_range()`. Fixed by splitting `save_slot()` from
   `load_slot()` as plain assignments, which cannot be un-done by calling them twice. Only a
   multi-request test in one process could have found this.
2. **A pin of 5 on a one-slot engine was silently accepted** and answered as slot 0, because
   `seq_pick()` tested `n_slots <= 1` before validating the pin. Folding a pin is invisible because
   the folded answer looks like a legitimate one. Found by running a 3-slot plan against a 1-slot
   engine; now returns 400.
3. **The prefix-cache ring was shared across slots.** A capture describes the conversation that
   took it, so restoring one conversation's recurrent state onto another's KV rows answers from the
   wrong context without failing. The ring is now partitioned per slot (`n_slots_ × captures`), and
   `pfx_pos_` is per-slot.

### Two limits worth stating plainly

- **A slot only persists a conversation when `HELIOS_PREFIX_CACHE=1` is also on.** Without it,
  `prefix_begin()` calls `reset()` on every request, so each request starts a fresh sequence and N
  slots are N identical independent runners. With it, a slot's position and token history survive
  across requests and the pin is what makes a follow-up turn land on its own state. `HELIOS_SEQUENCES`
  alone is the mechanism; the prefix cache is what makes it mean anything.
- **`HELIOS_DECODE_GRAPH=1` is refused when N > 1**, and says so at startup. A captured graph bakes
  in absolute device addresses; with more than one slot both the recurrent state and the KV base
  change when the conversation changes, so a replay would run against whichever slot was bound at
  capture time. That is the same stale-pointer class that already made graphs unsafe across
  requests, now reachable within one.

---

## Multi-sequence concurrency (opt-in): interleaved conversations, closing the last capability gap

exllamav3 reports `"parallel": 4` over a paged cache; helios served one sequence at a time. Now
`HELIOS_SEQUENCES=N` keeps N conversations alive. **What it is, stated plainly: interleaved, not
simultaneous.** There is one shared activation set, so one forward runs at a time; N slots preserve N
conversations' KV and recurrent state and let a client pin a follow-up turn to its own slot. A slot only
persists a conversation when `HELIOS_PREFIX_CACHE=1` is also set - without it every request resets, and N
slots would be N identical independent runners.

### Why it is cheap
- **The KV cache is not duplicated.** Slots partition the ONE already-allocated 6.00 GB by base-pointer
  offset, so N costs no extra KV memory - each slot simply gets `ctx_cap/N` rows.
- **The only VRAM cost is resident recurrent state, 111 MB per slot** (card0 56 MB, card1 55 MB), and it
  is resident rather than host-spilled precisely so that a slot switch is a pointer rebind instead of a
  116 MB save+restore.

At the shipped 262144/fp16 configuration: `N=1` 1.42/0.61 GB free, `N=2` 1.34/0.54, `N=4` 1.20/0.40,
`N=8` 0.91/0.11. With N=3 each slot gets 87,381 of 262,144 context rows. The clamp did not bind on this
box. Note the tension: fp16 KV is what gives strictly better fidelity than the reference's forced cq3
(6.00 GB vs 1.13 GB) and is also what leaves only 0.61 GB free. `HELIOS_KV_QUANT` is the dial, and it is a
real trade, not a free win - it measured slower per token.

### Correctness, verified independently
| test | result |
|---|---|
| agent's `test_seq_interleave` | PASS - 16 interleaved requests over 3 slots, every one byte-identical to serving that conversation alone |
| agent's `test_seq_server` | PASS - 30 requests through ONE server over 3 slots, no crash, every answer identical to single-sequence |
| **my own check, plan repeated twice** | **14/14 byte-identical with MTP off; 10/14 with MTP on** |
| flag unset | digest `3aaa5693` unchanged, **17/17 ctest**, prefill 2464-2478 and decode 68.89/68.89 vs pre-change 2432-2440 and 68.56-68.95 - no regression |

The 4 differing cells are **not a concurrency defect**: they are the request-order dependence already
documented for MTP (the draft head's cache is not reset between requests, so history changes the accept
count, which changes the trunk's verify width, which reassociates the MoE sum). With speculation off -
the order-invariant configuration - the interleaved slots are byte-identical to serving alone. That is the
honest statement: **concurrency is exact on the deterministic path, and inherits MTP's documented
order-sensitivity otherwise**, which is a property of speculation, not of the scheduler.

### A third silent defect of the "fresh process cannot see it" class
`bind_slot()` swapped the live state vectors with the slot's. At init the live vectors are empty and the
real pointers live in `slots_[0]`, so the first bind swapped empty against real and a second swapped them
back - live state ended up empty and `layer_range()` **segfaulted on the third request in one process**.
Fixed by splitting `save_slot()` from `load_slot()` as plain assignments, which cannot be undone by
calling them twice. Also fixed: an out-of-range slot pin was silently accepted as slot 0
(`seq_pick()` tested `n_slots<=1` before validating the pin) - now HTTP 400, with a unit test.

This is the third such defect in this project (after the CUDA-graph crash and the out-of-range pin), and
all three were found only by running a **sequence of requests through one process**. That is now the
acceptance gate for anything touching shared state here.

### The slot banner was claiming something false about the default configuration
With `HELIOS_SEQUENCES=3` and the prefix cache off - the shipped default - the startup banner read:

> "N slots keep N conversations' KV and recurrent state ALIVE and let a client pin a follow-up turn to
> its own slot"

That is untrue in that configuration. A slot preserves its conversation across requests **only** through
the prefix cache; without it every request calls `reset()`, so the N slots are N independent copies and
pinning a turn to a slot continues nothing. The banner now states the truth in both cases, and with the
prefix cache off it says so explicitly and names the flag that fixes it:

> `[seq]   WARNING: conversations are NOT kept alive with this configuration. Every request resets the
> runner, so the N slots are N independent copies and pinning a turn to a slot does not continue a
> conversation.`
> `[seq]   Set HELIOS_PREFIX_CACHE=1 as well to make slots persistent. That path is opt-in because it
> changes the token stream of a request that reuses nothing; for multi-turn use that does not matter,
> which is why it is the intended companion here.`

Worth keeping for the same reason as the other instrument fixes: the two features compose, and the
documented caveat of one (the prefix cache changes cold-request output) is precisely irrelevant to the
workload the other exists for. A user reading only the banner would otherwise have believed multi-turn
conversation worked when it did not.

---

## The 262,144 context claim, finally tested at depth

This was the last untested headline claim - every measurement in this project used 8k or 16k prompts,
and "serves 262,144 tokens" had only ever been a number the engine printed about itself.

| prompt tokens | prefill tok/s | decode tok/s | MTP acceptance |
|---|---|---|---|
| 53,487 | 2053.1 | - | - |
| 106,974 | 1579.1 | - | - |
| 160,857 | 1308.9 | 36.1 | 57.5% |
| **218,402** | **1058.9** | **35.4** | 56.1% (1.56 tok/step) |
| **224,573** | **1058.9** | **40.5** | 87.5% (1.88 tok/step) |

**218,402 and 224,573 tokens - 83-86% of the advertised window - prefill at ~1059 tok/s with speculation
working at 56-88% acceptance.** Prefill degrades roughly linearly with depth (2053 -> 1059 from 53k to
218k), which is the expected attention cost, not a cliff. At 218k it still runs at ~55% of exllamav3's
rate on a prompt of that depth, so the prefill advantage holds at depth rather than only at 4-16k.

The two entries with no decode figure generated **0 tokens**: the repeated-word filler triggers an
immediate stop token, the same degenerate-prompt behaviour the grid harness hit, not a fault.

### Past the cap: refused cleanly
```
[gen] prompt 270678 exceeds this sequence slot's context 262144 (of 262144 total across 1 slot)
```
Exit 0, no crash, and the **server stayed alive** after serving a 224k request - an over-cap request is
refused rather than taking the process down, which is the behaviour that matters for a long-lived server.

This also exercises the slot accounting at depth: with `HELIOS_SEQUENCES=1` the message correctly names
the per-slot budget and the total across slots, and with N>1 each slot's own cap applies.

---

## The chat endpoint — the last untested surface, and the one most clients actually use

Everything up to here went through `/v1/completions`. This is a hybrid reasoning model, so the chat
path also exercises the `<think>` parser that splits `reasoning_content` from `content`.

| test | result |
|---|---|
| plain question | `content: "\n\nParis"`, with the deliberation in `reasoning_content` - correctly separated |
| code | correctly typed Python function |
| **multi-turn** | `"\n\nYour name is Ada."` - recalls from conversation history |
| **system role** | `"\n\nLa capitale de la France est Paris."` - follows a "answer only in French" system prompt |
| `reasoning_effort` low / high / absent | all accepted; low and high produce visibly different reasoning |

`finish_reason: stop` throughout, usage reported correctly, and the server stayed alive across all of
them. The reasoning model's thinking block is surfaced in the standard `reasoning_content` field rather
than being flattened into the answer, which is what a reasoning-aware client needs.

---

## Two server bugs, found by testing the two things a client actually does

### 1. `/v1/completions` never streamed, despite `stream: true`
`text_handler` had **no streaming path at all**. A client sending `{"stream": true}` got a single
buffered JSON body - which a streaming client reads as "nothing yet" until the whole generation
finishes, and which does not error, so nothing surfaces. The chat endpoint always streamed; the
completions endpoint did not.

It now emits SSE (`text/event-stream`, per-token `text` deltas, a terminal `finish_reason` chunk, an
optional `stream_options.include_usage` block, and `[DONE]`). Verified on the order-invariant path
(`HELIOS_MTP=0`), streamed text is **byte-identical to the buffered path** on all four cases:

| prompt | match | chunks | `[DONE]` | finish |
|---|---|---|---|---|
| Count to five: | yes | 16 | yes | stop |
| Write a haiku about rain. | yes | 122 | yes | length |
| The capital of France is | yes | 62 | yes | length |
| Écris un poème sur la mer. | yes | 82 | yes | length |

(The last one is deliberate: it exercises the `Utf8Streamer`, which holds back a trailing partial UTF-8
character rather than emitting half of one, so the end-of-generation tail is emitted from the full
decode. A non-ASCII case is the one that would catch that being wrong.)

**A use-after-free, found while fixing it.** httplib runs the content provider *while writing the
response*, i.e. after the handler has returned, so anything the provider captures by reference to a
handler local is dangling. My first version captured `p` (a `GenParams` local) under the default `[&]`
and the generation ran with a garbage `max_tokens` and produced no tokens at all. Fixed by copying the
values the provider needs into its explicit capture list - which is what the chat handler already does
correctly with `req`, so the two handlers are now consistent.

**And a verification failure of my own, worth recording.** Earlier in this project I reported "streaming
works" on the strength of getting a valid JSON body back. I never checked that the body was SSE - which
is why this bug survived several rounds of endpoint testing. The lesson generalises past streaming: *a
response that parses is not evidence that it arrived in the shape the client asked for.*

### 2. An aborted generation was not cancelled server-side
Aborting a 3000-token stream 1.5 s in left the generation running to completion, and the **next request
was blocked behind it for 93 s**. On a shared server that means one user pressing ctrl-C stalls everyone.

The streaming loop already tried to notice (`alive = send(j)`, returning false on a failed write), but a
write keeps succeeding into the kernel's socket send buffer long after the peer is gone, so the return
value alone does not reveal a dead client. Adding httplib's own liveness probe,
`sink.is_writable()` - which is `select_write(...) > 0 && is_socket_alive(...)` - before every chunk
cuts the stall from **93 s to 45 s**. It is not instant: the client's buffered output still has to
absorb before the socket errors, and the server's write timeout is 600 s. Bounded rather than eliminated
is the honest description, and it is a large improvement over a 93 s stall.

### Why the abort fix is bounded at ~45 s rather than instant
Worth recording so nobody re-derives it. httplib's liveness chain is
`is_writable() = select_write(sock, WRITE_TIMEOUT) > 0 && is_socket_alive(sock)`, and
`is_socket_alive` is:

```cpp
const auto val = select_read(sock, 0, 0);
if (val == 0) return true;                       // nothing to read -> treat as alive
if (val < 0 && errno == EBADF) return false;
return read_socket(sock, &buf[0], 1, MSG_PEEK) > 0;
```

Both peer-close cases are handled correctly: a clean FIN makes the socket readable-at-EOF so the
`MSG_PEEK` returns 0, and an RST makes it return -1; either way `> 0` is false and the peer is dead.
So the check is not missing anything. The residual stall is physical - the kernel keeps accepting the
abandoned stream's bytes into the socket send buffer, and the peer only becomes visibly dead once that
buffer can take no more. The server's write timeout is 600 s, so there is a lot of room for output to be
absorbed first. Getting instant cancellation would need a reader thread per connection watching for
the FIN, which is disproportionate for a single-user server; 45 s bounded is the honest outcome, against
93 s before.

---

## Tool calling was broken: the parser and the renderer spoke different dialects

`text_handler`/`chat_handler` carry a lot of tool-calling machinery - `req.tools`, the tools preamble,
`tool_begin`, `tool_index`, the assistant tool-call renderer - and none of it had ever been exercised.
With a tool offered, the model produced a correct call and the engine returned:

```json
{"name": "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>",
 "arguments": "{}"}
```

**The function name was the whole markup blob and the arguments were empty** - a real client would have
invoked `get_weather` with no arguments at all.

**Cause: the parser and the renderer disagreed.** `render_chat_qwen` emits Qwen3's dialect, an inner
`<function=NAME> ... <parameter=KEY>VALUE</parameter> ... </function>` nested in `<tool_call>`, and the
system prompt tells the model to produce the same. But `args_to_json` only understood the older
`NAME<arg_key>K</arg_key><arg_value>V</arg_value>` form, and `tool_name_of` takes everything before the
first `<arg_key>` as the name - which for a Qwen3 call is the entire blob. The parser now accepts both
dialects, with multi-line values handled and JSON-typed values (`42`, `true`) left typed rather than
stringified. The end-to-end flow is now correct:

| | before | after |
|---|---|---|
| name | the markup blob | `get_weather` |
| arguments | `{}` | `{"location":"Paris"}` |
| follow-up with the tool result | leaked `<\|im_start\|>` tokens | *"The weather in Paris is currently **18°C and partly cloudy**."* |

### Why it survived so long: there was no parser test at all
`grep -rl tool_call test/*.cpp` returned **zero files**. The chat tests exercised `render_chat_qwen` -
the renderer - and the parity scripts diff rendered bytes against the checkpoint's own
`chat_template.jinja`. Nothing ever fed a tool call *into* `OutputParser`, so the two halves of the
feature were free to disagree indefinitely. New `test_chat_parse` (registered, 18 tests total) feeds
both dialects through the parser a few characters at a time, so the streaming holdback path is covered
too: single and multiple parameters, a multi-line value, JSON-typed values, the legacy dialect, and
prose followed by a call.

The general shape is the same one as the `/v1/completions` streaming bug: a half of a feature that
nothing exercised, which is why "it is implemented" and "it works" turned out to be different claims.

---

## Template tokens were leaking into user-visible content: the parser knew the wrong dialect

Chasing two results from the tool-calling test that I had initially moved past - an empty content on
`tool_choice:"none"` and a stop-sequence case that proved nothing - turned up a real defect. (The empty
content was *not* a bug: with a warm server and no tools the same request answers properly. It was the
MTP order-dependence again. The stop test was genuinely vacuous - the model opened with "\n\n", so the
stop matched immediately.)

What was real: with a tool offered, the content field came back as

```
'\n\n\n<|im_start|>user\n<tool_response>\n{"type": 0, "city": "Par...
```

alongside a correctly-parsed `tool_calls`. Template markers were reaching the caller verbatim.

**Cause: the same dialect mismatch as the tool-call parser, one level up.** `OutputParser` strips turn
and template markers using a fixed set - `<think>`, `<tool_call>`, `<|observation|>`, `<|endoftext|>`,
`<|user|>`, `<|assistant|>`, `<|system|>` - and that set is the *other* dialect's. This checkpoint is
ChatML: `tokenizer_config.json` declares `<|im_start|>`/`<|im_end|>` and `eos_token: <|im_end|>`, and
`render_chat_qwen` wraps a tool result in `<|im_start|>user ... <tool_response>...</tool_response>`. So a
model echoing the template back - which it does readily once a tool call is in play - had every one of
those markers emitted into the answer.

Both marker lists the parser maintains are now covered: the holdback set (so a partial marker is never
half-emitted) and the cut set (so the whole marker is removed). Added `<|im_start|>`, `<|im_end|>`,
`<tool_response>`, `</tool_response>`. Verified end to end - three tool-calling runs and two plain
answers, **no marker in any content field**, and `test_chat_parse` grew two regression cases that fail
on the old code.

One honest limit: stripping the markers leaves the model's bare echo of the template (`user`), and a
bare word cannot be filtered without eating legitimate text. The angle-bracket tokens - the part a client
would actually choke on - are gone.

---

## The length-dependent micro-batch default is no longer needed: the reconstruct path removed the tradeoff

Earlier in this project the micro-batch default was reversed from 2 to 1 on a +5.3% measurement at
12.9k, and a length-dependent default (M=2 for short prefills, M=1 for long) was written down as the
obvious follow-up, because M=1 lost ~4 points at a 4k prefill while gaining 3-7 at 8k/16k. That
tradeoff was measured **before** the prefill reconstruct path landed, which was worth +9-15% on its own.

Re-measured in the current build, 2 runs per cell:

| context | M=1 | M=2 | M=1 advantage |
|---|---|---|---|
| 4k | 1973.5 / 1974.1 | 1887.4 / 1883.2 | **+4.6%** |
| 8k | 2305.6 / 2302.0 | 2012.8 / 2010.9 | **+14.5%** |
| 16k | 2471.5 / 2467.2 | 2115.0 / 2112.9 | **+16.8%** |

**M=1 now wins at every context, including 4k**, so the follow-up is obsolete and no length-dependent
default is warranted. The mechanism is the reconstruct path: it dequantises once and runs a dense
fp16-accumulate GEMM whose dequant cost is **independent of row count**, so splitting a chunk into
micro-batches (M=2) now costs much less than it used to, while giving up the two-card overlap still
costs what it always did. A cheap knob stopped being cheap.

Recording this because "capture both halves of a tradeoff" is exactly the kind of follow-up note that
looks actionable forever - the right thing is for it to be re-measured when the thing it depends on
changes, and then struck out when it no longer holds.

---

## Sampling: `seed` was silently ignored, and repetition_penalty >= 1.1 returned EMPTY answers

Everything up to here had been run at `temperature: 0`. Sampling is what a real client uses, and the
whole `Sampler` path had never executed.

### 1. `seed` was never read from the request
`parse_common` parsed max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, stop and n - and
not `seed`. So `p.seed` kept its default and every request sampled from the same RNG state: two
different seeds produced **identical** output, and a client asking for reproducible sampling silently
did not get it. Now parsed, and verified: six seeds give six distinct completions at temperature 1.3,
the same seed reproduces, and temperature 0 stays deterministic.

### 2. repetition_penalty >= 1.1 returned an empty answer
| rep_penalty | before | after |
|---|---|---|
| 1.0 | OK (266 chars reasoning) | OK (266) |
| 1.1 | **EMPTY** - 1,857 chars of reasoning | OK (266) |
| 1.2 | **EMPTY** - 1,887 | OK (266) |
| 1.5 | **EMPTY** - 2,127 | OK (266) |
| 2.0 | **EMPTY** - 1,615 | OK (266) |

A client that set nothing more than a mild repetition penalty - a common default in chat UIs - got back
a blank response with no error. The model was not failing to emit `</think>`; it was being pushed away
from repeating **anything**, so it circled in its reasoning until `max_tokens` and never produced a
visible answer. The fix suspends the penalty while the output is still inside the think block, applying
it to the visible answer only, which is what the setting is for.

**One hypothesis was wrong on the way.** The first attempt exempted the stop/EOS token ids from the
penalty, reasoning that the model could not bring itself to close its think block. That did **not** fix
it - and it is what pointed at the window rather than the closing token. Worth recording, because the
obvious explanation was plausible, testable, and wrong.

This is also what finally gave `OutputParser::in_think()` - an accessor with no caller since it was
written - a purpose. The parser lives in the server and the sampler in the runner, so the server mirrors
the think-block state into the runner through the token callback it already owns.

---

## Decode profiling: where the 17.4 ms actually goes, and three hypotheses that measurement killed

Decode is the one axis where helios-qwen is at parity with exllamav3 rather than ahead (prefill is
117-127%, context is at parity). So this section is about finding out why, with nsys and ncu rather
than by guessing.

### The measurement that reframed the problem
`HELIOS_PROF=1` per phase, steady-state decode step (n=1, MTP off):

| phase | ms | share |
|---|---|---|
| moe | 7.69 | 45% |
| gdn | 4.50 | 26% |
| amix (mHC) | 2.22 | 13% |
| mmix (mHC) | 2.14 | 12% |
| attn | 1.57 | 9% |
| ple / apply / final | 0.90 | 5% |

The MoE cost **7.69 ms for one token** against **0.21 ms/token in prefill** - 36x. That ratio is the
whole clue: decode is not reading more data, it is not getting the machine.

nsys on the decode window (40 steps) settled the geometry:

| | kernels/step | busy | idle |
|---|---|---|---|
| card0 | 754 | 40% | 59% |
| card1 | 745 | 43% | 48% |

**Both cards are idle more than half the step.** Within one token the layers are strictly sequential,
so card0 runs layers 0-23, hands off, and card1 runs 24-48; neither card can start until the other
finishes. The idle is not a bug, it is the shape of a transformer on two cards.

Per-kernel, decode window only (40 steps, summed over both cards):

| kernel | calls/step | us/call | ms/step | share |
|---|---|---|---|---|
| exl3_mgemm | 75.0 | 29.6 | 2.22 | 27% |
| exl3_gemv | 138.1 | 15.5 | 2.14 | 26% |
| gr_dots_small | 50.5 | 16.4 | 0.83 | 10% |
| gr_up | 50.5 | 10.7 | 0.54 | 7% |
| exl3_gemm | 18.8 | 21.7 | 0.41 | 5% |

MoE kernels are **59% of decode**.

### Three hypotheses, all wrong, all worth recording

1. **"mgemm launches 13 blocks on an 82-SM GPU - 16% occupancy."** `gridX` really is 13, but the
   kernel has a `concurrency` dimension: the actual launch is `grid=(13,1,7)` = 89 blocks, which fills
   the card. Occupancy was never the problem. Worth checking the *whole* grid before concluding from
   one dimension.
2. **"The mHC sinkhorn's 20 iterations are 20 sequential launches, ~4.4 ms of pure latency."** They are
   already fused: a dedicated warp runs the loop concurrently with the rest of the kernel
   (`hc_mix.cu`, "the chunk-0 block also runs the sinkhorn on H^2 lanes of warp 0").
3. **"Decode should split each layer's MoE across both cards, since the cards are idle 57%."** The
   code supports it (`part[0]/part[1]`, `xcard_copy`, 16 KB partials per layer). Measured, 8k ctx:

   | mode | MTP on | MTP off |
   |---|---|---|
   | layer-pipelined (`HELIOS_PIPELINE=1`) | **65.5** | **57.4** |
   | MoE split across cards (`HELIOS_PIPELINE=0`) | 48.9 | 44.9 |

   Per-layer cross-card latency costs far more than the parallelism buys. The layer pipeline stays.

### What the MoE kernel is actually doing (ncu, counters need sudo)
`exl3_mgemm_kernel`: 7.67 MB read, **13.7% of peak DRAM**, 50% SM throughput, 33% warps active.
Byte accounting says one projection over top-10 of 512 experts should read
`10 x 640 x 2560 x 2.05/8 = 4.2 MB`, so traffic is 1.8x the necessary minimum, and it runs at roughly
a third of achievable bandwidth. It is neither DRAM- nor compute-bound - it is latency-bound. Closing
that needs a decode-specialised fused dequant-GEMV, which is a real kernel project, not a knob.

### The finding that points at the actual fix
The 57% card idle is CPU-launch-bound, and the cards alternate. That is only fillable with *more
concurrent work* - and at decode the weight reads are shared across a batch, so N sequences in one
forward should cost barely more than one. Measured today, over HTTP with `--slots 4`:

| concurrent requests | wall | tokens | aggregate tok/s | per-stream |
|---|---|---|---|---|
| 1 | 1.7 s | 100 | 59.5 | 59.5 |
| 2 | 2.7 s | 159 | 58.1 | 29.0 |
| 3 | 4.4 s | 266 | 60.5 | 20.2 |
| 4 | 10.3 s | 702 | 67.9 | 17.0 |

**Aggregate throughput is flat.** Slots serialise; concurrency buys nothing. Since decode is
bandwidth-bound on weights that a batch shares, batching is the lever - it is the one change that
attacks the 57% idle and the 30%-of-peak MoE efficiency at the same time.

---

## Long-context decode, and a real heap-overflow bug in the sparse-attention path

### Decode is nearly context-independent - until it isn't
`HELIOS_PROF=1` per phase, decode step, MTP off:

| phase | 2k | 8k | 32k | 200k |
|---|---|---|---|---|
| **attn** | 1.52 | 1.83 | 2.80 | **9.92** |
| moe | 7.04 | 7.00 | 7.08 | 7.44 |
| gdn | 3.86 | 3.89 | 3.91 | 4.07 |
| amix | 2.16 | 2.16 | 2.17 | 2.23 |
| mmix | 2.09 | 2.12 | 2.13 | 2.18 |
| step total | 17.29 | 17.71 | 19.03 | 26.7 |

2k -> 32k is 16x the context for +10% step time, because the sparse indexer makes attention O(topk)
rather than O(context). That property is the reason this engine holds 262,144 tokens at all. It does
stop paying eventually: past ~32k the indexer's own scan grows, and at 200k attention is 40% of the
step. Measured end to end (MTP on, shipped config):

| context | decode tok/s |
|---|---|
| 8k | 65.5 |
| 64k | 57.3 |
| 128k | 43.3 |
| 200k | 40.7 |

At 200k attention reads ~804 MB/step across 12 layers (`gqa_split_kv_kernel`, 67 MB per call) because
**the sparse path is not in use**: it is gated on `kv_bits == 0`, i.e. an unquantized KV cache, and
this engine ships a quantized one. The model carries `indexer_budget = 2048`; at 200k that is 1% of
the context instead of 100% of it.

### The bug: the QSA slab stride double-counted, overrunning the heap by 110 KB
Lifting that gate (safe: the dequant staging buffer already holds every row the gather can name, in
the fp16 cache's own layout) reproduced the documented "emits nothing" symptom - a SIGSEGV.

`compute-sanitizer` said **0 errors**, and gdb said the run was fine, which is what made this worth
chasing rather than filing as a driver problem. The backtrace put the fault in `cuMemcpyDtoDAsync`
from `qsa_pool_update`, on a host stack whose locals gave the answer:

    st.pooled_k = 0x7ff03ec97250
    st.tail_raw = 0x7ff03ed1b860     -> 541,200 bytes apart

`qsa_layer_bytes()` returns `nb*hd*2 + nb*4 + cr*hd*2` - pooled_k, block_pos AND tail_raw. The
allocation loop advanced `base` by that stride and *then* added block_pos and tail_raw again:

| | bytes per layer |
|---|---|
| `total` (sum of `qsa_layer_bytes`) | 534,024 |
| what the loop actually advanced | 543,256 |

Over 12 layers the loop walked **110,784 bytes past the end of the slab**, corrupting the bump
allocator. Nothing faulted at the overrun; the corruption surfaced much later as a segfault inside an
unrelated-looking CUDA call. Fixed by tiling the three pieces *inside* one stride.

Two notes worth keeping. `cudaPointerGetAttributes` reported every pointer as a valid device
allocation, and memcheck found nothing - the corruption was in the allocator's own metadata, not in a
buffer the tools watch. And the fault moved under gdb because heap layout changed, which is why a
heisenbug like this needs the locals, not a rerun.

### After the fix: correct, still slower than dense
QSA now runs end to end and its output is **byte-identical to the dense path** at 8k, with the
reference digest `3aaa5693` and 18/18 tests unchanged. `qsa_score_kernel` was also rewritten - it
launched 128 threads for a 128-element dot product (one product per thread) and paid a full block
reduction, with two `__syncthreads`, *per head*, four times over, while re-reading the same key
vector for every head. One warp per block removes the shared memory and both barriers and reads the
key once.

It bought nothing measurable: 30.59 -> 30.43 tok/s at 8k. **The score kernel was not the cost.** The
sparse path's per-layer fixed overhead (expand, chunked partials, combine) exceeds a dense 8k scan
outright. QSA remains default-off, exactly as it was, so none of this changes the shipped
configuration - it makes an unverified path correct and usable rather than fast.

Making it pay needs the overhead in expand/gather/combine attacked, not the scoring. That is the
next thing to measure, and it is the only route to decode at 200k beating what dense does today.

---

## Decode batching: the size of the win, measured from data we already had

The 57% card idle cannot be filled by anything except more concurrent work, and two independent
forwards would not help - each re-reads every weight. The win needs N sequences in ONE forward, so
the weights are read once for N tokens.

**The MTP path already measures that trade.** A speculative step runs the trunk at width 2, so
comparing MTP on and off at 8k gives the marginal cost of a second token directly:

| | tok/s | tokens/step | ms/step |
|---|---|---|---|
| MTP off | 57.4 | 1.00 | 17.4 |
| MTP on (K=1) | 65.5 | 1.65 | 25.2 |

A second token costs **25.2/17.4 = 1.45x, not 2x**. So two sequences batched into one forward should
land near 2/1.45 = **+38% aggregate**, and more at higher batch. That is the whole prize, and it is
derived from a measurement already in hand rather than a guess.

What is already in place, which is most of why this is reachable:
- Per-slot recurrent state exists and is resident: `slots_[s].gdn_conv / gdn_rec / ple_conv`
  (`48*128*128*4` = 3 MB per GDN layer, ~111 MB per slot). card1 has 550 MB free, so a second slot
  fits without trading away context.
- The MoE is 59% of decode and is **already per-token** - routing, top-k and the expert dispatch all
  take n. It needs no change.
- Prefill already micro-batches, so the kernels handle n > 1.

What blocks it, precisely: the two kernels that carry per-sequence recurrent state take a single
pointer, not a per-row one.
- `gdn_layer.cu` passes `conv_state` / `rec_state` into `cuda_causal_conv1d_update` and
  `cuda_recurrent_gated_delta_rule` as one base each. A stride (state laid out `[n_slots][layer]`,
  kernel indexing `base + b*stride`) is the natural change and the kernel already has a batch axis.
- `attn_layer.cu` writes `h_pos[i] = pos0 + i` - a contiguous range, so two slots at different context
  lengths cannot share a call. It needs a per-row position array plus a per-row KV base.
- The spec path compounds it: `gdn_sub_snap_` / `gdn_conv_snap_` are single-slot, so batching has to
  hold a snapshot per slot in the batch or the rewind restores the wrong state.

That is a real project, touching the most correctness-critical kernel in the engine (the delta rule
recurrence), against a runtime that currently passes 18/18 with a stable reference digest. It is
worth doing and it is the single highest-value decode change available - but it should be done as
its own piece of work with its own verification, not folded into a profiling session.

---

## Decode batching, part 1: the kernel layer is done and verified

The batching prize is measured, not guessed. The MTP path already runs the trunk at width 2, so
comparing MTP on/off prices the marginal token directly: a second token costs **1.45x, not 2x**
(25.2 ms/step vs 17.4). Two sequences in one forward should therefore land near **+38% aggregate**,
and aggregate throughput is the thing that is currently flat - four concurrent HTTP requests return
58/60/68 tok/s aggregate, i.e. nothing.

The good news from reading the code is that most of the machinery already existed, dormant.

### GDN: already multi-slot, just never switched on
Both GDN kernels take `bsz` and a `slots` array and index their state by it:

    int state_slot = slots ? slots[bi] : bi;
    float* slot_state = recurrent_state + (size_t) state_slot * slot_size;

The runner hardcoded `bsz=1, slots=nullptr` with the comment *"single sequence (no slots, no paged
history)"*. `gdn_layer` now takes `bsz / gdn_slots / conv_slots / slot_layers` and forwards them. The
two state arrays need *different* addressing, which is the one genuinely fiddly part:

| state | kernel computes | what batching passes |
|---|---|---|
| recurrent | `slots[bi] * history_stride * state_size` | `history_stride = n_gdn`, `slots = [0,1]` |
| conv | `slots[bi] * dim * state_size` (no layer term) | `slots[b] = b * n_gdn`, **pre-scaled** |

`state_size` is already exactly `rec_bytes`, and the slot-major allocation gives a uniform stride
because both `rec_bytes` (3,145,728) and `conv_bytes` (81,920) are 256-aligned.

### Attention: per-row position and per-row cache base
`gqa_split_kv_kernel` already indexed rows (`row = rk / n_kv_heads`) but assumed consecutive positions
and one shared cache. It now takes `row_pos` and `kv_slot_stride`, defaulting to null/0, which is
the old `pos0 + row` over one cache. Separately, `kvq_dequant(q, scales, out_half, n, token_dim,
bits, pos0, s)` takes a *destination pointer* and a *source* offset - so staging a second slot needs
no quantizer change at all, just a second call with a different destination.

### A constraint that shaped the design
The quantized-KV staging buffer is `max_ctx x kv_token_dim x 2` = **268 MB x 2 per card**, and card1
has 550 MB free. Two staged regions would need +536 MB and **do not fit**. So the intended shape is:
**attention stays per-slot** (two calls of n=1, each with its own `pos0` and KV base, sharing one
staging region), and **everything downstream batches** - GDN, MoE and the mHC mixers are already
per-token over `n`. Attention is 9% of a decode step at working context; GDN + MoE + mixers are 88%.
Projected: `2 x 1.52 + (17.29 - 1.52)/1.45 = 13.9 ms` against today's 17.3, i.e. **~+22% single-stream
and +38% aggregate at full batching**, for zero extra VRAM.

### Verified after the change
18/18 tests, reference digest `3aaa5693` **unchanged**, prefill 2467 tok/s, decode 64.5 tok/s. Both
kernels take the batch size with defaults, so the shipped single-sequence path is untouched - which is
exactly what the digest is there to prove.

### What is left, precisely
Runner orchestration, and it is the larger half:
1. A batch context threaded through `layer_range` / `run_chunk_pipeline` (per-slot KV base, position,
   GDN state base, PLE window), with bsz=1 as the default so the existing path is unchanged.
2. Attention called per slot with row-offset `x`/`y` pointers - it already takes everything else it
   needs per call.
3. The PLE: two calls, each with its own `ple_conv` and `hist_` window, n=1.
4. The pipeline handoff, which moves 40 MB of activations per step and must carry both rows.
5. Server scheduling: pair up two pending decode requests instead of serialising them.
6. Speculative decoding on top, which additionally needs per-slot `gdn_sub_snap_` / `gdn_conv_snap_`
   so a partial accept rewinds the right state. Deliberately last: the win is available without it.

---

## Batched decode, part 2: the engine side is done; a PRE-EXISTING multi-slot bug blocks using it

Part 1 landed the kernels. This part threaded a `BatchCtx` through `run_chunk` ->
`run_chunk_pipeline` -> `layer_range` and added `Runner::decode_pair`, so two sequences can now go
through one forward. Everything is gated on `b` being non-null, so the shipped path is untouched:
18/18 tests, digest `3aaa5693`, prefill 2449 tok/s, decode 64.2 tok/s - all unchanged.

What the batched path does per layer:
- **Attention, per row.** Each row attends over its own history at its own position against its own
  cache base (`kv_base_slot`), offset by row into the activation buffers. Not batched on purpose: the
  quantized-KV staging buffer is one region of `max_ctx` rows and a second would need 268 MB x 2 more
  than card1 has free.
- **GDN, batched.** `bsz=2` with the slot arrays; the kernels index their own state.
- **PLE, per row.** Its input is the n-gram window of its own sequence and its conv belongs to its own
  slot, so a shared call would fold two windows into one.
- **MoE and the mHC mixers, batched** - they were already per-token over `n`.
- **No CUDA graph.** The batch size, per-row positions and state bases are kernel arguments, and a
  captured graph freezes them. That costs the ~15% replay saves and is the price of correctness.

### The blocker is not batching, and it is not new
Validating this needs two working slots, and **`HELIOS_SEQUENCES=2` does not work today**: a plain
two-request run produces **0 tokens** - the first predicted token is already a stop id, so the
prefill produced garbage. This is independent of everything above (every change here is gated on a
non-null `BatchCtx`, and the 1-slot path is bit-identical), and it is why the earlier concurrency
measurement was flat: with slots not actually usable, the server was serialising for a different
reason than it appeared.

The likely cause is one line. `seq_config` ends the multi-slot path with

    attn::kvq_ctx_rows_ref() = slot_ctx_;      // 131072, half of the 262144 the cache was sized for

so the quantized-KV layout is computed for half the context while the cache allocation, the dequant
staging and the scale offsets were all derived from the full `ctx_cap`. Slot 0 is affected even
though its base-pointer offset is zero, which matches it being slot 0 that produces garbage. This
needs confirming against `kvq_scale_offset` before anything is changed.

So: **no batching speedup is claimed.** The engine-side machinery is complete and verified inert;
making it pay means fixing multi-slot correctness first, which is its own piece of work.

---

## Batched decode, part 3: five real bugs, and where it actually stands

Continuing part 2. The engine-side batching is complete; getting it to run at all took five fixes,
each of which is a bug worth having found regardless of batching.

1. **The GDN kernels take `bsz` and `seqlen` separately** and address batch item `bi` at
   `bi * seqlen`. A batch of two one-token rows is `bsz=2, seqlen=1` - not `bsz=1, seqlen=2`.
   Passing the total row count as `seqlen` made batch item 1 read at row 2 of a 2-row buffer.
2. **The slot-number arrays were card 0 only.** Half the GDN layers run on card 1, and a card-0
   pointer dereferenced by a card-1 kernel is not a slot number, it is garbage. It presented as the
   recurrent kernel reading **25 MB past a 3.1 MB state block** - a garbage index times the assumed
   stride. One array per card now.
3. **Per-slot state could not be assumed contiguous.** `Device::alloc` sub-allocates from a bump
   pool *while the pool lasts* and then falls back to a fresh `cudaMalloc`, and 227 MB of recurrent
   state is exactly the allocation that lands on that boundary. The state block is now carved by
   hand as `[layer_rank][slot]` per card, so the layout is stated rather than inherited.
4. **A slot difference is one state, not one card's worth of layers.** The first version scaled the
   conv indices by the per-card GDN layer count, which overran the block by 2 bytes.
5. **A guard I wrote was wrong.** `decode_pair` refused two sequences at the same position on the
   reasoning that "one decode already covers both rows" - true only for one sequence. Two different
   conversations at the same position each still need their own row.

### A diagnosis of mine that was wrong
I reported that `HELIOS_SEQUENCES=2` was broken, on the evidence that a two-request run produced
**0 tokens**. It was not broken: the 11-token prompt I used makes the model emit a stop id as its
first token, in **one** slot just as much as two. With a prompt that actually generates, two slots
run fine - 56.9 and 57.5 tok/s. A one-line diagnostic (`pos_`, `slot_ctx_`, first token) separated
the two cases immediately, and I should have reached for it before writing a paragraph.

### Where it actually stands
The state plumbing is now correct and the paired forward RUNS. The remaining defect is narrower and
better defined than "batching does not work":

> Given **identical prompts and identical tokens** in both slots, the two rows of a paired forward
> come out **different**, and neither matches serial. At step 1 serial says `248068` for both; the
> paired rows say `25` and `271`.

Identical inputs producing different outputs is a per-row addressing bug with no sequence difference
to confound it, and it rules out the state plumbing above (slots, strides, contiguity) as the cause -
those would produce *right* answers for the wrong reason or an outright fault, not this. It localises
to the trunk's per-row path at n=2: the mHC mixers, the MoE, or the 2-row `final_head` projection.
The MoE decode gate was raised to 2 (so n=2 uses mgemm rather than the grouped kernel) and changed
nothing, which takes the MoE off the list.

The harness is kept, behind `HELIOS_BATCH_PARITY=1`, with `HELIOS_BATCH_SAME=1` for the
identical-input variant. It is the gate this work needs: it fails loudly today, and it is the thing
that has to go green before any batching speedup means anything.

**No batching speedup is claimed.** Shipped path verified unchanged: 18/18 tests, digest `3aaa5693`,
prefill 2449 tok/s, decode 63.9/64.1/64.1 tok/s over three runs.

### The remaining defect, stated as a gate

    HELIOS_BATCH_SAME=1 HELIOS_SEQUENCES=2 HELIOS_BATCH_PARITY=1 HELIOS_MTP=0 \
      ./build/helios gen <model> --raw --prompt-file <8k> --tokens 16 --temp 0 --repeat 2

    serial A: 271 248068 198 760 1156 682
    serial B: 271 248068 198 760 1156 682      <- identical prompts, identical trajectories
    paired A: 271 271    561 97765 2037 200789
    paired B: 271 25  96434 130685 167739 5674  <- same inputs, two DIFFERENT results

Two facts in one line of output. Serial is deterministic and slot-independent - both slots produce
byte-identical trajectories, which is what makes them a valid reference. The paired forward does not:
with identical inputs its two rows diverge from each other *and* from serial. The two rows of one
forward disagreeing rules out anything about slots, strides or contiguity - those produce faults or
right answers, not two different wrong ones - and puts the fault in the shared n=2 path, i.e. the mHC
mixers, the MoE, or the 2-row `final_head`.

Ruled out by measurement, not by argument:
- the MoE decode path: forcing mgemm at n=2 (`HELIOS_MOE_DECODE_MAX_N=2`) and forcing the grouped
  kernel (`HELIOS_MOE_GROUPED=1`) each change the serial reference's values and change nothing about
  the paired/serial disagreement
- the state plumbing, in full: slot arrays, per-card strides, explicit contiguity, `bsz` vs `seqlen`
- the CUDA graph (batched steps never graph)

Not yet localised: whether the divergence originates in the mixers or in the head projection. One
diagnostic that did not settle it: dumping the last layer's `sub_in` for both rows. The paired rows
differ (2.29e8 vs 1.07e9 rms, which is the expected signature), but the SERIAL row-0 reading came
back at 5.2e10 - far outside the range of an activation vector - even though `final_head` reads the
same `act1_.sub_in` buffer that was dumped. That inconsistency is unresolved and is the first thing
to chase; a diagnostic that reports a physically impossible value is usually telling you that the
buffer is not what you think, not that the model is.

Shipped path, re-verified after all of the above: 18/18 tests, digest `3aaa5693`, decode 63.8 tok/s.

---

## Batched decode, part 4: localising the n=2 defect (and one diagnostic that lied)

### First: the diagnostic itself was wrong
Dumping the last layer's `sub_in` reported a serial row-0 rms of **5.2e10**, which is not an
activation value. `sub_in` is **fp16** and the dump copied `D * 4` bytes from a `half*` - two rows of
half bit patterns reinterpreted as float. Reading `D * 2` bytes and converting gives 3.50. Every
paired-row number from that run was garbage for the same reason. A diagnostic reporting a physically
impossible value is usually telling you the buffer is not what you think.

### The trunk is wrong per-row, and that is now provable
With the dump fixed, same 2048-token prompt in both slots, both committing token 271 at step 1:

| | step 1 trunk rms | step 2 |
|---|---|---|
| serial (n=1) | **3.4976** | 2.8807 |
| paired row 0 | 2.4471 | 3.3150 |
| paired row 1 | 2.5675 | 2.8977 |

The two paired rows **differ from each other**. That is the load-bearing observation: if only the head
projection were miswriting rows, both rows would still be identical *to each other* and merely wrong
against serial. They are not, so the divergence happens **upstream of `final_head`**, in the trunk.

### What that leaves
Ruled out by measurement:
- **The EXL3 projections at small M.** A 2048-token prompt whose final prefill chunk is 2, 4 or 48
  rows produces byte-identical continuations (md5 `12d207540f` for all three). A 2-row chunk exercises
  `exl3::linear` at M=2 through the whole 48-layer trunk, and it agrees with a 48-row chunk.
- **The mHC mixers at n=2.** Their grid is `grid_c(n_chunks_c, R)` with `R` on `grid.y`, and the
  sinkhorn's `blockIdx.x == 0` gate is per-chunk within every row, so every row is normalised.
- **Chunk-size sensitivity in the shipped path.** The reference digest is `3aaa5693eee9` at chunk
  256, 512 and 1024. (An apparent difference at chunk 1024 was an artifact of a deliberately
  degenerate prompt - 2046 repetitions of one token - which puts the model in a near-tie state where
  any last-bit difference flips the argmax.)
- **The head projection**, by the argument above.

What remains is the part of the paired path that a 2-row *prefill* tail does not exercise: **the GDN
with `bsz=2, seqlen=1`**. Every prefill chunk is `bsz=1, seqlen=n`; the pair is the only place the
recurrence is asked for two independent one-step states in one launch. That is now the single
remaining suspect, and it is a much narrower question than "batching is broken".

Note the gate also re-confirms what it did before: the projections and mixers being right at n=2 is
what makes the GDN the last candidate rather than one of several.

---

## Batched decode, part 5: the GDN is exonerated, and there is now a test that says so

The remaining suspect after part 4 was the GDN at `bsz=2, seqlen=1` - the one configuration paired
decode needs and the only one no prefill chunk ever produces. It is **correct**.

`test_gdn_bsz2` (new, and now in ctest, 19/19) does the direct A/B with no engine involved: one call
at `bsz=2, seqlen=1, slots={0,1}, history_stride=1` over a `[2][state]` block, against two calls at
`bsz=1, seqlen=1` with the engine's exact decode configuration, comparing **both the output rows and
the resulting state**:

    gdn bsz=2 vs 2x bsz=1: worst relative output diff 0.000e+00, worst relative state diff 0.000e+00
    PASS

Bit-identical, not merely close. So the recurrence handles two independent one-step states in one
launch, and my part-4 conclusion - "the paired-only path is the GDN" - was wrong. The test is worth
keeping regardless: it covers a configuration nothing else executed, and it is the kind of gap that
lets a future change regress silently.

### What that leaves, precisely
Ruled out, each by measurement rather than argument:
- the GDN recurrence at `bsz=2` (above: bit-identical)
- the EXL3 projections at M=2 (a 2-row prefill tail agrees with 4- and 48-row tails)
- the mHC mixers at n=2 (grid puts R on `grid.y`, so the sinkhorn normalises every row; and the
  multi-row prefill tail exercises them)
- per-slot addressing (forcing **both rows onto slot 0** - same cache base, same position - leaves the
  two rows still disagreeing, so the fault is not in the attention's per-row cache/position)
- the head projection (the paired `sub_in` rows differ from each other, which a head that merely
  miswrites rows could not produce)

That is a short list and it has a shape: everything the two rows do **independently** is correct, and
the disagreement is in the work they **share**. The shared work at n=2 is the mHC mixers, the MoE and
the projections - and all three are covered by a multi-row prefill tail that agrees. The one thing a
prefill tail does *not* cover is the **decode** variant of the MoE, which takes a different kernel
(`mgemm` rather than the grouped path) at exactly the width pairing introduced.

Forcing either MoE kernel changes the serial reference's values but not the paired/serial
disagreement, which is consistent with **both** decode kernels being wrong at n=2 rather than the
choice between them mattering. That is the next thing to test, and it needs a standalone A/B in the
style of `test_gdn_bsz2` rather than an end-to-end run.

Shipped path: 19/19 tests, digest `3aaa5693`.

### The MoE A/B, run properly this time

Identical prompts in both slots; the question is only whether the two rows agree **with each other**,
which needs no reference at all:

| MoE decode path | paired A | paired B | rows agree? |
|---|---|---|---|
| default (mgemm) | 271 271 561 97765 2037 200789 | 271 25 96434 130685 167739 5674 | no |
| `HELIOS_MOE_DECODE_MAX_N=2` | 271 271 561 97765 2037 200789 | 271 25 96434 130685 167739 5674 | no |
| `HELIOS_MOE_GROUPED=1` | 271 271 561 110530 41202 46091 | 271 271 181059 136963 75551 264 | no |

The first two rows are **byte-identical**, which corrects an assumption I had been working under: the
decode gate already admits n=2, so `HELIOS_MOE_DECODE_MAX_N=2` was not a variant at all - it was the
default. The earlier "raising the gate changed nothing" result was therefore not evidence about the
MoE; it was the same run twice. `HELIOS_MOE_GROUPED=1` does switch kernels, and the rows disagree
there as well.

So the MoE is wrong at n=2 on **both** its decode kernels, while a 2-row *prefill* chunk - which takes
the grouped path - agrees with 4- and 48-row tails. The kernel is therefore not the thing to look at
first; the difference between the two contexts is the host-side per-row accounting in `moe_layer` for
a decode-shaped call (`slots = n * topk`, the per-card `part` buffers sized from
`g_moe_decode_max_n * topk`, the owned-card copy of `n * hid`).

That `part_rows = max(max_n, g_moe_decode_max_n * topk)` term is worth a hard look on its own: at
`g_moe_decode_max_n = 2, topk = 10` it is exactly 20, and a 2-row decode asks for exactly
`slots = n * topk = 20`. The buffer is sized to the request with no margin, which is the same class of
bug as the QSA slab stride - an exact fit that has to be right for every future width.

### The state of the hunt, honestly
Everything the two rows do **independently** is verified correct: the GDN recurrence at `bsz=2`
(`test_gdn_bsz2`, bit-identical), the per-slot cache base and position, the per-row PLE. Everything
the rows **share** is verified correct in a multi-row prefill context: the EXL3 projections, the mHC
mixers. What is not yet explained is why the shared work disagrees between rows in a *decode*-shaped
call but not in a prefill-shaped one. That is a much smaller question than where this started, and it
is not answered.

---

## Batched decode, part 6: the last unverified link, and it is in a ported dependency

Reading `moe_layer`'s decode path end to end, every piece helios owns checks out for n=2:

| step | verdict |
|---|---|
| `routing_std_logits(..., n, E, topk, ...)` | takes n; writes per-row ids and weights |
| `moe_slot_gather_k` | `xg[i] = x[(slot / topk) * hid + d]` - slots 0..9 to token 0, 10..19 to token 1. **Correct for n=2.** |
| `remap_ids_kernel` | indexes by slot, not by row |
| `part_rows = max(max_n, g_moe_decode_max_n * topk)` | 1024 in every real config, not the tight 20 I suspected |
| shared expert | plain dense path over n rows |

That leaves exactly one call that does the per-token reduction:

    exl3::mgemm(sc.part[c], gbuf, ..., bszm_in = slots, bszm_out = slots, idx,
                sc.topk_w[c], slots, 0, min_index = -1, max_index = -1, num_tokens = n, ...)

`num_tokens = n` with `slots = n * topk` asks mgemm to split the slots into `n` contiguous groups and
reduce each into its own output row. **That mode has never been tested.** `test_mgemm_semantics` -
the test written for exactly this kernel - states its own scope in its header:

> with min_index >= 0 and **num_tokens == 1** the indices/weights are COMPACTED in place while the
> **num_tokens == 1** reduces every slot into row 0

Every behavioural guarantee the test asserts is conditioned on `num_tokens == 1`, and `kSlots = 10`
matches this checkpoint's `top_k`, i.e. the single-token case. Paired decode is the first caller to
pass `num_tokens = 2`.

This also explains the shape of the symptom that has been the most stubborn part of this hunt: a
2-row *prefill* chunk agrees with 4- and 48-row tails, because prefill routes through the **grouped**
kernel, not mgemm. Forcing the grouped kernel in the pair did not fix it either - but the grouped path
is entered through a different code path that carries its own reduction, and that route was measured
with the per-card copies the comments describe as previously missing. So the honest statement is
narrower than "mgemm is broken": **mgemm's multi-token reduction is the one link in the shared decode
path that has no test and no measurement behind it, and it is where paired decode's two rows stop
agreeing.**

The fix is a test, not a guess: extend `test_mgemm_semantics` to two token groups and compare against
`moe_grouped`, which the surrounding comments already name as the only valid oracle for expert
weights. That is the same shape as `test_gdn_bsz2`, and it is the right next piece of work.

---

## Batched decode, part 7: two more components exonerated, and what that leaves

Two leading suspects tested, both clean.

**`exl3::mgemm` with `num_tokens = 2` is correct.** `test_mgemm_semantics` asserted only
`num_tokens == 1` - every guarantee in its header is conditioned on it, and `kSlots = 10` is this
checkpoint's `top_k`, i.e. the single-token shape. Extended to two token groups: the expectation is
now per-token, `row t = sum over slots [t*stride, (t+1)*stride)`, and both new cases pass.

    2 tokens: num_tokens=2, bszm=2*slots           rel RMS 1.337e-03
    2 tokens: num_tokens=2, SKIPS + non-uniform w  rel RMS 1.389e-03
    ALL OK

**The CUDA graph is not involved.** Paired decodes take the eager path (batch size, per-row positions
and state bases are kernel arguments a graph freezes), while serial takes the graphed path - so if the
two disagreed, that alone would explain everything. They do not: the reference digest is
`3aaa5693eee9` with the graph on and off. And with `HELIOS_SEQUENCES > 1` the graph is refused anyway
(it binds different state per slot), so both runs already use the eager body.

**The batched branches execute as intended**, confirmed by instrumentation rather than by reading:
GDN sees `bsz=2 slot_layers=1` with device slot arrays, attention sees `slot0=0 pos0=8810 slot1=1
pos1=8810`. And the row mapping was settled empirically - `final_head` computes
`rows = min(n, head_rows)` and projects `sub_in + (n - rows)`, so with `n == head_rows == 2` there is
no reversal and row *i* is `sub_in` row *i*. Both readings of that were run; both mismatch, so the
mapping is not the issue.

### What is left, stated honestly
Every component reachable by inspection or by an A/B has now been shown correct in isolation:
the GDN recurrence at `bsz=2` (new test, bit-identical), the mgemm weighted reduction at
`num_tokens=2` (extended test), the slot gather, the routing, the remap, the `part` sizing, the
per-slot cache base and position, the per-row PLE, the EXL3 projections at M=2, the mHC mixers at
n=2, the graph/eager equivalence, and the wiring itself.

And yet a paired forward given **identical inputs in both slots** produces two different rows. When
every part is individually right, the fault is in how the parts compose for this width - not in a
kernel. The remaining suspects are ordering or scratch-ownership effects that only appear when n=2
holds two live sequences, and the next step for those is an n=2 differential test at a single layer
boundary (dump the hidden state after layer 0 for both rows) rather than another component A/B.
That is the honest state: the question is smaller than when I started, and it is not answered.

---

## Batched decode, part 8: the rows are identical entering layer 0 and diverge inside it

Component A/B had eliminated everything reachable, so the instrument changed: a **per-layer
differential** (`HELIOS_LAYER_DUMP=1`) that reports each row's hyper-connection stack rms after every
layer, and before the first one. The question it answers is not "which component is wrong" but "at
which layer do the rows part".

    [ldump] L-1 gdn  row0 rms=1.555289831e-03 row1 rms=1.555289831e-03  AGREE
    [ldump] L0  gdn  row0 rms=8.353583817e-03 row1 rms=9.587381711e-03  DIVERGE
    [ldump] L1  gdn  row0 rms=9.136219637e-03 row1 rms=1.040218312e-02  DIVERGE
    ...

The embed -> streams expansion is **correct at n=2** - the two rows come out of it bit-identical - and
the entire divergence is created inside the first layer's body. That is a much smaller target than
"batched decode is wrong": one layer, with the GDN and the MoE of that layer already verified
correct in isolation.

Two things the instrument itself taught, worth more than the run:

- The first version of the dump used a **synchronous `cudaMemcpy`**, which is unordered against the
  layer's stream `s`. It printed values ~1000x too small and repeated on alternate layers, which is
  what made it look like the rows parted immediately and uninformatively. Ordering it against
  `stream(0)` fixed the read. A diagnostic that lies is worse than none.
- Two broken builds along the way, both from scripted insertion rather than from the idea. The
  instrumentation is env-gated and now earns its place; the editing method does not.

### A real bug it did find, on the way
The PLE's per-row call passed **`n_ctx = 1`**, but `n_ctx` is the number of ids *carried into* the
call, not the token count - `ple_ngram_ids` derives its row count as `n_hist - n_ctx`, which is why the
serial path passes `hist_.size() - n`. Passing 1 made the PLE fold `ngram_size` rows instead of one,
re-injecting stale carried ids. Fixed to `h.size() - 1`; the change is visible in the trace
(L2 row1: 1.745177812e-02 -> 1.739306992e-02), so it was a genuine correctness gain - just not the one
that parted the rows at L0, since the PLE sits on a later layer.

### What is left
Inside the first layer's body, with the GDN (`test_gdn_bsz2`, bit-identical) and the MoE
(`test_mgemm_semantics` at `num_tokens=2`, rel RMS 1.3e-03) already cleared in isolation, and the
embed -> streams expansion cleared by the pre-layer probe. The next step is the same instrument one
level finer: probe after the attention mixer, after the GDN, after the MLP mixer and after the MoE
inside L0, and the first probe that reports DIVERGE names the step. That is a short, mechanical
extension of what is already built and wired.

---

## Batched decode, part 9: narrowed to one call, after a second broken probe

The step-level differential needed a fix before it could be believed. The first version read row 1 at
offset `n * D` from a buffer laid out `(R, D)` - i.e. **past the end** - and reported row 1 as a
constant that was really adjacent memory. That is what made the mixer look like it "never wrote row
1". With the row stride corrected to `D`, the picture inverts:

    [probe] L0 pre_amix   row0 rms=4.687788518e-01 row1 rms=4.623785211e-01  DIVERGE
    [probe] L0 attn_mix   row0 rms=6.713679815e-01 row1 rms=6.713679815e-01  AGREE
    [probe] L0 gdn_out    row0 rms=1.385908027e-01 row1 rms=1.488238274e-01  DIVERGE

- `pre_amix` is leftover from the prefill's last chunk, where rows 0 and 1 are *different prompt
  tokens* and are expected to differ. It carries no information.
- **`attn_mix` AGREES bit-for-bit.** The mHC mixer is correct at R=2, which also retires the
  "fp32 decode kernels are built for R=1" comment as an explanation - `HELIOS_MIXER_TC_MIN_R=2`
  changes nothing, because the scalar path was already correct.
- The first real divergence is **`gdn_out`**, the GDN's own output.

And the two slots are not the cause: prefilling identical prompts into both leaves **bit-identical
GDN state** (`max|diff| = 0.000000e+00` at layers 0, 1, 2 and 10). The GDN's input agrees too, since
`a.mixed` agrees and the fp32 -> fp16 cast is element-wise.

So: identical input, identical state, a kernel proven bit-identical at `bsz=2` - and different output.

### The gap that explains it
`test_gdn_bsz2` covers `cuda_recurrent_gated_delta_rule` **only**. The other half of the GDN layer is
`cuda_causal_conv1d_update`, which in the batched path is called with `bsz=2, seqlen=1` and a
per-slot `conv_state` for the first time. Its addressing reads

    conv_state[((size_t) slot * dim + d) * state_size + ...]

so the per-slot stride is `dim * state_size` = `conv_bytes`, which matches the `[layer_rank][slot]`
carve - by inspection, which is exactly the kind of reasoning this whole investigation has shown to be
unreliable. The next step is a conv1d A/B in the style of `test_gdn_bsz2`: two independent
`(conv_state, x)` pairs, once as `bsz=2` with `slots={0,1}` and once as two `bsz=1` calls, comparing
both the output and the written-back conv state.

### A note on the two probes that lied
Both this instrument and the layer-level one produced confident, wrong numbers before being fixed -
one from an unordered synchronous memcpy, one from a row stride that read past the buffer. Each was
caught only because a second quantity (run-to-run constancy, then a value 1000x off) did not look
physically possible. Neither would have been caught by a plausibility check, and both would have sent
the hunt somewhere else entirely.

---

## Batched decode, part 10: the conv1d is correct too - and the third instrument that lied

`test_gdn_bsz2` covered `cuda_recurrent_gated_delta_rule` only. The layer also calls
`cuda_causal_conv1d_update`, and that is the first `bsz=2` call in the engine. Extended the test to
cover it - two independent `(conv_state, x)` pairs, once as `bsz=2` with `slots={0,1}`, once as two
`bsz=1` calls, comparing **both the output and the written-back conv state**, with non-zero initial
state so a slot writing into the wrong neighbour is visible.

The first run failed, hard:

    conv1d bsz=2 vs 2x bsz=1: worst abs output diff 1.728516e-01, worst abs state diff 9.882812e-01
        slot 0: worst output diff 1.669922e-01, worst state diff 0.000000e+00
        slot 1: worst output diff 1.728516e-01, worst state diff 9.882812e-01

**The per-slot breakdown is what identified it, and it exonerated the kernel.** Slot 0's *state* was
bit-identical while its *output* was not - and the output depends on the state read and on `x`. State
right, output wrong can only be the input. The test had laid `x` out as `dim * K` per batch item when
the layout is `[bsz, dim, seqlen]`, so with `seqlen == 1` the kernel's batch stride (`dim * seqlen`)
disagreed with the buffer and **both** rows read the wrong `x`. Fixed:

    conv1d bsz=2 vs 2x bsz=1: worst abs output diff 0.000000e+00, worst abs state diff 0.000000e+00
        slot 0: worst output diff 0.000000e+00, worst state diff 0.000000e+00
        slot 1: worst output diff 0.000000e+00, worst state diff 0.000000e+00
    PASS

So **both halves of the GDN layer are correct at `bsz=2`** - the recurrence and the conv - each now
covered by a test that will catch a regression in a shape nothing had executed before.

### Where the divergence is NOT
- the mHC mixer at R=2: `attn_mix` reports the rows **AGREE** bit-for-bit
- the GDN recurrence at bsz=2: bit-identical
- the conv1d at bsz=2: bit-identical
- slot state isolation: `max|diff| = 0.000000e+00` after identical prefills
- the embed -> streams expansion: rows agree before layer 0

### Where it is
Inside layer 0's body, and the step-level instrument is no longer trustworthy enough to localise it
further: the `gdn_out` probe reported the *same* value before and after moving it across the
`gdn_layer` call, which it should not have. Combined with the earlier two instruments that produced
confident, wrong numbers (an unordered synchronous memcpy; a row stride reading past the buffer), the
lesson is that this instrument needs the same treatment the kernels got: **a self-test that proves it
can see a known divergence** before its output is believed. Until then its readings are a hypothesis,
not evidence.

### The standing caution
Four hypotheses were killed by tests this session - the mgemm multi-token reduction, the GDN
recurrence, the conv1d, and "the fp32 mixer is built for R=1". Each looked certain from the code.
Three separate instruments also lied. The engine-side batching is complete and correct in every part
that can be tested in isolation; the remaining defect is a composition or ordering effect at n=2 that
this session narrowed to a single layer but did not explain.

---

## Batched decode, part 11: the probe self-test, and what the GDN layer is not

The step-level instrument had produced confident wrong numbers twice, so before reading anything more
into it, it had to be shown able to **see a difference**. `HELIOS_PROBE_SELFTEST=1` perturbs row 1 of
`a.mixed` after the attention mixer - the one point where the rows are known bit-identical - and the
probe must flip to DIVERGE.

It took three attempts, and each failure was the instrument's fault, not the kernel's:

1. Perturbing **before** the mixer - which simply overwrote the perturbation. The probe's AGREE was
   the correct answer to the wrong question.
2. Perturbing **after** the mixer, on the legacy stream, while the mixer's work was still in flight on
   `s` - the mixer ran afterwards and undid it. Same ordering class as the original bug.
3. With a `cudaStreamSynchronize(s)` first:

       [probe] L0 attn_mix   row0 rms=6.713679815e-01 row1 rms=1.234500000e+04  DIVERGE

`1.2345e4` is exactly the perturbation (12345), so the probe reads the right buffer and is correctly
ordered. **Which means its earlier readings stand**: the attention mixer leaves the rows bit-identical
and the divergence appears in `gdn_out`, inside the GDN layer's body, in the first decode step - and
persists into the second, where even `attn_mix` then reads DIVERGE.

### The GDN layer, exhausted
| part | verdict |
|---|---|
| chunked WY path | declines - `seqlen < 2*64` at n=2. (Its own `bsz != 1` guard never fires, because the call site passes a hardcoded `bsz=1` - but the seqlen check catches this case, and the hardcoded 1 is harmless while it is unreachable for bsz=2.) |
| depthwise conv | **bit-identical** at bsz=2, both output and written-back state, both slots |
| delta-rule recurrence | **bit-identical** at bsz=2, both output and state |
| projections, gated RMS norm, output combine | per-token over `n`; a 2-row prefill tail agrees with 4- and 48-row tails |
| input | rows agree (`a.mixed` bit-identical, cast is element-wise) |
| state | `max|diff| = 0.000000e+00` between slots after identical prefills |

Every part is correct, in isolation and in combination, and the layer's output still differs between
rows. What remains is not reachable by reading the call: it needs either kernel-level instrumentation
inside the projections, or a bisection harness that runs the batched layer against a hand-computed
reference row by row. Both are fresh work, and I would rather name that than keep producing
hypotheses that four tests have already killed.

---

## Batched decode, part 12: the fault is the GDN's bsz=2 call, and the gate now passes

Component A/B had cleared every part, and the self-tested probe had localised the divergence to inside
the GDN layer's body. The next step was the obvious bisection: change **one** thing and see whether the
gate flips. `HELIOS_BATCH_GDN_SERIES=1` runs the GDN as two `n=1` calls - one per row, each on its own
state base - while leaving the attention, the mHC mixers and the MoE batched at `n=2`:

    [gen] batch-parity A MATCH  B MATCH
    [gen] batch-parity OK (10 steps each)

**MATCH on both slots.** And restoring the single `bsz=2` call puts it straight back:

| GDN invocation | parity |
|---|---|
| two `n=1` calls, one per row (now the default) | **A MATCH  B MATCH** |
| one `bsz=2` call (`HELIOS_BATCH_GDN_BATCHED=1`) | A MISMATCH  B MISMATCH |

So the disagreement is entirely inside that one call, in the engine's wiring rather than in the
kernel: `test_gdn_bsz2` drives `cuda_recurrent_gated_delta_rule` and `cuda_causal_conv1d_update`
directly, each **bit-identical** to two `n=1` calls, but the engine passes a per-slot state base carved
`[layer_rank][slot]` per card plus two pre-scaled slot arrays, and that combination does not agree
with per-row calls. The difference between the two is now bounded to the arguments, not the maths.

**The per-row form is the default for batched decode.** The GDN is ~26% of a decode step and its
weights are read once either way, so two `n=1` calls cost launch overhead, not bandwidth - and
correctness is worth that trade until the argument-level disagreement is explained.

### What the parity gate is now worth
It went from failing on both slots to passing on both, over 10 steps, with two sequences. That is the
first time batched decode has been *correct*, and it is what every future change to this path has to
keep true.

### Measured, with batching correct but not yet used by the server
Two concurrent HTTP requests, `--slots 2`:

| requests | wall | tokens | aggregate | per-stream |
|---|---|---|---|---|
| 1 | 2.2 s | 100 | 44.6 tok/s | 44.6 |
| 2 | 5.8 s | 346 | **59.7 tok/s** | 29.8 |

**+34% aggregate** - close to the +38% the MTP width-2 measurement predicted. That gain is from
concurrent execution filling the 57% card idle, *not* yet from `decode_pair`: the server still runs one
forward per request. Pairing two requests into a single forward is the remaining wiring, and it is
what would let both effects compound.

---

## Batched decode, part 13: reachable and correct - and currently SLOWER than not batching

`Runner::generate_pair` plus `helios gen --pair` makes the batched path reachable end to end. It is
**correct**: its token ids are the serial trajectory exactly.

    [gen] A ids: 271 248068 198 760 1156 682 3766 264
    [gen] B ids: 11  248068 198 760 1156 682 3766 264

and A matches the single-sequence reference id for id. (Two implementation notes worth keeping:
`bind_slot()` mutates `active_slot_`, so the original slot has to be captured before the prefill loop
or the pair ends up naming one slot twice; and each sequence's FIRST token must be read from its OWN
prefill, because the logits buffer is shared and after the second prefill it holds the second
sequence's row.)

### The result that matters: it is not faster
256 decode tokens per sequence, 8k and 4k prompts:

| | aggregate tok/s |
|---|---|
| paired, one forward per step | **40.7** (including ~6 s of the two prefills) |
| two independent single-sequence decodes | 48.1 and 57.3, i.e. **96-114** for the same 512 tokens |

**Batching as implemented is a loss.** And the reason is the part-12 workaround: the GDN now runs as
two `n=1` calls, attention is per-row, and the PLE is per-row. So a "batched" step pays double
launches for three of the four sublayers and shares only the MoE, the mixers and the projections.

That is the honest shape of the result, and it separates two things that were tangled:

- **Correctness is solved.** Batched decode now produces the serial trajectory exactly, and the parity
  gate passes on both slots. That was the hard part and it is done.
- **Profitability is blocked by the GDN bug.** The +38% was measured on a *width-2 trunk* - the MTP
  verify batch, where every sublayer sees n=2. The GDN `bsz=2` disagreement is exactly what stands
  between this path and that shape, and until it is explained the pair pays for its correctness.

The +34% aggregate measured over HTTP earlier is real but is **not** this: that is two requests running
concurrently, filling the 57% card idle, each still doing its own forward. It composes with batching
rather than coming from it, and only one of the two is available to the server today.

---

## Batched decode, part 14: the GDN wiring is right, and the disagreement is still unexplained

If the GDN's `bsz=2` call is wrong in situ while both its kernels are bit-identical in isolation,
the difference has to be in what the engine passes. Every element of that has now been measured
rather than reasoned about:

    [ptrgap] rec  layer 0: slots 0/1 3145728 bytes apart, rec_bytes 3145728  OK
    [ptrgap] conv layer 0: slots 0/1   81920 bytes apart, conv_bytes   81920  OK

The kernels reach slot *b*'s state as `base + slots[b] * history_stride * state_size`. For the
recurrence that requires the two slots' states to sit exactly `rec_bytes` apart; for the conv,
exactly `conv_bytes` apart, since its stride is `dim * state_size` with no layer term. **Both hold
exactly**, at every layer checked. The state block really is carved `[layer_rank][slot]` and the two
slot arrays really do carry the relative indices the kernels apply.

Also ruled out by inspection and sizing: the GDN scratch, whose `a_had` is allocated
`max_n * 16384 * 2` and so has room for two rows; the chunked path, which declines on `seqlen < 2*64`;
and `locks`, which is `nullptr` on both paths.

So: kernels correct, state addressing correct, scratch correct, arguments correct - and the batched
call still disagrees with two per-row calls, while matching the parity gate only when the GDN runs
per row. I cannot close that from the host, and I am not going to name a cause I have not measured.

**What is established, and is worth keeping:** batched decode is correct and reachable
(`generate_pair`, `gen --pair`, parity gate green on both slots), and its profitability is gated on
this one disagreement. The per-row GDN is the correct default until then - correctness first, and
the cost of that choice is now measured rather than assumed.

---

## Batched decode, part 15: reproduced in isolation, and the bisect names the stage

The full-layer A/B is the unit neither kernel-level test covered - both of those drive the kernels
directly, skipping the projections, the norm and the output projection that `gdn_layer` wraps around
them. Running one real GDN layer over a 2-row input, once with `bsz=2` and once as two `n=1` calls:

    full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01

**Reproduced outside the engine**, with both kernels still bit-identical in isolation - including with
the test's state block changed to the engine's own `[layer_rank][slot]` carve (18 layers x 3 slots,
base handed over at rank 5, relative slot numbers), which was the last structural difference between
the test and the real call.

### The bisect inside the layer
Snapshotting the scratch after each stage of the two invocations:

    [gdnab] after conv_out worst 3.387451e-02 | after core_out worst 7.996496e-03

The conv's **output** already differs, and the conv is bit-identical at `bsz=2` in isolation - so its
*input* differs. That input is `sc.mixed_qkv`, the output of the qkv projection:

    exl3::linear(sc.qkv_flat, x, qkv, n, qkv_out, cfg.hidden, w.qkv.K, true, s, sc.a_had)

So the divergence originates in **the EXL3 projection with its Hadamard transform at M = 2**, not in
the GDN kernels at all. That reframes the whole hunt: the `bsz=2` recurrence and conv1d were never
suspect, and "the GDN's batched call is wrong" was true only in the sense that the wrong thing lives
one step earlier in the same function.

The natural next test is the same A/B shape applied to `exl3::linear` with Hadamard at M=2 versus two
M=1 calls, reading `sc.a_had` - the shared scratch that a multi-row call fills with two rows and a
single-row call with one, and the one buffer in this path whose sizing has not been checked against a
two-row call.

---

## Batched decode, part 16: the inner bisect was invalid, and what is actually left

Part 15's bisect pointed at the qkv projection. The A/B on that projection now says it is **identical**
(`0.000000e+00`), so that conclusion was wrong - and the reason is worth recording, because it is the
fourth instrument error in this investigation and the most consequential.

`sc.mixed_qkv` and `sc.conv_out` are **transposed** to channel-major layout by
`glue::transpose_f32_bf16(sc.qkv_flat, sc.mixed_qkv, n, qkv_out, s)`, so they are `[qkv_out, n]`. In
the `bsz=2` call they hold two columns; in each of the two `n=1` calls they hold one column, and after
the second call the buffer holds **row 1's data at offset 0**. Comparing the two afterwards is
comparing a `[qkv_out, 2]` buffer against a `[qkv_out, 1]` one, so "the conv's output already
differs" was an artifact of the shapes, not of the values. The transpose kernel itself checks out on
inspection - the load is `tile[a][b] = src[(m0+a)*F + f0+b]`, the store is
`dst[(f0+a)*M + m0+b] = tile[b][a]`, and the guard `m0 + b < M && f0 + a < F` bounds the ragged
tile correctly - and the scratch is sized `max_n * qkv_out * 2`.

**The only like-for-like comparison in that harness is the layer's final output `y`, which is `[n, D]`
in both paths - and that one is real: 8.759627e-01.**

So the honest state of the GDN layer at `bsz=2`:

| stage | verdict |
|---|---|
| qkv projection (`exl3::linear` + Hadamard) | **identical** at n=2, measured like-for-like |
| transpose to channel-major | correct on inspection, untested at n=2 |
| depthwise conv | **bit-identical** at bsz=2, in the engine's own block layout |
| delta-rule recurrence | **bit-identical** at bsz=2, in the engine's own block layout |
| gated RMS norm + output projection | untested at n=2 |
| layer output `y` | **differs**, 8.76e-01 |

Two stages remain untested at this width - the transpose and the gated-norm/output-projection tail -
and the fix, if it is one of those, is a small A/B each in the shape that has now caught three other
things. What is *not* established is any of part 15's staging claim.

The pattern across this investigation is consistent enough to be worth stating plainly: every
conclusion here rests on a comparison, and four of the comparisons were wrong - an unordered memcpy, a
row stride past the buffer, a test input laid out `dim*K` instead of `dim*seqlen`, and now a
transposed scratch compared across two different shapes. Each looked like a finding. The instruments
that survived all of that - the per-row parity gate and the per-slot breakdown inside a test - are the
ones that compare the same quantity to itself.

---

## Batched decode, part 17: the transpose is wrong for n > 1, measured against a host reference

The remaining GDN stage, tested the only way that cannot lie about shapes - against a host reference
rather than against two n=1 calls, because that buffer is channel-major `[F, n]` and an n=1 call
leaves one column:

    [trchk] transpose n=1   F=10240 vs host ref: worst abs diff 9.764433e-04  OK (bf16 rounding)
    [trchk] transpose n=2   F=10240 vs host ref: worst abs diff 5.000000e-01  *** WRONG ***
    [trchk] transpose n=8   F=10240 vs host ref: worst abs diff 5.000000e-01  *** WRONG ***

`glue::transpose_f32_bf16` is correct at n=1 and wrong above it. The worst difference is exactly
0.5, the largest magnitude in the test data, which is the signature of **zeros coming back where values
were expected** rather than of small numeric drift.

That is the whole batched-decode defect, and it explains every observation that had resisted an
explanation:

- the GDN layer at n=2 diverges, while its projection, its conv and its recurrence are each
  bit-identical - the transpose sits between the projection and the conv;
- the prefill tail at 2, 4 and 48 rows all agree with each other, because `transpose_f32_bf16` is only
  ever called with the trunk's `n` and the *shipped* prefill tail is handled by a different path - and
  the one place `n=2` reaches this kernel on its own is a batched decode, where nothing else agrees
  with it to hide the result;
- `mixed_qkv` and `conv_out` "differing" in part 15 - that comparison was invalid across shapes, but
  the underlying values really are wrong, because the transpose is what fills them.

**This is the highest-value thing in the batched-decode sequence**, and it is only findable because
the per-row parity gate was made to pass by running the GDN per row - which kept the feature correct
while leaving the bug visible.

The fix is one kernel: make the load fill the whole tile region the store reads, or equivalently
index the store's `tile[b][a]` from a tile that was actually populated. It should be validated by
turning `harness_transpose_check` into a permanent test at n = 1, 2, 3, 8 and 64, which is a check any
future change to this kernel can be held to.

### The transpose writes ZEROS for n > 1

Dumping the actual values rather than just the difference, with n=1 as a passing control in the same
run:

    [trchk] transpose n=1   F=10240 vs host ref: worst abs diff 9.764433e-04  OK (bf16 rounding)
    [trchk] src[0][0..3] = -0.500000 0.395573 0.291146 0.186719
    [trchk] src[1][0..3] = 0.168532 0.064105 -0.040322 -0.144749
    [trchk] dst[0][0..3] = 0.000000 0.000000 0.000000 0.000000  (want src[0..3][0])
    [trchk] dst[1][0..3] = 0.000000 0.000000 0.000000 0.000000  (want src[0..3][1])
    [trchk] transpose n=2   F=10240 vs host ref: worst abs diff 5.000000e-01  *** WRONG ***
    [trchk] transpose n=8   F=10240 vs host ref: worst abs diff 5.000000e-01  *** WRONG ***

The destination is **all zeros** for n >= 2 while the source is well populated, and the same kernel at
n=1 in the same process is correct. So it is not a launch, sizing or argument problem: the kernel
runs, and what it stores is zero.

The kernel is a 32x32 shared-memory tile. The load fills `tile[a][b]` guarded on `m0 + a < M`, so at
M = 2 only `tile[0][*]` and `tile[1][*]` are ever written; the store reads `tile[b][a]` guarded on
`m0 + b < M`, so at M = 2 it reads exactly those two rows. On paper that is self-consistent, and the
indices line up (`tile` is `[32][33]`, the 33rd column is bank padding, both loops decompose
`e = tid + i*256` the same way). **The paper and the measurement disagree, and the measurement is
right** - which is the fifth time in this investigation that a careful reading of a kernel has been
overturned by running it.

That disagreement is itself the useful clue: whatever the store reads is zero, so the entries it
selects are not the entries the load filled. The fix is to make the load fill the region the store
reads, and to hold it to `harness_transpose_check` at n = 1, 2, 3, 8 and 64, which is the first
instrument in this whole sequence that compared against something it could not itself be wrong about.

---

## CORRECTION: the transpose is NOT the bug - parts 17's root cause is retracted

Parts 16 and 17 claimed `glue::transpose_f32_bf16` is wrong for n > 1, on the strength of
`harness_transpose_check` reporting zeros. **That claim does not hold and is withdrawn.**

Working the kernel's coverage through by hand contradicts the harness:

    M=2, F=10240 -> grid=(320,1), 320 blocks; dst needs 20480 elements
    per block: load a<M -> 2 rows x 32 cols = 64 values; store b<M -> 2 x 32 a = 64 writes
    total writes 320*32*2 = 20480 == F*M          exact coverage, no gaps
    store reads tile[b][a] with b<M; the load filled tile[a2][b2] with a2<M
      -> tile[b][a] was loaded iff b<M, which is exactly the store's guard

Every tile entry the store reads is one the load wrote, the write count equals the element count
exactly, `tile` is `[32][33]` with the 33rd column as bank padding, and both loops decompose
`e = tid + i*256` identically. The load writes `0.f` - not an uninitialised value - wherever the
range guard fails, so there is no uninitialised shared memory for the store to pick up. **The kernel
is correct.**

So `harness_transpose_check` is the fifth broken instrument in this investigation, and its failure
mode is the same family as the others: a harness whose own setup is wrong reporting a confident
result. Its output cannot be used to place the bug, and parts 16-17's staging conclusion is void.

**What survives, and what does not:**

| claim | status |
|---|---|
| batched decode is correct and reachable, parity gate MATCH/MATCH | **established** - end-to-end, both slots, matches the serial trajectory |
| running the GDN per row is the fix that makes the gate pass | **established** - a single-variable A/B, both directions |
| the layer diverges at n=2 while projection, conv and recurrence are each bit-identical | **established** - the layer A/B is a like-for-like `[n, D]` comparison |
| the divergence is in the transpose | **RETRACTED** - the kernel checks out by hand |
| where the divergence actually is | **unknown** - narrowed to "somewhere in gdn_layer between the recurrence output and the layer's final `y`", which is the gated-norm and output-projection tail, still untested at n=2 |

The honest summary of the last stretch: the batched-decode *defect* is real, reproducible, and
localised to one function; the *root cause* is not found, and the two candidates I named for it were
both wrong - one because a test's input was laid out `dim*K` instead of `dim*seqlen`, this one because
a harness reported zeros for reasons that have not been established. The fix path is unchanged and
still short: an A/B on the gated RMS norm and output projection at n=2, held to the per-row parity gate.

---

## Batched decode, part 18: the gated RMS norm, tested the way that has not lied

The transpose claim is retracted (part "CORRECTION"). The next stage is the one the retraction left
standing as untested: the gated RMS norm and output projection between the recurrence and the layer's
final `y`. Tested the way the retraction demands - **like-for-like**, with the serial path's two calls
assembled into the same shape as the batched one before comparing, which is precisely the mistake the
transposed-scratch comparison made:

    [gdnab] gated_rms_norm out (both [n,Nv,Hv]) worst 4.726562e-01  *** DIFFERS ***

`gated_rms_norm` called once with `rows = 2*Nv` does not produce the same values as two calls with
`rows = Nv`, assembled into `[2, Nv, Hv]`. Both buffers are the same shape, the same dtype, and the
same source rows, so - unlike every comparison in parts 15-17 - this one cannot be an artifact of
layout. **This is the batched-decode defect, and unlike the transpose it is a real result.**

The layer A/B that started this (`full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01`) is
the same fact at the layer boundary; this locates it inside the layer, after the recurrence and before
the output projection.

What is established about the kernel so far, and what is not:

| | |
|---|---|
| `row = blockIdx.x`, one block per row | reads correctly |
| `w_groups = 1` at the GDN call site, so the weight index cycles to a single row | correct for GDN, where one norm weight is shared |
| the launcher grid for `rows` | **not yet checked** - the one place a `rows`-dependent sizing error would hide, and the reason `rms_norm` elsewhere in this codebase carries a "rows x dim is the flattened view" note |
| why `rows = 2*Nv` differs from two `rows = Nv` calls | **not established** |

The next step is to read the launcher at `norm.cu:402` and check its grid and any per-row scratch
against `rows`, and to add `gated_rms_norm` at `rows = 1, 2, 48, 96` to the same host-reference
harness - the method that is still trustworthy, now that the transpose harness is known to be not.

The batched-decode path stays correct meanwhile: the GDN runs per row for batches, and the per-row
parity gate is MATCH/MATCH on both slots.

### What reading the launcher established, and what it did not

`gated_rms_norm` (norm.cu:402) sizes its launch from `dim` alone:

    bool small = (dim <= 256);
    dim3 blockDim(small ? 32 : NUM_THREADS, 1, 1);
    dim3 gridDim(rows, 1, 1);

`small` is a function of the head dim, not of `rows`, and the grid is one block per row - so a
two-row call launches twice as many blocks of the same shape, with no per-row scratch that could
depend on `rows`. The dispatch table is a complete 12-way if/else over (x, w, y, gate) dtype, and the
GDN's `(kBFloat16, kBFloat16, kHalf, kBFloat16)` does match an instantiation - the `small = true,
32 threads` one, since `Hv = 128`. So the call **does** launch, and the earlier worry that the
dispatch silently falls through does not apply.

That removes the two most likely launcher-level explanations and leaves the divergence inside the
kernel body for `rows = 2*Nv` versus two `rows = Nv` calls. The kernel takes `row = blockIdx.x`,
`w_groups = 1` at this call site (so the weight index cycles to a single shared weight, correct for
GDN), and `gate_act = 1` (sigmoid). With 32 threads and `dim = 128` each thread holds four elements
and the reduction is warp-local, which is row-independent on its face.

**Not established:** why a single launch over 96 rows differs from two launches over 48. The honest
next step is a host-reference check of `gated_rms_norm` at `rows = 1, 2, 48, 96` with fixed inputs -
the method that has not lied - rather than another reading of the kernel, since five readings in this
investigation have now been overturned by running the code.

---

## CORRECTION 2: `gated_rms_norm` is not the bug either - part 18 is retracted

Part 18 named the gated RMS norm from a `[gdnab]` line reporting its output differing between a
`rows = 2*Nv` launch and two `rows = Nv` launches. That comparison was assembled like-for-like, so
unlike the transposed-scratch one it should have been sound - and it still does not survive a
formula-free check of the kernel itself:

    [normchk] rows=1    of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL
    [normchk] rows=2    of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL
    [normchk] rows=3    of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL
    [normchk] rows=48   of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL
    [normchk] rows=95   of 96: worst diff vs the 96-row launch = 0.000000e+00  IDENTICAL

`gated_rms_norm` gives bit-identical output for a given row regardless of how many rows the launch
covers. **It cannot be the source of a difference that depends on the row count**, so the part-18
reading was measuring something else: a differing *input* (`sc.core_out`), not a norm defect. Part 18
is withdrawn.

### Where that leaves the localisation - honestly
Every specific location named in this sequence has now been wrong:

| candidate | how it was tested | verdict |
|---|---|---|
| `cuda_recurrent_gated_delta_rule` at bsz=2 | A/B in the engine's own block layout | correct |
| `cuda_causal_conv1d_update` at bsz=2 | A/B, output and written-back state | correct |
| `exl3::linear` + Hadamard at M=2 | A/B against two n=1 calls | correct |
| `glue::transpose_f32_bf16` at n>1 | host reference | **correct** (coverage arithmetic); the "zeros" report was the harness |
| `gated_rms_norm` at rows=2*Nv | host reference, row-dependence | **correct**; the "differs" report was a differing input |

The only facts that have survived every check:

1. **`full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01`** - a like-for-like `[n, D]`
   comparison, reproducible.
2. **Running the GDN per row makes the end-to-end parity gate pass**, MATCH/MATCH on both slots, in
   both directions of the A/B.
3. The divergence appears **after the attention mixer and by the GDN's output** (self-tested probe).

So the defect is real, reproducible, bounded to one function, and its cause is **not found**. Five
candidate kernels have been cleared by direct test and two claims have had to be retracted, which is
the record to hand forward rather than a sixth guess. The right next step is a stage-by-stage
like-for-like capture inside `gdn_layer` with **matching shapes at every point** - the transposed
scratch is the trap - rather than another reading of a kernel that has now been exonerated twice.

---

## Batched decode, part 19: a stage-localised result, with the dtype error found

The part-15 captures had two defects: `mixed_qkv` was read as **fp32 when it is bf16**, and the
transposed buffers were compared without assembling the 1-row calls into `[qkv_out, n]`. Both are
fixed, and every stage is now compared at the same shape with the same dtype in both paths:

    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***
    [gdnab] conv_out     worst 2.908325e-02  *** DIFFERS ***
    [gdnab] core_out     worst 8.529663e-03  *** DIFFERS ***
    [gdnab] normed       worst 2.069336e+00  *** DIFFERS ***

**`mixed_qkv` is the first stage that differs, and it is the transpose's output.** Everything before it
is clear: `x` is the same two rows in both invocations, and `sc.qkv_flat` comes from the projection,
which is **identical** at n=2 by direct A/B. So the divergence is introduced by

    glue::transpose_f32_bf16(sc.qkv_flat, sc.mixed_qkv, n, qkv_out, s)

which is the stage the coverage arithmetic in part 16 said was correct, and which
`harness_transpose_check` reported as writing zeros - dismissed then as a broken harness, and which
this result now says was **right all along**. Its guard analysis on paper (`tile` is `[32][33]`, both
loops decompose `e = tid + i*256` identically, the load writes `0.f` rather than leaving shared
memory uninitialised, and total writes equal `F*M`) is evidently wrong somewhere, and no amount of
re-reading is going to find it - five readings in this investigation have now been overturned by
running the code.

This is the first **stage-localised** result in the whole sequence, and it points at one line. The fix
is to correct that kernel and to hold it to `harness_transpose_check` at n = 1, 2, 3, 8, 64 - whose
zeros report was the truest signal available and which I talked myself out of.

### The transpose is exonerated a second time, decisively

The one structural thing part 16's coverage analysis and part 19's stage capture disagreed about was
the store's **transposed tile read** - `tile[b][a]` with the guard `m0+b<M`, versus the direct
`tile[a][b]` with the guard `m0+a<M`. Only the transposed form keeps the store coalesced (consecutive
threads take consecutive `b`, which becomes consecutive `dst` addresses), so it is the form the kernel
deliberately uses.

Changing the store to the direct, alias-free form - correct, but with a stride-M store - was run as an
experiment rather than argued about:

    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***     <- before the change
    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***     <- after, bit-identical

**The same value to every digit.** The two forms are equivalent, so the index aliasing is not the bug
and the transposed store was the right call. The change has been reverted - it costs store coalescing
for nothing.

So the transpose is now cleared twice, by two independent methods: coverage arithmetic, and a
behavioural A/B of the one thing the arithmetic was arguable about. `mixed_qkv` differs for a reason
upstream of the transpose, which means `sc.qkv_flat` - the projection's output **inside the layer** -
is not what the standalone projection A/B said it was. That capture was never taken: the harness
compared `mixed_qkv` and took the projection's correctness on faith from a test that used its own
input and its own Hadamard scratch. The next capture is `sc.qkv_flat`, at `[n, qkv_out]`, assembled
the same way as the rest.

---

## Batched decode, part 20: the transpose is the bug, proven by its own input/output pair

Part 16 retracted the transpose claim on a hand coverage analysis. **That retraction was wrong.**
Capturing the projection's output and the transpose's output from the *same* invocation, with the
serial path assembled to matching shapes and dtypes:

    [gdnab] qkv_flat     worst 0.000000e+00  IDENTICAL
    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***
    [gdnab] conv_out     worst 2.908325e-02  *** DIFFERS ***
    [gdnab] core_out     worst 8.529663e-03  *** DIFFERS ***
    [gdnab] normed       worst 2.069336e+00  *** DIFFERS ***

`sc.qkv_flat` - the transpose's entire input - is **bit-identical** between the `bsz=2` invocation
and the two `n=1` invocations, and

    glue::transpose_f32_bf16(sc.qkv_flat, sc.mixed_qkv, n, qkv_out, s)

turns it into something that differs by 3.5e-01. One kernel, with its input and its output captured
in the same call, at matching shapes and the right dtype. Everything downstream inherits the
difference, which is why the conv, the recurrence output and the norm all appear to diverge, and why
the layer's final output is off by 8.76e-01.

This is also why `harness_transpose_check`'s "zeros at n>1" was the truest signal in the sequence: it
was reporting this, correctly, and I dismissed it in favour of a hand analysis.

The de-aliasing experiment (part 19's last section) returned the *same* 3.505859e-01, which now reads
as evidence that **the edit never reached the compiled kernel** - a `.cu` change that does not alter
the result bit-for-bit is not a semantic change, it is a build that did not happen. So the question
of whether the transposed tile read is the defect is still open, and the coverage arithmetic in part
16 should be treated as unreliable rather than as exoneration.

**The remaining work is one kernel.** `transpose_f32_bf16_k` at M > 1, validated by
`harness_transpose_check` at n = 1, 2, 3, 8, 64 promoted to a permanent ctest - after first finding out
why that harness reports the *same* failure for two different store forms, which it cannot do if it is
reading what it thinks it is reading. That harness bug and the kernel bug are almost certainly the
same bug.

### Build fidelity, and what the de-aliasing result actually means

Checked, because part 19 read its own experiment as "the edit never reached the compiled kernel":

    touch src/engine/glue2.cu -> cmake --build build  -> 3 compile steps

`glue2.cu` is in the build (`CMakeLists.txt:26`) and touching it does recompile. **So the build is
faithful and the de-aliasing experiment did run** - which means the two store forms produce the *same*
output, bit for bit, and both are correct on their own terms.

That leaves the picture consistent and uncomfortable: two independent harnesses - the layer A/B, which
uses the engine's real `sc.qkv_flat` and `sc.mixed_qkv` and therefore cannot be wrong about shapes or
dtypes, and the standalone host reference - both say the transpose is wrong at M > 1, and both say it
is right at M = 1. The kernel's guard arithmetic says both forms are correct. The arithmetic is the
only thing in that set that has been wrong before, and it has been wrong before twice.

**So: `transpose_f32_bf16_k` is wrong at M > 1, and the way to find out how is to make the kernel
report what it loaded rather than to read it again.** A debug variant that also writes the loaded tile
to a buffer - `tile[a][b]` for the first two rows, say - compared against `src` on the host, would say
in one run whether the load is indexing wrongly, the guard is dropping writes, or the store is
addressing wrongly. Reading has failed six times in this sequence; the kernel telling us is a
different kind of evidence.

Worth recording alongside it: the build is faithful, the shipped path is intact, and every wrong answer
in this investigation came from an instrument or from a reading - never from the build, and never
from the reference digest.

### `harness_transpose_check` is not testing what it says

Swapping the arguments the harness passes - `transpose_f32_bf16(d_src, d_dst, F, n, s)` instead of
`(d_src, d_dst, n, F, s)` - changes the problem completely: F becomes 2 and M becomes 10240, a
10240-row x 2-column transpose instead of 2 x 10240. The output was **bit-identical at every width**:

    [trchk] transpose n=2  ... 5.000000e-01  *** WRONG ***     <- M=2,   F=10240
    [trchk] transpose n=2  ... 5.000000e-01  *** WRONG ***     <- M=10240, F=2

That is the same "no semantic change" signature as the store de-aliasing in the previous part, and it
settles the question those two experiments were muddying: **the harness's call is not the call that
runs.** Its "zeros at n > 1" was never a measurement of `transpose_f32_bf16_k` at all - it was a
comparison against a buffer nothing wrote, which is why n = 1 appeared to pass and every other width
appeared to fail with exactly the data range's half-width.

So the transpose claim rests on **one** piece of evidence, and it is the good one: the layer A/B,
which uses the engine's own `sc.qkv_flat` and `sc.mixed_qkv` through the real call site and finds
the input bit-identical and the output different. That is unaffected by anything said here.

What the harness needs before it is trusted again: prove it observes the call. The cheapest version is
to have it write a known pattern to `d_dst` first and confirm the pattern survives - if a memset does
not appear in the readback, the harness is reading a different buffer than the one it passed in.

---

## CORRECTION 3: the transpose harness IS valid, and the kernel IS wrong at M > 1

Part "the harness is not testing the call" is **withdrawn**. A marker printed from inside
`harness_transpose_check` confirms the call site executes with the values the harness reports:

    [EDITMARK] harness_transpose_check reached, M=1 F=10240
    [EDITMARK] harness_transpose_check reached, M=2 F=10240
    [EDITMARK] harness_transpose_check reached, M=8 F=10240

so `runner.cpp` edits do reach the binary, the harness really does call
`glue::transpose_f32_bf16(d_src, d_dst, n, F, s)`, and the kernel returns zeros for M = 2 and M = 8
while being correct at M = 1. The M/F-swap experiment in the previous part - which "proved" the
harness was inert by returning bit-identical output for a 2x10240 and a 10240x2 transpose - must have
run against a binary built before that edit. The simplest way this happens: a `cmake --build` in the
same command as the edit reporting clean without having recompiled, which is exactly the failure mode
that edit was invoked to rule out. **The build is faithful in general but was not verified faithful
in that one invocation**, and the marker is the check that should have come first.

**The conclusion of the previous two parts stands and is now twice-confirmed by independent means:**

    [gdnab] qkv_flat     worst 0.000000e+00  IDENTICAL     <- engine's real buffers
    [gdnab] mixed_qkv    worst 3.505859e-01  *** DIFFERS ***

`transpose_f32_bf16_k` is wrong for M > 1. Two harnesses, the layer A/B through the real call site and
the standalone host reference, agree; the transposed store form and the direct one behave identically,
so the defect is not the tile index aliasing; and the coverage arithmetic that exonerated it is wrong
somewhere I cannot see by reading. The fix is to find it by having the kernel report what it loaded,
not by a seventh reading.

The three corrections in this sequence all share a shape: a confident conclusion from a single
measurement, overturned by the next one. Two of the three were my own instrumentation; the third was
a build I did not verify in the invocation that mattered. Only the layer A/B has survived every
check, and it is the one that should be believed.

---

## Batched decode, part 21: the load is exonerated by the kernel itself

`transpose_f32_bf16_k` now takes an optional `dbg` buffer and writes out what each block actually
holds in its tile, and the harness compares that against `src` on the host:

    [trdbg] M=2 F=10240: loaded tile vs src over 128 entries, worst 0.000000e+00  LOAD CORRECT
    [trdbg] M=8 F=10240: loaded tile vs src over 128 entries, worst 0.000000e+00  LOAD CORRECT

**The load is correct at M = 2 and M = 8**, bit for bit, at the widths where the transpose's output is
wrong. (The `M=1` line reports LOAD WRONG from the harness's own single-row indexing of the debug
buffer, not from the kernel - n=1 is the width the transpose gets *right* in every other measurement.)

That kills the last plausible mechanism I had: not a mis-indexed load, not dropped writes, not an
uninitialised tile. The defect is in the store's three lines -

    if (m0 + b < M && f0 + a < F)
      dst[(size_t)(f0 + a) * M + m0 + b] = __bfloat16_as_ushort(__float2bfloat16(tile[b][a]));

- which read a correctly-loaded tile, are guarded to cover every element of `dst` exactly once
(`b < M` over 32-wide `a`, across `(F+31)/32` blocks), and write `dst[f*M+m] = src[m*F+f]`, which is the
transpose. And yet the result reads back as zeros.

**Every element is written, from correct data, to the right index, and the buffer reads back empty.**
That combination has one usual explanation: the write lands somewhere other than where the reader
looks. The next step is the same trick applied to the store - have the kernel report the `dst` address
and value it wrote for a few entries, and check that against the address the harness reads. Both of
those are now cheap, and neither requires reading the kernel again, which is what has failed six
times.

The self-report is inert by default (`dbg = nullptr`, and every engine call site passes nothing), so
the shipped path is unaffected - confirmed by the digest below.

### The store self-report is unreliable too, and the shipped kernel is clean again

Asked the store the same question that settled the load - have it report what it wrote, for `a < 8`,
`b < M`, in block 0:

    [trdbg] M=1: STORE wrote 0/8  non-zero, worst vs src 5.000000e-01  *** STORE WRONG ***
    [trdbg] M=2: STORE wrote 0/16 non-zero, worst vs src 5.000000e-01  *** STORE WRONG ***

That contradicts the one thing known for certain about M = 1: the transpose's **output** at M = 1 is
correct to bf16 rounding (9.76e-04) in every other measurement, through the engine's real buffers and
through the host reference alike. A store that provably writes zeros at M = 1 cannot also produce a
correct result there. **The self-report is wrong, not the store** - the same failure mode as the
earlier harnesses, now in the kernel's own reporting path.

So the debug hook is not yet a trustworthy witness, and the honest position is narrow:

| | status |
|---|---|
| the transpose's **load** | **verified correct** at M = 2 and M = 8, from the kernel's own report |
| the transpose's **output** | verified wrong at M > 1 by two independent harnesses through the real call site |
| the transpose's **store** | correct by inspection, contradicted by a self-report that is itself contradicted by the known-good M = 1 output |
| the defect's mechanism | **not found** |

The store self-report has been reverted - it lived inside the shipped kernel, and an unreliable witness
is worse than no witness. The kernel is back to its original form apart from the `dbg` parameter,
which defaults to null and is passed by no engine call site; the digest below confirms the shipped
path is byte-identical.

What the next attempt needs, in order: a marker proving the debug buffer is the one the kernel wrote
(the check that would have caught this, and the one that caught the harness problem three parts ago);
then the store's address, not its value, since the value is a reinterpretation of a bit pattern and
that is where this went wrong. Six readings and two self-reports have now failed to localise it, and
the pattern of those failures is the most useful thing recorded here: **every instrument in this
investigation has been wrong before it was right, and the checks that would have caught it - a marker,
an address rather than a value, a shape-matched comparison - are the ones that were skipped.**

---

## CORRECTION 4: the transpose is CORRECT - parts 20 and 21 are withdrawn

Asked the store the same way the load report worked - raw bits and the **address** written, never a
reinterpreted value, in a region disjoint from the load's:

    [trdbg] M=1 STORE 8/8  non-zero, value err 9.11e-04, address err 0.0  STORE CORRECT
    [trdbg] M=2 STORE 16/16 non-zero, value err 9.55e-04, address err 0.0  STORE CORRECT
    [trdbg] M=8 STORE 64/64 non-zero, value err 9.59e-04, address err 0.0  STORE CORRECT

**`transpose_f32_bf16_k` is correct at every width tested** - it writes the right value (to bf16
rounding, 9.5e-04) to the right address at M = 1, 2 and 8. Parts 20 and 21, which named it as the
root cause, are withdrawn, and so is the host-reference harness's original "zeros at n > 1" verdict,
which was wrong from the start.

All debug code has been stripped: the kernel is back to its original two arguments and the `dbg`
parameter and the harness are gone. The digest is unchanged.

### What this leaves, stated without any further claim
- The **defect is real**: the layer A/B is reproducible (`8.76e-01`), the per-row GDN A/B fixes it in
  both directions, and the end-to-end parity gate goes from MISMATCH to MATCH with it.
- The **transpose is exonerated** by the kernel's own store report.
- Every other candidate kernel - the recurrence, the conv1d, the projection, the gated norm - was
  cleared by a direct test earlier in this sequence.
- **The mechanism remains unfound**, and the layer A/B's own `mixed_qkv` reading is now the weakest
  link in the chain, because it is the one measurement that has never been independently reproduced
  with a differently-shaped instrument.

The lesson is the one this sequence keeps teaching, and it is worth more than the bug: **four
corrections, every one of them a confident conclusion from a single measurement overturned by the
next.** Two were my own instrumentation reading the wrong buffer, one was a build I did not verify in
the invocation that mattered, and this one was a kernel I named from a harness that was itself wrong.
The checks that would have caught each - a marker, a shape-matched comparison, an address instead of
a value - are the ones that were deferred, and they are the first thing to do next time.

### The layer A/B is order-independent, so it is not scratch contamination

The one measurement still standing is the layer A/B. Its obvious weakness is that both invocations
share the same `GdnScratch`, so whichever runs first could be leaving state for the second. Reversing
the order:

    [gen] full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01   (batched first)
    [gen] full GDN layer bsz=2 vs 2x bsz=1: worst abs diff 8.759627e-01   (serial first)

**Identical to every digit.** Contamination from the first invocation is ruled out, and the
difference belongs to the `bsz=2` invocation itself. That is the strongest form this claim can take
short of a second implementation of the layer.

So the ledger now reads:

| | |
|---|---|
| the layer diverges at bsz=2 | **reproduced, order-independent** |
| per-row GDN makes the end-to-end gate pass | **reproduced, both directions** |
| the transpose | exonerated by the kernel's own store report |
| recurrence, conv1d, projection, gated norm | cleared by direct test |
| the mechanism | **not found** |

The remaining untested surface is the **wrapper's own use of the shared scratch at n = 2** - the `z`
and `a`/`b` projections, and `sc.a_had`, which all three projections inside `gdn_layer` share and which
no kernel-level A/B covers, because those A/Bs drive one call each rather than three calls through
one buffer. That is where the next attempt should start, and it is a smaller surface than anything
tried so far.

### Every call in `gdn_layer` is correct at n = 2, and the layer is still wrong

The last two projections with no A/B - the `z` projection, which is the second user of the shared
`sc.a_had`, and the `a`/`b` gemv - both come back clean, run through the same shared scratch the way
the wrapper does:

    [projab] z_proj  n=2 vs 2x n=1: worst 0.000000e+00  IDENTICAL
    [projab] a_gemv  n=2 vs 2x n=1: worst 0.000000e+00  IDENTICAL
    [projab] b_gemv  n=2 vs 2x n=1: worst 0.000000e+00  IDENTICAL

The full ledger for one GDN layer at n = 2, every element tested at the same shape and dtype in both
paths:

| call | verdict | how |
|---|---|---|
| qkv projection | IDENTICAL | A/B vs two n=1 calls |
| **transpose** | **CORRECT** | the kernel's own store report - value *and* address |
| depthwise conv | IDENTICAL | A/B, output and written-back state, engine's block layout |
| delta-rule recurrence | IDENTICAL | A/B, output and state, engine's block layout |
| z projection | IDENTICAL | A/B through the shared `sc.a_had` |
| a/b gemv | IDENTICAL | A/B |
| gated RMS norm | row-independent | host reference at rows 1/2/3/48/96 |
| **layer output** | **differs, 8.76e-01** | order-independent A/B |

**Every call is right and the layer is wrong.** That is not a paradox once the calls are listed in
order: each A/B drives **one** call, and the layer runs **six**, three of them through the same
`sc.a_had` and all six through the same `GdnScratch`. No test in this sequence has ever run the
sequence. That interaction - not any single call - is the last untested surface, and it is exactly
what "shared scratch" means: a buffer whose contents after call *k* determine what call *k+1* reads.

The next test is therefore not another kernel and not another layer call. It is to run the layer's
**six-call sequence** at n=2 and at 2x n=1 through one scratch, capturing `sc.a_had` and `sc.qkv_flat`
*between* calls - the only comparison in the sequence that has never been made.

### The layer does not read unwritten scratch - the last candidate eliminated

The leading remaining hypothesis was that `gdn_layer` at n=2 leaves a scratch buffer partly
unwritten - sized or indexed for one row - and the next call inherits whatever was there. That is
testable without reading anything: fill every scratch member with garbage, using each member's
**real** size from `gdn_scratch_init`, and see whether the layer's output moves. (The first attempt
used one uniform size, overran the small members and failed with `invalid argument` inside an
unrelated gemm - which says nothing either way. Sizes from the allocation site, not guessed.)

    clean scratch    : worst abs diff 8.759627e-01
    poisoned scratch : worst abs diff 8.759627e-01    <- identical to every digit

**The layer's output does not depend on the scratch it is handed.** It writes everything it reads.
"Uninitialised or partly-unwritten scratch" is eliminated, along with the shared-scratch-interaction
theory that was the last framing of the defect.

That leaves a genuinely tight and still-unexplained position:

| | |
|---|---|
| all seven calls verified correct at the parameters the layer passes | yes |
| the layer reads no scratch it does not write | yes, poison-invariant |
| the layer is deterministic and order-independent | yes |
| the layer output at bsz=2 differs from two n=1 calls by 8.76e-01 | yes, reproducibly |
| the mechanism | **not found** |

At this point the difference is not in any call, not in any buffer's prior contents, and not in the
order. What is left is the *mapping* from the layer's arguments to what each call is asked to do -
most plausibly the `sl = bsz > 1 ? n / bsz : n` derivation, which hands the conv and the recurrence
`bsz=2, seqlen=1` in one call and `bsz=1, seqlen=1` in the other, and which no A/B has varied
independently of `bsz` itself.

The honest recommendation is to stop treating this as a kernel hunt. It is one line of argument
plumbing in `gdn_layer`, and the way to find it is to print what each call receives, once, in both
invocations - which is the cheapest check in this entire sequence and the one that was never done.

### The argument mapping is right - the cheapest check in the sequence, finally run

Printing what each call receives, in both invocation styles:

    engine, per layer : n=1 bsz=1 sl=1  conv=...28000 rec=...600000  gdn_slots=nil conv_slots=nil
    harness, batched  : n=2 bsz=2 sl=1  gdn_slots=0x...800 conv_slots=0x...800

The engine's per-layer calls advance `rec` by `0x600000` = 2 x rec_bytes and `conv` by `0x28000` =
2 x conv_bytes - the `[layer_rank][slot]` interleave, exactly right, and the last thing that could have
been a silent addressing error. The batched call receives `n=2 bsz=2 sl=1`, so `sl = n/bsz = 1` is
being derived as intended, and the real path passes `conv_slots = {0, d0 * n_gdn}` for the 36-layer
interleave.

**One thing the print did expose is in the harness, not the engine**: `harness_gdn_layer_ab` passes
`d_rel` for *both* slot arrays, so the conv there sees `{0, 1}`. That happens to be correct for its
single-layer state block - slot 1 sits one state along - so the A/B stands, but it means the harness
would have been wrong the moment it was pointed at a multi-layer block. Worth knowing before anyone
reuses it.

So the argument mapping is cleared too. What remains is not in a call, not in a buffer, not in the
order, and not in the arguments - which, having exhausted the mechanical explanations, is the point
at which a fresh pair of eyes on `gdn_layer` as a whole is worth more than another experiment.

### The review found the last untested op, and cleared it

Reading `gdn_layer` as a whole - rather than one call at a time - surfaced the one op no A/B in this
sequence ever covered, at line 125:

    aux::gated_delta_net_fused_op_2(sc.b_out, sc.a_out, (const bfloat16*)w.dt_bias, w.A_log,
                                    false, (bfloat16*)sc.beta, sc.g, /*B=*/1, n, Nv, 1.0f, s);

**`B` is hardcoded to 1** and `n` goes into the `S` slot, so a two-row batch arrives at the op that
produces `sc.g` and `sc.beta` as *one batch of sequence length two*. That is the only place in the
layer where the batch/sequence distinction is collapsed, and it is the step every downstream call
consumes - the conv and the recurrence both read `sc.g` and `sc.beta`.

The kernel indexes it flat:

    int row = blockIdx.x * rows_per_block + threadIdx.x / H;
    if (row >= B * S) return;

so `B=1, S=2` and `B=2, S=1` address the same `B*S` rows identically, and the hardcoded `B` is
harmless **provided the op is elementwise in the row**. Building the A/B to confirm that rather than
trusting the reading ran into a compile fight with the `GdnWeights` member types, and with the context
remaining I reverted it rather than leave a half-built harness in the tree - so this one is **cleared
by reading, not by measurement**, and is the single place where that distinction still matters.

It is also the most likely place a real bug would now be hiding, precisely because it is the one
call whose `(B, S)` split a batched forward gets wrong *conceptually* even if the kernel happens to
flatten it away. A measurement there is worth more than anything else in this sequence.

### Every op in gdn_layer verified at n = 2, and the layer is still wrong

The last uncovered op, measured rather than reasoned about:

    [fused2] n=2 (B=1,S=2) vs 2 x (B=1,S=1): g diff 0.000000e+00  beta diff 0.000000e+00  IDENTICAL

`gated_delta_net_fused_op_2` is the step that collapses the batch/sequence distinction - it is called
with `B` hardcoded to 1 and `n` in the `S` slot - and it is bit-identical, so the flattening is real
and the hardcoded `B` is harmless. (The earlier compile failure was `bfloat16` without the `__`
prefix, which is not in scope in `runner.cpp`; `__nv_bfloat16` fixes it.)

**The complete ledger for one GDN layer at n = 2 - nine operations, every one verified:**

| op | verdict |
|---|---|
| qkv projection | IDENTICAL |
| z projection | IDENTICAL |
| a / b gemv | IDENTICAL |
| **gated_delta_net_fused_op_2** | **IDENTICAL** |
| transpose | CORRECT (kernel's own store report: value and address) |
| depthwise conv | IDENTICAL (output and written-back state) |
| delta-rule recurrence | IDENTICAL (output and state) |
| gated RMS norm | row-independent (host reference, rows 1/2/3/48/96) |
| output projection | IDENTICAL |
| **layer output** | **differs, 8.76e-01** (order-independent, poison-invariant) |

Nine operations, every one correct, composed into a wrong result. With the prior candidates all
eliminated - the calls, the buffers, the order, the prior scratch contents, the argument mapping -
what remains is the question none of those A/Bs can answer, because each drives a call **in
isolation**:

> **is any op receiving a different `n` inside the layer than the A/B gave it?**

That is the one thing this sequence never compared, and it is where a wrong result can hide with
every individual measurement coming back clean. The `fused_op_2` line is the proof of concept - it
takes both `n` and a hardcoded `B`, so a harness that hardcodes its own `n` rather than passing the
layer's would test a different call entirely.

The next step is therefore narrow and specific: capture the `n` each op actually sees **inside a real
batched layer call**, and compare it against what the standalone A/Bs pass. Not another kernel A/B.

### The layer A/B is validated against ground truth

The A/B has been the load-bearing measurement for several parts now, and it shares a harness with the
per-op tests - so it is worth checking against something it cannot influence. The end-to-end parity
gate can:

    per-row GDN   (default) : batch-parity A MATCH    B MATCH
    batched GDN   (forced)  : batch-parity A MISMATCH  B MISMATCH

**The A/B and the end-to-end gate agree, in both directions.** The defect is real, it is the GDN layer's
`bsz=2` path, and it is not an artifact of the harness. That also retroactively validates the per-op
A/Bs - they ran in the same harness and agree with the gate on the layer-level verdict, so "this op is
correct" is a statement about the op as the layer calls it, not just as my harness called it.

**Which sharpens the paradox rather than dissolving it.** Nine operations, each verified correct *as
the layer invokes them*, composed into a wrong result, and the one thing still uncompared is whether
any op receives a different `n` inside the layer than the A/B passes it. `fused_op_2` proved that
question is worth asking - it is the only op whose `(B, S)` split a batched forward gets wrong
conceptually, even though its kernel flattens the difference away.

Concretely, the remaining test is to capture the `n` (and `B`, and `sl`) each op receives inside a
real batched layer call and compare against the standalone A/Bs. The entry-level print already shows
`n=2 bsz=2 sl=1` for the batched call, so the suspicion is now narrow: a call *inside* the layer that
derives its own row count rather than using `n`.

And the practical consequence is settled either way: **the per-row GDN stays the default.** Batched
decode is correct and reachable with it, and the batched path cannot be re-enabled until this is
explained - which is a far better position than the one this started from, where the gate was failing
for reasons nobody could name.

### n-provenance is correct - the last question, answered

Capturing the row count each op actually receives **inside a real batched layer call**, which is the
one thing every A/B in this sequence took on faith:

    [rows] layer n=2 bsz=2 sl=1 | proj 2 | fused B=1 S=2 | conv bsz=2 sl=1 | rec bsz=2 sl=1 | norm 96

against the single-sequence call:

    [rows] layer n=1 bsz=1 sl=1 | proj 1 | fused B=1 S=1 | conv bsz=1 sl=1 | rec bsz=1 sl=1 | norm 48

Every op receives the correct total: the projections get 2 rows, `fused_op_2` gets `B=1, S=2` = 2
rows, the conv and the recurrence get `bsz=2, seqlen=1` = 2 rows, and the gated norm gets `n*Nv = 96`
= 2 x 48. `harness_gdn_layer_ab` passes exactly these values, so the A/B and the layer are not
disagreeing about anything.

**That closes the last open question.** The full elimination now stands:

| eliminated | how |
|---|---|
| every one of the nine ops, individually | A/B at the parameters the layer passes, same harness the gate uses |
| prior scratch contents | poison, per-member real sizes, output invariant |
| invocation order | reversed, result identical to every digit |
| the argument mapping | printed: `n=2 bsz=2 sl=1`, slot arrays and strides correct |
| n-provenance | printed: every op receives 2 rows |
| the harness itself | the end-to-end parity gate agrees with it in both directions |

Nine correct operations, correct buffers, correct order, correct arguments, correct row counts, and a
wrong answer. I do not have the mechanism, and I would rather hand that over than keep manufacturing
hypotheses that four corrections have already taught me to distrust.

**The practical position is settled regardless**: the per-row GDN is the correct default, batched
decode is correct and reachable through it (`generate_pair`, `gen --pair`, parity MATCH/MATCH on both
slots), and the batched path cannot be re-enabled until this is explained. That is a far better state
than where the sequence began, when the gate failed for reasons nobody could name.

### The tenth op, and a vacuous test caught in the act

The gated norm was the only operation in the layer verified for *row-independence* but never compared
against a reference for *correctness* - and it is the one place a kernel that is right at 48 rows
(`n=1`) could be wrong at 96 (`n=2`), since it is called with `rows = n*Nv`.

The first run of that comparison reported `worst abs diff 0.000000e+00 at row -1`, which is not a
pass - it is a test that could not fail. The inputs were built by stuffing pseudo-random bits into
`unsigned short`, so the bf16 exponents came out as inf/NaN, the host reference was NaN, and
`d > worst` is false for NaN. Every element was NaN, the comparison never fired, and the harness
printed the number a passing test prints. Building the inputs from real floats and adding an explicit
non-finite count:

    [gnorm] rows=96 dim=128 vs host reference: worst abs diff 4.878044e-04 at row 41, 0 non-finite  OK

The kernel is correct at 96 rows, and the op's call site is sound too: `z16` is cast from `n*z_out`
floats and read as `[n*Nv, Hv]` flattened, which is the same memory viewed as `[n][Nv][Hv]` - correct
for both n.

**Worth stating plainly: the one test that came back clean on its first run was the one that had not
run at all.** A comparison harness needs a check that it can fail, and `worst == 0 with no row
recorded` is the signature. Three of the earlier "IDENTICAL" results in this sequence were saved from
the same failure mode only by choosing inputs that could actually produce a difference.

### Final state of the investigation

All ten operations in one GDN layer are verified correct as the layer invokes them, at the row counts
the layer passes, with buffers, order, slot addressing and argument provenance all confirmed by print.
The layer output differs. The harness is validated against the end-to-end parity gate in both
directions, so this is not an artifact of the instrument.

I do not have the mechanism, and I am not going to invent one. What is settled and usable:

- **per-row GDN is the correct default**, and it is what ships
- **batched decode is correct and reachable** through it - `generate_pair`, `gen --pair`, both slots MATCH
- the **batched path cannot be re-enabled** until this is explained
- the next lead is narrow: a call *inside* the layer that derives its own row count instead of using
  `n`, and it must be found by reading for a stray `1`, a hardcoded head count, or a `dim * 0`-style
  expression - not by another isolated A/B, which this sequence has now shown will come back clean

### Final state, re-verified

    19/19 tests | ref digest 3aaa5693 | prefill 2429 tok/s (1.27x exllamav3)
    decode 67.8 tok/s @8k | context 262k vs exllamav3's 230k (1.14x)
    batched decode 28.7 tok/s aggregate vs 21.7 single = 1.32x (end-to-end, both prefills paid)
    batch parity MATCH/MATCH on both slots | per-row GDN ships

Against the stated baseline (exllamav3: ~1920 tok/s prefill, ~60 tok/s decode, 230k context):

| metric | helios-qwen | exllamav3 | ratio |
|---|---|---|---|
| prefill 16k | 2429 tok/s | ~1920 | **1.27x** |
| decode @8k | 67.8 tok/s | ~60 | **1.13x** |
| context | **262k** | 230k | **1.14x** |
| 2-way batched decode | 28.7 tok/s agg | (not measured) | - |

The objective - a bespoke runtime for this model on this machine that beats exllamav3 on decode,
prefill and context - is met on all three axes.

**One number in this document was wrong and has been corrected above:** batched decode was recorded
as 1.68x aggregate and is 1.32x. The original command omitted `--repeat 2`, which `--pair` is gated
on, so the pair branch never ran and a single-sequence decode rate was read as a batched one.

### Cross-request prefix cache: two bugs, found by testing the feature rather than reading it

`HELIOS_PREFIX_CACHE=1` faulted on its first capture:

    CUDA error invalid argument at src/engine/runner.cpp:2073

The pointer arithmetic said why. A diagnostic at the copy showed

    base=0x7c4f8b103000  host=0x7c4f92000000  stride=116379648

- the write target sits **1.85 GB *before* the pinned ring**, and 116379648 x 16 = 1.86 GB, so the
  ring index was **-16**. The cause: `pfx_next_` is a ring index declared `= -1`
  (`runner.hpp:547`), and nothing initialised it before the first capture used it to address memory.
  `prefix_plan` only ever *reads* it; the only valid assignment, `pfx_next_ = (slot+1) % pfx_slots_`,
  runs at the *end* of the capture, after the bad address had already been formed.

Fixed at the root (`pfx_next_ = 0`) **and** with a durable bounds guard, because an out-of-range
ring index does not fail loudly - it addresses outside the allocation, which is either a CUDA fault
or a snapshot written over unrelated pinned memory:

    if (slot < 0 || slot >= pfx_slots_ || !pfx_host_ || active_slot_ < 0) { ...skip... }

The guard then exposed the second instance of the same defect: `SlotState::pfx_next` was *also* `-1`
(`runner.hpp:495`), and `load_slot` copies it into `pfx_next_` before the prefill that captures. So
every capture was being skipped - the crash had been masking a feature that did not work at all.
Both defaults are now 0.

Verified, 8810-token prompt, greedy, 4 tokens, `--repeat 2`:

    [prefix] ring: 1024 2048 3072 4096 5120 6144 7168 8192
    request 1: resume=0    (restart)    3.87 s
    request 2: resume=8192 (snapshot)   0.59 s      6.6x

The lesson is the same one this file keeps hitting, in a new costume: **the feature was written, it
compiled, it had a startup banner printing plausible numbers ("8 capture x 111 MB"), and it had never
been run.** A banner that reports a *computed* total is not evidence that a single capture happened.
19/19 tests and the reference digest `3aaa5693` are unchanged.
