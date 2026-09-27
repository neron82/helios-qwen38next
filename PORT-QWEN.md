# helios-qwen: bespoke runtime for Qwen3.8-Flash-Next-exl3

Port of the helios engine (GLM-5.3-Flash, 2x RTX 3090 + 128 GB DDR4) to this checkpoint, targeting
the exllamav3 baseline (`~/models/qwen38_next_server.sh`). Local `config.json` and the checkpoint's
own `chat_template.jinja` are authoritative; exllamav3 is the correctness oracle and the speed bar.

## Baseline to beat (measured, greedy, MTP as configured)

| workload | prefill | decode |
|---|---|---|
| 815-token prompt | 738 tok/s | 85.3 tok/s (MTP on) / 75.2 (off) |
| 11.8k prompt | 1,644 tok/s | 83.4 (on) / 71.2 (off) |
| 47k prompt | 1,730 tok/s | 86.5 (on) |
| 146.7k prompt | 1,666 tok/s | 73.4 (on) |
| MTP value | — | **+13.4 % at 1k, +17.1 % at 16k** |

Prefill is flat with depth (1.64-1.73k tok/s from 12k to 147k), so depth does not strain prefill
throughput; decode loses ~15 % by 147k (86.5 -> 73.4), which is the attention/indexer cost at depth
and a concrete target for the port.

Context ceiling: 229,888 tokens, and the launcher is explicit that this is a *VRAM* limit — "a full
262144-token KV reservation does not fit next to the ~36 GB of weights on 2x24 GB VRAM". Prefill is
flat with depth (~1.7k tok/s), so depth is not the prefill problem; allocation is.

## Where this checkpoint differs from GLM-5.3 (the port surface)

Measured category split (from the safetensors headers of the 5 shards):

| category | GB | tensors |
|---|---|---|
| routed experts (48 layers x 512) | 31.31 | 301,056 |
| hyper-connections | 1.31 | 395 |
| embedding (bf16) | 1.27 | 1 |
| GDN (36 layers) | 1.06 | 648 |
| vision (out of scope) | 0.45 | 987 |
| full attention (12 layers) | 0.33 | 312 |
| lm_head (4-bit) | 0.32 | 4 |
| MoE router | 0.13 | 49 |
| shared experts | 0.12 | 637 |
| PLE layer | 0.07 | 6 |
| **total** | **36.37** | **304,105** |

plus `ngram_embedding.safetensors`: 26.24 GB, 132 tensors - 128 shards of `trellis [2500012, 41]`
I16, one F16 and three I64. 41 words x 16 bits = 656 bits, i.e. ~320 weights at 2.05 bpw = exactly
`hc_lowrank` 320, and `heads_per_ngram` 8 x 320 = 2560 = `ple_embed_dim`. So each token gathers 8
low-rank vectors (one per n-gram head) and the table is *large* but the per-token traffic is ~0.7 KB
- a gather, not a bandwidth problem.

### 262,144-token VRAM budget (the context lever)

| component | GB |
|---|---|
| text weights (excl. vision, excl. n-gram table) | 35.92 |
| KV fp16 (12 layers x 2 kv heads x 256 dim x 2 B x 2 for K+V) | 6.44 |
| KV int8 equivalent | 3.22 |
| indexer planes (compress 4) | 0.10 |
| GDN recurrent + conv states | 0.11 |
| **total with fp16 KV** | **42.57** |
| **total with int8 KV** | **40.94** |

So the *native* 262,144 context fits with fp16 KV and ~5 GB to spare, before any quantization. The
baseline's 229,888 ceiling is not a KV-size limit in the abstract - it is what is left after the
vision tower, its KV, the MTP draft buffers and ev3's reservations on a 22+22 split. A text-only
runtime does not carry those, so matching the native window is the conservative target and int8 KV
is the margin, not the prerequisite.

(Correction: an earlier revision of this table said 3.22 GB for fp16 KV, counting only one of K and V
per layer. Both are 2 kv heads x 256 dims x 2 B = 1024 B/token/layer, so 12 layers cost 24,576 B per
token, i.e. 6.44 GB at 262,144 - still inside 48 GB, but with ~5 GB of headroom rather than ~8.)

| aspect | GLM-5.3 | Qwen3.8-Flash-Next | consequence |
|---|---|---|---|
| expert residency | 73 GB of experts, 3053/13248 slots, census-ranked streaming | **30.9 GB** (48 layers x 512 experts x 1.26 MB) | **fully resident, no slot streaming at all** |
| attention mix | MLA at 12 layers + KDA | 36x GDN (linear) + 12x GQA full attention | new attention path, simpler than MLA |
| GDN decay | channelwise (`A_log[8192]`) | **head-wise** (`A_log[48]`, `dt_bias[48]`) | simpler kernel |
| GDN value expansion | 64 k-heads x 128 = v | 16 k-heads, **48 v-heads** (3:1) | needs v-head expansion |
| GDN output gate | `gated_rmsnorm` + sigmoid(gate) | same shape, `output_gate_type: sigmoid` | reusable |
| full attention | MLA absorbed, no RoPE, 64 heads | **GQA 24 q / 2 kv, head_dim 256**, partial rotary 0.25, `interleaved_gate` (q_proj emits q+gate, 12288 = 2x6144) | new |
| q/k norm | none | per-head RMSNorm with `constant_bias = 1.0` | new |
| sparse indexer | DSA, 32x128, topk 2048 pools | QSA: 4 heads x 128, 1 kv head, budget 2048, compress 4 | same family, new dims |
| hyper-connections | mHC, f32 `fn[24,16384]`, sinkhorn routing | **gated residual, low-rank 320**: `hc_norm[10240]`, `input_mix_weight_down[320,10240]`, `up[10240,320]`, `block_inject[4,10240]`; combine-less mixer before the head; no final norm | rewrite |
| MoE | 288 experts, top-8, 2048 inter, router bias | **512 experts, top-10, 640 inter**, `shared_expert_gate[1,2560]`, no router bias | reshape + shared gate |
| PLE | none | **`layers.1.ple`**: conv1d[10240,1,4], key_proj[10240,2560], value_proj, 3 norms + 26 GB EXL3-quantized n-gram table (128 shards of [2500012,41] I16), injected into the 4 hc streams before layer 1 | **new, mandatory for correctness** |
| MTP | 1 full sparse layer, own experts | 1 full layer (`mtp.layers.0`) + `fc_embedding`/`fc_hidden` + own hc mixer (external patch file) | adapt |
| embedding | bf16 [154880,4096] | bf16 [248320,2560] (1.27 GB) | reshape |
| vision | none | `model.visual.*` (27 blocks, 4-bit) | **out of scope** (text-only runtime) |
| quant | exl3 1.4.4, bits 2.05, head_bits 5, mul1, out_scales always | **same, head_bits 4** | **helios quant core reused unchanged** |

## MTP draft head (implemented, `src/engine/mtp.cu`)

The head is one draft block plus an input combine over the trunk's **pre-collapse** stream stack
(the trunk mixer's input, exported as `target_hidden`). Per draft token:

    streams_i = fc_hidden( rmsnorm_over_D(stack_i) * (w_i + 1) ) + fc_embedding( rmsnorm(embed(token)) )

with the `fc_hidden` branch applied per stream and the embedding branch broadcast into all four, so
the draft block starts from the trunk's stream identities. `stream_tap = false` (average the normed
streams, then broadcast) is implemented as a switch, because **upstream states in its own docstring
that there is no reference implementation for this head and that this choice is a semantic guess to
be confirmed by acceptance rate** - so acceptance rate is the acceptance criterion, not a spec.

Two loader gaps this closed: `mtp.pre_fc_norm_hidden` (the per-stream tap norm) and
`mtp.pre_fc_norm_embedding` are now loaded, both with the `(w + 1)` constant-bias convention that
`gr_mix` already uses for `hc_norm`.

Speculation loop (`Runner::spec_step`): the trunk consumes the verified token and captures its tap;
the draft head runs at the **same** position against its own KV cache and proposes the next token;
on agreement the drafted token is committed with a second trunk pass, which leaves logits row 0
predicting the token after it. Because the draft consumes the same position as the trunk rather than
a speculative one, every state advances monotonically with committed tokens - a rejection needs no
rollback at all, so no GDN/PLE snapshot machinery is required. The KV cache likewise needs none: a
rejected token's K/V rows sit past the committed position and are overwritten, since every attention
read is bounded by the query position.

Contract: on return, logits row 0 always predicts the token after the last committed position, so
`next_token()` feeds the following call whether or not the draft was accepted.

## Correctness status (measured, not asserted)

The port runs end to end: `helios load` brings up all 85 GB (227,906 jobs, 471 device tensors
verified, 0 mismatches, 8.8 s at 6.6 GB/s), and `helios gen` produces coherent text.

Every stage was checked against exllamav3 rather than assumed, using two oracles built for the
purpose: a recursive module-hook dump of the reference's per-layer hidden states
(`test/dump_ref_states.py`) and an op-by-op replay of the PLE chain (`test/ple_ref_stage.py`).

| stage | reference | helios | ratio |
|---|---|---|---|
| embedding gather | - | - | bit-exact |
| GatedResidual mix (`_mix_ref`) | mixed 0.849918, post 0.107115 | mixed 0.849919, post 0.107115 | exact |
| layer-0 GDN output | 0.087205 | 0.087197 | 0.9999 |
| PLE output (after conv fix) | 0.017761 | 0.017764 | 1.0002 |
| MoE output L0 / L1 / L2 | .039989 / .008852 / .011581 | .039998 / .008852 / .011584 | 1.0002 / exact / 1.0003 |
| per-layer stream RMS, 48 layers | - | - | median 1.04 |

`gr_mix` was verified independently by recomputing the reference's own `_mix_ref` in NumPy on the
real hc weights and the captured input tensor.

### Two limitations that are properties of the oracle, not the port

1. **The baseline is not deterministic at temperature 0.** Three identical `/v1/completions`
   requests produce three different continuations once the model is past its confident prefix
   (e.g. `# 2026年中国GDP增长目标与政策展望`, `# 2026年中国AI产业趋势报告`,
   `# 10 Best Places to Visit in France`). Token-identical output is therefore not a well-defined
   acceptance criterion against it. `test/parity.py` compares against the prefix the baseline
   *reproduces across its own runs* instead; on that measure the port matches 3 of 5 fixed prompts,
   including the full 24-token `def fibonacci(n):` continuation.
2. **Both engines are non-deterministic run to run.** The MoE permutation uses `atomicAdd` to place
   tokens within an expert, so the accumulation order varies between runs at the 1e-6 level. On
   near-tie positions (e.g. choosing among digit tokens) that is enough to flip a token, which then
   propagates. A deterministic (offset-based, atomic-free) scatter would fix the port's side of it.

### Remaining divergence

The residual is a small per-layer numerical difference (median 1.04) that is large enough to change
the output where the model's choice is fine-grained - e.g. the reasoning-delimiter tokens for the
`Q: ... A:` prompt. Closing it means matching the reference's intermediate precision more exactly,
starting from the layers whose ratio is furthest from 1.

## Design decisions

1. **Fully resident weights.** 36.37 GB of weights fit in 48 GB, so the expert slot manager, PCIe
   streaming, census ranking and the GPU0/GPU1 split that dominated helios are all unnecessary. What
   VRAM is left (~11 GB) goes to KV, GDN states and workspaces — and that is exactly where the
   baseline is squeezed, which is the context lever.
2. **Quantized KV as the context lever.** The baseline reports `kv_cache_quantization_supported:
   false` for this checkpoint, and its 229,888 ceiling exists only because fp16 KV does not fit. A
   q8/int8 KV with a Hadamard rotation is already validated in helios (1.96e-04 relative attention
   error against the engine's own fp16 kernel noise of 1.7-2.8e-04), so context beyond 230k, up to
   and past the native 262,144, is a realistic deliverable rather than a stretch.
3. **MTP is load-bearing** (+13-17 % measured), so it ships in the first correct version, not as an
   afterthought: 1 full-attention layer with its own hc sites, `fc_embedding`/`fc_hidden`, greedy
   acceptance at temperature 0 for the benchmark, sampled acceptance later.
4. **CUDA-graph decode from the start.** helios deferred it; for a fully resident model at M=1 the
   launch overhead is the same class of cost as the arithmetic, and the baseline already benefits.
5. **Reuse, do not rewrite:** the EXL3 quant core (`src/cuda/quant`, kernel-identical format), the
   OpenAI server (tool calls, reasoning separation, SSE, disconnect cancel — all now proven), the
   tokenizer loader, the sampler, the prefix-cache/EAGLE-style state snapshot machinery (adapted
   from KDA to GDN), the benchmark harness, and the chat-parity/test discipline.

## Milestones

1. **Config + inventory + template parity** — parse `config.json`, classify the 304k tensors, render
   the Qwen ChatML template byte-identically to the checkpoint's `chat_template.jinja`
   (`test/chat_parity.py` adapted).
2. **Loader** — arena layout, EXL3 slabs for experts/attn/head, fp16/bf16 for norms, the 26 GB
   n-gram table mapped (not copied) from disk, MTP weights placed.
3. **Forward pass** — embedding + ExpandStreams, GDN layers, GQA+QSA layers, MoE with shared expert,
   PLE injection before layer 1, combine-less mixer, head. Verify **token-identical at temperature 0
   against exllamav3** on fixed prompts, then numerically against a torch oracle per stage.
4. **Server parity** — chat template, thinking/reasoning separation, tool calls, streaming.
5. **Speed** — profile prefill and decode, then close the gap to (and past) 1.7k / 86 tok/s.
6. **Context** — int8 KV + Hadamard, verify error against fp16 at depth, and report the reachable
   context versus the 229,888 baseline.

## Guardrails

* Every optimisation gets a measurement, and every correctness claim gets a test that would fail if
  the claim were false — the GLM build's most expensive lesson was that a test which cannot fail is
  worse than no test.
* exllamav3 is the oracle for both tokens and timings; the same benchmark harness measures both
  (`/tmp/bench_engine.py`, nonce-unique prompts so no prompt cache flatters either side).
* Vision is explicitly out of scope for this deliverable.
