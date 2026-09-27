# Qwen3.8-Flash-Next: GDN / QSA indexer / MTP reconnaissance

From a read-only scout over exllamav3 (full spec was too large to persist through the artifact
reader; read the sources directly when implementing - paths below are the map).

## Correctness scoping (decide before writing kernels)

- **GDN prefill vs decode use different algorithms.** Decode uses the sequential CUDA kernel; prefill
  (seqlen >= 48, no history) calls **flash-linear-attention's chunked WY algorithm** (`chunk_size 64`,
  `l2norm eps 1e-6`, `scale 1/sqrt(128)`) at
  `/home/neron/shared-venv-gpu/lib/python3.12/site-packages/fla/ops/gated_delta_rule/`. The two differ
  in rounding, so **a port using one sequential recurrence everywhere matches to bf16 noise, not bit
  exactly** - the same scoping the GLM KDA port used (its oracle was per-stage numeric comparison
  rather than token-identity).
- **MTP has no HF reference for its tap.** `Qwen4ExpMTPInputLayer.stream_tap=True` is documented
  in-repo as a *semantic guess* (`modules/arch_specific/qwen4_exp_mtp.py:17-21`), and the trunk tap
  (`GatedResidual.forward` appending `x.flatten(-2).half()` when its key is in
  `params['export_state_norm_keys']`) is a bespoke exllamav3 mechanism. Both are mandatory for MTP
  parity *with exllamav3*; the acceptance rate is the only oracle.
- `A_log` / `dt_bias` load with `allow_bf16=True` (`gated_delta_net.py:808-809`) and the kernels
  accept fp32 or bf16 - the checkpoint stores **bf16** (verified from the file header), so upcast.

## Files to read when implementing

| module | source |
|---|---|
| GDN forward, weight names, kernel selection | `modules/gated_delta_net.py` (1260 lines), `modules/gated_delta_net_fn/` |
| GDN decode kernel choice at load time (`is_quantized_split` / `is_quantized_kda`) | `gated_delta_net.py:588-700` |
| QSA indexer | `modules/qsa_indexer.py`, `qsa_triton.py:150-303` |
| full attention (BC split/combine, paged) | `modules/attn.py`, `libtorch/attention.cpp:596-670`, `bc_attn.py:383-405` |
| MTP wiring and tap | `architecture/qwen4_exp_mtp.py`, `modules/arch_specific/qwen4_exp_mtp.py` |
| MoE router + experts | `modules/block_sparse_mlp.py:1002-1440`, `routing.cu`, exl3 moe kernels |

## Router detail (confirmed)

Router is **`std`**: softmax over the **top-k logits only**, no bias, no `routed_scaling_factor` -
`config.json` has no `e_score_correction_bias` and the tensor map confirms only `mlp.gate.weight`
(512x2560) plus `shared_expert_gate` (1x2560). This differs from GLM, which had a centred fp16 bias
and a 2.5x routed scale.
