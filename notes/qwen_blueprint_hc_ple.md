# Qwen3.8-Flash-Next blueprint: hyper-connections + PLE + gated RMSNorm

Extracted from exllamav3 by a read-only scout; every formula is backed by a quoted line.
Ground truth for parity is `hyperconnections.py::_mix_ref` (pure fp32 torch), not the fused
`gr_mix` kernels (R<=32) or the half-GEMM path (R>32), which disagree in low-order bits.

## Uncertainties recorded by the scout

- NO WRITE TOOL IN THIS ROLE: I am the read-only investigation subagent and my tool set is {read, grep, glob, web_search, yield}. I could not create local://qwen-blueprint-hc-ple.md. The complete spec document is delivered inline inside `key_formulas` (one entry per section, in document order); the main agent should write it verbatim to local://qwen-blueprint-hc-ple.md.
- Bit-exactness caveat: exllamav3 runs GatedResidual through TWO different arithmetics (`FUSED_MAX_R = 32`): the fused ext.gr_mix kernels (fp32 dot products on RAW fp32 streams, fp16 fn/up weights) for R<=32 and a half-GEMM path (ext.rms_norm -> half, torch.matmul half -> half, silu on half, matmul -> half) for R>32. The two disagree in low-order bits, so a port cannot be bit-identical to BOTH. The code itself designates `_mix_ref()` (pure fp32 torch, hyperconnections.py:331-342) as the ground truth the parity tests compare against.
- Rounding-order caveat inside the fp32 reference itself: hc_mix.cu gr_finalize computes `sigmoid(g)*rmr[h]*(1/H)*w*streams` (hc_mix.cu:452-456), while _mix_ref computes `(sigmoid(g)*normed).mean(-2)` with normed = x*rsqrt* w (hyperconnections.py:335,339); and norm.cu apply4 does `x*w*rmf` (norm.cu:118) whereas rmsnorm.py forward_torch does `x*rsqrt*constant_scale` then `*w` (rmsnorm.py:78-84). Same algebra, different fp32 association order.
- The n-gram table's per-head `head_offsets` / `head_vocab_sizes` / `layer_multipliers` and the trellis `K` are data inside ngram_embedding.safetensors; I could not read that header (24.4 GB binary, my grep tool only scans the first 4 MB and the file is binary). The port must read them at runtime, per ngram_embedding.py:104-175 and conversion/ngram.py:496-509. Shard count is 128 (config split_ngram_parts=128); row count per shard comes from the file header shape (conversion/ngram.py:213-224). Size arithmetic is consistent with 128 shards x 2.5M rows = 320M rows x 41 int16 words (K=4) = 26.2 GB = 24.4 GiB, i.e. num_rows = 16 heads x 20M (ngram_vocab_size_base) [INFERENCE - not read from the file].
- I did not verify the trunk final-mixer tensor names in model.safetensors.index.json: the grep tool only scans the first 4 MB of that 31 MB file and the hyper_connection_mixer/lm_head entries sit past it. Names `model.language_model.hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}.weight` are derived from qwen4_exp.py:285-293 + hyperconnections.py:268-270. Site-form names ARE verified in the index (lines 6165-6168, 6187-6190, ...).
- MTP mixer shapes not verified: mtp_hyper_connection_mixer_patch.safetensors is binary (12.5 MB) and its header is unreadable with my tools. Per assignment context it holds mtp.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}; shapes are inferred [INFERENCE] to be the same family as the trunk (hc_norm [10240], down [320,10240], up [10240,320]) and the MTP mixer is created with use_combine=False (qwen4_exp_mtp.py:100-108).
- The GatedRMSNorm fp32 torch fallback (gated_rmsnorm.py:131-137) and the bf16 CUDA kernel (norm.cu:498-598) differ in dtype/order (`x*rsqrt*...` vs `apply4 = x*w*rmf`); for the qwen4_exp GDN output norm the bf16 kernel path is the one taken (x = core_attn_out is bf16) but I did not execute it to confirm.
- hc_eps/sinkhorn_iters belong to mHC class HyperConnection/HyperHead only; class GatedResidual (the qwen4_exp mixer) has no hc_eps and no combine matrix - the spec states this explicitly rather than guessing values.

---

SECTION 0 - SCOPE, MODEL WIRING AND EXECUTION ORDER (why the three modules see the shapes they see)

Config (authoritative, /home/neron/models/Qwen3.8-Flash-Next-exl3/config.json): text_config.hidden_size=2560, hc_count=4, hc_lowrank=320, rms_norm_eps=1e-6, ngram_size=3, heads_per_ngram=8, ple_embed_dim=2560, ple_conv_kernel_size=4, ple_layer_ids=[2], split_ngram_parts=128, ngram_vocab_size_base=20000000, eos_token_id=248044, image_token_id=248056, output_gate_type="sigmoid". Derived: H=hc_mult=4, D=hidden=2560, hc_hidden=H*D=10240, rank=320, num_heads=(3-1)*8=16, ROW_DIM=ple_embed_dim/num_heads=160.

Module list (qwen4_exp.py:241-294):
  Embedding(embed_tokens)                                          -> then
  ExpandStreams(key="hc_expand", hc_mult=config.hc_mult)           (qwen4_exp.py:244-248)
  then for idx in range(48): if (idx+1) in ple_layer_ids: PLELayer(key=f"model.language_model.layers.{idx}.ple", layer_idx=-(idx+1), ...) (qwen4_exp.py:255-263) BEFORE build_qwen4_block(...); so with ple_layer_ids=[2] the PLE layer runs after block 0 and before block 1 (layer_idx stored as -2 to keep the recurrent-state key distinct from block 1: ple.py:195-196 "assert layer_idx < 0").
  final: GatedResidual(key="model.language_model.hyper_connection_mixer", hc_mult=4, hidden_size=2560, rms_norm_eps=config.rms_norm_eps, use_combine=False, out_dtype=torch.half) then Linear(key="lm_head", qmap="block", qbits_key="head_bits") (qwen4_exp.py:283-294). qwen4_exp.py:283 comment: "# No final model norm: the combine-less mixer collapses the stream stack".

Sequential execution, no per-module branching: model_ls.py:265-276 `for module, instance, idx in self.fwd_modules: ... x = module.prepare_for_device(x, params); x = module.forward(x, params)`.

Block sites (TransformerBlock, transformer.py:147-195), executed in this order per block:
  x fp32 streams; `hc_post, hc_comb, y = self.attn_hc.mix(x, params)` (151); `y = y.half()` (152); no attn_norm in qwen4_exp (build_qwen4_block passes attn_norm=None), so `y = self.attn.forward(y, params)` (159); `x = self.attn_hc.apply_(x, y, hc_post, hc_comb, params)` (164-165); then `hc_post, hc_comb, y = self.mlp_hc.mix(x, params)` (174-175); `y = y.half()` (176); `y = self.mlp.forward(y, params)` (187); `x = self.mlp_hc.apply_(x, y, hc_post, hc_comb, params)` (190-191). For a GatedResidual site, hc_comb is always None (hyperconnections.py:378 `return post.view(b, s, self.hc_mult), None, mixed.view(b, s, self.hidden_size)`) and apply_ ignores it. Note there is NO pre-norm and NO post-norm at these sites: the mix() collapse replaces the input norm, apply() replaces the residual add (transformer.py:56-63 asserts post-norms/residual scalars are None).

Weight names verified in the converted checkpoint index (model.safetensors.index.json):
  "model.language_model.layers.{L}.mlp_hyper_connection.{input_mix_weight_down,input_mix_weight_up,block_inject_weight,hc_norm}.weight" (index lines 6165-6168 for L=0)
  "model.language_model.layers.{L}.attn_hyper_connection.{...}" (index lines 6187-6190)
  "model.language_model.layers.1.ple.{norm_conv,norm_key,norm_query,conv1d,key_proj,value_proj}.weight" (index lines 6191-6196)
  Site form uses block_inject_weight (use_combine=True default, qwen4_exp.py:96-102); the trunk final mixer uses use_combine=False so it has NO block_inject tensor (hyperconnections.py:271-272 loads it only `if self.use_combine`).

SECTION 1 - ExpandStreams (hyperconnections.py:19-49)

Weights: NONE. `def optimizer_targets(self): return []` (hyperconnections.py:24-26); the class stores only self.hc_mult (line 25).

Forward signature and shapes: `def forward(self, x, params, out_dtype=None)` (31); body is exactly one line:
  hyperconnections.py:32  `return x.float().unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()`
  x: (bsz, seq, 2560). Callers pass the Embedding output, which for qwen4_exp is fp32 (Embedding default `out_dtype: torch.dtype | None = torch.float`, embedding.py:21; instantiated without out_dtype at qwen4_exp.py:240-243).
  out: (bsz, seq, 4, 2560) fp32, contiguous; out[:, :, h, :] == x for all four h.
  out_dtype argument is ignored (cast is unconditional .float()).
This is the only initialisation of the stream stack: streams = broadcast embedding; there is no learned stream embedding, scale, or extra norm here.

SECTION 2 - GatedResidual: constructor, weights, load-time preparation (hyperconnections.py:211-325)

Constructor: `GatedResidual(config, key, hc_mult, hidden_size, rms_norm_eps, use_combine=True, out_dtype=None)` (237-246). Buffers (253-262): norm_w_raw (checkpoint weight), norm_w (H,D) fp32 = w_raw+1, w_h (H*D) fp16 = w_raw+1, down_h (rank, H*D) fp16, up_h (H*D, rank) fp16 "checkpoint orientation", upx_h (H, D/4, rank, 4) fp16 fused layout, inject_h (H, H*D) fp16 site form only, proj_h = cat(down, inject) fp16, fn_h = cat(down, inject)*w fp16, rank=0, FUSED_MAX_R=32 (line 235), rms_eps=rms_norm_eps.

Weights read (load, 265-273):
  hyperconnections.py:268  `self.norm_w_raw = stc.get_tensor(f"{self.key}.hc_norm.weight", device, no_defer = True)` -> shape [H*D] = [10240], raw checkpoint dtype (zero-init source; used as 1+w).
  hyperconnections.py:269  `down = stc.get_tensor(f"{self.key}.input_mix_weight_down.weight", ...)` -> [rank, H*D] = [320, 10240].
  hyperconnections.py:270  `up = stc.get_tensor(f"{self.key}.input_mix_weight_up.weight", ...)` -> [H*D, rank] = [10240, 320], row index = h*2560 + d, column = rank index.
  hyperconnections.py:271-272 `inject = stc.get_tensor(f"{self.key}.block_inject_weight.weight", ...) if self.use_combine else None` -> [H, H*D] = [4, 10240] (site form only; NOT loaded for use_combine=False).

_prepare (275-303) - exact order and dtypes, this is the part a port must replicate step by step:
  hyperconnections.py:284  `self.norm_w = (self.norm_w_raw.float() + 1.0).view(H, Dh).contiguous()`   # fp32, +1 constant bias
  hyperconnections.py:285  `self.w_h = self.norm_w.flatten().half().contiguous()`                   # fp16 (H*D,)
  hyperconnections.py:286  `self.rank = down.shape[0]`                                                # 320
  hyperconnections.py:288  `self.proj_h = down.half().contiguous()`                                  # use_combine=False branch
  hyperconnections.py:293  `self.down_h = self.proj_h[: self.rank]`                                   # view
  hyperconnections.py:295-298  `tmp = g_tensor_cache...view(M, H * Dh); tmp.copy_(self.proj_h); tmp *= self.w_h.float()`
  hyperconnections.py:299  `self.fn_h = tmp.half().contiguous()`
  => fn_h[i, h*D+d] = fp16( fp32(fp16(down[i, h*D+d])) * (w_raw[h*D+d] + 1) ). NOTE the double rounding: down is rounded to fp16 FIRST (line 288/291 .half()), then multiplied by the fp32 (w_raw+1), then rounded to fp16 again. M = fn_h.shape[0] = 320 when use_combine=False, 324 when True.
  hyperconnections.py:300  `self.up_h = up.half().contiguous()`   # (H*D, rank) = (10240, 320)
  hyperconnections.py:301-303 `self.upx_h = self.up_h.view(H, Dh // 4, 4, self.rank).permute(0, 1, 3, 2).contiguous()`
  => upx_h[h, d//4, i, j] = up_h[h*Dh + (d//4)*4 + j, i], i.e. contiguous over the rank index for each of the 4 lane elements of a float4 column (fused-kernel layout; gr_finalize reads `upt + (((h * (D / 4) + c) * LR + i) * 4)` for elements x..w, hc_mix.cu:429).

weights_numel (324-325): `n + 2 * rank * n + (hc_mult * n if use_combine else 0)` with n = H*D (already converted fp16 counts).

SECTION 3 - GatedResidual fp32 reference math (_mix_ref, hyperconnections.py:331-342) - THE ORACLE

`def _mix_ref(self, streams)` returns (post (b,s,H) or None, mixed (b,s,D)), both fp32. Streams (b, s, H, D).
  hyperconnections.py:335  `normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.rms_eps) * self.norm_w`
      => rmr[r,h] = rsqrt( (1/D) * sum_d x[r,h,d]^2 + eps ),  normed[r,h,d] = x[r,h,d] * rmr[r,h] * (w_raw[h,d]+1)
         (per-STREAM RMS over the D=2560 channels; mean is over the last dim = D, not over H*D)
  hyperconnections.py:336  `flat = normed.flatten(-2)`                       # (R, H*D)
  hyperconnections.py:337  `t = F.silu(F.linear(flat, self.down_h.float()) / self.hc_mult)`
      => t[r,i] = silu( (1/H) * sum_{h,d} normed[r,h,d] * down[i,h*D+d] ), i in [0,rank)
  hyperconnections.py:338  `w = torch.sigmoid(F.linear(t, self.up_h.float()))`
      => w[r,h*D+d] = sigmoid( sum_i t[r,i] * up[h*D+d, i] )
  hyperconnections.py:339  `mixed = (w.unflatten(-1, (self.hc_mult, self.hidden_size)) * normed).mean(dim = -2)`
      => mixed[r,d] = (1/H) * sum_h sigmoid(g[r,h,d]) * normed[r,h,d]
  hyperconnections.py:340-341 `post = 2.0 * torch.sigmoid(F.linear(flat, self.inject_h.float()) / self.hc_mult) if self.use_combine else None`
      => post[r,h] = 2*sigmoid( (1/H) * sum_{d} normed[r,h,d] * inject[h, h*D+d] )   [site form only; None here]
Activations: silu(x) = x*sigmoid(x); sigmoid = 1/(1+exp(-x)); eps = self.rms_eps = config.rms_norm_eps = 1e-6. There is NO combine matrix, NO Sinkhorn, NO hc_eps in this class (contrast mHC HyperConnection, hyperconnections.py:39-44/149-153).

SECTION 4 - GatedResidual runtime path A: fused ext.gr_mix (R <= FUSED_MAX_R = 32)

Dispatcher (_mix, hyperconnections.py:344-372). Streams are first coerced to fp32 contiguous: lines 348-352 `s3 = streams.reshape(R, H, Dh); if s3.dtype != torch.float: s3 = s3.float(); if not s3.is_contiguous(): s3 = s3.contiguous()`. Then:
  hyperconnections.py:354-355 `post = torch.empty((R, H), dtype = torch.float, device = dev) if self.use_combine else None`
  hyperconnections.py:357-360 `if R <= self.FUSED_MAX_R:` / `dots = torch.empty((R, self.fn_h.shape[0] + 1, H), dtype=torch.float, device=dev)` / `mixed = torch.empty((R, Dh), dtype=torch.half, device=dev)` / `ext.gr_mix(s3, self.fn_h, self.upx_h, self.w_h, self.rms_eps, dots, post, mixed)`
  Workspace dots: (R, M+1, H) = (R, 321, 4) for use_combine=False.

Kernel 1, gr_dots_kernel (hc_mix.cu:291-357), grid (M+1, R), 128 threads; one block per (fn row j | sumsq row j==M) and per row r:
  For j < M it accumulates, per stream h, sum over that stream's D channels of stream[h,d] * fn[j, h*D+d] with fp32 FMA on fp16 weights:
    hc_mix.cu:326-329 `a = fmaf(s0.x, LOW_TO_FLOAT(w0.x), a); ...`
    hc_mix.cu:355  `dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;`
  For j == M it accumulates the per-stream sum of squares (hc_mix.cu:339-346 branch `acc[M] = fmaf(s.x, s.x, ...)` is the mHC analogue; GatedResidual uses `a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))))`), giving dots[r, M, h] = sum_d streams[r,h,d]^2.
  MATHS: dots[r, j, h] = sum_d fn[j, h*D+d] * streams[r,h,d]   for j<M;  dots[r, M, h] = sum_d streams[r,h,d]^2.
  Note the dot is taken against the RAW streams; the per-stream rms factor is applied in kernel 2, and the norm weight is already folded into fn (hc_mix.cu:264-271 comment).

Kernel 2, gr_finalize_kernel<H=4, HALF_OUT=true> (hc_mix.cu:359-467), grid (n_chunks, R), 256 threads, shared t_s[LR] fp32:
  hc_mix.cu:384 `rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);`  => rmr[h] = rsqrt(dots[r,M,h]/D + eps)
  hc_mix.cu:386 `const float inv_h = 1.0f / (float) H;`
  hc_mix.cu:387-394 `for (int i = threadIdx.x; i < LR; i += NUM_THREADS) { v = sum_h rmr_s[h]*dr[i*H+h] (line 392); v *= inv_h (393); t_s[i] = v * sigmoidf_(v); (394) }`  => t[i] = silu( (1/H) * sum_h rmr[h]*dots[r,i,h] )
  hc_mix.cu:396-402 (only when post != nullptr) `post[r,h] = 2.0f * sigmoidf_(v * inv_h)` with v = sum_h rmr_s[h]*dr[(LR+h)*H+h]  => post[h] = 2*sigmoid( (1/H) * sum_h rmr[h]*dots[r,LR+h,h] )
  column loop (hc_mix.cu:421-456): one warp per 4-column group; per rank i, `g[h].x = fmaf(ti, LOW_TO_FLOAT(u.x), g[h].x)` etc. where u = upx element for (h, column, i) (429-432), shfl-reduced over the 32 lanes (440-443), then for lane 0:
    hc_mix.cu:451-456 `half4 wq = *(const half4*) (w + h*D + 4*c); float coef = rmr_s[h] * inv_h; o.x = fmaf(sigmoidf_(g[h].x) * coef * LOW_TO_FLOAT(wq.x), s.x, o.x); ...`
    MATHS: g[r,h,d] = sum_i up[h*D+d, i] * t[r,i];  mixed[r,d] = sum_h sigmoid(g[r,h,d]) * rmr[h] * (1/H) * (w[h,d]+1) * streams[r,h,d] = (1/H) * sum_h sigmoid(g[r,h,d]) * normed[r,h,d].
  HALF_OUT path stores with `__floats2half2_rn` (hc_mix.cu:458-464) => mixed is fp16.

apply() (SECTION 5).

SECTION 5 - GatedResidual runtime path B: half-GEMM (R > 32) and the residual injection apply_

_mix GEMM branch, hyperconnections.py:361-371 (exact dtype chain):
  362 `normed = torch.empty((R * H, Dh), dtype = torch.half, device = dev)`
  363-364 `ext.rms_norm(s3.view(R * H, Dh), self.w_h, normed, self.rms_eps, 0.0, 1.0, False, False, H)`  # constant_bias=0 (the +1 is already inside w_h), constant_scale=1, add_residual=False, w_groups=H
  365 `dm = torch.matmul(normed.view(R, H * Dh), self.proj_h.t())`  # half x half -> half, (R, rank [+ H])
  366 `t = F.silu(dm[:, : self.rank] / H)`                          # half division, silu in half
  367-368 `if self.use_combine: post.copy_(2.0 * torch.sigmoid(dm[:, self.rank :].float() / H))`
  369 `g = torch.matmul(t, self.up_h.t())`                          # half -> (R, H*Dh)
  370-371 `mixed = (torch.sigmoid(g.float()).view(R, H, Dh) * normed.float().view(R, H, Dh)).mean(dim = -2).half()`
  MATHS: same as the fused path; sigmoid is evaluated after upcasting g to fp32, the mean is over dim H, and the final result is rounded to half.

mix() wrapper, hyperconnections.py:374-378: input (b, s, H, D) fp32, returns (post (b,s,H) fp32 or None, None, collapsed (b,s,D) fp16).

apply_ (hyperconnections.py:380-403) - the ONLY place the sublayer output is added:
  389-390 `if "quant_preserve" in params or "capture" in params: return x + post.unsqueeze(-1) * y.float().unsqueeze(-2)`  # non-in-place guard
  393-401 `b, s = x.shape[:2]; y2 = y.reshape(b * s, self.hidden_size); if y2.dtype not in (torch.half, torch.float): y2 = y2.half(); ext.hc_apply(x.view(b*s, H, D), y2.contiguous(), post.reshape(b*s, H).contiguous(), None)`
  403 `return x`  # IN PLACE
  The kernel is hc_apply with comb == nullptr (hc_mix.cu:714-765 dispatch `<4, half|float, false>`), whose math is documented at hc_mix.cu:470-473: "Without comb (GatedResidual): x[h, d] <- post[h] * y[d] + x[h, d]. Pure per-column mix of the H stream rows, so it runs in place", and implemented as `o.x = post_r[h] * yv.x; ... else { o.x += xv[h].x; ... }` (hc_mix.cu:519-544). post[h] is a scalar PER STREAM, shared across all D channels: every stream receives the SAME sublayer output y scaled by its own gate. In qwen4_exp, y from attn/mlp is fp32 (out_dtype=torch.float, qwen4_exp.py:141/... site modules), so the fp32 branch of hc_apply is used.

SECTION 6 - GatedResidual combine-less FINAL MIXER (forward, use_combine=False)

hyperconnections.py:406-420:
  408 `assert not self.use_combine, "site-form GatedResidual is consumed via mix()/apply_()"`
  411-415 MTP trunk tap: `if self.key in params.get("export_state_norm_keys", ()): ... states.append(x.flatten(-2).half())`  # appends the PRE-collapse stack (bsz, seq, H*D=10240) fp16; used by Qwen4ExpMTPModel.attach_to (qwen4_exp_mtp.py:152-157)
  417 `_, mixed = self._mix(x)`
  418-420 `mixed = mixed.view(b, s, self.hidden_size); dt = out_dtype or self.out_dtype; return mixed if dt is None else mixed.to(dt)`
  In qwen4_exp the final mixer is constructed with out_dtype=torch.half (qwen4_exp.py:292), so the model output is (bsz, seq, 2560) fp16 - i.e. the collapse produces the hidden state directly.
  THERE IS NO FINAL NORM anywhere after this point: qwen4_exp.py:283 "# No final model norm: the combine-less mixer collapses the stream stack", and the next module is the lm_head Linear (qwen4_exp.py:294+). The mixer is a pure weighted mean (weights = sigmoid-gated per channel), and with use_combine=False there is no inject gate and no block_inject tensor at all (load, hyperconnections.py:271-272).
  Same code path in the MTP head: qwen4_exp_mtp.py:100-108 creates `GatedResidual(key="mtp.hyper_connection_mixer", use_combine=False, out_dtype=torch.half)` and sample_from_state() (qwen4_exp_mtp.py:~195-205) reshapes the flattened stack to (b, s, 4, 2560) and calls mixer.forward before the shared lm_head.

SECTION 7 - PLELayer: construction, weights, shapes (ple.py:121-236)

Constructor args (ple.py:123-137): (config, key, layer_idx, hidden_size=2560, hc_mult=4, ple_embed_dim=2560, ngram_size=3, heads_per_ngram=8, eos_token_id=248044, conv_kernel_size=4, rms_norm_eps=1e-6, qmap=None, stream_from_disk=None, out_dtype=None, mm_token_id=248056).
  ple.py:146-150 `self.conv_dilation = ngram_size`; `self.conv_state_len = (conv_kernel_size - 1) * self.conv_dilation` = (4-1)*3 = 9; `self.gate_scale = 1.0 / math.sqrt(hidden_size)` = 1/sqrt(2560) = 0.019764235...; hc_hidden = 4*2560 = 10240.
Weights:
  `{key}.ple_embedding.ngram_embedding.*` - NGramEmbedding submodule (ple.py:152-160), its table is out-of-line in ngram_embedding.safetensors (see SECTION 9).
  `{key}.key_proj.weight` [10240, 2560] (Linear in=2560 out=hc_hidden=10240, qmap=None -> fp16, out_dtype=torch.half; ple.py:161-168).
  `{key}.value_proj.weight` [2560, 2560] (Linear, out_dtype=torch.half; ple.py:169-176).
  `{key}.norm_key.weight`, `{key}.norm_query.weight`, `{key}.norm_conv.weight` - each [4, 2560] grouped RMSNorm weights: ple.py:179-184 `def norm(name): return RMSNorm(config, f"{key}.{name}", rms_norm_eps, constant_bias = 1.0, groups = hc_mult)`. Semantics: ROW (r,h) is normalised by the plain per-row RMS and multiplied by (w_raw[h, :] + 1.0), the weight row selected by (row % 4) (rmsnorm.py:56-58 groups comment and 138-140 "the kernel selects the weight row by (row % groups)"). Source weights are zero-init, so the effective initial weight is 1.0.
  `{key}.conv1d.weight` - depthwise conv kernel, shape (hc_hidden, 1, ksize) = (10240, 1, 4) (ple.py:207-208 `self.conv_w = self.config.stc.get_tensor(f"{self.key}.conv1d.weight", device, float2half = True, no_defer = True)`).
  Index verification: model.safetensors.index.json lines 6191-6196 list exactly model.language_model.layers.1.ple.{norm_conv,norm_key,norm_query,conv1d,key_proj,value_proj}.weight (plain .weight, not trellis -> key_proj/value_proj really are fp16, which is what unlocks the fused ext.ple_forward_streams path at ple.py:272-274).

Recurrent state (PLELayerState, ple.py:31-118): conv_state (max_batch_size, hc_hidden=10240, win+max_history) fp16 with win = conv_state_len = 9, zeroed at alloc (ple.py:69-72); id_state (max_batch_size, ctx+max_history) int64 ON CPU, pre-filled with eos_token_id (ple.py:73-77 comment "the n-gram hashing runs host-side on the (pinned) input ids, so the id history must never round-trip through the device"); ctx = ple_embedding.context_len = ngram_size-1 = 2 (ple.py:49). rewind() copies the trailing num_tokens columns back to the front (ple.py:94-101). Layer state is keyed by (layer_idx, instance) = (-2, 0) so it cannot collide with decoder block 1.

SECTION 8 - PLELayer forward: ids, n-gram hash, gather, dot-product gate, dilated conv, injection

PyTorch reference (forward_streams_reference, ple.py:290-306) - quote by quote:
  294 `key = self.key_proj.forward(emb, params).view(bsz, seq, H, D)`        # emb (bsz, seq, 2560) fp16 -> fp16 (bsz,seq,4,2560)
  295 `key = self.norm_key.forward(key, params, out_dtype = torch.float)`    # grouped RMS (4 rows of 2560) -> fp32
  296 `value = self.value_proj.forward(emb, params)`                         # (bsz, seq, 2560) fp16
  297 `query = self.norm_query.forward(streams, params, out_dtype = torch.float)`  # streams (bsz,seq,4,2560) fp32 -> grouped RMS -> fp32
  300 `gate = torch.bmm(query.view(-1, 1, D), key.reshape(-1, D, 1)).view(bsz, seq, H)`   # per-stream dot, fp32
  302 `ext.ple_gate(gate, value, gated, self.gate_scale)`
  303 `normed = self.norm_conv.forward(gated, params, out_dtype = torch.half).flatten(-2)`  # grouped RMS -> fp16 (bsz,seq,10240)
  304 `conv_out, conv_stream = self._short_conv(normed, conv_state)`
  305 `delta = gated + conv_out.view(bsz, seq, H, D)`
  306 `return delta, conv_stream`
  gate kernel math, ple.cu:19-20 comment and 43-53 body:
    gate_scale: ple.cu:43 `float g = gate[rh] * gate_scale;`
    signed sqrt: ple.cu:44-45 `float a = sqrtf(fmaxf(fabsf(g), 1e-6f)); float ss = g > 0.0f ? a : (g < 0.0f ? -a : 0.0f);`  (sign(0)=0, clamp |g| at 1e-6)
    gate: ple.cu:46 `float s = __fdividef(1.0f, 1.0f + __expf(-ss));`  => s = sigmoid(ss)
    output: ple.cu:50-53 `o4.x = s * LOW_TO_FLOAT(v4.x); ...` with v4 = value[r, d..d+3] (fp16) => out fp32.
    FORMULA: gated[b,s,h,d] = sigmoid( sign(x)*sqrt(max(|x|,1e-6)) ) * value[b,s,d], where x = gate_scale * sum_c query_n[b,s,h,c]*key_n[b,s,h,c], and gate_scale = 1/sqrt(2560).
  Short conv (_short_conv, ple.py:238-252), depthwise, causal, dilated:
    249-250 `if conv_state is None: conv_state = xt.new_zeros((bsz, ch, self.conv_state_len))` / `xt = torch.cat((conv_state.to(xt.dtype), xt), dim = -1)`   # padded column stream (bsz, 10240, 9 + seq)
    251 `y = F.conv1d(xt, self.conv_w, groups = ch, dilation = self.conv_dilation)`  # groups=10240 (depthwise), dilation=3, stride 1, padding 0
    252 `return F.silu(y).transpose(1, 2), xt`   # silu evaluated on the fp16 conv output -> half
    FORMULA: with columns indexed 0..9+seq-1, out[j] = silu( sum_{k=0..3} w[d, 0, k] * xt[d, j + 3k] ) for j in [0, 9+seq-1-9]; output length = 9 + seq - 9 = seq. In new-column terms t' = j - 9: out[t'] = silu( sum_k w[d,0,k] * normed[t' - 3k] ), i.e. a 4-tap causal dilated conv with receptive field 10 positions (t', t'-3, t'-6, t'-9). conv_out is fp16; delta = gated (fp32) + conv_out (type-promotes to fp32).
  Fused ext path (ple.py:271-288) which IS the one used in this model, mirroring the reference exactly (ple.cu:113-194):
    ple.cu:160 `at::Tensor key = at::matmul(emb2, key_w);` (half x half) then ple.cu:162 `rms_norm(key.view({R*H, D}), norm_key_w, key_n, eps, 1.0f, 1.0f, false, false, (int) H);`  # constant_bias = 1.0, constant_scale = 1.0, w_groups = 4, fp32 output
    ple.cu:166 `rms_norm(s2, norm_query_w, query_n, eps, 1.0f, 1.0f, false, false, (int) H);`  # s2 = streams.view(R*H, D)
    ple.cu:168-169 `gate = at::bmm(query_n.view({R*H,1,D}), key_n.view({R*H,D,1})).view({bsz, seq, H});`  # fp32 x fp32 dot
    ple.cu:170 `at::Tensor value = at::matmul(emb2, value_w).view({bsz, seq, D}).contiguous();`  (half)
    ple.cu:173 `ple_gate(gate.contiguous(), value, gated, (float) gate_scale);`
    ple.cu:176 `rms_norm(gated.view({R * H, D}), norm_conv_w, normed, eps, 1.0f, 1.0f, false, false, (int) H);`  # fp16 out buffer
    ple.cu:186-190 conv_stream[:, :, 9:9+seq] = normed.T; then `at::conv1d(conv_stream, conv_w, nullopt, {1}, {0}, {conv_dilation}, hc)` (groups = 10240)
    ple.cu:191 `at::Tensor conv_out = at::silu(y).transpose(1, 2);`
    ple.cu:194 `at::add_out(d2, gated.view({bsz, seq, hc}), conv_out);`  # delta = gated + conv_out, fp32
  Injection into the streams (ple.py:308-360):
    324 `ids = ids.to("cpu", torch.int64)` (ids come from params["input_ids"], set by Embedding.forward, embedding.py:83-84)
    325-330 multimodal handling: `if self.mm_token_id is not None and int(ids.max()) >= FIRST_MM_EMBEDDING_INDEX: ids = ids.clone(); ids[ids >= FIRST_MM_EMBEDDING_INDEX] = self.mm_token_id` (the literal placeholder token is hashed, matching HF)
    341-345 with recurrent states: `prev_ids = torch.stack([id_state[s, :ctx] for s in slots]); history = torch.cat((prev_ids, ids), dim=1)` (ctx=2 previous ids prepended); `cs = torch.stack([conv_state[s, :, :win] for s in slots]).to(x.device)`; `delta, conv_stream = self.forward_streams(x, history, params, conv_state=cs)`
    346-352 state writeback: with save_history the full conv_stream tail and history are written right-aligned (window width w = min(state dim, conv_stream width)); otherwise `conv_state[s, :, :win].copy_(conv_stream[i, :, -win:]); id_state[s, :ctx].copy_(history[i, -ctx:])`
    356-359 without recurrent states (position 0 only): `pad = ids.new_full((bsz, ctx), self.ple_embedding.eos_token_id); delta, _ = self.forward_streams(x, torch.cat((pad, ids), dim=1), params)`
    360 `return x + delta`  => streams (bsz, seq, 4, 2560) fp32; the delta is a broadcast add over all four streams. The PLE output is NOT gated by any learned scalar and there is no residual mixing here.

SECTION 9 - n-gram hashing and 128-shard table indexing (NGramEmbedding)

Head geometry (ngram_embedding.py:59-68): ngram_size=3, context_len=ngram_size-1=2, heads_per_ngram=8, num_heads=(ngram_size-1)*heads_per_ngram=16, head_dim=ple_embed_dim/num_heads=2560/16=160, asserted `assert self.head_dim == ROW_DIM` with ROW_DIM=160 (ngram_codec.py:31).
Aux tensors, host-side only (ngram_embedding.py:104-114): head_offsets (16,) int64, head_vocab_sizes (16,) int64, layer_multipliers (ngram_size=3,) int64 - all loaded to CPU (the hashing runs on the CPU next to the token ids); head_bias (16,160) fp16 goes to the device for dequant. For the trellis layout the names are `{key}.head_offsets` etc., for an unquantized table `{parent}.ngram_heads_offsets` etc. with parent = model.language_model.layers.1.ple.ple_embedding (ngram_embedding.py:145-171).
CSR-style gather: fetch_rows/embed_ids (ngram_embedding.py:258-284, 312-321) unique the flat (bsz*seq*16) ids, gather the unique rows, and scatter back through the inverse map, then reshape to (bsz, seq, 16*160=2560).

N-gram id formula (torch reference, ngram_embedding.py:298-310):
  th = token_history.long(); shifted[s] = _shift_right_ignore_eos(th, s) for s in 0..ngram_size-1;
  for ngram in 2..ngram_size: lo = (ngram-2)*heads_per_ngram; hi = lo + heads_per_ngram;
    mixed = shifted[0] * layer_multipliers[0];  for position in 1..ngram-1: mixed ^= shifted[position] * layer_multipliers[position];
    block = (mixed.unsqueeze(-1) % head_vocab_sizes[lo:hi]) + head_offsets[lo:hi];
  ngram_ids = cat(blocks, -1)[:, -out_len:]
EOS segmentation (_shift_right_ignore_eos, ngram_embedding.py:279-295): `valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)` where position_in_segment counts positions since the last eos inclusive; any shifted source that would cross an eos boundary (or is < 0) is replaced by eos_token_id. So an n-gram never spans a segment boundary; at sequence start the 2 carried context ids are eos (ple.py:358).
C++ hot path, byte-for-byte the same semantics (ngram.cu:43-124, and ngram.cu:27-30 comment "Exact port of the torch reference (NGramEmbedding.compute_ngram_ids + torch.unique), which the HF parity test pins down bitwise"):
  ngram.cu:82-84 `if (p > 0 && row[p - 1] == eos_token) prev_eos = p - 1;` ... `int64_t seg_start = prev_eos + 1; uint64_t mixed = (uint64_t) row[p] * (uint64_t) mult[0];`
  ngram.cu:88-96 `for (int64_t s = 1; s < ngram_size; ++s) { int64_t src = (p - s >= seg_start && p - s >= 0) ? row[p - s] : eos_token; mixed ^= (uint64_t) src * (uint64_t) mult[s]; int64_t lo = (s - 1) * heads_per_ngram; for (h = lo; h < lo + heads_per_ngram; ++h) { int64_t m = (int64_t) mixed % szs[h]; if (m < 0) m += szs[h]; ... hk[idx] = { m + offs[h], idx }; } }`
  => head block h in [(s-1)*8, s*8) is the s+1-gram head set; mixed always includes the multiplier-0 term for the current token; the XOR is over uint64 products (torch uses bitwise_xor on int64 products - equivalent for the low bits, and torch.remainder matches the negative-corrected % in C++).
  Dedup + ordering (ngram.cu:105-123): rows are sorted by (hash, idx) with std::sort, unique ids emitted ascending, `inv_p[hk[i].idx] = u` gives the inverse map, and each unique id's head is `upper_bound(offs, offs+H, last) - offs - 1` (offsets are ascending starts of disjoint per-head ranges). Python mirrors this for the bias with searchsorted(right=True)-1 then clamp(0, num_heads-1) (ngram_embedding.py:264-267).

Sharding (config split_ngram_parts=128): the table file holds shard_0..shard_127.trellis, contiguous row arrays; all shards but the last hold rows_per_shard rows (ngram_embedding.py:117-140 asserts this and reads rows_per_shard from the first shard shape; conversion/ngram.py:213-224, 355-380 writes them that way). Routing is a plain division:
  ngram_embedding.py:261-263 `shard = uids_cpu // self.rows_per_shard; local = uids_cpu - shard * self.rows_per_shard;` (with `if len(store) == 1: return gather(0, uids_cpu)` - disk mode merges the 128 contiguous handles into one handle spanning the whole table, ngram_embedding.py:180-193).
Row decode (ngram_codec.py:31, 36-40, 85-100): each row is (1 + 10*K) int16 words; word 0 is the fp16 row scale, the rest an exl3_ngram_trellis ring bitstream over the mul1 codebook (MUL1 = 0x83DCD12D). `row[i] = codebook[state_i] * scale + head_bias[head]`, all fp32 math on fp16 inputs, then cast to the requested dtype (`out = codebook[states].float() * scales.float().unsqueeze(1); if bias is not None: out = out + bias.float()`), with `head = searchsorted(head_offsets, uid, right=True) - 1`. The CUDA decode (ext.ngram_dequant, ngram.cu "ngram_dequant: one block per row ... Matches dequant_rows() in ngram_codec.py (fp16 codebook rounding included)") must match this bit-for-bit. ROW_DIM=160 -> the concatenated 16 heads give 2560 = ple_embed_dim, which is the input width of key_proj/value_proj.

SECTION 10 - GatedRMSNorm with gate_activation="sigmoid" (gated_rmsnorm.py)

Signature: `GatedRMSNorm(config, key, rms_norm_eps, out_dtype=None, qmap=None, constant_bias=0.0, groups=1, gate_first=False, gate_activation="silu")` (gated_rmsnorm.py:10-24); `assert gate_activation in ("silu", "sigmoid")` (41). Weight: `weight = self.config.stc.get_tensor(f"{self.key}.weight", self.device, allow_bf16 = True)` (51) - i.e. it keeps bf16 (no float2half), shape [dim]; `self.bc = ext.BC_GatedRMSNorm(weight, eps, constant_bias, groups, gate_first, 1 if gate_activation == "sigmoid" else 0)` (55-62).
Forward (123-140):
  gated_rmsnorm.py:130 `gate_act = 1 if self.gate_activation == "sigmoid" else 0`
  gated_rmsnorm.py:131 `if gate_act and not (x.dtype == torch.bfloat16 and x.is_contiguous() and gate.is_contiguous()):`
  fp32 torch fallback (132-137), used for sigmoid whenever x is NOT bf16 (e.g. fp16 or non-contiguous), quoted:
    133-137 `h = x.to(torch.float32); h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps); h = self.weight.to(torch.float32) * h; h = h * torch.sigmoid(gate.to(torch.float32)); return h.to(out_dtype or self.out_dtype or x.dtype)`
    FORMULA: y = ( fp32(w) * ( fp32(x) * rsqrt(mean_c fp32(x)^2 + eps) ) ) * sigmoid(fp32(gate)), cast to out_dtype.
  CUDA path (138-139): `y = torch.empty_like(x, dtype = out_dtype or self.out_dtype); ext.gated_rms_norm(x, self.weight, y, gate, self.rms_norm_eps, self.constant_bias, self.groups, self.gate_first, gate_act)`; kernel body norm.cu:498-598:
    norm.cu:515 `#define _gate_fn(v) (gate_act == 1 ? _sigmoid_f(v) : _silu(v))` with `_sigmoid_f(x) = __fdividef(1.0f, 1.0f + __expf(-x))` (norm.cu:133-136) and `_silu(x) = x * __fdividef(1.0f, 1.0f + __expf(-x))` (norm.cu:124-129)
    norm.cu:550 `float rmf = rsqrtf(sum / (float)dim + epsilon);`  # sum of squares accumulated in fp32 over the bf16 input row
    norm.cu:569-582: weight (bfloat16 or float32 ONLY - the dispatch table at norm.cu:655-670 has no half weight), `if (constant_bias != 0.0f) w4 += constant_bias;`, then `apply4(x4, w4, rmf); x4 *= _gate_fn(g4);`
    norm.cu:118-124 `apply4: x4.x = x4.x * w4.x * rmf;` (i.e. (x*w)*rmf, a different association than the torch fallback)
    FORMULA (ext path): y = (x * (w + constant_bias) * rmf) * sigmoid(gate), with x/w read as bf16->fp32, gate read as fp32 or bf16, and the result rounded to out_dtype (half or float) by write_half4/write_float4 (norm.cu:80-93).
    Dispatch requires x bf16 (norm.cu:502 `const bfloat16* x`), weight bf16|fp32, weight row selected by `(row % w_groups)` (norm.cu:521 `const weight_t* w_row = w + (size_t) (row % w_groups) * dim;`), row count = product of all dims but the last (norm.cu:634-636), and dim <= 256 selects a 32-thread block, else 1024 (norm.cu:640-642).

Usage in qwen4_exp (why sigmoid matters here): build_qwen4_block passes `norm = GatedRMSNorm(config, f"{key}.linear_attn.norm", config.rms_norm_eps, out_dtype=torch.half, gate_activation=config.output_gate_type)` (qwen4_exp.py:133-137) with output_gate_type="sigmoid" - so constant_bias=0.0, groups=1, gate_first=False, weight = checkpoint `...linear_attn.norm.weight` [128]. The GDN forward calls `core_attn_out = self.norm.forward(core_attn_out, params, gate = z)` (gated_delta_net.py:1063) where core_attn_out is bf16 (gated_delta_rule.py:180-184 / 237-241 `dtype = torch.bfloat16`) and z is fp32 (z_proj out_dtype=torch.float, gated_delta_net.py:431-437, called at 1014-1015; for the KDA variant z is the low-rank g_a/g_b pair, 996-998). x bf16 contiguous + gate fp32 contiguous therefore satisfies the ext-path condition at gated_rmsnorm.py:131, and the dispatch row is (kBFloat16 x, kFloat weight, kHalf y, kFloat gate, small) - norm.cu:657 - producing y fp16 which is then reshaped to (b, s, 48*128) (gated_delta_net.py:1064) and fed to out_proj.

SECTION 11 - NUMERICS TABLE FOR THE PORT (dtypes, epsilons, order)

Global: rms_norm_eps = 1e-6 for PLELayer norms, GatedResidual.rms_eps, GatedRMSNorm, and the mHC norm (all from config.text_config.rms_norm_eps). PLE gate clamp epsilon = 1e-6 on |gate*scale| (ple.cu:44). hc_eps and sinkhorn_iters exist ONLY for class HyperConnection/HyperHead; GatedResidual has neither.

ExpandStreams: in fp32 (bsz,seq,2560) -> out fp32 (bsz,seq,4,2560), no weights, no arithmetic beyond the cast/broadcast.

GatedResidual per site (R = bsz*seq):
  weights: hc_norm.weight fp16-equivalent stored (raw, [10240]); down/up/inject loaded then rounded to fp16 (down_h/up_h/inject_h, lines 288-293, 300); fn_h = fp16( fp32(fp16(proj)) * fp32(w_raw+1) ), w_h = fp16(w_raw+1).
  fused path (R<=32): streams read fp32; dots summed fp32 with fp16 weights; rmr fp32 (1/sqrt of fp32 mean + 1e-6); t = silu(fp32) -> fp32; g = fp32 accumulation of fp16 up weights * fp32 t; sigmoid in fp32 (`1/(1+__expf(-x))`); mixed accumulated fp32 and rounded to fp16 once at the end.
  GEMM path (R>32): norm output fp16 (rounding point!), half GEMM accumulate fp32 -> fp16 dm, division by H in fp16, silu in fp16, second half GEMM -> fp16 g, sigmoid after fp32 upcast, mean in fp32, final fp16.
  Both paths agree with _mix_ref except for these intermediate roundings.
  apply_: x fp32 in place, y fp32 (block out_dtype=torch.float) or fp16, post fp32; x[h,d] += post[h]*y[d].
  final mixer: input fp32 (b,s,4,2560) -> output fp16 (b,s,2560) via the same _mix; NO final norm.

PLELayer (per token):
  emb (from the n-gram table) -> fp16 (bsz, seq, 2560); key_proj/value_proj are fp16 GEMMs -> fp16; norm_key/norm_query output fp32; the dot is fp32 x fp32 -> fp32; gate = sigmoid(signed_sqrt(dot * 1/sqrt(2560))) fp32; value fp16; gated = fp32; norm_conv output ROUNDED TO FP16 before the conv; conv input/weights/activation fp16 (silu evaluated in fp16), conv accumulation fp32 per ATen conv1d; delta = gated (fp32) + conv_out (fp16 -> promoted); streams stay fp32 (delta added by x + delta). The injected delta is the sum of the gated value and the conv of the *fp16-rounded* normed gated values - the fp16 rounding of normed before the conv is a required numerics detail (ple.py:303, ple.cu:176).
  key_proj/value_proj weights: fp16 with NO bias (`... and self.key_proj.inner.bias is None and self.value_proj.inner.bias is None`, ple.py:274) - which is the condition that selects the fused ext path.

GatedRMSNorm (sigmoid): x bf16, weight bf16 (or fp32), norm in fp32, weight+bias multiply then rms scale as (x*w)*rmf, gate sigmoid in fp32, output half; eps 1e-6; constant_bias 0.0 for the GDN output norm; gate_first False.

SECTION 12 - EXPLICITLY NOT DETERMINABLE FROM THESE FILES

1. The concrete values of head_offsets, head_vocab_sizes, layer_multipliers, head_bias and K (plus the shard row count) live in ngram_embedding.safetensors and are read at runtime (ngram_embedding.py:104-175, conversion/ngram.py:496-509). They are data, not code; the formulas above are complete but the numbers must be taken from the file header/metadata (`self.K = int(self.metadata["K"])`, `row_dim`, `shard_rows`, conversion/ngram.py:497-509). Size arithmetic is consistent with 128 shards x 2.5M rows x 41 int16 words (K=4) [INFERENCE].
2. The trunk final-mixer and lm_head tensor entries in model.safetensors.index.json were not readable with my search tools (31 MB file, tool scans the first 4 MB); the names in SECTION 0/2 are derived from the constructor code.
3. Which of the two GatedResidual arithmetics a given forward takes depends only on R = bsz*seq (<=32 fused, else GEMM), so no single implementation can be bit-identical to both; the code's own oracle is _mix_ref (hyperconnections.py:331-333 "fp32 torch reference of the mix (the parity tests' ground truth)").
4. PLELayer.forward asserts it needs recurrent states for position > 0 (ple.py:356-357); the prefill-only branch passes a zero conv_state. The exact history-window sizing (win + max_history) is a cache configuration detail outside these files.
