// Launchers + workspace plumbing for the chunked gated delta rule (kernels in gdn_chunked.cuh).
// The three stage launches are ported from ninfer-3090
// src/ops/linear_attention/gated_delta_net/chunked/launch.{cu,h}, with the tensor plumbing replaced
// by helios' raw-pointer + explicit-stream convention and the q/k L2 normalization (a separate op
// in ninfer) folded in as a fourth, tiny launch.
#include "gdn.cuh"
#include "gdn_chunked.cuh"
#include "helios_shim.cuh"

#include <cstdint>

namespace helios {

namespace {

using gdn::BT;
using gdn::kStateDim;

constexpr size_t kAlign = 256;

size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// The output stage keeps at most one resident wave of CTAs and distributes the (chunk x head) jobs
// evenly over it. ninfer hard-codes the RTX 5090's 170 SMs; this engine runs on a 3090 (82), so the
// count is queried once instead of baked in.
int sm_count()
{
    static int n = [] {
        int dev = 0, v = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 82;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || v <= 0) return 82;
        return v;
    }();
    return n;
}

}  // namespace

namespace aux {

using helios::align_up;

size_t gdn_chunked_workspace_bytes(int num_v_heads, int num_k_heads, int tokens)
{
    if (tokens <= 0 || num_v_heads <= 0 || num_k_heads <= 0) return 0;
    const size_t t = (size_t)((tokens / BT) * BT);
    const size_t chunks = t / BT;
    const size_t panel = (size_t)kStateDim * num_v_heads * t * 2;  // W / U / v_new, bf16
    size_t bytes = 0;
    bytes += align_up((size_t)num_v_heads * t * 4);                               // g_cumsum, fp32
    bytes += align_up(panel);                                                    // W
    bytes += align_up(panel);                                                    // U
    bytes += align_up(panel);                                                    // v_new
    bytes += align_up((size_t)kStateDim * kStateDim * num_v_heads * chunks * 2);  // h_chunk
    bytes += align_up((size_t)t * num_k_heads * kStateDim * 2);                   // q_norm
    bytes += align_up((size_t)t * num_k_heads * kStateDim * 2);                   // k_norm
    return bytes;
}

gdn::ChunkWorkspace gdn_chunked_bind_workspace(void* arena, int num_v_heads, int num_k_heads, int tokens)
{
    gdn::ChunkWorkspace w;
    if (!arena || tokens <= 0) return w;
    const int t = (tokens / BT) * BT;
    const int chunks = t / BT;
    const size_t panel_elems = (size_t)kStateDim * num_v_heads * t;

    char* p = (char*)arena;
    auto take = [&](size_t bytes) {
        char* out = p;
        p += align_up(bytes);
        return out;
    };

    w.g_cumsum = (float*)take((size_t)num_v_heads * t * 4);
    w.W = (__nv_bfloat16*)take(panel_elems * 2);
    w.U = (__nv_bfloat16*)take(panel_elems * 2);
    w.v_new = (__nv_bfloat16*)take(panel_elems * 2);
    w.h_chunk = (__nv_bfloat16*)take((size_t)kStateDim * kStateDim * num_v_heads * chunks * 2);
    w.q_norm = (__nv_bfloat16*)take((size_t)t * num_k_heads * kStateDim * 2);
    w.k_norm = (__nv_bfloat16*)take((size_t)t * num_k_heads * kStateDim * 2);
    w.tokens = t;
    return w;
}

}  // namespace aux

namespace gdn {

using helios::sm_count;

namespace {

template <int KPanelCols, int WuPanelCols, int BlockWarps>
cudaError_t launch_prepare_fixed(const prepare_wy_wu_config& cfg, dim3 grid)
{
    using dims = prepare_wy_wu::kernel_dims<KPanelCols, WuPanelCols>;
    constexpr int smem_bytes = dims::SMEM_FLOATS * (int)sizeof(float);
    constexpr int threads = BlockWarps * ptx::kWarpSize;

    cudaError_t err = cudaFuncSetAttribute(prepare_wy_wu::prepare_wy_wu_kernel<KPanelCols, WuPanelCols, BlockWarps>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) return err;

    prepare_wy_wu::prepare_wy_wu_kernel<KPanelCols, WuPanelCols, BlockWarps>
        <<<grid, dim3(threads, 1, 1), smem_bytes, cfg.stream>>>(cfg.k_in, cfg.v_in, cfg.g_in, cfg.beta, cfg.W,
                                                                cfg.U, cfg.g_cumsum_out, cfg.geom);
    return cudaGetLastError();
}

template <int NStrip>
cudaError_t launch_state_fixed(const state_passing_config& cfg)
{
    using D = state_passing::kernel_dims<NStrip>;
    constexpr int smem_bytes = state_passing::smem_layout<NStrip>::SMEM_FLOATS * (int)sizeof(float);

    cudaError_t err = cudaFuncSetAttribute(state_passing::state_passing_kernel<NStrip>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) return err;

    const dim3 grid((unsigned)((int64_t)cfg.geom.H_v * D::D_STRIPS), 1, 1);
    const dim3 block(D::THREADS, 1, 1);
    state_passing::state_passing_kernel<NStrip><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.W, cfg.U, cfg.k_in, cfg.g_cumsum, cfg.state_in, cfg.v_new, cfg.h_chunk, cfg.state_out, cfg.geom,
        cfg.chunks);
    return cudaGetLastError();
}

template <bool MULTI_JOB>
cudaError_t launch_output_fixed(const chunk_output_config& cfg, dim3 grid)
{
    constexpr int smem_bytes = output::kernel_dims::SMEM_BYTES;

    cudaError_t err = cudaFuncSetAttribute(output::output_kernel<MULTI_JOB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_bytes);
    if (err != cudaSuccess) return err;

    const dim3 block(output::THREADS, 1, 1);
    output::output_kernel<MULTI_JOB><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.q_in, cfg.k_in, cfg.v_new, cfg.g_cumsum, cfg.h_chunk, cfg.attn_out, cfg.geom, cfg.scale, cfg.chunks);
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_l2norm_qk(const __nv_bfloat16* mixed_qkv, int64_t src_col_off, __nv_bfloat16* out,
                             const ChunkGeom& geom, int tokens, cudaStream_t stream)
{
    if (mixed_qkv == nullptr || out == nullptr || tokens <= 0) return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    // gdn_l2norm_qk_kernel gives each WARP one (token, head) row - the sum-of-squares reduction is a
    // __shfl_down tree inside a single warp - so the grid must cover rows in units of warps per
    // block. Sizing it per thread (kBlock rows per block) left 7/8 of the panel unwritten, which fed
    // stage 1 a garbage k panel: A = -beta*(k.k) was meaningless, T_inv was garbage, and the whole
    // delta-rule update collapsed to the decayed carry-in state.
    constexpr int kRowsPerBlock = kBlock / (int)ptx::kWarpSize;
    const int rows = tokens * geom.H_qk;
    const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    gdn_l2norm_qk_kernel<<<blocks, kBlock, 0, stream>>>(mixed_qkv, src_col_off, out, geom, tokens);
    return cudaGetLastError();
}

cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config& cfg)
{
    if (cfg.chunks <= 0 || cfg.k_in == nullptr || cfg.v_in == nullptr || cfg.g_in == nullptr ||
        cfg.beta == nullptr || cfg.W == nullptr || cfg.U == nullptr || cfg.g_cumsum_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid((unsigned)cfg.chunks, (unsigned)cfg.geom.H_v, 1);
    if (cfg.geom.H_v == 32) return launch_prepare_fixed<32, 16, 4>(cfg, grid);
    return launch_prepare_fixed<64, 32, 8>(cfg, grid);
}

cudaError_t launch_state_passing(const state_passing_config& cfg)
{
    if (cfg.chunks <= 0 || cfg.W == nullptr || cfg.U == nullptr || cfg.k_in == nullptr || cfg.g_cumsum == nullptr ||
        cfg.state_in == nullptr || cfg.v_new == nullptr || cfg.h_chunk == nullptr || cfg.state_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    // H_v >= 48 uses 8 warps per 16-wide value strip (two CTAs/SM); narrower heads get one CTA of
    // 16 warps per 32-wide strip instead.
    if (cfg.geom.H_v >= 48) return launch_state_fixed<16>(cfg);
    return launch_state_fixed<32>(cfg);
}

cudaError_t launch_output(const chunk_output_config& cfg)
{
    if (cfg.chunks <= 0 || cfg.q_in == nullptr || cfg.k_in == nullptr || cfg.v_new == nullptr ||
        cfg.g_cumsum == nullptr || cfg.h_chunk == nullptr || cfg.attn_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    const int64_t target = (int64_t)sm_count() * 4;
    const int64_t jobs = (int64_t)cfg.chunks * cfg.geom.H_v;
    const int64_t jobs_per_block = (jobs + target - 1) / target;
    const int64_t grid_chunks = (cfg.chunks + jobs_per_block - 1) / jobs_per_block;
    const dim3 grid((unsigned)grid_chunks, (unsigned)cfg.geom.H_v, 1);
    if (jobs_per_block <= 1) return launch_output_fixed<false>(cfg, grid);
    return launch_output_fixed<true>(cfg, grid);
}

}  // namespace gdn

namespace aux {

// L2-normalize the first `tokens` tokens' q and k into the workspace panels, then run the three
// stages. `tokens` must be a non-zero multiple of 64 and no larger than the workspace capacity.
// The recurrent state is updated in place (the caller passes the same pointer as in and out).
//
// Launch failures abort rather than return: by the time stage 2 runs, the recurrent state has
// already been partially advanced, so a soft failure would hand back a silently wrong state. The
// shape preconditions are checked by the caller before anything is launched.
void gdn_chunked_stages(const bfloat16* mixed_qkv, const float* g, const bfloat16* beta, float* recurrent_state,
                        bfloat16* core_attn_out, int tokens, int num_k_heads, int num_v_heads, void* workspace,
                        cudaStream_t stream)
{
    HELIOS_AUX_CHECK(mixed_qkv && g && beta && recurrent_state && core_attn_out && workspace,
                     "cuda_chunked_gated_delta_rule: null pointer");
    HELIOS_AUX_CHECK(tokens > 0 && tokens % BT == 0, "chunked path needs a positive multiple of 64 tokens");

    gdn::ChunkWorkspace ws = gdn_chunked_bind_workspace(workspace, num_v_heads, num_k_heads, tokens);
    HELIOS_AUX_CHECK(ws.tokens >= tokens, "chunked workspace smaller than the token count");

    gdn::ChunkGeom geom;
    geom.H_qk = num_k_heads;
    geom.H_v = num_v_heads;
    geom.qk_row_stride = (int64_t)num_k_heads * kStateDim;                    // normalized q / k panels
    geom.v_row_stride = (int64_t)(2 * num_k_heads * kStateDim + num_v_heads * kStateDim);  // packed conv row
    geom.v_row_off = (int64_t)(2 * num_k_heads * kStateDim);
    geom.k_src_col_off = (int64_t)num_k_heads * kStateDim;

    // q and k are normalized here (ninfer runs l2norm as a separate op) so the stages can read
    // contiguous panels instead of striding through the packed conv row. This is the same
    // rsqrtf(sum + 1e-6) normalization the serial kernel performs internally, over the same bf16
    // inputs - only the summation order differs, and the reduction tree is fixed either way.
    HELIOS_CUDA_CHECK(gdn::launch_l2norm_qk(mixed_qkv, 0, ws.q_norm, geom, tokens, stream));
    HELIOS_CUDA_CHECK(gdn::launch_l2norm_qk(mixed_qkv, geom.k_src_col_off, ws.k_norm, geom, tokens, stream));

    const float scale = 1.0f / sqrtf((float)kStateDim);

    gdn::prepare_wy_wu_config prep{};
    prep.geom = geom;
    prep.k_in = ws.k_norm;
    prep.v_in = mixed_qkv;
    prep.g_in = g;
    prep.beta = beta;
    prep.W = ws.W;
    prep.U = ws.U;
    prep.g_cumsum_out = ws.g_cumsum;
    prep.chunks = tokens / BT;
    prep.stream = stream;
    HELIOS_CUDA_CHECK(gdn::launch_prepare_wy_wu(prep));

    gdn::state_passing_config st{};
    st.geom = geom;
    st.W = ws.W;
    st.U = ws.U;
    st.k_in = ws.k_norm;
    st.g_cumsum = ws.g_cumsum;
    st.state_in = recurrent_state;
    st.v_new = ws.v_new;
    st.h_chunk = ws.h_chunk;
    st.state_out = recurrent_state;
    st.chunks = tokens / BT;
    st.stream = stream;
    HELIOS_CUDA_CHECK(gdn::launch_state_passing(st));

    gdn::chunk_output_config out{};
    out.geom = geom;
    out.q_in = ws.q_norm;
    out.k_in = ws.k_norm;
    out.v_new = ws.v_new;
    out.g_cumsum = ws.g_cumsum;
    out.h_chunk = ws.h_chunk;
    out.attn_out = core_attn_out;
    out.scale = scale;
    out.chunks = tokens / BT;
    out.stream = stream;
    HELIOS_CUDA_CHECK(gdn::launch_output(out));
    HELIOS_CUDA_CHECK(cudaGetLastError());
}

}}  // namespace helios::aux
