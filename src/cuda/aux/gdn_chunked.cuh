#pragma once
// Chunked (WY-representation) gated delta rule - the three kernel stages ported from
// ninfer-3090 src/ops/linear_attention/gated_delta_net/chunked/.
//
// The serial kernel (gdn.cu, cuda_recurrent_gated_delta_rule_kernel_128) walks the sequence one
// token at a time, so a 1024-token prefill chunk costs 1024 fully-serial steps per v-head and is
// the second-largest prefill phase. This path computes the same recurrence with the UT-transform
// of Yang et al.'s Gated Delta Networks, in three kernels:
//
//   1. prepare_wy_wu  (grid: chunk x v-head)   g cumsum, A = -beta*(k.k) decayed, T_inv in smem,
//                                              U = T_inv @ (beta*V), W = T_inv @ (beta*2^g*K)
//   2. state_passing  (grid: v-head x d-strip) chunk-sequential inter-chunk state scan
//   3. output         (grid: chunk x v-head)   inter (q @ h_chunk^T) + intra (A @ v_new) output
//
// It is the same recurrence, not the same arithmetic: the W/U/v_new/h_chunk workspace is bf16, four
// matmuls run on TF32, and the state summation order differs, so chunked and serial agree to
// ~1e-2 relative rather than bit-exactly (upstream accepts chunked-vs-serial at rtol/atol 5e-2).
//
// Determinism: no atomics anywhere. Every reduction is either a fixed-index smem loop, a fixed
// __shfl_down tree, or a single mma whose internal order is fixed by hardware, so two launches on
// identical inputs are byte-identical. The engine's reproducibility guarantee therefore survives
// the port (test_gdn_chunked_parity asserts it).
//
// Geometry adaptation vs ninfer: ninfer is handed separate contiguous q/k/v tensors, while
// helios' conv writes one packed row per token - [q (Nk*128) | k (Nk*128) | v (Nv*128)] - so the
// per-token row strides and the in-row offsets of k and v are passed explicitly in ChunkGeom
// instead of being derived from the head counts. Everything else is the ninfer kernel body.

#include "mma.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace helios { namespace gdn {

// Fixed geometry: 128x128 heads, 64-token chunks. kStateDim/kChunkSize come from ninfer's
// gated_delta_net/common.h; the kernels hard-code BT = 64 = 4 * BC = 16 and D = 128 throughout.
inline constexpr int kStateDim = 128;
inline constexpr int kChunkSize = 64;

inline constexpr int BT = kChunkSize;  // tokens per chunk
inline constexpr int BC = 16;          // sub-block of the triangular solve
inline constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 8;

static_assert(BT % BC == 0, "BT must be a multiple of BC");
static_assert(BT % MMA_M == 0, "BT must be a multiple of MMA_M");

// Addressing. q and k are read from the L2-normalized panels produced by gdn_l2norm_qk_kernel
// (contiguous [T, H_qk*D], qk_row_stride elements per token); v is read straight out of the packed
// conv row, where it starts at v_row_off. helios' conv writes one row per token laid out
// [q (Nk*D) | k (Nk*D) | v (Nv*D)], so v_row_stride is the packed width (2*H_qk*D + H_v*D) and is
// not derivable from the head counts - hence the explicit fields.
struct ChunkGeom
{
    int H_qk = 0;
    int H_v = 0;
    int64_t qk_row_stride = 0;  // elements between tokens in the normalized q / k panels
    int64_t v_row_stride = 0;   // elements between tokens in the packed row holding v
    int64_t v_row_off = 0;      // element offset of the v block inside the packed row
    int64_t k_src_col_off = 0;  // element offset of k inside the packed row (l2norm source)

    // v-head h shares the q/k head of its group of 3. Plain division: evaluated once per block, so
    // the reciprocal-multiply ninfer uses here would only add a magic-number failure mode.
    __host__ __device__ __forceinline__ int qk_head(int h_v) const { return h_v / (H_v / H_qk); }
};

// FP32 shared-memory tile with ninfer's XOR swizzle. The swizzle is what keeps the smem fragment
// loads (and the two-way STS in the Schur scratch) bank-conflict free.
template <int STRIDE>
struct SmemTile
{
    float* __restrict__ base;
    static_assert(STRIDE == 16 || STRIDE >= 32, "SmemTile: only STRIDE in {16, 32, 64, 128, ...} supported");

    __device__ __forceinline__ int swz_xor(int row) const
    {
        if constexpr (STRIDE >= 32) { return ((row & 3) << 3) | (row & 4); }
        else                        { return ((row >> 1) & 3) << 2; }
    }

    __device__ __forceinline__ float& at(int row, int col) const
    {
        return base[row * STRIDE + (col ^ swz_xor(row))];
    }

    __device__ __forceinline__ float4& vec4_at(int row, int col) const
    {
        return *reinterpret_cast<float4*>(&base[row * STRIDE + (col ^ swz_xor(row))]);
    }
};

// Bulk BF16 -> FP32 staging of one [ROWS, STRIDE] panel into a SmemTile.
template <int ROWS, int STRIDE, int THREADS, class View>
__device__ __forceinline__ void
issue_load_bf16_to_float_vec4(View view, const __nv_bfloat16* __restrict__ gmem_base_row0,
                              int64_t gmem_row_stride_elems, int tid)
{
    static_assert(STRIDE % 4 == 0, "issue_load_bf16_to_float_vec4: STRIDE must be a multiple of 4");
    constexpr int VEC_PER_ROW = STRIDE / 4;
    constexpr int N_VEC = ROWS * VEC_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VEC; v += THREADS) {
        const int row = v / VEC_PER_ROW;
        const int col4 = v - row * VEC_PER_ROW;
        const __nv_bfloat16* gmem_ptr =
            gmem_base_row0 + static_cast<int64_t>(row) * gmem_row_stride_elems + col4 * 4;
        const ptx::Bf16x4Pack packed = ptx::load_vec<ptx::Bf16x4Pack>(gmem_ptr);
        const float2 lo = ptx::bf16x2_to_float2(packed.pair[0]);
        const float2 hi = ptx::bf16x2_to_float2(packed.pair[1]);
        view.vec4_at(row, col4 * 4) = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
}

namespace prepare_wy_wu {

using ptx::Cache;
using ptx::cp_async;
using ptx::cp_commit;
using ptx::cp_wait;
using ptx::ldmatrix_x2;
using ptx::ldmatrix_x4;
using ptx::mma_bf16;
using ptx::mma_tf32;
using ptx::pack_bf16x2_rn;
using ptx::pack_bf16x2_rz;
using ptx::smem_addr;
using ptx::store_vec;

static_assert(kChunkSize == 64, "prepare_wy_wu: kChunkSize must be 64 (BT = 4 * BC = 16 is hard-coded)");
static_assert(kStateDim == 128);

constexpr int N_SUB = BT / BC;    // 4
constexpr int WY_WARPS = N_SUB;    // 4 warps own the triangular construction
constexpr int N_K_TILES = BT / MMA_K;  // 8 (recompute_wu)
constexpr int BF16_MMA_K = 16;

// Phase D scratch row stride. Smallest value >= 16 (per-warp data needs 16 cols) that preserves
// the 2-way-write / 0-way-read property:
//   * stride % 32 = 20 -> (20*lane_g + 2*lane_t) lands all 32 lanes on 16 distinct even banks
//     (strict 2-way write); cumulative shifts {0,20,8,28,16,4,24,12} are all distinct mod 32.
//   * Reads (20*lane_g + lane_t) span 32 distinct banks (0-way read).
// Equivalent BC to stride=36 but saves 1024B of scratch_floats.
constexpr int SCR_STRIDE = 20;

template <int KPanelCols, int WuPanelCols>
struct kernel_dims
{
    static_assert(KPanelCols == 32 || KPanelCols == 64);
    static_assert(WuPanelCols == 16 || WuPanelCols == 32);
    static_assert(KPanelCols % BF16_MMA_K == 0);
    static_assert(WuPanelCols % MMA_N == 0);

    static constexpr int N_K_PANELS = kStateDim / KPanelCols;
    static constexpr int K_TILES_PER_PANEL = KPanelCols / BF16_MMA_K;
    static constexpr int N_WU_PANELS = kStateDim / WuPanelCols;

    static constexpr int T_inv_floats = BT * BT;                    // 16 KiB
    static constexpr int scratch_floats = WY_WARPS * BC * SCR_STRIDE;  // 5 KiB
    static constexpr int wu_stage_floats = BT * WuPanelCols;
    static constexpr int k_stage_bf16 = BT * KPanelCols;
    static constexpr int k_stage_floats = k_stage_bf16 / 2;
    static constexpr int stage_floats =
        scratch_floats > wu_stage_floats ? scratch_floats : wu_stage_floats;
    static constexpr int output_stage_floats = BT * WuPanelCols / 2;
    static_assert(stage_floats >= k_stage_floats);
    static_assert(stage_floats >= scratch_floats);

    static constexpr int g_floats = BT;
    static constexpr int beta_floats = BT;
    static constexpr int bg_floats = BT;
    static constexpr int SMEM_FLOATS =
        T_inv_floats + stage_floats + output_stage_floats + g_floats + beta_floats + bg_floats;
};

template <int STRIDE>
struct Bf16SmemTile
{
    __nv_bfloat16* __restrict__ base;
    static_assert(STRIDE == 32 || STRIDE == 64);

    __device__ __forceinline__ int swizzled_col(int row, int col) const
    {
        return col ^ ((row & (STRIDE / 8 - 1)) << 3);
    }

    __device__ __forceinline__ __nv_bfloat16* ptr(int row, int col) const
    {
        return base + row * STRIDE + swizzled_col(row, col);
    }
};

template <int Stride>
__device__ __forceinline__ int wu_output_swizzled_col(int row, int col)
{
    return (((col >> 3) ^ (row & (Stride / 8 - 1))) << 3) | (col & 7);
}

__device__ __forceinline__ void
scatter_frag_to_scr(const float frag[8], float* __restrict__ scr_smem, int warp, int lane)
{
    float* Sptr = scr_smem + warp * BC * SCR_STRIDE;
    const int lane_g = lane >> 2;
    const int col_2t = (lane & 3) << 1;
    Sptr[lane_g * SCR_STRIDE + col_2t] = frag[0];
    Sptr[lane_g * SCR_STRIDE + col_2t + 1] = frag[1];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t] = frag[2];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 1] = frag[3];
    Sptr[lane_g * SCR_STRIDE + col_2t + 8] = frag[4];
    Sptr[lane_g * SCR_STRIDE + col_2t + 9] = frag[5];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 8] = frag[6];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 9] = frag[7];
}

// 16x16x16 mma: A from raw row-major scratch (stride SCR_STRIDE), B from swizzled M_view at
// (M_row_off, M_col_off).
__device__ __forceinline__ void mma16_raw_x_swiz(float D[8], int lane, const float* __restrict__ A_buf,
                                                 SmemTile<BT> M_view, int M_row_off, int M_col_off)
{
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0 = A_buf[lane_g * SCR_STRIDE + (k_off + lane_t)];
        const float a1 = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t)];
        const float a2 = A_buf[lane_g * SCR_STRIDE + (k_off + lane_t + 4)];
        const float a3 = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t + 4)];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0 = M_view.at(M_row_off + k_off + lane_t, M_col_off + n_off + lane_g);
            const float b1 = M_view.at(M_row_off + k_off + lane_t + 4, M_col_off + n_off + lane_g);
            mma_tf32(D[nt * 4 + 0], D[nt * 4 + 1], D[nt * 4 + 2], D[nt * 4 + 3], a0, a1, a2, a3, b0, b1);
        }
    }
}

// 16x16x16 mma: A from swizzled M_view, B from raw row-major scratch.
__device__ __forceinline__ void mma16_swiz_x_raw(float D[8], int lane, SmemTile<BT> M_view, int M_row_off,
                                                 int M_col_off, const float* __restrict__ B_buf)
{
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0 = M_view.at(M_row_off + lane_g, M_col_off + k_off + lane_t);
        const float a1 = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t);
        const float a2 = M_view.at(M_row_off + lane_g, M_col_off + k_off + lane_t + 4);
        const float a3 = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t + 4);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0 = B_buf[(k_off + lane_t) * SCR_STRIDE + (n_off + lane_g)];
            const float b1 = B_buf[(k_off + lane_t + 4) * SCR_STRIDE + (n_off + lane_g)];
            mma_tf32(D[nt * 4 + 0], D[nt * 4 + 1], D[nt * 4 + 2], D[nt * 4 + 3], a0, a1, a2, a3, b0, b1);
        }
    }
}

// Templated on (MY_W, MY_J) so A_reg[k] resolves to compile-time indices after unrolling;
// otherwise nvcc parks A_reg in local memory.
template <int MY_W, int MY_J>
__device__ __forceinline__ void compute_off_diag(float out[8], int warp, int lane, const float A_reg[N_SUB][8],
                                                 float* __restrict__ scr_smem, SmemTile<BT> M_view)
{
    static_assert(0 <= MY_J && MY_J < MY_W && MY_W <= N_SUB);

    float sum[8] = {};

#pragma unroll
    for (int k = MY_J; k < MY_W; ++k) {
        scatter_frag_to_scr(A_reg[k], scr_smem, warp, lane);
        __syncwarp();
        const float* A_buf = scr_smem + warp * BC * SCR_STRIDE;
        mma16_raw_x_swiz(sum, lane, A_buf, M_view, k * BC, MY_J * BC);
        __syncwarp();
        if (k == MY_J) {  // diagonal-block correction (folded after unroll)
#pragma unroll
            for (int e = 0; e < 8; ++e) sum[e] += A_reg[k][e];
        }
    }

    scatter_frag_to_scr(sum, scr_smem, warp, lane);
    __syncwarp();
    const float* B_buf = scr_smem + warp * BC * SCR_STRIDE;

    float prod[8] = {};
    mma16_swiz_x_raw(prod, lane, M_view, MY_W * BC, MY_W * BC, B_buf);

#pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = prod[e] + sum[e];
}

__device__ __forceinline__ void store_frag_to_M(const float frag[8], int my_w, int my_j, int lane_g,
                                                int lane_t, SmemTile<BT> M_view)
{
    const int row_g0 = my_w * BC + lane_g;
    const int row_g1 = row_g0 + 8;
    const int col_base = my_j * BC + 2 * lane_t;
    M_view.at(row_g0, col_base) = frag[0];
    M_view.at(row_g0, col_base + 1) = frag[1];
    M_view.at(row_g1, col_base) = frag[2];
    M_view.at(row_g1, col_base + 1) = frag[3];
    M_view.at(row_g0, col_base + 8) = frag[4];
    M_view.at(row_g0, col_base + 9) = frag[5];
    M_view.at(row_g1, col_base + 8) = frag[6];
    M_view.at(row_g1, col_base + 9) = frag[7];
}

// In-place forward substitution of the 16x16 diagonal block of (I + A). One warp per block, 16 of
// its lanes own one column, and the 15 elimination steps are separated by __syncwarp - so the order
// is fixed and no step races another lane's column.
template <int DIAG_BLOCK>
__device__ __forceinline__ void solve_diag_block(int lane, SmemTile<BT> M_view)
{
    constexpr int diag_off = DIAG_BLOCK * BC;
    const int wcol = lane & 15;
    for (int i = 1; i < BC; ++i) {
        const int row_i = diag_off + i;
        const int col = diag_off + wcol;
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < BC - 1; ++j) {
            if (j < i) sum += M_view.at(row_i, diag_off + j) * M_view.at(diag_off + j, col);
        }
        __syncwarp();
        if (lane < 16 && wcol < i) M_view.at(row_i, col) += sum;
        __syncwarp();
    }
}

template <int WU_PANEL_COLS, int BLOCK_THREADS>
__device__ __forceinline__ void
load_scaled_wu_panel(SmemTile<WU_PANEL_COLS> panel, const __nv_bfloat16* __restrict__ input_row0,
                     std::int64_t input_row_stride, const float* __restrict__ scale, int panel_col, int tid)
{
    constexpr int VEC_PER_ROW = WU_PANEL_COLS / 4;
    constexpr int N_VEC = BT * VEC_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VEC; v += BLOCK_THREADS) {
        const int row = v / VEC_PER_ROW;
        const int col4 = (v - row * VEC_PER_ROW) * 4;
        const ptx::Bf16x4Pack packed =
            ptx::load_vec<ptx::Bf16x4Pack>(input_row0 + (std::int64_t)row * input_row_stride + panel_col + col4);
        const float2 lo = ptx::bf16x2_to_float2(packed.pair[0]);
        const float2 hi = ptx::bf16x2_to_float2(packed.pair[1]);
        const float s = scale[row];
        panel.vec4_at(row, col4) = make_float4(lo.x * s, lo.y * s, hi.x * s, hi.y * s);
    }
}

template <int WU_PANEL_COLS, int BLOCK_THREADS>
__device__ __forceinline__ void
load_scaled_wu_panel_from_smem(SmemTile<WU_PANEL_COLS> panel, Bf16SmemTile<WU_PANEL_COLS> input,
                               const float* __restrict__ scale, int tid)
{
    constexpr int VEC_PER_ROW = WU_PANEL_COLS / 4;
    constexpr int N_VEC = BT * VEC_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VEC; v += BLOCK_THREADS) {
        const int row = v / VEC_PER_ROW;
        const int col4 = (v - row * VEC_PER_ROW) * 4;
        const ptx::Bf16x4Pack packed = ptx::load_vec<ptx::Bf16x4Pack>(input.ptr(row, col4));
        const float2 lo = ptx::bf16x2_to_float2(packed.pair[0]);
        const float2 hi = ptx::bf16x2_to_float2(packed.pair[1]);
        const float s = scale[row];
        panel.vec4_at(row, col4) = make_float4(lo.x * s, lo.y * s, hi.x * s, hi.y * s);
    }
}

template <bool InterleaveOutput, int WU_PANEL_COLS, int BLOCK_WARPS>
__device__ __forceinline__ void
compute_store_wu_panel(SmemTile<BT> T_view, SmemTile<WU_PANEL_COLS> panel, __nv_bfloat16* __restrict__ output_smem,
                       __nv_bfloat16* __restrict__ output_row0, std::int64_t output_row_stride, int panel_col,
                       int warp, int lane)
{
    static_assert(BLOCK_WARPS == 4 || BLOCK_WARPS == 8);
    constexpr int WARPS_PER_ROW = BLOCK_WARPS / N_SUB;
    constexpr int WARP_PANEL_COLS = WU_PANEL_COLS / WARPS_PER_ROW;
    constexpr int WU_N_TILES = WARP_PANEL_COLS / MMA_N;

    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    const int row_tile = warp / WARPS_PER_ROW;
    const int warp_panel_col = (warp - row_tile * WARPS_PER_ROW) * WARP_PANEL_COLS;
    const int row_g0 = row_tile * MMA_M + lane_g;
    const int row_g1 = row_g0 + 8;
    const int col_pair = lane_t << 1;

    float D[WU_N_TILES][4] = {};
#pragma unroll
    for (int k_tile = 0; k_tile < N_K_TILES; ++k_tile) {
        const int k_off = k_tile * MMA_K;
        const int col_t0 = k_off + lane_t;
        const int col_t1 = col_t0 + 4;
        const float a0 = T_view.at(row_g0, col_t0);
        const float a1 = T_view.at(row_g1, col_t0);
        const float a2 = T_view.at(row_g0, col_t1);
        const float a3 = T_view.at(row_g1, col_t1);

        const int row_t0 = k_off + lane_t;
        const int row_t1 = row_t0 + 4;
#pragma unroll
        for (int n = 0; n < WU_N_TILES; ++n) {
            const int col = warp_panel_col + n * MMA_N + lane_g;
            const float b0 = panel.at(row_t0, col);
            const float b1 = panel.at(row_t1, col);
            mma_tf32(D[n][0], D[n][1], D[n][2], D[n][3], a0, a1, a2, a3, b0, b1);
        }
    }

    __nv_bfloat16* const warp_output = output_smem + warp * MMA_M * WARP_PANEL_COLS;
#pragma unroll
    for (int n = 0; n < WU_N_TILES; ++n) {
        const int col = n * MMA_N + col_pair;
        store_vec(&warp_output[lane_g * WARP_PANEL_COLS + wu_output_swizzled_col<WARP_PANEL_COLS>(lane_g, col)],
                  pack_bf16x2_rn(D[n][0], D[n][1]));
        store_vec(&warp_output[(lane_g + 8) * WARP_PANEL_COLS +
                               wu_output_swizzled_col<WARP_PANEL_COLS>(lane_g + 8, col)],
                  pack_bf16x2_rn(D[n][2], D[n][3]));
    }
    __syncwarp();

    constexpr int STORE_ELEMS = 8;
    constexpr int STORE_PER_ROW = WARP_PANEL_COLS / STORE_ELEMS;
    constexpr int STORE_PER_WARP = MMA_M * STORE_PER_ROW;
#pragma unroll
    for (int v = lane; v < STORE_PER_WARP; v += ptx::kWarpSize) {
        const int row = v / STORE_PER_ROW;
        const int col8 = (v - row * STORE_PER_ROW) * STORE_ELEMS;
        uint4 packed = ptx::load_vec<uint4>(&warp_output[row * WARP_PANEL_COLS +
                                                      wu_output_swizzled_col<WARP_PANEL_COLS>(row, col8)]);
        if constexpr (InterleaveOutput) {
            // W is a private prepare -> state-passing workspace. Store every group of eight columns
            // as {0,4,1,5,2,6,3,7}, so the consumer's native-BF16 ldmatrix.x2 directly produces its
            // TF32 A fragment.
            packed = {
                (packed.x & 0x0000ffffU) | (packed.z << 16),
                (packed.x >> 16) | (packed.z & 0xffff0000U),
                (packed.y & 0x0000ffffU) | (packed.w << 16),
                (packed.y >> 16) | (packed.w & 0xffff0000U),
            };
        }
        store_vec(output_row0 + (std::int64_t)(row_tile * MMA_M + row) * output_row_stride + panel_col +
                      warp_panel_col + col8,
                  packed);
    }
}

// One CTA per (64-token chunk, v-head). The wide route (H_v = 48) runs 8 warps / 28.75 KiB
// dynamic smem for 3 CTAs/SM; the narrow one (H_v = 32) 4 warps / 23.75 KiB for 4 CTAs/SM. Both
// retain a 16 KiB FP32 T_inv and 0.75 KiB of controls, and T_inv never crosses HBM.
//
// Deliberately no __launch_bounds__: without one nvcc caps the kernel at 64 registers/thread,
// which is what lets three 256-thread CTAs share the 64K register file next to 28.75 KiB of smem
// each. A launch bound would let it take more registers and drop to one CTA/SM.
template <int K_PANEL_COLS, int WU_PANEL_COLS, int BLOCK_WARPS>
__global__ void prepare_wy_wu_kernel(const __nv_bfloat16* __restrict__ k_in, const __nv_bfloat16* __restrict__ v_in,
                                     const float* __restrict__ g_in, const __nv_bfloat16* __restrict__ beta_in,
                                     __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ U,
                                     float* __restrict__ g_cumsum_out, ChunkGeom geom)
{
    static_assert(BLOCK_WARPS == 4 || BLOCK_WARPS == 8);
    static_assert(BLOCK_WARPS % N_SUB == 0);
    static_assert(WU_PANEL_COLS % (BLOCK_WARPS / N_SUB) == 0);
    using dims = kernel_dims<K_PANEL_COLS, WU_PANEL_COLS>;
    constexpr int BLOCK_THREADS = BLOCK_WARPS * ptx::kWarpSize;
    constexpr int N_K_PANELS = dims::N_K_PANELS;
    constexpr int N_WU_PANELS = dims::N_WU_PANELS;

    extern __shared__ float smem[];
    float* const T_inv_smem = smem;
    float* const stage_smem = smem + dims::T_inv_floats;
    auto* const output_smem = reinterpret_cast<__nv_bfloat16*>(stage_smem + dims::stage_floats);
    float* const g_smem = stage_smem + dims::stage_floats + dims::output_stage_floats;
    float* const beta_smem = g_smem + BT;
    float* const bg_smem = beta_smem + BT;

    // stage_smem aliases BF16 K (WY-B), FP32 Schur scratch (WY-D), and one pre-scaled FP32 V/K
    // panel (WU). All handovers are block-barrier guarded.
    auto* const K_smem = reinterpret_cast<__nv_bfloat16*>(stage_smem);
    float* const scr_smem = stage_smem;

    SmemTile<BT> M_view{T_inv_smem};
    SmemTile<BT> T_view{T_inv_smem};
    Bf16SmemTile<K_PANEL_COLS> K_view{K_smem};
    SmemTile<WU_PANEL_COLS> WU_view{stage_smem};

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (ptx::kWarpSize - 1);
    const int warp = tid / ptx::kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const int chunk = static_cast<int>(blockIdx.x);
    const int h_v = static_cast<int>(blockIdx.y);
    const std::int64_t cs = static_cast<std::int64_t>(chunk) * BT;
    const std::int64_t H_v = geom.H_v;
    const std::int64_t qk_row_stride = geom.qk_row_stride;
    const std::int64_t v_row_stride = geom.v_row_stride;

    // === Phase WY-A: cooperative load of beta + warp-0 in-place scan of g ===
    //
    // beta load: 64 threads (warp 0 + 1) load BT entries.
    // g scan: only warp 0 participates -- each lane handles 2 consecutive tokens (a[2L], a[2L+1])
    // of g_in for this (chunk, h_v), runs a Hillis-Steele inclusive scan via shfl, and writes both
    // g_smem[] (consumed by Phase WY-C / WU-A) AND HBM g_cumsum_out (consumed by stages 2/3
    // unchanged). This folds the standalone g-cumsum kernel into here at zero added latency: the
    // scan is hidden behind Phase WY-B's K load + KKT mma.
    //
    // No __syncthreads here -- the per-chunk K loader below issues one before any read of
    // g_smem / beta_smem can race the scan stores.
    if (tid < BT) {
        const int64_t boff = (cs + tid) * H_v + h_v;
        beta_smem[tid] = __bfloat162float(beta_in[boff]);
    }

    if (warp == 0) {
        const int64_t g_row_base = cs * H_v + h_v;
        const int t0 = 2 * lane;  // 0, 2, ..., 62
        const int t1 = t0 + 1;    // 1, 3, ..., 63

        const float a = g_in[g_row_base + (int64_t)t0 * H_v];
        const float bv = g_in[g_row_base + (int64_t)t1 * H_v];

        // Hillis-Steele inclusive scan over per-lane partials (a + bv).
        float partial = a + bv;
#pragma unroll
        for (int o = 1; o < ptx::kWarpSize; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, partial, o);
            if (lane >= o) partial += n;
        }
        // Inclusive -> exclusive shift: lane 0's prefix is 0.
        const float prev_inc = __shfl_up_sync(0xffffffffu, partial, 1);
        const float ex_prefix = (lane == 0) ? 0.0f : prev_inc;

        // c_t = sum_{i <= t} g_i: the INCLUSIVE chunk cumsum. Every later stage indexes it this
        // way (2^{G_t} is the chunk-local decay applied through token t), so an off-by-one here
        // would rescale the whole head by one token's decay.
        const float c0 = ex_prefix + a;
        const float c1 = c0 + bv;

        g_smem[t0] = c0;
        g_smem[t1] = c1;
        g_cumsum_out[g_row_base + (int64_t)t0 * H_v] = c0;
        g_cumsum_out[g_row_base + (int64_t)t1 * H_v] = c1;
    }

    // === Phase WY-B: native BF16 KKT on the lower-tri 4x4 sub-block grid ===
    //
    // K is a represented BF16 Op input. Keeping it BF16 through shared memory and m16n8k16 changes
    // no operand precision relative to a BF16->FP32->TF32 path: every BF16 value is exactly
    // representable in TF32. The KKT accumulator remains FP32.
    float A_reg[N_SUB][8] = {};

    const int64_t k_base = cs * qk_row_stride + static_cast<int64_t>(geom.qk_head(h_v)) * kStateDim;
    const int64_t v_base = cs * v_row_stride + geom.v_row_off + static_cast<int64_t>(h_v) * kStateDim;
    constexpr int K_VECS_PER_ROW = K_PANEL_COLS / 8;
    constexpr int K_STAGE_VECS = BT * K_VECS_PER_ROW;

    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;

    auto kkt_strip = [&]<int N_OWNED>() {
#pragma unroll
        for (int k_tile = 0; k_tile < dims::K_TILES_PER_PANEL; ++k_tile) {
            const int k_off = k_tile * BF16_MMA_K;
            unsigned af[4];
            ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(K_view.ptr(warp * BC + a_rowoff, k_off + a_coloff)));

#pragma unroll
            for (int j_sub = 0; j_sub < N_OWNED; ++j_sub) {
#pragma unroll
                for (int n_tile = 0; n_tile < 2; ++n_tile) {
                    const int row_b = j_sub * BC + n_tile * MMA_N + b_rin;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1], smem_addr(K_view.ptr(row_b, k_off + b_koff)));
                    mma_bf16(A_reg[j_sub][n_tile * 4 + 0], A_reg[j_sub][n_tile * 4 + 1], A_reg[j_sub][n_tile * 4 + 2],
                             A_reg[j_sub][n_tile * 4 + 3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
                }
            }
        }
    };

#pragma unroll
    for (int kp = 0; kp < N_K_PANELS; ++kp) {
        const int panel_col = kp * K_PANEL_COLS;

#pragma unroll
        for (int v = tid; v < K_STAGE_VECS; v += BLOCK_THREADS) {
            const int row = v / K_VECS_PER_ROW;
            const int col8 = (v - row * K_VECS_PER_ROW) * 8;
            const __nv_bfloat16* src = k_in + k_base + (int64_t)row * qk_row_stride + panel_col + col8;
            cp_async<16, Cache::cg>(K_view.ptr(row, col8), src);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        switch (warp) {
        case 0: kkt_strip.template operator()<1>(); break;
        case 1: kkt_strip.template operator()<2>(); break;
        case 2: kkt_strip.template operator()<3>(); break;
        case 3: kkt_strip.template operator()<4>(); break;
        }

        // The wide route's four W/U helper warps would otherwise wait for the triangular KKT work.
        // Use that interval to stage the first BF16 V panel in output_smem; scaling stays FP32.
        if constexpr (BLOCK_WARPS == 8) {
            if (kp == 0 && warp >= WY_WARPS) {
                constexpr int HELPER_THREADS = (BLOCK_WARPS - WY_WARPS) * ptx::kWarpSize;
                constexpr int VECS_PER_ROW = WU_PANEL_COLS / 8;
                constexpr int N_VECS = BT * VECS_PER_ROW;
                const int helper_tid = tid - WY_WARPS * ptx::kWarpSize;
                Bf16SmemTile<WU_PANEL_COLS> preload{output_smem};
#pragma unroll
                for (int v = helper_tid; v < N_VECS; v += HELPER_THREADS) {
                    const int row = v / VECS_PER_ROW;
                    const int col8 = (v - row * VECS_PER_ROW) * 8;
                    cp_async<16, Cache::cg>(preload.ptr(row, col8),
                                            v_in + v_base + static_cast<int64_t>(row) * v_row_stride + col8);
                }
                cp_commit();
                cp_wait<0>();
            }
        }

        if (kp + 1 < N_K_PANELS) __syncthreads();
    }

    if (warp < WY_WARPS) {
        const int r_g0 = warp * BC + lane_g;
        const int r_g1 = r_g0 + 8;
        const float beta_r0 = beta_smem[r_g0];
        const float beta_r1 = beta_smem[r_g1];
        const float g_r0 = g_smem[r_g0];
        const float g_r1 = g_smem[r_g1];
        const float nbeta_r0 = -beta_r0;
        const float nbeta_r1 = -beta_r1;

#pragma unroll
        for (int j_sub = 0; j_sub < N_SUB; ++j_sub) {
            if (j_sub > warp) continue;

            const int c_base = j_sub * BC + 2 * lane_t;
            const int c0 = c_base;
            const int c1 = c_base + 1;
            const int c2 = c_base + 8;
            const int c3 = c_base + 9;

            const float g_c0 = g_smem[c0];
            const float g_c1 = g_smem[c1];
            const float g_c2 = g_smem[c2];
            const float g_c3 = g_smem[c3];

            const bool is_diag = (j_sub == warp);

            // Diagonal block: keep strict lower triangle (r > c), zero the rest. Off-diagonal
            // blocks (j_sub < warp): keep all entries.
            //
            // Footgun: the cumsum is monotone decreasing, so on the diagonal block's strict-upper
            // triangle (r < c) g_r - g_c is large positive and exp can saturate to +inf. A
            // `* mask` (mask = 0/1 float) would then produce inf * 0 = NaN. The conditional select
            // below overwrites the bad product with 0.0f instead, so no NaN escapes (same pattern
            // as the output stage).
            A_reg[j_sub][0] = (!is_diag || r_g0 > c0) ? nbeta_r0 * A_reg[j_sub][0] * expf(g_r0 - g_c0) : 0.0f;
            A_reg[j_sub][1] = (!is_diag || r_g0 > c1) ? nbeta_r0 * A_reg[j_sub][1] * expf(g_r0 - g_c1) : 0.0f;
            A_reg[j_sub][2] = (!is_diag || r_g1 > c0) ? nbeta_r1 * A_reg[j_sub][2] * expf(g_r1 - g_c0) : 0.0f;
            A_reg[j_sub][3] = (!is_diag || r_g1 > c1) ? nbeta_r1 * A_reg[j_sub][3] * expf(g_r1 - g_c1) : 0.0f;
            A_reg[j_sub][4] = (!is_diag || r_g0 > c2) ? nbeta_r0 * A_reg[j_sub][4] * expf(g_r0 - g_c2) : 0.0f;
            A_reg[j_sub][5] = (!is_diag || r_g0 > c3) ? nbeta_r0 * A_reg[j_sub][5] * expf(g_r0 - g_c3) : 0.0f;
            A_reg[j_sub][6] = (!is_diag || r_g1 > c2) ? nbeta_r1 * A_reg[j_sub][6] * expf(g_r1 - g_c2) : 0.0f;
            A_reg[j_sub][7] = (!is_diag || r_g1 > c3) ? nbeta_r1 * A_reg[j_sub][7] * expf(g_r1 - g_c3) : 0.0f;
        }
    }

    __syncthreads();

    {
        constexpr int N = BT * BT;
#pragma unroll
        for (int idx = tid; idx < N; idx += BLOCK_THREADS) T_inv_smem[idx] = 0.0f;
    }
    __syncthreads();

    switch (warp) {
    case 0: store_frag_to_M(A_reg[0], 0, 0, lane_g, lane_t, M_view); break;
    case 1: store_frag_to_M(A_reg[1], 1, 1, lane_g, lane_t, M_view); break;
    case 2: store_frag_to_M(A_reg[2], 2, 2, lane_g, lane_t, M_view); break;
    case 3: store_frag_to_M(A_reg[3], 3, 3, lane_g, lane_t, M_view); break;
    }
    __syncwarp();

    switch (warp) {
    case 0: solve_diag_block<0>(lane, M_view); break;
    case 1: solve_diag_block<1>(lane, M_view); break;
    case 2: solve_diag_block<2>(lane, M_view); break;
    case 3: solve_diag_block<3>(lane, M_view); break;
    }
    __syncthreads();

    // === Phase WY-D: block-Schur off-diagonal completion (3 waves) ===
    // Switch on `warp` so (MY_W, MY_J) are compile-time per case.
    auto wave_compute_store = [&]<int MY_W, int MY_J>() {
        float out[8];
        compute_off_diag<MY_W, MY_J>(out, warp, lane, A_reg, scr_smem, M_view);
        store_frag_to_M(out, MY_W, MY_J, lane_g, lane_t, M_view);
    };

    switch (warp) {
    case 1: wave_compute_store.template operator()<1, 0>(); break;
    case 2: wave_compute_store.template operator()<2, 1>(); break;
    case 3: wave_compute_store.template operator()<3, 2>(); break;
    }
    __syncthreads();

    switch (warp) {
    case 2: wave_compute_store.template operator()<2, 0>(); break;
    case 3: wave_compute_store.template operator()<3, 1>(); break;
    }
    __syncthreads();

    if (warp == 3) wave_compute_store.template operator()<3, 0>();
    __syncthreads();

    // === Phase WY-E: +I on the diagonal of T_inv (no HBM write; sync fused into WU-A) ===
    if (tid < BT) M_view.at(tid, tid) += 1.0f;

    // W/U preserve the BF16->FP32->FP32-scale->TF32 precision path. In particular T_inv and the
    // scaled V/K are never down-cast to BF16.
    if (tid < BT) bg_smem[tid] = beta_smem[tid] * expf(g_smem[tid]);

    const int64_t out_base = cs * H_v * kStateDim + static_cast<int64_t>(h_v) * kStateDim;
    const int64_t out_row_stride = H_v * kStateDim;

    // === Phase WU-A/B: U = T_inv @ (beta * V), one 64x32 FP32 panel at a time ===
#pragma unroll
    for (int panel = 0; panel < N_WU_PANELS; ++panel) {
        if (panel != 0) __syncthreads();
        const int panel_col = panel * WU_PANEL_COLS;
        // The wide route's helper warps preloaded V panel 0 into output_smem during the KKT phase,
        // so panel 0 is scaled straight out of smem; the narrow route has no helper warps and reads
        // every panel from global memory.
        if constexpr (BLOCK_WARPS == 8) {
            if (panel == 0) {
                load_scaled_wu_panel_from_smem<WU_PANEL_COLS, BLOCK_THREADS>(
                    WU_view, Bf16SmemTile<WU_PANEL_COLS>{output_smem}, beta_smem, tid);
            } else {
                load_scaled_wu_panel<WU_PANEL_COLS, BLOCK_THREADS>(WU_view, v_in + v_base, v_row_stride,
                                                                   beta_smem, panel_col, tid);
            }
        } else {
            load_scaled_wu_panel<WU_PANEL_COLS, BLOCK_THREADS>(WU_view, v_in + v_base, v_row_stride, beta_smem,
                                                               panel_col, tid);
        }
        __syncthreads();
        compute_store_wu_panel<false, WU_PANEL_COLS, BLOCK_WARPS>(T_view, WU_view, output_smem, U + out_base,
                                                                    out_row_stride, panel_col, warp, lane);
    }

    // === Phase WU-C/D: W = T_inv @ (beta * exp(g) * K), same FP32 panel path ===
    __syncthreads();
#pragma unroll
    for (int panel = 0; panel < N_WU_PANELS; ++panel) {
        if (panel != 0) __syncthreads();
        const int panel_col = panel * WU_PANEL_COLS;
        load_scaled_wu_panel<WU_PANEL_COLS, BLOCK_THREADS>(WU_view, k_in + k_base, qk_row_stride, bg_smem,
                                                           panel_col, tid);
        __syncthreads();
        compute_store_wu_panel<true, WU_PANEL_COLS, BLOCK_WARPS>(T_view, WU_view, output_smem, W + out_base,
                                                                   out_row_stride, panel_col, warp, lane);
    }
}

} // namespace prepare_wy_wu

// ---------------------------------------------------------------------------
// Stage 2: chunk-sequential state passing.
// ---------------------------------------------------------------------------
namespace state_passing {

using ptx::cp_commit;
using ptx::cp_wait;
using ptx::exp2_approx;
using ptx::kLog2E;
using ptx::kWarpSize;
using ptx::ldmatrix_x2;
using ptx::ldmatrix_x2_t;
using ptx::load_ldg;
using ptx::mma_tf32_bits;
using ptx::pack_bf16x2_rn;
using ptx::pack_bf16x2_rz;
using ptx::smem_addr;
using ptx::store_vec;

static_assert(kChunkSize == 64, "state_passing: kChunkSize must be 64");
static_assert(kStateDim == 128);

template <int NStrip>
struct kernel_dims;

template <>
struct kernel_dims<16> {
    static constexpr int N_STRIP_PER_BLOCK = 16;
    static constexpr int D_STRIPS = kStateDim / N_STRIP_PER_BLOCK;
    static constexpr int DT_TILES_PER_BLOCK = N_STRIP_PER_BLOCK / MMA_N;
    static constexpr int BT_SPLITS = 4;
    static constexpr int N_WARPS = DT_TILES_PER_BLOCK * BT_SPLITS;  // 8
    static constexpr int THREADS = N_WARPS * kWarpSize;
    static constexpr int MIN_BLOCKS = 2;
};

template <>
struct kernel_dims<32> {
    static constexpr int N_STRIP_PER_BLOCK = 32;
    static constexpr int D_STRIPS = kStateDim / N_STRIP_PER_BLOCK;
    static constexpr int DT_TILES_PER_BLOCK = N_STRIP_PER_BLOCK / MMA_N;
    static constexpr int BT_SPLITS = 4;
    static constexpr int N_WARPS = DT_TILES_PER_BLOCK * BT_SPLITS;  // 16
    static constexpr int THREADS = N_WARPS * kWarpSize;
    static constexpr int MIN_BLOCKS = 1;
};

template <int NStrip>
struct smem_layout
{
    using D = kernel_dims<NStrip>;
    static constexpr int W_STRIDE = kStateDim;
    static constexpr int W_LOAD_BF16 = BT * W_STRIDE;
    static constexpr int W_STORAGE_FLT = W_LOAD_BF16 / 2;
    static constexpr int UVD_FLT = BT * D::N_STRIP_PER_BLOCK;  // U-vd alias
    static constexpr int K_LOAD_ROWS = BT;
    static constexpr int K_LOAD_BF16 = K_LOAD_ROWS * kStateDim;
    static constexpr int K_STORAGE_FLT = K_LOAD_BF16 / 2;
    static constexpr int M_TILES_H_PW = (kStateDim / D::BT_SPLITS) / MMA_M;
    static constexpr int SNAP_K_ROWS = MMA_M * M_TILES_H_PW;
    static constexpr int SNAP_FLT = SNAP_K_ROWS * D::N_STRIP_PER_BLOCK;
    static constexpr int M_TILES_H_GLOB = kStateDim / MMA_M;
    static constexpr int N_SNAP_ITERS = M_TILES_H_GLOB / M_TILES_H_PW;
    // One buffer per sit so all warps can scatter their owned h_frag once at chunk start (Phase A)
    // and then read from any sit's buffer in the unified matmul1 / coop_write paths without a
    // per-sit re-scatter+sync.
    static constexpr int N_SNAP_BUF = N_SNAP_ITERS;
    static constexpr int SMEM_FLOATS = W_STORAGE_FLT + UVD_FLT + K_STORAGE_FLT + SNAP_FLT * N_SNAP_BUF + BT;
};

// Snapshot stores the FP32 state operand as [value, state] with a stride-32 swizzle shared by the
// scatter, cooperative h_chunk write, and MM1 LDSM read.
struct SnapView
{
    float* __restrict__ base;
    static constexpr int kStride = kStateDim / 4;
    static_assert(kStride == 32);

    __device__ __forceinline__ int swz_xor(int row) const { return (row & 7) << 2; }

    __device__ __forceinline__ float& at(int row, int col) const { return base[row * kStride + (col ^ swz_xor(row))]; }

    __device__ __forceinline__ float4& vec4_at(int row, int col) const
    {
        return *reinterpret_cast<float4*>(&base[row * kStride + (col ^ swz_xor(row))]);
    }
};

// prepare_wy_wu stores W's private workspace in the native-BF16 fragment order {0,4,1,5,2,6,3,7}
// within each group of eight logical columns.
struct Bf16WView
{
    __nv_bfloat16* __restrict__ base;

    __device__ __forceinline__ int swizzled_col(int row, int col) const
    {
        return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
    }

    __device__ __forceinline__ __nv_bfloat16* ptr(int row, int col) const
    {
        return &base[row * kStateDim + swizzled_col(row, col)];
    }
};

template <int THREADS>
__device__ __forceinline__ void issue_load_w_bf16(Bf16WView view, const __nv_bfloat16* gmem_base_row0,
                                                  int64_t gmem_row_stride, int tid)
{
    constexpr int ELEMS_PER_COPY = 8;
    constexpr int COPIES_PER_ROW = kStateDim / ELEMS_PER_COPY;
    constexpr int COPIES = BT * COPIES_PER_ROW;
#pragma unroll
    for (int copy = tid; copy < COPIES; copy += THREADS) {
        const int row = copy / COPIES_PER_ROW;
        const int col = (copy - row * COPIES_PER_ROW) * ELEMS_PER_COPY;
        ptx::cp_async<16>(view.ptr(row, col), gmem_base_row0 + static_cast<int64_t>(row) * gmem_row_stride + col);
    }
}

// K is staged in native BF16 because it is already representable exactly as a TF32 operand. Within
// each group of eight token rows, physical rows are ordered {0,4,1,5,2,6,3,7}. ldmatrix.x2.trans
// then packs token columns {t,t+4} into the two halves of each register, matching the TF32 A
// fragment.
struct Bf16KView
{
    __nv_bfloat16* __restrict__ base;

    __device__ __forceinline__ int physical_row(int logical_row) const
    {
        const int row8 = logical_row & 7;
        return (logical_row & ~7) + ((row8 & 3) << 1) + (row8 >> 2);
    }

    __device__ __forceinline__ int swizzled_col(int physical_row_, int col) const
    {
        return (col & ~63) + ((((col & 63) >> 3) ^ (physical_row_ & 7)) << 3) + (col & 7);
    }

    __device__ __forceinline__ __nv_bfloat16* logical_ptr(int row, int col) const
    {
        const int row_phys = physical_row(row);
        return &base[row_phys * kStateDim + swizzled_col(row_phys, col)];
    }

    __device__ __forceinline__ __nv_bfloat16* physical_ptr(int row, int col) const
    {
        return &base[row * kStateDim + swizzled_col(row, col)];
    }
};

template <int THREADS>
__device__ __forceinline__ void issue_load_k_bf16(Bf16KView view, const __nv_bfloat16* gmem_base_row0,
                                                  int64_t gmem_row_stride, int tid)
{
    constexpr int ELEMS_PER_COPY = 8;
    constexpr int COPIES_PER_ROW = kStateDim / ELEMS_PER_COPY;
    constexpr int COPIES = BT * COPIES_PER_ROW;
#pragma unroll
    for (int copy = tid; copy < COPIES; copy += THREADS) {
        const int row = copy / COPIES_PER_ROW;
        const int col = (copy - row * COPIES_PER_ROW) * ELEMS_PER_COPY;
        ptx::cp_async<16>(view.logical_ptr(row, col),
                          gmem_base_row0 + static_cast<int64_t>(row) * gmem_row_stride + col);
    }
}

__device__ __forceinline__ void unpack_bf16x2_to_fp32_bits(unsigned packed, unsigned& low, unsigned& high)
{
    low = packed << 16;
    high = packed & 0xffff0000U;
}

// The narrow geometry targets two 128-register CTAs per SM. The wide geometry uses one 512-thread
// CTA; both expose 16 resident warps without local spills.
template <int NStrip>
__launch_bounds__(kernel_dims<NStrip>::THREADS, kernel_dims<NStrip>::MIN_BLOCKS) __global__
void state_passing_kernel(const __nv_bfloat16* __restrict__ W_in, const __nv_bfloat16* __restrict__ U_in,
                          const __nv_bfloat16* __restrict__ k_in, const float* __restrict__ g_cumsum,
                          const float* state_in, __nv_bfloat16* __restrict__ v_new,
                          __nv_bfloat16* __restrict__ h_chunk, float* state_out, ChunkGeom geom, int chunks)
{
    using D = kernel_dims<NStrip>;
    using L = smem_layout<NStrip>;
    constexpr int N_STRIP_PER_BLOCK = D::N_STRIP_PER_BLOCK;
    constexpr int D_STRIPS = D::D_STRIPS;
    constexpr int BT_SPLITS = D::BT_SPLITS;
    constexpr int THREADS_K = D::THREADS;
    constexpr int W_STRIDE = L::W_STRIDE;

    constexpr int BT_PER_WARP = BT / BT_SPLITS;
    constexpr int S_PER_WARP = kStateDim / BT_SPLITS;
    constexpr int M_TILES_MM1_PW = BT_PER_WARP / MMA_M;
    constexpr int M_TILES_H_PW = S_PER_WARP / MMA_M;
    constexpr int M_TILES_H_GLOB = kStateDim / MMA_M;
    constexpr int K_TILES_MM2 = BT / MMA_K;

    static_assert(M_TILES_H_PW >= 1, "S_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_MM1_PW >= 1, "BT_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_H_GLOB == BT_SPLITS * M_TILES_H_PW, "BT_SPLITS partition of state dim must be exact");
    static_assert(W_STRIDE >= 16, "SmemTile<W_STRIDE> requires stride >= 16");

    constexpr int SNAP_K_ROWS = L::SNAP_K_ROWS;
    constexpr int N_SNAP_ITERS = L::N_SNAP_ITERS;
    constexpr int K_TILES_PER_SNAP_ITER = SNAP_K_ROWS / MMA_K;

    static_assert(N_SNAP_ITERS == BT_SPLITS, "design assumes one snap iter per s_idx");

    constexpr int W_STORAGE_FLT = L::W_STORAGE_FLT;
    constexpr int UVD_FLT = L::UVD_FLT;
    constexpr int K_STORAGE_FLT = L::K_STORAGE_FLT;
    constexpr int SNAP_FLT = L::SNAP_FLT;
    constexpr int N_SNAP_BUF = L::N_SNAP_BUF;

    // Smem partition. U and vd alias (uvd_smem) in disjoint phases. snap is per-sit so the
    // chunk-start unified scatter feeds every sit's MM1.
    extern __shared__ float smem[];
    auto* const W_smem = reinterpret_cast<__nv_bfloat16*>(smem);  // W_LOAD_BF16
    float* const uvd_smem = smem + W_STORAGE_FLT;                  // UVD_FLT
    auto* const k_smem = reinterpret_cast<__nv_bfloat16*>(uvd_smem + UVD_FLT);
    float* const snap_smem = uvd_smem + UVD_FLT + K_STORAGE_FLT;  // SNAP_FLT * N_SNAP_BUF
    float* const g_smem = snap_smem + SNAP_FLT * N_SNAP_BUF;      // BT

    Bf16WView W_view{W_smem};
    SmemTile<N_STRIP_PER_BLOCK> vd_view{uvd_smem};
    Bf16KView k_view{k_smem};
    SmemTile<N_STRIP_PER_BLOCK> U_view{uvd_smem};
    // One SnapView per sit so each owning warp scatters into a unique buffer (Phase A unified
    // scatter). The array is sized on N_SNAP_BUF (= N_SNAP_ITERS) rather than hard-coded.
    SnapView snap_views[N_SNAP_BUF];
#pragma unroll
    for (int b_ = 0; b_ < N_SNAP_BUF; ++b_) snap_views[b_] = SnapView{snap_smem + b_ * SNAP_FLT};

    // Block / lane indexing.
    //   grid.x = hd in [0, H_v*D_STRIPS).
    //   warp = dt_idx * BT_SPLITS + s_idx.
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const int s_idx = warp % BT_SPLITS;
    const int dt_idx = warp / BT_SPLITS;
    const int warp_d_local = dt_idx * MMA_N;

    const int hd = static_cast<int>(blockIdx.x);
    const int h_v = hd / D_STRIPS;
    const int strip_idx = hd - h_v * D_STRIPS;
    const int d_off = strip_idx * N_STRIP_PER_BLOCK;
    const int warp_d_global = d_off + warp_d_local;
    const std::int64_t H_v = geom.H_v;

    // === Phase 0: load state_in (transposed) -> per-warp h_frag ===
    // Read-only here; state_out is written in Phase Z, so the caller may pass the same buffer for
    // both (in-place on rec_state).
    float h_frag[M_TILES_H_PW][4];
    // NOTE the layout: helios' recurrent_state is [v_heads, v_dim, k_dim], i.e. the K index is
    // the ROW (stride v_dim) and the value index is the column - the opposite of ninfer's
    // [value][k] state. The smem snapshot keeps the ninfer orientation ([value][k], which is what
    // the ldmatrix patterns want), so only these two global accesses transpose.
    {
        const int64_t st_base = static_cast<int64_t>(h_v) * kStateDim * kStateDim;
        const int row_off = s_idx * S_PER_WARP;
#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
            const int k_g0 = row_off + m * MMA_M + lane_g;
            const int k_g1 = k_g0 + 8;
            const int d0 = warp_d_global + 2 * lane_t;
            const int d1 = d0 + 1;
            h_frag[m][0] = load_ldg<float>(state_in + st_base + (int64_t)k_g0 * kStateDim + d0);
            h_frag[m][1] = load_ldg<float>(state_in + st_base + (int64_t)k_g0 * kStateDim + d1);
            h_frag[m][2] = load_ldg<float>(state_in + st_base + (int64_t)k_g1 * kStateDim + d0);
            h_frag[m][3] = load_ldg<float>(state_in + st_base + (int64_t)k_g1 * kStateDim + d1);
        }
    }

    // Chunk 0 is staged before the loop. Later W/K async groups and converted U are issued
    // together at the preceding chunk's Phase-E boundary. All global bases advance additively
    // across the sequential chunk loop.
    const int64_t W_stride = H_v * kStateDim;
    const int64_t k_stride = geom.qk_row_stride;
    const int64_t W_chunk_stride = (int64_t)BT * W_stride;
    const int64_t k_chunk_stride = (int64_t)BT * k_stride;
    const int64_t hc_chunk_stride = H_v * kStateDim * kStateDim;
    const int64_t vn_stride = W_stride;
    const int64_t vn_chunk_stride = (int64_t)BT * vn_stride;
    const int64_t g_chunk_step = (int64_t)BT * H_v;

    const int64_t W_block_base = static_cast<int64_t>(h_v) * kStateDim;
    const int64_t k_block_base = static_cast<int64_t>(geom.qk_head(h_v)) * kStateDim;
    const int64_t hc_block_base = static_cast<int64_t>(h_v) * kStateDim * kStateDim;
    const int64_t vn_block_base = static_cast<int64_t>(h_v) * kStateDim;
    const int64_t g_block_base = h_v;
    const int64_t g_thread_base = g_block_base + (int64_t)tid * H_v;

    // W/K stay native BF16 in shared memory. W is committed first for MM1; K is the later group
    // and remains in flight until MM2. U is expanded synchronously while both async groups make
    // progress.
    {
        issue_load_w_bf16<THREADS_K>(W_view, W_in + W_block_base, W_stride, tid);
        cp_commit();
        issue_load_k_bf16<THREADS_K>(k_view, k_in + k_block_base, k_stride, tid);
        cp_commit();
        issue_load_bf16_to_float_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(U_view, U_in + W_block_base + d_off,
                                                                        W_stride, tid);
    }

    // === Main chunk loop ===
    int64_t W_base = W_block_base;
    int64_t k_base = k_block_base;
    int64_t hc_base = hc_block_base;
    int64_t vn_base = vn_block_base;
    int64_t g_cs_offset = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int64_t W_base_next = W_base + W_chunk_stride;
        const int64_t k_base_next = k_base + k_chunk_stride;

        // === Phase A: drain W + scatter h_frag to snap ===
        //
        // Every warp scatters its owned h_frag into snap_views[s_idx] BEFORE the drain sync. The
        // single sync below covers W/U/g_smem AND snap visibility, so Phase B needs no per-sit
        // scatter+sync.
        if (tid < BT) g_smem[tid] = g_cumsum[g_thread_base + g_cs_offset];

        {
            SnapView snap = snap_views[s_idx];
#pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int k_g0 = m * MMA_M + lane_g;
                const int k_g1 = k_g0 + 8;
                const int d0 = warp_d_local + 2 * lane_t;
                const int d1 = d0 + 1;
                snap.at(d0, k_g0) = h_frag[m][0];
                snap.at(d1, k_g0) = h_frag[m][1];
                snap.at(d0, k_g1) = h_frag[m][2];
                snap.at(d1, k_g1) = h_frag[m][3];
            }
        }

        cp_wait<1>();     // drain W; leave K in flight
        __syncthreads();  // gates W/U/g_smem STS and scatter visibility

        // === Phase B: per-sit coop_write + matmul1 (no per-sit scatter sync) ===
        float vnew_frag[M_TILES_MM1_PW][4] = {};

#pragma unroll
        for (int sit = 0; sit < N_SNAP_ITERS; ++sit) {
            const int k_row_off = sit * SNAP_K_ROWS;
            SnapView snap = snap_views[sit];

            // Coop float4 gmem write of h_chunk for this snap block. No sync above: snap was
            // populated by the Phase A unified scatter, and snap[sit] is read-only from now on.
            {
                constexpr int K_VEC_PER_D = SNAP_K_ROWS / 4;
                constexpr int N_VEC_SNAP = SNAP_FLT / 4;
#pragma unroll
                for (int v = tid; v < N_VEC_SNAP; v += THREADS_K) {
                    const int d_local = v / K_VEC_PER_D;
                    const int kvec = v - d_local * K_VEC_PER_D;
                    const int k_off = kvec * 4;
                    float4 val = snap.vec4_at(d_local, k_off);
                    const int d_global = d_off + d_local;
                    __nv_bfloat16* out =
                        &h_chunk[hc_base + (int64_t)d_global * kStateDim + k_row_off + k_off];
                    store_vec(out, pack_bf16x2_rn(val.x, val.y));
                    store_vec(out + 2, pack_bf16x2_rn(val.z, val.w));
                }
            }

            // W's producer-interleaved BF16 layout lets ldmatrix.x2 form the four exact FP32/TF32 A
            // operands. The FP32 state snapshot uses ldmatrix.x2 as a b16 transport for the two B
            // operand bit patterns.
            const int lane_in_8 = lane & 7;
            const int matrix_half = (lane >> 3) & 1;
#pragma unroll
            for (int kt = 0; kt < K_TILES_PER_SNAP_ITER; ++kt) {
                const int W_k_local = k_row_off + kt * MMA_K;
                const int snap_k = kt * MMA_K;

                const int b_row = warp_d_local + lane_in_8;
                const int b_col = snap_k + matrix_half * 4;
                const unsigned b_addr = smem_addr(&snap.at(b_row, b_col));
                unsigned ub0, ub1;
                ldmatrix_x2(ub0, ub1, b_addr);

#pragma unroll
                for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
                    const int row_base = s_idx * BT_PER_WARP + m_mm1 * MMA_M;
                    const int a_row = row_base + matrix_half * 8 + lane_in_8;
                    const unsigned a_addr = smem_addr(W_view.ptr(a_row, W_k_local));
                    unsigned packed0, packed1;
                    ldmatrix_x2(packed0, packed1, a_addr);
                    unsigned ua0, ua1, ua2, ua3;
                    unpack_bf16x2_to_fp32_bits(packed0, ua0, ua2);
                    unpack_bf16x2_to_fp32_bits(packed1, ua1, ua3);

                    mma_tf32_bits(vnew_frag[m_mm1][0], vnew_frag[m_mm1][1], vnew_frag[m_mm1][2],
                                  vnew_frag[m_mm1][3], ua0, ua1, ua2, ua3, ub0, ub1);
                }
            }
        }

        // === Phase C: subtract converted FP32 U ===
#pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_local + 2 * lane_t;
            const float2 u_top = ptx::load_vec<float2>(&U_view.at(row_g0, col_d0));
            const float2 u_bot = ptx::load_vec<float2>(&U_view.at(row_g1, col_d0));
            vnew_frag[m_mm1][0] = u_top.x - vnew_frag[m_mm1][0];
            vnew_frag[m_mm1][1] = u_top.y - vnew_frag[m_mm1][1];
            vnew_frag[m_mm1][2] = u_bot.x - vnew_frag[m_mm1][2];
            vnew_frag[m_mm1][3] = u_bot.y - vnew_frag[m_mm1][3];
        }

        // === Phase D: STG vnew (UNDECAYED), STS v_decay -> vd_view, scale h_frag ===
        const float g_C = g_smem[BT - 1];
        float gamma_C = 0.0f;
        if (lane == 0) gamma_C = exp2_approx(g_C * kLog2E);
        gamma_C = __shfl_sync(0xffffffffU, gamma_C, 0);

#pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_global + 2 * lane_t;

            float dec_top = 0.0f;
            float dec_bot = 0.0f;
            if (lane_t == 0) {
                dec_top = exp2_approx((g_C - g_smem[row_g0]) * kLog2E);
                dec_bot = exp2_approx((g_C - g_smem[row_g1]) * kLog2E);
            }
            const int decay_src_lane = lane & ~3;
            dec_top = __shfl_sync(0xffffffffU, dec_top, decay_src_lane);
            dec_bot = __shfl_sync(0xffffffffU, dec_bot, decay_src_lane);

            const float v0 = vnew_frag[m_mm1][0];
            const float v1 = vnew_frag[m_mm1][1];
            const float v2 = vnew_frag[m_mm1][2];
            const float v3 = vnew_frag[m_mm1][3];

            store_vec(&v_new[vn_base + (int64_t)row_g0 * vn_stride + col_d0], pack_bf16x2_rn(v0, v1));
            store_vec(&v_new[vn_base + (int64_t)row_g1 * vn_stride + col_d0], pack_bf16x2_rn(v2, v3));

            const int col_d0_loc = warp_d_local + 2 * lane_t;
            store_vec(&vd_view.at(row_g0, col_d0_loc), make_float2(v0 * dec_top, v1 * dec_top));
            store_vec(&vd_view.at(row_g1, col_d0_loc), make_float2(v2 * dec_bot, v3 * dec_bot));
        }

#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
#pragma unroll
            for (int e = 0; e < 4; ++e) h_frag[m][e] *= gamma_C;
        }

        // Drain native-BF16 K, then make K and Phase-D v_decay visible to MM2.
        cp_wait<0>();
        __syncthreads();

        // === Phase E: matmul2 over the full chunk of k ===
#pragma unroll
        for (int kt = 0; kt < K_TILES_MM2; ++kt) {
            const int k_off_local = kt * MMA_K;

            const int row_t0 = k_off_local + lane_t;
            const int row_t1 = row_t0 + 4;
            const int col_g = warp_d_local + lane_g;
            const float b0 = vd_view.at(row_t0, col_g);
            const float b1 = vd_view.at(row_t1, col_g);

#pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int col_a_base = s_idx * S_PER_WARP + m * MMA_M;
                const int a_row = k_off_local + (lane & 7);
                const int a_col = col_a_base + ((lane >> 3) & 1) * 8;
                unsigned packed0, packed1;
                ldmatrix_x2_t(packed0, packed1, smem_addr(k_view.physical_ptr(a_row, a_col)));
                unsigned ua0, ua1, ua2, ua3;
                unpack_bf16x2_to_fp32_bits(packed0, ua0, ua2);
                unpack_bf16x2_to_fp32_bits(packed1, ua1, ua3);

                mma_tf32_bits(h_frag[m][0], h_frag[m][1], h_frag[m][2], h_frag[m][3], ua0, ua1, ua2, ua3,
                              __float_as_uint(b0), __float_as_uint(b1));
            }
        }

        __syncthreads();  // gates MM2 before the next chunk overwrites W/K/U

        // The next chunk repeats the W-then-K async group order used by the prologue. Its Phase A
        // drains W only; Phase E drains K.
        if (chunk + 1 < chunks) {
            issue_load_w_bf16<THREADS_K>(W_view, W_in + W_base_next, W_stride, tid);
            cp_commit();
            issue_load_k_bf16<THREADS_K>(k_view, k_in + k_base_next, k_stride, tid);
            cp_commit();
            issue_load_bf16_to_float_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(U_view,
                                                                            U_in + W_base_next + d_off, W_stride,
                                                                            tid);
        }

        // Advance loop-carried bases for the next chunk.
        W_base = W_base_next;
        k_base = k_base_next;
        hc_base += hc_chunk_stride;
        vn_base += vn_chunk_stride;
        g_cs_offset += g_chunk_step;
    }

    // === Phase Z: store h_frag -> state_out (transposed to helios' [k][v] layout, see Phase 0) ===
    const int64_t st_base = static_cast<int64_t>(h_v) * kStateDim * kStateDim;

#pragma unroll
    for (int m = 0; m < M_TILES_H_PW; ++m) {
        const int k_g0 = s_idx * S_PER_WARP + m * MMA_M + lane_g;
        const int k_g1 = k_g0 + 8;
        const int d0 = warp_d_global + 2 * lane_t;
        const int d1 = d0 + 1;
        state_out[st_base + (int64_t)k_g0 * kStateDim + d0] = h_frag[m][0];
        state_out[st_base + (int64_t)k_g0 * kStateDim + d1] = h_frag[m][1];
        state_out[st_base + (int64_t)k_g1 * kStateDim + d0] = h_frag[m][2];
        state_out[st_base + (int64_t)k_g1 * kStateDim + d1] = h_frag[m][3];
    }
}

} // namespace state_passing

// ---------------------------------------------------------------------------
// Stage 3: per-chunk output.
// ---------------------------------------------------------------------------
namespace output {

using ptx::Cache;
using ptx::cp_async;
using ptx::cp_commit;
using ptx::cp_wait;
using ptx::exp2_approx;
using ptx::kLog2E;
using ptx::kWarpSize;
using ptx::ldmatrix_x2;
using ptx::ldmatrix_x4;
using ptx::mma_bf16;
using ptx::mma_tf32;
using ptx::pack_bf16x2_rn;
using ptx::pack_bf16x2_rz;
using ptx::smem_addr;
using ptx::store_vec;

static_assert(kChunkSize == 64, "output: kChunkSize must be 64 (BT = 64 is hard-coded)");
static_assert(kStateDim == 128);

constexpr int N_WARPS = 4;
constexpr int THREADS = N_WARPS * kWarpSize;

static_assert(BT == N_WARPS * MMA_M, "kernel assigns one 16-row strip per warp; BT must equal N_WARPS * MMA_M");

constexpr int BF16_MMA_K = 16;
constexpr int K_PANEL = 32;
constexpr int D_PANEL = 16;
constexpr int N_K_PANELS = kStateDim / K_PANEL;
constexpr int N_D_PANELS = kStateDim / D_PANEL;
constexpr int N_TILES_BT = BT / MMA_N;
constexpr int K_TILES_BT = BT / MMA_K;

static_assert(K_PANEL % BF16_MMA_K == 0);
static_assert(D_PANEL % MMA_N == 0);

struct kernel_dims
{
    static constexpr int Q_BF16 = BT * kStateDim;
    static constexpr int K_PANEL_BF16 = BT * K_PANEL;
    static constexpr int H_PANEL_BF16 = D_PANEL * kStateDim;
    static constexpr int V_PANEL_BF16 = BT * D_PANEL;
    static constexpr int STAGE_BF16 = K_PANEL_BF16 > H_PANEL_BF16 ? K_PANEL_BF16 : H_PANEL_BF16;
    static_assert(STAGE_BF16 >= V_PANEL_BF16);

    static constexpr int BF16_SMEM_ELEMS = Q_BF16 + 2 * STAGE_BF16;
    static constexpr int SMEM_BYTES = BF16_SMEM_ELEMS * static_cast<int>(sizeof(__nv_bfloat16));
};

template <int STRIDE>
struct Bf16SmemTile
{
    __nv_bfloat16* __restrict__ base;
    static_assert(STRIDE == 16 || STRIDE == 32 || STRIDE == 128);

    __device__ __forceinline__ int swizzled_col(int row, int col) const
    {
        return col ^ ((row & (STRIDE / 8 - 1)) << 3);
    }

    __device__ __forceinline__ __nv_bfloat16* ptr(int row, int col) const
    {
        return base + row * STRIDE + swizzled_col(row, col);
    }
};

template <int ROWS, int COLS, int BLOCK_THREADS>
__device__ __forceinline__ void issue_cp_bf16(Bf16SmemTile<COLS> dst, const __nv_bfloat16* __restrict__ src_row0,
                                              std::int64_t src_row_stride, int tid)
{
    static_assert(COLS % 8 == 0);
    constexpr int VECS_PER_ROW = COLS / 8;
    constexpr int N_VECS = ROWS * VECS_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VECS; v += BLOCK_THREADS) {
        const int row = v / VECS_PER_ROW;
        const int col8 = (v - row * VECS_PER_ROW) * 8;
        ptx::cp_async<16, Cache::cg>(dst.ptr(row, col8),
                                     src_row0 + static_cast<std::int64_t>(row) * src_row_stride + col8);
    }
}

template <int N_TILES, int K_TILES, int A_STRIDE, int B_STRIDE>
__device__ __forceinline__ void mma_bf16_panel(float (&D)[N_TILES][4], Bf16SmemTile<A_STRIDE> A,
                                               Bf16SmemTile<B_STRIDE> B, int a_row_base, int a_col_base,
                                               int b_col_base, int lane)
{
    const int lane_in_8 = lane & 7;
    const int lane_q = lane >> 3;
    const int a_row = a_row_base + lane_in_8 + ((lane_q & 1) << 3);
    const int a_col = a_col_base + ((lane_q >> 1) << 3);
    const int b_col = b_col_base + ((lane_q & 1) << 3);

#pragma unroll
    for (int kt = 0; kt < K_TILES; ++kt) {
        const int k_off = kt * BF16_MMA_K;
        unsigned af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(A.ptr(a_row, a_col + k_off)));

#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            unsigned bf[2];
            ldmatrix_x2(bf[0], bf[1], smem_addr(B.ptr(nt * MMA_N + lane_in_8, b_col + k_off)));
            mma_bf16(D[nt][0], D[nt][1], D[nt][2], D[nt][3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }
    }
}

template <int N_TILES>
__device__ __forceinline__ void mma_av_panel(float (&D)[N_TILES][4], const float A_a[K_TILES_BT][4],
                                              Bf16SmemTile<D_PANEL> V, int lane_g, int lane_t)
{
#pragma unroll
    for (int kt = 0; kt < K_TILES_BT; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0 = A_a[kt][0];
        const float a1 = A_a[kt][1];
        const float a2 = A_a[kt][2];
        const float a3 = A_a[kt][3];

#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            const int col = nt * MMA_N + lane_g;
            const float b0 = __bfloat162float(*V.ptr(k_off + lane_t, col));
            const float b1 = __bfloat162float(*V.ptr(k_off + lane_t + 4, col));
            mma_tf32(D[nt][0], D[nt][1], D[nt][2], D[nt][3], a0, a1, a2, a3, b0, b1);
        }
    }
}

__device__ __forceinline__ void output_job(const __nv_bfloat16* __restrict__ q_in,
                                           const __nv_bfloat16* __restrict__ k_in,
                                           const __nv_bfloat16* __restrict__ v_new_in,
                                           const float* __restrict__ g_cumsum_in,
                                           const __nv_bfloat16* __restrict__ h_chunk_in,
                                           __nv_bfloat16* __restrict__ attn_out, ChunkGeom geom, float scale,
                                           int chunk, int h_v, float* smem)
{
    auto* const bf16_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    auto* const q_smem = bf16_smem;
    auto* const stage0 = q_smem + kernel_dims::Q_BF16;
    auto* const stage1 = stage0 + kernel_dims::STAGE_BF16;
    float* const g_smem = reinterpret_cast<float*>(stage0 + kernel_dims::V_PANEL_BF16);

    Bf16SmemTile<kStateDim> q_view{q_smem};
    Bf16SmemTile<K_PANEL> k_stage0{stage0};
    Bf16SmemTile<K_PANEL> k_stage1{stage1};
    Bf16SmemTile<kStateDim> h_view{stage0};
    Bf16SmemTile<D_PANEL> v_view{stage1};

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const std::int64_t cs = static_cast<std::int64_t>(chunk) * BT;
    const std::int64_t H_v = geom.H_v;
    const std::int64_t qk_stride_t = geom.qk_row_stride;
    const std::int64_t qk_head_idx = static_cast<std::int64_t>(geom.qk_head(h_v)) * kStateDim;
    const std::int64_t q_base = cs * qk_stride_t + qk_head_idx;
    const std::int64_t k_base = q_base;
    const std::int64_t vn_base = cs * H_v * kStateDim + static_cast<std::int64_t>(h_v) * kStateDim;
    const std::int64_t hc_base = (static_cast<std::int64_t>(chunk) * H_v + h_v) * kStateDim * kStateDim;

    const std::int64_t value_row_stride = H_v * kStateDim;

    // Q is permanent. K uses two 64x32 BF16 buffers and is prefetched one panel ahead while the
    // current panel feeds BF16 MMA.
    issue_cp_bf16<BT, kStateDim, THREADS>(q_view, q_in + q_base, qk_stride_t, tid);
    issue_cp_bf16<BT, K_PANEL, THREADS>(k_stage0, k_in + k_base, qk_stride_t, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    float A_strip[N_TILES_BT][4] = {};

#pragma unroll
    for (int panel = 0; panel < N_K_PANELS; ++panel) {
        Bf16SmemTile<K_PANEL> current = (panel & 1) == 0 ? k_stage0 : k_stage1;
        if (panel + 1 < N_K_PANELS) {
            Bf16SmemTile<K_PANEL> next = (panel & 1) == 0 ? k_stage1 : k_stage0;
            issue_cp_bf16<BT, K_PANEL, THREADS>(next, k_in + k_base + static_cast<std::int64_t>(panel + 1) * K_PANEL,
                                                qk_stride_t, tid);
            cp_commit();
        } else if (tid < BT) {
            // stage0 was consumed by panel 2 and is now dead. Reuse its otherwise-idle tail for g
            // while panel 3 consumes stage1.
            g_smem[tid] = g_cumsum_in[(cs + static_cast<std::int64_t>(tid)) * H_v + h_v];
        }

        mma_bf16_panel<N_TILES_BT, K_PANEL / BF16_MMA_K>(A_strip, q_view, current, warp * MMA_M, panel * K_PANEL, 0, lane);

        if (panel + 1 < N_K_PANELS) {
            cp_wait<0>();
            __syncthreads();
        }
    }
    __syncthreads();

    // Apply causal decay in FP32. Upper-triangle exp2 values may overflow, so the conditional select
    // must replace the entire product rather than multiply by a zero mask.
    const int row_g0 = warp * MMA_M + lane_g;
    const int row_g1 = row_g0 + 8;
    const float g_r0 = g_smem[row_g0];
    const float g_r1 = g_smem[row_g1];
    const float gamma0 = exp2_approx(g_r0 * kLog2E);
    const float gamma1 = exp2_approx(g_r1 * kLog2E);

#pragma unroll
    for (int nt = 0; nt < N_TILES_BT; ++nt) {
        const int s0 = nt * MMA_N + 2 * lane_t;
        const int s1 = s0 + 1;
        const float g_s0 = g_smem[s0];
        const float g_s1 = g_smem[s1];

        const float dec00 = exp2_approx((g_r0 - g_s0) * kLog2E);
        const float dec01 = exp2_approx((g_r0 - g_s1) * kLog2E);
        const float dec10 = exp2_approx((g_r1 - g_s0) * kLog2E);
        const float dec11 = exp2_approx((g_r1 - g_s1) * kLog2E);

        A_strip[nt][0] = (s0 <= row_g0) ? A_strip[nt][0] * dec00 : 0.0f;
        A_strip[nt][1] = (s1 <= row_g0) ? A_strip[nt][1] * dec01 : 0.0f;
        A_strip[nt][2] = (s0 <= row_g1) ? A_strip[nt][2] * dec10 : 0.0f;
        A_strip[nt][3] = (s1 <= row_g1) ? A_strip[nt][3] * dec11 : 0.0f;
    }

    // Convert the m16n8 accumulator layout into the TF32 A-operand layout needed by A @ V. A stays
    // FP32 throughout.
    float A_a[K_TILES_BT][4];
    {
        const int src_lo = (lane_g << 2) | (lane_t >> 1);
        const int src_hi = src_lo + 2;
        const bool t_odd = (lane_t & 1) != 0;
        constexpr unsigned mask = 0xFFFFFFFFu;

#pragma unroll
        for (int kt = 0; kt < K_TILES_BT; ++kt) {
            const float d0_lo = __shfl_sync(mask, A_strip[kt][0], src_lo);
            const float d1_lo = __shfl_sync(mask, A_strip[kt][1], src_lo);
            const float d2_lo = __shfl_sync(mask, A_strip[kt][2], src_lo);
            const float d3_lo = __shfl_sync(mask, A_strip[kt][3], src_lo);
            const float d0_hi = __shfl_sync(mask, A_strip[kt][0], src_hi);
            const float d1_hi = __shfl_sync(mask, A_strip[kt][1], src_hi);
            const float d2_hi = __shfl_sync(mask, A_strip[kt][2], src_hi);
            const float d3_hi = __shfl_sync(mask, A_strip[kt][3], src_hi);

            A_a[kt][0] = t_odd ? d1_lo : d0_lo;
            A_a[kt][1] = t_odd ? d3_lo : d2_lo;
            A_a[kt][2] = t_odd ? d1_hi : d0_hi;
            A_a[kt][3] = t_odd ? d3_hi : d2_hi;
        }
    }

    // All K reads must finish before stage0 becomes H panel 0.
    __syncthreads();
    issue_cp_bf16<D_PANEL, kStateDim, THREADS>(h_view, h_chunk_in + hc_base, kStateDim, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    // H and V use disjoint buffers. V[c] is fetched while Q @ H[c]^T runs; H[c+1] is fetched
    // while the FP32 A @ V[c] path runs.
#pragma unroll 1
    for (int panel = 0; panel < N_D_PANELS; ++panel) {
        const int d_off = panel * D_PANEL;

        issue_cp_bf16<BT, D_PANEL, THREADS>(v_view, v_new_in + vn_base + static_cast<std::int64_t>(d_off),
                                             value_row_stride, tid);
        cp_commit();

        float D_frag[D_PANEL / MMA_N][4] = {};
        mma_bf16_panel<D_PANEL / MMA_N, kStateDim / BF16_MMA_K>(D_frag, q_view, h_view, warp * MMA_M, 0, 0, lane);

#pragma unroll
        for (int nt = 0; nt < D_PANEL / MMA_N; ++nt) {
            D_frag[nt][0] *= gamma0;
            D_frag[nt][1] *= gamma0;
            D_frag[nt][2] *= gamma1;
            D_frag[nt][3] *= gamma1;
        }

        cp_wait<0>();
        __syncthreads();

        if (panel + 1 < N_D_PANELS) {
            issue_cp_bf16<D_PANEL, kStateDim, THREADS>(
                h_view, h_chunk_in + hc_base + static_cast<std::int64_t>(panel + 1) * D_PANEL * kStateDim, kStateDim, tid);
            cp_commit();
        }

        mma_av_panel(D_frag, A_a, v_view, lane_g, lane_t);

#pragma unroll
        for (int nt = 0; nt < D_PANEL / MMA_N; ++nt) {
            const int d_global = d_off + nt * MMA_N + 2 * lane_t;
            const __nv_bfloat162 out0 = pack_bf16x2_rz(scale * D_frag[nt][0], scale * D_frag[nt][1]);
            const __nv_bfloat162 out1 = pack_bf16x2_rz(scale * D_frag[nt][2], scale * D_frag[nt][3]);
            store_vec(&attn_out[vn_base + static_cast<std::int64_t>(row_g0) * value_row_stride + d_global], out0);
            store_vec(&attn_out[vn_base + static_cast<std::int64_t>(row_g1) * value_row_stride + d_global], out1);
        }

        if (panel + 1 < N_D_PANELS) {
            cp_wait<0>();
            __syncthreads();
        }
    }
}

template <bool MULTI_JOB>
__launch_bounds__(THREADS, 4) __global__
void output_kernel(const __nv_bfloat16* __restrict__ q_in, const __nv_bfloat16* __restrict__ k_in,
                   const __nv_bfloat16* __restrict__ v_new_in,
                   const float* __restrict__ g_cumsum_in, const __nv_bfloat16* __restrict__ h_chunk_in,
                   __nv_bfloat16* __restrict__ attn_out, ChunkGeom geom, float scale, int chunks)
{
    extern __shared__ float smem[];

    const int h_v = static_cast<int>(blockIdx.y);
    if constexpr (MULTI_JOB) {
        const int chunk_stride = static_cast<int>(gridDim.x);
        for (int chunk = static_cast<int>(blockIdx.x); chunk < chunks; chunk += chunk_stride) {
            output_job(q_in, k_in, v_new_in, g_cumsum_in, h_chunk_in, attn_out, geom, scale, chunk, h_v, smem);
            if (chunk + chunk_stride < chunks) __syncthreads();
        }
    } else {
        output_job(q_in, k_in, v_new_in, g_cumsum_in, h_chunk_in, attn_out, geom, scale,
                   static_cast<int>(blockIdx.x), h_v, smem);
    }
}

} // namespace output

// ---------------------------------------------------------------------------
// L2 normalization of q and k (ninfer runs this as a separate op before the chunked stages).
// ---------------------------------------------------------------------------

// One warp per (token, head). Reads the packed conv row, writes a contiguous [T, H_qk*128] bf16
// panel. The reduction is a fixed __shfl_down tree, so the normalization is bit-reproducible, and
// it matches the serial kernel's rsqrtf(sum + 1e-6) exactly in form (it normalizes the same bf16
// inputs the serial kernel reads, so both paths see identical q/k values up to summation order).
// src_col_off selects q (0) or k (geom.qk_row_off) inside the packed row, and out is the
// contiguous [T, H_qk*128] panel the chunked stages read.
// static: the only non-template kernel in this header, so it would otherwise be emitted into every
// including TU and collide at link time with the copy in gdn_chunked.cu.
static __global__ __launch_bounds__(256) void gdn_l2norm_qk_kernel(const __nv_bfloat16* __restrict__ mixed_qkv,
                                                           int64_t src_col_off, __nv_bfloat16* __restrict__ out,
                                                           ChunkGeom geom, int tokens)
{
    constexpr int kBlock = 256;
    constexpr int kWarpsPerBlock = kBlock / ptx::kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (ptx::kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / ptx::kWarpSize;
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= tokens * geom.H_qk) return;

    const int t = row / geom.H_qk;
    const int h = row - t * geom.H_qk;
    const __nv_bfloat16* src = mixed_qkv + (int64_t)t * geom.v_row_stride + src_col_off +
                               (int64_t)h * kStateDim;
    __nv_bfloat16* dst = out + (int64_t)row * kStateDim;

    constexpr int pairs = kStateDim / 2;
    __nv_bfloat162 values[pairs / ptx::kWarpSize];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < pairs / ptx::kWarpSize; ++k) {
        const int pair = lane + k * ptx::kWarpSize;
        values[k] = __halves2bfloat162(src[2 * pair], src[2 * pair + 1]);
        const float2 xf = __bfloat1622float2(values[k]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }
    sum = ptx::warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum + 1e-6f) : 0.0f;
    inv = __shfl_sync(ptx::kFullWarpMask, inv, 0);
#pragma unroll
    for (int k = 0; k < pairs / ptx::kWarpSize; ++k) {
        const int pair = lane + k * ptx::kWarpSize;
        const float2 xf = __bfloat1622float2(values[k]);
        ptx::store_vec(&dst[2 * pair], ptx::pack_bf16x2_rn(xf.x * inv, xf.y * inv));
    }
}

// ---------------------------------------------------------------------------
// Workspace + launch config
// ---------------------------------------------------------------------------

// g_cumsum: fp32 [H_v, T]; W / U / v_new: bf16 [D, H_v, T]; h_chunk: bf16 [D, D, H_v, chunks].
// The sub-tensors are carved out of one 256-byte-aligned arena by the caller.
struct ChunkWorkspace
{
    float* g_cumsum = nullptr;
    __nv_bfloat16* W = nullptr;
    __nv_bfloat16* U = nullptr;
    __nv_bfloat16* v_new = nullptr;
    __nv_bfloat16* h_chunk = nullptr;
    // Normalized q / k panels, bf16 [T, H_qk*D] each.
    __nv_bfloat16* q_norm = nullptr;
    __nv_bfloat16* k_norm = nullptr;
    int tokens = 0;  // capacity in tokens
};

struct prepare_wy_wu_config
{
    ChunkGeom geom;
    const __nv_bfloat16* k_in = nullptr;
    const __nv_bfloat16* v_in = nullptr;
    const float* g_in = nullptr;
    const __nv_bfloat16* beta = nullptr;
    __nv_bfloat16* W = nullptr;
    __nv_bfloat16* U = nullptr;
    float* g_cumsum_out = nullptr;
    int chunks = 0;  // tokens / 64
    cudaStream_t stream = nullptr;
};

struct state_passing_config
{
    ChunkGeom geom;
    const __nv_bfloat16* W = nullptr;
    const __nv_bfloat16* U = nullptr;
    const __nv_bfloat16* k_in = nullptr;
    const float* g_cumsum = nullptr;
    const float* state_in = nullptr;
    __nv_bfloat16* v_new = nullptr;
    __nv_bfloat16* h_chunk = nullptr;
    float* state_out = nullptr;
    int chunks = 0;  // tokens / 64
    cudaStream_t stream = nullptr;
};

struct chunk_output_config
{
    ChunkGeom geom;
    const __nv_bfloat16* q_in = nullptr;
    const __nv_bfloat16* k_in = nullptr;
    const __nv_bfloat16* v_new = nullptr;
    const float* g_cumsum = nullptr;
    const __nv_bfloat16* h_chunk = nullptr;
    __nv_bfloat16* attn_out = nullptr;
    float scale = 0.0f;
    int chunks = 0;  // tokens / 64
    cudaStream_t stream = nullptr;
};

cudaError_t launch_l2norm_qk(const __nv_bfloat16* mixed_qkv, int64_t src_col_off, __nv_bfloat16* out,
                             const ChunkGeom& geom, int tokens, cudaStream_t stream);
cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config& cfg);
cudaError_t launch_state_passing(const state_passing_config& cfg);
cudaError_t launch_output(const chunk_output_config& cfg);

// Bytes the chunked workspace needs for `tokens` tokens (rounded down to a 64-token multiple) at
// the given head counts, including the two normalized q/k panels.
size_t chunk_workspace_bytes(int value_heads, int qk_heads, int tokens);
ChunkWorkspace bind_workspace(void* arena, size_t bytes, int value_heads, int qk_heads, int tokens);

}} // namespace helios::gdn
