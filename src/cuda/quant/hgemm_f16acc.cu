// Dense fp16 GEMM with fp16-accumulator tensor-core MMA and fp32 accumulation across K.
// Ported from exllamav3_ext/hgemm_f16acc.cu. See hgemm_f16acc.cuh for what changed and why.

#include <cuda_fp16.h>
#include "hgemm_f16acc.cuh"
#include "helios_shim.cuh"

#include <mutex>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>

namespace helios
{
namespace exl3
{

namespace f16acc
{

constexpr int MIN_ROWS = 384;
constexpr int KSLICE = 32;      // fp32 flush cadence along K, asserted against the k16 step count

// The layout/shape sweep was measured on GeForce Blackwell upstream; other Ampere+ parts (this
// engine's sm_86 target) retain the existing layout, and the rate probe still decides whether
// the fp16 MMA is worthwhile at all.
template <int TILE_N, bool TUNED> struct Config
{
    static constexpr int BM = 128, BN = TILE_N, BK = 64, PAD = TUNED ? 0 : 8;
    static constexpr int WARPS_M = 2, WARPS_N = BN / 32;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int STAGES = 2, GROUP_M = TUNED ? 16 : 8;
    static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    static constexpr int MT = WM / 16, NT = WN / 8;
    static constexpr int AS_STRIDE = BK + PAD, BS_STRIDE = BN + PAD;
    static constexpr int A_STAGE = BM * AS_STRIDE, B_STAGE = BK * BS_STRIDE;
    static constexpr size_t SMEM_BYTES = (size_t) STAGES * (A_STAGE + B_STAGE) * sizeof(half);
};
// The only configuration this port instantiates: the non-tuned 128-wide tile, which is what
// upstream keeps for every device that is not compute capability 12. The shape heuristics below
// read their tile sizes from it rather than from a second copy of the constants.
using Def = Config<128, false>;


__device__ __forceinline__ void add_half_pair(float& a, float& b, uint32_t h)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // PTX 8.6+: convert an FP16 partial and add it to the FP32 total in one instruction.
    asm volatile("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; "
                 "add.rn.f32.f16 %0, lo, %0; add.rn.f32.f16 %1, hi, %1; }"
                 : "+f"(a), "+f"(b) : "r"(h));
#else
    float2 f = __half22float2(*reinterpret_cast<half2*>(&h));
    a += f.x;
    b += f.y;
#endif
}


__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred)
{
    int src_size = pred ? 16 : 0;    // 0 -> zero-fill, no global read
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_u32(smem)), "l"(gmem), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_f16(uint32_t* c, const uint32_t* a, const uint32_t* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f32(float* c, const uint32_t* a, const uint32_t* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <bool OUT_F32, int TILE_N = 128, bool TUNED = false>
__global__ void __launch_bounds__((Config<TILE_N, TUNED>::THREADS), 1)
gemm_kernel
(
    const half* __restrict__ A, const half* __restrict__ B, void* __restrict__ C,
    int M, int N, int K, int ldc,
    long long strideA, long long strideB, long long strideC
)
{
    using CF = Config<TILE_N, TUNED>;
    constexpr int BM = CF::BM, BN = CF::BN, BK = CF::BK, THREADS = CF::THREADS;
    constexpr int STAGES = CF::STAGES, GROUP_M = CF::GROUP_M;
    constexpr int WARPS_M = CF::WARPS_M, WARPS_N = CF::WARPS_N;
    constexpr int WM = CF::WM, WN = CF::WN, MT = CF::MT, NT = CF::NT;
    constexpr int AS_STRIDE = CF::AS_STRIDE, BS_STRIDE = CF::BS_STRIDE;
    constexpr int A_STAGE = CF::A_STAGE, B_STAGE = CF::B_STAGE;

    extern __shared__ __align__(16) half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * A_STAGE;

    // Grouped raster: consecutive blocks walk GROUP_M M-stripes over one N column before moving
    // on, so the B stripe stays L2-hot across GROUP_M A-stripes
    const int grid_n = gridDim.x, grid_m = gridDim.y;
    const int bid = blockIdx.y * grid_n + blockIdx.x;
    const int group_size = GROUP_M * grid_n;
    const int group = bid / group_size;
    const int first_m = group * GROUP_M;
    const int gm_eff = min(GROUP_M, grid_m - first_m);
    const int in_group = bid - group * group_size;
    const int bm = (first_m + in_group % gm_eff) * BM;
    const int bn = (in_group / gm_eff) * BN;

    const int bz = blockIdx.z;
    A += (long long) bz * strideA;
    B += (long long) bz * strideB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;

    constexpr int A_CPR = BK / 8, B_CPR = BN / 8;
    constexpr int A_ITERS = BM * A_CPR / THREADS, B_ITERS = BK * B_CPR / THREADS;
    auto load_tile = [&](int stage, int kt)
    {
        const int k0 = kt * BK;
        half* as = As + stage * A_STAGE;
        half* bs = Bs + stage * B_STAGE;
        #pragma unroll
        for (int i = 0; i < A_ITERS; ++i)
        {
            int c = tid + i * THREADS;
            int row = c / A_CPR, chunk = c % A_CPR;
            int grow = bm + row;
            bool pred = grow < M;
            const half* src = A + (long long) (pred ? grow : 0) * K + k0 + chunk * 8;
            if constexpr (TUNED) cp_async16(as + row * AS_STRIDE + (chunk ^ (row % A_CPR)) * 8, src, pred);
            else cp_async16(as + row * AS_STRIDE + chunk * 8, src, pred);
        }
        #pragma unroll
        for (int i = 0; i < B_ITERS; ++i)
        {
            int c = tid + i * THREADS;
            int row = c / B_CPR, chunk = c % B_CPR;
            const half* src = B + (long long) (k0 + row) * N + bn + chunk * 8;
            if constexpr (TUNED) cp_async16(bs + row * BS_STRIDE + (chunk ^ (row % B_CPR)) * 8, src, true);
            else cp_async16(bs + row * BS_STRIDE + chunk * 8, src, true);
        }
    };

    float acc[MT][NT][4];
    uint32_t hacc[MT][NT][2];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
        {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
            hacc[i][j][0] = hacc[i][j][1] = 0u;
        }
    auto flush = [&]()
    {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float2 f0 = __half22float2(*reinterpret_cast<half2*>(&hacc[i][j][0]));
                float2 f1 = __half22float2(*reinterpret_cast<half2*>(&hacc[i][j][1]));
                acc[i][j][0] += f0.x; acc[i][j][1] += f0.y;
                acc[i][j][2] += f1.x; acc[i][j][3] += f1.y;
                hacc[i][j][0] = 0u; hacc[i][j][1] = 0u;
            }
    };

    const int KT = K / BK;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }

    const int a_lrow = lane & 15, a_lcol = (lane >> 4) * 8;
    const int b_lrow = (lane & 7) + ((lane >> 3) & 1) * 8, b_lcol = (lane >> 4) * 8;
    constexpr int KK = BK / 16;
    static_assert(KSLICE == 16 * 2, "flush cadence below assumes kslice 32 = two k16 steps");

    uint32_t af[2][MT][4];
    uint32_t bf[2][NT][2];
    auto load_frags = [&](int buf, const half* as, const half* bs, int kk)
    {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            if constexpr (TUNED)
                ldmatrix_x4(af[buf][i], as + (i * 16 + a_lrow) * AS_STRIDE +
                            (((kk * 16 + a_lcol) / 8) ^ ((i * 16 + a_lrow) % A_CPR)) * 8);
            else
                ldmatrix_x4(af[buf][i], as + (i * 16 + a_lrow) * AS_STRIDE + kk * 16 + a_lcol);
        #pragma unroll
        for (int j = 0; j < NT; j += 2)
        {
            uint32_t r[4];
            if constexpr (TUNED)
                ldmatrix_x4_trans(r, bs - wn * WN + (kk * 16 + b_lrow) * BS_STRIDE +
                                  (((wn * WN + j * 8 + b_lcol) / 8) ^ ((kk * 16 + b_lrow) % B_CPR)) * 8);
            else
                ldmatrix_x4_trans(r, bs + (kk * 16 + b_lrow) * BS_STRIDE + j * 8 + b_lcol);
            bf[buf][j][0] = r[0]; bf[buf][j][1] = r[1]; bf[buf][j + 1][0] = r[2]; bf[buf][j + 1][1] = r[3];
        }
    };

    for (int kt = 0; kt < KT; ++kt)
    {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KT) load_tile(nk % STAGES, nk);
            cp_async_commit();
        }
        const half* as = As + (kt % STAGES) * A_STAGE + wm * WM * AS_STRIDE;
        const half* bs = Bs + (kt % STAGES) * B_STAGE + wn * WN;

        if constexpr (TUNED)
        {
            // Preserve each pair of k16 MMAs and the FP32 addition order, but flush a
            // fragment immediately so conversions/additions can overlap other MMAs.
            #pragma unroll
            for (int kk = 0; kk < KK; kk += 2)
            {
                load_frags(0, as, bs, kk);
                load_frags(1, as, bs, kk + 1);
                #pragma unroll
                for (int i = 0; i < MT; ++i)
                    #pragma unroll
                    for (int j = 0; j < NT; ++j)
                    {
                        uint32_t h[2] = {};
                        mma_f16(h, af[0][i], bf[0][j]);
                        mma_f16(h, af[1][i], bf[1][j]);
                        add_half_pair(acc[i][j][0], acc[i][j][1], h[0]);
                        add_half_pair(acc[i][j][2], acc[i][j][3], h[1]);
                    }
            }
        }
        else
        {
            load_frags(0, as, bs, 0);
            #pragma unroll
            for (int kk = 0; kk < KK; ++kk)
            {
                if (kk + 1 < KK) load_frags((kk + 1) & 1, as, bs, kk + 1);
                const int cur = kk & 1;
                #pragma unroll
                for (int i = 0; i < MT; ++i)
                    #pragma unroll
                    for (int j = 0; j < NT; ++j)
                        mma_f16(hacc[i][j], af[cur][i], bf[cur][j]);
                if (kk & 1) flush();     // every 32 of K
            }
        }
    }

    const int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
        {
            int col = bn + wn * WN + j * 8 + t * 2;
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                int row = bm + wm * WM + i * 16 + g + h * 8;
                if (row >= M) continue;
                float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
                if constexpr (OUT_F32)
                {
                    float* c = reinterpret_cast<float*>(C) + (long long) bz * strideC + (long long) row * ldc + col;
                    *reinterpret_cast<float2*>(c) = make_float2(v0, v1);
                }
                else
                {
                    half* c = reinterpret_cast<half*>(C) + (long long) bz * strideC + (long long) row * ldc + col;
                    *reinterpret_cast<half2*>(c) = __floats2half2_rn(v0, v1);
                }
            }
        }
}

// Rate probe: independent register-resident MMA chains, no memory traffic
template <bool F16>
__global__ void __launch_bounds__(256) rate_kernel(int iters, float* sink)
{
    uint32_t a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 5, b1 = a0 + 7;
    uint32_t a[4] = { a0, a1, a2, a3 }, b[2] = { b0, b1 };
    uint32_t h[8][2] = {};
    float f[8][4] = {};
    for (int i = 0; i < iters; ++i)
    {
        #pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            if constexpr (F16) mma_f16(h[c], a, b);
            else mma_f32(f[c], a, b);
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) s += F16 ? __uint_as_float(h[c][0] ^ h[c][1]) : f[c][0] + f[c][3];
    if (s == 123.456f) sink[threadIdx.x] = s;
}

// Per-device decision: -1 unknown, 0 off, 1 on
constexpr int MAX_DEVICES = 16;
static int g_enabled[MAX_DEVICES];
static bool g_init = false;
static std::mutex g_mutex;

static float probe_ms(bool f16, int blocks, int iters, float* sink, cudaStream_t stream)
{
    auto run = [&]() { if (f16) rate_kernel<true><<<blocks, 256, 0, stream>>>(iters, sink);
                       else rate_kernel<false><<<blocks, 256, 0, stream>>>(iters, sink); };
    run();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int r = 0; r < 3; ++r) run();
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

// Device properties are read once per device: both are driver round-trips and linear() calls
// this launcher ~300 times per prefill chunk. Deliberately not guarded by g_mutex, which
// enabled() already holds - hence its own once_flag array.
struct DevInfo { int major = 0; int sms = 1; };

static DevInfo dev_info(int device)
{
    static DevInfo cache[MAX_DEVICES];
    static std::once_flag once[MAX_DEVICES];
    if (device < 0 || device >= MAX_DEVICES) return DevInfo{};
    std::call_once(once[device], [&]()
    {
        cudaDeviceGetAttribute(&cache[device].major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&cache[device].sms, cudaDevAttrMultiProcessorCount, device);
    });
    return cache[device];
}

static bool enabled(int device)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_init)
    {
        for (int i = 0; i < MAX_DEVICES; ++i) g_enabled[i] = -1;
        g_init = true;
    }
    if (device < 0 || device >= MAX_DEVICES) return false;
    if (g_enabled[device] >= 0) return g_enabled[device] == 1;

    int on = 0;
    const char* env = std::getenv("HELIOS_HGEMM_F16ACC");
    DevInfo di = dev_info(device);
    if (di.major < 8)
        on = 0;                                   // cp.async / ldmatrix.x4 / m16n8k16 need sm80+
    else if (env && std::strcmp(env, "0") == 0)
        on = 0;
    else if (env && std::strcmp(env, "1") == 0)
        on = 1;
    else
    {
        // Auto: enable where the fp16-accumulator MMA is at least 1.5x faster than the
        // fp32-accumulator one (GeForce: 2.0x; workstation / datacenter parts: 1.0x)
        float* sink = nullptr;
        cudaMalloc(&sink, 256 * sizeof(float));
        float t32 = probe_ms(false, di.sms * 4, 512, sink, 0);
        float t16 = probe_ms(true, di.sms * 4, 512, sink, 0);
        cudaFree(sink);
        cudaGetLastError();
        on = (t32 > 0.f && t16 > 0.f && t32 / t16 >= 1.5f) ? 1 : 0;
    }
    g_enabled[device] = on;
    return on == 1;
}

// Hard shape coverage of the kernel (independent of the device decision).
static bool covered(const void* a, const void* b, const void* c, int M, int N, int K, int ldc,
                    bool c_fp32, int device)
{
    if (!a || !b || !c) return false;
    if (M < 1 || N < 1 || K < 1) return false;
    if (K % Def::BK != 0 || N % Def::BN != 0) return false;
    if ((M + Def::BM - 1) / Def::BM > 65535) return false;
    if (ldc < N || ldc % 2 != 0) return false;
    if (M > std::numeric_limits<int>::max() || N > std::numeric_limits<int>::max() ||
        K > std::numeric_limits<int>::max()) return false;
    const uintptr_t output_alignment = c_fp32 ? 8 : 4;
    if (((uintptr_t) a & 15) || ((uintptr_t) b & 15) || ((uintptr_t) c & (output_alignment - 1)))
        return false;
    return dev_info(device).major >= 8;
}

// Shape heuristics decide whether it pays. Non-Blackwell parts only use the 128-wide tile, so
// the narrow-tile branch is unreachable here and the block count is the whole test.
static bool worthwhile(int M, int N, int K, int device)
{
    if (M < MIN_ROWS) return false;
    int64_t blocks = ((M + Def::BM - 1) / Def::BM) * (N / Def::BN);

    return blocks >= dev_info(device).sms;
}

// Upstream's worthwhile() is a fill-the-GPU test, not a correctness limit: when it declines,
// exllamav3 falls back to cuBLAS, which is still a tensor-core GEMM. helios has no cuBLAS, so its
// fallback is the EXL3 trellis kernel - and for a narrow projection (N = 512 or 640 at prefill
// width) that is not a near miss, it is two orders of magnitude slower. HELIOS_HGEMM_SMALL_SHAPES=1
// therefore keeps upstream's threshold for reporting but lets the caller take the fp16 GEMM
// anyway; the shape is still required to satisfy covered() in full.
static bool small_shapes_allowed()
{
    static int v = -1;
    if (v < 0)
    {
        // ON by default: measured +15.0% prefill at 13.6k and +14.6% at 3.4k (3 runs each, <1%
        // spread), on top of the reconstruct path, with no decode cost. The shapes concerned are
        // the narrow projections (attn k/v at N=512, shared-expert gate/up at N=640) whose block
        // counts are 64 and 80 at M=2048 against 82 SMs - a full wave each, which is exactly the
        // regime this kernel tiles badly. HELIOS_HGEMM_SMALL_SHAPES=0 restores upstream's threshold.
        const char* e = std::getenv("HELIOS_HGEMM_SMALL_SHAPES");
        v = (!e || e[0] == 0 || std::strcmp(e, "0") != 0) ? 1 : 0;
    }
    return v == 1;
}

template <bool OUT_F32, int TILE_N, bool TUNED>
static void launch_config(const half* a, const half* b, void* c, int M, int N, int K, int ldc,
                          cudaStream_t stream)
{
    using CF = Config<TILE_N, TUNED>;
    auto kern = gemm_kernel<OUT_F32, TILE_N, TUNED>;
    static std::once_flag attr_set[MAX_DEVICES];
    int device = 0;
    cudaGetDevice(&device);
    HELIOS_ASSERT(device >= 0 && device < MAX_DEVICES, "hgemm_f16acc: device index");
    std::call_once(attr_set[device], [&]()
    {
        cuda_check(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int) CF::SMEM_BYTES));
    });
    dim3 grid(N / CF::BN, (M + CF::BM - 1) / CF::BM, 1);
    kern<<<grid, CF::THREADS, CF::SMEM_BYTES, stream>>>
    (
        a, b, c,
        M, N, K, ldc,
        0, 0, 0
    );
    cuda_check(cudaPeekAtLastError());
}

} // namespace f16acc

int hgemm_f16acc_status(int device)
{
    return f16acc::enabled(device) ? 1 : 0;
}

bool hgemm_f16acc_try
(
    void*               c,
    const void*         a,
    const void*         b,
    int                 M,
    int                 N,
    int                 K,
    int                 ldc,
    bool                c_fp32,
    Stream              s
)
{
    int device = 0;
    cudaGetDevice(&device);
    if (!f16acc::covered(a, b, c, M, N, K, ldc, c_fp32, device)) return false;
    if (!f16acc::worthwhile(M, N, K, device) && !f16acc::small_shapes_allowed()) return false;
    if (!f16acc::enabled(device)) return false;
    if (c_fp32) f16acc::launch_config<true, 128, false>((const half*) a, (const half*) b, c, M, N, K, ldc, s);
    else f16acc::launch_config<false, 128, false>((const half*) a, (const half*) b, c, M, N, K, ldc, s);
    return true;
}

} // namespace exl3
} // namespace helios
