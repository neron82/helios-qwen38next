#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "exl3_devctx.cuh"
#include "helios_shim.cuh"

//DevCtx::DevCtc()
//{
//    int num_sms[MAX_DEVICES] = {};
//    int cc[MAX_DEVICES] = {};
//    void* locks[MAX_DEVICES] = {};
//    std::mutex mtx;
//}

namespace helios
{
namespace exl3
{

DevCtx& DevCtx::instance()
{
    static DevCtx ctx;
    return ctx;
}

int DevCtx::get_num_sms(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!num_sms[device])
        cuda_check(cudaDeviceGetAttribute(&num_sms[device], cudaDevAttrMultiProcessorCount, device));
    return num_sms[device];
}

int DevCtx::get_cc(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!cc[device])
    {
        cudaDeviceProp prop;
        cuda_check(cudaGetDeviceProperties(&prop, device));
        if (prop.major >= 10) cc[device] = CC_BLACKWELL;
        else if (prop.major >= 9) cc[device] = CC_HOPPER;
        else if (prop.major >= 8 && prop.minor >= 9) cc[device] = CC_ADA;
        else if (prop.major >= 8) cc[device] = CC_AMPERE;
        else cc[device] = CC_OLD;
    }
    return cc[device];
}

void* DevCtx::get_ws(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!ws[device])
    {
        cudaSetDevice(device);
        HELIOS_CUDA_CHECK(cudaMalloc(&ws[device], WORKSPACE_SIZE));
    }
    return ws[device];
}

int* DevCtx::get_locks(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!locks[device])
    {
        cudaSetDevice(device);
        size_t size = (MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS) * sizeof(int);
        HELIOS_CUDA_CHECK(cudaMalloc(&locks[device], size));
        HELIOS_CUDA_CHECK(cudaMemset(locks[device], 0, size));
    }
    return (int*) locks[device];
}

long long* DevCtx::get_moe_fp_scratch(int device, size_t n)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (n > moe_fp_n[device])
    {
        int cur = -1;
        cudaGetDevice(&cur);
        cudaSetDevice(device);
        if (moe_fp[device]) HELIOS_CUDA_CHECK(cudaFree(moe_fp[device]));
        // Exactly n, no headroom: max_chunk is fixed once the runner initialises, so a "slightly
        // larger chunk later" cannot happen. The 25% pad was pure VRAM cost, and this buffer sits
        // alongside a multi-GB KV cache at long context.
        HELIOS_CUDA_CHECK(cudaMalloc(&moe_fp[device], n * sizeof(long long)));
        moe_fp_n[device] = n;
        cudaSetDevice(cur);
    }
    return (long long*) moe_fp[device];
}

int g_get_cc(int device)
{
    return DevCtx::instance().get_cc(device);
}

int g_get_num_sms(int device)
{
    return DevCtx::instance().get_num_sms(device);
}

void prepare_ctx(int device)
{
    DevCtx::instance().get_num_sms(device);
    DevCtx::instance().get_cc(device);
    DevCtx::instance().get_locks(device);
}

} // namespace exl3
} // namespace helios
