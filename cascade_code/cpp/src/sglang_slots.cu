#include "sglang_slots.hpp"
#include <cuda_runtime.h>

namespace cascade {
namespace {


template <bool GATHER>
__global__ void slots_kernel(const unsigned long long* __restrict__ k_ptrs,
                             const unsigned long long* __restrict__ v_ptrs,
                             const long long* __restrict__ slots,
                             unsigned char* __restrict__ flat,
                             int num_layers, int num_tokens, int token_stride) {
    const long long vec_per_token = token_stride / 16;
    const long long total =
        (long long)2 * num_layers * num_tokens * vec_per_token;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x) {
        long long v      = idx % vec_per_token;
        long long rest   = idx / vec_per_token;
        long long tok    = rest % num_tokens;
        long long rest2  = rest / num_tokens;
        long long layer  = rest2 % num_layers;
        long long kv     = rest2 / num_layers;

        unsigned long long base = kv ? v_ptrs[layer] : k_ptrs[layer];
        uint4* pool = reinterpret_cast<uint4*>(
            base + (unsigned long long)slots[tok] * (unsigned long long)token_stride);
        uint4* side = reinterpret_cast<uint4*>(flat + idx * 16);

        if (GATHER) {
            *side = pool[v];
        } else {
            pool[v] = *side;
        }
    }
}

int launch(bool gather, uintptr_t k_ptrs_dev, uintptr_t v_ptrs_dev,
           uintptr_t slots_dev, uintptr_t flat, int num_layers, int num_tokens,
           int token_stride, uintptr_t stream) {
    if (num_layers <= 0 || num_tokens <= 0 || token_stride <= 0) return 0;


    if (token_stride % 16 != 0) return static_cast<int>(cudaErrorInvalidValue);

    const long long vec_per_token = token_stride / 16;
    const long long total = (long long)2 * num_layers * num_tokens * vec_per_token;
    const int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (gather) {
        slots_kernel<true><<<(int)blocks, threads, 0, s>>>(
            reinterpret_cast<const unsigned long long*>(k_ptrs_dev),
            reinterpret_cast<const unsigned long long*>(v_ptrs_dev),
            reinterpret_cast<const long long*>(slots_dev),
            reinterpret_cast<unsigned char*>(flat),
            num_layers, num_tokens, token_stride);
    } else {
        slots_kernel<false><<<(int)blocks, threads, 0, s>>>(
            reinterpret_cast<const unsigned long long*>(k_ptrs_dev),
            reinterpret_cast<const unsigned long long*>(v_ptrs_dev),
            reinterpret_cast<const long long*>(slots_dev),
            reinterpret_cast<unsigned char*>(flat),
            num_layers, num_tokens, token_stride);
    }
    return static_cast<int>(cudaGetLastError());
}

}

int gather_slots(uintptr_t k, uintptr_t v, uintptr_t slots, uintptr_t dst,
                 int num_layers, int num_tokens, int token_stride, uintptr_t stream) {
    return launch(true, k, v, slots, dst, num_layers, num_tokens, token_stride, stream);
}

int scatter_slots(uintptr_t k, uintptr_t v, uintptr_t slots, uintptr_t src,
                  int num_layers, int num_tokens, int token_stride, uintptr_t stream) {
    return launch(false, k, v, slots, src, num_layers, num_tokens, token_stride, stream);
}

}
