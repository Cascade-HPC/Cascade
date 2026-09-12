#pragma once
#include <cstdint>

namespace cascade {


int gather_slots(uintptr_t k_ptrs_dev, uintptr_t v_ptrs_dev, uintptr_t slots_dev,
                 uintptr_t dst, int num_layers, int num_tokens, int token_stride,
                 uintptr_t stream);

int scatter_slots(uintptr_t k_ptrs_dev, uintptr_t v_ptrs_dev, uintptr_t slots_dev,
                  uintptr_t src, int num_layers, int num_tokens, int token_stride,
                  uintptr_t stream);

}
