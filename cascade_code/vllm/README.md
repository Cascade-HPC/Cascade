# CASCADE for vLLM

External KV store for vLLM. KV blocks are held in CASCADE's distributed store
rather than only in GPU memory. The store is in [../cpp](../cpp).

This package implements no cache policy. Tier placement, eviction,
deduplication and RDMA are in the C++ store. No file in the vLLM tree is
modified.

## Contents

| file | lines | role |
|---|---|---|
| `cascade_vllm/connector.py` | 92 | the two classes vLLM resolves by name |
| `cascade_vllm/manager.py` | 194 | scheduler side, reports block availability |
| `cascade_vllm/handler.py` | 562 | worker side, transfers blocks between GPU pages and the store |
| `cascade_vllm/store.py` | 1053 | store wrapper, block key, node rank access |

## Attachment

Through the vLLM V1 offloading interface. Place this directory on `PYTHONPATH`
and declare both classes in the engine's KV transfer config. They resolve from
the same module.

```python
kv_transfer_config = {
    "kv_connector": "CascadeConnectorV1",
    "kv_connector_module_path": "cascade_vllm.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "spec_name": "CascadeOffloadingSpec",
        "spec_module_path": "cascade_vllm.connector",
    },
}
```

Tensor parallelism is supported. Pipeline and data parallelism are not.

## Build and configuration

Build with `build_cpp.sh` at the repository root. Set `CASCADE_CPP_PATH` to the
build directory.

Tier sizes are passed in `kv_connector_extra_config`.

| key | default | meaning |
|---|---|---|
| `num_gpus_per_node` | 1 | GPUs the node's store serves. Set to the tensor parallel size, otherwise the GPU tier is inactive on every rank but the first |
| `dram_capacity_gb` | 64 | DRAM tier |
| `gpu_capacity_gb` | 0 | GPU tier, 0 disables it |
| `lustre_path` | unset | shared filesystem tier |

One setting is an environment variable rather than a config key.

| variable | default | meaning |
|---|---|---|
| `CASCADE_CROSS_LAYER` | 1 | ask vLLM for a block-major KV cache, so one logical block is a single contiguous byte range and the store reads the page itself instead of gathering one fragment per layer. The cost is that attention then reads a layer's consecutive blocks a whole block apart rather than one layer apart, so the trade is worth measuring on a given model and backend |

The remaining `CASCADE_*` variables control the transfer workers and the timing
instrumentation. All have defaults.
