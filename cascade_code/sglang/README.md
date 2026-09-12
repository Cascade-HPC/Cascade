# CASCADE for SGLang

External KV store for SGLang. KV pages are held in CASCADE's distributed store
rather than only in GPU memory. The store is in [../cpp](../cpp) and is shared
with the vLLM adapter in [../vllm](../vllm).

This package implements no cache policy. Tier placement, eviction,
deduplication and RDMA are in the C++ store. No file in the SGLang tree is
modified.

## Contents

| file | lines | role |
|---|---|---|
| `cascade_sglang/hook.py` | 40 | installs the device level cache |
| `cascade_sglang/radix_cache.py` | 556 | device level cache, GPU slot transfers |
| `cascade_sglang/backend.py` | 293 | host level storage backend |
| `cascade_sglang/shm_pool.py` | 182 | host KV pool shared by the node's ranks |
| `cascade_sglang/device_pool.py` | 95 | chunked device transfers |
| `cascade_sglang/phase_timing.py` | 228 | per-phase timings, disabled by default |
| `cascade_sglang/store.py` | 1031 | store wrapper, page key, node rank access |

## Attachment

Two attachment points. Which one is active is a configuration setting.

### Host level

`backend.py` implements the SGLang storage backend interface. It sits below
SGLang's hierarchical cache and fills a host KV pool. SGLang retains the host
to device transfer and CASCADE acts as a backing store rather than as the KV
path. Launch the engine with

```
--hicache-storage-backend dynamic
--hicache-mem-layout page_first
--hicache-storage-backend-extra-config '{"backend_name": "cascade",
    "module_path": "cascade_sglang.backend",
    "class_name": "CascadeHiCacheBackend", "interface_v1": 1}'
```

`interface_v1` makes the SGLang cache controller select its zero copy page
functions, which hand the backend host pool indices rather than materialised
tensors. Three methods carry the work.

| SGLang calls | the store does |
|---|---|
| `batch_exists` | `available_prefix` |
| `batch_get_v1` | `get_batch` into the host pool page |
| `batch_set_v1` | `put_batch` from the host pool page |

In the `page_first` layout a page is two contiguous byte spans, the K span and
the V span, so it needs no gather and no staging buffer. Each span is one entry.
Availability is answered from the K entries alone, because the two are always
written together.

### Device level

`radix_cache.py` replaces the SGLang radix cache. A prefix miss is served
directly into the engine's GPU slots and a finished request is published
directly from them. It overrides the two methods the SGLang device level cache
overrides, which is the only interface exposing the device side KV with its
slot indices.

SGLang selects its cache implementation at run time inside a branch, and the
only device level branch it offers is the one for its LMCache integration.
`hook.py` substitutes the module that branch imports before the scheduler
reaches it, binding the name it looks for to `CascadeRadixCache`. The
constructor signature is unchanged and LMCache itself is never imported.

Set `CASCADE_SGLANG_DEVICE=1` and launch the engine with `--enable-lmcache`,
which in that process selects CASCADE rather than LMCache. State this in any
run configuration so a reader is never misled about which store produced a
number.

## Build and configuration

Build with `build_cpp.sh` at the repository root. Set `CASCADE_CPP_PATH` to the
build directory.

This adapter is configured through the environment. SGLang provides no path for
passing a configuration object.

| variable | default | meaning |
|---|---|---|
| `CASCADE_SGLANG_DEVICE` | 0 | 1 installs the device level cache, otherwise only the host level backend attaches |
| `CASCADE_NUM_GPUS` | 4 | GPUs the node's store serves, set to the tensor parallel size |
| `CASCADE_DRAM_GB` | 64 | DRAM tier |
| `CASCADE_GPU_GB` | 0 | GPU tier, 0 disables it |
| `CASCADE_LUSTRE_PATH` | unset | shared filesystem tier |
| `CASCADE_CHUNK_TOKENS` | 256 | tokens per transfer chunk |

The remaining `CASCADE_*` variables control the transfer workers and the timing
instrumentation. All have defaults.
