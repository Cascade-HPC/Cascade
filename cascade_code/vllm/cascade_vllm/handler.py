import atexit
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import os
import threading
import time

import numpy as np
import torch

from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
try:
    from vllm.v1.kv_offload.spec import CanonicalKVCaches
except ImportError:
    CanonicalKVCaches = object
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from .store import CascadeLoadStoreSpec, make_block_key


logger = logging.getLogger(__name__)


class CascadeOffloadingHandler(OffloadingHandler):

    def __init__(self, config, kv_caches, store, attn_backends=None):
        self._store = store
        self._model = config.model_config.model
        self._tp_size = config.parallel_config.tensor_parallel_size
        self._tp_rank = config.parallel_config.rank
        self._extra = config.kv_transfer_config.kv_connector_extra_config
        self._dtype = self._extra.get(
            "kv_dtype", str(getattr(config.model_config, "dtype", "native"))
        )
        self._layout = self._extra.get("layout", "vllm-canonical-v1")
        self._block_size = config.cache_config.block_size
        self._tensors = []
        self._block_dims = []
        self._page_sizes = []
        transfer_workers = max(
            1,
            int(
                os.environ.get(
                    "OFFLOADING_TRANSFER_WORKERS",
                    os.environ.get("CASCADE_TRANSFER_WORKERS", "1"),
                )
            ),
        )
        self._executor = ThreadPoolExecutor(
            max_workers=transfer_workers, thread_name_prefix="cascade"
        )
        self._futures: dict[int, tuple[Future, tuple[str, str], int, float]] = {}
        self._completed: list[TransferResult] = []
        self._debug_put_count = 0
        self._debug_load_count = 0
        self._poll_sync = os.environ.get("CASCADE_POLL_SYNC", "1") == "1"

        if getattr(config.parallel_config, "pipeline_parallel_size", 1) != 1 or getattr(
            config.parallel_config, "data_parallel_size", 1
        ) != 1:
            raise ValueError(
                "Cascade vLLM integration currently requires PP=1 and DP=1"
            )

        if isinstance(kv_caches, dict):
            if not attn_backends:
                raise ValueError(
                    "raw vLLM KV caches require attention backend metadata"
                )
            for layer_name, gpu_tensor in kv_caches.items():
                backend = attn_backends[layer_name]
                test_shape = backend.get_kv_cache_shape(
                    num_blocks=1234,
                    block_size=16,
                    num_kv_heads=8,
                    head_size=256,
                )
                has_layers_dim = len(gpu_tensor.shape) != len(test_shape)
                split_k_and_v = False
                if has_layers_dim:
                    if len(gpu_tensor.shape) != len(test_shape) + 1:
                        raise ValueError(
                            f"Unsupported Cascade KV layout for {layer_name}: "
                            f"tensor={tuple(gpu_tensor.shape)} backend={test_shape}"
                        )
                    try:
                        stride_order = backend.get_kv_cache_stride_order(
                            include_num_layers_dimension=True
                        )
                        physical_shape = (80,) + test_shape
                        test_shape = tuple(physical_shape[i] for i in stride_order)
                    except (AttributeError, NotImplementedError):
                        test_shape = (80,) + test_shape
                elif test_shape[0] != 1234:
                    if test_shape[0] != 2 or test_shape[1] != 1234:
                        raise ValueError(
                            f"Unsupported Cascade KV layout for {layer_name}: "
                            f"tensor={tuple(gpu_tensor.shape)} backend={test_shape}"
                        )
                    split_k_and_v = True

                block_dim = test_shape.index(1234)
                tensors = gpu_tensor.unbind(0) if split_k_and_v else (gpu_tensor,)
                for tensor in tensors:
                    self._tensors.append(tensor)
                    self._block_dims.append(block_dim - 1 if split_k_and_v else block_dim)
        else:
            for cache in kv_caches.tensors:
                tensor = cache.tensor.view(torch.int8).view(-1, cache.page_size_bytes)
                self._tensors.append(tensor)
                self._block_dims.append(0)
                self._page_sizes.append(cache.page_size_bytes)

        if not self._page_sizes:
            self._page_sizes = [
                tensor.select(block_dim, 0).numel() * tensor.element_size()
                for tensor, block_dim in zip(self._tensors, self._block_dims)
            ]

        if not self._tensors:
            raise ValueError("Cascade handler received no canonical KV tensors")

        self._block_bytes = None
        if len(self._tensors) == 1 and self._block_dims[0] == 0:
            tensor = self._tensors[0]
            if tensor.is_contiguous():
                self._block_bytes = tensor.view(torch.uint8).reshape(
                    tensor.shape[0], -1
                )
        self._page_total = sum(self._page_sizes)
        self._device = self._tensors[0].device

        self._metrics = {
            "store_jobs": 0,
            "store_blocks": 0,
            "store_bytes": 0,
            "store_service_s": 0.0,
            "store_wait_s": 0.0,
            "load_jobs": 0,
            "load_blocks": 0,
            "store_dedup_blocks": 0,
            "store_dropped_blocks": 0,
            "load_bytes": 0,
            "load_service_s": 0.0,
            "load_native_s": 0.0,
            "load_scatter_s": 0.0,
            "queue_s": 0.0,

            "lookup_s": 0.0,
            "lookup_calls": 0,
            "_load_spans": [],
            "_store_spans": [],
        }
        self._metrics_lock = threading.Lock()
        self._metrics_path = os.environ.get("CASCADE_METRICS_PATH")
        self._metrics_interval = float(
            os.environ.get("CASCADE_METRICS_INTERVAL_S", "1.0")
        )
        atexit.register(self._report_metrics)
        if self._metrics_path:
            threading.Thread(
                target=self._metrics_loop, name="cascade-metrics", daemon=True
            ).start()

    def _key(self, block_hash) -> str:
        return make_block_key(
            model=self._model,
            block_hash=block_hash,
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
            dtype=self._dtype,
            layout=self._layout,
            block_size=self._block_size,
        )

    def _page_bytes(self, block_id: int) -> torch.Tensor:
        if self._block_bytes is not None:
            return self._block_bytes[block_id]
        pages = [
            tensor.select(self._block_dims[index], block_id).detach()
            for index, tensor in enumerate(self._tensors)
        ]
        if len(pages) == 1 and pages[0].is_contiguous():
            return pages[0].view(torch.uint8).reshape(-1)

        payload = torch.empty(
            sum(page.numel() * page.element_size() for page in pages),
            dtype=torch.uint8,
            device=pages[0].device,
        )
        offset = 0
        for page in pages:
            page_bytes = page.contiguous().view(torch.uint8).reshape(-1)
            payload[offset : offset + page_bytes.numel()].copy_(
                page_bytes, non_blocking=True
            )
            offset += page_bytes.numel()
        return payload

    def _unpack(self, block_id: int, payload: np.ndarray) -> None:
        offset = 0
        for index, (tensor, page_size) in enumerate(
            zip(self._tensors, self._page_sizes)
        ):
            end = offset + page_size
            if end > len(payload):
                raise ValueError(
                    f"Cascade payload is too small: need {end}, got {len(payload)}"
                )
            page = payload[offset:end]
            source = torch.from_numpy(page)
            target = tensor.select(self._block_dims[index], block_id)
            target.view(torch.uint8).reshape(-1).copy_(source, non_blocking=True)
            offset = end
        if offset != len(payload):
            raise ValueError(
                f"Cascade payload has trailing bytes: consumed {offset}, "
                f"got {len(payload)}"
            )
        if torch.cuda.is_available():
            torch.cuda.current_stream(device=self._tensors[0].device).synchronize()

    def _unpack_device(self, block_id: int, payload: torch.Tensor) -> None:
        payload = payload.view(torch.uint8).reshape(-1)
        offset = 0
        for index, (tensor, page_size) in enumerate(
            zip(self._tensors, self._page_sizes)
        ):
            end = offset + page_size
            if end > payload.numel():
                raise ValueError(
                    f"Cascade device payload is too small: need {end}, got {payload.numel()}"
                )
            page = payload[offset:end]
            target = tensor.select(self._block_dims[index], block_id)
            target.view(torch.uint8).reshape(-1).copy_(page, non_blocking=True)
            offset = end
        if offset != payload.numel():
            raise ValueError(
                f"Cascade device payload has trailing bytes: consumed {offset}, "
                f"got {payload.numel()}"
            )

    def _store_blocks(self, src: GPULoadStoreSpec, dst: CascadeLoadStoreSpec,
                      ready: "torch.cuda.Event | None" = None):
        if len(src.block_ids) != len(dst.block_hashes):
            raise ValueError("GPU and Cascade block counts do not match")
        keys = [self._key(block_hash) for block_hash in dst.block_hashes]
        payloads = [
            self._page_bytes(int(block_id)) for block_id in src.block_ids.tolist()
        ]
        wait_started = time.perf_counter()
        if ready is not None:
            if self._poll_sync:
                while not ready.query():
                    time.sleep(0.0002)
            else:
                ready.synchronize()
        else:
            torch.cuda.current_stream(device=self._device).synchronize()
        wait_seconds = time.perf_counter() - wait_started
        stored = self._store.put_device_many(keys, payloads, is_prefix=True)
        if stored != len(keys):
            lookup_started = time.perf_counter()
            present = self._store.available_batch(keys)
            with self._metrics_lock:
                self._metrics["lookup_s"] += time.perf_counter() - lookup_started
                self._metrics["lookup_calls"] += 1
            missing = [key for key, ok in zip(keys, present) if not ok]
            self._metrics["store_dedup_blocks"] += len(keys) - stored - len(missing)
            if missing:
                self._metrics["store_dropped_blocks"] += len(missing)
                if stored == 0:
                    raise RuntimeError(
                        f"Cascade put_batch stored nothing of {len(keys)} blocks"
                    )
                if self._metrics["store_dropped_blocks"] <= 8:
                    print(
                        f"Cascade could not admit {len(missing)}/{len(keys)} "
                        f"blocks, they will not be reusable "
                        f"(dropped so far {self._metrics['store_dropped_blocks']})",
                        flush=True,
                    )
        if (
            (os.environ.get("CASCADE_REMOTE_REUSE", "0") == "1"
             or os.environ.get("REMOTE_REUSE", "0") == "1")
            and self._debug_put_count < 100
        ):
            print(
                "Cascade handler put "
                f"rank={self._store.rank} blocks={len(keys)} "
                f"first_key={keys[0] if keys else 'none'}",
                flush=True,
            )
            self._debug_put_count += 1
        with self._metrics_lock:
            self._metrics["store_jobs"] += 1
            self._metrics["store_blocks"] += len(keys)
            self._metrics["store_bytes"] += len(keys) * self._page_total
            self._metrics["store_wait_s"] += wait_seconds

    def _load_blocks(self, src: CascadeLoadStoreSpec, dst: GPULoadStoreSpec):
        if len(src.block_hashes) != len(dst.block_ids):
            raise ValueError("Cascade and GPU block counts do not match")
        keys = [self._key(block_hash) for block_hash in src.block_hashes]
        output_buffers = []
        fallback_targets = []
        for block_id in dst.block_ids.tolist():
            if self._block_bytes is not None:
                output_buffers.append(self._block_bytes[int(block_id)])
                fallback_targets.append(None)
                continue
            pages = [
                tensor.select(self._block_dims[index], int(block_id)).detach()
                for index, tensor in enumerate(self._tensors)
            ]
            if len(pages) == 1 and pages[0].is_contiguous():
                output_buffers.append(pages[0].view(torch.uint8).reshape(-1))
                fallback_targets.append(None)
                continue
            size = sum(page.numel() * page.element_size() for page in pages)
            temporary = torch.empty(size, dtype=torch.uint8, device=pages[0].device)
            output_buffers.append(temporary)
            fallback_targets.append((pages, temporary))
        native_started = time.perf_counter()
        success, sizes = self._store.get_device_many(keys, output_buffers)
        native_seconds = time.perf_counter() - native_started
        if success != len(keys):
            raise RuntimeError(
                f"Cascade native device get returned {success}/{len(keys)} blocks"
            )
        if (
            (os.environ.get("CASCADE_REMOTE_REUSE", "0") == "1"
             or os.environ.get("REMOTE_REUSE", "0") == "1")
            and self._debug_load_count < 100
        ):
            print(
                "Cascade handler load "
                f"rank={self._store.rank} blocks={len(keys)} "
                f"first_key={keys[0] if keys else 'none'}",
                flush=True,
            )
            self._debug_load_count += 1
        scatter_started = time.perf_counter()
        needs_scatter = any(target is not None for target in fallback_targets)
        if needs_scatter:
            for fallback in fallback_targets:
                if fallback is None:
                    continue
                pages, payload = fallback
                offset = 0
                for page in pages:
                    page_bytes = page.numel() * page.element_size()
                    source = payload[offset : offset + page_bytes].view(page.dtype)
                    page.copy_(source.reshape(page.shape), non_blocking=True)
                    offset += page_bytes
            torch.cuda.current_stream(device=self._device).synchronize()
        scatter_seconds = time.perf_counter() - scatter_started
        with self._metrics_lock:
            self._metrics["load_jobs"] += 1
            self._metrics["load_blocks"] += len(keys)
            self._metrics["load_bytes"] += sum(int(size) for size in sizes)
            self._metrics["load_native_s"] += native_seconds
            self._metrics["load_scatter_s"] += scatter_seconds

    def _run_transfer(self, spec: TransferSpec, ready=None):
        service_started = time.perf_counter()
        src, dst = spec
        if isinstance(src, GPULoadStoreSpec) and isinstance(
            dst, CascadeLoadStoreSpec
        ):
            self._store_blocks(src, dst, ready)
            transfer_type = ("GPU", dst.medium())
            num_bytes = len(src.block_ids) * self._page_total
            key = "store_service_s"
        elif isinstance(src, CascadeLoadStoreSpec) and isinstance(
            dst, GPULoadStoreSpec
        ):
            self._load_blocks(src, dst)
            transfer_type = (src.medium(), "GPU")
            num_bytes = len(dst.block_ids) * self._page_total
            key = "load_service_s"
        else:
            raise TypeError(f"Unsupported transfer {type(src)} -> {type(dst)}")
        service_ended = time.perf_counter()
        service_seconds = service_ended - service_started
        with self._metrics_lock:
            bucket = ("_load_spans" if key == "load_service_s"
                      else "_store_spans")
            self._metrics[bucket].append((service_started, service_ended))
            self._metrics[key] += service_seconds
        return transfer_type, num_bytes, service_seconds

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        started = time.perf_counter()
        src, dst = spec
        ready = None
        if isinstance(src, GPULoadStoreSpec) and isinstance(dst, CascadeLoadStoreSpec):
            transfer_type = ("GPU", dst.medium())
            num_bytes = len(src.block_ids) * self._page_total
            if torch.cuda.is_available():
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device=self._device))
        elif isinstance(src, CascadeLoadStoreSpec) and isinstance(dst, GPULoadStoreSpec):
            transfer_type = (src.medium(), "GPU")
            num_bytes = len(dst.block_ids) * self._page_total
        else:
            raise TypeError(f"Unsupported transfer {type(src)} -> {type(dst)}")

        future = self._executor.submit(self._run_transfer, spec, ready)
        self._futures[job_id] = (future, transfer_type, num_bytes, started)
        return True

    def _finish(self, job_id: int) -> TransferResult:
        future, transfer_type, num_bytes, started = self._futures.pop(job_id)
        service_time = None
        try:
            transfer_result = future.result()
            if isinstance(transfer_result, tuple) and len(transfer_result) == 3:
                service_time = float(transfer_result[2])
            success = True
        except Exception:
            logger.exception(
                "Cascade transfer failed: job_id=%s type=%s bytes=%s",
                job_id,
                transfer_type,
                num_bytes,
            )
            success = False
        transfer_time = time.perf_counter() - started
        if service_time is not None:
            with self._metrics_lock:
                self._metrics["queue_s"] += max(0.0, transfer_time - service_time)
        if os.environ.get("CASCADE_TRANSFER_DEBUG", "0") == "1":
            queue_time = (
                max(0.0, transfer_time - service_time)
                if service_time is not None
                else -1.0
            )
            print(
                "Cascade transfer "
                f"job={job_id} type={transfer_type} bytes={num_bytes} "
                f"seconds={transfer_time:.6f} "
                f"service_seconds={service_time if service_time is not None else -1.0:.6f} "
                f"queue_seconds={queue_time:.6f} success={success}",
                flush=True,
            )
        return TransferResult(
            job_id=job_id,
            success=success,
            transfer_size=num_bytes,
            transfer_time=transfer_time,
            transfer_type=transfer_type,
        )

    def get_finished(self):
        for job_id, (future, _, _, _) in list(self._futures.items()):
            if future.done():
                self._completed.append(self._finish(job_id))
        completed = self._completed
        self._completed = []
        return completed

    def wait(self, job_ids: set[int]) -> None:
        for job_id in list(job_ids):
            if job_id in self._futures:
                self._completed.append(self._finish(job_id))

    def _metrics_loop(self) -> None:
        while True:
            time.sleep(self._metrics_interval)
            with self._metrics_lock:
                active = self._metrics["load_jobs"] or self._metrics["store_jobs"]
            if active:
                self._write_metrics()

    def _write_metrics(self) -> None:
        try:
            metrics = self._collect_metrics()
            node = os.environ.get("SLURM_PROCID", "0")
            target = f"{self._metrics_path}.node{node}.tp{self._tp_rank}.json"
            temporary = f"{target}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            os.replace(temporary, target)
        except OSError:
            pass

    @staticmethod
    def _span_summary(spans: list, prefix: str) -> dict:
        if not spans:
            return {f"{prefix}_busy_s": 0.0, f"{prefix}_span_s": 0.0,
                    f"{prefix}_concurrency": 0.0, f"{prefix}_serial_s": 0.0,
                    f"{prefix}_peak_inflight": 0}
        busy = sum(end - start for start, end in spans)
        merged, union = [], 0.0
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            union += end - start
        events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
        peak = live = 0
        for _, delta in events:
            live += delta
            peak = max(peak, live)
        concurrency = busy / union if union else 0.0
        return {f"{prefix}_busy_s": busy, f"{prefix}_span_s": union,
                f"{prefix}_concurrency": concurrency,
                f"{prefix}_serial_s": busy / concurrency if concurrency else 0.0,
                f"{prefix}_peak_inflight": peak}

    def _collect_metrics(self) -> dict:
        with self._metrics_lock:
            metrics = dict(self._metrics)
        metrics.update(self._span_summary(metrics.pop("_load_spans", []), "load"))
        metrics.update(self._span_summary(metrics.pop("_store_spans", []), "store"))
        metrics["tp_rank"] = self._tp_rank
        metrics["page_bytes"] = self._page_total
        try:
            stats = self._store.local_stats()
            for field in (
                "local_gpu_hits",
                "local_dram_hits",
                "remote_gpu_hits",
                "remote_dram_hits",
                "lustre_hits",
                "misses",
                "dedup_hits",
                "prefetched_blocks",
                "prefetch_hits",
                "local_gpu_used",
                "local_dram_used",
                "gpu_evictions",
                "dram_evictions",
                "total_blocks",
            ):
                metrics[field] = int(getattr(stats, field, 0))
        except Exception as exc:
            metrics["stats_error"] = f"{type(exc).__name__}: {exc}"
        native = metrics.get("load_native_s", 0.0)
        local = int(metrics.get("local_dram_hits", 0) or 0) + int(
            metrics.get("local_gpu_hits", 0) or 0)
        remote = int(metrics.get("remote_dram_hits", 0) or 0) + int(
            metrics.get("remote_gpu_hits", 0) or 0) + int(
            metrics.get("lustre_hits", 0) or 0)
        served = local + remote
        metrics["cache_s"] = native * local / served if served else 0.0
        metrics["transfer_s"] = native * remote / served if served else 0.0
        return metrics

    def _report_metrics(self) -> None:
        metrics = self._collect_metrics()
        print("CASCADE_HANDLER_METRICS " + json.dumps(metrics), flush=True)
        self._write_metrics()
