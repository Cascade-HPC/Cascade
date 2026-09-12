from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

from cascade_sglang.shm_pool import current_geometry
from cascade_sglang.store import CascadeStore, page_keys

logger = logging.getLogger(__name__)


class _Timings:

    __slots__ = ("lock", "get_calls", "get_pages", "get_bytes", "get_seconds",
                 "set_calls", "set_pages", "set_bytes", "set_seconds",
                 "exists_calls", "exists_keys", "exists_seconds", "hit_pages",
                 "miss_pages")

    def __init__(self):
        self.lock = threading.Lock()
        for name in self.__slots__:
            if name != "lock":
                setattr(self, name, 0)

    def as_dict(self) -> dict:
        with self.lock:
            return {name: getattr(self, name) for name in self.__slots__
                    if name != "lock"}


class CascadeHiCacheBackend(HiCacheStorage):

    def __init__(self, storage_config: HiCacheStorageConfig, extra: Optional[dict] = None):
        self.config = storage_config
        cfg = dict(storage_config.extra_config or {})
        if isinstance(extra, dict):
            cfg.update({k: v for k, v in extra.items() if k not in cfg})

        self.tp_rank = int(storage_config.tp_rank)
        self.tp_size = int(storage_config.tp_size)
        self.model_name = storage_config.model_name
        self.is_mla = bool(storage_config.is_mla_model)
        self.timings = _Timings()
        self.geometry = None
        self.mem_pool_host = None
        self.stats_dir = cfg.get("stats_dir", "") or ""
        self._stats_stop = threading.Event()
        if self.stats_dir:
            threading.Thread(target=self._stats_loop, name="cascade-stats",
                             daemon=True).start()

        self.disabled = os.environ.get("CASCADE_SGLANG_DISABLE_STORE", "") in (
            "1", "true", "yes"
        )

        extra_store = {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "gpu_capacity_gb": float(cfg.get("gpu_capacity_gb", 0.0)),
            "dram_capacity_gb": float(cfg.get("dram_capacity_gb", 64.0)),
            "num_gpus_per_node": int(cfg.get("num_gpus_per_node", 4)),
            "gpu_device_offset": cfg.get("gpu_device_offset", "current"),
            "lustre_path": cfg.get("lustre_path", ""),
            "dedup_enabled": bool(cfg.get("dedup_enabled", True)),
            "kv_compression": False,
            "prefix_replication": bool(cfg.get("prefix_replication", True)),
        }
        if self.disabled:
            self.store = None
            logger.warning(
                "Cascade backend attached with the store disabled: hooks fire, "
                "no KV byte moves. This measures the cost of attaching a "
                "backend at all."
            )
        else:
            self.store = CascadeStore(extra_store)
            logger.info(
                "Cascade backend ready: tp_rank=%d tp_size=%d rank=%s world=%s",
                self.tp_rank, self.tp_size,
                getattr(self.store, "rank", "?"),
                getattr(self.store, "world_size", "?"),
            )


    def register_mem_pool_host(self, mem_pool_host) -> None:
        super().register_mem_pool_host(mem_pool_host)
        self.mem_pool_host = mem_pool_host
        if self.disabled:
            return
        self.geometry = current_geometry(mem_pool_host)
        logger.info(
            "Cascade host pool: segment=%s pages=%d page=%d tokens "
            "%.2f MiB layout=%s",
            self.geometry.segment,
            self.geometry.size // self.geometry.page_size,
            self.geometry.page_size,
            self.geometry.page_nbytes() / (1 << 20),
            self.geometry.layout,
        )


    def _keys(self, page_hashes: List[str]) -> List[str]:
        return page_keys(
            page_hashes,
            model_name=self.model_name,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            is_mla=self.is_mla,
        )

    @staticmethod
    def _kv_keys(base_keys: List[str]) -> List[str]:
        out: List[str] = []
        for key in base_keys:
            out.append(key + ":k")
            out.append(key + ":v")
        return out

    def _page_starts(self, host_indices: torch.Tensor, count: int) -> List[int]:
        page = self.geometry.page_size
        flat = host_indices.tolist() if torch.is_tensor(host_indices) else list(host_indices)
        return [int(flat[i * page]) for i in range(count)]


    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        if self.disabled or not keys:
            return 0
        start = time.perf_counter()
        k_keys = [key + ":k" for key in self._keys(keys)]
        try:
            hits = int(self.store.available_prefix(k_keys))
        except Exception as exc:
            logger.warning("Cascade availability lookup failed: %s", exc)
            return 0
        with self.timings.lock:
            self.timings.exists_calls += 1
            self.timings.exists_keys += len(keys)
            self.timings.exists_seconds += time.perf_counter() - start
        return hits

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if self.disabled or not keys:
            return [False] * len(keys)
        start = time.perf_counter()
        starts = self._page_starts(host_indices, len(keys))
        cascade_keys = self._kv_keys(self._keys(keys))
        try:
            sizes = self.store.get_pages(cascade_keys, self.geometry, starts)
        except Exception as exc:
            logger.warning("Cascade page restore failed: %s", exc)
            return [False] * len(keys)

        expect = self.geometry.page_size * self.geometry.layout_dim
        results = [
            sizes[2 * i] == expect and sizes[2 * i + 1] == expect
            for i in range(len(keys))
        ]
        hit = sum(1 for flag in results if flag)
        with self.timings.lock:
            self.timings.get_calls += 1
            self.timings.get_pages += len(keys)
            self.timings.get_bytes += hit * self.geometry.page_nbytes()
            self.timings.get_seconds += time.perf_counter() - start
            self.timings.hit_pages += hit
            self.timings.miss_pages += len(keys) - hit
        return results

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if self.disabled or not keys:
            return [True] * len(keys)
        start = time.perf_counter()
        starts = self._page_starts(host_indices, len(keys))
        cascade_keys = self._kv_keys(self._keys(keys))
        try:
            stored = int(self.store.put_pages(cascade_keys, self.geometry, starts))
        except Exception as exc:
            logger.warning("Cascade page store failed: %s", exc)
            return [False] * len(keys)
        with self.timings.lock:
            self.timings.set_calls += 1
            self.timings.set_pages += len(keys)
            self.timings.set_bytes += len(keys) * self.geometry.page_nbytes()
            self.timings.set_seconds += time.perf_counter() - start
        ok = stored >= len(cascade_keys)
        return [ok] * len(keys)


    def exists(self, key: str) -> bool:
        return self.batch_exists([key]) > 0

    def get(self, key, target_location=None, target_sizes=None):
        raise NotImplementedError(
            "Cascade uses the zero copy interface. Set interface_v1 in the "
            "backend extra config so SGLang calls batch_get_v1."
        )

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        raise NotImplementedError(
            "Cascade uses the zero copy interface. Set interface_v1 in the "
            "backend extra config so SGLang calls batch_get_v1."
        )

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        raise NotImplementedError(
            "Cascade uses the zero copy interface. Set interface_v1 in the "
            "backend extra config so SGLang calls batch_set_v1."
        )

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        raise NotImplementedError(
            "Cascade uses the zero copy interface. Set interface_v1 in the "
            "backend extra config so SGLang calls batch_set_v1."
        )


    def get_stats(self):
        payload: dict[str, Any] = {"transfer": self.timings.as_dict()}
        if self.store is not None:
            try:
                stats = self.store.local_stats()
                payload["tiers"] = {
                    name: getattr(stats, name)
                    for name in dir(stats)
                    if not name.startswith("_")
                }
            except Exception as exc:
                payload["tiers_error"] = str(exc)
        return payload

    def _stats_loop(self) -> None:
        while not self._stats_stop.wait(2.0):
            self._dump_stats()

    def _dump_stats(self) -> None:
        if not self.stats_dir:
            return
        try:
            import pathlib

            node = os.environ.get("SLURM_PROCID", "0")
            directory = pathlib.Path(self.stats_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"cascade_stats_rank{node}_tp{self.tp_rank}.json"
            payload = self.get_stats()
            payload["node_rank"] = int(node)
            payload["tp_rank"] = self.tp_rank
            path.write_text(json.dumps(payload, default=str, indent=2))
        except Exception:
            pass

    def clear(self) -> None:
        if self.store is not None:
            try:
                self.store.flush()
            except Exception:
                pass

    def close(self) -> None:
        try:
            stats = self.get_stats()
            print("CASCADE_SGLANG_STATS tp_rank=%d %s"
                  % (self.tp_rank, json.dumps(stats, default=str)), flush=True)
        except Exception:
            pass
        if self.store is not None:
            try:
                self.store.close()
            except Exception:
                pass
