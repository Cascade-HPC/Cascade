from __future__ import annotations

import logging
import os
import threading
import time

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.hicache_storage import get_hash_str

from cascade_sglang.device_pool import DeviceChunkMover
from cascade_sglang.store import page_keys

logger = logging.getLogger(__name__)


class _Counters:
    __slots__ = ("lock", "get_calls", "get_chunks", "get_bytes", "get_seconds",
                 "put_calls", "put_chunks", "put_bytes", "put_seconds",
                 "lookup_calls", "lookup_seconds", "hit_chunks", "miss_chunks")

    def __init__(self):
        self.lock = threading.Lock()
        for n in self.__slots__:
            if n != "lock":
                setattr(self, n, 0)

    def as_dict(self):
        with self.lock:
            return {n: getattr(self, n) for n in self.__slots__ if n != "lock"}


class CascadeRadixCache(RadixCache):

    def __init__(self, params, model_config=None, tp_size: int = 1,
                 rank: int = 0, tp_group=None):
        super().__init__(params)
        from cascade_sglang.store import CascadeStore

        self.tp_size = int(tp_size)
        self.tp_rank = int(rank)
        self.tp_group = tp_group
        self._tp_pg = None
        self.model_name = getattr(model_config, "model_path", None) if model_config else None
        self.counters = _Counters()
        globals()["COUNTERS"] = self.counters
        self._local = threading.local()
        try:
            self._cuda_index = torch.cuda.current_device()
        except Exception:
            self._cuda_index = 0
        self.chunk_tokens = int(os.environ.get("CASCADE_CHUNK_TOKENS", "256"))
        self.stats_dir = os.environ.get("CASCADE_STATS_DIR", "")
        self.enabled = os.environ.get("CASCADE_SGLANG_DISABLE_STORE", "") not in ("1", "true", "yes")

        self.store = None
        self.mover = None
        if self.enabled:
            extra = {
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
                "gpu_capacity_gb": float(os.environ.get("CASCADE_GPU_GB", "0")),
                "dram_capacity_gb": float(os.environ.get("CASCADE_DRAM_GB", "64")),
                "num_gpus_per_node": int(os.environ.get("CASCADE_NUM_GPUS", "4")),
                "gpu_device_offset": "current",
                "lustre_path": os.environ.get("CASCADE_LUSTRE_PATH", ""),
                "prefix_replication": os.environ.get(
                    "CASCADE_PREFIX_REPLICATION", "0") in ("1", "true", "yes"),
                "kv_compression": False,
            }

            pg = (getattr(self.tp_group, "cpu_group", None)
                  or getattr(self.tp_group, "device_group", None)) if self.tp_group else None
            self._tp_pg = pg

            def _wait():
                if pg is None or self.tp_size <= 1:
                    return
                try:
                    torch.distributed.barrier(group=pg)
                except Exception as exc:
                    logger.warning("Cascade startup barrier failed: %s", exc)

            if self.tp_rank == 0:
                self.store = CascadeStore(extra)
                _wait()
            else:
                _wait()
                self.store = CascadeStore(extra)

            try:
                from cascade_sglang.store import CascadeStore as _CS
                logger.info("Cascade rpc socket tp_rank=%d procid=%s path=%s owner=%s",
                            self.tp_rank, os.environ.get("SLURM_PROCID"),
                            _CS._rpc_socket_path(), self.tp_rank == 0)
            except Exception:
                pass

            self.mover = DeviceChunkMover(
                self.token_to_kv_pool_allocator.get_kvcache(),
                self.store._cpp, self.chunk_tokens, self.device,
            )
            try:
                from cascade_sglang.store import CascadeStore as _CS
                logger.info("Cascade rpc socket tp_rank=%d procid=%s path=%s owner=%s",
                            self.tp_rank, os.environ.get("SLURM_PROCID"),
                            _CS._rpc_socket_path(), self.tp_rank == 0)
            except Exception:
                pass
            logger.info(
                "Cascade device cache ready: tp_rank=%d chunk=%d tokens "
                "%.2f MiB layers=%d token_stride=%d",
                self.tp_rank, self.chunk_tokens,
                self.mover.chunk_bytes / (1 << 20),
                self.mover.layer_num, self.mover.token_stride,
            )
        import queue as _queue

        self._publish_q = _queue.Queue()
        self._buf_lock = threading.Lock()
        self._buf_free: list = []
        self._buf_base: dict = {}
        self._buf_live = 0
        self._buf_max = int(os.environ.get("CASCADE_PUBLISH_BUFFERS", "3"))
        self._buf_chunks = max(
            int(os.environ.get("CASCADE_PUBLISH_GROUP", "4")),
            int(os.environ.get("CASCADE_FETCH_GROUP", "4")))
        try:
            free_bytes = torch.cuda.mem_get_info(self.mover.device)[0]
            share = float(os.environ.get("CASCADE_STAGING_SHARE", "0.12"))
            if free_bytes < 1.5 * (1 << 30):
                share = min(share, 0.05)
            afford = int(free_bytes * share) // self.mover.chunk_bytes
            want = max(self._buf_chunks, self._chunks_per_request())
            self._buf_chunks = max(1, min(want, afford // self._buf_max))
            if afford < self._buf_max:
                self._buf_max = max(1, afford)
                self._buf_chunks = 1
            logger.info(
                "Cascade staging pool: %d buffers x %d chunks = %.0f MiB, "
                "free was %.2f GiB",
                self._buf_max, self._buf_chunks,
                self._buf_max * self._buf_chunks
                * self.mover.chunk_bytes / (1 << 20),
                free_bytes / (1 << 30))
        except Exception as exc:
            logger.warning("Cascade staging pool sizing failed, using the "
                           "configured pool: %s", exc)
        self._publish_thread = threading.Thread(
            target=self._publish_loop, daemon=True, name="cascade-publish")
        self._publish_thread.start()

        if self.stats_dir:
            threading.Thread(target=self._stats_loop, daemon=True,
                             name="cascade-stats").start()


    def _chunk_keys(self, token_ids, start_chunk: int, count: int, prior: str | None):
        hashes, h = [], prior
        for c in range(start_chunk, start_chunk + count):
            lo, hi = c * self.chunk_tokens, (c + 1) * self.chunk_tokens
            h = get_hash_str(list(token_ids[lo:hi]), h)
            hashes.append(h)
        return page_keys(hashes, model_name=self.model_name,
                         tp_rank=self.tp_rank, tp_size=self.tp_size), h

    def _chunks_per_request(self) -> int:
        try:
            tokens = int(getattr(self, "_max_request_tokens", 0)) or int(
                os.environ.get("CASCADE_MAX_PREFIX_TOKENS", "0"))
            if tokens <= 0:
                tokens = int(self.token_to_kv_pool_allocator.size)
            return max(1, tokens // self.chunk_tokens)
        except Exception:
            return self._buf_chunks

    def _agree(self, value: int) -> int:
        pg = self._tp_pg
        if pg is None or self.tp_size <= 1:
            return value
        try:
            dev = "cpu"
            try:
                if torch.distributed.get_backend(pg) == "nccl":
                    dev = self.mover.device
            except Exception:
                pass
            t = torch.tensor([int(value)], dtype=torch.int64, device=dev)
            torch.distributed.all_reduce(
                t, op=torch.distributed.ReduceOp.MIN, group=pg)
            return int(t.item())
        except Exception as exc:
            logger.warning("Cascade rank agreement failed, serving nothing "
                           "this step: %s", exc)
            return 0


    def match_prefix(self, params) -> MatchResult:
        base = super().match_prefix(params)
        if not self.enabled or self.disable:
            return base
        if getattr(self._local, "in_cascade", False):
            return base
        key = params.key
        if not key:
            return base

        matched = base.device_indices.numel()
        total = len(key)
        first_chunk = matched // self.chunk_tokens
        avail_chunks = total // self.chunk_tokens
        want = avail_chunks - first_chunk
        if want <= 0:
            return base

        token_ids = list(key.token_ids) if hasattr(key, "token_ids") else list(key)

        started = time.perf_counter()
        keys, _ = self._chunk_keys(token_ids, 0, avail_chunks, None)
        try:
            present = int(self.store.available_prefix(keys))
        except Exception as exc:
            logger.warning("Cascade lookup failed: %s", exc)
            present = 0
        with self.counters.lock:
            self.counters.lookup_calls += 1
            self.counters.lookup_seconds += time.perf_counter() - started

        usable = min(max(present - first_chunk, 0), want)
        usable = self._agree(usable)
        if usable <= 0:
            with self.counters.lock:
                self.counters.miss_chunks += want
            return base

        need = usable * self.chunk_tokens
        if self.token_to_kv_pool_allocator.available_size() < need:
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams
            self.evict(EvictParams(num_tokens=need))
        slots = self.token_to_kv_pool_allocator.alloc(need)

        started = time.perf_counter()
        fetched = 0
        if slots is not None:
            try:
                fetched = self._fetch(keys[first_chunk:first_chunk + usable], slots)
            except Exception as exc:
                logger.warning("Cascade fetch failed: %s", exc)
                fetched = 0
        fetched = max(0, min(fetched, need))

        fetched = self._agree(fetched)
        fetched -= fetched % self.chunk_tokens

        if fetched <= 0:
            if slots is not None:
                self.token_to_kv_pool_allocator.free(slots)
            return base
        if fetched < need:
            self.token_to_kv_pool_allocator.free(slots[fetched:])
            slots = slots[:fetched]

        with self.counters.lock:
            self.counters.get_calls += 1
            self.counters.get_chunks += fetched // self.chunk_tokens
            self.counters.get_bytes += (fetched // self.chunk_tokens) * self.mover.chunk_bytes
            self.counters.get_seconds += time.perf_counter() - started
            self.counters.hit_chunks += fetched // self.chunk_tokens

        last_node = base.last_device_node
        node = TreeNode(priority=getattr(last_node, "priority", 0))
        node.key = key[matched:matched + fetched]
        node.value = slots
        node.parent = last_node
        last_node.children[self.get_child_key_fn(node.key)] = node
        self.evictable_size_ += fetched

        return MatchResult(
            device_indices=torch.cat([base.device_indices, slots]),
            last_device_node=node,
            last_host_node=node,
        )

    def _fetch(self, keys, slots) -> int:
        group = self._buf_chunks
        done = 0
        for start in range(0, len(keys), group):
            count = min(group, len(keys) - start)
            buf = self._take_buffer(count)
            if buf is None:
                break
            try:
                views = self.mover.chunk_views(buf, count)
                _t_wire = time.perf_counter()
                _, sizes = self.store.get_device_many(list(keys[start:start + count]), views)
                _wire = time.perf_counter() - _t_wire
                good = 0
                for size in sizes:
                    if int(size) != self.mover.chunk_bytes:
                        break
                    good += 1
                if good == 0:
                    break
                lo = done
                hi = done + good * self.chunk_tokens
                _t_load = time.perf_counter()
                self.mover.scatter(slots[lo:hi].to(torch.int64), buf)
                self.mover.sync()
                _load = time.perf_counter() - _t_load
                try:
                    from cascade_sglang import phase_timing as _pt
                    _pt._add("wire_seconds", _wire)
                    _pt._add("wire_calls", 1)
                    _pt._add("load_seconds", _load)
                    _pt._add("load_calls", 1)
                except Exception:
                    pass
                done = hi
                if good < count:
                    break
            finally:
                self._return_buffer(buf)
        return done

    def cache_finished_req(self, req, is_insert: bool = True) -> None:
        super().cache_finished_req(req, is_insert=is_insert)
        if not self.enabled or self.disable:
            return
        try:
            self._enqueue_publish(req)
        except Exception as exc:
            logger.warning("Cascade publish enqueue failed: %s", exc)

    def _enqueue_publish(self, req) -> None:
        from sglang.srt.server_args import get_global_server_args
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        topk = get_global_server_args().speculative_eagle_topk
        if topk is None or topk == 1:
            committed = int(req.kv_committed_len)
        else:
            committed = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

        token_ids = (list(req.origin_input_ids) + list(req.output_ids))[:committed]
        chunks = len(token_ids) // self.chunk_tokens
        if chunks <= 0:
            return

        indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : chunks * self.chunk_tokens
        ].clone()

        node = None
        try:
            node = super().match_prefix(
                MatchPrefixParams(key=RadixKey(token_ids, req.extra_key))
            ).last_device_node
            if node is not None:
                self.inc_lock_ref(node)
        except Exception:
            node = None

        group = self._buf_chunks
        keys, _ = self._chunk_keys(token_ids, 0, chunks, None)
        try:
            for start in range(0, chunks, group):
                count = min(group, chunks - start)
                buf = self._take_buffer(count)
                if buf is None:
                    logger.warning("Cascade publish dropped: no staging buffer for "
                                   "%d chunks", count)
                    break
                views = self.mover.chunk_views(buf, count)
                began = time.perf_counter()
                for c in range(count):
                    lo = (start + c) * self.chunk_tokens
                    hi = lo + self.chunk_tokens
                    self.mover.gather(indices[lo:hi].to(torch.int64), views[c])
                self.mover.sync()
                with self.counters.lock:
                    self.counters.put_seconds += time.perf_counter() - began
                self._publish_q.put((keys[start:start + count], buf, count))
        finally:
            if node is not None:
                try:
                    self.dec_lock_ref(node)
                except Exception:
                    pass
        return

    def _take_buffer(self, chunks: int):
        want = chunks * self.mover.chunk_bytes
        full = self._buf_chunks * self.mover.chunk_bytes
        if want > full:
            return None
        deadline = time.time() + 30.0
        while True:
            with self._buf_lock:
                if self._buf_free:
                    return self._buf_free.pop()[:want]
                if self._buf_live < self._buf_max:
                    self._buf_live += 1
                    try:
                        base = torch.empty(full, dtype=torch.uint8,
                                           device=self.mover.device)
                    except Exception:
                        self._buf_live -= 1
                        return None
                    self._buf_base[base.data_ptr()] = base
                    return base[:want]
            if time.time() >= deadline:
                logger.warning("Cascade publish buffers exhausted, dropping a publish")
                return None
            time.sleep(0.005)

    def _return_buffer(self, buf) -> None:
        with self._buf_lock:
            base = self._buf_base.get(buf.data_ptr())
            self._buf_free.append(base if base is not None else buf)

    def _publish_loop(self) -> None:
        try:
            torch.cuda.set_device(self._cuda_index)
            logger.info("Cascade publish thread bound to cuda:%d (tp_rank=%d)",
                        self._cuda_index, self.tp_rank)
        except Exception as exc:
            logger.warning("Cascade publish thread could not set device: %s", exc)
        while True:
            item = self._publish_q.get()
            if item is None:
                return
            keys, buf, chunks = item
            try:
                started = time.perf_counter()
                views = self.mover.chunk_views(buf, chunks)
                stored = int(self.store.put_device_many(list(keys), views))
                with self.counters.lock:
                    self.counters.put_calls += 1
                    self.counters.put_chunks += chunks
                    self.counters.put_bytes += chunks * self.mover.chunk_bytes
                    self.counters.put_seconds += time.perf_counter() - started
                if stored < chunks:
                    logger.debug("Cascade stored %d of %d chunks", stored, chunks)
            except Exception as exc:
                logger.warning("Cascade publish failed: %s", exc)
            finally:
                self._return_buffer(buf)

    def _publish_chunks(self, token_ids, indices, chunks) -> None:
        keys, _ = self._chunk_keys(token_ids, 0, chunks, None)
        started = time.perf_counter()
        buf = self.mover.staging(chunks)
        views = self.mover.chunk_views(buf, chunks)
        for c in range(chunks):
            lo, hi = c * self.chunk_tokens, (c + 1) * self.chunk_tokens
            self.mover.gather(indices[lo:hi].to(torch.int64), views[c])
        self.mover.sync()
        stored = int(self.store.put_device_many(list(keys), views))
        with self.counters.lock:
            self.counters.put_calls += 1
            self.counters.put_chunks += chunks
            self.counters.put_bytes += chunks * self.mover.chunk_bytes
            self.counters.put_seconds += time.perf_counter() - started
        if stored < chunks:
            logger.debug("Cascade stored %d of %d chunks", stored, chunks)

    def drain_publishes(self, timeout: float = 600.0) -> None:
        deadline = time.time() + timeout
        while not self._publish_q.empty() and time.time() < deadline:
            time.sleep(0.05)

    def _publish_old(self, req) -> None:
        from sglang.srt.server_args import get_global_server_args

        topk = get_global_server_args().speculative_eagle_topk
        if topk is None or topk == 1:
            committed = int(req.kv_committed_len)
        else:
            committed = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

        token_ids = (list(req.origin_input_ids) + list(req.output_ids))[:committed]
        chunks = len(token_ids) // self.chunk_tokens
        if chunks <= 0:
            return
        indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :committed]
        keys, _ = self._chunk_keys(token_ids, 0, chunks, None)

        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        node = None
        try:
            node = self.match_prefix(
                MatchPrefixParams(key=RadixKey(token_ids, req.extra_key))
            ).last_device_node
            if node is not None:
                self.inc_lock_ref(node)
        except Exception:
            node = None

        started = time.perf_counter()
        try:
            buf = self.mover.staging(chunks)
            views = self.mover.chunk_views(buf, chunks)
            for c in range(chunks):
                lo, hi = c * self.chunk_tokens, (c + 1) * self.chunk_tokens
                self.mover.gather(indices[lo:hi].to(torch.int64), views[c])
            self.mover.sync()
            stored = int(self.store.put_device_many(list(keys), views))
        finally:
            if node is not None:
                try:
                    self.dec_lock_ref(node)
                except Exception:
                    pass
        with self.counters.lock:
            self.counters.put_calls += 1
            self.counters.put_chunks += chunks
            self.counters.put_bytes += chunks * self.mover.chunk_bytes
            self.counters.put_seconds += time.perf_counter() - started
        if stored < chunks:
            logger.debug("Cascade stored %d of %d chunks", stored, chunks)


    def get_cascade_stats(self):
        payload = {"transfer": self.counters.as_dict()}
        if self.store is not None:
            try:
                s = self.store.local_stats()
                payload["tiers"] = {n: getattr(s, n) for n in dir(s)
                                    if not n.startswith("_")}
            except Exception as exc:
                payload["tiers_error"] = str(exc)
        return payload

    def _stats_loop(self):
        import json, pathlib
        try:
            torch.cuda.set_device(self._cuda_index)
        except Exception:
            pass
        node = os.environ.get("SLURM_PROCID", "0")
        path = pathlib.Path(self.stats_dir) / f"cascade_stats_rank{node}_tp{self.tp_rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            time.sleep(2.0)
            try:
                d = self.get_cascade_stats()
                d["node_rank"], d["tp_rank"] = int(node), self.tp_rank
                path.write_text(json.dumps(d, default=str, indent=2))
            except Exception:
                pass
