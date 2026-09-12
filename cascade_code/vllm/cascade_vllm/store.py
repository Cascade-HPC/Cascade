from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import pickle
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.v1.kv_offload.abstract import LoadStoreSpec


def _field(value) -> str:
    if isinstance(value, bytes):
        value = value.hex()
    return str(value)


class CascadeKey(NamedTuple):

    model: str
    block_hash: str
    tp_rank: int
    tp_size: int
    dtype: str
    layout: str
    block_size: int
    version: int = 1

    def encode(self) -> str:
        fields = (
            ("v", self.version),
            ("model", self.model),
            ("block", self.block_hash),
            ("tp_rank", self.tp_rank),
            ("tp_size", self.tp_size),
            ("dtype", self.dtype),
            ("layout", self.layout),
            ("block_size", self.block_size),
        )
        canonical = "|".join(
            f"{name}={_field(value)}" for name, value in fields
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        return f"cascade-kv-v{self.version}-{digest}"


def make_block_key(
    *,
    model: str,
    block_hash: str,
    tp_rank: int,
    tp_size: int,
    dtype: str,
    layout: str,
    block_size: int,
) -> str:

    return CascadeKey(
        model=model,
        block_hash=block_hash,
        tp_rank=tp_rank,
        tp_size=tp_size,
        dtype=dtype,
        layout=layout,
        block_size=block_size,
    ).encode()


class CascadeLoadStoreSpec(LoadStoreSpec):

    def __init__(self, block_hashes):
        self.block_hashes = list(block_hashes)

    @staticmethod
    def medium() -> str:
        return "CASCADE"

    def __repr__(self) -> str:
        return f"CascadeLoadStoreSpec({len(self.block_hashes)} blocks)"


_STATS_FIELDS = (
    "local_gpu_used",
    "local_dram_used",
    "cluster_gpu_used",
    "cluster_dram_used",
    "local_gpu_hits",
    "local_dram_hits",
    "remote_gpu_hits",
    "remote_dram_hits",
    "lustre_hits",
    "misses",
    "dedup_hits",
    "dedup_bytes_saved",
    "gpu_evictions",
    "dram_evictions",
    "prefix_blocks_protected",
    "promotions_to_local",
    "compression_savings",
    "total_blocks",
    "prefix_blocks",
    "prefetched_blocks",
    "prefetch_hits",
)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Cascade RPC peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock: socket.socket) -> Any:
    size = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    if size > 2 * 1024**3:
        raise ValueError(f"Cascade RPC frame too large: {size}")
    return pickle.loads(_recv_exact(sock, size))


def _send_message(sock: socket.socket, message: Any) -> None:
    payload = pickle.dumps(message, protocol=5)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


class _CascadeNodeServer:

    def __init__(
        self, native, socket_path: Path, tp_size: int, marker_root: Path | None
    ):
        self.native = native
        self.socket_path = socket_path
        self.tp_size = tp_size
        self.marker_root = marker_root
        self._native_lock = threading.RLock()
        self._collective_lock = threading.Condition()
        self._collectives = {
            name: {"generation": 0, "count": 0, "results": {}}
            for name in ("put_complete", "sync_metadata", "flush", "stats")
        }
        self._ipc_exports = {}
        self._ipc_export_id = 0
        self._metadata_sync_lock = threading.Lock()
        self._metadata_synced = False
        self._metadata_syncing = False
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=int(os.environ.get("CASCADE_PREFETCH_WORKERS", "2")),
            thread_name_prefix="cascade-prefetch",
        )
        self._prefetch_inflight: set[str] = set()
        self._prefetch_lock = threading.Lock()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(socket_path))
        self._listener.listen(max(128, tp_size * 32))
        self._thread = threading.Thread(
            target=self._accept_loop, name="cascade-node-rpc", daemon=True
        )
        self._thread.start()
        threading.Thread(
            target=self._metadata_sync_loop, name="cascade-meta-sync", daemon=True
        ).start()

    def _accept_loop(self):
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve, args=(conn,), daemon=True,
                name="cascade-rpc-conn",
            ).start()

    def _serve(self, conn: socket.socket):
        with conn:
            while True:
                try:
                    request = _recv_message(conn)
                except (ConnectionError, OSError, EOFError):
                    return
                try:
                    value = self._dispatch(request)
                    response = {"ok": True, "value": value}
                except Exception as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                try:
                    _send_message(conn, response)
                except (ConnectionError, OSError):
                    return

    def _native_call(self, name: str, *args):
        with self._native_lock:
            return getattr(self.native, name)(*args)

    def _native_write_lock(self):
        if os.environ.get("CASCADE_WRITE_PARALLEL", "1") == "1":
            return nullcontext()
        return self._native_lock

    def _native_read_lock(self):
        if os.environ.get("CASCADE_READ_PARALLEL", "0") == "1":
            return nullcontext()
        return self._native_lock

    def _publish_lustre_marker(self, keys):
        if (
            os.environ.get("CASCADE_LUSTRE_MARKERS", "0") == "1"
            and self.marker_root is not None
        ):
            for key in keys:
                marker = self.marker_root / key[:2] / key[2:4] / f"{key}.kv"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch(exist_ok=True)

    def _collective(self, name: str):
        with self._collective_lock:
            state = self._collectives[name]
            generation = state["generation"]
            state["count"] += 1
            if state["count"] == self.tp_size:
                try:
                    with self._native_lock:
                        if name == "put_complete":
                            self.native._store.sync_metadata()
                            self.native._store.flush()
                            native_value = None
                        else:
                            native_value = getattr(self.native, name)()
                    if name == "stats":
                        value = {
                            field: getattr(native_value, field)
                            for field in _STATS_FIELDS
                        }
                    else:
                        value = native_value
                    state["results"][generation] = (True, value)
                except Exception as exc:
                    state["results"][generation] = (
                        False,
                        f"{type(exc).__name__}: {exc}",
                    )
                state["count"] = 0
                state["generation"] += 1
                self._collective_lock.notify_all()
            else:
                while state["generation"] == generation:
                    self._collective_lock.wait()
            ok, value = state["results"][generation]
            if not ok:
                raise RuntimeError(value)
            return value

    def _run_prefetch(self, keys):
        try:
            with self._native_read_lock():
                self.native._store.prefetch_batch(keys)
        except Exception:
            pass
        finally:
            with self._prefetch_lock:
                self._prefetch_inflight.difference_update(keys)

    def _metadata_sync_loop(self):
        interval = float(os.environ.get("CASCADE_META_SYNC_POLL_S", "0.05"))
        while not self._metadata_synced:
            self._ensure_metadata_sync()
            if self._metadata_synced:
                return
            time.sleep(interval)

    def _ensure_metadata_sync(self, block: bool = True):
        if (
            os.environ.get("CASCADE_SYNC_METADATA", "0") != "1"
            or self.native.world_size <= 1
        ):
            return
        if self._metadata_synced:
            return
        barrier_dir = os.environ.get("CASCADE_PHASE_BARRIER_DIR")
        if barrier_dir:
            expected = [
                Path(barrier_dir) / f"pass1_rank_{peer}"
                for peer in range(int(self.native.world_size))
            ]
            if not all(path.exists() for path in expected):
                return
        if not self._metadata_sync_lock.acquire(blocking=block):
            return
        try:
            if not self._metadata_synced:
                with self._native_lock:
                    self.native._store.sync_metadata()
                    self.native._store.flush()
                    warm = getattr(self.native._store, "warm_remote", None)
                    if warm is not None:
                        warm()
                self._metadata_synced = True
        finally:
            self._metadata_sync_lock.release()

    def _dispatch(self, request):
        op = request["op"]
        if op == "hello":
            return {"rank": self.native.rank, "world_size": self.native.world_size}
        if op == "put_batch":
            payloads = [
                np.frombuffer(payload, dtype=np.uint8)
                for payload in request["payloads"]
            ]
            with self._native_lock:
                stored = int(
                    self.native._store.put_batch(
                        request["keys"], payloads, request["is_prefix"]
                    )
                )
            self._publish_lustre_marker(request["keys"])
            return stored
        if op == "put_ipc_batch":
            descriptors = request["descriptors"]
            with self._native_write_lock():
                stored = int(self.native._store.put_ipc_batch(
                    request["keys"],
                    [item["handle"] for item in descriptors],
                    [int(item["offset"]) for item in descriptors],
                    [int(item["size"]) for item in descriptors],
                    request["is_prefix"],
                ))
            self._publish_lustre_marker(request["keys"])
            return stored
        if op == "get_batch":
            outputs = [
                np.empty(int(request["max_size"]), dtype=np.uint8)
                for _ in request["keys"]
            ]
            with self._native_lock:
                _, sizes = self.native._store.get_batch(
                    request["keys"], outputs
                )
            return [
                bytes(output[: int(size)]) if int(size) else None
                for output, size in zip(outputs, sizes)
            ]
        if op == "get_ipc_batch":
            self._ensure_metadata_sync()
            buffers = [
                torch.empty(int(capacity), dtype=torch.uint8,
                            device=f"cuda:{torch.cuda.current_device()}")
                for capacity in request["capacities"]
            ]
            with self._native_read_lock():
                success, sizes = self.native._store.get_device_batch(
                    request["keys"], buffers
                )
            export_id = str(self._ipc_export_id)
            self._ipc_export_id += 1
            self._ipc_exports[export_id] = buffers
            descriptors = CascadeStore._ipc_descriptors(buffers)
            return {
                "success": int(success),
                "sizes": [int(size) for size in sizes],
                "export_id": export_id,
                "descriptors": descriptors,
            }
        if op == "get_ipc_device_batch":
            descriptors = request["descriptors"]
            self._ensure_metadata_sync()
            t0 = time.perf_counter()
            handles = [item["handle"] for item in descriptors]
            offsets = [int(item["offset"]) for item in descriptors]
            capacities = [int(item["size"]) for item in descriptors]
            t1 = time.perf_counter()
            with self._native_read_lock():
                success, sizes = self.native._store.get_ipc_device_batch(
                    request["keys"], handles, offsets, capacities
                )
                try:
                    self.native._store.unpin_promised(request["keys"])
                except AttributeError:
                    pass
            t2 = time.perf_counter()
            return {"success": int(success), "sizes": [int(size) for size in sizes],
                    "resolve_ms": (t1 - t0) * 1000.0,
                    "server_ms": (t2 - t1) * 1000.0}
        if op == "release_ipc":
            self._ipc_exports.pop(str(request["export_id"]), None)
            return True
        if op == "contains":
            return bool(self._native_call("contains", request["key"]))
        if op == "prefetch":
            self._ensure_metadata_sync()
            keys = request["keys"]
            with self._prefetch_lock:
                fresh = [key for key in keys if key not in self._prefetch_inflight]
                self._prefetch_inflight.update(fresh)
            if fresh:
                self._prefetch_pool.submit(self._run_prefetch, fresh)
            return len(fresh)
        if op == "available_prefix":
            self._ensure_metadata_sync()
            keys = request["keys"]
            with self._native_read_lock():
                hits = int(self.native._store.available_prefix(keys))
                if hits:
                    try:
                        self.native._store.pin_promised(
                            keys[:hits], float(os.environ.get(
                                "CASCADE_PIN_SECONDS", "120")))
                    except AttributeError:
                        pass
                return hits
        if op == "available_batch":
            self._ensure_metadata_sync()
            keys = request["keys"]
            with self._native_read_lock():
                return [bool(flag) for flag in
                        self.native._store.available_batch(keys)]
        if op == "stats_local":
            with self._native_read_lock():
                native_value = self.native._store.get_stats_local()
            return {field: getattr(native_value, field) for field in _STATS_FIELDS}
        if op in ("sync_metadata", "flush", "stats"):
            return self._collective(op)
        if op == "barrier":
            return self._collective("flush")
        raise ValueError(f"Unknown Cascade RPC operation: {op}")


class _CascadeNodeClient:

    def __init__(self, socket_path: Path, timeout_s: float = 300.0):
        self.socket_path = socket_path
        self._local = threading.local()
        deadline = time.monotonic() + timeout_s
        last_error = None
        while time.monotonic() < deadline:
            try:
                sock = self._connect()
                _send_message(sock, {"op": "hello"})
                response = _recv_message(sock)
                if not response["ok"]:
                    raise RuntimeError(response["error"])
                self.rank = response["value"]["rank"]
                self.world_size = response["value"]["world_size"]
                return
            except (OSError, ConnectionError, RuntimeError) as exc:
                last_error = exc
                self._drop()
                time.sleep(0.1)
        raise RuntimeError(
            f"Timed out waiting for Cascade TP0 RPC server at {socket_path}: "
            f"{last_error}"
        )

    def _connect(self) -> socket.socket:
        sock = getattr(self._local, "sock", None)
        if sock is not None:
            return sock
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(300.0)
        sock.connect(str(self.socket_path))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._local.sock = sock
        return sock

    def _drop(self) -> None:
        sock = getattr(self._local, "sock", None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
            self._local.sock = None

    def call(self, request):
        deadline = time.monotonic() + 300.0
        last_error = None
        while time.monotonic() < deadline:
            try:
                sock = self._connect()
                _send_message(sock, request)
                response = _recv_message(sock)
                if not response["ok"]:
                    raise RuntimeError(response["error"])
                return response["value"]
            except RuntimeError:
                raise
            except (BlockingIOError, ConnectionRefusedError,
                    ConnectionResetError, TimeoutError, InterruptedError,
                    ConnectionError, OSError) as exc:
                last_error = exc
                self._drop()
                time.sleep(0.01)
        raise RuntimeError(
            f"Timed out waiting for Cascade TP0 RPC server at {self.socket_path}: "
            f"{last_error}"
        )


class CascadeIndexClient:

    def __init__(self, connect_timeout_s: float = 2.0):
        self._connect_timeout_s = connect_timeout_s
        self._client: _CascadeNodeClient | None = None
        self._next_attempt = 0.0
        self.rpc_calls = 0
        self.rpc_seconds = 0.0
        self.unavailable = 0

    @staticmethod
    def _candidate_sockets() -> list[Path]:
        preferred = CascadeStore._rpc_socket_path()
        candidates = [preferred]
        namespace = os.environ.get("CASCADE_RPC_NAMESPACE") or os.environ.get(
            "SLURM_JOB_ID"
        )
        if namespace:
            for path in sorted(Path("/tmp").glob(f"cascade-vllm-{namespace}-*.sock")):
                if path != preferred:
                    candidates.append(path)
        return candidates

    def _connect(self) -> "_CascadeNodeClient | None":
        if self._client is not None:
            return self._client
        now = time.monotonic()
        if now < self._next_attempt:
            return None
        for candidate in self._candidate_sockets():
            try:
                self._client = _CascadeNodeClient(
                    candidate, timeout_s=self._connect_timeout_s
                )
                return self._client
            except RuntimeError:
                continue
        self._next_attempt = now + 1.0
        self.unavailable += 1
        return None

    def _call(self, request):
        client = self._connect()
        if client is None:
            return None
        started = time.perf_counter()
        try:
            value = client.call(request)
        except Exception:
            self._client = None
            self._next_attempt = time.monotonic() + 1.0
            self.unavailable += 1
            return None
        self.rpc_calls += 1
        self.rpc_seconds += time.perf_counter() - started
        return value

    def available_prefix(self, keys: list[str]) -> int | None:
        if not keys:
            return 0
        return self._call({"op": "available_prefix", "keys": list(keys)})

    def prefetch(self, keys: list[str]) -> int | None:
        if not keys:
            return 0
        return self._call({"op": "prefetch", "keys": list(keys)})

    def available_batch(self, keys: list[str]) -> list[bool] | None:
        if not keys:
            return []
        value = self._call({"op": "available_batch", "keys": list(keys)})
        if value is None:
            return None
        return [bool(flag) for flag in value]


def _load_cascade_cpp(extra: dict[str, Any]):
    module_stem = (
        extra.get("cascade_module")
        or os.environ.get("CASCADE_MODULE")
        or "cascade_cpp"
    )
    if module_stem not in ("cascade_cpp", "cascade_cc"):
        raise ValueError(f"unknown Cascade native module {module_stem!r}")
    module_path = (
        extra.get("cascade_cpp_path")
        or os.environ.get("CASCADE_CPP_PATH")
        or os.environ.get("CASCADE_NATIVE_PATH")
    )
    candidates = []
    if module_path:
        candidates.append(Path(module_path).expanduser())
    cpp_root = Path(__file__).resolve().parents[2] / "cpp"
    candidates.extend(
        [
            cpp_root / "build_vllm_native",
            cpp_root / "build_cascade_cpp",
        ]
    )
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    errors = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        module_files = [
            path
            for path in candidate.glob(f"{module_stem}*.so")
            if path.name.endswith(suffixes)
        ]
        for module_file in sorted(module_files):
            try:
                spec = importlib.util.spec_from_file_location(
                    module_stem, module_file
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"no loader for {module_file}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_stem] = module
                spec.loader.exec_module(module)
                return module
            except ImportError as exc:
                errors.append(f"{module_file}: {exc}")
                sys.modules.pop("cascade_cpp", None)
    detail = "; ".join(errors)
    raise RuntimeError(
        "Cascade native module is required. Build the C++ store with "
        "USE_MPI=ON and set CASCADE_CPP_PATH to the directory containing "
        f"a Python-compatible cascade_cpp*.so. {detail}"
    )


class CascadeStore:

    def __init__(self, extra: dict[str, Any]):
        self._cpp = None
        self._store = None
        self._server = None
        self._client = None
        self._tp_size = int(extra.get("tp_size", 1))
        self._tp_rank = int(extra.get("tp_rank", 0))
        self._gpu_free_before: list[int] = []

        use_tp_fan_in = self._tp_size > 1 and os.environ.get(
            "CASCADE_TP_RPC", "1"
        ) == "1"
        if use_tp_fan_in and self._tp_rank != 0:
            socket_path = self._rpc_socket_path()
            self._client = _CascadeNodeClient(socket_path)
            self._cpp = _load_cascade_cpp(extra)
            self.rank = self._client.rank
            self.world_size = self._client.world_size
            return

        self._cpp = _load_cascade_cpp(extra)
        config = self._cpp.DistributedConfig()

        def set_if_present(name: str, value: Any) -> None:
            if hasattr(config, name) and value is not None:
                setattr(config, name, value)

        requested_gpu_gb = float(extra.get("gpu_capacity_gb", 0.0))
        num_gpus = int(extra.get("num_gpus_per_node", 1))
        gpu_capacity_gb, gpu_headroom = self._fit_gpu_capacity(
            requested_gpu_gb, num_gpus
        )
        set_if_present(
            "gpu_capacity_per_device", int(gpu_capacity_gb * 1024**3)
        )
        set_if_present(
            "dram_capacity",
            int(float(extra.get("dram_capacity_gb", 64.0)) * 1024**3),
        )
        set_if_present("num_gpus_per_node", num_gpus)
        gpu_device_offset = extra.get("gpu_device_offset", 0)
        if gpu_device_offset == "current":
            try:
                import torch

                gpu_device_offset = torch.cuda.current_device()
            except (ImportError, RuntimeError):
                gpu_device_offset = 0
        set_if_present("gpu_device_offset", int(gpu_device_offset))
        set_if_present("lustre_path", extra.get("lustre_path", ""))
        set_if_present("dedup_enabled", bool(extra.get("dedup_enabled", True)))
        set_if_present("semantic_eviction", bool(extra.get("semantic_eviction", True)))
        set_if_present("locality_aware", bool(extra.get("locality_aware", True)))
        set_if_present("prefix_replication", bool(extra.get("prefix_replication", True)))
        set_if_present("kv_compression", bool(extra.get("kv_compression", False)))
        set_if_present("aggregated_lustre", bool(extra.get("aggregated_lustre", True)))

        previous_cuda_device = None
        try:
            import torch

            if torch.cuda.is_available():
                previous_cuda_device = torch.cuda.current_device()
        except (ImportError, RuntimeError):
            pass

        self._store = self._cpp.DistributedStore(config)

        free_after = self._device_free_bytes(num_gpus)
        print(
            "CASCADE gpu_memory tp_rank=%d requested_gb=%.3f granted_gb=%.3f "
            "reserve_gb=%.3f free_before_gib=%s free_after_gib=%s"
            % (
                self._tp_rank,
                requested_gpu_gb,
                gpu_capacity_gb,
                gpu_headroom,
                [round(value / 1024**3, 2) for value in self._gpu_free_before],
                [round(value / 1024**3, 2) for value in free_after],
            ),
            flush=True,
        )

        if previous_cuda_device is not None:
            import torch

            torch.cuda.set_device(previous_cuda_device)
        self.rank = getattr(self._store, "rank", 0)
        self.world_size = getattr(self._store, "world_size", 1)

        self._server = _CascadeNodeServer(
            self,
            self._rpc_socket_path(),
            self._tp_size,
            Path(extra["lustre_path"]) if extra.get("lustre_path") else None,
        )

    @staticmethod
    def _device_free_bytes(num_gpus: int) -> list[int]:
        try:
            import torch

            if not torch.cuda.is_available():
                return []
            count = min(num_gpus, torch.cuda.device_count())
            return [torch.cuda.mem_get_info(device)[0] for device in range(count)]
        except (ImportError, RuntimeError):
            return []

    def _fit_gpu_capacity(self, requested_gb: float, num_gpus: int):
        reserve_gb = float(os.environ.get("CASCADE_GPU_RESERVE_GB", "1.6"))
        self._gpu_free_before = self._device_free_bytes(num_gpus)
        if requested_gb <= 0 or not self._gpu_free_before:
            return max(0.0, requested_gb), reserve_gb
        free_min_gb = min(self._gpu_free_before) / 1024**3
        allowed_gb = max(0.0, free_min_gb - reserve_gb)
        granted_gb = min(requested_gb, allowed_gb)
        if granted_gb < requested_gb:
            print(
                "CASCADE gpu_tier clamped requested_gb=%.3f free_min_gib=%.3f "
                "reserve_gb=%.3f granted_gb=%.3f"
                % (requested_gb, free_min_gb, reserve_gb, granted_gb),
                flush=True,
            )
        return granted_gb, reserve_gb

    @staticmethod
    def _rpc_socket_path() -> Path:
        namespace = os.environ.get("CASCADE_RPC_NAMESPACE") or os.environ.get(
            "SLURM_JOB_ID", str(os.getppid())
        )
        node_rank = os.environ.get("SLURM_PROCID", "0")
        return Path(f"/tmp/cascade-vllm-{namespace}-{node_rank}.sock")

    def _rpc_call(self, request):
        if self._client is None:
            raise RuntimeError("Cascade RPC client is not initialized")
        return self._client.call(request)

    @staticmethod
    def _array(data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return np.ascontiguousarray(data, dtype=np.uint8)
        return np.frombuffer(memoryview(data), dtype=np.uint8)

    def put(self, key: str, data: Any, *, is_prefix: bool = True) -> bool:
        if self._client is not None:
            return self.put_many([key], [data], is_prefix=is_prefix) == 1
        payload = self._array(data)
        if self._server is not None:
            with self._server._native_lock:
                return bool(self._store.put(key, payload, is_prefix))
        return bool(self._store.put(key, payload, is_prefix))

    def get(self, key: str, max_size: int):
        if self._client is not None:
            return self.get_many([key], max_size)[0]
        output = np.empty(int(max_size), dtype=np.uint8)
        if self._server is not None:
            with self._server._native_lock:
                found, size = self._store.get(key, output)
        else:
            found, size = self._store.get(key, output)
        if not found:
            return None
        return output[: int(size)].copy()

    def contains(self, key: str) -> bool:
        if self._client is not None:
            return bool(self._rpc_call({"op": "contains", "key": key}))
        if self._server is not None:
            with self._server._native_lock:
                return bool(self._store.contains(key))
        return bool(self._store.contains(key))

    def put_many(self, keys: list[str], payloads: list[Any], *, is_prefix: bool = True) -> int:
        if self._client is not None:
            return int(
                self._rpc_call(
                    {
                        "op": "put_batch",
                        "keys": keys,
                        "payloads": [self._array(payload).tobytes() for payload in payloads],
                        "is_prefix": [bool(is_prefix)] * len(keys),
                    }
                )
            )
        arrays = [self._array(payload) for payload in payloads]
        if hasattr(self._store, "put_batch"):
            flags = [bool(is_prefix)] * len(keys)
        if self._server is not None:
            with self._server._native_lock:
                stored = int(self._store.put_batch(keys, arrays, flags))
            self._server._publish_lustre_marker(keys)
            return stored
            return int(self._store.put_batch(keys, arrays, flags))
        return sum(self.put(key, payload, is_prefix=is_prefix) for key, payload in zip(keys, arrays))

    _ipc_handle_cache: dict[int, bytes] = {}

    @classmethod
    def _ipc_descriptors(cls, buffers: list[Any]):
        descriptors = []
        for buffer in buffers:
            if not isinstance(buffer, torch.Tensor) or not buffer.is_cuda:
                raise TypeError("Cascade device path requires CUDA tensors")
            storage = buffer.untyped_storage()
            base = storage.data_ptr()
            handle = cls._ipc_handle_cache.get(base)
            if handle is None:
                shared = storage._share_cuda_()
                handle = shared[1]
                cls._ipc_handle_cache[base] = handle
            descriptors.append({
                "handle": handle,
                "offset": int(buffer.data_ptr() - base),
                "size": int(buffer.numel() * buffer.element_size()),
            })
        return descriptors

    def put_device_many(self, keys: list[str], buffers: list[Any], *, is_prefix: bool = True) -> int:
        if len(keys) != len(buffers):
            raise ValueError("Cascade key/device buffer count mismatch")
        flags = [bool(is_prefix)] * len(keys)
        if self._client is not None:
            descriptors = self._ipc_descriptors(buffers)
            return int(self._rpc_call({
                "op": "put_ipc_batch",
                "keys": keys,
                "descriptors": descriptors,
                "is_prefix": flags,
            }))
        if not hasattr(self._store, "put_device_batch"):
            raise RuntimeError("Cascade native module lacks put_device_batch")
        if self._server is not None:
            with self._server._native_write_lock():
                stored = int(self._store.put_device_batch(keys, buffers, flags))
            self._server._publish_lustre_marker(keys)
            return stored
        return int(self._store.put_device_batch(keys, buffers, flags))

    def get_many(
        self,
        keys: list[str],
        max_size: int,
        outputs: list[np.ndarray] | None = None,
    ) -> list[np.ndarray | None]:
        if self._client is not None:
            payloads = self._rpc_call(
                {"op": "get_batch", "keys": keys, "max_size": int(max_size)}
            )
            if outputs is None:
                return [
                    np.frombuffer(payload, dtype=np.uint8).copy()
                    if payload is not None
                    else None
                    for payload in payloads
                ]
            result = []
            for output, payload in zip(outputs, payloads):
                if payload is None:
                    result.append(None)
                else:
                    view = np.frombuffer(payload, dtype=np.uint8)
                    output[: len(view)] = view
                    result.append(output[: len(view)])
            return result
        owns_outputs = outputs is None
        outputs = outputs or [np.empty(int(max_size), dtype=np.uint8) for _ in keys]
        if hasattr(self._store, "get_batch"):
            if self._server is not None:
                with self._server._native_lock:
                    success, sizes = self._store.get_batch(keys, outputs)
            else:
                success, sizes = self._store.get_batch(keys, outputs)
            return [
                (output[: int(size)].copy() if owns_outputs else output[: int(size)])
                if int(size) > 0
                else None
                for output, size in zip(outputs, sizes)
            ]
        return [self.get(key, max_size) for key in keys]

    def get_device_many(self, keys: list[str], buffers: list[Any]):
        if len(keys) != len(buffers):
            raise ValueError("Cascade key/device buffer count mismatch")
        if self._client is not None:
            debug = os.environ.get("CASCADE_TRANSFER_DEBUG", "0") == "1"
            t0 = time.perf_counter()
            descriptors = self._ipc_descriptors(buffers)
            t1 = time.perf_counter()
            result = self._rpc_call({
                "op": "get_ipc_device_batch",
                "keys": keys,
                "capacities": [int(item["size"]) for item in descriptors],
                "descriptors": descriptors,
            })
            t2 = time.perf_counter()
            if debug:
                print(
                    f"CASCADE rpc_load blocks={len(keys)} "
                    f"descriptors_ms={(t1 - t0) * 1000:.2f} "
                    f"roundtrip_ms={(t2 - t1) * 1000:.2f} "
                    f"server_ms={float(result.get('server_ms', 0.0)):.2f} "
                    f"resolve_ms={float(result.get('resolve_ms', 0.0)):.2f}",
                    flush=True,
                )
            return int(result["success"]), [int(size) for size in result["sizes"]]
        if not hasattr(self._store, "get_device_batch"):
            raise RuntimeError("Cascade native module lacks get_device_batch")
        if self._server is not None:
            self._server._ensure_metadata_sync()
            with self._server._native_read_lock():
                success, sizes = self._store.get_device_batch(keys, buffers)
        else:
            success, sizes = self._store.get_device_batch(keys, buffers)
        return int(success), [int(size) for size in sizes]

    def available_prefix(self, keys: list[str]) -> int:
        if not keys:
            return 0
        if self._client is not None:
            return int(
                self._rpc_call({"op": "available_prefix", "keys": list(keys)})
            )
        if self._server is not None:
            self._server._ensure_metadata_sync()
            with self._server._native_read_lock():
                return int(self._store.available_prefix(list(keys)))
        return int(self._store.available_prefix(list(keys)))

    def available_batch(self, keys: list[str]) -> list[bool]:
        if not keys:
            return []
        if self._client is not None:
            return [
                bool(flag)
                for flag in self._rpc_call(
                    {"op": "available_batch", "keys": list(keys)}
                )
            ]
        if self._server is not None:
            self._server._ensure_metadata_sync()
            with self._server._native_read_lock():
                return [bool(f) for f in self._store.available_batch(list(keys))]
        return [bool(f) for f in self._store.available_batch(list(keys))]

    def local_stats(self):
        from types import SimpleNamespace

        if self._client is not None:
            return SimpleNamespace(**self._rpc_call({"op": "stats_local"}))
        if self._server is not None:
            with self._server._native_read_lock():
                native_value = self._store.get_stats_local()
        else:
            native_value = self._store.get_stats_local()
        return SimpleNamespace(
            **{field: getattr(native_value, field) for field in _STATS_FIELDS}
        )

    def stats(self):
        if self._client is not None:
            from types import SimpleNamespace

            return SimpleNamespace(
                **self._rpc_call({"op": "stats"})
            )
        if self._server is not None:
            from types import SimpleNamespace

            return SimpleNamespace(**self._server._collective("stats"))
        return self._store.get_stats()

    def sync_metadata(self) -> None:
        if self._server is not None:
            self._server._collective("sync_metadata")
            return
        if self._client is not None:
            self._rpc_call({"op": "sync_metadata"})
            return
        self._store.sync_metadata()

    def flush(self) -> None:
        if self._server is not None:
            self._server._collective("flush")
            return
        if self._client is not None:
            self._rpc_call({"op": "flush"})
            return
        self._store.flush()

    def close(self) -> None:
        return None

    def __enter__(self) -> "CascadeStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()