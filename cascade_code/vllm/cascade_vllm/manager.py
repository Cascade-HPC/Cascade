from collections.abc import Iterable
import os
import time

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)

from .store import (
    CascadeIndexClient,
    CascadeLoadStoreSpec,
    make_block_key,
)


class CascadeOffloadingManager(OffloadingManager):

    def __init__(self, config):
        self._model = config.model_config.model
        self._tp_size = config.parallel_config.tensor_parallel_size
        self._tp_rank = config.parallel_config.rank
        self._dtype = config.kv_transfer_config.kv_connector_extra_config.get(
            "kv_dtype", str(getattr(config.model_config, "dtype", "native"))
        )
        self._layout = config.kv_transfer_config.kv_connector_extra_config.get(
            "layout", "vllm-canonical-v1"
        )
        self._block_size = config.cache_config.block_size
        self.block_size = self._block_size
        self.medium = CascadeLoadStoreSpec.medium()
        self.events = []
        self._index = CascadeIndexClient()
        self._known_hashes: set[str] = set()
        self._debug = os.environ.get("CASCADE_LOOKUP_DEBUG", "0") == "1" or (
            os.environ.get("CASCADE_TRANSFER_DEBUG", "0") == "1"
        )
        self._debug_lookup_count = 0
        self._warned_unavailable = False
        self._prefetch_enabled = os.environ.get("CASCADE_PREFETCH", "1") == "1"
        self._prefetch_sent: set[str] = set()
        self.prefetch_requests = 0
        self.prefetch_blocks = 0
        self._store_disabled = os.environ.get(
            "CASCADE_DISABLE_STORE", "0"
        ) == "1"
        self.lookup_calls = 0
        self.lookup_blocks = 0
        self.lookup_hits = 0
        self.lookup_seconds = 0.0

        if getattr(config.parallel_config, "pipeline_parallel_size", 1) != 1 or getattr(
            config.parallel_config, "data_parallel_size", 1
        ) != 1:
            raise ValueError(
                "Cascade vLLM integration currently requires PP=1 and DP=1; "
                "tensor parallel workers are supported"
            )

    def _key(self, block_hash: BlockHash, tp_rank: int | None = None) -> str:
        return make_block_key(
            model=self._model,
            block_hash=block_hash,
            tp_rank=self._tp_rank if tp_rank is None else tp_rank,
            tp_size=self._tp_size,
            dtype=self._dtype,
            layout=self._layout,
            block_size=self._block_size,
        )

    def _prefetch(self, block_hashes) -> None:
        if not self._prefetch_enabled or not block_hashes:
            return
        keys = [
            self._key(block_hash, tp_rank)
            for tp_rank in range(self._tp_size)
            for block_hash in block_hashes
        ]
        pending = [key for key in keys if key not in self._prefetch_sent]
        if not pending:
            return
        self._prefetch_sent.update(pending)
        issued = self._index.prefetch(pending)
        if issued:
            self.prefetch_requests += 1
            self.prefetch_blocks += int(issued)

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        block_hashes = list(block_hashes)
        if not block_hashes:
            return 0
        started = time.perf_counter()
        keys = [self._key(block_hash) for block_hash in block_hashes]

        hits = 0
        while hits < len(keys) and keys[hits] in self._known_hashes:
            hits += 1
        if hits < len(keys):
            remote_hits = self._index.available_prefix(keys[hits:])
            if remote_hits is None:
                if not self._warned_unavailable:
                    self._warned_unavailable = True
                    print(
                        "CASCADE WARNING scheduler cannot reach the Cascade "
                        "index socket; external KV hits are disabled for now",
                        flush=True,
                    )
            elif remote_hits:
                self._known_hashes.update(keys[hits : hits + remote_hits])
                self._prefetch(block_hashes[hits : hits + remote_hits])
                hits += remote_hits

        elapsed = time.perf_counter() - started
        self.lookup_calls += 1
        self.lookup_blocks += len(keys)
        self.lookup_hits += hits
        self.lookup_seconds += elapsed
        if self._debug and self._debug_lookup_count < 200:
            print(
                f"{self.medium} manager lookup blocks={len(keys)} hits={hits} "
                f"seconds={elapsed:.6f} rpc_calls={self._index.rpc_calls} "
                f"rpc_seconds={self._index.rpc_seconds:.6f} "
                f"index_unavailable={self._index.unavailable}",
                flush=True,
            )
            self._debug_lookup_count += 1
        return hits

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        return CascadeLoadStoreSpec(list(block_hashes))

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        block_hashes = list(block_hashes)
        if self._store_disabled:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=CascadeLoadStoreSpec([]),
                block_hashes_evicted=[],
            )
        keys = [self._key(block_hash) for block_hash in block_hashes]
        unknown = [
            index for index, key in enumerate(keys) if key not in self._known_hashes
        ]
        available = {}
        if unknown:
            flags = self._index.available_batch([keys[index] for index in unknown])
            if flags is not None:
                for index, flag in zip(unknown, flags):
                    available[index] = flag
                    if flag:
                        self._known_hashes.add(keys[index])
        new_hashes = [
            block_hash
            for index, block_hash in enumerate(block_hashes)
            if keys[index] not in self._known_hashes and not available.get(index, False)
        ]
        return PrepareStoreOutput(
            block_hashes_to_store=new_hashes,
            store_spec=CascadeLoadStoreSpec(new_hashes),
            block_hashes_evicted=[],
        )

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        if success:
            block_hashes = list(block_hashes)
            self._known_hashes.update(
                self._key(block_hash) for block_hash in block_hashes
            )
            self.events.append(
                OffloadingEvent(
                    block_hashes=block_hashes,
                    block_size=self.block_size,
                    medium=self.medium,
                    removed=False,
                )
            )

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        return

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        return

    def take_events(self):
        events = self.events
        self.events = []
        yield from events
