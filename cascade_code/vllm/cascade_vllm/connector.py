import os
from collections.abc import Iterator
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

try:
    from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
except ImportError:
    from vllm.v1.kv_offload.spec import OffloadingSpec

    CanonicalKVCaches = Any

from .handler import CascadeOffloadingHandler
from .manager import CascadeOffloadingManager
from .store import CascadeLoadStoreSpec, CascadeStore


class CascadeConnectorV1(OffloadingConnector):

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return os.environ.get("CASCADE_CROSS_LAYER", "1") == "1"


class CascadeOffloadingSpec(OffloadingSpec):

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        self._store = None
        self._manager = None
        self._handler = None

        if self.block_size_factor != 1:
            raise ValueError(
                "Cascade integration initially requires offloaded block size "
                "to equal the vLLM GPU block size"
            )

        parallel_config = vllm_config.parallel_config
        pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
        dp_size = getattr(parallel_config, "data_parallel_size", 1)
        if pp_size != 1 or dp_size != 1:
            raise ValueError(
                "Cascade integration currently supports PP=1 and DP=1; "
                "tensor parallel workers are supported"
            )

    def _get_store(self):
        if self._store is None:
            extra = dict(self.extra_config)
            extra["tp_rank"] = int(self.vllm_config.parallel_config.rank)
            extra["tp_size"] = int(
                self.vllm_config.parallel_config.tensor_parallel_size
            )
            self._store = CascadeStore(extra)
        return self._store

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = CascadeOffloadingManager(self.vllm_config)
        return self._manager

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches | dict[str, Any],
        attn_backends: dict[str, Any] | None = None,
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if self._handler is None:
            self._handler = CascadeOffloadingHandler(
                self.vllm_config,
                kv_caches,
                self._get_store(),
                attn_backends=attn_backends,
            )
        yield GPULoadStoreSpec, CascadeLoadStoreSpec, self._handler
        yield CascadeLoadStoreSpec, GPULoadStoreSpec, self._handler


OffloadingSpecFactory.register_spec(
    "CascadeOffloadingSpec",
    "cascade_vllm.connector",
    "CascadeOffloadingSpec",
)
