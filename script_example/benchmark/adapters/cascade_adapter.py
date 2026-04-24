
import sys
from pathlib import Path
from typing import Optional, Dict, Any

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from .base import StorageAdapter
except ImportError:
    from base import StorageAdapter

class CascadeAdapter(StorageAdapter):

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("Cascade", config)
        self.store = None

        self.gpu_capacity_gb = config.get("gpu_capacity_gb", 32.0)
        self.shm_capacity_gb = config.get("shm_capacity_gb", 64.0)
        self.lustre_path = config.get("lustre_path", "${REPO_ROOT}/benchmark/cascade_store")
        self.use_gpu = config.get("use_gpu", False)
        self.use_compression = config.get("use_compression", True)
        self.use_sharding = config.get("use_sharding", True)

    def initialize(self) -> bool:
        try:

            from cascade import CascadeStore, CascadeConfig, create_login_node_store
            import os

            try:
                from mpi4py import MPI
                if MPI.Is_initialized():
                    world_size = MPI.COMM_WORLD.Get_size()
                else:

                    world_size = 1
            except ImportError:
                world_size = int(os.environ.get('SLURM_NTASKS', 1))

            if world_size > 1:

                import cascade_cpp

                cfg = cascade_cpp.DistributedConfig()
                cfg.gpu_capacity_per_device = int(self.gpu_capacity_gb * 1024**3)
                cfg.dram_capacity = int(self.shm_capacity_gb * 1024**3)

                num_gpus = 0
                try:
                    import torch
                    if torch.cuda.is_available():
                        num_gpus = torch.cuda.device_count()
                except ImportError:
                    pass

                if num_gpus == 0:
                    try:
                        import subprocess
                        output = subprocess.check_output("nvidia-smi -L | wc -l", shell=True).decode().strip()
                        num_gpus = int(output)
                    except:
                        num_gpus = 4

                cfg.num_gpus_per_node = num_gpus if num_gpus > 0 else 1
                cfg.dedup_enabled = True
                cfg.locality_aware = True
                cfg.kv_compression = True
                cfg.prefix_replication = True
                cfg.lustre_path = self.lustre_path
                self.store = cascade_cpp.DistributedStore(cfg)
                print(f"[CascadeAdapter] Initialized DistributedStore (world={world_size})")
            elif self.use_gpu:
                cfg = CascadeConfig()
                cfg.gpu_capacity_bytes = int(self.gpu_capacity_gb * 1024**3)
                cfg.shm_capacity_bytes = int(self.shm_capacity_gb * 1024**3)
                cfg.lustre_path = self.lustre_path
                cfg.dedup_enabled = True
                cfg.use_gpu = True
                self.store = CascadeStore(cfg)
                print(f"[CascadeAdapter] Initialized Local CascadeStore (GPU=True)")
            else:

                self.store = create_login_node_store()
                print(f"[CascadeAdapter] Initialized Login Node Store (GPU=False)")

            self._initialized = True
            return True

        except ImportError as e:
            print(f"[CascadeAdapter] Import error: {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"[CascadeAdapter] Init error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def sync_metadata(self):
        if self._initialized and hasattr(self.store, "sync_metadata"):
            self.store.sync_metadata()

    def barrier(self):
        if self._initialized and hasattr(self.store, "barrier"):
            self.store.barrier()

    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized:
            return False

        import numpy as np

        data = np.frombuffer(key_data + value_data, dtype=np.uint8)
        return self.store.put(block_id, data, is_prefix=False)

    def put_prefix(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized:
            return False

        import numpy as np
        data = np.frombuffer(key_data + value_data, dtype=np.uint8)
        return self.store.put(block_id, data, is_prefix=True)

    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized:
            return None

        import numpy as np

        if not hasattr(self, "_get_buf"):
            self._get_buf = np.empty(512 * 1024 * 1024, dtype=np.uint8)

        found, size = self.store.get(block_id, self._get_buf)

        if found and size > len(self._get_buf):

            self._get_buf = np.empty(int(size * 1.1), dtype=np.uint8)
            found, size = self.store.get(block_id, self._get_buf)

        if not found or size == 0:
            return None

        data = self._get_buf[:size]

        mid = len(data) // 2
        return (data[:mid], data[mid:])

    def contains(self, block_id: str) -> bool:
        if not self._initialized:
            return False
        return self.store.contains(block_id)

    def delete(self, block_id: str) -> bool:

        return False

    def clear(self) -> None:
        if self._initialized and self.store:
            self.store.clear()

    def flush(self) -> None:
        if self._initialized and self.store:
            self.store.flush()

    def get_stats(self) -> Dict[str, Any]:
        if not self._initialized:
            return {}
        stats = self.store.get_stats()

        gpu_used = getattr(stats, "cluster_gpu_used", getattr(stats, "gpu_used", getattr(stats, "local_gpu_used", 0)))
        shm_used = getattr(stats, "cluster_dram_used", getattr(stats, "shm_used", getattr(stats, "local_dram_used", 0)))

        gpu_hits = getattr(stats, "local_gpu_hits", getattr(stats, "gpu_hits", 0))
        shm_hits = getattr(stats, "local_dram_hits", getattr(stats, "shm_hits", 0))

        return {
            "gpu_used": gpu_used,
            "shm_used": shm_used,
            "gpu_hits": gpu_hits,
            "shm_hits": shm_hits,
            "lustre_hits": stats.lustre_hits,
            "misses": stats.misses,
            "dedup_hits": stats.dedup_hits
        }

    def close(self) -> None:
        if self.store:
            try:
                self.store.flush()
            except:
                pass
        self._initialized = False
