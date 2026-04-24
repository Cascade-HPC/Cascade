
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    from .base import StorageAdapter
except ImportError:
    from benchmark.adapters.base import StorageAdapter

class PDCAdapter(StorageAdapter):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("PDC", config)
        self.storage_path = config.get("storage_path", "${REPO_ROOT}/benchmark/pdc_store")

        self._reads = 0
        self._writes = 0

    def initialize(self) -> bool:
        try:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            self._initialized = True
            return True
        except Exception as e:
            print(f"[PDCAdapter] Init error: {e}")
            return False

    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized: return False

        try:
            path = Path(self.storage_path) / f"{block_id}.pdc"
            with open(path, 'wb') as f:
                f.write(key_data + value_data)
                f.flush()

                os.fsync(f.fileno())

            if hasattr(os, 'sync'):
                os.sync()

            self._writes += 1
            return True
        except Exception as e:
            print(f"[PDCAdapter] Put error: {e}")
            return False

    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized: return None

        try:
            path = Path(self.storage_path) / f"{block_id}.pdc"
            if not path.exists(): return None

            with open(path, 'rb') as f:
                data = f.read()

            self._reads += 1
            mid = len(data) // 2
            return (data[:mid], data[mid:])
        except Exception as e:
            return None

    def contains(self, block_id: str) -> bool:
        path = Path(self.storage_path) / f"{block_id}.pdc"
        return path.exists()

    def delete(self, block_id: str) -> bool:
        path = Path(self.storage_path) / f"{block_id}.pdc"
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:

        rank = int(os.environ.get('SLURM_PROCID', 0))
        if rank == 0:
            if os.path.exists(self.storage_path):
                for f in os.listdir(self.storage_path):
                    if f.endswith(".pdc"):
                        try:
                            os.remove(os.path.join(self.storage_path, f))
                        except: pass

        self._reads = 0
        self._writes = 0

    def flush(self) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "reads": self._reads,
            "writes": self._writes,
            "path": self.storage_path
        }

    def close(self) -> None:
        self._initialized = False
