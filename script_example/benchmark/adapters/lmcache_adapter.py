
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from .base import StorageAdapter
except ImportError:
    from benchmark.adapters.base import StorageAdapter

class LMCacheAdapter(StorageAdapter):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("LMCache", config)
        self.storage_path = config.get("storage_path", "${REPO_ROOT}/benchmark/lmcache_store")
        self.max_size_gb = config.get("max_size_gb", 100.0)
        self.base_dir = Path(self.storage_path)

    def initialize(self) -> bool:
        try:
            rank = int(os.environ.get('SLURM_PROCID', 0))
            if rank == 0:
                self.base_dir.mkdir(parents=True, exist_ok=True)
            time.sleep(1)
            self._initialized = True
            return True
        except Exception as e:
            if 'File exists' in str(e):
                self._initialized = True
                return True
            print(f"[LMCacheAdapter] Init error: {e}")
            return False

    def _get_path(self, block_id: str) -> Path:
        return self.base_dir / f"{block_id}.kv"

    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized: return False
        try:
            p = self._get_path(block_id)
            tmp_p = p.with_suffix('.tmp')
            with open(tmp_p, 'wb') as f:
                f.write(key_data)
                f.write(value_data)
            os.rename(tmp_p, p)
            return True
        except Exception as e:
            return False

    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized: return None
        try:
            p = self._get_path(block_id)
            if not p.exists(): return None
            with open(p, 'rb') as f:
                data = f.read()
            mid = len(data) // 2
            return (data[:mid], data[mid:])
        except:
            return None

    def contains(self, block_id: str) -> bool:
        return self._get_path(block_id).exists()

    def delete(self, block_id: str) -> bool:
        try:
            p = self._get_path(block_id)
            if p.exists():
                p.unlink()
                return True
            return False
        except:
            return False

    def clear(self) -> None:
        rank = int(os.environ.get('SLURM_PROCID', 0))
        if rank == 0:
            try:
                if self.base_dir.exists():
                    shutil.rmtree(self.base_dir)
                self.base_dir.mkdir(parents=True, exist_ok=True)
            except:
                pass

    def flush(self) -> None:

        pass

    def get_stats(self) -> Dict[str, Any]:
        try:
            size_bytes = sum(f.stat().st_size for f in self.base_dir.glob('**/*') if f.is_file())
            num_blocks = len(list(self.base_dir.glob('*.kv')))
            return {
                "mode": "simulated_disk",
                "num_blocks": num_blocks,
                "size_bytes": size_bytes
            }
        except:
            return {}

    def is_available(self) -> bool:
        return self._initialized
