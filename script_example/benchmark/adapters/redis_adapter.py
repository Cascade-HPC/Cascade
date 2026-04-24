
import os
import time
import numpy as np
from typing import Optional, Dict, Any

try:
    from .base import StorageAdapter
except ImportError:
    from benchmark.adapters.base import StorageAdapter

class RedisAdapter(StorageAdapter):
    CHUNK_SIZE = 50 * 1024 * 1024

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Redis", config)
        self.host = config.get("host", os.environ.get("REDIS_HOST", "localhost"))
        self.port = int(config.get("port", os.environ.get("REDIS_PORT", 6379)))
        self.password = config.get("password", os.environ.get("REDIS_PASSWORD", None))
        self.client = None

        self._reads = 0
        self._writes = 0

    def initialize(self) -> bool:
        try:
            import redis

            for attempt in range(10):
                try:
                    self.client = redis.Redis(
                        host=self.host,
                        port=self.port,
                        password=self.password,
                        socket_timeout=60,
                        socket_connect_timeout=20,
                        retry_on_timeout=True,
                    )
                    self.client.ping()
                    self._initialized = True
                    return True
                except Exception as e:
                    if attempt == 9:
                        print(f"[RedisAdapter] Failed to connect to {self.host}:{self.port} after 10 attempts: {e}")
                    time.sleep(2 + attempt)
            return False
        except ImportError:
            print("[RedisAdapter] redis-py not installed")
            return False

    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if not self._initialized: return False

        try:
            raw = key_data + value_data
            total = len(raw)

            if total <= self.CHUNK_SIZE:

                self.client.set(block_id, raw)
                self.client.set(f"{block_id}:meta", str(total).encode())
            else:

                n_chunks = (total + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
                pipe = self.client.pipeline(transaction=False)
                for i in range(n_chunks):
                    start = i * self.CHUNK_SIZE
                    end = min(start + self.CHUNK_SIZE, total)
                    pipe.set(f"{block_id}:chunk:{i}", raw[start:end])
                pipe.set(f"{block_id}:meta", f"{total}:{n_chunks}".encode())
                pipe.execute()

            self._writes += 1
            return True
        except Exception as e:
            print(f"[RedisAdapter] Put error: {e}")
            return False

    def get(self, block_id: str) -> Optional[tuple]:
        if not self._initialized: return None

        try:
            meta = self.client.get(f"{block_id}:meta")
            if meta is None: return None

            meta_str = meta.decode()
            if ':' not in meta_str:

                val = self.client.get(block_id)
                if val:
                    data = val
                else:
                    return None
            else:

                total, n_chunks = meta_str.split(':')
                total, n_chunks = int(total), int(n_chunks)
                pipe = self.client.pipeline(transaction=False)
                for i in range(n_chunks):
                    pipe.get(f"{block_id}:chunk:{i}")
                chunks = pipe.execute()

                data = b''.join(chunks)

            self._reads += 1
            mid = len(data) // 2
            return (data[:mid], data[mid:])

        except Exception as e:
            print(f"[RedisAdapter] Get error: {e}")
            return None

    def contains(self, block_id: str) -> bool:
        if not self._initialized: return False
        return self.client.exists(f"{block_id}:meta") > 0

    def delete(self, block_id: str) -> bool:
        if not self._initialized: return False
        try:
            meta = self.client.get(f"{block_id}:meta")
            if meta:
                meta_str = meta.decode()
                if ':' in meta_str:
                    _, n_chunks = meta_str.split(':')
                    keys = [f"{block_id}:chunk:{i}" for i in range(int(n_chunks))]
                    keys.append(f"{block_id}:meta")
                    self.client.delete(*keys)
                else:
                    self.client.delete(block_id, f"{block_id}:meta")
                return True
            return False
        except:
            return False

    def clear(self) -> None:
        if self._initialized:
            self.client.flushdb()
            self._reads = 0
            self._writes = 0

    def flush(self) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "reads": self._reads,
            "writes": self._writes,
            "connected": self._initialized
        }

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None
