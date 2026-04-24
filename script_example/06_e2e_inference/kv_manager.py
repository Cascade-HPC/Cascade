import hashlib
import struct
import time
import numpy as np
import torch
from typing import Optional, Tuple

from .config import ModelConfig

HEADER_FMT = "<IIIIH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

DTYPE_MAP = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}

def extract_kv_pairs(past_key_values) -> list:

    if hasattr(past_key_values, 'key_cache'):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))

    result = []
    for layer in past_key_values:
        if isinstance(layer, (tuple, list)):
            result.append((layer[0], layer[1]))
        else:
            raise ValueError(f"Unexpected layer type: {type(layer)}")
    return result

def compute_prefix_hash(token_ids: torch.Tensor, num_tokens: int) -> str:

    prefix = token_ids[0, :num_tokens].cpu().numpy().tobytes()
    return hashlib.sha256(prefix).hexdigest()[:32]

def serialize_kv_cache(past_key_values) -> bytes:

    kv_pairs = extract_kv_pairs(past_key_values)

    first_k = kv_pairs[0][0]
    num_layers = len(kv_pairs)
    _, num_kv_heads, seq_len, head_dim = first_k.shape

    dtype_id = DTYPE_MAP[torch.float16]

    header = struct.pack(HEADER_FMT, num_layers, num_kv_heads, seq_len, head_dim, dtype_id)

    chunks = [header]
    for k, v in kv_pairs:
        k_bytes = k.squeeze(0).contiguous().to(torch.float16).cpu().numpy().view(np.uint8).tobytes()
        v_bytes = v.squeeze(0).contiguous().to(torch.float16).cpu().numpy().view(np.uint8).tobytes()
        chunks.append(k_bytes)
        chunks.append(v_bytes)

    return b''.join(chunks)

def deserialize_kv_cache(
    data: bytes,
    device: torch.device = torch.device("cuda:0"),
) -> tuple:

    header = data[:HEADER_SIZE]
    num_layers, num_kv_heads, seq_len, head_dim, dtype_id = struct.unpack(HEADER_FMT, header)
    dtype = DTYPE_REVERSE[dtype_id]

    payload = data[HEADER_SIZE:]
    np_dtype = np.float16 if dtype == torch.float16 else np.float32
    bytes_per_element = 2 if np_dtype == np.float16 else 4
    elements_per_tensor = num_kv_heads * seq_len * head_dim
    tensor_bytes = elements_per_tensor * bytes_per_element
    past_key_values = []

    offset = 0
    for _ in range(num_layers):
        k_np = np.frombuffer(payload[offset:offset + tensor_bytes], dtype=np_dtype)
        k = torch.from_numpy(k_np.copy()).reshape(1, num_kv_heads, seq_len, head_dim).to(dtype=dtype, device=device)
        offset += tensor_bytes

        v_np = np.frombuffer(payload[offset:offset + tensor_bytes], dtype=np_dtype)
        v = torch.from_numpy(v_np.copy()).reshape(1, num_kv_heads, seq_len, head_dim).to(dtype=dtype, device=device)
        offset += tensor_bytes

        past_key_values.append((k, v))

    return tuple(past_key_values)

def kv_cache_size_bytes(config: ModelConfig) -> int:
    return config.kv_block_bytes

class KVCacheStore:

    def __init__(self, adapter, config: ModelConfig):
        self.adapter = adapter
        self.config = config
        self.stats = {
            "hits": 0,
            "misses": 0,
            "put_time_ms": [],
            "get_time_ms": [],
            "serialize_time_ms": [],
            "deserialize_time_ms": [],
        }

    def store(self, prefix_hash: str, past_key_values: tuple) -> bool:
        t0 = time.perf_counter()
        kv_bytes = serialize_kv_cache(past_key_values)
        ser_ms = (time.perf_counter() - t0) * 1000
        self.stats["serialize_time_ms"].append(ser_ms)

        mid = len(kv_bytes) // 2
        t1 = time.perf_counter()
        ok = self.adapter.put(prefix_hash, kv_bytes[:mid], kv_bytes[mid:])
        put_ms = (time.perf_counter() - t1) * 1000
        self.stats["put_time_ms"].append(put_ms)

        return ok

    def load(
        self,
        prefix_hash: str,
        device: torch.device = torch.device("cuda:0"),
    ) -> Optional[tuple]:
        t0 = time.perf_counter()
        result = self.adapter.get(prefix_hash)
        get_ms = (time.perf_counter() - t0) * 1000
        self.stats["get_time_ms"].append(get_ms)

        if result is None:
            self.stats["misses"] += 1
            return None

        self.stats["hits"] += 1
        key_data, value_data = result

        t1 = time.perf_counter()

        kv_bytes = bytes(key_data) + bytes(value_data)
        past_kv = deserialize_kv_cache(kv_bytes, device=device)
        deser_ms = (time.perf_counter() - t1) * 1000
        self.stats["deserialize_time_ms"].append(deser_ms)

        return past_kv

    def contains(self, prefix_hash: str) -> bool:
        return self.adapter.contains(prefix_hash)
