
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class LLaMAConfig:
    num_layers: int = 80
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "float16"

    @property
    def kv_size_per_token(self) -> int:

        element_size = 2
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * element_size

@dataclass
class BenchmarkConfig:

    base_path: Path = Path("${REPO_ROOT}")
    data_path: Path = Path("${REPO_ROOT}/benchmark/data")
    results_path: Path = Path("${REPO_ROOT}/benchmark/results")

    total_data_size_gb: float = 500.0
    block_size_tokens: int = 256
    num_unique_prefixes: int = 100
    prefix_length_tokens: int = 2048

    num_sessions: int = 1000
    read_write_ratio: float = 0.8

    num_gpus_per_node: int = 4
    gpu_memory_gb: float = 40.0
    dram_per_node_gb: float = 256.0
    shm_capacity_gb: float = 128.0

    systems: tuple = ("cascade", "hdf5", "redis", "lmcache", "pdc")

    def __post_init__(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

LLAMA_CONFIG = LLaMAConfig()
BENCHMARK_CONFIG = BenchmarkConfig()
