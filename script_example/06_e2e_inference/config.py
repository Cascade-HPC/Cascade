from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-72B"
    dtype: str = "float16"
    tp_size: int = 4
    max_seq_len: int = 4096
    trust_remote_code: bool = True
    quantization: str = "none"

    block_token_size: int = 2048

    num_layers: int = 80
    num_kv_heads: int = 8
    head_dim: int = 128

    @property
    def kv_block_bytes(self) -> int:

        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.block_token_size * 2

@dataclass
class CascadeConfig:
    gpu_capacity_gb: float = 32.0
    shm_capacity_gb: float = 64.0
    use_gpu: bool = True
    use_compression: bool = True
    lustre_path: Optional[str] = None

@dataclass
class BenchmarkConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)

    dataset_path: str = "benchmark/data_external/sharegpt/sharegpt_cleaned.json"
    num_sessions: int = 100
    max_new_tokens: int = 64
    warmup_requests: int = 2

    results_dir: str = "inference_benchmark/results"

LLAMA_70B = ModelConfig(
    name="meta-llama/Llama-3-70B",
    num_layers=80,
    num_kv_heads=8,
    head_dim=128,
    block_token_size=2048,
)

QWEN_72B = ModelConfig(
    name="models/Qwen2.5-72B",
    num_layers=80,
    num_kv_heads=8,
    head_dim=128,
    block_token_size=2048,
)
