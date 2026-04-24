
import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

from .config import BENCHMARK_CONFIG, LLAMA_CONFIG
from .adapters import (
    StorageAdapter, BenchmarkStats, CascadeAdapter,
    HDF5IndependentIOAdapter, HDF5CollectiveIOAdapter,
    LMCacheAdapter, RedisAdapter, PDCAdapter
)

def get_adapter(name: str, config: Dict = None) -> StorageAdapter:
    adapters = {
        "cascade": CascadeAdapter,
        "hdf5": HDF5IndependentIOAdapter,
        "hdf5-independent": HDF5IndependentIOAdapter,
        "hdf5-indep": HDF5IndependentIOAdapter,
        "hdf5-collective": HDF5CollectiveIOAdapter,
        "hdf5-coll": HDF5CollectiveIOAdapter,

        "lmcache": LMCacheAdapter,
        "lmcache-redis": RedisAdapter,
        "redis": RedisAdapter,
        "pdc": PDCAdapter,
    }

    name_lower = name.lower()
    if name_lower not in adapters:
        raise ValueError(f"Unknown system: {name}. Available: {list(adapters.keys())}")

    return adapters[name_lower](config or {})

def load_test_data(data_path: Path, num_blocks: int = 1000) -> List[Dict]:
    metadata_path = data_path / "metadata.json"

    if not metadata_path.exists():
        print(f"Warning: No pre-generated data at {data_path}")
        print("Generating synthetic data...")
        return generate_synthetic_data(num_blocks)

    with open(metadata_path) as f:
        metadata = json.load(f)

    blocks = []
    block_meta = metadata["blocks"][:num_blocks]

    for meta in block_meta:
        block_id = meta["block_id"]
        is_prefix = meta["is_prefix"]

        if is_prefix:
            prefix_id = meta["prefix_id"]
            block_idx = meta["block_index"] % (BENCHMARK_CONFIG.prefix_length_tokens // BENCHMARK_CONFIG.block_size_tokens)
            path = data_path / "prefixes" / f"prefix_{prefix_id:03d}_block_{block_idx:03d}.npz"
        else:
            unique_idx = meta["block_index"] - len([b for b in block_meta if b["is_prefix"]])
            shard = unique_idx // 1000
            path = data_path / "unique" / f"shard_{shard:05d}" / f"block_{unique_idx:08d}.npz"

        if path.exists():
            data = np.load(path)
            blocks.append({
                "block_id": block_id,
                "key_data": data["key"].tobytes(),
                "value_data": data["value"].tobytes(),
                "is_prefix": is_prefix,
                "size_bytes": meta["size_bytes"]
            })

    return blocks

def generate_synthetic_data(num_blocks: int = 1000) -> List[Dict]:
    config = BENCHMARK_CONFIG
    llama = LLAMA_CONFIG

    block_shape = (
        config.block_size_tokens,
        llama.num_layers,
        llama.num_kv_heads,
        llama.head_dim
    )

    blocks = []
    for i in range(num_blocks):

        rng = np.random.RandomState(i)

        key_data = rng.randn(*block_shape).astype(np.float16)
        value_data = rng.randn(*block_shape).astype(np.float16)

        import hashlib
        content = key_data.tobytes() + value_data.tobytes()
        block_id = hashlib.sha256(content).hexdigest()[:32]

        blocks.append({
            "block_id": block_id,
            "key_data": key_data.tobytes(),
            "value_data": value_data.tobytes(),
            "is_prefix": i < num_blocks * 0.1,
            "size_bytes": key_data.nbytes + value_data.nbytes
        })

    return blocks

def run_throughput_benchmark(
    adapter: StorageAdapter,
    blocks: List[Dict],
    num_iterations: int = 1
) -> Dict[str, BenchmarkStats]:
    results = {}

    print(f"\\n  [WRITE] {adapter.name}...")
    write_stats = BenchmarkStats(system_name=adapter.name, operation="write")

    for iteration in range(num_iterations):
        adapter.clear()

        start_time = time.perf_counter()

        for block in blocks:
            t0 = time.perf_counter()
            success = adapter.put(
                block["block_id"],
                block["key_data"],
                block["value_data"]
            )
            latency = (time.perf_counter() - t0) * 1000

            if success:
                write_stats.total_ops += 1
                write_stats.total_bytes += block["size_bytes"]
                write_stats.latencies.append(latency)
            else:
                write_stats.errors += 1

        adapter.flush()

    write_stats.duration_seconds = time.perf_counter() - start_time
    results["write"] = write_stats

    print(f"    Ops/s: {write_stats.ops_per_second:,.0f}")
    print(f"    Throughput: {write_stats.throughput_gbps:.2f} GB/s")
    print(f"    Avg latency: {write_stats.avg_latency_ms:.2f} ms")

    print(f"\\n  [READ] {adapter.name}...")
    read_stats = BenchmarkStats(system_name=adapter.name, operation="read")

    start_time = time.perf_counter()

    for iteration in range(num_iterations):

        indices = np.random.permutation(len(blocks))

        for idx in indices:
            block = blocks[idx]

            t0 = time.perf_counter()
            result = adapter.get(block["block_id"])
            latency = (time.perf_counter() - t0) * 1000

            if result is not None:
                read_stats.total_ops += 1
                read_stats.total_bytes += block["size_bytes"]
                read_stats.hits += 1
                read_stats.latencies.append(latency)
            else:
                read_stats.misses += 1

    read_stats.duration_seconds = time.perf_counter() - start_time
    results["read"] = read_stats

    print(f"    Ops/s: {read_stats.ops_per_second:,.0f}")
    print(f"    Throughput: {read_stats.throughput_gbps:.2f} GB/s")
    print(f"    Hit rate: {read_stats.hit_rate:.1%}")
    print(f"    Avg latency: {read_stats.avg_latency_ms:.2f} ms")

    return results

def run_shared_prefix_benchmark(
    adapter: StorageAdapter,
    blocks: List[Dict],
    num_sessions: int = 100
) -> BenchmarkStats:
    print(f"\\n  [SHARED PREFIX] {adapter.name}...")

    prefix_blocks = [b for b in blocks if b["is_prefix"]]
    unique_blocks = [b for b in blocks if not b["is_prefix"]]

    stats = BenchmarkStats(system_name=adapter.name, operation="shared_prefix")
    adapter.clear()

    start_time = time.perf_counter()

    for session_id in range(num_sessions):

        for block in prefix_blocks[:8]:
            t0 = time.perf_counter()
            success = adapter.put(
                block["block_id"],
                block["key_data"],
                block["value_data"]
            )
            latency = (time.perf_counter() - t0) * 1000

            stats.total_ops += 1
            stats.total_bytes += block["size_bytes"]
            stats.latencies.append(latency)

        unique_start = (session_id * 5) % len(unique_blocks)
        for i in range(5):
            block = unique_blocks[(unique_start + i) % len(unique_blocks)]

            adapter.put(
                block["block_id"],
                block["key_data"],
                block["value_data"]
            )
            stats.total_ops += 1
            stats.total_bytes += block["size_bytes"]

    adapter.flush()
    stats.duration_seconds = time.perf_counter() - start_time

    system_stats = adapter.get_stats()

    print(f"    Sessions: {num_sessions}")
    print(f"    Total ops: {stats.total_ops:,}")
    print(f"    Ops/s: {stats.ops_per_second:,.0f}")
    print(f"    System stats: {system_stats}")

    return stats

def main():
    parser = argparse.ArgumentParser(description="KV Cache Benchmark Runner")
    parser.add_argument("--systems", type=str, default="cascade,hdf5",
                        help="Comma-separated list of systems to benchmark")
    parser.add_argument("--workload", type=str, default="throughput",
                        choices=["throughput", "shared_prefix", "all"],
                        help="Workload type")
    parser.add_argument("--num_blocks", type=int, default=1000,
                        help="Number of blocks to test")
    parser.add_argument("--data_path", type=str,
                        default="${REPO_ROOT}/benchmark/data",
                        help="Path to benchmark data")
    parser.add_argument("--output", type=str,
                        default="${REPO_ROOT}/benchmark/results",
                        help="Output directory for results")
    args = parser.parse_args()

    systems = args.systems.split(",") if args.systems != "all" else [
        "cascade", "hdf5-independent", "hdf5-collective", "lmcache", "lmcache-redis", "pdc"
    ]

    data_path = Path(args.data_path)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KV Cache Benchmark")
    print("=" * 60)
    print(f"Systems: {systems}")
    print(f"Workload: {args.workload}")
    print(f"Blocks: {args.num_blocks}")
    print(f"Data: {data_path}")

    print("\\nLoading test data...")
    blocks = load_test_data(data_path, args.num_blocks)
    if not blocks:
        blocks = generate_synthetic_data(args.num_blocks)
    print(f"Loaded {len(blocks)} blocks")

    all_results = {}

    for system_name in systems:
        print(f"\\n{'=' * 40}")
        print(f"Benchmarking: {system_name.upper()}")
        print("=" * 40)

        try:
            adapter = get_adapter(system_name)

            if not adapter.initialize():
                print(f"  SKIP: {system_name} failed to initialize")
                continue

            if args.workload in ["throughput", "all"]:
                results = run_throughput_benchmark(adapter, blocks)
                all_results[f"{system_name}_write"] = results["write"].to_dict()
                all_results[f"{system_name}_read"] = results["read"].to_dict()

            if args.workload in ["shared_prefix", "all"]:
                stats = run_shared_prefix_benchmark(adapter, blocks)
                all_results[f"{system_name}_shared_prefix"] = stats.to_dict()

            adapter.close()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"benchmark_{args.workload}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"\\n{'System':12} {'GB/s':>10} {'Latency(ms)':>12}")
    print("-" * 64)

    for key, stats in all_results.items():
        print(f"{stats['system']:12s} {stats['throughput_gbps']:>10.2f} "
              f"{stats['avg_latency_ms']:>12.2f}")

    print(f"\\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
