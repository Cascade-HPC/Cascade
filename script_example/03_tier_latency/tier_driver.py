
import numpy as np
import time
import json
import argparse
from pathlib import Path
import sys

try:
    import cascade_cpp
except ImportError:
    print("ERROR: cascade_cpp not found. Make sure CASCADE is built.")
    sys.exit(1)

def measure_tier_latency(store, block_size_mb, num_iterations=100, tier_name="unknown"):
    block_size = block_size_mb * 1024 * 1024

    data = np.random.randint(0, 255, size=block_size, dtype=np.uint8)

    write_times = []
    read_times = []

    print(f"\n  Measuring {tier_name} latency ({num_iterations} iterations)...")

    for i in range(num_iterations):
        key = f"benchmark_block_{tier_name}_{i}"

        start = time.perf_counter()
        store.put(key, data)
        write_time = (time.perf_counter() - start) * 1000
        write_times.append(write_time)

        buffer = np.empty_like(data)
        start = time.perf_counter()
        found, size = store.get(key, buffer)
        read_time = (time.perf_counter() - start) * 1000

        if found:
            read_times.append(read_time)

        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{num_iterations}", end='\r', flush=True)

    print(f"    Progress: {num_iterations}/{num_iterations} ")

    results = {
        'tier': tier_name,
        'block_size_mb': block_size_mb,
        'num_iterations': num_iterations,
        'write': {
            'mean_ms': np.mean(write_times),
            'median_ms': np.median(write_times),
            'p95_ms': np.percentile(write_times, 95),
            'p99_ms': np.percentile(write_times, 99),
            'std_ms': np.std(write_times),
        },
        'read': {
            'mean_ms': np.mean(read_times),
            'median_ms': np.median(read_times),
            'p95_ms': np.percentile(read_times, 95),
            'p99_ms': np.percentile(read_times, 99),
            'std_ms': np.std(read_times),
        }
    }

    print(f"    Write: {results['write']['mean_ms']:.2f} ms (P95: {results['write']['p95_ms']:.2f} ms)")
    print(f"    Read:  {results['read']['mean_ms']:.2f} ms (P95: {results['read']['p95_ms']:.2f} ms)")

    return results

def run_tier_microbenchmark(args):
    print(f"\n{'='*80}")
    print(f"STORAGE TIER LATENCY MICROBENCHMARK")
    print(f"{'='*80}")
    print(f"Block Size: {args.block_size_mb} MB")
    print(f"Iterations per tier: {args.num_iterations}")
    print(f"{'='*80}")

    cfg = cascade_cpp.DistributedConfig()

    if args.tier == 'gpu':

        cfg.gpu_capacity_per_device = 10 * 1024**3
        cfg.dram_capacity = 0
        tier_name = "GPU (Tier 1)"

    elif args.tier == 'dram':

        cfg.gpu_capacity_per_device = 0
        cfg.dram_capacity = 10 * 1024**3
        tier_name = "DRAM (Tier 2)"

    elif args.tier == 'lustre':

        cfg.gpu_capacity_per_device = 0
        cfg.dram_capacity = 0
        cfg.enable_lustre = True
        tier_name = "Lustre (Tier 5)"

    else:
        print(f"ERROR: Unknown tier '{args.tier}'")
        sys.exit(1)

    cfg.num_gpus_per_node = 1
    cfg.dedup_enabled = False

    print(f"\nInitializing CASCADE store for {tier_name}...")
    store = cascade_cpp.DistributedStore(cfg)

    results = measure_tier_latency(
        store,
        args.block_size_mb,
        args.num_iterations,
        tier_name
    )

    output_file = Path(args.output_json)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n Results saved to: {output_file}")
    print(f"{'='*80}\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Storage tier latency microbenchmark")
    parser.add_argument("--tier", type=str, required=True,
                       choices=['gpu', 'dram', 'lustre'],
                       help="Storage tier to benchmark")
    parser.add_argument("--block-size-mb", type=int, default=320,
                       help="Block size in MB")
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="Number of iterations")
    parser.add_argument("--output-json", type=str, required=True,
                       help="Output JSON file")

    args = parser.parse_args()

    try:
        run_tier_microbenchmark(args)
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
