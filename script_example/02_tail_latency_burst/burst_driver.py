import os
import time
import argparse
import numpy as np
from pathlib import Path
import sys
import json
import threading
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

from benchmark.run_benchmark import get_adapter

rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

def print_rank0(*args):
    if rank == 0:
        print(*args, flush=True)

def worker_thread(adapter, block_data, key_prefix, num_ops, latencies, thread_id):
    for i in range(num_ops):
        key = f"{key_prefix}_t{thread_id}_op{i}"
        bk, bv = block_data

        t0 = time.time()
        adapter.put(key, bk, bv)
        write_lat = (time.time() - t0) * 1000

        t0 = time.time()
        adapter.get(key)
        read_lat = (time.time() - t0) * 1000

        latencies[thread_id].append({
            'write': write_lat,
            'read': read_lat
        })

def run_phase(adapter, block_data, phase_name, num_threads, ops_per_thread, key_prefix):
    print_rank0(f"  Phase: {phase_name} ({num_threads} concurrent threads, {ops_per_thread} ops each)")

    latencies = defaultdict(list)
    threads = []

    t_start = time.time()
    for tid in range(num_threads):
        t = threading.Thread(
            target=worker_thread,
            args=(adapter, block_data, f"{key_prefix}_{phase_name}", ops_per_thread, latencies, tid)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    duration = time.time() - t_start

    all_write_lats = []
    all_read_lats = []
    for thread_lats in latencies.values():
        for entry in thread_lats:
            all_write_lats.append(entry['write'])
            all_read_lats.append(entry['read'])

    total_ops = len(all_write_lats)
    throughput = total_ops / duration if duration > 0 else 0

    return {
        'phase': phase_name,
        'threads': num_threads,
        'total_ops': total_ops,
        'duration_s': duration,
        'throughput_ops_s': throughput,
        'write_avg_ms': np.mean(all_write_lats),
        'write_p95_ms': np.percentile(all_write_lats, 95),
        'write_p99_ms': np.percentile(all_write_lats, 99),
        'read_avg_ms': np.mean(all_read_lats),
        'read_p95_ms': np.percentile(all_read_lats, 95),
        'read_p99_ms': np.percentile(all_read_lats, 99),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--node-count", type=int, default=1)
    parser.add_argument("--block-size-mb", type=int, default=160, help="160=Llama, 320=Qwen")
    args = parser.parse_args()

    block_size = args.block_size_mb * 1024 * 1024
    model_name = "Llama-2" if args.block_size_mb == 160 else "Qwen-2.5-72B"

    config = {}
    if args.system.lower() == "cascade":
        config = {
            "gpu_capacity_gb": 32.0,
            "shm_capacity_gb": 64.0,
            "semantic_eviction": True,
            "lustre_path": f"${REPO_ROOT}/benchmark/cascade_bursty_{job_id}"
        }
    elif "redis" in args.system.lower():
        tmp_h_dir = REPO_ROOT / "benchmark" / "tmp" / f"hosts_{job_id}"
        tmp_h_dir.mkdir(parents=True, exist_ok=True)
        wait_count = 0
        while not (tmp_h_dir / "redis_host").exists() and wait_count < 30:
            time.sleep(1)
            wait_count += 1
        with open(tmp_h_dir / "redis_host", 'r') as f:
            shared_redis_host = f.read().strip()
        config = {"host": shared_redis_host, "port": 16379}
    elif args.system.lower() == "lmcache":
        config = {"storage_path": f"${REPO_ROOT}/benchmark/lmcache_bursty_{job_id}"}
    elif args.system.lower() == "pdc":
        config = {"storage_path": f"${REPO_ROOT}/benchmark/pdc_bursty_{job_id}"}
    elif "hdf5" in args.system.lower():
        config = {"file_path": f"${REPO_ROOT}/benchmark/tmp/h5_bursty_{job_id}.h5", "use_mpi": True}

    adapter = get_adapter(args.system, config)
    if not adapter.initialize():
        print(f"Rank {rank}: Failed to initialize {args.system}")
        return

    adapter.clear()
    if hasattr(adapter, "barrier"): adapter.barrier()

    block_data_raw = np.random.randint(0, 255, block_size, dtype=np.uint8).tobytes()
    block_data = (block_data_raw[:len(block_data_raw)//2], block_data_raw[len(block_data_raw)//2:])

    print_rank0(f"\n{'='*70}")
    print_rank0(f" Bursty Traffic Benchmark: {args.system} | N={args.node_count} | {model_name} ({args.block_size_mb}MB)")
    print_rank0(f"{'='*70}")

    results = []
    key_prefix = f"bursty_n{args.node_count}_r{rank}"

    results.append(run_phase(adapter, block_data, "calm_baseline", num_threads=2, ops_per_thread=20, key_prefix=key_prefix))
    adapter.flush()
    if hasattr(adapter, "barrier"): adapter.barrier()
    time.sleep(2)

    results.append(run_phase(adapter, block_data, "burst_10x", num_threads=20, ops_per_thread=10, key_prefix=key_prefix))
    adapter.flush()
    if hasattr(adapter, "barrier"): adapter.barrier()
    time.sleep(2)

    results.append(run_phase(adapter, block_data, "calm_recovery", num_threads=2, ops_per_thread=20, key_prefix=key_prefix))
    adapter.flush()
    if hasattr(adapter, "barrier"): adapter.barrier()

    res_dir = REPO_ROOT / "benchmark" / "tmp" / f"bursty_res_{job_id}"
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(res_dir / f"rank_{rank}_n{args.node_count}.json", "w") as f:
        json.dump(results, f, indent=2)

    if hasattr(adapter, "barrier"): adapter.barrier()

    if rank == 0:
        time.sleep(2)
        all_results = []
        for r in range(world):
            try:
                with open(res_dir / f"rank_{r}_n{args.node_count}.json", "r") as f:
                    all_results.extend(json.load(f))
            except: pass

        phase_stats = defaultdict(lambda: {
            'total_ops': 0,
            'total_duration': 0,
            'write_lats': [],
            'read_lats': []
        })

        for res in all_results:
            phase = res['phase']
            phase_stats[phase]['total_ops'] += res['total_ops']
            phase_stats[phase]['total_duration'] += res['duration_s']
            phase_stats[phase]['write_lats'].append(res['write_p99_ms'])
            phase_stats[phase]['read_lats'].append(res['read_p99_ms'])

        print_rank0(f"\n{'='*70}")
        print_rank0(f" Aggregated Results: {args.system} @ {args.node_count} Nodes")
        print_rank0(f"{'='*70}")

        for phase in ['calm_baseline', 'burst_10x', 'calm_recovery']:
            stats = phase_stats[phase]
            avg_throughput = stats['total_ops'] / (stats['total_duration'] / world) if stats['total_duration'] > 0 else 0
            write_p99 = np.mean(stats['write_lats'])
            read_p99 = np.mean(stats['read_lats'])

            print_rank0(f"\n[{phase.upper()}]")
            print_rank0(f"  Total Ops: {stats['total_ops']}")
            print_rank0(f"  Throughput: {avg_throughput:.2f} ops/s")
            print_rank0(f"  Write P99: {write_p99:.2f} ms")
            print_rank0(f"  Read P99: {read_p99:.2f} ms")

        baseline_p99 = np.mean(phase_stats['calm_baseline']['read_lats'])
        burst_p99 = np.mean(phase_stats['burst_10x']['read_lats'])
        recovery_p99 = np.mean(phase_stats['calm_recovery']['read_lats'])

        degradation = ((burst_p99 - baseline_p99) / baseline_p99 * 100) if baseline_p99 > 0 else 0
        recovery_rate = ((baseline_p99 - recovery_p99) / baseline_p99 * 100) if baseline_p99 > 0 else 0

        print_rank0(f"\n{'='*70}")
        print_rank0(f" Stability Analysis:")
        print_rank0(f"  Burst Degradation: {degradation:+.1f}%")
        print_rank0(f"  Recovery Quality: {recovery_rate:+.1f}% (vs baseline)")
        print_rank0(f"  Status: {' STABLE' if abs(degradation) < 50 else ' DEGRADED'}")
        print_rank0(f"{'='*70}\n")

    adapter.close()

if __name__ == "__main__":
    main()
