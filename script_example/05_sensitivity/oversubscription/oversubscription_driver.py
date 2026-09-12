import os
import time
import argparse
import numpy as np
from pathlib import Path
import sys
import json

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

from benchmark.run_benchmark import get_adapter

rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

def print_rank0(*args):
    if rank == 0:
        print(*args, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--node-count", type=int, default=1)
    parser.add_argument("--block-size-mb", type=int, default=160, help="160=Llama, 320=Qwen")

    parser.add_argument("--total-data-per-node-gb", type=float, default=400.0)
    args = parser.parse_args()

    block_size = args.block_size_mb * 1024 * 1024
    model_name = "Llama-2" if args.block_size_mb == 160 else "Qwen-2.5-72B"

    config = {}
    if args.system.lower() == "cascade":
        config = {
            "gpu_capacity_gb": 32.0,
            "shm_capacity_gb": 64.0,
            "semantic_eviction": True,
            "lustre_path": f"${REPO_ROOT}/benchmark/cascade_10x_{job_id}"
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
        config = {"storage_path": f"${REPO_ROOT}/benchmark/lmcache_10x_{job_id}"}
    elif args.system.lower() == "pdc":
        config = {"storage_path": f"${REPO_ROOT}/benchmark/pdc_10x_{job_id}"}
    elif "hdf5" in args.system.lower():
        config = {"file_path": f"${REPO_ROOT}/benchmark/tmp/h5_10x_{job_id}.h5", "use_mpi": True}

    adapter = get_adapter(args.system, config)
    if not adapter.initialize():
        print(f"Rank {rank}: Failed to initialize {args.system}")
        return

    adapter.clear()
    if hasattr(adapter, "barrier"): adapter.barrier()

    blocks_per_node = int((args.total_data_per_node_gb * 1024**3) / block_size)
    num_important = int(blocks_per_node * 0.1)
    num_disposable = blocks_per_node - num_important

    gpu_hbm_gb = 40 * 4
    oversubscription_ratio = args.total_data_per_node_gb / gpu_hbm_gb

    print_rank0(f"\n{'='*70}")
    print_rank0(f" Oversubscription 10x Test: {args.system} | N={args.node_count} | {model_name}")
    print_rank0(f"{'='*70}")
    print_rank0(f"  GPU HBM per node: {gpu_hbm_gb} GB")
    print_rank0(f"  Total data per node: {args.total_data_per_node_gb} GB")
    print_rank0(f"  Oversubscription ratio: {oversubscription_ratio:.1f}x")
    print_rank0(f"  Blocks per node: {blocks_per_node}")
    print_rank0(f"  Important blocks (10%): {num_important}")
    print_rank0(f"  Disposable blocks (90%): {num_disposable}")
    print_rank0(f"{'='*70}\n")

    prefix_keys = [f"prefix_10x_n{args.node_count}_r{rank}_b{i}" for i in range(num_important)]
    important_data = np.random.randint(0, 255, block_size, dtype=np.uint8).tobytes()
    ik, iv = important_data[:len(important_data)//2], important_data[len(important_data)//2:]

    print_rank0(f"Step 1: Writing {num_important} important blocks (System Prompts)...")

    t0 = time.time()
    for i, key in enumerate(prefix_keys):
        if hasattr(adapter, "put_prefix"):
            adapter.put_prefix(key, ik, iv)
        else:
            adapter.put(key, ik, iv)
        if rank == 0 and i % 100 == 0 and i > 0:
            print(f"  [Progress] {i}/{num_important} important blocks...", flush=True)

    adapter.flush()
    if hasattr(adapter, "barrier"): adapter.barrier()
    t_write_important = time.time() - t0
    print_rank0(f"   Wrote {num_important} important blocks in {t_write_important:.2f}s")

    print_rank0(f"\nStep 2: Writing {num_disposable} disposable blocks (Trigger 10x Eviction)...")

    t0 = time.time()
    for i in range(num_disposable):
        unique_data = np.full(block_size, (rank + i) % 256, dtype=np.uint8).tobytes()
        uk, uv = unique_data[:len(unique_data)//2], unique_data[len(unique_data)//2:]
        adapter.put(f"disposable_10x_n{args.node_count}_r{rank}_b{i}", uk, uv)

        if rank == 0 and i % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (num_disposable - i) / rate if rate > 0 else 0
            print(f"  [Progress] {i}/{num_disposable} blocks ({rate:.1f} blk/s, ETA: {eta:.1f}s)...", flush=True)

    adapter.flush()
    if hasattr(adapter, "barrier"): adapter.barrier()
    t_write_disposable = time.time() - t0

    write_throughput_gb_s = (num_disposable * block_size * world) / (t_write_disposable * 1024**3) if t_write_disposable > 0 else 0
    print_rank0(f"   Wrote {num_disposable} disposable blocks in {t_write_disposable:.2f}s")
    print_rank0(f"  Write Throughput: {write_throughput_gb_s:.2f} GB/s (aggregate)")

    if hasattr(adapter, "sync_metadata"): adapter.sync_metadata()
    if hasattr(adapter, "barrier"): adapter.barrier()

    print_rank0(f"\nStep 3: Measuring TTFT for important blocks after 10x oversubscription...")
    prefix_latencies = []
    prefix_hits = 0

    t0_read = time.time()
    for key in prefix_keys:
        t_req = time.time()
        res = adapter.get(key)
        lat = (time.time() - t_req) * 1000
        if res:
            prefix_hits += 1
            prefix_latencies.append(lat)

    t_read_important = time.time() - t0_read

    avg_prefix_ttft = np.mean(prefix_latencies) if prefix_latencies else 0
    p95_ttft = np.percentile(prefix_latencies, 95) if prefix_latencies else 0
    p99_ttft = np.percentile(prefix_latencies, 99) if prefix_latencies else 0
    retention_rate = (prefix_hits / num_important) * 100 if num_important > 0 else 0

    res_dir = REPO_ROOT / "benchmark" / "tmp" / f"oversubscription_10x_res_{job_id}"
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(res_dir / f"rank_{rank}_n{args.node_count}.json", "w") as f:
        json.dump({
            "ttft_avg": avg_prefix_ttft,
            "ttft_p95": p95_ttft,
            "ttft_p99": p99_ttft,
            "retention": retention_rate,
            "write_disposable_s": t_write_disposable,
            "write_throughput_gb_s": (num_disposable * block_size) / (t_write_disposable * 1024**3) if t_write_disposable > 0 else 0
        }, f)

    if hasattr(adapter, "barrier"): adapter.barrier()

    if rank == 0:
        time.sleep(2)
        all_ttfts_avg = []
        all_ttfts_p95 = []
        all_ttfts_p99 = []
        all_retentions = []
        all_write_throughputs = []

        for r in range(world):
            try:
                with open(res_dir / f"rank_{r}_n{args.node_count}.json", "r") as f:
                    d = json.load(f)
                    all_ttfts_avg.append(d["ttft_avg"])
                    all_ttfts_p95.append(d["ttft_p95"])
                    all_ttfts_p99.append(d["ttft_p99"])
                    all_retentions.append(d["retention"])
                    all_write_throughputs.append(d["write_throughput_gb_s"])
            except: pass

        final_ttft_avg = np.mean(all_ttfts_avg)
        final_ttft_p95 = np.mean(all_ttfts_p95)
        final_ttft_p99 = np.mean(all_ttfts_p99)
        final_retention = np.mean(all_retentions)
        final_write_throughput = sum(all_write_throughputs)

        print_rank0(f"\n{'='*70}")
        print_rank0(f" Result: {args.system} | N={args.node_count} | {model_name} ({args.block_size_mb}MB)")
        print_rank0(f"{'='*70}")
        print_rank0(f"  Oversubscription: {oversubscription_ratio:.1f}x ({args.total_data_per_node_gb} GB / {gpu_hbm_gb} GB)")
        print_rank0(f"  Important Block Retention: {final_retention:.1f}%")
        print_rank0(f"  TTFT (Avg): {final_ttft_avg:.2f} ms")
        print_rank0(f"  TTFT (P95): {final_ttft_p95:.2f} ms")
        print_rank0(f"  TTFT (P99): {final_ttft_p99:.2f} ms")
        print_rank0(f"  Write Throughput: {final_write_throughput:.2f} GB/s")

        if final_retention < 50:
            status = " CRITICAL - Low Retention"
        elif final_ttft_p99 < 100:
            status = " EXCELLENT"
        elif final_ttft_p99 < 200:
            status = " ACCEPTABLE"
        else:
            status = " DEGRADED"

        print_rank0(f"  Status: {status}")
        print_rank0(f"{'='*70}\n")

    adapter.close()

if __name__ == "__main__":
    main()
