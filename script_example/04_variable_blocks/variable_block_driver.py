
import os
import sys
import time
import json
import numpy as np
import argparse
from pathlib import Path
import subprocess

rank = int(os.environ.get('SLURM_PROCID', 0))
world = int(os.environ.get('SLURM_NTASKS', 1))
job_id = os.environ.get('SLURM_JOB_ID', 'local')

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

from benchmark.run_benchmark import get_adapter

def print_rank0(*args):
    if rank == 0:
        print(*args, flush=True)

def file_barrier(name):
    bar_dir = REPO_ROOT / "benchmark" / "tmp" / f"bar_{job_id}_{name}"
    bar_dir.mkdir(parents=True, exist_ok=True)
    (bar_dir / f"rank_{rank}").touch()
    while True:
        try:
            if len(list(bar_dir.iterdir())) >= world: break
        except: pass
        time.sleep(1)
    time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--block-size-mb", type=int, default=160)
    parser.add_argument("--num-write-blocks", type=int, default=100)
    parser.add_argument("--num-read-ops", type=int, default=500)
    parser.add_argument("--storage-path", type=str, default=None, help="Override default storage path")
    parser.add_argument("--redis-port", type=int, default=16379, help="Redis port (default: 16379)")
    parser.add_argument("--sigma", type=float, default=0.8, help="Log-normal sigma (default: 0.8)")
    args = parser.parse_args()

    block_size_bytes = args.block_size_mb * 1024 * 1024

    config = {}
    if args.system.lower() == "cascade":
        l_path = args.storage_path if args.storage_path else f"${REPO_ROOT}/benchmark/cascade_tail_{job_id}"
        config = {"gpu_capacity_gb": 32.0, "shm_capacity_gb": 64.0, "use_gpu": True, "lustre_path": l_path}
    elif "redis" in args.system.lower():
        tmp_h_dir = REPO_ROOT / "benchmark" / "tmp" / f"hosts_{job_id}"
        wait_count = 0
        while not (tmp_h_dir / "redis_host").exists() and wait_count < 30:
            time.sleep(1)
            wait_count += 1
        with open(tmp_h_dir / "redis_host", 'r') as f:
            shared_redis_host = f.read().strip()
        config = {"host": shared_redis_host, "port": args.redis_port}
    elif args.system.lower() == "lmcache":
        l_path = args.storage_path if args.storage_path else f"${REPO_ROOT}/benchmark/lmcache_tail_{job_id}"
        config = {"storage_path": l_path}
    elif args.system.lower() == "pdc":
        l_path = args.storage_path if args.storage_path else f"${REPO_ROOT}/benchmark/pdc_tail_{job_id}"
        config = {"storage_path": l_path}
    elif "hdf5" in args.system.lower():
        l_path = args.storage_path if args.storage_path else f"${REPO_ROOT}/benchmark/tmp/h5_tail_{job_id}.h5"
        config = {"file_path": l_path, "use_mpi": True}

    adapter = get_adapter(args.system, config=config)
    if not adapter.initialize():
        print(f"Rank {rank}: Failed to initialize {args.system}")
        return

    print_rank0(f"Phase 1: Writing {args.num_write_blocks} variable blocks (mean {args.block_size_mb} MB) with {args.system}...")
    my_keys = [f"tail_r{rank}_b{i}" for i in range(args.num_write_blocks)]

    np.random.seed(rank + 42)

    mu = np.log(args.block_size_mb) - (args.sigma**2 / 2)
    mb_sizes = np.random.lognormal(mu, args.sigma, args.num_write_blocks)
    mb_sizes = np.clip(mb_sizes, 1.0, 1024.0)

    max_bytes = int(np.max(mb_sizes) * 1024 * 1024)
    data_buffer = np.random.randint(0, 256, max_bytes, dtype=np.uint8).tobytes()

    key_size_map = {}

    for i, key in enumerate(my_keys):
        s_bytes = int(mb_sizes[i] * 1024 * 1024)
        s_bytes = (s_bytes // 8) * 8
        mk, mv = data_buffer[:s_bytes//2], data_buffer[s_bytes//2:s_bytes]
        adapter.put(key, mk, mv)
        key_size_map[key] = s_bytes

    adapter.flush()
    if hasattr(adapter, "sync_metadata"): adapter.sync_metadata()
    file_barrier("warmup")

    print_rank0(f"Phase 2: Performing {args.num_read_ops} variable random reads concurrent across {world} ranks...")

    all_ranks = list(range(world))
    latencies = []
    total_bytes_read = 0

    def get_rank_sizes(r_id):
        np.random.seed(r_id + 42)
        mu_r = np.log(args.block_size_mb) - (args.sigma**2 / 2)
        sizes = np.random.lognormal(mu_r, args.sigma, args.num_write_blocks)
        sizes = np.clip(sizes, 1.0, 1024.0)
        return [(int(s * 1024 * 1024) // 8) * 8 for s in sizes]

    file_barrier("measure_start")
    start_time = time.perf_counter()

    for _ in range(args.num_read_ops):
        target_r = np.random.choice(all_ranks)
        target_b = np.random.randint(0, args.num_write_blocks)
        key = f"tail_r{target_r}_b{target_b}"

        target_sizes = get_rank_sizes(target_r)
        total_bytes_read += target_sizes[target_b]

        t0 = time.perf_counter()
        adapter.get(key)
        t_lat = (time.perf_counter() - t0) * 1000
        latencies.append(t_lat)

    end_time = time.perf_counter()
    duration = end_time - start_time
    file_barrier("measure_done")

    local_lats = np.array(latencies)
    p50 = np.percentile(local_lats, 50)
    p95 = np.percentile(local_lats, 95)
    p99 = np.percentile(local_lats, 99)
    p999 = np.percentile(local_lats, 99.9)
    avg = np.mean(local_lats)
    std = np.std(local_lats)
    mx = np.max(local_lats)

    thru = args.num_read_ops / duration
    bw_gbps = (total_bytes_read / (1024.0**3)) / duration

    res_dir = REPO_ROOT / "benchmark" / "tmp" / f"tail_res_{job_id}"
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(res_dir / f"rank_{rank}.json", "w") as f:
        json.dump({
            "avg": avg, "std": std, "p50": p50, "p95": p95, "p99": p99, "p999": p999, "max": mx,
            "thru": thru, "bw": bw_gbps,
            "all_lats": local_lats.tolist()
        }, f)

    file_barrier("results_gathered")

    if rank == 0:
        global_lats = []
        global_thrus = []
        global_bws = []
        for r in range(world):
            with open(res_dir / f"rank_{r}.json", "r") as f:
                data = json.load(f)
                global_lats.extend(data["all_lats"])
                global_thrus.append(data["thru"])
                global_bws.append(data["bw"])

        global_lats = np.array(global_lats)
        gp50 = np.percentile(global_lats, 50)
        gp95 = np.percentile(global_lats, 95)
        gp99 = np.percentile(global_lats, 99)
        gp999 = np.percentile(global_lats, 99.9)
        gavg = np.mean(global_lats)
        agg_thru = np.sum(global_thrus)
        agg_bw = np.sum(global_bws)

        print(f"\n" + "="*60)
        print(f" TTFT & BANDWIDTH DISTRIBUTION: {args.system} ({world} Nodes)")
        print(f"="*60)
        print(f"  Samples:         {len(global_lats)}")
        print(f"  Avg TTFT:        {gavg:8.2f} ms")
        print(f"  P50 TTFT:        {gp50:8.2f} ms")
        print(f"  P99 TTFT:        {gp99:8.2f} ms")
        print(f"  P99.9 TTFT:      {gp999:8.2f} ms")
        print(f"  Max TTFT:        {np.max(global_lats):8.2f} ms")
        print(f"  --------------------------------------------------")
        print(f"  Avg Throughput:  {agg_thru:8.2f} req/s")
        print(f"  Aggregated BW:   {agg_bw:8.2f} GB/s")
        print(f"="*60)
        print(" Done")

    adapter.close()

if __name__ == "__main__":
    main()
