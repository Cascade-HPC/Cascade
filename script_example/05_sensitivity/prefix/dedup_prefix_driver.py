import os
import time
import argparse
import numpy as np
from pathlib import Path
import subprocess
import socket
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

def file_barrier(name, run_id=None):
    rid = run_id or job_id
    bar_dir = REPO_ROOT / "benchmark" / "tmp" / f"bar_{rid}_{name}"
    bar_dir.mkdir(parents=True, exist_ok=True)
    (bar_dir / f"rank_{rank}").touch()
    while True:
        try:
            if len(list(bar_dir.iterdir())) >= world: break
        except: pass
        time.sleep(1)
    time.sleep(2)

def force_lustre_sync(path):
    try:
        p = Path(path)
        if p.exists():
            subprocess.run(["ls", "-ld", str(p)], capture_output=True, timeout=5)
            if p.is_dir():
                subprocess.run(["ls", "-a", str(p)], capture_output=True, timeout=5)
    except: pass

class SyntheticKVLoader:
    def __init__(self, block_size_bytes: int):
        self.block_size = block_size_bytes

    def load(self, block_id: str):
        seed = abs(hash(block_id)) % (2**31)
        rng = np.random.default_rng(seed)
        data = rng.integers(0, 256, self.block_size, dtype=np.uint8).tobytes()
        mid = len(data) // 2
        return data[:mid], data[mid:]

def run_dedup_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--prefix-blocks", type=int, default=32, help="Number of blocks in the shared prefix")
    parser.add_argument("--block-size-mb", type=int, default=160)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    block_size_bytes = args.block_size_mb * 1024 * 1024
    loader = SyntheticKVLoader(block_size_bytes)

    name = args.system
    rid = args.run_id or job_id

    print_rank0(f"\n" + "="*60)
    print_rank0(f" Global Dedup & Prefix Sharing: {name} | {args.block_size_mb}MB blocks")
    print_rank0(f"Ranks: {world}, Prefix Size: {args.prefix_blocks} blocks")
    print_rank0("="*60)

    config = {}
    if name.lower() == "cascade":
        config = {"gpu_capacity_gb": 30.0, "shm_capacity_gb": 140.0, "use_gpu": True}
    elif "redis" in name.lower():
        r_port = int(os.environ.get("REDIS_PORT", 16379))

        tmp_h_dir = REPO_ROOT / "benchmark" / "tmp" / f"hosts_{job_id}"
        tmp_h_dir.mkdir(parents=True, exist_ok=True)

        wait_count = 0
        while not (tmp_h_dir / "redis_host").exists() and wait_count < 30:
            time.sleep(1)
            wait_count += 1

        with open(tmp_h_dir / "redis_host", 'r') as f:
            shared_redis_host = f.read().strip()
        config = {"host": shared_redis_host, "port": r_port}
    elif name.lower() == "lmcache":
        config = {"storage_path": f"${REPO_ROOT}/benchmark/lmcache_store_dedup_{rid}"}
    elif name.lower() == "pdc":
        config = {"storage_path": f"${REPO_ROOT}/benchmark/pdc_store_dedup_{rid}"}
    elif "hdf5" in name.lower():
        config = {"file_path": f"${REPO_ROOT}/benchmark/tmp/h5_dedup_{rid}.h5", "use_mpi": True}

    adapter = get_adapter(name, config)
    if not adapter.initialize():
        print_rank0(f" [{name}] Failed")
        return

    try:
        adapter.clear()
        file_barrier("cleared", rid)

        prefix_keys = [f"shared_prefix_b{i}" for i in range(args.prefix_blocks)]

        if rank == 0:
            print(f"[{name}] Rank 0 writing shared prefix...", flush=True)
            for rk in prefix_keys:
                k_data, v_data = loader.load(rk)
                if hasattr(adapter, "put_prefix"):
                    adapter.put_prefix(rk, k_data, v_data)
                else:
                    adapter.put(rk, k_data, v_data)

        adapter.flush()

        if hasattr(adapter, "sync_metadata"):
            adapter.sync_metadata()

        file_barrier("prefix_written", rid)

        if name.lower() != "cascade" and "redis" not in name.lower():
            time.sleep(10)
            force_lustre_sync(config.get("storage_path", "/tmp"))

        print(f"[Rank {rank}] Starting shared prefix read...", flush=True)
        read_latencies = []
        f0 = time.time()
        for rk in prefix_keys:
            t_req = time.time()
            res = None
            for attempt in range(20):
                res = adapter.get(rk)
                if res: break
                time.sleep(0.5)
            if res:
                read_latencies.append((time.time() - t_req) * 1000)

        local_avg_ttft = np.mean(read_latencies) if read_latencies else 0
        local_duration = time.time() - f0
        local_throughput = len(read_latencies) / local_duration if local_duration > 0 else 0

        stats_dir = REPO_ROOT / "benchmark" / "tmp" / f"stats_dedup_{rid}"
        stats_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_dir / f"rank_{rank}.json", 'w') as f:
            json.dump({"ttft": local_avg_ttft, "thru": local_throughput, "total": len(read_latencies)}, f)

        file_barrier("stats_ready", rid)

        if rank == 0:
            all_ttfts = []
            all_thrus = []
            total_blocks = 0
            for r in range(world):
                try:
                    with open(stats_dir / f"rank_{r}.json", 'r') as f:
                        s = json.load(f)
                        all_ttfts.append(s["ttft"])
                        all_thrus.append(s["thru"])
                        total_blocks += s["total"]
                except: continue

            print(f"\n---  Dedup/Sharing Results ({world} Nodes) ---")
            print(f"Avg TTFT: {np.mean(all_ttfts):.2f} ms")
            print(f"Aggregate Throughput: {np.sum(all_thrus):.2f} req/s")
            print(f"Total Shared Blocks Served: {total_blocks}")
            print(f"Efficiency: {np.sum(all_thrus) / (np.sum(all_thrus[0]) if all_thrus else 1):.2f}x speedup vs single")

    finally:
        adapter.close()

    file_barrier("done", rid)

if __name__ == "__main__":
    run_dedup_benchmark()
