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

def force_lustre_sync(path):
    try:
        p = Path(path)
        if p.exists():
            subprocess.run(["ls", "-ld", str(p)], capture_output=True, timeout=5)
            if p.is_dir():
                subprocess.run(["ls", "-a", str(p)], capture_output=True, timeout=5)
        else:
            subprocess.run(["ls", "-ld", str(p.parent)], capture_output=True, timeout=5)
    except: pass

def get_all_hosts(run_id=None):
    rid = run_id or job_id
    my_host = socket.gethostname()
    tmp_dir = REPO_ROOT / "benchmark" / "tmp" / f"hosts_{rid}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with open(tmp_dir / f"rank_{rank}", 'w') as f:
        f.write(my_host)

    all_hosts = {}
    for _ in range(120):
        try:
            files = list(tmp_dir.iterdir())
            if len(files) >= world:
                for f in files:
                    try:
                        r_idx = int(f.name.split('_')[1])
                        with open(f, 'r') as hf:
                            all_hosts[r_idx] = hf.read().strip()
                    except: continue
                if len(all_hosts) >= world: break
        except: pass
        time.sleep(1)
    return all_hosts

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

class SyntheticKVLoader:
    def __init__(self, block_size_bytes: int, num_blocks: int = 3200):
        self.block_size = block_size_bytes
        self.block_ids = [f"synth_b{i}" for i in range(num_blocks)]
        print_rank0(f"Synthetic loader: {num_blocks} blocks x {block_size_bytes/1024/1024:.0f} MB")

    def load(self, block_id: str):

        seed = abs(hash(block_id)) % (2**31)
        rng = np.random.default_rng(seed)
        data = rng.integers(0, 256, self.block_size, dtype=np.uint8).tobytes()
        mid = len(data) // 2
        return data[:mid], data[mid:]

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, help="System name (e.g. Cascade, HDF5-Native, PDC)")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--use-mpi-io", action="store_true", help="Force MPI-IO for HDF5")
    parser.add_argument("--run-id", type=str, default=None, help="Unique ID for this run within a job")
    parser.add_argument("--block-size-mb", type=int, default=160,
                        help="Block size in MB (default: 160 for Llama-3-70B, use 320 for Qwen-2.5-72B)")
    parser.add_argument("--tier-mode", choices=["hot", "warm", "cold"], default=None,
                        help="Tier retrieval mode for Fig 8. hot=GPU HBM, warm=DRAM, cold=Lustre.")
    args = parser.parse_args()

    block_size_bytes = args.block_size_mb * 1024 * 1024
    real_block_size = 160 * 1024 * 1024
    if args.block_size_mb != 160:

        loader = SyntheticKVLoader(block_size_bytes, num_blocks=3200)
        print_rank0(f"Using SYNTHETIC {args.block_size_mb}MB blocks (Qwen-2.5-72B equivalent)")
    else:

        real_loader = type('RealKVLoader', (), {})()
        real_loader.base_dir = Path("${SCRATCH}/cascade_kv_cache")
        real_loader.all_blocks = {}
        real_loader.block_ids = []
        index_path = real_loader.base_dir / "global_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    real_loader.all_blocks = data['blocks']
                    real_loader.block_ids = list(real_loader.all_blocks.keys())
            except Exception as e:
                print_rank0(f"Error loading real KV index: {e}")
        if real_loader.block_ids:
            print_rank0(f"KV data loader initialized with {len(real_loader.block_ids)} real 160MB blocks.")

            def _load(block_id):
                if not real_loader.all_blocks:
                    data = np.zeros(160*1024*1024, dtype=np.uint8).tobytes()
                    return data[:len(data)//2], data[len(data)//2:]
                loc = real_loader.all_blocks[block_id]
                path = real_loader.base_dir / loc['file']
                with open(path, 'rb') as f:
                    f.seek(loc['offset'])
                    d = f.read(loc['size'])
                    return d[:len(d)//2], d[len(d)//2:]
            real_loader.load = _load
            loader = real_loader
        else:
            print_rank0("WARNING: Real KV data not found! Using synthetic 160MB fallback.")
            loader = SyntheticKVLoader(160*1024*1024, num_blocks=3200)

    name = args.system
    rid = args.run_id or job_id
    print_rank0(f"\n" + "="*60)
    print_rank0(f" Standalone Benchmark: {name} | {args.block_size_mb}MB blocks (Rank {rank}/{world}, RunID: {rid})")
    print_rank0("="*60)

    config = {}

    my_host = socket.gethostname()

    all_hosts = get_all_hosts(rid)

    if name.lower() == "cascade":
        config = {"gpu_capacity_gb": 30.0, "shm_capacity_gb": 140.0, "use_gpu": True}
    elif "redis" in name.lower():
        r_port = int(os.environ.get("REDIS_PORT", 16379))

        config = {"host": my_host, "port": r_port}

        time.sleep(10)
    elif name.lower() == "lmcache":
        config = {"storage_path": f"${REPO_ROOT}/benchmark/lmcache_store_{rid}"}
    elif name.lower() == "pdc":
        config = {"storage_path": f"${REPO_ROOT}/benchmark/pdc_store_{rid}"}
    elif "hdf5" in name.lower():
        config = {
            "file_path": f"${REPO_ROOT}/benchmark/tmp/h5_store_{rid}.h5",
            "use_mpi": args.use_mpi_io
        }

    if args.use_mpi_io:
        try:
            from mpi4py import MPI
            if not MPI.Is_initialized():
                pass
        except ImportError:
            pass

    adapter = get_adapter(name, config)
    if not adapter.initialize():
        print_rank0(f" [{name}] Failed to initialize adapter")
        return

    try:

        print_rank0(f"[{name}] Clearing storage...")
        adapter.clear()
        file_barrier(f"{name}_cleared", rid)

        req_keys = loader.block_ids[:args.num_requests]
        my_write_reqs = req_keys[rank::world]

        print_rank0(f"[{name}] Write Phase: {len(my_write_reqs)} reqs/rank...")
        for rk in my_write_reqs:
            k_data, v_data = loader.load(rk)
            adapter.put(rk, k_data, v_data)

        adapter.flush()
        if hasattr(adapter, "sync_metadata"):
            print_rank0(f"[{name}] Synchronizing metadata...")
            adapter.sync_metadata()
        file_barrier(f"{name}_written", rid)

        if name.lower() == "cascade":
            time.sleep(10)
        elif "redis" not in name.lower():
            print_rank0(f"[{name}] Waiting for Lustre/FS stabilization...")
            time.sleep(20)
            store_type = name.lower().split('-')[0]
            force_lustre_sync(REPO_ROOT / "benchmark" / f"{store_type}_store")
            force_lustre_sync(REPO_ROOT / "benchmark" / "tmp")

        file_barrier(f"{name}_ready_to_read", rid)

        if args.tier_mode == "hot":
            target_rank = rank
        else:
            target_rank = (rank + 1) % world
        my_read_reqs = req_keys[target_rank::world]

        if args.tier_mode == "cold":
            if hasattr(adapter, "evict_all"):
                adapter.evict_all()
            elif hasattr(adapter, "flush_cache"):
                adapter.flush_cache()
            time.sleep(5)
        elif args.tier_mode == "warm":
            if hasattr(adapter, "evict_gpu"):
                adapter.evict_gpu()

        if "redis" in name.lower():
            target_host = all_hosts.get(target_rank, "127.0.0.1")
            adapter.close()
            adapter.host = target_host
            adapter.initialize()

        print(f"[Rank {rank}] Read Phase: fetching {len(my_read_reqs)} blocks from rank {target_rank}...", flush=True)

        read_latencies = []
        f0 = time.time()
        synced_once = False
        for rk in my_read_reqs:
            t_req = time.time()
            res = None
            max_attempts = 3 if name.lower() == "cascade" else 30
            for attempt in range(max_attempts):
                res = adapter.get(rk)
                if res: break
                if name.lower() == "cascade" and not synced_once:

                    if hasattr(adapter, "sync_metadata"):
                        adapter.sync_metadata()
                    synced_once = True
                elif "redis" not in name.lower() and name.lower() != "cascade":
                    store_type = name.lower().split('-')[0]
                    force_lustre_sync(REPO_ROOT / "benchmark" / f"{store_type}_store" / f"{rk}.kv")
                time.sleep(0.5)

            if res:
                read_latencies.append((time.time() - t_req) * 1000)
            else:
                print(f"[Rank {rank}] {name}: Missed key {rk} from rank {target_rank}")

        if read_latencies:
            local_avg_ttft = np.mean(read_latencies)
            local_duration = time.time() - f0
            local_throughput = len(read_latencies) / local_duration

            stats_dir = REPO_ROOT / "benchmark" / "tmp" / f"stats_{rid}"
            if rank == 0:
                if stats_dir.exists():
                    import shutil
                    shutil.rmtree(stats_dir)

            file_barrier(f"{name}_stats_clean", rid)
            stats_dir.mkdir(parents=True, exist_ok=True)

            stats_file = stats_dir / f"rank_{rank}.json"
            with open(stats_file, 'w') as f:
                json.dump({"ttft": local_avg_ttft, "thru": local_throughput, "total": len(read_latencies)}, f)

            file_barrier(f"{name}_stats_ready", rid)

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

                final_avg_ttft = np.mean(all_ttfts)
                final_agg_thru = np.sum(all_thrus)

                print(f"\n---  {name} Benchmark Results ({world} Nodes) ---")
                print(f"Avg TTFT (Per Req): {final_avg_ttft:.2f} ms")
                print(f"Aggregate Throughput: {final_agg_thru:.2f} req/s")
                print(f"Total Read Count: {total_blocks} blocks")
        else:
            print(f"[Rank {rank}]  [{name}] All read requests failed.")

    except Exception as e:
        print_rank0(f" [{name}] Crash: {e}")
    finally:
        adapter.close()

    file_barrier(f"{name}_done", rid)
    print_rank0(f"\n Run {rid} complete.")

if __name__ == "__main__":
    run_benchmark()
