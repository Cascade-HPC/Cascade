
import os
import time
import argparse
import numpy as np
from pathlib import Path
import subprocess
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

def get_dir_size(path):
    try:
        output = subprocess.check_output(['du', '-sb', str(path)]).split()[0].decode('utf-8')
        return int(output)
    except:
        return 0

def get_file_size(path):
    try:
        return os.path.getsize(path)
    except:
        return 0

def cleanup_path(path):
    if not path: return
    p = Path(path)
    if not p.exists(): return
    if p.is_dir():
        subprocess.run(['rm', '-rf', str(p)])
    else:
        p.unlink(missing_ok=True)

class DedupDataGenerator:
    def __init__(self, block_size_bytes, num_total, sharing_rate):
        self.block_size = block_size_bytes
        self.num_total = num_total
        self.sharing_rate = sharing_rate
        self.num_unique = max(1, int(num_total * (1.0 - sharing_rate)))

        self.unique_blocks = []
        for i in range(self.num_unique):

            rng = np.random.default_rng(i + 12345)
            self.unique_blocks.append(rng.integers(0, 256, self.block_size, dtype=np.uint8).tobytes())

    def get_block(self, index):

        target_unique_idx = index % self.num_unique
        return self.unique_blocks[target_unique_idx]

def run_sensitivity():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--sharing-rate", type=float, default=0.5, help="0.0 to 1.0")
    parser.add_argument("--num-blocks", type=int, default=100, help="Total blocks per node")
    parser.add_argument("--block-size-mb", type=int, default=160)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    block_size_bytes = args.block_size_mb * 1024 * 1024
    total_blocks_per_rank = args.num_blocks
    total_blocks_cluster = total_blocks_per_rank * world

    num_unique_cluster = max(1, int(total_blocks_cluster * (1.0 - args.sharing_rate)))
    num_logical_bytes = total_blocks_cluster * block_size_bytes

    name = args.system
    rid = args.run_id or job_id

    store_path = None
    config = {}
    if name.lower() == "cascade":
        config = {"gpu_capacity_gb": 38.0, "shm_capacity_gb": 128.0, "use_gpu": True,
                  "lustre_path": f"${REPO_ROOT}/benchmark/tmp/cas_dedup_{rid}"}
        store_path = config["lustre_path"]
    elif name.lower() == "lmcache":
        store_path = f"${REPO_ROOT}/benchmark/tmp/lmcache_dedup_{rid}"
        config = {"storage_path": store_path}
    elif name.lower() == "pdc":
        store_path = f"${REPO_ROOT}/benchmark/tmp/pdc_dedup_{rid}"
        config = {"storage_path": store_path}
    elif "hdf5" in name.lower():
        store_path = f"${REPO_ROOT}/benchmark/tmp/hdf5_dedup_{rid}.h5"
        config = {"file_path": store_path, "use_mpi": True}

    adapter = get_adapter(name, config)
    if not adapter.initialize():
        print_rank0(f" [{name}] Failed")
        return

    try:
        adapter.clear()

        unique_pool = []
        for i in range(num_unique_cluster):

            pass

        print_rank0(f"\n---  Dedup Sensitivity: {name} (Rate: {args.sharing_rate}, Nodes: {world}) ---")
        print_rank0(f"Logical Data: {num_logical_bytes / 1024**3:.2f} GB ({total_blocks_cluster} blocks)")

        t0 = time.time()

        for i in range(total_blocks_per_rank):
            global_idx = rank * total_blocks_per_rank + i

            unique_idx = global_idx % num_unique_cluster

            rng = np.random.default_rng(unique_idx + 999)
            data = rng.integers(0, 256, block_size_bytes, dtype=np.uint8).tobytes()

            mid = len(data) // 2
            k_data = data[:mid]
            v_data = data[mid:]

            key = f"block_{global_idx}"
            adapter.put(key, k_data, v_data)

            if (i+1) % 10 == 0:
                print(f"[Rank {rank}] Written {i+1}/{total_blocks_per_rank} blocks", flush=True)

        adapter.flush()
        if hasattr(adapter, "sync_metadata"): adapter.sync_metadata()

        if world > 1:
            try:
                import mpi4py.MPI as MPI
                MPI.COMM_WORLD.Barrier()
            except:
                time.sleep(5)

        physical_bytes = 0
        dedup_saved_bytes = 0

        if name.lower() == "cascade":

            stats = adapter.get_stats()

        if rank == 0:
            print_rank0(f"---  Measuring Physical Occupancy (Path: {store_path}) ---")

            if name.lower() == "cascade":

                gpu_used = stats.get("gpu_used", 0)
                shm_used = stats.get("shm_used", 0)
                lustre_used = 0
                if store_path:
                    lustre_used = get_dir_size(store_path)

                physical_bytes = gpu_used + shm_used + lustre_used
                dedup_saved_bytes = max(0, num_logical_bytes - physical_bytes)

                print(f"Cascade Detail: GPU={gpu_used/1024**3:.2f}GB, SHM={shm_used/1024**3:.2f}GB, Lustre={lustre_used/1024**3:.2f}GB", flush=True)
            else:

                time.sleep(2)
                if store_path:
                    p = Path(store_path)
                    if p.exists():
                        if p.is_dir():
                            physical_bytes = get_dir_size(store_path)
                        else:
                            physical_bytes = get_file_size(store_path)
                    else:
                        print(f" Warning: store_path {store_path} does not exist for measurement!")

                        physical_bytes = num_logical_bytes

                dedup_saved_bytes = max(0, num_logical_bytes - physical_bytes)

            saving_ratio = (dedup_saved_bytes / num_logical_bytes) * 100 if num_logical_bytes > 0 else 0
            phys_logical_ratio = physical_bytes / num_logical_bytes if num_logical_bytes > 0 else 1.0

            print(f"\n[RESULTS] {name} | Sharing Rate: {args.sharing_rate}")
            print(f"Logical Bytes:  {num_logical_bytes / 1024**3:8.2f} GB")
            print(f"Physical Bytes: {physical_bytes / 1024**3:8.2f} GB")
            print(f"Dedup Savings:  {saving_ratio:6.1f} %")
            print(f"Phys/Logical:   {phys_logical_ratio:6.2f} x")

            res = {
                "system": name,
                "nodes": world,
                "sharing_rate": args.sharing_rate,
                "logical_gb": num_logical_bytes / 1024**3,
                "physical_gb": physical_bytes / 1024**3,
                "savings_pct": saving_ratio,
                "ratio": phys_logical_ratio
            }
            log_dir = REPO_ROOT / "benchmark" / "results" / "dedup_sens"
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / f"{name}_n{world}_r{args.sharing_rate}.json", 'w') as f:
                json.dump(res, f)

    finally:
        if args.cleanup:

            try: adapter.clear()
            except: pass
            adapter.close()

            if rank == 0:
                cleanup_path(store_path)
        else:
            adapter.close()

if __name__ == "__main__":
    run_sensitivity()
