print("[Proxy] Initializing DeepCAM Multi-System Proxy...")
import sys
import os
import argparse
import time
import h5py
import numpy as np
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "benchmark"))

deepcam_dir = os.environ.get("DEEPCAM_SRC", "./mlcommons_hpc/deepcam/src/deepCam")
sys.path.insert(0, deepcam_dir)

from utils import parser as deepcam_parser
from train import main as deepcam_train_main
import data.cam_hdf5_dataset as ds

def setup_distributed_env():
    if 'SLURM_PROCID' in os.environ:
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        print(f"[Proxy] Slurm detected: RANK={os.environ['RANK']}, WORLD_SIZE={os.environ['WORLD_SIZE']}")

setup_distributed_env()

def patch_dataset(system, adapter):
    original_getitem = ds.CamDataset.__getitem__

    def system_getitem(self, idx):
        filename = os.path.join(self.source, self.files[idx])
        if adapter is not None:
            basename = os.path.basename(filename)

            DATA_SIZE = 768 * 1152 * 16 * 4

            payload = None
            if hasattr(adapter, 'get_raw'):
                payload = adapter.get_raw(basename)
            else:
                res = adapter.get(basename)
                if res:
                    payload = np.concatenate([res[0], res[1]]) if isinstance(res, tuple) else res

            if payload is None or len(payload) <= DATA_SIZE:
                with h5py.File(filename, "r") as f:
                    disk_data = f["climate/data"][...]
                    disk_label = f["climate/labels_0"][...]

                adapter.put(basename, disk_data.tobytes(), disk_label.tobytes())

                data = disk_data[..., self.channels]
                label = disk_disk_label = disk_label.astype(np.int64)
                data = self.data_scale * (data - self.data_shift)
                if self.transpose:
                    data = np.transpose(data, (2, 0, 1))
                return data, label, filename

            if payload is not None and len(payload) > DATA_SIZE:
                data_payload = payload[:DATA_SIZE]
                label_payload = payload[DATA_SIZE:]

                data = np.frombuffer(data_payload, dtype=np.float32).reshape(768, 1152, 16)
                label = np.frombuffer(label_payload, dtype=np.int64).reshape(768, 1152)

                data = data[..., self.channels]
                data = self.data_scale * (data - self.data_shift)

                if self.transpose:
                    data = np.transpose(data, (2, 0, 1))
                return data, label, filename

        return original_getitem(self, idx)

    ds.CamDataset.__getitem__ = system_getitem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--system', type=str, default='hdf5',
                        choices=['hdf5', 'cascade', 'lmcache_disk', 'lmcache_redis', 'pdc', 'redis'])
    parser.add_argument('--use_cascade', action='store_true', help="Use Cascade cache")
    parser.add_argument('--cascade_lustre_path', type=str, default='')
    parser.add_argument('--aggregated_io', action='store_true', default=True)
    parser.add_argument('--no_aggregated_io', action='store_false', dest='aggregated_io', help="Disable Aggregated I/O")
    parser.add_argument('--storage_path', type=str, default='')
    parser.add_argument('--redis_host', type=str, default='localhost')
    parser.add_argument('--redis_port', type=int, default=6379)

    args, unknown = parser.parse_known_args()
    if args.use_cascade: args.system = 'cascade'

    sys.argv = [sys.argv[0]] + unknown
    deepcam_pargs = deepcam_parser.parse_arguments()

    adapter = None
    if args.system == 'cascade':
        from adapters.cascade_adapter_ml import CascadeAdapter
        class MLProxyAdapter(CascadeAdapter):
            def initialize(self):
                import cascade_cpp
                world_size = int(os.environ.get("SLURM_NTASKS", "1"))
                if world_size > 1:
                    cfg = cascade_cpp.DistributedConfig()
                    cfg.gpu_capacity_per_device = 0
                    cfg.dram_capacity = 50 * 1024**3
                    cfg.num_gpus_per_node = 4
                    cfg.kv_compression = False
                    cfg.locality_aware = True
                    cfg.prefix_replication = False
                    cfg.aggregated_lustre = False
                    cfg.lustre_path = self.lustre_path
                    self.store = cascade_cpp.DistributedStore(cfg)
                    return True
                return super().initialize()
        adapter = MLProxyAdapter({"lustre_path": args.cascade_lustre_path})
    elif args.system == 'lmcache_disk':
        from adapters.lmcache_adapter import LMCacheAdapter
        adapter = LMCacheAdapter({"storage_path": args.storage_path})
    elif args.system == 'lmcache_redis' or args.system == 'redis':
        from adapters.redis_adapter import RedisAdapter
        adapter = RedisAdapter({"host": args.redis_host, "port": args.redis_port})
    elif args.system == 'pdc':
        from adapters.pdc_adapter import PDCAdapter
        adapter = PDCAdapter({"storage_path": args.storage_path})

    if adapter:
        adapter.initialize()
        patch_dataset(args.system, adapter)
    else:
        patch_dataset('hdf5', None)

    deepcam_train_main(deepcam_pargs)
