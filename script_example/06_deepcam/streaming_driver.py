print("[Proxy] Initializing DeepCAM Multi-System Proxy (STREAMING MODE)...")
import sys
import os
import argparse
import time
import h5py
import numpy as np
import torch
import concurrent.futures

_put_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

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
                    payload = res[0] + res[1] if isinstance(res, tuple) else res

            if payload is None or len(payload) <= DATA_SIZE:
                with h5py.File(filename, "r") as f:
                    disk_data = f["climate/data"][...]
                    disk_label = f["climate/labels_0"][...]

                _put_executor.submit(adapter.put, basename, disk_data.tobytes(), disk_label.tobytes())

                data = disk_data[..., self.channels]
                label = disk_label.astype(np.int64)
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
    parser.add_argument('--system', type=str, default='cascade')
    parser.add_argument('--cascade_lustre_path', type=str, default='')
    parser.add_argument('--no_aggregated_io', action='store_true')

    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)

    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    deepcam_pargs = deepcam_parser.parse_arguments()

    from adapters.cascade_adapter_ml import CascadeAdapter
    class MLProxyAdapter(CascadeAdapter):
        def initialize(self):
            import cascade_cpp
            cfg = cascade_cpp.DistributedConfig()
            cfg.gpu_capacity_per_device = 0
            cfg.dram_capacity = 32 * 1024**3
            cfg.num_gpus_per_node = 4
            cfg.dedup_enabled = False
            cfg.lustre_path = self.lustre_path
            self.store = cascade_cpp.DistributedStore(cfg)
            return True

    adapter = MLProxyAdapter({"lustre_path": args.cascade_lustre_path})
    adapter.initialize()
    patch_dataset('cascade', adapter)

    deepcam_train_main(deepcam_pargs)
