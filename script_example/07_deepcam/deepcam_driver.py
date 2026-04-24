import sys
import os
import argparse
import time
import h5py
import numpy as np
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
deepcam_dir = os.environ.get("DEEPCAM_SRC", "./mlcommons_hpc/deepcam/src/deepCam")
sys.path.insert(0, deepcam_dir)

from utils import parser as deepcam_parser
from train import main as deepcam_train_main
import data.cam_hdf5_dataset as ds

def patch_dataset(use_cascade, adapter):
    original_getitem = ds.CamDataset.__getitem__

    def cascade_getitem(self, idx):
        filename = os.path.join(self.source, self.files[idx])
        if use_cascade and adapter is not None:
            basename = os.path.basename(filename)
            res = adapter.get(basename)
            if res is not None:
                buf = res[0] + res[1]
                data_size = 768 * 1152 * 16 * 4

                data = np.frombuffer(buf[:data_size], dtype=np.float32).reshape(768, 1152, 16)
                label = np.frombuffer(buf[data_size:], dtype=np.int64).reshape(768, 1152)

                data = data[..., self.channels]
                data = self.data_scale * (data - self.data_shift)

                if self.transpose:
                    data = np.transpose(data, (2, 0, 1))
                return data, label, filename

        return original_getitem(self, idx)

    ds.CamDataset.__getitem__ = cascade_getitem

def ingest_to_cascade(adapter, data_dir):
    import glob
    train_files = glob.glob(os.path.join(data_dir, "train", "*.h5"))
    val_files = glob.glob(os.path.join(data_dir, "validation", "*.h5"))
    all_files = train_files + val_files

    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    rank = int(os.environ.get('SLURM_PROCID', 0))

    if rank == 0:
        print(f"[Cascade] Starting Ingestion of {len(all_files)} files...")

    for i, fpath in enumerate(all_files):
        if i % world_size == rank:
            basename = os.path.basename(fpath)
            with h5py.File(fpath, "r") as f:
                data = f["climate/data"][...]
                label = f["climate/labels_0"][...]

            payload = data.tobytes() + label.tobytes()
            mid = len(payload) // 2
            adapter.put(basename, payload[:mid], payload[mid:])

    adapter.flush()
    adapter.barrier()
    if rank == 0:
        print(f"[Cascade] Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--use_cascade', action='store_true', help="Use Cascade cache")
    parser.add_argument('--cascade_lustre_path', type=str, default='')

    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    deepcam_pargs = deepcam_parser.parse_arguments()

    if args.use_cascade:
        sys.path.insert(0, os.path.join(repo_root, "benchmark"))
        from adapters.cascade_adapter import CascadeAdapter
        config = {
            "gpu_capacity_gb": 32.0,
            "shm_capacity_gb": 128.0,
            "use_gpu": True,
            "lustre_path": args.cascade_lustre_path
        }
        adapter = CascadeAdapter(config)
        adapter.initialize()

        ingest_to_cascade(adapter, deepcam_pargs.data_dir_prefix)
        patch_dataset(True, adapter)
    else:
        patch_dataset(False, None)

    deepcam_train_main(deepcam_pargs)
