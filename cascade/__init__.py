from cascade_cpp import CascadeStore, CascadeConfig, compute_block_id, GPUBackend, ShmBackend, LustreBackend

def create_login_node_store():
    """Create a CPU-only store for login node testing."""
    cfg = CascadeConfig()
    cfg.use_gpu = False
    cfg.shm_capacity_bytes = 1 * 1024 * 1024 * 1024  # 1GB
    cfg.shm_path = "/dev/shm/cascade_login"
    cfg.lustre_path = "/tmp/cascade_lustre_test"
    return CascadeStore(cfg)

def compute_block_id_from_bytes(data):
    """Wrapper for compute_block_id that takes bytes."""
    import numpy as np
    arr = np.frombuffer(data, dtype=np.uint8)
    return compute_block_id(arr)
