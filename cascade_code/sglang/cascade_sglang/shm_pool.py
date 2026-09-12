from __future__ import annotations

import mmap
import os
import threading
from dataclasses import dataclass

SHM_DIR = "/dev/shm"
ENV_ENABLE = "CASCADE_SGLANG_SHM"


@dataclass(frozen=True)
class PoolGeometry:

    segment: str
    nbytes: int
    size: int
    page_size: int
    layer_num: int
    token_stride: int
    layout: str

    def as_dict(self) -> dict:
        return {
            "segment": self.segment,
            "nbytes": self.nbytes,
            "size": self.size,
            "page_size": self.page_size,
            "layer_num": self.layer_num,
            "token_stride": self.token_stride,
            "layout": self.layout,
        }

    @property
    def layout_dim(self) -> int:
        return self.token_stride * self.layer_num

    @property
    def v_base(self) -> int:
        return self.size * self.layout_dim

    def page_spans(self, host_index: int) -> list[tuple[int, int]]:
        if self.layout != "page_first":
            raise ValueError(
                f"Cascade needs the page_first host layout, got {self.layout}. "
                "Launch SGLang with --hicache-mem-layout page_first."
            )
        length = self.page_size * self.layout_dim
        k_off = host_index * self.layout_dim
        return [(k_off, length), (self.v_base + k_off, length)]

    def page_nbytes(self) -> int:
        return 2 * self.page_size * self.layout_dim


class _Segment:

    __slots__ = ("name", "fd", "mm", "memview")

    def __init__(self, name: str, nbytes: int, create: bool):
        path = os.path.join(SHM_DIR, name)
        if create:
            self.fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(self.fd, nbytes)
        else:
            self.fd = os.open(path, os.O_RDWR)
        self.name = name
        self.mm = mmap.mmap(self.fd, nbytes, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE)
        self.memview = memoryview(self.mm)

    def view(self, offset: int, length: int) -> memoryview:
        return self.memview[offset : offset + length]


class SegmentMapper:

    def __init__(self):
        self._segments: dict[str, _Segment] = {}
        self._lock = threading.Lock()

    def get(self, geometry: PoolGeometry) -> _Segment:
        seg = self._segments.get(geometry.segment)
        if seg is not None:
            return seg
        with self._lock:
            seg = self._segments.get(geometry.segment)
            if seg is None:
                seg = _Segment(geometry.segment, geometry.nbytes, create=False)
                self._segments[geometry.segment] = seg
        return seg

    def page_views(self, geometry: PoolGeometry, host_index: int) -> list[memoryview]:
        seg = self.get(geometry)
        return [seg.view(off, length) for off, length in geometry.page_spans(host_index)]


_ALLOCATED: list = []
_COUNTER = [0]


def _next_name() -> str:
    _COUNTER[0] += 1
    return f"cascade_sgl_{os.getpid()}_{_COUNTER[0]}"


def make_allocator(base_cls):

    class ShmHostTensorAllocator(base_cls):

        def __init__(self):
            super().__init__()
            self.segment = None
            self.nbytes = 0

        def allocate(self, dims: tuple, dtype, device: str):
            import torch

            self.dtype = dtype
            self.dims = dims
            if device != "cpu":
                return super().allocate(dims, dtype, device)

            numel = 1
            for d in dims:
                numel *= int(d)
            nbytes = numel * torch.tensor([], dtype=dtype).element_size()

            name = _next_name()
            seg = _Segment(name, nbytes, create=True)
            tensor = torch.frombuffer(seg.mm, dtype=dtype).view(*dims)

            self.segment = name
            self.nbytes = nbytes
            _ALLOCATED.append((seg, tensor))
            return tensor

    return ShmHostTensorAllocator


_installed = [False]


def install() -> bool:
    if _installed[0] or os.environ.get(ENV_ENABLE, "") not in ("1", "true", "yes"):
        return False
    try:
        from sglang.srt.mem_cache import memory_pool_host as mph
    except Exception:
        return False

    base = mph.HostTensorAllocator
    cls = make_allocator(base)
    original = mph.get_allocator_from_storage

    def get_allocator_from_storage(allocator_type):
        if allocator_type in ("dynamic", "cascade"):
            return cls()
        return original(allocator_type)

    mph.get_allocator_from_storage = get_allocator_from_storage
    _installed[0] = True
    return True


def current_geometry(mem_pool_host) -> PoolGeometry:
    allocator = getattr(mem_pool_host, "allocator", None)
    segment = getattr(allocator, "segment", None)
    if segment is None:
        raise RuntimeError(
            "The SGLang host KV pool was not allocated in shared memory. "
            f"Set {ENV_ENABLE}=1 so the Cascade allocator hook is installed."
        )
    return PoolGeometry(
        segment=segment,
        nbytes=int(allocator.nbytes),
        size=int(mem_pool_host.size),
        page_size=int(mem_pool_host.page_size),
        layer_num=int(mem_pool_host.layer_num),
        token_stride=int(mem_pool_host.token_stride_size),
        layout=str(mem_pool_host.layout),
    )
