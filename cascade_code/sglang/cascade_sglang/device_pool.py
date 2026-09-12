from __future__ import annotations

import torch


class DeviceChunkMover:

    def __init__(self, kv_pool, cascade_cpp, chunk_tokens: int, device: torch.device):
        self.cpp = cascade_cpp
        self.chunk_tokens = int(chunk_tokens)
        self.device = device

        layer_num = int(kv_pool.layer_num)
        getk = getattr(kv_pool, "_get_key_buffer", None) or kv_pool.get_key_buffer
        getv = getattr(kv_pool, "_get_value_buffer", None) or kv_pool.get_value_buffer
        k_list = [getk(i + kv_pool.start_layer) for i in range(layer_num)]
        v_list = [getv(i + kv_pool.start_layer) for i in range(layer_num)]
        self.pool_rows = int(k_list[0].shape[0])

        probe = k_list[0]
        self.layer_num = layer_num
        self.dtype = probe.dtype
        self.token_stride = int(probe[0].numel() * probe.element_size())

        self._k_list, self._v_list = k_list, v_list
        self.k_ptrs = torch.tensor([t.data_ptr() for t in k_list],
                                   dtype=torch.uint64, device=device)
        self.v_ptrs = torch.tensor([t.data_ptr() for t in v_list],
                                   dtype=torch.uint64, device=device)

        self.chunk_bytes = 2 * layer_num * self.chunk_tokens * self.token_stride

        self._staging: dict[int, torch.Tensor] = {}

        self.stream = torch.cuda.Stream(device=device)

        self._pending: list[torch.Tensor] = []

    def staging(self, count: int) -> torch.Tensor:
        buf = self._staging.get(count)
        if buf is None:
            buf = torch.empty(self.chunk_bytes * count, dtype=torch.uint8,
                              device=self.device)
            self._staging[count] = buf
        return buf

    def chunk_views(self, buf: torch.Tensor, count: int) -> list[torch.Tensor]:
        return [buf[i * self.chunk_bytes:(i + 1) * self.chunk_bytes] for i in range(count)]

    def check_slots(self, slots: torch.Tensor) -> None:
        if slots.numel() == 0:
            return
        lo = int(slots.min().item())
        hi = int(slots.max().item())
        if lo < 0 or hi >= self.pool_rows:
            raise IndexError(
                f"slot index out of range: [{lo}, {hi}] against a pool of "
                f"{self.pool_rows} rows"
            )

    def _keep(self, *tensors: torch.Tensor) -> None:
        for t in tensors:
            if t.is_cuda:
                t.record_stream(self.stream)
            self._pending.append(t)

    def gather(self, slots: torch.Tensor, out: torch.Tensor) -> None:
        slots = slots.contiguous()
        self.check_slots(slots)
        self._keep(slots, out)
        rc = self.cpp.gather_slots(
            int(self.k_ptrs.data_ptr()), int(self.v_ptrs.data_ptr()),
            int(slots.data_ptr()), int(out.data_ptr()),
            self.layer_num, int(slots.numel()), self.token_stride,
            int(self.stream.cuda_stream),
        )
        if rc != 0:
            raise RuntimeError(f"Cascade gather_slots failed with cuda error {rc}")

    def scatter(self, slots: torch.Tensor, src: torch.Tensor) -> None:
        slots = slots.contiguous()
        self.check_slots(slots)
        self._keep(slots, src)
        rc = self.cpp.scatter_slots(
            int(self.k_ptrs.data_ptr()), int(self.v_ptrs.data_ptr()),
            int(slots.data_ptr()), int(src.data_ptr()),
            self.layer_num, int(slots.numel()), self.token_stride,
            int(self.stream.cuda_stream),
        )
        if rc != 0:
            raise RuntimeError(f"Cascade scatter_slots failed with cuda error {rc}")

    def sync(self) -> None:
        self.stream.synchronize()
        self._pending.clear()
