#include "cascade_distributed.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <climits>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

namespace cascade {


static constexpr int kCentralRank = 0;


namespace distributed {


#ifdef USE_MPI
DistributedDRAMBackend::DistributedDRAMBackend(size_t capacity, MPI_Comm comm)
    : capacity_(capacity), comm_(comm) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &world_size_);


  const unsigned host_flags =
      std::getenv("CASCADE_HOST_ALLOC_DEFAULT")
          ? cudaHostAllocDefault
          : cudaHostAllocPortable;
  CUDA_CHECK(cudaHostAlloc(&dram_base_, capacity_, host_flags));
  memset(dram_base_, 0, capacity_);


  const char *no_window = std::getenv("CASCADE_NO_RMA_WINDOW");
  if (no_window && no_window[0] == '1') {
    window_ = MPI_WIN_NULL;
    if (rank_ == 0) {
      printf("[DRAM Backend] RMA window DISABLED, remote reads will fail; "
             "this is a bandwidth measurement mode only\n");
    }
  } else {
    MPI_Win_create(dram_base_, capacity_, 1, MPI_INFO_NULL, comm_, &window_);
    MPI_Win_lock_all(0, window_);
  }

  if (rank_ == 0) {
    printf("[DRAM Backend] Initialized %.2f GB per node, %d nodes total\n",
           capacity_ / (1024.0 * 1024.0 * 1024.0), world_size_);
  }
}
#else
DistributedDRAMBackend::DistributedDRAMBackend(size_t capacity)
    : capacity_(capacity) {
  CUDA_CHECK(cudaHostAlloc(&dram_base_, capacity_, cudaHostAllocDefault));
  memset(dram_base_, 0, capacity_);
}
#endif

DistributedDRAMBackend::~DistributedDRAMBackend() {
#ifdef USE_MPI
  MPI_Barrier(comm_);
  MPI_Win_unlock_all(window_);
  MPI_Win_free(&window_);
#endif
  if (dram_base_) {
    cudaFreeHost(dram_base_);
  }
}


size_t DistributedDRAMBackend::allocate(size_t size) {
  size_t aligned = (size + 63) & ~63ULL;


  {
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    auto best = free_list_.end();
    size_t best_waste = SIZE_MAX;

    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
      if (it->size >= aligned) {
        size_t waste = it->size - aligned;
        if (waste < best_waste) {
          best = it;
          best_waste = waste;
          if (waste == 0) break;
        }
      }
    }

    if (best != free_list_.end()) {
      size_t offset = best->offset;
      if (best->size == aligned) {
        free_list_.erase(best);
      } else {
        best->offset += aligned;
        best->size -= aligned;
      }
      return offset;
    }
  }


  size_t offset = write_offset_.fetch_add(aligned);
  if (offset + aligned > capacity_) {
    write_offset_.fetch_sub(aligned);
    return SIZE_MAX;
  }
  return offset;
}

void DistributedDRAMBackend::deallocate(size_t offset, size_t size) {
  size_t aligned = (size + 63) & ~63ULL;

  std::lock_guard<std::mutex> lock(free_list_mutex_);


  auto it = free_list_.begin();
  while (it != free_list_.end() && it->offset < offset) {
    ++it;
  }
  auto inserted = free_list_.insert(it, {offset, aligned});


  auto next = std::next(inserted);
  if (next != free_list_.end() &&
      inserted->offset + inserted->size == next->offset) {
    inserted->size += next->size;
    free_list_.erase(next);
  }


  if (inserted != free_list_.begin()) {
    auto prev = std::prev(inserted);
    if (prev->offset + prev->size == inserted->offset) {
      prev->size += inserted->size;
      free_list_.erase(inserted);
    }
  }
}

bool DistributedDRAMBackend::put_local(const BlockId &id, const uint8_t *data,
                                       size_t size, bool is_prefix) {
  size_t offset = allocate(size);
  if (offset == SIZE_MAX) {
    return false;
  }

  memcpy(static_cast<uint8_t *>(dram_base_) + offset, data, size);

  DRAMBlock block{offset, size, is_prefix};
  index_.put(id, block, size);
  used_.fetch_add(size);


  {
    std::lock_guard<std::mutex> lock(lru_mutex_);
    auto it = lru_map_.find(id);
    if (it != lru_map_.end()) {
      lru_list_.erase(it->second);
    }
    lru_list_.push_front(id);
    lru_map_[id] = lru_list_.begin();
  }

  return true;
}

bool DistributedDRAMBackend::get_local(const BlockId &id, uint8_t *out,
                                       size_t *out_size) {
  auto block = index_.get(id);
  if (!block)
    return false;

  memcpy(out, static_cast<uint8_t *>(dram_base_) + block->offset, block->size);
  *out_size = block->size;


  {
    std::lock_guard<std::mutex> lock(lru_mutex_);
    auto it = lru_map_.find(id);
    if (it != lru_map_.end()) {
      lru_list_.erase(it->second);
      lru_list_.push_front(id);
      lru_map_[id] = lru_list_.begin();
    }
  }

  return true;
}

uint8_t *DistributedDRAMBackend::stage_alloc(size_t size, size_t *out_offset) {
  size_t offset = allocate(size);
  if (offset == SIZE_MAX) return nullptr;
  if (out_offset) *out_offset = offset;
  return static_cast<uint8_t *>(dram_base_) + offset;
}

void DistributedDRAMBackend::stage_commit(const BlockId &id, size_t offset,
                                          size_t size, bool is_prefix) {


  auto previous = index_.get(id);
  if (previous && previous->offset != offset) {
    deallocate(previous->offset, previous->size);
    used_.fetch_sub(previous->size);
  }
  DRAMBlock block{offset, size, is_prefix};
  index_.put(id, block, size);
  used_.fetch_add(size);
  std::lock_guard<std::mutex> lock(lru_mutex_);
  auto it = lru_map_.find(id);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
  }
  lru_list_.push_front(id);
  lru_map_[id] = lru_list_.begin();
}

void DistributedDRAMBackend::stage_release(size_t offset, size_t size) {
  deallocate(offset, size);
}

const uint8_t *DistributedDRAMBackend::local_ptr(const BlockId &id,
                                                 size_t *out_size) const {
  auto block = index_.get(id);
  if (!block) return nullptr;
  if (out_size) *out_size = block->size;
  return static_cast<const uint8_t *>(dram_base_) + block->offset;
}

bool DistributedDRAMBackend::get_remote(int target_rank, size_t offset,
                                        uint8_t *out, size_t size) {
#ifdef USE_MPI
  MPI_Get(out, size, MPI_BYTE, target_rank, offset, size, MPI_BYTE, window_);
  MPI_Win_flush(target_rank, window_);
  return true;
#else
  return false;
#endif
}

void DistributedDRAMBackend::init_central_allocator() {
#ifdef USE_MPI
  if (alloc_win_ != MPI_WIN_NULL) return;
  MPI_Aint bytes = sizeof(MPI_Aint);
  MPI_Win_allocate(bytes, sizeof(MPI_Aint), MPI_INFO_NULL, comm_,
                   &alloc_counter_, &alloc_win_);
  *alloc_counter_ = 0;
  MPI_Win_lock_all(0, alloc_win_);
  MPI_Barrier(comm_);
#endif
}

long long DistributedDRAMBackend::reserve_on_central(size_t size) {
#ifdef USE_MPI
  if (alloc_win_ == MPI_WIN_NULL) return -1;

  const MPI_Aint step = static_cast<MPI_Aint>((size + 4095) & ~static_cast<size_t>(4095));
  MPI_Aint offset = 0;
  if (MPI_Fetch_and_op(&step, &offset, MPI_AINT, 0, 0, MPI_SUM, alloc_win_)
      != MPI_SUCCESS) {
    return -1;
  }
  MPI_Win_flush(0, alloc_win_);
  if (static_cast<size_t>(offset) + size > capacity_) {


    return -1;
  }
  return static_cast<long long>(offset);
#else
  (void)size;
  return -1;
#endif
}

bool DistributedDRAMBackend::put_remote(int target_rank, size_t offset,
                                        const uint8_t *data, size_t size) {
#ifdef USE_MPI
  if (MPI_Put(const_cast<uint8_t *>(data), static_cast<int>(size), MPI_BYTE,
              target_rank, static_cast<MPI_Aint>(offset),
              static_cast<int>(size), MPI_BYTE, window_) != MPI_SUCCESS) {
    return false;
  }
  return MPI_Win_flush(target_rank, window_) == MPI_SUCCESS;
#else
  (void)target_rank; (void)offset; (void)data; (void)size;
  return false;
#endif
}

long long DistributedDRAMBackend::reserve_local(const BlockId &id, size_t size,
                                                bool is_prefix) {


  size_t offset = allocate(size);
  if (offset == SIZE_MAX) return -1;

  DRAMBlock block{offset, size, is_prefix};
  index_.put(id, block, size);
  used_.fetch_add(size);

  {
    std::lock_guard<std::mutex> lock(lru_mutex_);
    auto it = lru_map_.find(id);
    if (it != lru_map_.end()) {
      lru_list_.erase(it->second);
    }
    lru_list_.push_front(id);
    lru_map_[id] = lru_list_.begin();
  }
  return static_cast<long long>(offset);
}

bool DistributedDRAMBackend::device_rma_supported() {
#ifdef USE_MPI
  int cached = device_rma_state_.load();
  if (cached >= 0) return cached == 1;
  std::lock_guard<std::mutex> lock(device_rma_mutex_);
  cached = device_rma_state_.load();
  if (cached >= 0) return cached == 1;

  int result = 0;


  const char *env = std::getenv("CASCADE_DIRECT_GPU_RMA");
  const std::string mode = env ? env : "0";
  if (mode == "0") {
    device_rma_state_.store(0);
    return false;
  }
  if (mode == "1") {
    device_rma_state_.store(1);
    return true;
  }

  void *probe = nullptr;
  if (cudaMalloc(&probe, 64) == cudaSuccess && dram_base_) {
    const int target = world_size_ > 1 ? (rank_ + 1) % world_size_ : rank_;
    if (MPI_Get(probe, 1, MPI_BYTE, target, 0, 1, MPI_BYTE, window_) ==
            MPI_SUCCESS &&
        MPI_Win_flush(target, window_) == MPI_SUCCESS) {
      result = 1;
    }
    cudaFree(probe);
  }
  if (rank_ == 0) {
    printf("[DRAM Backend] CUDA-aware MPI RMA origin: %s\n",
           result ? "supported, remote reads land in the vLLM KV page"
                  : "not supported, remote reads stage through the RMA window");
    fflush(stdout);
  }
  device_rma_state_.store(result);
  return result == 1;
#else
  return false;
#endif
}

void DistributedDRAMBackend::warm_remote() {
#ifdef USE_MPI
  if (world_size_ <= 1 || !dram_base_) return;


  uint8_t scratch = 0;
  void *device_scratch = nullptr;
  const bool device_ok = device_rma_supported();
  if (device_ok) cudaMalloc(&device_scratch, 64);
  for (int target = 0; target < world_size_; ++target) {
    if (target == rank_) continue;
    if (MPI_Get(&scratch, 1, MPI_BYTE, target, 0, 1, MPI_BYTE, window_) ==
        MPI_SUCCESS) {
      MPI_Win_flush(target, window_);
    }
    if (device_scratch &&
        MPI_Get(device_scratch, 1, MPI_BYTE, target, 0, 1, MPI_BYTE,
                window_) == MPI_SUCCESS) {
      MPI_Win_flush(target, window_);
    }
  }
  if (device_scratch) cudaFree(device_scratch);
#endif
}

namespace {


constexpr int kMaxEvictRounds = 8;

constexpr size_t kMaxRmaBytes = 1ULL << 30;

size_t cascade_rma_chunk_bytes() {
  static const size_t value = [] {
    const char *env = std::getenv("CASCADE_RMA_MAX_BYTES");
    size_t bytes = env ? strtoull(env, nullptr, 10) : kMaxRmaBytes;
    if (bytes == 0 || bytes > static_cast<size_t>(INT_MAX)) bytes = kMaxRmaBytes;
    return bytes;
  }();
  return value;
}

}

bool DistributedDRAMBackend::get_remote_batch(
    const std::vector<int> &target_ranks, const std::vector<size_t> &offsets,
    const std::vector<uint8_t *> &outs, const std::vector<size_t> &sizes) {
  if (target_ranks.size() != offsets.size() ||
      target_ranks.size() != outs.size() ||
      target_ranks.size() != sizes.size()) {
    return false;
  }
#ifdef USE_MPI


  std::unordered_set<int> targets;
  std::unordered_map<int, std::vector<size_t>> by_target;
  for (size_t i = 0; i < target_ranks.size(); ++i) {
    if (sizes[i] != 0) by_target[target_ranks[i]].push_back(i);
  }
  for (const auto &[target, indices] : by_target) {
    size_t min_offset = std::numeric_limits<size_t>::max();
    size_t max_end = 0;
    size_t requested_bytes = 0;
    for (size_t i : indices) {
      min_offset = std::min(min_offset, offsets[i]);
      max_end = std::max(max_end, offsets[i] + sizes[i]);
      requested_bytes += sizes[i];
    }
    size_t span_bytes = max_end - min_offset;


    static const bool span_enabled = [] {
      const char *env = std::getenv("CASCADE_RMA_SPAN");
      return env && std::string(env) == "1";
    }();
    const bool compact = span_enabled && indices.size() > 1 &&
                         span_bytes <= requested_bytes * 8;
    const char *rma_debug_env = std::getenv("CASCADE_RMA_DEBUG");
    if (rma_debug_env && std::string(rma_debug_env) == "1") {
      printf("CASCADE RMA target=%d blocks=%zu requested=%zu span=%zu compact=%d\n",
             target, indices.size(), requested_bytes, span_bytes,
             compact ? 1 : 0);
    }
    const size_t chunk_bytes = cascade_rma_chunk_bytes();
    auto issue = [&](uint8_t *dst, size_t remote_offset, size_t bytes) -> bool {
      size_t done = 0;
      while (done < bytes) {
        const size_t piece = std::min(chunk_bytes, bytes - done);
        if (MPI_Get(dst + done, static_cast<int>(piece), MPI_BYTE, target,
                    remote_offset + done, static_cast<int>(piece), MPI_BYTE,
                    window_) != MPI_SUCCESS) {
          return false;
        }
        done += piece;
      }
      return true;
    };

    if (compact) {
      std::vector<uint8_t> span(span_bytes);
      if (!issue(span.data(), min_offset, span_bytes)) return false;
      targets.insert(target);
      for (size_t i : indices) {
        memcpy(outs[i], span.data() + (offsets[i] - min_offset), sizes[i]);
      }
    } else {
      for (size_t i : indices) {
        if (!issue(outs[i], offsets[i], sizes[i])) return false;
      }
      targets.insert(target);
    }
  }
  for (int target : targets) {
    MPI_Win_flush(target, window_);
  }
  return true;
#else
  return false;
#endif
}

bool DistributedDRAMBackend::get_remote_device_batch(
    const std::vector<int>& target_ranks,
    const std::vector<size_t>& offsets,
    const std::vector<void*>& outs,
    const std::vector<size_t>& sizes) {
  if (target_ranks.size() != offsets.size() ||
      target_ranks.size() != outs.size() ||
      target_ranks.size() != sizes.size()) {
    return false;
  }
#ifdef USE_MPI


  std::unordered_set<int> targets;
  for (size_t i = 0; i < target_ranks.size(); ++i) {
    if (sizes[i] == 0) continue;
    const size_t chunk_bytes = cascade_rma_chunk_bytes();
    size_t done = 0;
    while (done < sizes[i]) {
      const size_t piece = std::min(chunk_bytes, sizes[i] - done);
      if (MPI_Get(static_cast<uint8_t *>(outs[i]) + done,
                  static_cast<int>(piece), MPI_BYTE, target_ranks[i],
                  offsets[i] + done, static_cast<int>(piece), MPI_BYTE,
                  window_) != MPI_SUCCESS) {
        return false;
      }
      done += piece;
    }
    targets.insert(target_ranks[i]);
  }
  for (int target : targets) {
    if (MPI_Win_flush(target, window_) != MPI_SUCCESS) return false;
  }
  return true;
#else
  return false;
#endif
}

bool DistributedDRAMBackend::remove_local(const BlockId &id) {
  auto block = index_.get(id);
  if (!block) return false;

  deallocate(block->offset, block->size);
  used_.fetch_sub(block->size);
  index_.remove(id);

  {
    std::lock_guard<std::mutex> lock(lru_mutex_);
    auto it = lru_map_.find(id);
    if (it != lru_map_.end()) {
      lru_list_.erase(it->second);
      lru_map_.erase(it);
    }
  }
  return true;
}

std::vector<std::pair<BlockId, std::vector<uint8_t>>>
DistributedDRAMBackend::evict_for_space(size_t needed, bool protect_prefix) {
  std::vector<std::pair<BlockId, std::vector<uint8_t>>> evicted;
  size_t freed = 0;

  std::lock_guard<std::mutex> lock(lru_mutex_);


  auto scan = [&](bool skip_prefix) {
    auto it = lru_list_.end();
    while (it != lru_list_.begin() && freed < needed) {
      --it;
      BlockId id = *it;
      auto block = index_.get(id);
      if (!block) {
        continue;
      }
      if (skip_prefix && block->is_prefix) {
        continue;
      }


      if (is_pinned(id)) {
        continue;
      }


      std::vector<uint8_t> data(block->size);
      memcpy(data.data(),
             static_cast<uint8_t *>(dram_base_) + block->offset,
             block->size);
      evicted.push_back({id, std::move(data)});
      freed += block->size;


      deallocate(block->offset, block->size);
      used_.fetch_sub(block->size);
      index_.remove(id);


      lru_map_.erase(id);
      it = lru_list_.erase(it);
    }
  };

  scan(protect_prefix);
  if (protect_prefix && freed < needed) {
    const size_t cold_only = evicted.size();
    scan(false);
    prefix_evictions_ += evicted.size() - cold_only;
  }

  return evicted;
}


void DistributedDRAMBackend::pin_batch(const std::vector<BlockId> &ids,
                                       double seconds) {
  const auto deadline = std::chrono::steady_clock::now()
      + std::chrono::milliseconds(static_cast<long long>(seconds * 1000.0));
  std::lock_guard<std::mutex> lock(pin_mutex_);
  for (const auto &id : ids) {
    auto &entry = pinned_blocks_[id];
    ++entry.count;

    if (entry.deadline < deadline) entry.deadline = deadline;
  }
}

void DistributedDRAMBackend::unpin_batch(const std::vector<BlockId> &ids) {
  std::lock_guard<std::mutex> lock(pin_mutex_);
  for (const auto &id : ids) {
    auto it = pinned_blocks_.find(id);
    if (it == pinned_blocks_.end()) continue;
    if (--it->second.count <= 0) pinned_blocks_.erase(it);
  }
}

bool DistributedDRAMBackend::is_pinned(const BlockId &id) const {
  std::lock_guard<std::mutex> lock(pin_mutex_);
  auto it = pinned_blocks_.find(id);
  if (it == pinned_blocks_.end()) return false;


  if (std::chrono::steady_clock::now() > it->second.deadline) return false;
  return it->second.count > 0;
}

size_t DistributedDRAMBackend::pinned_count() const {
  std::lock_guard<std::mutex> lock(pin_mutex_);
  return pinned_blocks_.size();
}

void DistributedDRAMBackend::barrier() {
#ifdef USE_MPI
  MPI_Barrier(comm_);
#endif
}

size_t DistributedDRAMBackend::get_offset(const BlockId &id) const {
  auto block = index_.get(id);
  if (!block)
    return 0;
  return block->offset;
}


#ifdef USE_MPI
DistributedGPUBackend::DistributedGPUBackend(size_t cap_per_gpu, int num_gpus,
                                             MPI_Comm comm, int device_offset)
    : cap_per_gpu_(cap_per_gpu), num_gpus_(num_gpus),
      device_offset_(device_offset), comm_(comm) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &world_size_);
#else
DistributedGPUBackend::DistributedGPUBackend(size_t cap_per_gpu, int num_gpus,
                                             int device_offset)
    : cap_per_gpu_(cap_per_gpu), num_gpus_(num_gpus),
      device_offset_(device_offset) {
#endif


  for (int i = 0; i < num_gpus_; i++) {
    int device = device_offset_ + i;
    CUDA_CHECK(cudaSetDevice(device));
    gpus_.push_back(std::make_unique<GPUBackend>(cap_per_gpu_, device));
  }


  setup_nvlink();


  CUDA_CHECK(cudaSetDevice(device_offset_));
  for (int i = 0; i < 32; i++) {
    CUDA_CHECK(cudaHostAlloc(&pinned_[i], staging_size_, cudaHostAllocDefault));
  }


  init_window();

  if (rank_ == 0) {
    printf(
        "[GPU Backend] %d GPUs/node, %.2f GB/GPU, %d nodes = %.2f TB total GPU\n",
        num_gpus_, cap_per_gpu_ / (1024.0 * 1024.0 * 1024.0), world_size_,
        (num_gpus_ * cap_per_gpu_ * world_size_) /
            (1024.0 * 1024.0 * 1024.0 * 1024.0));
  }
}

DistributedGPUBackend::~DistributedGPUBackend() {
#ifdef USE_MPI
  MPI_Barrier(comm_);
  if (window_ != MPI_WIN_NULL) {
    MPI_Win_unlock_all(window_);
    MPI_Win_free(&window_);
  }
#endif
  for (int i = 0; i < 32; i++) {
    if (pinned_[i]) {
      cudaFreeHost(pinned_[i]);
    }
  }
}

void DistributedGPUBackend::setup_nvlink() {
  for (int i = 0; i < num_gpus_; i++) {
    for (int j = 0; j < num_gpus_; j++) {
      if (i != j) {
        int can_access;
        int src_device = device_offset_ + i;
        int dst_device = device_offset_ + j;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, src_device, dst_device));
        peer_[i][j] = (can_access != 0);
        if (can_access) {
          CUDA_CHECK(cudaSetDevice(src_device));
          cudaDeviceEnablePeerAccess(dst_device, 0);
        }
      }
    }
  }

  if (rank_ == 0) {
    printf("[NVLink] Peer access matrix:\n");
    for (int i = 0; i < num_gpus_; i++) {
      printf("  GPU %d: ", i);
      for (int j = 0; j < num_gpus_; j++) {
        printf("%c ", peer_[i][j] ? 'Y' : 'N');
      }
      printf("\n");
    }
  }
}

void DistributedGPUBackend::init_window() {
#ifdef USE_MPI
  MPI_Win_create(pinned_[0], staging_size_, 1, MPI_INFO_NULL, comm_, &window_);
  MPI_Win_lock_all(0, window_);
#endif
}

int DistributedGPUBackend::get_target_node(const BlockId &id) const {
  return std::hash<std::string>{}(id) % world_size_;
}

int DistributedGPUBackend::get_target_gpu(const BlockId &id) const {
  return (std::hash<std::string>{}(id) / world_size_) % num_gpus_;
}

bool DistributedGPUBackend::put(const BlockId &id, const uint8_t *data,
                                size_t size, bool is_prefix) {


  int target_gpu = get_target_gpu(id);
  return put_local(id, data, size, target_gpu, is_prefix);
}

bool DistributedGPUBackend::put_device(const BlockId &id, const void *data,
                                       size_t size, bool is_prefix) {


  int target_gpu = get_target_gpu(id);
  if (target_gpu < 0 || target_gpu >= num_gpus_) return false;

  CUDA_CHECK(cudaSetDevice(device_offset_ + target_gpu));
  bool ok = gpus_[target_gpu]->put_device(id, data, size);
  if (ok) {
    BlockLocation loc;
    loc.node_id = rank_;
    loc.gpu_id = device_offset_ + target_gpu;
    loc.offset = gpus_[target_gpu]->get_offset(id);
    loc.size = size;
    loc.dram_offset = 0;
    loc.dram_size = 0;
    loc.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    loc.is_gpu = true;
    loc.is_prefix = is_prefix;
    loc.has_dram_shadow = false;
    global_index_.put(id, loc, size);
  }
  return ok;
}

size_t DistributedGPUBackend::put_device_batch(
    const std::vector<BlockId> &ids, const std::vector<const void *> &data,
    const std::vector<size_t> &sizes,
    const std::vector<bool> &is_prefix_flags) {
  if (ids.size() != data.size() || ids.size() != sizes.size()) return 0;


  static const bool affinity = [] {
    const char *env = std::getenv("CASCADE_GPU_AFFINITY");
    return !env || std::string(env) == "1";
  }();


  int affinity_gpu = -1;
  if (affinity && !data.empty()) {
    cudaPointerAttributes attrs{};
    if (cudaPointerGetAttributes(&attrs, data[0]) == cudaSuccess &&
        attrs.type == cudaMemoryTypeDevice) {
      affinity_gpu = attrs.device - device_offset_;
    }
    if (affinity_gpu < 0 || affinity_gpu >= num_gpus_) affinity_gpu = -1;
  }

  std::vector<std::vector<size_t>> by_gpu(num_gpus_);
  for (size_t i = 0; i < ids.size(); ++i) {
    int target_gpu = affinity_gpu >= 0 ? affinity_gpu : get_target_gpu(ids[i]);
    if (target_gpu >= 0 && target_gpu < num_gpus_)
      by_gpu[target_gpu].push_back(i);
  }

  size_t success = 0;
  for (int gpu = 0; gpu < num_gpus_; ++gpu) {
    if (by_gpu[gpu].empty()) continue;
    std::vector<BlockId> gpu_ids;
    std::vector<const void *> gpu_data;
    std::vector<size_t> gpu_sizes;
    gpu_ids.reserve(by_gpu[gpu].size());
    gpu_data.reserve(by_gpu[gpu].size());
    gpu_sizes.reserve(by_gpu[gpu].size());
    std::vector<bool> was_present;
    was_present.reserve(by_gpu[gpu].size());
    for (size_t i : by_gpu[gpu]) {
      gpu_ids.push_back(ids[i]);
      gpu_data.push_back(data[i]);
      gpu_sizes.push_back(sizes[i]);
      was_present.push_back(gpus_[gpu]->contains(ids[i]));
    }

    CUDA_CHECK(cudaSetDevice(device_offset_ + gpu));
    gpus_[gpu]->put_device_batch(gpu_ids, gpu_data, gpu_sizes);
    for (size_t j = 0; j < gpu_ids.size(); ++j) {
      if (was_present[j] || !gpus_[gpu]->contains(gpu_ids[j])) continue;
      const size_t original = by_gpu[gpu][j];
      BlockLocation loc;
      loc.node_id = rank_;
      loc.gpu_id = device_offset_ + gpu;
      loc.offset = gpus_[gpu]->get_offset(gpu_ids[j]);
      loc.size = gpu_sizes[j];
      loc.dram_offset = 0;
      loc.dram_size = 0;
      loc.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
      loc.is_gpu = true;
      loc.is_prefix = original < is_prefix_flags.size() && is_prefix_flags[original];
      loc.has_dram_shadow = false;
      global_index_.put(gpu_ids[j], loc, gpu_sizes[j]);
      ++success;
    }
  }
  return success;
}

bool DistributedGPUBackend::put_local(const BlockId &id, const uint8_t *data,
                                      size_t size, int gpu, bool is_prefix) {
  if (gpu < 0 || gpu >= num_gpus_)
    return false;

  CUDA_CHECK(cudaSetDevice(device_offset_ + gpu));
  bool ok = gpus_[gpu]->put(id, data, size);

  if (ok) {
    BlockLocation loc;
    loc.node_id = rank_;
    loc.gpu_id = device_offset_ + gpu;
    loc.offset = gpus_[gpu]->get_offset(id);
    loc.size = size;
    loc.dram_offset = 0;
    loc.dram_size = 0;
    loc.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    loc.is_gpu = true;
    loc.is_prefix = is_prefix;
    loc.has_dram_shadow = false;
    global_index_.put(id, loc, size);
  }

  return ok;
}

bool DistributedGPUBackend::get(const BlockId &id, uint8_t *out,
                                size_t *out_size) {
  if (get_local(id, out, out_size)) {
    return true;
  }

  auto loc = global_index_.get(id);
  if (!loc)
    return false;

  if (loc->node_id == rank_) {
    return false;
  }

  return get_remote(loc->node_id, loc->offset, loc->size, out);
}

bool DistributedGPUBackend::get_local(const BlockId &id, uint8_t *out,
                                      size_t *out_size) {
  for (int i = 0; i < num_gpus_; i++) {
    CUDA_CHECK(cudaSetDevice(device_offset_ + i));
    if (gpus_[i]->get(id, out, out_size)) {
      return true;
    }
  }
  return false;
}

bool DistributedGPUBackend::get_device(const BlockId &id, void *out,
                                       size_t out_capacity, size_t *out_size) {


  auto loc = global_index_.get(id);
  if (!loc || loc->node_id != rank_ || !loc->is_gpu) return false;
  int gpu = loc->gpu_id - device_offset_;
  if (gpu < 0 || gpu >= num_gpus_) return false;
  CUDA_CHECK(cudaSetDevice(device_offset_ + gpu));
  size_t size = 0;
  if (!gpus_[gpu]->get_device(id, out, out_capacity, &size)) return false;
  *out_size = size;
  return true;
}

bool DistributedGPUBackend::get_device_async(const BlockId &id, void *out,
                                            size_t out_capacity,
                                            size_t *out_size, void *stream) {
  auto loc = global_index_.get(id);
  if (!loc || loc->node_id != rank_ || !loc->is_gpu) return false;
  int gpu = loc->gpu_id - device_offset_;
  if (gpu < 0 || gpu >= num_gpus_) return false;


  size_t size = 0;
  if (!gpus_[gpu]->get_device_async(id, out, out_capacity, &size, stream))
    return false;
  *out_size = size;
  return true;
}

bool DistributedGPUBackend::get_remote(int target_rank, size_t offset,
                                       size_t size, uint8_t *out) {
#ifdef USE_MPI
  MPI_Get(pinned_[1], size, MPI_BYTE, target_rank, offset, size, MPI_BYTE,
          window_);
  MPI_Win_flush(target_rank, window_);

  CUDA_CHECK(cudaMemcpy(out, pinned_[1], size, cudaMemcpyDefault));
  return true;
#else
  return false;
#endif
}

std::optional<BlockLocation>
DistributedGPUBackend::locate(const BlockId &id) const {
  return global_index_.get(id);
}

size_t DistributedGPUBackend::used_bytes() const {
  size_t total = 0;
  for (const auto &gpu : gpus_) {
    total += gpu->used_bytes();
  }
  return total;
}

void DistributedGPUBackend::sync_all() {
  for (int i = 0; i < num_gpus_; i++) {
    CUDA_CHECK(cudaSetDevice(device_offset_ + i));
    gpus_[i]->sync_all();
  }
}

void DistributedGPUBackend::barrier() {
#ifdef USE_MPI
  MPI_Barrier(comm_);
#endif
}

std::vector<GPUBackend::EvictedBlock> DistributedGPUBackend::evict_gpu_for_space(
    int gpu_id, size_t needed_bytes,
    const std::function<bool(const BlockId&)>& is_prefix) {
  if (gpu_id < 0 || gpu_id >= num_gpus_) {
    return {};
  }
  CUDA_CHECK(cudaSetDevice(device_offset_ + gpu_id));
  return gpus_[gpu_id]->evict_for_space(needed_bytes, is_prefix);
}


#ifdef USE_MPI
DistributedStore::DistributedStore(const DistributedConfig &cfg, MPI_Comm comm)
    : cfg_(cfg), comm_(comm) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }

  if (comm_ == MPI_COMM_WORLD) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &world_size_);
  } else {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &world_size_);
  }
#else
DistributedStore::DistributedStore(const DistributedConfig &cfg)
    : cfg_(cfg) {
  rank_ = 0;
  world_size_ = 1;
#endif

#ifdef USE_MPI
  gpu_ = std::make_unique<DistributedGPUBackend>(cfg_.gpu_capacity_per_device,
                                                 cfg_.num_gpus_per_node, comm_,
                                                 cfg_.gpu_device_offset);
  dram_ = std::make_unique<DistributedDRAMBackend>(cfg_.dram_capacity, comm_);
#else
  gpu_ = std::make_unique<DistributedGPUBackend>(cfg_.gpu_capacity_per_device,
                                                 cfg_.num_gpus_per_node,
                                                 cfg_.gpu_device_offset);
  dram_ = std::make_unique<DistributedDRAMBackend>(cfg_.dram_capacity);
#endif


#ifdef CASCADE_CENTRALIZED
  if (dram_) dram_->init_central_allocator();
#endif

  if (!cfg_.lustre_path.empty()) {
    if (cfg_.aggregated_lustre) {
      agg_lustre_ = std::make_unique<AggregatedLustreBackend>(
          cfg.lustre_path, cfg.agg_file_size);
    } else {
      lustre_ = std::make_unique<LustreBackend>(cfg.lustre_path);
    }
  }

  if (rank_ == 0) {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║   Cascade Distributed Store V6 Initialized  ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  Nodes: %d                                   \n", world_size_);
    printf("║  GPUs/node: %d                               \n", cfg_.num_gpus_per_node);
    printf("║  GPU capacity: %.2f GB/device                \n",
           cfg_.gpu_capacity_per_device / (1024.0 * 1024.0 * 1024.0));
    printf("║  DRAM capacity: %.2f GB/node                 \n",
           cfg_.dram_capacity / (1024.0 * 1024.0 * 1024.0));
    printf("║  Total cluster GPU: %.2f TB                  \n",
           (cfg_.gpu_capacity_per_device * cfg_.num_gpus_per_node * world_size_) /
               (1024.0 * 1024.0 * 1024.0 * 1024.0));
    printf("║  Total cluster DRAM: %.2f TB                 \n",
           (cfg_.dram_capacity * world_size_) /
               (1024.0 * 1024.0 * 1024.0 * 1024.0));
    printf("║  Features:                                   \n");
    printf("║    Semantic eviction: %s                     \n",
           cfg_.semantic_eviction ? "ON" : "OFF");
    printf("║    Distributed dedup: %s                     \n",
           cfg_.dedup_enabled ? "ON" : "OFF");
    printf("║    Locality-aware:    %s                     \n",
           cfg_.locality_aware ? "ON" : "OFF");
    printf("║    KV compression:    %s                     \n",
           cfg_.kv_compression ? "ON" : "OFF");
    printf("║    Lustre tier:       %s                     \n",
           cfg_.lustre_path.empty() ? "OFF" : "ON");
    printf("╚══════════════════════════════════════════════╝\n");
  }
}

DistributedStore::~DistributedStore() { barrier(); }


bool DistributedStore::put(const BlockId &id, const uint8_t *data,
                           size_t size, bool is_pf) {

  if (cfg_.dedup_enabled && global_dedup_.contains(id)) {
    dedup_hits_++;
    dedup_bytes_saved_ += size;
    return true;
  }


  std::vector<uint8_t> compressed_buf;
  const uint8_t *store_data = data;
  size_t store_size = size;

  if (cfg_.kv_compression && size >= 64) {
    CompressionMeta meta;
    compressed_buf = KVCompressor::compress(data, size, meta);
    store_data = compressed_buf.data();
    store_size = compressed_buf.size();
    compression_savings_ += (size - store_size);
  }

  bool stored = false;
  bool gpu_stored = false;


  if (gpu_) {
    gpu_stored = gpu_->put(id, data, size, is_pf);
    if (!gpu_stored) {

      if (evict_gpu_to_dram(size)) {
        gpu_stored = gpu_->put(id, data, size, is_pf);
      }
    }
    stored = gpu_stored;
  }


  bool dram_shadow_ok = false;
  if (dram_) {
#ifdef CASCADE_CENTRALIZED


    if (rank_ == kCentralRank) {
      dram_shadow_ok = dram_->put_local(id, store_data, store_size, is_pf);
      if (!dram_shadow_ok && evict_dram_to_lustre(store_size)) {
        dram_shadow_ok = dram_->put_local(id, store_data, store_size, is_pf);
      }
    } else {


      const long long offset = dram_->reserve_on_central(store_size);
      if (offset >= 0) {
        dram_shadow_ok = dram_->put_remote(kCentralRank,
                                           static_cast<size_t>(offset),
                                           store_data, store_size);
      }
    }
#else
    dram_shadow_ok = dram_->put_local(id, store_data, store_size, is_pf);
    if (!dram_shadow_ok) {

      if (evict_dram_to_lustre(store_size)) {
        dram_shadow_ok = dram_->put_local(id, store_data, store_size, is_pf);
      }
    }
#endif
    if (!stored) stored = dram_shadow_ok;
  }


  if (dram_shadow_ok && dram_ && gpu_) {
    auto existing = gpu_->global_index_.get(id);
    BlockLocation loc;
    if (existing) loc = *existing;
    loc.node_id = rank_;
    loc.dram_offset = dram_->get_offset(id);
    loc.dram_size = store_size;
    loc.has_dram_shadow = true;
    if (!gpu_stored) {
      loc.gpu_id = -1;
      loc.offset = loc.dram_offset;
      loc.size = store_size;
      loc.is_gpu = false;
    }
    gpu_->global_index_.put(id, loc, store_size);
    mark_dirty(id, loc);
  }


  const char *write_through_env = std::getenv("CASCADE_WRITE_THROUGH_LUSTRE");
  const bool write_through_lustre =
      write_through_env && std::string(write_through_env) == "1";
  if ((lustre_ || agg_lustre_) && (write_through_lustre || !stored)) {
    bool l_ok = lustre_put(id, store_data, store_size);
    if (!stored) stored = l_ok;
  }


  if (stored) {
    if (cfg_.dedup_enabled) {
      global_dedup_.put(id, true);
    }
    if (is_pf) {
      std::unique_lock lock(prefix_mutex_);
      prefix_registry_.insert(id);
    }

    if (!gpu_stored || !dram_shadow_ok) {
      auto loc_opt = gpu_ ? gpu_->global_index_.get(id) : std::nullopt;
      if (loc_opt) mark_dirty(id, *loc_opt);
    }
  }

  return stored;
}

bool DistributedStore::put_device(const BlockId &id, const void *data,
                                  size_t size, bool is_pf) {


  if (cfg_.dedup_enabled && global_dedup_.contains(id)) {
    dedup_hits_++;
    dedup_bytes_saved_ += size;
    return true;
  }

  std::vector<uint8_t> host(size);
  cudaError_t copy_err = cudaMemcpy(host.data(), data, size, cudaMemcpyDefault);
  if (copy_err != cudaSuccess) return false;

  std::vector<uint8_t> compressed_buf;
  const uint8_t *store_data = host.data();
  size_t store_size = size;
  if (cfg_.kv_compression && size >= 64) {
    CompressionMeta meta;
    compressed_buf = KVCompressor::compress(host.data(), size, meta);
    store_data = compressed_buf.data();
    store_size = compressed_buf.size();
    compression_savings_ += (size - store_size);
  }

  bool stored = false;
  bool gpu_stored = false;
  if (gpu_) {
    gpu_stored = gpu_->put_device(id, data, size, is_pf);
    for (int attempt = 0; !gpu_stored && attempt < kMaxEvictRounds; ++attempt) {
      if (!evict_gpu_to_dram(size)) break;
      gpu_stored = gpu_->put_device(id, data, size, is_pf);
    }
    stored = gpu_stored;
  }

  bool dram_shadow_ok = false;
  if (dram_) {
    dram_shadow_ok = dram_->put_local(id, store_data, store_size, is_pf);


    for (int attempt = 0; !dram_shadow_ok && attempt < kMaxEvictRounds;
         ++attempt) {
      if (!evict_dram_to_lustre(store_size)) break;
      dram_shadow_ok = dram_->put_local(id, store_data, store_size, is_pf);
    }
    if (!stored) stored = dram_shadow_ok;
  }

  if (dram_shadow_ok && dram_ && gpu_) {
    auto existing = gpu_->global_index_.get(id);
    BlockLocation loc;
    if (existing) loc = *existing;
    loc.node_id = rank_;
    loc.dram_offset = dram_->get_offset(id);
    loc.dram_size = store_size;
    loc.has_dram_shadow = true;
    if (!gpu_stored) {
      loc.gpu_id = -1;
      loc.offset = loc.dram_offset;
      loc.size = store_size;
      loc.is_gpu = false;
    }
    gpu_->global_index_.put(id, loc, store_size);
    mark_dirty(id, loc);
  }

  const char *write_through_env = std::getenv("CASCADE_WRITE_THROUGH_LUSTRE");
  const bool write_through_lustre =
      write_through_env && std::string(write_through_env) == "1";
  if ((lustre_ || agg_lustre_) && (write_through_lustre || !stored)) {
    bool l_ok = lustre_put(id, store_data, store_size);
    if (!stored) stored = l_ok;
  }

  if (stored) {
    if (cfg_.dedup_enabled) global_dedup_.put(id, true);
    if (is_pf) {
      std::unique_lock lock(prefix_mutex_);
      prefix_registry_.insert(id);
    }
    if (!gpu_stored || !dram_shadow_ok) {
      auto loc_opt = gpu_ ? gpu_->global_index_.get(id) : std::nullopt;
      if (loc_opt) mark_dirty(id, *loc_opt);
    }
  }
  return stored;
}


bool DistributedStore::get(const BlockId &id, uint8_t *out, size_t *out_size) {

  if (gpu_ && gpu_->get_local(id, out, out_size)) {
    local_gpu_hits_++;
    record_access(id, TierType::LOCAL_GPU);
    return true;
  }


  if (dram_ && dram_->get_local(id, out, out_size)) {
    local_dram_hits_++;
    record_access(id, TierType::LOCAL_DRAM);


    if (cfg_.kv_compression && *out_size >= sizeof(CompressionMeta)) {
      CompressionMeta meta;
      memcpy(&meta, out, sizeof(CompressionMeta));
      size_t orig_size = KVCompressor::original_size(*out_size);
      std::vector<uint8_t> tmp(out, out + *out_size);
      KVCompressor::decompress(tmp.data(), *out_size, meta, out, orig_size);
      *out_size = orig_size;
    }


    if (cfg_.locality_aware && should_promote_local(id)) {
      promote_to_local_gpu(id, out, *out_size);
    }
    return true;
  }


  if (dram_ && gpu_) {
    auto loc = gpu_->locate(id);
    if (loc && !loc->is_local(rank_) && loc->has_dram_shadow) {

      if (dram_->get_remote(loc->node_id, loc->dram_offset, out, loc->dram_size)) {

        if (loc->is_gpu) {
          remote_gpu_hits_++;
          record_access(id, TierType::REMOTE_GPU);
        } else {
          remote_dram_hits_++;
          record_access(id, TierType::REMOTE_DRAM);
        }
        *out_size = loc->dram_size;


        if (dram_) {
          dram_->put_local(id, out, *out_size, is_prefix(id));
        }


        if (cfg_.kv_compression && *out_size >= sizeof(CompressionMeta)) {
          CompressionMeta meta;
          memcpy(&meta, out, sizeof(CompressionMeta));
          size_t orig_size = KVCompressor::original_size(*out_size);
          std::vector<uint8_t> tmp(out, out + *out_size);
          KVCompressor::decompress(tmp.data(), *out_size, meta, out, orig_size);
          *out_size = orig_size;
        }


        if (cfg_.locality_aware && should_promote_local(id)) {
          promote_to_local_gpu(id, out, *out_size);
        }
        return true;
      }
    }
  }


  if (lustre_get(id, out, out_size)) {
    lustre_hits_++;
    record_access(id, TierType::LUSTRE);


    if (dram_) {
      dram_->put_local(id, out, *out_size, is_prefix(id));
    }


    if (cfg_.kv_compression && *out_size >= sizeof(CompressionMeta)) {
      CompressionMeta meta;
      memcpy(&meta, out, sizeof(CompressionMeta));
      size_t orig_size = KVCompressor::original_size(*out_size);
      std::vector<uint8_t> tmp(out, out + *out_size);
      KVCompressor::decompress(tmp.data(), *out_size, meta, out, orig_size);
      *out_size = orig_size;
    }
    return true;
  }

  misses_++;
  return false;
}

bool DistributedStore::get_device(const BlockId &id, void *out,
                                  size_t out_capacity, size_t *out_size) {
  if (gpu_ && gpu_->get_device(id, out, out_capacity, out_size)) {
    local_gpu_hits_++;
    record_access(id, TierType::LOCAL_GPU);
    return true;
  }


  std::vector<uint8_t> host(out_capacity);
  size_t host_size = 0;
  if (!get(id, host.data(), &host_size)) return false;
  if (host_size > out_capacity) return false;
  cudaError_t copy_err = cudaMemcpy(out, host.data(), host_size,
                                    cudaMemcpyDefault);
  if (copy_err != cudaSuccess) return false;
  *out_size = host_size;
  return true;
}

void *DistributedStore::transfer_stream() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) device = 0;


  static thread_local std::unordered_map<int, void *> streams;
  auto it = streams.find(device);
  if (it != streams.end()) return it->second;
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    stream = nullptr;
  }
  streams[device] = stream;
  {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    transfer_streams_[static_cast<int>(transfer_streams_.size())] = stream;
  }
  return stream;
}

bool DistributedStore::available(const BlockId &id) const {


  if (gpu_ && gpu_->locate(id).has_value()) return true;
  if (dram_ && dram_->contains(id)) return true;
  return false;
}

std::vector<uint8_t> DistributedStore::available_batch(
    const std::vector<BlockId> &ids) const {
  std::vector<uint8_t> result(ids.size(), 0);
  for (size_t i = 0; i < ids.size(); ++i) {
    result[i] = available(ids[i]) ? 1 : 0;
  }
  return result;
}


void DistributedStore::pin_promised(const std::vector<BlockId> &ids,
                                    double seconds) {
  if (dram_) dram_->pin_batch(ids, seconds);
}

void DistributedStore::unpin_promised(const std::vector<BlockId> &ids) {
  if (dram_) dram_->unpin_batch(ids);
}

size_t DistributedStore::pinned_blocks() const {
  return dram_ ? dram_->pinned_count() : 0;
}

size_t DistributedStore::available_prefix(
    const std::vector<BlockId> &ids) const {
  size_t hits = 0;
  for (const auto &id : ids) {
    if (!available(id)) break;
    ++hits;
  }
  return hits;
}

bool DistributedStore::contains(const BlockId &id) const {
  if (gpu_ && gpu_->locate(id).has_value()) return true;
  if (dram_ && dram_->contains(id)) return true;
  if (lustre_contains(id)) return true;
  return false;
}

std::optional<BlockLocation> DistributedStore::locate(const BlockId &id) const {
  return gpu_->locate(id);
}


size_t DistributedStore::put_batch(const std::vector<BlockId> &ids,
                                   const std::vector<const uint8_t *> &data,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<bool> &is_prefix_flags) {

  std::vector<size_t> non_dedup_indices;
  non_dedup_indices.reserve(ids.size());
  size_t dedup_saved = 0;

  for (size_t i = 0; i < ids.size(); i++) {
    if (cfg_.dedup_enabled && global_dedup_.contains(ids[i])) {
      dedup_hits_++;
      dedup_bytes_saved_ += sizes[i];
      dedup_saved++;
    } else {
      non_dedup_indices.push_back(i);
    }
  }


  size_t success = dedup_saved;
  for (size_t idx : non_dedup_indices) {
    bool pf = (idx < is_prefix_flags.size()) ? is_prefix_flags[idx] : false;
    if (put(ids[idx], data[idx], sizes[idx], pf)) {
      success++;
    }
  }
  return success;
}


namespace {

inline bool cascade_transfer_debug() {
  static const bool enabled = [] {
    const char *env = std::getenv("CASCADE_TRANSFER_DEBUG");
    return env && std::string(env) == "1";
  }();
  return enabled;
}

inline bool cascade_rma_span() {
  static const bool enabled = [] {
    const char *env = std::getenv("CASCADE_RMA_SPAN");
    return env && std::string(env) == "1";
  }();
  return enabled;
}


inline bool cascade_write_through_lustre() {
  static const bool enabled = [] {
    const char *env = std::getenv("CASCADE_WRITE_THROUGH_LUSTRE");
    return env && std::string(env) == "1";
  }();
  return enabled;
}

using clock_type = std::chrono::steady_clock;


struct CascadeCopy {
  void *dst;
  const void *src;
  size_t size;
};

inline size_t cascade_flush_copies(std::vector<CascadeCopy> &copies,
                                   cudaStream_t stream, size_t *issued) {
  if (copies.empty()) return 0;
  size_t enqueued = 0, operations = 0;
  size_t index = 0;
  while (index < copies.size()) {
    size_t run = 1;
    while (index + run < copies.size()) {
      const CascadeCopy &prev = copies[index + run - 1];
      const CascadeCopy &next = copies[index + run];
      if (static_cast<const uint8_t *>(prev.src) + prev.size != next.src) break;
      if (static_cast<uint8_t *>(prev.dst) + prev.size != next.dst) break;
      ++run;
    }
    size_t bytes = 0;
    for (size_t k = 0; k < run; ++k) bytes += copies[index + k].size;
    if (cudaMemcpyAsync(copies[index].dst, copies[index].src, bytes,
                        cudaMemcpyDefault, stream) == cudaSuccess) {
      enqueued += run;
    }
    ++operations;
    index += run;
  }
  if (issued) *issued = operations;
  copies.clear();
  return enqueued;
}

inline cudaError_t cascade_stream_wait(cudaStream_t stream) {
  static const bool poll = [] {
    const char *env = std::getenv("CASCADE_POLL_SYNC");
    return !env || std::string(env) == "1";
  }();
  if (!poll) {
    return stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
  }
  if (!stream) return cudaDeviceSynchronize();
  for (;;) {
    cudaError_t status = cudaStreamQuery(stream);
    if (status != cudaErrorNotReady) return status;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

inline double ms_since(const clock_type::time_point &start) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - start)
      .count();
}

}

size_t DistributedStore::put_device_batch(
    const std::vector<BlockId> &ids, const std::vector<const void *> &data,
    const std::vector<size_t> &sizes,
    const std::vector<bool> &is_prefix_flags) {
  if (ids.size() != data.size() || ids.size() != sizes.size()) return 0;

  const bool debug = cascade_transfer_debug();
  const auto t_begin = clock_type::now();


  int caller_device = 0;
  const bool have_device = cudaGetDevice(&caller_device) == cudaSuccess;


  std::vector<size_t> pending;
  pending.reserve(ids.size());
  size_t success = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (cfg_.dedup_enabled && global_dedup_.contains(ids[i])) {
      dedup_hits_++;
      dedup_bytes_saved_ += sizes[i];
      ++success;
      continue;
    }
    pending.push_back(i);
  }
  if (pending.empty()) return success;


  const auto t_gpu = clock_type::now();
  std::unordered_set<size_t> gpu_stored;
  if (gpu_) {
    std::vector<BlockId> gpu_ids;
    std::vector<const void *> gpu_data;
    std::vector<size_t> gpu_sizes;
    std::vector<bool> gpu_prefix;
    gpu_ids.reserve(pending.size());
    gpu_data.reserve(pending.size());
    gpu_sizes.reserve(pending.size());
    gpu_prefix.reserve(pending.size());
    for (size_t i : pending) {
      gpu_ids.push_back(ids[i]);
      gpu_data.push_back(data[i]);
      gpu_sizes.push_back(sizes[i]);
      gpu_prefix.push_back(i < is_prefix_flags.size() && is_prefix_flags[i]);
    }
    gpu_->put_device_batch(gpu_ids, gpu_data, gpu_sizes, gpu_prefix);
    for (size_t k = 0; k < gpu_ids.size(); ++k) {
      auto loc = gpu_->global_index_.get(gpu_ids[k]);
      if (loc && loc->is_gpu && loc->node_id == rank_) gpu_stored.insert(pending[k]);
    }
  }
  const double gpu_ms = debug ? ms_since(t_gpu) : 0.0;
  if (have_device) cudaSetDevice(caller_device);


  const auto t_dram = clock_type::now();
  cudaStream_t stream = static_cast<cudaStream_t>(transfer_stream());
  struct Slot {
    size_t index;
    size_t offset;
    size_t size;
  };
  std::vector<Slot> slots;
  slots.reserve(pending.size());
  if (dram_) {
    for (size_t i : pending) {
      size_t offset = 0;
      uint8_t *host = dram_->stage_alloc(sizes[i], &offset);
      if (!host && evict_dram_to_lustre(sizes[i])) {
        host = dram_->stage_alloc(sizes[i], &offset);
      }
      if (!host) continue;
      cudaError_t err = cudaMemcpyAsync(host, data[i], sizes[i],
                                        cudaMemcpyDefault, stream);
      if (err != cudaSuccess) {
        dram_->stage_release(offset, sizes[i]);
        continue;
      }
      slots.push_back({i, offset, sizes[i]});
    }
    if (!slots.empty()) {
      cudaError_t sync_err = cascade_stream_wait(stream);
      if (sync_err != cudaSuccess) {
        for (const auto &slot : slots) dram_->stage_release(slot.offset, slot.size);
        slots.clear();
      }
    }
  }
  const double dram_ms = debug ? ms_since(t_dram) : 0.0;


  const auto t_publish = clock_type::now();
  std::unordered_set<size_t> shadowed;
  size_t shadow_bytes = 0;
  for (const auto &slot : slots) {
    const size_t i = slot.index;
    const bool pf = i < is_prefix_flags.size() && is_prefix_flags[i];
    dram_->stage_commit(ids[i], slot.offset, slot.size, pf);
    shadowed.insert(i);
    shadow_bytes += slot.size;
    if (gpu_) {
      BlockLocation loc{};
      auto existing = gpu_->global_index_.get(ids[i]);
      if (existing) {
        loc = *existing;
      } else {
        loc.gpu_id = -1;
        loc.offset = slot.offset;
        loc.size = slot.size;
        loc.is_gpu = false;
        loc.timestamp =
            std::chrono::steady_clock::now().time_since_epoch().count();
      }
      loc.node_id = rank_;
      loc.dram_offset = slot.offset;
      loc.dram_size = slot.size;
      loc.has_dram_shadow = true;
      loc.is_prefix = pf;
      gpu_->global_index_.put(ids[i], loc, slot.size);
      mark_dirty(ids[i], loc);
    }
  }

  for (size_t i : pending) {
    const bool pf = i < is_prefix_flags.size() && is_prefix_flags[i];
    const bool stored = shadowed.count(i) || gpu_stored.count(i);


    if ((lustre_ || agg_lustre_) && (cascade_write_through_lustre() || !stored)) {
      const uint8_t *host = nullptr;
      if (shadowed.count(i)) {
        size_t shadow_size = 0;
        host = dram_->local_ptr(ids[i], &shadow_size);
      }
      if (host) lustre_put(ids[i], host, sizes[i]);
    }
    if (!stored) continue;
    if (cfg_.dedup_enabled) global_dedup_.put(ids[i], true);
    if (pf) {
      std::unique_lock lock(prefix_mutex_);
      prefix_registry_.insert(ids[i]);
    }
    ++success;
  }
  const double publish_ms = debug ? ms_since(t_publish) : 0.0;

  if (debug) {
    printf("CASCADE put_batch blocks=%zu stored=%zu gpu=%zu shadow=%zu "
           "bytes=%zu gpu_ms=%.3f dram_ms=%.3f publish_ms=%.3f total_ms=%.3f\n",
           ids.size(), success, gpu_stored.size(), shadowed.size(),
           shadow_bytes, gpu_ms, dram_ms, publish_ms, ms_since(t_begin));
    fflush(stdout);
  }
  return success;
}


size_t DistributedStore::get_batch(const std::vector<BlockId> &ids,
                                   std::vector<uint8_t *> &out,
                                   std::vector<size_t> &sizes) {

  std::vector<size_t> local_indices, remote_indices;
  local_indices.reserve(ids.size());
  remote_indices.reserve(ids.size());

  for (size_t i = 0; i < ids.size(); i++) {
    auto loc = gpu_ ? gpu_->locate(ids[i]) : std::nullopt;
    if (!loc || loc->is_local(rank_)) {
      local_indices.push_back(i);
    } else {
      remote_indices.push_back(i);
    }
  }


  size_t success = 0;
  for (size_t idx : local_indices) {
    if (get(ids[idx], out[idx], &sizes[idx])) success++;
  }
  for (size_t idx : remote_indices) {
    if (get(ids[idx], out[idx], &sizes[idx])) success++;
  }
  return success;
}


bool cascade_tier_has_room(const cascade::distributed::DistributedDRAMBackend *dram) {
  if (!dram || dram->capacity() == 0) return false;
  static const double max_fill = [] {
    const char *env = std::getenv("CASCADE_SHADOW_MAX_FILL");
    return env ? atof(env) : 0.85;
  }();
  return static_cast<double>(dram->used_bytes()) / dram->capacity() < max_fill;
}

size_t DistributedStore::prefetch_batch(const std::vector<BlockId> &ids) {
  if (!dram_ || !gpu_ || ids.empty()) return 0;


  const double fill = dram_->capacity() == 0 ? 1.0
      : static_cast<double>(dram_->used_bytes()) / dram_->capacity();
  static const double max_fill = [] {
    const char *env = std::getenv("CASCADE_PREFETCH_MAX_FILL");
    return env ? atof(env) : 0.75;
  }();
  if (fill >= max_fill) {
    prefetch_skipped_full_ += ids.size();
    return 0;
  }

  struct Pending {
    BlockId id;
    int target_rank;
    size_t remote_offset;
    size_t size;
    size_t local_offset;
    uint8_t *host;
  };
  std::vector<Pending> pending;
  pending.reserve(ids.size());
  for (const auto &id : ids) {
    size_t resident = 0;
    if (dram_->local_ptr(id, &resident)) {
      prefetch_hits_++;
      continue;
    }
    auto loc = gpu_->locate(id);
    if (!loc || loc->is_local(rank_) || !loc->has_dram_shadow) continue;
    if (cfg_.kv_compression) continue;
    pending.push_back({id, loc->node_id, loc->dram_offset, loc->dram_size, 0,
                       nullptr});
  }
  if (pending.empty()) return 0;


  std::unordered_map<int, std::vector<size_t>> by_target;
  for (size_t i = 0; i < pending.size(); ++i)
    by_target[pending[i].target_rank].push_back(i);

  std::vector<int> targets;
  std::vector<size_t> offsets;
  std::vector<uint8_t *> outs;
  std::vector<size_t> sizes;
  std::vector<std::pair<size_t, size_t>> chunks;
  std::vector<bool> covered(pending.size(), false);

  for (const auto &[target, indices] : by_target) {
    size_t min_offset = std::numeric_limits<size_t>::max();
    size_t max_end = 0, requested = 0;
    for (size_t i : indices) {
      min_offset = std::min(min_offset, pending[i].remote_offset);
      max_end = std::max(max_end, pending[i].remote_offset + pending[i].size);
      requested += pending[i].size;
    }
    const size_t span = max_end - min_offset;
    if (indices.size() >= 4 && span <= requested * 5 / 4) {
      size_t chunk_offset = 0;
      uint8_t *chunk = dram_->stage_alloc(span, &chunk_offset);
      if (!chunk && evict_dram_to_lustre(span))
        chunk = dram_->stage_alloc(span, &chunk_offset);
      if (chunk) {
        for (size_t i : indices) {
          const size_t delta = pending[i].remote_offset - min_offset;
          pending[i].local_offset = chunk_offset + delta;
          pending[i].host = chunk + delta;
          covered[i] = true;
        }
        targets.push_back(target);
        offsets.push_back(min_offset);
        outs.push_back(chunk);
        sizes.push_back(span);
        chunks.emplace_back(chunk_offset, span);
        continue;
      }
    }
    for (size_t i : indices) {
      uint8_t *host = dram_->stage_alloc(pending[i].size, &pending[i].local_offset);
      if (!host && evict_dram_to_lustre(pending[i].size))
        host = dram_->stage_alloc(pending[i].size, &pending[i].local_offset);
      if (!host) continue;
      pending[i].host = host;
      covered[i] = true;
      targets.push_back(target);
      offsets.push_back(pending[i].remote_offset);
      outs.push_back(host);
      sizes.push_back(pending[i].size);
      chunks.emplace_back(pending[i].local_offset, pending[i].size);
    }
  }
  if (targets.empty()) return 0;

  if (!dram_->get_remote_batch(targets, offsets, outs, sizes)) {
    for (const auto &chunk : chunks) dram_->stage_release(chunk.first, chunk.second);
    return 0;
  }

  size_t installed = 0;
  for (size_t i = 0; i < pending.size(); ++i) {
    if (!covered[i] || !pending[i].host) continue;
    dram_->stage_commit(pending[i].id, pending[i].local_offset, pending[i].size,
                        is_prefix(pending[i].id));
    ++installed;
  }
  prefetched_blocks_ += installed;
  if (cascade_transfer_debug()) {
    printf("CASCADE prefetch requested=%zu installed=%zu rma_ops=%zu\n",
           ids.size(), installed, targets.size());
    fflush(stdout);
  }
  return installed;
}

size_t DistributedStore::get_device_batch(
    const std::vector<BlockId> &ids, const std::vector<void *> &out,
    const std::vector<size_t> &capacities, std::vector<size_t> &sizes) {
  if (ids.size() != out.size() || ids.size() != capacities.size()) return 0;
  sizes.assign(ids.size(), 0);
  if (ids.empty()) return 0;

  const bool debug = cascade_transfer_debug();
  const auto t_begin = clock_type::now();

  cudaStream_t stream = static_cast<cudaStream_t>(transfer_stream());
  size_t success = 0;
  size_t enqueued = 0;
  size_t local_gpu_blocks = 0, local_dram_blocks = 0;
  size_t remote_blocks = 0, fallback_blocks = 0;
  size_t remote_bytes = 0;

  struct RemoteRequest {
    size_t index;
    int target_rank;
    size_t remote_offset;
    size_t size;
    size_t local_offset;
    uint8_t *local_host;
    bool owns_slot;
  };
  std::vector<RemoteRequest> remotes;
  std::vector<size_t> fallback;
  std::vector<CascadeCopy> pending_copies;
  size_t copy_operations = 0;


  const auto t_local = clock_type::now();
  for (size_t i = 0; i < ids.size(); ++i) {


    if (gpu_ && gpu_->get_device_async(ids[i], out[i], capacities[i], &sizes[i],
                                       stream)) {
      local_gpu_hits_++;
      record_access(ids[i], TierType::LOCAL_GPU);
      ++success;
      ++enqueued;
      ++local_gpu_blocks;
      continue;
    }


    if (dram_ && !cfg_.kv_compression) {
      size_t shadow_size = 0;
      const uint8_t *host = dram_->local_ptr(ids[i], &shadow_size);
      if (host && shadow_size <= capacities[i]) {
        pending_copies.push_back({out[i], host, shadow_size});
        sizes[i] = shadow_size;
        local_dram_hits_++;
        record_access(ids[i], TierType::LOCAL_DRAM);
        ++success;
        ++enqueued;
        ++local_dram_blocks;
        continue;
      }
    }


    if (dram_ && gpu_) {
      auto loc = gpu_->locate(ids[i]);
      if (loc && !loc->is_local(rank_) && loc->has_dram_shadow &&
          loc->dram_size <= capacities[i] && !cfg_.kv_compression) {
        remotes.push_back({i, loc->node_id, loc->dram_offset, loc->dram_size, 0,
                           nullptr, false});
        continue;
      }
    }

    fallback.push_back(i);
  }
  const double local_ms = debug ? ms_since(t_local) : 0.0;


  double rma_ms = 0.0;
  size_t rma_operations = 0;
  if (!remotes.empty()) {
    const auto t_rma = clock_type::now();


    static const double span_slack = [] {
      const char *env = std::getenv("CASCADE_RMA_SPAN_SLACK");
      return env ? atof(env) : 1.25;
    }();
    static const size_t span_min_blocks = [] {
      const char *env = std::getenv("CASCADE_RMA_SPAN_MIN_BLOCKS");
      return env ? static_cast<size_t>(atoi(env)) : 4;
    }();

    std::unordered_map<int, std::vector<size_t>> by_target;
    for (size_t i = 0; i < remotes.size(); ++i) {
      by_target[remotes[i].target_rank].push_back(i);
    }
    struct SpanFetch {
      int target_rank;
      size_t remote_offset;
      size_t bytes;
      uint8_t *host;
    };
    std::vector<SpanFetch> spans;
    std::vector<bool> covered(remotes.size(), false);
    const bool direct_mode = dram_->device_rma_supported();
    if (!direct_mode && span_slack > 0.0) {
      for (const auto &[target, indices] : by_target) {
        if (indices.size() < span_min_blocks) continue;
        size_t min_offset = std::numeric_limits<size_t>::max();
        size_t max_end = 0;
        size_t requested = 0;
        for (size_t i : indices) {
          min_offset = std::min(min_offset, remotes[i].remote_offset);
          max_end = std::max(max_end, remotes[i].remote_offset + remotes[i].size);
          requested += remotes[i].size;
        }
        const size_t span_bytes = max_end - min_offset;
        if (static_cast<double>(span_bytes) >
            static_cast<double>(requested) * span_slack) {
          continue;
        }
        size_t chunk_offset = 0;
        uint8_t *chunk = nullptr;
        if (cascade_tier_has_room(dram_.get())) {
          chunk = dram_->stage_alloc(span_bytes, &chunk_offset);
        }
        if (!chunk) continue;
        for (size_t i : indices) {
          const size_t delta = remotes[i].remote_offset - min_offset;
          remotes[i].local_offset = chunk_offset + delta;
          remotes[i].local_host = chunk + delta;
          remotes[i].owns_slot = true;
          covered[i] = true;
          remote_bytes += remotes[i].size;
        }
        spans.push_back({target, min_offset, span_bytes, chunk});
      }
    }


    std::vector<std::vector<uint8_t>> scratch;
    for (size_t i = 0; i < remotes.size(); ++i) {
      if (covered[i]) continue;
      auto &request = remotes[i];
      if (direct_mode) {

        remote_bytes += request.size;
        continue;
      }
      uint8_t *host = nullptr;
      if (cascade_tier_has_room(dram_.get())) {
        host = dram_->stage_alloc(request.size, &request.local_offset);
      }
      if (host) {
        request.local_host = host;
        request.owns_slot = true;
      } else {
        scratch.emplace_back(request.size);
        request.local_host = scratch.back().data();
        request.owns_slot = false;
      }
      remote_bytes += request.size;
    }

    std::vector<int> targets;
    std::vector<size_t> offsets;
    std::vector<uint8_t *> outs;
    std::vector<void *> device_outs;
    std::vector<size_t> rma_sizes;
    targets.reserve(remotes.size());
    offsets.reserve(remotes.size());
    outs.reserve(remotes.size());
    device_outs.reserve(remotes.size());
    rma_sizes.reserve(remotes.size());
    for (const auto &span : spans) {
      targets.push_back(span.target_rank);
      offsets.push_back(span.remote_offset);
      outs.push_back(span.host);
      rma_sizes.push_back(span.bytes);
    }
    for (size_t i = 0; i < remotes.size(); ++i) {
      if (covered[i]) continue;
      const auto &request = remotes[i];
      targets.push_back(request.target_rank);
      offsets.push_back(request.remote_offset);
      outs.push_back(request.local_host);
      rma_sizes.push_back(request.size);
    }
    for (const auto &request : remotes) {
      device_outs.push_back(out[request.index]);
    }
    rma_operations = targets.size();

    bool ok = false;
    const bool direct = dram_->device_rma_supported();
    if (direct) {


      std::vector<int> direct_targets;
      std::vector<size_t> direct_offsets;
      std::vector<size_t> direct_sizes;
      direct_targets.reserve(remotes.size());
      direct_offsets.reserve(remotes.size());
      direct_sizes.reserve(remotes.size());
      for (const auto &request : remotes) {
        direct_targets.push_back(request.target_rank);
        direct_offsets.push_back(request.remote_offset);
        direct_sizes.push_back(request.size);
      }
      ok = dram_->get_remote_device_batch(direct_targets, direct_offsets,
                                          device_outs, direct_sizes);
    } else {
      ok = dram_->get_remote_batch(targets, offsets, outs, rma_sizes);
    }

    if (ok) {
      for (auto &request : remotes) {
        const size_t i = request.index;
        auto loc = gpu_->locate(ids[i]);
        if (loc && loc->is_gpu) {
          remote_gpu_hits_++;
          record_access(ids[i], TierType::REMOTE_GPU);
        } else {
          remote_dram_hits_++;
          record_access(ids[i], TierType::REMOTE_DRAM);
        }
        ++remote_blocks;
        if (direct) {

          if (request.owns_slot)
            dram_->stage_release(request.local_offset, request.size);
          sizes[i] = request.size;
          ++success;
          continue;
        }
        pending_copies.push_back({out[i], request.local_host, request.size});
        sizes[i] = request.size;
        ++success;
        ++enqueued;
        if (request.owns_slot) {
          dram_->stage_commit(ids[i], request.local_offset, request.size,
                              is_prefix(ids[i]));
        }
      }
      if (!scratch.empty() && enqueued > 0) {

        cascade_flush_copies(pending_copies, stream, &copy_operations);
        cascade_stream_wait(stream);
        enqueued = 0;
      }
    } else {
      for (size_t i = 0; i < remotes.size(); ++i) {
        auto &request = remotes[i];
        if (request.owns_slot && !covered[i])
          dram_->stage_release(request.local_offset, request.size);
        fallback.push_back(request.index);
      }
      for (const auto &span : spans) {

        dram_->stage_release(
            static_cast<size_t>(span.host - dram_->base_ptr()), span.bytes);
      }
      remote_blocks = 0;
      remote_bytes = 0;
    }
    rma_ms = debug ? ms_since(t_rma) : 0.0;
  }


  double fallback_ms = 0.0;
  if (!fallback.empty()) {
    const auto t_fallback = clock_type::now();
    for (size_t i : fallback) {
      std::vector<uint8_t> host(capacities[i]);
      size_t host_size = 0;
      if (!get(ids[i], host.data(), &host_size)) continue;
      if (host_size > capacities[i]) continue;
      if (cudaMemcpy(out[i], host.data(), host_size, cudaMemcpyDefault) !=
          cudaSuccess) {
        continue;
      }
      sizes[i] = host_size;
      ++success;
      ++fallback_blocks;
    }
    fallback_ms = debug ? ms_since(t_fallback) : 0.0;
  }


  const auto t_sync = clock_type::now();
  size_t flushed = 0;
  cascade_flush_copies(pending_copies, stream, &flushed);
  copy_operations += flushed;
  if (enqueued > 0 && cascade_stream_wait(stream) != cudaSuccess) return 0;
  const double sync_ms = debug ? ms_since(t_sync) : 0.0;

  if (debug) {
    printf("CASCADE get_batch blocks=%zu hit=%zu local_gpu=%zu local_dram=%zu "
           "remote=%zu lustre=%zu remote_bytes=%zu rma_ops=%zu h2d_ops=%zu "
           "classify_ms=%.3f rma_ms=%.3f lustre_ms=%.3f h2d_sync_ms=%.3f "
           "total_ms=%.3f\n",
           ids.size(), success, local_gpu_blocks, local_dram_blocks,
           remote_blocks, fallback_blocks, remote_bytes, rma_operations,
           copy_operations, local_ms, rma_ms, fallback_ms, sync_ms,
           ms_since(t_begin));
    fflush(stdout);
  }
  return success;
}


bool DistributedStore::evict_gpu_to_dram(size_t needed_bytes) {
  if (!gpu_ || !dram_) return false;

  bool any_evicted = false;

  for (int g = 0; g < gpu_->num_gpus(); g++) {
    std::function<bool(const BlockId &)> is_pf_fn = nullptr;
    if (cfg_.semantic_eviction) {
      is_pf_fn = [this](const BlockId &bid) -> bool {
        return is_prefix(bid);
      };
    }

    auto evicted_blocks = gpu_->evict_gpu_for_space(g, needed_bytes, is_pf_fn);

    for (auto &evicted : evicted_blocks) {


      const uint8_t *dram_data = evicted.data.data();
      size_t dram_size = evicted.size;
      std::vector<uint8_t> compressed_buf;

      if (cfg_.kv_compression && evicted.size >= 64) {
        CompressionMeta meta;
        compressed_buf = KVCompressor::compress(evicted.data.data(), evicted.size, meta);
        dram_data = compressed_buf.data();
        dram_size = compressed_buf.size();
      }


      bool dram_ok = dram_->put_local(evicted.id, dram_data,
                                       dram_size, is_prefix(evicted.id));
      if (!dram_ok) {

        evict_dram_to_lustre(dram_size);
        dram_ok = dram_->put_local(evicted.id, dram_data,
                                    dram_size, is_prefix(evicted.id));
      }
      if (!dram_ok) {

        lustre_put(evicted.id, dram_data, dram_size);
      }

      if (dram_ok) {
        BlockLocation loc;
        loc.node_id = rank_;
        loc.gpu_id = -1;
        loc.offset = dram_->get_offset(evicted.id);
        loc.size = dram_size;
        loc.dram_offset = loc.offset;
        loc.dram_size = dram_size;
        loc.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        loc.is_gpu = false;
        loc.is_prefix = is_prefix(evicted.id);
        loc.has_dram_shadow = true;
        gpu_->global_index_.put(evicted.id, loc);
      }
      gpu_evictions_++;
      any_evicted = true;
    }

    if (any_evicted) break;
  }

  return any_evicted;
}

bool DistributedStore::evict_dram_to_lustre(size_t needed_bytes) {
  if (!dram_) return false;


  static const bool evict_debug =
      std::getenv("CASCADE_EVICT_DEBUG") != nullptr;
  static std::atomic<uint64_t> ev_calls{0}, ev_blocks{0};
  static std::atomic<uint64_t> ns_scan{0}, ns_put{0}, ns_index{0};
  auto clock_now = [] { return std::chrono::steady_clock::now(); };
  auto t0 = clock_now();

  auto evicted = dram_->evict_for_space(needed_bytes, cfg_.semantic_eviction);
  auto t1 = clock_now();


  for (auto &[id, data] : evicted) {
    auto t2 = clock_now();

    lustre_put(id, data.data(), data.size());
    auto t3 = clock_now();
    if (evict_debug) {
      ns_put += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2)
                    .count();
      ev_blocks++;
    }
    dram_evictions_++;


    if (gpu_) {
      auto loc = gpu_->global_index_.get(id);
      if (loc && loc->is_local(rank_)) {
        BlockLocation updated = *loc;
        updated.has_dram_shadow = false;
        updated.dram_offset = 0;
        updated.dram_size = 0;
        auto t4 = clock_now();
        gpu_->global_index_.put(id, updated, 0);
        mark_dirty(id, updated);
        if (evict_debug) {
          ns_index +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  clock_now() - t4).count();
        }
      }
    }
  }

  if (evict_debug) {
    ns_scan += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                   .count();
    uint64_t n = ++ev_calls;
    if (n % 200 == 0) {
      std::fprintf(stderr,
                   "Cascade evict rank=%d calls=%llu blocks=%llu "
                   "scan=%.1fs lustre=%.1fs index=%.1fs\n",
                   rank_, (unsigned long long)n,
                   (unsigned long long)ev_blocks.load(),
                   ns_scan.load() / 1e9, ns_put.load() / 1e9,
                   ns_index.load() / 1e9);
      std::fflush(stderr);
    }
  }

  return !evicted.empty();
}

void DistributedStore::sync_prefix_registry() {
#ifdef USE_MPI

  std::vector<BlockId> local_prefixes;
  {
    std::shared_lock lock(prefix_mutex_);
    local_prefixes.assign(prefix_registry_.begin(), prefix_registry_.end());
  }


  std::vector<char> send_buf;
  uint32_t count = local_prefixes.size();
  send_buf.insert(send_buf.end(), reinterpret_cast<char*>(&count),
                  reinterpret_cast<char*>(&count) + sizeof(count));
  for (const auto &id : local_prefixes) {
    uint32_t len = id.size();
    send_buf.insert(send_buf.end(), reinterpret_cast<char*>(&len),
                    reinterpret_cast<char*>(&len) + sizeof(len));
    send_buf.insert(send_buf.end(), id.begin(), id.end());
  }


  int send_size = send_buf.size();
  std::vector<int> recv_sizes(world_size_);
  MPI_Allgather(&send_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, comm_);


  std::vector<int> displs(world_size_, 0);
  int total_recv = 0;
  for (int i = 0; i < world_size_; i++) {
    displs[i] = total_recv;
    total_recv += recv_sizes[i];
  }


  std::vector<char> recv_buf(total_recv);
  MPI_Allgatherv(send_buf.data(), send_size, MPI_CHAR,
                 recv_buf.data(), recv_sizes.data(), displs.data(),
                 MPI_CHAR, comm_);


  {
    std::unique_lock lock(prefix_mutex_);
    for (int r = 0; r < world_size_; r++) {
      if (r == rank_) continue;
      const char *ptr = recv_buf.data() + displs[r];
      uint32_t cnt;
      memcpy(&cnt, ptr, sizeof(cnt));
      ptr += sizeof(cnt);
      for (uint32_t i = 0; i < cnt; i++) {
        uint32_t len;
        memcpy(&len, ptr, sizeof(len));
        ptr += sizeof(len);
        BlockId id(ptr, ptr + len);
        ptr += len;
        prefix_registry_.insert(id);
      }
    }
  }

  if (rank_ == 0) {
    std::shared_lock lock(prefix_mutex_);
    printf("[Prefix Sync] Global prefix registry: %zu blocks\n",
           prefix_registry_.size());
  }


  if (cfg_.prefix_replication && world_size_ > 1) {
    broadcast_prefix_data();
  }
#endif
}


void DistributedStore::broadcast_prefix_data() {
#ifdef USE_MPI

  std::vector<BlockId> my_owned_prefixes;
  {
    std::shared_lock lock(prefix_mutex_);
    for (const auto& pid : prefix_registry_) {
      bool have_local = false;
      if (gpu_) {
        auto loc = gpu_->global_index_.get(pid);
        if (loc && loc->is_local(rank_)) have_local = true;
      }
      if (!have_local && dram_ && dram_->contains(pid)) {
        have_local = true;
      }
      if (have_local) {
        my_owned_prefixes.push_back(pid);
      }
    }
  }


  for (int r = 0; r < world_size_; r++) {

    std::vector<char> send_buf;
    uint32_t my_count = 0;

    if (r == rank_) {
      my_count = my_owned_prefixes.size();

      send_buf.insert(send_buf.end(), (char*)&my_count,
                      (char*)&my_count + sizeof(my_count));

      for (const auto& pid : my_owned_prefixes) {

        uint32_t id_len = pid.size();
        send_buf.insert(send_buf.end(), (char*)&id_len,
                        (char*)&id_len + sizeof(id_len));
        send_buf.insert(send_buf.end(), pid.begin(), pid.end());


        size_t data_size = 0;
        auto loc = gpu_ ? gpu_->global_index_.get(pid) : std::nullopt;
        if (loc) data_size = loc->size;

        send_buf.insert(send_buf.end(), (char*)&data_size,
                        (char*)&data_size + sizeof(data_size));

        if (data_size > 0) {
          size_t buf_offset = send_buf.size();
          send_buf.resize(buf_offset + data_size);
          size_t out_size = 0;
          if (gpu_) {
            gpu_->get_local(pid, (uint8_t*)&send_buf[buf_offset], &out_size);
          }
        }
      }
    }


    int64_t buf_size = send_buf.size();
    MPI_Bcast(&buf_size, sizeof(int64_t), MPI_BYTE, r, comm_);

    if (buf_size == 0) continue;


    if (r != rank_) {
      send_buf.resize(buf_size);
    }


    const size_t CHUNK = 1ULL * 1024 * 1024 * 1024;
    for (size_t offset = 0; offset < (size_t)buf_size; offset += CHUNK) {
      size_t remaining = (size_t)buf_size - offset;
      int chunk_size = (int)std::min(remaining, CHUNK);
      MPI_Bcast(send_buf.data() + offset, chunk_size, MPI_CHAR, r, comm_);
    }


    if (r != rank_) {
      const char* ptr = send_buf.data();
      const char* end = send_buf.data() + buf_size;

      if (ptr + sizeof(uint32_t) > end) continue;
      uint32_t count;
      memcpy(&count, ptr, sizeof(count));
      ptr += sizeof(count);

      for (uint32_t i = 0; i < count; i++) {
        if (ptr + sizeof(uint32_t) > end) break;
        uint32_t id_len;
        memcpy(&id_len, ptr, sizeof(id_len));
        ptr += sizeof(id_len);

        if (ptr + id_len > end) break;
        BlockId prefix_id(ptr, ptr + id_len);
        ptr += id_len;

        if (ptr + sizeof(size_t) > end) break;
        size_t data_size;
        memcpy(&data_size, ptr, sizeof(data_size));
        ptr += sizeof(data_size);

        if (data_size == 0 || ptr + data_size > end) {
          ptr += data_size;
          continue;
        }

        const uint8_t* block_data = (const uint8_t*)ptr;
        ptr += data_size;


        bool already_local = false;
        if (gpu_) {
          auto loc = gpu_->global_index_.get(prefix_id);
          if (loc && loc->is_local(rank_)) already_local = true;
        }
        if (!already_local && dram_ && dram_->contains(prefix_id)) {
          already_local = true;
        }
        if (already_local) continue;


        bool gpu_ok = false;
        if (gpu_) {
          int tgt_gpu = gpu_->get_target_gpu(prefix_id);
          gpu_ok = gpu_->put_local(prefix_id, block_data, data_size, tgt_gpu, true);
          if (!gpu_ok && evict_gpu_to_dram(data_size)) {
            gpu_ok = gpu_->put_local(prefix_id, block_data, data_size, tgt_gpu, true);
          }
        }


        bool dram_ok = false;
        if (dram_) {
          const uint8_t* dram_data = block_data;
          size_t dram_size = data_size;
          std::vector<uint8_t> comp_buf;

          if (cfg_.kv_compression && data_size >= 64) {
            CompressionMeta meta;
            comp_buf = KVCompressor::compress(block_data, data_size, meta);
            dram_data = comp_buf.data();
            dram_size = comp_buf.size();
          }

          dram_ok = dram_->put_local(prefix_id, dram_data, dram_size, true);
          if (!dram_ok && evict_dram_to_lustre(dram_size)) {
            dram_ok = dram_->put_local(prefix_id, dram_data, dram_size, true);
          }

          if (gpu_ok && dram_ok && gpu_) {
            auto existing = gpu_->global_index_.get(prefix_id);
            if (existing) {
              BlockLocation loc = *existing;
              loc.dram_offset = dram_->get_offset(prefix_id);
              loc.dram_size = dram_size;
              loc.has_dram_shadow = true;
              gpu_->global_index_.put(prefix_id, loc);
            }
          }
        }


        {
          std::unique_lock lock(prefix_mutex_);
          prefix_registry_.insert(prefix_id);
        }
        if (cfg_.dedup_enabled) {
          global_dedup_.put(prefix_id, true);
        }
      }
    }
  }

  if (rank_ == 0) {
    printf("[Prefix Replication] Batched broadcast %zu owned prefixes to %d nodes\n",
           my_owned_prefixes.size(), world_size_);
  }
#endif
}


void DistributedStore::record_access(const BlockId &id, TierType origin_tier) {
  auto existing = access_tracker_.get(id);
  AccessRecord rec;
  if (existing) {
    rec = *existing;
  }
  rec.total_count++;
  rec.window_total++;


  if (origin_tier == TierType::LOCAL_GPU || origin_tier == TierType::LOCAL_DRAM) {
    rec.local_count++;
  } else if (origin_tier == TierType::REMOTE_GPU || origin_tier == TierType::REMOTE_DRAM) {
    rec.remote_count++;
    rec.window_remote++;
  }


  if (rec.window_total >= AccessRecord::WINDOW_SIZE) {
    float current_rate = static_cast<float>(rec.window_remote) / rec.window_total;
    rec.ema_remote_rate = AccessRecord::EMA_ALPHA * current_rate
                        + (1.0f - AccessRecord::EMA_ALPHA) * rec.ema_remote_rate;
    rec.window_remote = 0;
    rec.window_total = 0;
  }

  rec.last_access_node = rank_;
  rec.last_access_time =
      std::chrono::steady_clock::now().time_since_epoch().count();
  access_tracker_.put(id, rec);
}


bool DistributedStore::should_promote_local(const BlockId &id) const {
  auto rec = access_tracker_.get(id);
  if (!rec) return false;


  return rec->total_count >= cfg_.promotion_threshold
      && rec->ema_remote_rate > 0.5f;
}

void DistributedStore::promote_to_local_gpu(const BlockId &id,
                                            const uint8_t *data, size_t size) {
  if (!gpu_) return;

  int target_gpu = gpu_->get_target_gpu(id);
  if (gpu_->put_local(id, data, size, target_gpu, is_prefix(id))) {
    promotions_to_local_++;
  }
}


bool DistributedStore::lustre_put(const BlockId &id, const uint8_t *data,
                                   size_t size) {
  if (agg_lustre_) return agg_lustre_->put(id, data, size);
  if (lustre_) return lustre_->put(id, data, size);
  return false;
}

bool DistributedStore::lustre_get(const BlockId &id, uint8_t *out,
                                   size_t *out_size) {
  if (agg_lustre_) return agg_lustre_->get(id, out, out_size);
  if (lustre_) return lustre_->get(id, out, out_size);
  return false;
}

bool DistributedStore::lustre_contains(const BlockId &id) const {
  if (agg_lustre_) return agg_lustre_->contains(id);
  if (lustre_) return lustre_->contains(id);
  return false;
}

bool DistributedStore::is_prefix(const BlockId &id) const {
  std::shared_lock lock(prefix_mutex_);
  return prefix_registry_.count(id) > 0;
}


void DistributedStore::warm_remote() {
  if (dram_) dram_->warm_remote();
}

void DistributedStore::barrier() {
#ifdef USE_MPI
  MPI_Barrier(comm_);
#endif
}


void DistributedStore::mark_dirty(const BlockId &id, const BlockLocation &loc) {
  std::lock_guard<std::mutex> lock(dirty_mutex_);
  dirty_blocks_.push_back({id, loc});
}


void DistributedStore::sync_metadata() {
#ifdef USE_MPI
  sync_prefix_registry();


  std::vector<std::pair<BlockId, BlockLocation>> dirty;
  {
    std::lock_guard<std::mutex> lock(dirty_mutex_);
    dirty.swap(dirty_blocks_);
  }
  sync_epoch_++;


  std::vector<char> send_buf;
  uint32_t count = dirty.size();
  send_buf.insert(send_buf.end(), (char*)&count, (char*)&count + sizeof(count));
  for (const auto& [key, loc] : dirty) {
    uint32_t klen = key.size();
    send_buf.insert(send_buf.end(), (char*)&klen, (char*)&klen + sizeof(klen));
    send_buf.insert(send_buf.end(), key.begin(), key.end());
    send_buf.insert(send_buf.end(), (char*)&loc, (char*)&loc + sizeof(BlockLocation));
  }


  int send_size = send_buf.size();
  std::vector<int> recv_sizes(world_size_);
  MPI_Allgather(&send_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, comm_);
  std::vector<int> displs(world_size_, 0);
  int total = 0;
  for(int i=0; i<world_size_; i++) { displs[i]=total; total+=recv_sizes[i]; }
  std::vector<char> recv_buf(total);
  MPI_Allgatherv(send_buf.data(), send_size, MPI_CHAR, recv_buf.data(),
                 recv_sizes.data(), displs.data(), MPI_CHAR, comm_);


  size_t new_blocks = 0;
  const char* ptr = recv_buf.data();
  const char* buf_end = recv_buf.data() + total;
  for(int r=0; r<world_size_; r++) {
    if(r==rank_) { ptr += recv_sizes[r]; continue; }
    if(ptr + sizeof(uint32_t) > buf_end) break;
    uint32_t cnt; memcpy(&cnt, ptr, sizeof(cnt)); ptr+=sizeof(cnt);
    for(uint32_t i=0; i<cnt; i++) {
      if(ptr + sizeof(uint32_t) > buf_end) break;
      uint32_t klen; memcpy(&klen, ptr, sizeof(klen)); ptr+=sizeof(klen);
      if(ptr + klen + sizeof(BlockLocation) > buf_end) break;
      BlockId id(ptr, ptr+klen); ptr+=klen;
      BlockLocation loc; memcpy(&loc, ptr, sizeof(loc)); ptr+=sizeof(loc);
      gpu_->global_index_.put(id, loc);
      if(cfg_.dedup_enabled) global_dedup_.put(id, true);
      new_blocks++;
    }
  }
  if(rank_==0) {
    printf("[Delta Sync] Epoch %lu: sent %u dirty, received %zu new (total global: %zu)\n",
           sync_epoch_.load(), count, new_blocks, gpu_->global_index_.size());
  }
#endif
  barrier();
}

DistributedStore::Stats DistributedStore::get_stats_local() {
  Stats s{};
  s.local_gpu_used = gpu_ ? gpu_->used_bytes() : 0;
  s.local_dram_used = dram_ ? dram_->used_bytes() : 0;
  s.local_gpu_hits = local_gpu_hits_.load();
  s.local_dram_hits = local_dram_hits_.load();
  s.remote_gpu_hits = remote_gpu_hits_.load();
  s.remote_dram_hits = remote_dram_hits_.load();
  s.lustre_hits = lustre_hits_.load();
  s.misses = misses_.load();
  s.dedup_hits = dedup_hits_.load();
  s.dedup_bytes_saved = dedup_bytes_saved_.load();
  s.gpu_evictions = gpu_evictions_.load();
  s.dram_evictions = dram_evictions_.load();
  s.prefix_blocks_protected = prefix_blocks_protected_.load();
  s.promotions_to_local = promotions_to_local_.load();
  s.compression_savings = compression_savings_.load();
  s.total_blocks = global_dedup_.size();
  {
    std::shared_lock lock(prefix_mutex_);
    s.prefix_blocks = prefix_registry_.size();
  }
  s.prefetched_blocks = prefetched_blocks_.load();
  s.prefetch_hits = prefetch_hits_.load();
  s.cluster_gpu_used = s.local_gpu_used;
  s.cluster_dram_used = s.local_dram_used;
  return s;
}

DistributedStore::Stats DistributedStore::get_stats() {
  Stats s{};
  s.local_gpu_used = gpu_ ? gpu_->used_bytes() : 0;
  s.local_dram_used = dram_ ? dram_->used_bytes() : 0;
  s.local_gpu_hits = local_gpu_hits_.load();
  s.local_dram_hits = local_dram_hits_.load();
  s.remote_gpu_hits = remote_gpu_hits_.load();
  s.remote_dram_hits = remote_dram_hits_.load();
  s.lustre_hits = lustre_hits_.load();
  s.misses = misses_.load();
  s.dedup_hits = dedup_hits_.load();
  s.dedup_bytes_saved = dedup_bytes_saved_.load();
  s.gpu_evictions = gpu_evictions_.load();
  s.dram_evictions = dram_evictions_.load();
  s.prefix_blocks_protected = prefix_blocks_protected_.load();
  s.promotions_to_local = promotions_to_local_.load();
  s.compression_savings = compression_savings_.load();
  s.total_blocks = global_dedup_.size();
  s.prefetched_blocks = prefetched_blocks_.load();
  s.prefetch_hits = prefetch_hits_.load();

  {
    std::shared_lock lock(prefix_mutex_);
    s.prefix_blocks = prefix_registry_.size();
  }

#ifdef USE_MPI
  unsigned long long local_gpu = s.local_gpu_used;
  unsigned long long local_dram = s.local_dram_used;
  unsigned long long cluster_gpu = 0;
  unsigned long long cluster_dram = 0;

  MPI_Allreduce(&local_gpu, &cluster_gpu, 1, MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, comm_);
  MPI_Allreduce(&local_dram, &cluster_dram, 1, MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, comm_);

  s.cluster_gpu_used = static_cast<size_t>(cluster_gpu);
  s.cluster_dram_used = static_cast<size_t>(cluster_dram);
#else
  s.cluster_gpu_used = s.local_gpu_used;
  s.cluster_dram_used = s.local_dram_used;
#endif

  return s;
}


void DistributedDRAMBackend::clear() {
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    std::lock_guard<std::mutex> lru_lock(lru_mutex_);
    index_.clear();
    lru_list_.clear();
    lru_map_.clear();
    free_list_.clear();
    free_list_.push_back({0, capacity_});
    used_.store(0);
    write_offset_.store(0);
}

void DistributedGPUBackend::clear() {
    for (auto &gpu : gpus_) {
        gpu->clear();
    }
    global_index_.clear();
}

void DistributedStore::clear() {
    barrier();
    if (gpu_) gpu_->clear();
    if (dram_) dram_->clear();
    global_dedup_.clear();
    {
        std::unique_lock lock(prefix_mutex_);
        prefix_registry_.clear();
    }
    access_tracker_.clear();
    {
        std::lock_guard<std::mutex> lock(dirty_mutex_);
        dirty_blocks_.clear();
    }
    local_gpu_hits_ = 0; local_dram_hits_ = 0;
    remote_gpu_hits_ = 0; remote_dram_hits_ = 0;
    lustre_hits_ = 0; misses_ = 0;
    dedup_hits_ = 0; dedup_bytes_saved_ = 0;
    gpu_evictions_ = 0; dram_evictions_ = 0;
    prefix_blocks_protected_ = 0; promotions_to_local_ = 0;
    compression_savings_ = 0;
    if (rank_ == 0) {
        printf("[Cascade] Memory tiers and indices cleared.\n");
    }
    barrier();
}

void DistributedStore::flush_lustre_local() {
    if (lustre_) lustre_->flush();
    if (agg_lustre_) agg_lustre_->flush();
}

void DistributedStore::flush() {
    barrier();
    flush_lustre_local();
    barrier();
}

}
}
