/**
 * Cascade Distributed Backend Implementation
 *
 * 3 Core Novelties:
 *   1. Cross-node semantic-aware eviction (prefix block protection)
 *   2. Distributed content-addressed deduplication (SHA256-based global index)
 *   3. Locality-aware hierarchical placement (access frequency tracking)
 *
 *
 */

#include "cascade_distributed.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <cuda_runtime.h>
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


  CUDA_CHECK(cudaHostAlloc(&dram_base_, capacity_, cudaHostAllocDefault));
  memset(dram_base_, 0, capacity_);


  MPI_Win_create(dram_base_, capacity_, 1, MPI_INFO_NULL, comm_, &window_);
  MPI_Win_lock_all(0, window_);

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


  auto it = lru_list_.rbegin();
  while (it != lru_list_.rend() && freed < needed) {
    BlockId id = *it;
    auto block = index_.get(id);
    if (!block) {
      ++it;
      continue;
    }


    if (protect_prefix && block->is_prefix) {
      ++it;
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
    

    auto fwd_it = std::next(it).base();
    lru_map_.erase(id);
    lru_list_.erase(fwd_it);
    it = lru_list_.rbegin();
  }

  return evicted;
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
                                             MPI_Comm comm)
    : cap_per_gpu_(cap_per_gpu), num_gpus_(num_gpus), comm_(comm) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &world_size_);
#else
DistributedGPUBackend::DistributedGPUBackend(size_t cap_per_gpu, int num_gpus)
    : cap_per_gpu_(cap_per_gpu), num_gpus_(num_gpus) {
#endif


  for (int i = 0; i < num_gpus_; i++) {
    CUDA_CHECK(cudaSetDevice(i));
    gpus_.push_back(std::make_unique<GPUBackend>(cap_per_gpu_, i));
  }


  setup_nvlink();


  CUDA_CHECK(cudaSetDevice(0));
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
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
        peer_[i][j] = (can_access != 0);
        if (can_access) {
          CUDA_CHECK(cudaSetDevice(i));
          cudaDeviceEnablePeerAccess(j, 0);
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

bool DistributedGPUBackend::put_local(const BlockId &id, const uint8_t *data,
                                      size_t size, int gpu, bool is_prefix) {
  if (gpu < 0 || gpu >= num_gpus_)
    return false;

  CUDA_CHECK(cudaSetDevice(gpu));
  bool ok = gpus_[gpu]->put(id, data, size);

  if (ok) {
    BlockLocation loc;
    loc.node_id = rank_;
    loc.gpu_id = gpu;
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
    CUDA_CHECK(cudaSetDevice(i));
    if (gpus_[i]->get(id, out, out_size)) {
      return true;
    }
  }
  return false;
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
    CUDA_CHECK(cudaSetDevice(i));
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
  CUDA_CHECK(cudaSetDevice(gpu_id));
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
                                                 cfg_.num_gpus_per_node, comm_);
  dram_ = std::make_unique<DistributedDRAMBackend>(cfg_.dram_capacity, comm_);
#else
  gpu_ = std::make_unique<DistributedGPUBackend>(cfg_.gpu_capacity_per_device,
                                                 cfg_.num_gpus_per_node);
  dram_ = std::make_unique<DistributedDRAMBackend>(cfg_.dram_capacity);
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
    printf("║   Cascade Distributed Store Initialized  ║\n");
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
    dram_shadow_ok = dram_->put_local(id, store_data, store_size, is_pf);
    if (!dram_shadow_ok) {

      if (evict_dram_to_lustre(store_size)) {
        dram_shadow_ok = dram_->put_local(id, store_data, store_size, is_pf);
      }
    }
    if (!stored) stored = dram_shadow_ok;
  }


  if (gpu_stored && dram_shadow_ok && gpu_) {
    auto existing = gpu_->global_index_.get(id);
    if (existing) {
      BlockLocation loc = *existing;
      loc.dram_offset = dram_->get_offset(id);
      loc.dram_size = store_size;
      loc.has_dram_shadow = true;
      gpu_->global_index_.put(id, loc);

      mark_dirty(id, loc);
    }
  }


  if (lustre_ || agg_lustre_) {
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

  auto evicted = dram_->evict_for_space(needed_bytes, cfg_.semantic_eviction);


  for (auto &[id, data] : evicted) {

    lustre_put(id, data.data(), data.size());
    dram_evictions_++;
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

void DistributedStore::flush() {
    barrier();
    if (lustre_) lustre_->flush();
    if (agg_lustre_) agg_lustre_->flush();
    barrier();
}

}
}

