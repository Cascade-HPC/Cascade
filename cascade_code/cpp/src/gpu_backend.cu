#include "cascade.hpp"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
#include <list>
#include <mutex>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cascade {


static constexpr int NUM_STREAMS = 32;
static constexpr size_t PINNED_BUFFER_SIZE = 8 * 1024 * 1024;
static constexpr int NUM_PINNED_BUFFERS = 32;
static constexpr size_t GPU_ALIGNMENT = 256;


#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      return false;                                                            \
    }                                                                          \
  } while (0)


class GPUMemoryPool {
public:
  GPUMemoryPool(size_t total_size, int device_id) : capacity_(total_size) {
    cudaSetDevice(device_id);
    cudaError_t err = cudaMalloc(&pool_base_, total_size);
    if (err != cudaSuccess) {
      pool_base_ = nullptr;
      fprintf(stderr, "Failed to allocate GPU memory pool of %zu bytes\n",
              total_size);
    }
  }

  ~GPUMemoryPool() {
    if (pool_base_) {
      cudaFree(pool_base_);
    }
  }


  void *alloc(size_t size) {
    size_t aligned_size = (size + GPU_ALIGNMENT - 1) & ~(GPU_ALIGNMENT - 1);


    {
      std::lock_guard<std::mutex> lock(free_mutex_);
      auto best = free_list_.end();
      size_t best_waste = SIZE_MAX;

      for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
        if (it->size >= aligned_size) {
          size_t waste = it->size - aligned_size;
          if (waste < best_waste) {
            best = it;
            best_waste = waste;
            if (waste == 0)
              break;
          }
        }
      }

      if (best != free_list_.end()) {
        void *ptr =
            static_cast<uint8_t *>(pool_base_) + best->offset;
        if (best->size == aligned_size) {

          free_list_.erase(best);
        } else {

          best->offset += aligned_size;
          best->size -= aligned_size;
        }
        recycled_bytes_ += aligned_size;
        return ptr;
      }
    }


    size_t offset = current_offset_.fetch_add(aligned_size);
    if (offset + aligned_size > capacity_) {
      current_offset_ -= aligned_size;
      return nullptr;
    }
    return static_cast<uint8_t *>(pool_base_) + offset;
  }


  void free(void *ptr, size_t size) {
    if (!ptr || !pool_base_)
      return;

    size_t aligned_size = (size + GPU_ALIGNMENT - 1) & ~(GPU_ALIGNMENT - 1);
    size_t offset = static_cast<uint8_t *>(ptr) - static_cast<uint8_t *>(pool_base_);

    std::lock_guard<std::mutex> lock(free_mutex_);


    auto it = free_list_.begin();
    while (it != free_list_.end() && it->offset < offset) {
      ++it;
    }
    auto inserted = free_list_.insert(it, FreeBlock{offset, aligned_size});


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

  void reset() {
    std::lock_guard<std::mutex> lock(free_mutex_);
    current_offset_ = 0;
    free_list_.clear();
    recycled_bytes_ = 0;
  }

  size_t used() const { return current_offset_.load(); }
  size_t capacity() const { return capacity_; }
  size_t recycled_bytes() const { return recycled_bytes_; }
  size_t free_list_size() const {

    return free_list_.size();
  }
  void *get_base_ptr() const { return pool_base_; }

private:
  struct FreeBlock {
    size_t offset;
    size_t size;
  };

  void *pool_base_ = nullptr;
  size_t capacity_;
  std::atomic<size_t> current_offset_{0};

  std::list<FreeBlock> free_list_;
  std::mutex free_mutex_;
  size_t recycled_bytes_ = 0;
};


GPUBackend::GPUBackend(size_t capacity_bytes, int device_id)
    : capacity_(capacity_bytes), device_id_(device_id) {
  init_cuda();
}

GPUBackend::~GPUBackend() {
  sync_all();

  for (int i = 0; i < NUM_PINNED_BUFFERS; i++) {
    if (pinned_buffers_[i]) {
      cudaFreeHost(pinned_buffers_[i]);
      pinned_buffers_[i] = nullptr;
    }
  }
  for (int i = 0; i < NUM_STREAMS; i++) {
    if (cuda_streams_[i]) {
      cudaStreamDestroy(static_cast<cudaStream_t>(cuda_streams_[i]));
      cuda_streams_[i] = nullptr;
    }
  }
  index_.clear();
  used_ = 0;
  memory_pool_.reset();
}

bool GPUBackend::init_cuda() {
  cudaError_t err = cudaSetDevice(device_id_);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to set CUDA device %d\n", device_id_);
    return false;
  }


  for (int i = 0; i < NUM_PINNED_BUFFERS; i++) {
    err = cudaHostAlloc(&pinned_buffers_[i], PINNED_BUFFER_SIZE,
                        cudaHostAllocDefault | cudaHostAllocPortable);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate pinned buffer %d\n", i);
      return false;
    }
  }
  pinned_buffer_ = pinned_buffers_[0];
  pinned_size_ = PINNED_BUFFER_SIZE;


  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStream_t stream;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to create CUDA stream %d\n", i);
      return false;
    }
    cuda_streams_[i] = stream;
  }


  memory_pool_ = std::make_unique<GPUMemoryPool>(capacity_, device_id_);

  return true;
}

void *GPUBackend::alloc_gpu(size_t size) {
  if (memory_pool_) {
    return memory_pool_->alloc(size);
  }
  void *ptr;
  if (cudaMalloc(&ptr, size) != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

void GPUBackend::free_gpu(void *ptr, size_t size) {
  if (memory_pool_) {
    memory_pool_->free(ptr, size);
  } else {
    cudaFree(ptr);
  }
}

void GPUBackend::copy_h2d_async(void *dst, const void *src, size_t size,
                                int stream) {
  cudaMemcpyAsync(
      dst, src, size, cudaMemcpyHostToDevice,
      static_cast<cudaStream_t>(cuda_streams_[stream % NUM_STREAMS]));
}

void GPUBackend::copy_d2h_async(void *dst, const void *src, size_t size,
                                int stream) {
  cudaMemcpyAsync(
      dst, src, size, cudaMemcpyDeviceToHost,
      static_cast<cudaStream_t>(cuda_streams_[stream % NUM_STREAMS]));
}

void GPUBackend::sync_stream(int stream) {
  cudaStreamSynchronize(
      static_cast<cudaStream_t>(cuda_streams_[stream % NUM_STREAMS]));
}


bool GPUBackend::put(const BlockId &id, const uint8_t *data, size_t size) {

  if (index_.contains(id)) {
    return true;
  }


  if (used_.load() + size > capacity_) {
    return false;
  }


  void *gpu_ptr = alloc_gpu(size);
  if (!gpu_ptr) {
    return false;
  }


#ifdef _OPENMP
  int tid = omp_get_thread_num();
#else
  int tid = current_stream_.fetch_add(1);
#endif
  int stream_id = tid % NUM_STREAMS;
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_streams_[stream_id]);


  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, data);

  if (err == cudaSuccess && attrs.type == cudaMemoryTypeHost) {

    cudaMemcpyAsync(gpu_ptr, data, size, cudaMemcpyHostToDevice, stream);
  } else if (size <= PINNED_BUFFER_SIZE) {

    int buf_id = tid % NUM_PINNED_BUFFERS;
    void *pinned = pinned_buffers_[buf_id];
    memcpy(pinned, data, size);
    cudaMemcpyAsync(gpu_ptr, pinned, size, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
  } else {

    cudaMemcpy(gpu_ptr, data, size, cudaMemcpyHostToDevice);
  }


  GPUBlock block{gpu_ptr, size};
  index_.put(id, block, size);
  used_ += size;

  return true;
}


bool GPUBackend::get(const BlockId &id, uint8_t *out_data, size_t *out_size) {
  auto block_opt = index_.get(id);
  if (!block_opt) {
    return false;
  }

  GPUBlock block = *block_opt;
  *out_size = block.size;

  return read_block_to_host(block, out_data);
}


bool GPUBackend::get_and_touch(const BlockId &id, uint8_t *out_data,
                               size_t *out_size) {
  auto block_opt = index_.get_and_touch(id);
  if (!block_opt) {
    return false;
  }

  GPUBlock block = *block_opt;
  *out_size = block.size;

  return read_block_to_host(block, out_data);
}


bool GPUBackend::read_block_to_host(const GPUBlock &block, uint8_t *out_data) {
#ifdef _OPENMP
  int tid = omp_get_thread_num();
#else
  int tid = current_stream_.fetch_add(1);
#endif
  int stream_id = tid % NUM_STREAMS;
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_streams_[stream_id]);


  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, out_data);

  if (err == cudaSuccess && attrs.type == cudaMemoryTypeHost) {

    cudaMemcpyAsync(out_data, block.ptr, block.size, cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
  } else {

    int buf_id = tid % NUM_PINNED_BUFFERS;
    void *pinned = pinned_buffers_[buf_id];

    if (block.size <= PINNED_BUFFER_SIZE) {
      cudaMemcpyAsync(pinned, block.ptr, block.size, cudaMemcpyDeviceToHost,
                      stream);
      cudaStreamSynchronize(stream);
      memcpy(out_data, pinned, block.size);
    } else {

      cudaMemcpy(out_data, block.ptr, block.size, cudaMemcpyDeviceToHost);
    }
  }

  return true;
}


bool GPUBackend::remove(const BlockId &id) {
  auto block_opt = index_.get(id);
  if (!block_opt) {
    return false;
  }

  GPUBlock block = *block_opt;


  free_gpu(block.ptr, block.size);

  index_.remove(id);
  used_ -= block.size;

  return true;
}


std::optional<GPUBackend::EvictedBlock> GPUBackend::evict_lru() {

  auto victim_opt = index_.get_lru_victim();
  if (!victim_opt) {
    return std::nullopt;
  }

  auto &victim = *victim_opt;
  GPUBlock block = victim.value;


  EvictedBlock evicted;
  evicted.id = victim.key;
  evicted.size = block.size;
  evicted.data.resize(block.size);

  if (!read_block_to_host(block, evicted.data.data())) {
    return std::nullopt;
  }


  free_gpu(block.ptr, block.size);
  index_.remove(victim.key);
  used_ -= block.size;

  return evicted;
}


std::optional<GPUBackend::EvictedBlock> GPUBackend::evict_lru_semantic(
    const std::function<bool(const BlockId &)> &is_prefix) {
  if (!is_prefix) {
    return evict_lru();
  }

  auto victim_opt = index_.get_lru_victim_skip(is_prefix);
  if (!victim_opt) {

    return evict_lru();
  }

  auto &victim = *victim_opt;
  GPUBlock block = victim.value;

  EvictedBlock evicted;
  evicted.id = victim.key;
  evicted.size = block.size;
  evicted.data.resize(block.size);

  if (!read_block_to_host(block, evicted.data.data())) {
    return std::nullopt;
  }

  free_gpu(block.ptr, block.size);
  index_.remove(victim.key);
  used_ -= block.size;

  return evicted;
}


std::vector<GPUBackend::EvictedBlock> GPUBackend::evict_for_space(
    size_t needed_bytes,
    const std::function<bool(const BlockId &)> &is_prefix) {
  std::vector<EvictedBlock> evicted_blocks;
  size_t freed = 0;

  while (freed < needed_bytes) {
    std::optional<EvictedBlock> evicted;
    if (is_prefix) {
      evicted = evict_lru_semantic(is_prefix);
    } else {
      evicted = evict_lru();
    }

    if (!evicted) {
      break;
    }

    freed += evicted->size;
    evicted_blocks.push_back(std::move(*evicted));
  }

  return evicted_blocks;
}


bool GPUBackend::contains(const BlockId &id) const {
  return index_.contains(id);
}

void GPUBackend::clear() {
  sync_all();
  index_.clear();
  used_ = 0;
  if (memory_pool_) {
    memory_pool_->reset();
  }
}

void GPUBackend::sync_all() {
  for (int i = 0; i < NUM_STREAMS; i++) {
    if (cuda_streams_[i]) {
      cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_streams_[i]));
    }
  }
}

uint8_t *GPUBackend::get_base_ptr() const {
  if (memory_pool_)
    return static_cast<uint8_t *>(memory_pool_->get_base_ptr());
  return nullptr;
}

size_t GPUBackend::get_offset(const BlockId &id) const {
  auto block_opt = index_.get(id);
  if (!block_opt)
    return 0;
  if (!memory_pool_)
    return 0;
  return static_cast<uint8_t *>(block_opt->ptr) -
         static_cast<uint8_t *>(memory_pool_->get_base_ptr());
}

}

