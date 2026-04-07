/**
 * Cascade Core Implementation 
 *
 * - ShardedIndex: Per-shard LRU doubly-linked list + hash map
 * - ShmBackend: mmap + SSE2 streaming stores + free-list allocator
 * - LustreBackend: O_DIRECT aligned I/O for true cold storage
 * - AggregatedLustreBackend: Multi-block append files 
 * - KVCompressor: FP16→INT8 quantization
 * - PrefetchPipeline: Async Lustre→SHM background loading
 * - CascadeStore: 3-tier orchestration with LRU eviction,
 *                 tier promotion/demotion, semantic eviction,
 *                 and OpenMP-parallelized batch API
 */

#include "cascade.hpp"

#include <dirent.h>
#include <emmintrin.h> 
#include <fcntl.h>
#include <openssl/sha.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib> 
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cascade {

static constexpr size_t DIRECT_IO_ALIGNMENT = 4096;
namespace fs = std::filesystem;


BlockId compute_block_id(const uint8_t *data, size_t size) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(data, size, hash);

  char hex[33];
  for (int i = 0; i < 16; i++) {
    snprintf(hex + i * 2, 3, "%02x", hash[i]);
  }
  hex[32] = '\0';
  return std::string(hex);
}


template <typename V> ShardedIndex<V>::ShardedIndex() {}

template <typename V> ShardedIndex<V>::~ShardedIndex() { clear(); }

template <typename V>
bool ShardedIndex<V>::put(const BlockId &key, V value, size_t size) {
  size_t shard_id = get_shard_id(key);
  auto &shard = shards_[shard_id];

  std::unique_lock lock(shard.mutex);

  auto it = shard.index.find(key);
  if (it != shard.index.end()) {
    shard.lru_list.splice(shard.lru_list.begin(), shard.lru_list, it->second);
    it->second->value = value;
    it->second->size = size;
    return true;
  }

  shard.lru_list.emplace_front(LRUNode{key, value, size});
  shard.index[key] = shard.lru_list.begin();
  shard.total_size += size;

  return true;
}

template <typename V>
std::optional<V> ShardedIndex<V>::get(const BlockId &key) const {
  size_t shard_id = get_shard_id(key);
  const auto &shard = shards_[shard_id];

  std::shared_lock lock(shard.mutex);

  auto it = shard.index.find(key);
  if (it == shard.index.end()) {
    return std::nullopt;
  }
  return it->second->value;
}

template <typename V>
std::optional<V> ShardedIndex<V>::get_and_touch(const BlockId &key) {
  size_t shard_id = get_shard_id(key);
  auto &shard = shards_[shard_id];

  std::unique_lock lock(shard.mutex);

  auto it = shard.index.find(key);
  if (it == shard.index.end()) {
    return std::nullopt;
  }
  shard.lru_list.splice(shard.lru_list.begin(), shard.lru_list, it->second);
  return it->second->value;
}

template <typename V> bool ShardedIndex<V>::remove(const BlockId &key) {
  size_t shard_id = get_shard_id(key);
  auto &shard = shards_[shard_id];

  std::unique_lock lock(shard.mutex);

  auto it = shard.index.find(key);
  if (it == shard.index.end()) {
    return false;
  }

  size_t size = it->second->size;
  shard.lru_list.erase(it->second);
  shard.index.erase(it);
  shard.total_size -= size;

  return true;
}

template <typename V> bool ShardedIndex<V>::contains(const BlockId &key) const {
  size_t shard_id = get_shard_id(key);
  const auto &shard = shards_[shard_id];

  std::shared_lock lock(shard.mutex);
  return shard.index.find(key) != shard.index.end();
}


template <typename V>
std::optional<typename ShardedIndex<V>::LRUEntry>
ShardedIndex<V>::get_lru_victim() const {
  size_t best_shard = 0;
  size_t best_size = 0;

  for (size_t i = 0; i < NUM_SHARDS; i++) {
    size_t s = shards_[i].total_size.load();
    if (s > best_size) {
      best_size = s;
      best_shard = i;
    }
  }

  auto &shard = shards_[best_shard];
  std::shared_lock lock(shard.mutex);

  if (shard.lru_list.empty()) {
    return std::nullopt;
  }

  const auto &back = shard.lru_list.back();
  return LRUEntry{back.key, back.value, back.size};
}

template <typename V>
std::optional<typename ShardedIndex<V>::LRUEntry>
ShardedIndex<V>::get_lru_victim_skip(
    const std::function<bool(const BlockId &)> &should_skip) const {
  std::vector<size_t> shard_order(NUM_SHARDS);
  for (size_t i = 0; i < NUM_SHARDS; i++)
    shard_order[i] = i;

  std::sort(shard_order.begin(), shard_order.end(),
            [this](size_t a, size_t b) {
              return shards_[a].total_size.load() >
                     shards_[b].total_size.load();
            });

  for (size_t idx : shard_order) {
    auto &shard = shards_[idx];
    std::shared_lock lock(shard.mutex);

    if (shard.lru_list.empty())
      continue;

    for (auto rit = shard.lru_list.rbegin(); rit != shard.lru_list.rend();
         ++rit) {
      if (!should_skip(rit->key)) {
        return LRUEntry{rit->key, rit->value, rit->size};
      }
    }
  }

  return std::nullopt; 
}

template <typename V> size_t ShardedIndex<V>::total_size() const {
  size_t total = 0;
  for (const auto &shard : shards_) {
    total += shard.total_size.load();
  }
  return total;
}

template <typename V> size_t ShardedIndex<V>::total_count() const {
  size_t total = 0;
  for (const auto &shard : shards_) {
    std::shared_lock lock(shard.mutex);
    total += shard.index.size();
  }
  return total;
}

template <typename V> void ShardedIndex<V>::clear() {
  for (auto &shard : shards_) {
    std::unique_lock lock(shard.mutex);
    shard.lru_list.clear();
    shard.index.clear();
    shard.total_size = 0;
  }
}

template class ShardedIndex<bool>;

struct GPUBlockFwd {
  void *ptr;
  size_t size;
};
struct ShmBlockFwd {
  size_t offset;
  size_t size;
};

template class ShardedIndex<GPUBackend::GPUBlock>;
template class ShardedIndex<ShmBackend::ShmBlock>;


ShmBackend::ShmBackend(size_t capacity_bytes, const std::string &path)
    : capacity_(capacity_bytes), path_(path) {

  fs::create_directories(path_);

  std::string file_path = path_ + "/data.mmap";

  int fd = open(file_path.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd < 0) {
    perror("Failed to open mmap file");
    return;
  }

  if (ftruncate(fd, capacity_) < 0) {
    perror("Failed to extend mmap file");
    close(fd);
    return;
  }

  mmap_base_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, fd, 0);

  if (mmap_base_ == MAP_FAILED) {
    mmap_base_ =
        mmap(nullptr, capacity_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  }

  if (mmap_base_ == MAP_FAILED) {
    perror("mmap failed");
    mmap_base_ = nullptr;
    close(fd);
    return;
  }

  mmap_size_ = capacity_;
  close(fd);

  madvise(mmap_base_, mmap_size_, MADV_WILLNEED);
  madvise(mmap_base_, mmap_size_, MADV_HUGEPAGE);
}

ShmBackend::~ShmBackend() {
  if (mmap_base_ && mmap_base_ != MAP_FAILED) {
    msync(mmap_base_, mmap_size_, MS_SYNC);
    munmap(mmap_base_, mmap_size_);
  }
}

size_t ShmBackend::allocate(size_t size) {
  size_t aligned_size = (size + 63) & ~63ULL; 

  {
    std::lock_guard<std::mutex> lock(free_list_mutex_);
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
      size_t offset = best->offset;
      if (best->size == aligned_size) {
        free_list_.erase(best);
      } else {
        best->offset += aligned_size;
        best->size -= aligned_size;
      }
      return offset;
    }
  }

  size_t offset = write_offset_.fetch_add(aligned_size);
  if (offset + aligned_size > capacity_) {
    write_offset_ -= aligned_size;
    return SIZE_MAX;
  }
  return offset;
}

void ShmBackend::deallocate(size_t offset, size_t size) {
  size_t aligned_size = (size + 63) & ~63ULL;

  std::lock_guard<std::mutex> lock(free_list_mutex_);

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

void ShmBackend::write_block(uint8_t *dst, const uint8_t *src, size_t size) {
  if (size >= 4096) {
    size_t aligned_size = size & ~63ULL;
    const __m128i *src_vec = reinterpret_cast<const __m128i *>(src);
    __m128i *dst_vec = reinterpret_cast<__m128i *>(dst);

    for (size_t i = 0; i < aligned_size; i += 64) {
      __m128i v0 = _mm_loadu_si128(src_vec++);
      __m128i v1 = _mm_loadu_si128(src_vec++);
      __m128i v2 = _mm_loadu_si128(src_vec++);
      __m128i v3 = _mm_loadu_si128(src_vec++);
      _mm_stream_si128(dst_vec++, v0);
      _mm_stream_si128(dst_vec++, v1);
      _mm_stream_si128(dst_vec++, v2);
      _mm_stream_si128(dst_vec++, v3);
    }
    _mm_sfence();

    if (size > aligned_size) {
      memcpy(dst + aligned_size, src + aligned_size, size - aligned_size);
    }
  } else {
    memcpy(dst, src, size);
  }
}

void ShmBackend::read_block(const ShmBlock &block, uint8_t *out_data) {
  const uint8_t *src = static_cast<const uint8_t *>(mmap_base_) + block.offset;

  if (block.size >= 4096) {
    size_t size = block.size;
    size_t aligned_size = size & ~63ULL;

    for (size_t i = 0; i < aligned_size; i += 512) {
      _mm_prefetch(reinterpret_cast<const char *>(src + i + 512), _MM_HINT_T0);
    }

    const __m128i *src_vec = reinterpret_cast<const __m128i *>(src);
    __m128i *dst_vec = reinterpret_cast<__m128i *>(out_data);

    for (size_t i = 0; i < aligned_size; i += 64) {
      __m128i v0 = _mm_load_si128(src_vec++);
      __m128i v1 = _mm_load_si128(src_vec++);
      __m128i v2 = _mm_load_si128(src_vec++);
      __m128i v3 = _mm_load_si128(src_vec++);
      _mm_store_si128(dst_vec++, v0);
      _mm_store_si128(dst_vec++, v1);
      _mm_store_si128(dst_vec++, v2);
      _mm_store_si128(dst_vec++, v3);
    }

    if (size > aligned_size) {
      memcpy(out_data + aligned_size, src + aligned_size, size - aligned_size);
    }
  } else {
    memcpy(out_data, src, block.size);
  }
}

bool ShmBackend::put(const BlockId &id, const uint8_t *data, size_t size) {
  if (!mmap_base_)
    return false;

  if (index_.contains(id)) {
    return true;
  }


  size_t offset = allocate(size);
  if (offset == SIZE_MAX) {
    return false;
  }

  uint8_t *dst = static_cast<uint8_t *>(mmap_base_) + offset;
  write_block(dst, data, size);

  ShmBlock block{offset, size};
  index_.put(id, block, size);
  used_ += size;

  return true;
}

bool ShmBackend::get(const BlockId &id, uint8_t *out_data, size_t *out_size) {
  if (!mmap_base_)
    return false;

  auto block_opt = index_.get(id);
  if (!block_opt) {
    return false;
  }

  ShmBlock block = *block_opt;
  *out_size = block.size;

  read_block(block, out_data);
  return true;
}

bool ShmBackend::get_and_touch(const BlockId &id, uint8_t *out_data,
                               size_t *out_size) {
  if (!mmap_base_)
    return false;

  auto block_opt = index_.get_and_touch(id);
  if (!block_opt) {
    return false;
  }

  ShmBlock block = *block_opt;
  *out_size = block.size;

  read_block(block, out_data);
  return true;
}

bool ShmBackend::remove(const BlockId &id) {
  auto block_opt = index_.get(id);
  if (!block_opt) {
    return false;
  }

  ShmBlock block = *block_opt;

  deallocate(block.offset, block.size);

  index_.remove(id);
  used_ -= block.size;

  return true;
}

std::optional<ShmBackend::EvictedBlock> ShmBackend::evict_lru() {
  auto victim_opt = index_.get_lru_victim();
  if (!victim_opt) {
    return std::nullopt;
  }

  auto &victim = *victim_opt;
  ShmBlock block = victim.value;

  EvictedBlock evicted;
  evicted.id = victim.key;
  evicted.size = block.size;
  evicted.data.resize(block.size);
  read_block(block, evicted.data.data());

  deallocate(block.offset, block.size);
  index_.remove(victim.key);
  used_ -= block.size;

  return evicted;
}

std::optional<ShmBackend::EvictedBlock> ShmBackend::evict_lru_semantic(
    const std::function<bool(const BlockId &)> &is_prefix) {
  if (!is_prefix) {
    return evict_lru();
  }

  auto victim_opt = index_.get_lru_victim_skip(is_prefix);
  if (!victim_opt) {
    return evict_lru();
  }

  auto &victim = *victim_opt;
  ShmBlock block = victim.value;

  EvictedBlock evicted;
  evicted.id = victim.key;
  evicted.size = block.size;
  evicted.data.resize(block.size);
  read_block(block, evicted.data.data());

  deallocate(block.offset, block.size);
  index_.remove(victim.key);
  used_ -= block.size;

  return evicted;
}

std::vector<ShmBackend::EvictedBlock> ShmBackend::evict_for_space(
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

    if (!evicted)
      break;

    freed += evicted->size;
    evicted_blocks.push_back(std::move(*evicted));
  }

  return evicted_blocks;
}

bool ShmBackend::contains(const BlockId &id) const {
  return index_.contains(id);
}

void ShmBackend::clear() {
  index_.clear();
  write_offset_ = 0;
  used_ = 0;
  {
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    free_list_.clear();
  }
}


LustreBackend::LustreBackend(const std::string &path, size_t stripe_size,
                             int stripe_count)
    : base_path_(path), stripe_size_(stripe_size), stripe_count_(stripe_count) {

  fs::create_directories(path);

  std::string cmd = "lfs setstripe -S " + std::to_string(stripe_size) + " -c " +
                    std::to_string(stripe_count) + " " + path + " 2>/dev/null";
  system(cmd.c_str());
}

LustreBackend::~LustreBackend() { flush(); }

std::string LustreBackend::block_path(const BlockId &id) const {
  std::string subdir =
      base_path_ + "/" + id.substr(0, 2) + "/" + id.substr(2, 2);
  fs::create_directories(subdir);
  return subdir + "/" + id + ".kv";
}

bool LustreBackend::put(const BlockId &id, const uint8_t *data, size_t size) {
  std::string path = block_path(id);

  if (fs::exists(path)) {
    return true;
  }


  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
  if (fd < 0) {
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      return false;
    }
    ssize_t written = write(fd, data, size);
    close(fd);
    return written == static_cast<ssize_t>(size);
  }

  size_t aligned_size = (size + DIRECT_IO_ALIGNMENT - 1) & ~(DIRECT_IO_ALIGNMENT - 1);
  void *aligned_buf = nullptr;
  if (posix_memalign(&aligned_buf, DIRECT_IO_ALIGNMENT, aligned_size) != 0) {
    close(fd);
    return false;
  }

  memcpy(aligned_buf, data, size);
  if (aligned_size > size) {
    memset(static_cast<uint8_t *>(aligned_buf) + size, 0, aligned_size - size);
  }

  ssize_t written = write(fd, aligned_buf, aligned_size);
  close(fd);
  ::free(aligned_buf);

  if (written == static_cast<ssize_t>(aligned_size) && aligned_size != size) {
    truncate(path.c_str(), size);
  }

  return written == static_cast<ssize_t>(aligned_size);
}

bool LustreBackend::get(const BlockId &id, uint8_t *out_data,
                        size_t *out_size) {
  std::string path = block_path(id);

  if (!fs::exists(path)) {
    return false;
  }

  struct stat st;
  if (stat(path.c_str(), &st) < 0) {
    return false;
  }
  size_t file_size = st.st_size;

  int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      return false;
    }
    *out_size = file_size;
    ssize_t bytes_read = read(fd, out_data, file_size);
    close(fd);
    return bytes_read == static_cast<ssize_t>(file_size);
  }

  size_t aligned_size = (file_size + DIRECT_IO_ALIGNMENT - 1) & ~(DIRECT_IO_ALIGNMENT - 1);
  void *aligned_buf = nullptr;
  if (posix_memalign(&aligned_buf, DIRECT_IO_ALIGNMENT, aligned_size) != 0) {
    close(fd);
    return false;
  }

  ssize_t bytes_read = read(fd, aligned_buf, aligned_size);
  close(fd);

  if (bytes_read < static_cast<ssize_t>(file_size)) {
    ::free(aligned_buf);
    return false;
  }

  memcpy(out_data, aligned_buf, file_size);
  ::free(aligned_buf);
  *out_size = file_size;

  return true;
}

bool LustreBackend::remove(const BlockId &id) {
  std::string path = block_path(id);
  return fs::remove(path);
}

bool LustreBackend::contains(const BlockId &id) const {
  return fs::exists(block_path(id));
}

void LustreBackend::flush() { sync(); }


AggregatedLustreBackend::AggregatedLustreBackend(const std::string &path,
                                                 size_t max_file_size,
                                                 size_t stripe_size,
                                                 int stripe_count)
    : base_path_(path), max_file_size_(max_file_size),
      stripe_size_(stripe_size), stripe_count_(stripe_count) {
  fs::create_directories(base_path_);


  std::string cmd = "lfs setstripe -S " + std::to_string(stripe_size) +
                    " -c " + std::to_string(stripe_count) + " " + path +
                    " 2>/dev/null";
  system(cmd.c_str());

  open_new_file();
}

AggregatedLustreBackend::~AggregatedLustreBackend() {
  flush();
  if (current_fd_ >= 0) {
    close(current_fd_);
  }
}

std::string AggregatedLustreBackend::file_path(uint32_t file_id) const {
  return base_path_ + "/agg_" + std::to_string(file_id) + ".dat";
}

bool AggregatedLustreBackend::open_new_file() {
  if (current_fd_ >= 0) {
    close(current_fd_);
  }
  current_file_id_++;
  current_offset_ = 0;

  std::string path = file_path(current_file_id_);
  current_fd_ = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  return current_fd_ >= 0;
}

bool AggregatedLustreBackend::put(const BlockId &id, const uint8_t *data,
                                  size_t size) {
  std::lock_guard<std::mutex> lock(write_mutex_);

  {
    std::shared_lock<std::shared_mutex> rlock(index_mutex_);
    if (index_.count(id) > 0) return true;
  }

  if (current_offset_ + size + sizeof(uint64_t) > max_file_size_) {
    if (!open_new_file()) return false;
  }

  if (current_fd_ < 0) {
    if (!open_new_file()) return false;
  }

  uint64_t block_size = size;
  ssize_t w1 = write(current_fd_, &block_size, sizeof(uint64_t));
  ssize_t w2 = write(current_fd_, data, size);

  if (w1 != sizeof(uint64_t) || w2 != static_cast<ssize_t>(size)) {
    return false;
  }

  BlockLocation loc;
  loc.file_id = current_file_id_;
  loc.offset = current_offset_;
  loc.size = size;

  {
    std::unique_lock<std::shared_mutex> wlock(index_mutex_);
    index_[id] = loc;
  }

  current_offset_ += sizeof(uint64_t) + size;
  return true;
}

bool AggregatedLustreBackend::get(const BlockId &id, uint8_t *out_data,
                                  size_t *out_size) {
  BlockLocation loc;
  {
    std::shared_lock<std::shared_mutex> rlock(index_mutex_);
    auto it = index_.find(id);
    if (it == index_.end()) return false;
    loc = it->second;
  }

  std::string path = file_path(loc.file_id);
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;

  lseek(fd, loc.offset + sizeof(uint64_t), SEEK_SET);
  ssize_t bytes_read = read(fd, out_data, loc.size);
  close(fd);

  if (bytes_read != static_cast<ssize_t>(loc.size)) return false;

  *out_size = loc.size;
  return true;
}

bool AggregatedLustreBackend::contains(const BlockId &id) const {
  std::shared_lock<std::shared_mutex> rlock(index_mutex_);
  return index_.count(id) > 0;
}

std::vector<BlockId> AggregatedLustreBackend::list_blocks() const {
  std::shared_lock<std::shared_mutex> rlock(index_mutex_);
  std::vector<BlockId> result;
  result.reserve(index_.size());
  for (auto &[id, _] : index_) {
    result.push_back(id);
  }
  return result;
}

void AggregatedLustreBackend::flush() {
  if (current_fd_ >= 0) {
    fsync(current_fd_);
  }
}

std::vector<uint8_t> KVCompressor::compress(const uint8_t *data, size_t size,
                                             CompressionMeta &meta) {
  size_t num_elements = size / 2;
  const int16_t *fp16_data = reinterpret_cast<const int16_t *>(data);

  int16_t min_val = fp16_data[0];
  int16_t max_val = fp16_data[0];
  for (size_t i = 1; i < num_elements; i++) {
    if (fp16_data[i] < min_val) min_val = fp16_data[i];
    if (fp16_data[i] > max_val) max_val = fp16_data[i];
  }

  float range = static_cast<float>(max_val) - static_cast<float>(min_val);
  meta.scale = (range > 0.0f) ? range / 255.0f : 1.0f;
  meta.zero_point = static_cast<int8_t>(min_val / meta.scale);

  std::vector<uint8_t> compressed(sizeof(CompressionMeta) + num_elements);

  memcpy(compressed.data(), &meta, sizeof(CompressionMeta));

  int8_t *out = reinterpret_cast<int8_t *>(compressed.data() + sizeof(CompressionMeta));
  float inv_scale = 1.0f / meta.scale;
  float min_f = static_cast<float>(min_val);

  for (size_t i = 0; i < num_elements; i++) {
    float val = static_cast<float>(fp16_data[i]);
    int quantized = static_cast<int>((val - min_f) * inv_scale + 0.5f);
    out[i] = static_cast<int8_t>(std::max(0, std::min(255, quantized)) - 128);
  }

  return compressed;
}

bool KVCompressor::decompress(const uint8_t *compressed, size_t compressed_size,
                               const CompressionMeta &meta, uint8_t *out_data,
                               size_t original_size) {
  size_t num_elements = original_size / 2;

  if (compressed_size < sizeof(CompressionMeta) + num_elements) {
    return false;
  }

  const int8_t *quant = reinterpret_cast<const int8_t *>(
      compressed + sizeof(CompressionMeta));
  int16_t *fp16_out = reinterpret_cast<int16_t *>(out_data);

  float min_f = meta.zero_point * meta.scale;

  for (size_t i = 0; i < num_elements; i++) {
    float dequantized = (static_cast<float>(quant[i]) + 128.0f) * meta.scale + min_f;
    fp16_out[i] = static_cast<int16_t>(dequantized);
  }

  return true;
}


PrefetchPipeline::PrefetchPipeline(LustreBackend *lustre,
                                   AggregatedLustreBackend *agg_lustre,
                                   ShmBackend *shm, int num_threads,
                                   size_t queue_size)
    : lustre_(lustre), agg_lustre_(agg_lustre), shm_(shm),
      max_queue_size_(queue_size) {
  for (int i = 0; i < num_threads; i++) {
    workers_.emplace_back(&PrefetchPipeline::worker_loop, this);
  }
}

PrefetchPipeline::~PrefetchPipeline() { stop(); }

void PrefetchPipeline::stop() {
  running_ = false;
  queue_cv_.notify_all();
  for (auto &t : workers_) {
    if (t.joinable())
      t.join();
  }
  workers_.clear();
}

void PrefetchPipeline::submit(const BlockId &id, size_t expected_size) {
  if (shm_ && shm_->contains(id)) {
    skipped_++;
    return;
  }

  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (queue_.size() >= max_queue_size_) {
    return; 
  }
  queue_.push({id, expected_size});
  submitted_++;
  queue_cv_.notify_one();
}

void PrefetchPipeline::record_access(const BlockId &id) {
  std::lock_guard<std::mutex> lock(freq_mutex_);
  access_freq_[id]++;
}

PrefetchPipeline::Stats PrefetchPipeline::get_stats() const {
  return {submitted_.load(), completed_.load(), skipped_.load()};
}

void PrefetchPipeline::worker_loop() {
  std::vector<uint8_t> buffer(512ULL * 1024 * 1024);

  while (running_) {
    PrefetchRequest req;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
      if (!running_ && queue_.empty()) break;
      if (queue_.empty()) continue;
      req = std::move(queue_.front());
      queue_.pop();
    }

    if (shm_->contains(req.id)) {
      skipped_++;
      continue;
    }

    size_t read_size = 0;
    bool found = false;

    if (agg_lustre_ && agg_lustre_->contains(req.id)) {
      found = agg_lustre_->get(req.id, buffer.data(), &read_size);
    } else if (lustre_) {
      found = lustre_->get(req.id, buffer.data(), &read_size);
    }

    if (found && read_size > 0) {
      if (shm_->used_bytes() + read_size <= shm_->capacity()) {
        shm_->put(req.id, buffer.data(), read_size);
        completed_++;
      }
    }
  }
}


CascadeStore::CascadeStore(const CascadeConfig &config) : config_(config) {
  if (config.use_gpu && config.gpu_capacity_bytes > 0) {
    gpu_ = std::make_unique<GPUBackend>(config.gpu_capacity_bytes,
                                        config.gpu_device_id);
  }

  if (config.shm_capacity_bytes > 0) {
    shm_ = std::make_unique<ShmBackend>(config.shm_capacity_bytes,
                                        config.shm_path);
  }

  if (!config.lustre_path.empty()) {
    if (config.aggregated_lustre) {
      agg_lustre_ = std::make_unique<AggregatedLustreBackend>(
          config.lustre_path, config.agg_file_size,
          config.lustre_stripe_size, config.lustre_stripe_count);
    } else {
      lustre_ = std::make_unique<LustreBackend>(config.lustre_path,
                                                config.lustre_stripe_size,
                                                config.lustre_stripe_count);
    }
  }


  if (config.prefetch_enabled && shm_) {
    prefetcher_ = std::make_unique<PrefetchPipeline>(
        lustre_.get(), agg_lustre_.get(), shm_.get(),
        config.prefetch_threads, config.prefetch_queue_size);
  }
}

CascadeStore::~CascadeStore() { flush(); }

bool CascadeStore::is_prefix_block(const BlockId &id) const {
  return prefix_blocks_.contains(id);
}


bool CascadeStore::evict_gpu_to_shm(size_t needed_bytes) {
  if (!gpu_ || !shm_)
    return false;

  auto is_prefix = config_.semantic_eviction
                       ? std::function<bool(const BlockId &)>(
                             [this](const BlockId &id) { return is_prefix_block(id); })
                       : std::function<bool(const BlockId &)>(nullptr);

  auto evicted = gpu_->evict_for_space(needed_bytes, is_prefix);

  for (auto &block : evicted) {
    if (shm_->used_bytes() + block.size > shm_->capacity()) {
      evict_shm_to_lustre(block.size);
    }
    shm_->put(block.id, block.data.data(), block.size);
    gpu_evictions_++;
  }

  return !evicted.empty();
}

bool CascadeStore::evict_shm_to_lustre(size_t needed_bytes) {
  if (!shm_)
    return false;

  auto is_prefix = config_.semantic_eviction
                       ? std::function<bool(const BlockId &)>(
                             [this](const BlockId &id) { return is_prefix_block(id); })
                       : std::function<bool(const BlockId &)>(nullptr);

  auto evicted = shm_->evict_for_space(needed_bytes, is_prefix);

  for (auto &block : evicted) {
    lustre_put(block.id, block.data.data(), block.size);
    shm_evictions_++;
  }

  return !evicted.empty();
}

bool CascadeStore::promote_to_gpu(const BlockId &id, const uint8_t *data,
                                  size_t size, bool is_prefix) {
  if (!gpu_)
    return false;

  if (gpu_->used_bytes() + size > gpu_->capacity()) {
    evict_gpu_to_shm(size);
  }

  bool ok = gpu_->put(id, data, size);
  if (ok) {
    promotions_to_gpu_++;
    if (shm_) {
      shm_->remove(id);
    }
  }
  return ok;
}

bool CascadeStore::promote_to_shm(const BlockId &id, const uint8_t *data,
                                  size_t size) {
  if (!shm_)
    return false;

  if (shm_->used_bytes() + size > shm_->capacity()) {
    evict_shm_to_lustre(size);
  }

  bool ok = shm_->put(id, data, size);
  if (ok) {
    promotions_to_shm_++;
  }
  return ok;
}

bool CascadeStore::put(const BlockId &id, const uint8_t *data, size_t size,
                       bool is_prefix) {

  if (config_.dedup_enabled && known_blocks_.contains(id)) {
    dedup_hits_++;
    return true;
  }

  std::vector<uint8_t> compressed_buf;
  const uint8_t *store_data = data;
  size_t store_size = size;

  if (config_.kv_compression && size >= 64) {
    CompressionMeta meta;
    compressed_buf = KVCompressor::compress(data, size, meta);
    store_data = compressed_buf.data();
    store_size = compressed_buf.size();
    compression_savings_ += (size - store_size);
  }

  bool stored = false;


  if (gpu_) {
    if (gpu_->used_bytes() + size <= gpu_->capacity()) {
      stored = gpu_->put(id, data, size);
    } else {
      if (evict_gpu_to_shm(size)) {
        stored = gpu_->put(id, data, size);
      }
    }
  }


  if (!stored && shm_) {
    if (shm_->used_bytes() + store_size <= shm_->capacity()) {
      stored = shm_->put(id, store_data, store_size);
      if (stored) shm_puts_++;
    } else {
      if (evict_shm_to_lustre(store_size)) {
        stored = shm_->put(id, store_data, store_size);
        if (stored) shm_puts_++;
      }
    }
  }


  if (!stored) {
    stored = lustre_put(id, store_data, store_size);
    if (stored) lustre_puts_++;
  }


  if (stored && config_.dedup_enabled) {
    known_blocks_.put(id, true);
    if (is_prefix) {
      prefix_blocks_.put(id, true);
    }
  }

  return stored;
}


bool CascadeStore::get(const BlockId &id, uint8_t *out_data, size_t *out_size) {
  if (gpu_ && gpu_->get_and_touch(id, out_data, out_size)) {
    gpu_hits_++;
    if (prefetcher_) prefetcher_->record_access(id);
    return true;
  }

  if (shm_ && shm_->get_and_touch(id, out_data, out_size)) {
    shm_hits_++;
    if (prefetcher_) prefetcher_->record_access(id);

    if (config_.kv_compression && *out_size >= sizeof(CompressionMeta)) {
      CompressionMeta meta;
      memcpy(&meta, out_data, sizeof(CompressionMeta));
      size_t orig_size = KVCompressor::original_size(*out_size);
      std::vector<uint8_t> tmp(out_data, out_data + *out_size);
      KVCompressor::decompress(tmp.data(), *out_size, meta, out_data, orig_size);
      *out_size = orig_size;
    }

    if (config_.promotion_enabled && gpu_) {
      bool is_pf = is_prefix_block(id);
      promote_to_gpu(id, out_data, *out_size, is_pf);
    }

    return true;
  }

  bool lustre_found = lustre_get(id, out_data, out_size);
  if (lustre_found) {
    lustre_hits_++;
    if (prefetcher_) prefetcher_->record_access(id);

    if (config_.kv_compression && *out_size >= sizeof(CompressionMeta)) {
      CompressionMeta meta;
      memcpy(&meta, out_data, sizeof(CompressionMeta));
      size_t orig_size = KVCompressor::original_size(*out_size);
      std::vector<uint8_t> tmp(out_data, out_data + *out_size);
      KVCompressor::decompress(tmp.data(), *out_size, meta, out_data, orig_size);
      *out_size = orig_size;
    }

    if (config_.promotion_enabled && shm_) {
      promote_to_shm(id, out_data, *out_size);
    }

    return true;
  }

  misses_++;
  return false;
}


bool CascadeStore::contains(const BlockId &id) const {
  if (gpu_ && gpu_->contains(id))
    return true;
  if (shm_ && shm_->contains(id))
    return true;
  if (lustre_ && lustre_->contains(id))
    return true;
  return false;
}


size_t CascadeStore::put_batch(const std::vector<BlockId> &ids,
                               const std::vector<const uint8_t *> &data,
                               const std::vector<size_t> &sizes) {
  std::atomic<size_t> count{0};
  size_t n = ids.size();

#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic) if(n > 4)
#endif
  for (size_t i = 0; i < n; i++) {
    if (put(ids[i], data[i], sizes[i])) {
      count.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return count.load();
}

size_t CascadeStore::get_batch(const std::vector<BlockId> &ids,
                               std::vector<uint8_t *> &out_data,
                               std::vector<size_t> &out_sizes) {
  std::atomic<size_t> count{0};
  size_t n = ids.size();

#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic) if(n > 4)
#endif
  for (size_t i = 0; i < n; i++) {
    if (get(ids[i], out_data[i], &out_sizes[i])) {
      count.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return count;
}


CascadeStore::Stats CascadeStore::get_stats() const {
  Stats stats;
  stats.gpu_used = gpu_ ? gpu_->used_bytes() : 0;
  stats.shm_used = shm_ ? shm_->used_bytes() : 0;
  stats.gpu_hits = gpu_hits_.load();
  stats.shm_hits = shm_hits_.load();
  stats.lustre_hits = lustre_hits_.load();
  stats.misses = misses_.load();
  stats.dedup_hits = dedup_hits_.load();
  stats.gpu_evictions = gpu_evictions_.load();
  stats.shm_evictions = shm_evictions_.load();
  stats.promotions_to_gpu = promotions_to_gpu_.load();
  stats.promotions_to_shm = promotions_to_shm_.load();
  stats.shm_puts = shm_puts_.load();
  stats.lustre_puts = lustre_puts_.load();
  stats.compression_savings_bytes = compression_savings_.load();

  stats.prefetch_completed = prefetcher_ ? prefetcher_->get_stats().completed : 0;
  return stats;
}

void CascadeStore::clear() {
  if (prefetcher_) prefetcher_->stop();
  if (gpu_) gpu_->clear();
  if (shm_) shm_->clear();
  known_blocks_.clear();
  prefix_blocks_.clear();
  gpu_hits_ = 0;
  shm_hits_ = 0;
  lustre_hits_ = 0;
  misses_ = 0;
  dedup_hits_ = 0;
  gpu_evictions_ = 0;
  shm_evictions_ = 0;
  promotions_to_gpu_ = 0;
  promotions_to_shm_ = 0;
  shm_puts_ = 0;
  lustre_puts_ = 0;
  compression_savings_ = 0;
}

void CascadeStore::flush() {
  if (lustre_) lustre_->flush();
  if (agg_lustre_) agg_lustre_->flush();
}


bool CascadeStore::lustre_put(const BlockId &id, const uint8_t *data,
                               size_t size) {
  if (agg_lustre_) return agg_lustre_->put(id, data, size);
  if (lustre_) return lustre_->put(id, data, size);
  return false;
}

bool CascadeStore::lustre_get(const BlockId &id, uint8_t *out_data,
                               size_t *out_size) {
  if (agg_lustre_) return agg_lustre_->get(id, out_data, out_size);
  if (lustre_) return lustre_->get(id, out_data, out_size);
  return false;
}

bool CascadeStore::lustre_contains(const BlockId &id) const {
  if (agg_lustre_) return agg_lustre_->contains(id);
  if (lustre_) return lustre_->contains(id);
  return false;
}

}

