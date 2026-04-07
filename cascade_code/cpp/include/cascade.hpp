#pragma once

#include <string>
#include <vector>
#include <list>
#include <memory>
#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include <functional>
#include <optional>
#include <cstdint>
#include <cstring>
#include <array>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <deque>

namespace cascade {


struct CascadeConfig {

    size_t gpu_capacity_bytes = 32ULL * 1024 * 1024 * 1024;
    int gpu_device_id = 0;
    bool use_gpu = true;
    

    size_t shm_capacity_bytes = 64ULL * 1024 * 1024 * 1024;
    std::string shm_path = "/dev/shm/cascade";
    

    std::string lustre_path = "cascade_store";
    size_t lustre_stripe_size = 1024 * 1024;
    int lustre_stripe_count = 16;
    

    bool dedup_enabled = true;
    bool compression_enabled = true;
    int num_io_threads = 8;
    

    bool semantic_eviction = true;
    bool promotion_enabled = true;
    

    bool prefetch_enabled = true;
    int prefetch_threads = 2;
    size_t prefetch_queue_size = 64;
    

    bool kv_compression = false;
    

    bool aggregated_lustre = false;
    size_t agg_file_size = 256ULL * 1024 * 1024;
};


using BlockId = std::string;

BlockId compute_block_id(const uint8_t* data, size_t size);


template<typename V>
class ShardedIndex {
public:
    static constexpr size_t NUM_SHARDS = 256;
    
    ShardedIndex();
    ~ShardedIndex();
    
    bool put(const BlockId& key, V value, size_t size = 0);
    std::optional<V> get(const BlockId& key) const;
    bool remove(const BlockId& key);
    bool contains(const BlockId& key) const;
    

    std::optional<V> get_and_touch(const BlockId& key);
    

    struct LRUEntry {
        BlockId key;
        V value;
        size_t size;
    };
    std::optional<LRUEntry> get_lru_victim() const;
    

    std::optional<LRUEntry> get_lru_victim_skip(
        const std::function<bool(const BlockId&)>& should_skip) const;
    
    size_t total_size() const;
    size_t total_count() const;
    void clear();
    
private:
    struct LRUNode {
        BlockId key;
        V value;
        size_t size;
    };
    
    struct Shard {
        mutable std::shared_mutex mutex;
        

        std::list<LRUNode> lru_list;
        

        std::unordered_map<BlockId, typename std::list<LRUNode>::iterator> index;
        
        std::atomic<size_t> total_size{0};
    };
    
    std::array<Shard, NUM_SHARDS> shards_;
    
    size_t get_shard_id(const BlockId& key) const {
        return std::hash<BlockId>{}(key) % NUM_SHARDS;
    }
};


class GPUMemoryPool;

class GPUBackend {
public:
    GPUBackend(size_t capacity_bytes, int device_id = 0);
    ~GPUBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool remove(const BlockId& id);
    bool contains(const BlockId& id) const;
    

    bool get_and_touch(const BlockId& id, uint8_t* out_data, size_t* out_size);
    

    struct EvictedBlock {
        BlockId id;
        std::vector<uint8_t> data;
        size_t size;
    };
    

    std::optional<EvictedBlock> evict_lru();
    

    std::optional<EvictedBlock> evict_lru_semantic(
        const std::function<bool(const BlockId&)>& is_prefix);
    

    std::vector<EvictedBlock> evict_for_space(
        size_t needed_bytes,
        const std::function<bool(const BlockId&)>& is_prefix = nullptr);
    
    size_t used_bytes() const { return used_.load(); }
    size_t capacity() const { return capacity_; }
    
    void clear();
    void sync_all();
    

    uint8_t* get_base_ptr() const;
    size_t get_offset(const BlockId& id) const;
    
private:
    size_t capacity_;
    int device_id_;
    std::atomic<size_t> used_{0};
    

    std::unique_ptr<GPUMemoryPool> memory_pool_;
    

    static constexpr int NUM_PINNED_BUFFERS = 32;
    void* pinned_buffers_[32] = {nullptr};
    void* pinned_buffer_ = nullptr;
    size_t pinned_size_ = 8 * 1024 * 1024;
    

    static constexpr int NUM_STREAMS = 32;
    void* cuda_streams_[32] = {nullptr};
    std::atomic<int> current_stream_{0};
    

    struct GPUBlock {
        void* ptr;
        size_t size;
    };
    ShardedIndex<GPUBlock> index_;
    
    bool init_cuda();
    void* alloc_gpu(size_t size);
    void free_gpu(void* ptr, size_t size);
    void copy_h2d_async(void* dst, const void* src, size_t size, int stream);
    void copy_d2h_async(void* dst, const void* src, size_t size, int stream);
    void sync_stream(int stream);
    

    bool read_block_to_host(const GPUBlock& block, uint8_t* out_data);
};


class ShmBackend {
public:
    ShmBackend(size_t capacity_bytes, const std::string& path = "/dev/shm/cascade");
    ~ShmBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool remove(const BlockId& id);
    bool contains(const BlockId& id) const;
    

    bool get_and_touch(const BlockId& id, uint8_t* out_data, size_t* out_size);
    

    struct EvictedBlock {
        BlockId id;
        std::vector<uint8_t> data;
        size_t size;
    };
    
    std::optional<EvictedBlock> evict_lru();
    std::optional<EvictedBlock> evict_lru_semantic(
        const std::function<bool(const BlockId&)>& is_prefix);
    std::vector<EvictedBlock> evict_for_space(
        size_t needed_bytes,
        const std::function<bool(const BlockId&)>& is_prefix = nullptr);
    
    size_t used_bytes() const { return used_.load(); }
    size_t capacity() const { return capacity_; }
    
    void clear();
    
private:
    size_t capacity_;
    std::string path_;
    std::atomic<size_t> used_{0};
    

    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::atomic<size_t> write_offset_{0};
    

    struct FreeBlock {
        size_t offset;
        size_t size;
    };
    std::list<FreeBlock> free_list_;
    mutable std::mutex free_list_mutex_;
    

    size_t allocate(size_t size);

    void deallocate(size_t offset, size_t size);
    

    struct ShmBlock {
        size_t offset;
        size_t size;
    };
    ShardedIndex<ShmBlock> index_;
    

    void read_block(const ShmBlock& block, uint8_t* out_data);
    void write_block(uint8_t* dst, const uint8_t* src, size_t size);
};


class LustreBackend {
public:
    LustreBackend(const std::string& path, size_t stripe_size = 1024*1024, int stripe_count = 16);
    ~LustreBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool remove(const BlockId& id);
    bool contains(const BlockId& id) const;
    
    void flush();
    

    std::string get_base_path() const { return base_path_; }
    
private:
    std::string base_path_;
    size_t stripe_size_;
    int stripe_count_;
    
    std::string block_path(const BlockId& id) const;
};


class AggregatedLustreBackend {
public:
    AggregatedLustreBackend(const std::string& path, size_t max_file_size = 256ULL*1024*1024,
                            size_t stripe_size = 4*1024*1024, int stripe_count = 16);
    ~AggregatedLustreBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool contains(const BlockId& id) const;
    std::vector<BlockId> list_blocks() const;
    
    void flush();
    
private:
    struct BlockLocation {
        uint32_t file_id;
        size_t offset;
        size_t size;
    };
    
    std::string base_path_;
    size_t max_file_size_;
    size_t stripe_size_;
    int stripe_count_;
    

    int current_fd_ = -1;
    uint32_t current_file_id_ = 0;
    size_t current_offset_ = 0;
    std::mutex write_mutex_;
    

    mutable std::shared_mutex index_mutex_;
    std::unordered_map<BlockId, BlockLocation> index_;
    
    std::string file_path(uint32_t file_id) const;
    bool open_new_file();
};


struct CompressionMeta {
    float scale;
    int8_t zero_point;
};

class KVCompressor {
public:


    static std::vector<uint8_t> compress(const uint8_t* data, size_t size, CompressionMeta& meta);
    

    static bool decompress(const uint8_t* compressed, size_t compressed_size,
                           const CompressionMeta& meta, uint8_t* out_data, size_t original_size);
    

    static size_t compressed_size(size_t original_size) {
        return sizeof(CompressionMeta) + original_size / 2;
    }
    

    static size_t original_size(size_t compressed_size) {
        return (compressed_size - sizeof(CompressionMeta)) * 2;
    }
};


class PrefetchPipeline {
public:
    PrefetchPipeline(LustreBackend* lustre, AggregatedLustreBackend* agg_lustre,
                     ShmBackend* shm, int num_threads = 2, size_t queue_size = 64);
    ~PrefetchPipeline();
    

    void submit(const BlockId& id, size_t expected_size);
    

    void record_access(const BlockId& id);
    

    struct Stats {
        size_t submitted;
        size_t completed;
        size_t skipped;
    };
    Stats get_stats() const;
    
    void stop();
    
private:
    struct PrefetchRequest {
        BlockId id;
        size_t expected_size;
    };
    
    LustreBackend* lustre_;
    AggregatedLustreBackend* agg_lustre_;
    ShmBackend* shm_;
    
    std::vector<std::thread> workers_;
    std::queue<PrefetchRequest> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{true};
    size_t max_queue_size_;
    

    mutable std::mutex freq_mutex_;
    std::unordered_map<BlockId, uint32_t> access_freq_;
    

    std::atomic<size_t> submitted_{0};
    std::atomic<size_t> completed_{0};
    std::atomic<size_t> skipped_{0};
    
    void worker_loop();
};


class CascadeStore {
public:
    explicit CascadeStore(const CascadeConfig& config);
    ~CascadeStore();
    

    bool put(const BlockId& id, const uint8_t* data, size_t size, bool is_prefix = false);
    bool get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool contains(const BlockId& id) const;
    

    size_t put_batch(const std::vector<BlockId>& ids, 
                     const std::vector<const uint8_t*>& data,
                     const std::vector<size_t>& sizes);
    size_t get_batch(const std::vector<BlockId>& ids,
                     std::vector<uint8_t*>& out_data,
                     std::vector<size_t>& out_sizes);
    

    struct Stats {
        size_t gpu_used;
        size_t shm_used;
        size_t gpu_hits;
        size_t shm_hits;
        size_t lustre_hits;
        size_t misses;
        size_t dedup_hits;
        size_t gpu_evictions;
        size_t shm_evictions;
        size_t promotions_to_gpu;
        size_t promotions_to_shm;

        size_t prefetch_completed;
        size_t compression_savings_bytes;
        size_t shm_puts;
        size_t lustre_puts;
    };
    Stats get_stats() const;
    
    void clear();
    void flush();
    
private:
    CascadeConfig config_;
    
    std::unique_ptr<GPUBackend> gpu_;
    std::unique_ptr<ShmBackend> shm_;
    std::unique_ptr<LustreBackend> lustre_;
    std::unique_ptr<AggregatedLustreBackend> agg_lustre_;
    std::unique_ptr<PrefetchPipeline> prefetcher_;
    

    ShardedIndex<bool> known_blocks_;
    ShardedIndex<bool> prefix_blocks_;
    

    mutable std::atomic<size_t> gpu_hits_{0};
    mutable std::atomic<size_t> shm_hits_{0};
    mutable std::atomic<size_t> lustre_hits_{0};
    mutable std::atomic<size_t> misses_{0};
    mutable std::atomic<size_t> dedup_hits_{0};
    mutable std::atomic<size_t> gpu_evictions_{0};
    mutable std::atomic<size_t> shm_evictions_{0};
    mutable std::atomic<size_t> promotions_to_gpu_{0};
    mutable std::atomic<size_t> promotions_to_shm_{0};
    mutable std::atomic<size_t> shm_puts_{0};
    mutable std::atomic<size_t> lustre_puts_{0};
    mutable std::atomic<size_t> compression_savings_{0};
    

    bool is_prefix_block(const BlockId& id) const;
    
    bool evict_gpu_to_shm(size_t needed_bytes);
    bool evict_shm_to_lustre(size_t needed_bytes);
    
    bool promote_to_gpu(const BlockId& id, const uint8_t* data, size_t size, bool is_prefix);
    bool promote_to_shm(const BlockId& id, const uint8_t* data, size_t size);
    

    bool lustre_put(const BlockId& id, const uint8_t* data, size_t size);
    bool lustre_get(const BlockId& id, uint8_t* out_data, size_t* out_size);
    bool lustre_contains(const BlockId& id) const;
};

}

