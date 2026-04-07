#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <thread>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <array>
#include <queue>
#include <condition_variable>
#include <functional>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "cascade.hpp"

namespace cascade {
namespace distributed {


enum class TierType : uint8_t {
    LOCAL_GPU  = 0,
    LOCAL_DRAM = 1,
    REMOTE_GPU = 2,
    REMOTE_DRAM = 3,
    LUSTRE     = 4,
};

struct BlockLocation {
    int node_id;
    int gpu_id;
    size_t offset;
    size_t size;
    size_t dram_offset;
    size_t dram_size;
    uint64_t timestamp;
    bool is_gpu;
    bool is_prefix;
    bool has_dram_shadow;
    
    TierType tier() const {
        if (is_gpu) return TierType::LOCAL_GPU;
        return TierType::LOCAL_DRAM;
    }
    
    bool is_local(int my_rank) const { return node_id == my_rank; }
};


template<typename V>
class DistributedIndex {
public:
    static constexpr size_t NUM_SHARDS = 256;
    
    DistributedIndex() = default;
    ~DistributedIndex() = default;
    
    bool put(const BlockId& key, V value, size_t size = 0) {
        size_t shard_id = std::hash<BlockId>{}(key) % NUM_SHARDS;
        auto& shard = shards_[shard_id];
        std::unique_lock lock(shard.mutex);
        shard.data[key] = value;
        return true;
    }
    
    std::optional<V> get(const BlockId& key) const {
        size_t shard_id = std::hash<BlockId>{}(key) % NUM_SHARDS;
        const auto& shard = shards_[shard_id];
        std::shared_lock lock(shard.mutex);
        auto it = shard.data.find(key);
        if (it == shard.data.end()) return std::nullopt;
        return it->second;
    }
    
    bool contains(const BlockId& key) const {
        return get(key).has_value();
    }
    
    bool remove(const BlockId& key) {
        size_t shard_id = std::hash<BlockId>{}(key) % NUM_SHARDS;
        auto& shard = shards_[shard_id];
        std::unique_lock lock(shard.mutex);
        return shard.data.erase(key) > 0;
    }
    
    void clear() {
        for (auto& shard : shards_) {
            std::unique_lock lock(shard.mutex);
            shard.data.clear();
        }
    }
    

    std::vector<BlockId> keys() const {
        std::vector<BlockId> result;
        for (const auto& shard : shards_) {
            std::shared_lock lock(shard.mutex);
            for (const auto& [k, v] : shard.data) {
                result.push_back(k);
            }
        }
        return result;
    }
    
    size_t size() const {
        size_t total = 0;
        for (const auto& shard : shards_) {
            std::shared_lock lock(shard.mutex);
            total += shard.data.size();
        }
        return total;
    }
    
private:
    struct Shard {
        mutable std::shared_mutex mutex;
        std::unordered_map<BlockId, V> data;
    };
    std::array<Shard, NUM_SHARDS> shards_;
};


struct AccessRecord {
    uint32_t total_count = 0;
    uint32_t local_count = 0;
    uint32_t remote_count = 0;
    int last_access_node = -1;
    uint64_t last_access_time = 0;
    

    float ema_remote_rate = 0.0f;
    uint32_t window_remote = 0;
    uint32_t window_total = 0;
    static constexpr uint32_t WINDOW_SIZE = 8;
    static constexpr float EMA_ALPHA = 0.3f;
    
    float locality_score(int my_rank) const {
        if (total_count == 0) return 0.0f;
        return static_cast<float>(local_count) / total_count;
    }
};


struct DistributedConfig {

    size_t gpu_capacity_per_device = 32ULL * 1024 * 1024 * 1024;
    size_t dram_capacity = 64ULL * 1024 * 1024 * 1024;
    int num_gpus_per_node = 4;
    size_t staging_buffer_size = 16 * 1024 * 1024;
    int num_staging_buffers = 32;
    

    bool semantic_eviction = true;
    bool prefix_replication = true;
    

    bool dedup_enabled = true;
    

    bool locality_aware = true;
    uint32_t promotion_threshold = 3;
    

    bool kv_compression = false;
    bool sync_metadata = false;
    

    std::string lustre_path = "";
    bool aggregated_lustre = true;
    size_t agg_file_size = 256ULL * 1024 * 1024;
};


struct DRAMBlock {
    size_t offset;
    size_t size;
    bool is_prefix = false;
};

class DistributedDRAMBackend {
public:
#ifdef USE_MPI
    DistributedDRAMBackend(size_t capacity, MPI_Comm comm);
#else
    explicit DistributedDRAMBackend(size_t capacity);
#endif
    ~DistributedDRAMBackend();
    
    bool put_local(const BlockId& id, const uint8_t* data, size_t size, bool is_prefix = false);
    bool get_local(const BlockId& id, uint8_t* out, size_t* out_size);
    bool get_remote(int target_rank, size_t offset, uint8_t* out, size_t size);
    bool remove_local(const BlockId& id);
    

    std::vector<std::pair<BlockId, std::vector<uint8_t>>>
        evict_for_space(size_t needed, bool protect_prefix = true);
    
    bool contains(const BlockId& id) const { return index_.contains(id); }
    size_t used_bytes() const { return used_.load(); }
    size_t capacity() const { return capacity_; }
    size_t count() const { return index_.size(); }
    void barrier();
    size_t get_offset(const BlockId& id) const;
    void clear();
    
    DistributedIndex<DRAMBlock> index_;
    
private:
    size_t capacity_;
    void* dram_base_ = nullptr;
    std::atomic<size_t> write_offset_{0};
    std::atomic<size_t> used_{0};
    

    struct FreeBlock {
        size_t offset;
        size_t size;
    };
    std::list<FreeBlock> free_list_;
    mutable std::mutex free_list_mutex_;
    
    size_t allocate(size_t size);
    void deallocate(size_t offset, size_t size);
    

    mutable std::mutex lru_mutex_;
    std::list<BlockId> lru_list_;
    std::unordered_map<BlockId, std::list<BlockId>::iterator> lru_map_;
    
#ifdef USE_MPI
    MPI_Comm comm_;
    MPI_Win window_;
    int rank_, world_size_;
#endif
};


class DistributedGPUBackend {
public:
#ifdef USE_MPI
    DistributedGPUBackend(size_t cap_per_gpu, int num_gpus, MPI_Comm comm);
#else
    DistributedGPUBackend(size_t cap_per_gpu, int num_gpus);
#endif
    ~DistributedGPUBackend();
    
    bool put(const BlockId& id, const uint8_t* data, size_t size, bool is_prefix = false);
    bool get(const BlockId& id, uint8_t* out, size_t* out_size);
    bool put_local(const BlockId& id, const uint8_t* data, size_t size, int gpu, bool is_prefix = false);
    bool get_local(const BlockId& id, uint8_t* out, size_t* out_size);
    bool get_remote(int rank, size_t offset, size_t size, uint8_t* out);
    
    std::optional<BlockLocation> locate(const BlockId& id) const;
    size_t used_bytes() const;
    int get_target_node(const BlockId& id) const;
    int get_target_gpu(const BlockId& id) const;
    void sync_all();
    void barrier();
    void clear();
    

    std::vector<GPUBackend::EvictedBlock> evict_gpu_for_space(
        int gpu_id, size_t needed_bytes,
        const std::function<bool(const BlockId&)>& is_prefix = nullptr);
    
    int num_gpus() const { return num_gpus_; }
    

    DistributedIndex<BlockLocation> global_index_;
    
private:
    int num_gpus_;
    size_t cap_per_gpu_;
    int rank_ = 0, world_size_ = 1;
    int my_gpu_id_ = 0;
    
#ifdef USE_MPI
    MPI_Comm comm_;
    MPI_Win window_;
#endif
    
    std::vector<std::unique_ptr<GPUBackend>> gpus_;
    bool peer_[16][16] = {{false}};
    void* pinned_[32] = {nullptr};
    size_t staging_size_ = 16 * 1024 * 1024;
    
    void setup_nvlink();
    void init_window();
};


class DistributedStore {
public:
#ifdef USE_MPI
    DistributedStore(const DistributedConfig& cfg, MPI_Comm comm = MPI_COMM_WORLD);
#else
    explicit DistributedStore(const DistributedConfig& cfg);
#endif
    ~DistributedStore();
    

    bool put(const BlockId& id, const uint8_t* data, size_t size, bool is_prefix = false);
    bool get(const BlockId& id, uint8_t* out, size_t* out_size);
    bool contains(const BlockId& id) const;
    std::optional<BlockLocation> locate(const BlockId& id) const;
    

    size_t put_batch(const std::vector<BlockId>& ids,
                     const std::vector<const uint8_t*>& data,
                     const std::vector<size_t>& sizes,
                     const std::vector<bool>& is_prefix);
    size_t get_batch(const std::vector<BlockId>& ids,
                     std::vector<uint8_t*>& out,
                     std::vector<size_t>& sizes);
    

    bool evict_gpu_to_dram(size_t needed_bytes);
    bool evict_dram_to_lustre(size_t needed_bytes);
    void sync_prefix_registry();
    

    size_t dedup_hits() const { return dedup_hits_.load(); }
    size_t dedup_bytes_saved() const { return dedup_bytes_saved_.load(); }
    

    void record_access(const BlockId& id, TierType origin_tier);
    bool should_promote_local(const BlockId& id) const;
    void promote_to_local_gpu(const BlockId& id, const uint8_t* data, size_t size);
    
    void barrier();
    void sync_metadata();
    void flush();
    void clear();
    
    struct Stats {

        size_t local_gpu_used, local_dram_used;
        size_t cluster_gpu_used, cluster_dram_used;

        size_t local_gpu_hits, local_dram_hits;
        size_t remote_gpu_hits, remote_dram_hits;
        size_t lustre_hits;
        size_t misses;

        size_t dedup_hits;
        size_t dedup_bytes_saved;

        size_t gpu_evictions, dram_evictions;
        size_t prefix_blocks_protected;

        size_t promotions_to_local;

        size_t compression_savings;

        size_t total_blocks;
        size_t prefix_blocks;
    };
    Stats get_stats();
    
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    
private:
    DistributedConfig cfg_;
    int rank_, world_size_;
#ifdef USE_MPI
    MPI_Comm comm_;
#endif
    

    std::unique_ptr<DistributedGPUBackend> gpu_;
    std::unique_ptr<DistributedDRAMBackend> dram_;
    std::unique_ptr<LustreBackend> lustre_;
    std::unique_ptr<AggregatedLustreBackend> agg_lustre_;
    

    DistributedIndex<bool> global_dedup_;
    

    mutable std::shared_mutex prefix_mutex_;
    std::unordered_set<BlockId> prefix_registry_;
    

    DistributedIndex<AccessRecord> access_tracker_;
    

    std::atomic<size_t> local_gpu_hits_{0}, local_dram_hits_{0};
    std::atomic<size_t> remote_gpu_hits_{0}, remote_dram_hits_{0};
    std::atomic<size_t> lustre_hits_{0};
    std::atomic<size_t> misses_{0};
    std::atomic<size_t> dedup_hits_{0}, dedup_bytes_saved_{0};
    std::atomic<size_t> gpu_evictions_{0}, dram_evictions_{0};
    std::atomic<size_t> prefix_blocks_protected_{0};
    std::atomic<size_t> promotions_to_local_{0};
    std::atomic<size_t> compression_savings_{0};
    

    std::atomic<size_t> put_counter_{0};
    static constexpr size_t SYNC_INTERVAL = 100;
    

    mutable std::mutex dirty_mutex_;
    std::vector<std::pair<BlockId, BlockLocation>> dirty_blocks_;
    std::atomic<uint64_t> sync_epoch_{0};
    

    bool lustre_put(const BlockId& id, const uint8_t* data, size_t size);
    bool lustre_get(const BlockId& id, uint8_t* out, size_t* out_size);
    bool lustre_contains(const BlockId& id) const;
    bool is_prefix(const BlockId& id) const;
    void mark_dirty(const BlockId& id, const BlockLocation& loc);
    

    void broadcast_prefix_data();
};

}
}

