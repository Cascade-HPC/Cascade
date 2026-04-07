#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cascade.hpp"
#include "cascade_distributed.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cascade_cpp, m) {
    m.doc() = "High-performance Cascade KV Cache for LLM inference";
    
    // ========================================================================
    // CascadeConfig
    // ========================================================================
    py::class_<cascade::CascadeConfig>(m, "CascadeConfig")
        .def(py::init<>())
        .def_readwrite("gpu_capacity_bytes", &cascade::CascadeConfig::gpu_capacity_bytes)
        .def_readwrite("shm_capacity_bytes", &cascade::CascadeConfig::shm_capacity_bytes)
        .def_readwrite("shm_path", &cascade::CascadeConfig::shm_path)
        .def_readwrite("lustre_path", &cascade::CascadeConfig::lustre_path)
        .def_readwrite("lustre_stripe_size", &cascade::CascadeConfig::lustre_stripe_size)
        .def_readwrite("lustre_stripe_count", &cascade::CascadeConfig::lustre_stripe_count)
        .def_readwrite("gpu_device_id", &cascade::CascadeConfig::gpu_device_id)
        .def_readwrite("dedup_enabled", &cascade::CascadeConfig::dedup_enabled)
        .def_readwrite("compression_enabled", &cascade::CascadeConfig::compression_enabled)
        .def_readwrite("use_gpu", &cascade::CascadeConfig::use_gpu)
        .def_readwrite("semantic_eviction", &cascade::CascadeConfig::semantic_eviction)
        .def_readwrite("promotion_enabled", &cascade::CascadeConfig::promotion_enabled)
        // V6 features
        .def_readwrite("prefetch_enabled", &cascade::CascadeConfig::prefetch_enabled)
        .def_readwrite("prefetch_threads", &cascade::CascadeConfig::prefetch_threads)
        .def_readwrite("prefetch_queue_size", &cascade::CascadeConfig::prefetch_queue_size)
        .def_readwrite("kv_compression", &cascade::CascadeConfig::kv_compression)
        .def_readwrite("aggregated_lustre", &cascade::CascadeConfig::aggregated_lustre)
        .def_readwrite("agg_file_size", &cascade::CascadeConfig::agg_file_size);
    
    // ========================================================================
    // CascadeStore::Stats
    // ========================================================================
    py::class_<cascade::CascadeStore::Stats>(m, "CascadeStats")
        .def_readonly("gpu_used", &cascade::CascadeStore::Stats::gpu_used)
        .def_readonly("shm_used", &cascade::CascadeStore::Stats::shm_used)
        .def_readonly("gpu_hits", &cascade::CascadeStore::Stats::gpu_hits)
        .def_readonly("shm_hits", &cascade::CascadeStore::Stats::shm_hits)
        .def_readonly("lustre_hits", &cascade::CascadeStore::Stats::lustre_hits)
        .def_readonly("misses", &cascade::CascadeStore::Stats::misses)
        .def_readonly("dedup_hits", &cascade::CascadeStore::Stats::dedup_hits)
        .def_readonly("gpu_evictions", &cascade::CascadeStore::Stats::gpu_evictions)
        .def_readonly("shm_evictions", &cascade::CascadeStore::Stats::shm_evictions)
        .def_readonly("promotions_to_gpu", &cascade::CascadeStore::Stats::promotions_to_gpu)
        .def_readonly("promotions_to_shm", &cascade::CascadeStore::Stats::promotions_to_shm)
        // V6 stats
        .def_readonly("prefetch_completed", &cascade::CascadeStore::Stats::prefetch_completed)
        .def_readonly("compression_savings_bytes", &cascade::CascadeStore::Stats::compression_savings_bytes)
        .def_readonly("shm_puts", &cascade::CascadeStore::Stats::shm_puts)
        .def_readonly("lustre_puts", &cascade::CascadeStore::Stats::lustre_puts)
        .def("__repr__", [](const cascade::CascadeStore::Stats& s) {
            return "CascadeStats(gpu=" + std::to_string(s.gpu_used / (1024*1024)) + "MB"
                   ", shm=" + std::to_string(s.shm_used / (1024*1024)) + "MB"
                   ", hits=" + std::to_string(s.gpu_hits + s.shm_hits + s.lustre_hits) +
                   ", dedup=" + std::to_string(s.dedup_hits) +
                   ", evictions=" + std::to_string(s.gpu_evictions + s.shm_evictions) +
                   ", promotions=" + std::to_string(s.promotions_to_gpu + s.promotions_to_shm) +
                   ", prefetch=" + std::to_string(s.prefetch_completed) +
                   ", compression_saved=" + std::to_string(s.compression_savings_bytes / (1024*1024)) + "MB)";
        });
    
    // ========================================================================
    // CascadeStore (Main Interface)
    // ========================================================================
    py::class_<cascade::CascadeStore>(m, "CascadeStore")
        .def(py::init<const cascade::CascadeConfig&>())
        .def("put", [](cascade::CascadeStore& self, const std::string& block_id, py::array_t<uint8_t>& data, bool is_prefix) {
            py::buffer_info buf = data.request();
            return self.put(block_id, static_cast<const uint8_t*>(buf.ptr), buf.size, is_prefix);
        }, py::arg("block_id"), py::arg("data"), py::arg("is_prefix") = false)
        .def("get", [](cascade::CascadeStore& self, const std::string& block_id, py::array_t<uint8_t>& out_data) {
            py::buffer_info buf = out_data.request();
            size_t size = 0;
            bool found = self.get(block_id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        })
        .def("contains", &cascade::CascadeStore::contains)
        .def("get_stats", &cascade::CascadeStore::get_stats)
        .def("clear", &cascade::CascadeStore::clear)
        .def("flush", &cascade::CascadeStore::flush);

    // ========================================================================
    // Distributed Backend (Multi-Node)
    // ========================================================================
#ifdef USE_MPI
    py::class_<cascade::distributed::DistributedConfig>(m, "DistributedConfig")
        .def(py::init<>())
        .def_readwrite("gpu_capacity_per_device", &cascade::distributed::DistributedConfig::gpu_capacity_per_device)
        .def_readwrite("dram_capacity", &cascade::distributed::DistributedConfig::dram_capacity)
        .def_readwrite("num_gpus_per_node", &cascade::distributed::DistributedConfig::num_gpus_per_node)
        .def_readwrite("staging_buffer_size", &cascade::distributed::DistributedConfig::staging_buffer_size)
        .def_readwrite("num_staging_buffers", &cascade::distributed::DistributedConfig::num_staging_buffers)
        .def_readwrite("sync_metadata", &cascade::distributed::DistributedConfig::sync_metadata)
        // V6 novelty features
        .def_readwrite("semantic_eviction", &cascade::distributed::DistributedConfig::semantic_eviction)
        .def_readwrite("prefix_replication", &cascade::distributed::DistributedConfig::prefix_replication)
        .def_readwrite("dedup_enabled", &cascade::distributed::DistributedConfig::dedup_enabled)
        .def_readwrite("locality_aware", &cascade::distributed::DistributedConfig::locality_aware)
        .def_readwrite("promotion_threshold", &cascade::distributed::DistributedConfig::promotion_threshold)
        .def_readwrite("kv_compression", &cascade::distributed::DistributedConfig::kv_compression)
        .def_readwrite("lustre_path", &cascade::distributed::DistributedConfig::lustre_path)
        .def_readwrite("aggregated_lustre", &cascade::distributed::DistributedConfig::aggregated_lustre)
        .def_readwrite("agg_file_size", &cascade::distributed::DistributedConfig::agg_file_size);

    py::class_<cascade::distributed::BlockLocation>(m, "BlockLocation")
        .def_readonly("node_id", &cascade::distributed::BlockLocation::node_id)
        .def_readonly("gpu_id", &cascade::distributed::BlockLocation::gpu_id)
        .def_readonly("offset", &cascade::distributed::BlockLocation::offset)
        .def_readonly("size", &cascade::distributed::BlockLocation::size)
        .def_readonly("is_gpu", &cascade::distributed::BlockLocation::is_gpu)
        .def_readonly("is_prefix", &cascade::distributed::BlockLocation::is_prefix);

    py::class_<cascade::distributed::DistributedStore::Stats>(m, "DistributedStats")
        .def_readonly("local_gpu_used", &cascade::distributed::DistributedStore::Stats::local_gpu_used)
        .def_readonly("local_dram_used", &cascade::distributed::DistributedStore::Stats::local_dram_used)
        .def_readonly("cluster_gpu_used", &cascade::distributed::DistributedStore::Stats::cluster_gpu_used)
        .def_readonly("cluster_dram_used", &cascade::distributed::DistributedStore::Stats::cluster_dram_used)
        .def_readonly("local_gpu_hits", &cascade::distributed::DistributedStore::Stats::local_gpu_hits)
        .def_readonly("local_dram_hits", &cascade::distributed::DistributedStore::Stats::local_dram_hits)
        .def_readonly("remote_gpu_hits", &cascade::distributed::DistributedStore::Stats::remote_gpu_hits)
        .def_readonly("remote_dram_hits", &cascade::distributed::DistributedStore::Stats::remote_dram_hits)
        .def_readonly("lustre_hits", &cascade::distributed::DistributedStore::Stats::lustre_hits)
        .def_readonly("misses", &cascade::distributed::DistributedStore::Stats::misses)
        .def_readonly("dedup_hits", &cascade::distributed::DistributedStore::Stats::dedup_hits)
        .def_readonly("dedup_bytes_saved", &cascade::distributed::DistributedStore::Stats::dedup_bytes_saved)
        .def_readonly("gpu_evictions", &cascade::distributed::DistributedStore::Stats::gpu_evictions)
        .def_readonly("dram_evictions", &cascade::distributed::DistributedStore::Stats::dram_evictions)
        .def_readonly("prefix_blocks_protected", &cascade::distributed::DistributedStore::Stats::prefix_blocks_protected)
        .def_readonly("promotions_to_local", &cascade::distributed::DistributedStore::Stats::promotions_to_local)
        .def_readonly("compression_savings", &cascade::distributed::DistributedStore::Stats::compression_savings)
        .def_readonly("total_blocks", &cascade::distributed::DistributedStore::Stats::total_blocks)
        .def_readonly("prefix_blocks", &cascade::distributed::DistributedStore::Stats::prefix_blocks)
        .def("__repr__", [](const cascade::distributed::DistributedStore::Stats& s) {
            return "DistributedStats("
                   "gpu=" + std::to_string(s.local_gpu_used/(1024*1024)) + "MB"
                   ", dram=" + std::to_string(s.local_dram_used/(1024*1024)) + "MB"
                   ", hits=" + std::to_string(s.local_gpu_hits+s.local_dram_hits+s.remote_gpu_hits+s.remote_dram_hits+s.lustre_hits) +
                   ", dedup=" + std::to_string(s.dedup_hits) +
                   ", promotions=" + std::to_string(s.promotions_to_local) +
                   ", prefix=" + std::to_string(s.prefix_blocks) + ")";
        });

    py::class_<cascade::distributed::DistributedStore>(m, "DistributedStore")
        .def(py::init<const cascade::distributed::DistributedConfig&>())
        .def("put", [](cascade::distributed::DistributedStore& self, const std::string& block_id, py::array_t<uint8_t>& data, bool is_prefix) {
            py::buffer_info buf = data.request();
            return self.put(block_id, static_cast<const uint8_t*>(buf.ptr), buf.size, is_prefix);
        }, py::arg("block_id"), py::arg("data"), py::arg("is_prefix") = false)
        .def("get", [](cascade::distributed::DistributedStore& self, const std::string& block_id, py::array_t<uint8_t>& out_data) {
            py::buffer_info buf = out_data.request();
            size_t size = buf.size;
            bool found = self.get(block_id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        })
        .def("contains", &cascade::distributed::DistributedStore::contains)
        .def("locate", &cascade::distributed::DistributedStore::locate)
        .def("get_stats", &cascade::distributed::DistributedStore::get_stats)
        .def("sync_metadata", &cascade::distributed::DistributedStore::sync_metadata)
        .def("barrier", &cascade::distributed::DistributedStore::barrier)
        .def("clear", &cascade::distributed::DistributedStore::clear)
        .def("flush", &cascade::distributed::DistributedStore::flush)
        .def_property_readonly("rank", &cascade::distributed::DistributedStore::rank)
        .def_property_readonly("world_size", &cascade::distributed::DistributedStore::world_size);
#endif


    m.def("compute_block_id", [](py::array_t<uint8_t>& data) {
        py::buffer_info buf = data.request();
        return cascade::compute_block_id(static_cast<const uint8_t*>(buf.ptr), buf.size);
    }, py::arg("data"));
    
    py::class_<cascade::GPUBackend>(m, "GPUBackend")
        .def(py::init<size_t, int>(), py::arg("capacity_bytes"), py::arg("device_id") = 0)
        .def("put", [](cascade::GPUBackend& self, const std::string& id, py::array_t<uint8_t>& data) {
            py::buffer_info buf = data.request();
            return self.put(id, static_cast<const uint8_t*>(buf.ptr), buf.size);
        })
        .def("get", [](cascade::GPUBackend& self, const std::string& id, py::array_t<uint8_t>& out) {
            py::buffer_info buf = out.request();
            size_t size = 0;
            bool found = self.get(id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        })
        .def("contains", &cascade::GPUBackend::contains)
        .def("used_bytes", &cascade::GPUBackend::used_bytes)
        .def("clear", &cascade::GPUBackend::clear);

    py::class_<cascade::ShmBackend>(m, "ShmBackend")
        .def(py::init<size_t, const std::string&>(), py::arg("capacity_bytes"), py::arg("path") = "/dev/shm/cascade")
        .def("put", [](cascade::ShmBackend& self, const std::string& id, py::array_t<uint8_t>& data) {
            py::buffer_info buf = data.request();
            return self.put(id, static_cast<const uint8_t*>(buf.ptr), buf.size);
        })
        .def("get", [](cascade::ShmBackend& self, const std::string& id, py::array_t<uint8_t>& out) {
            py::buffer_info buf = out.request();
            size_t size = 0;
            bool found = self.get(id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        })
        .def("contains", &cascade::ShmBackend::contains)
        .def("used_bytes", &cascade::ShmBackend::used_bytes)
        .def("clear", &cascade::ShmBackend::clear);

    py::class_<cascade::LustreBackend>(m, "LustreBackend")
        .def(py::init<const std::string&, size_t, int>(), py::arg("path"), py::arg("stripe_size") = 4*1024*1024, py::arg("stripe_count") = 4)
        .def("put", [](cascade::LustreBackend& self, const std::string& id, py::array_t<uint8_t>& data) {
            py::buffer_info buf = data.request();
            return self.put(id, static_cast<const uint8_t*>(buf.ptr), buf.size);
        })
        .def("get", [](cascade::LustreBackend& self, const std::string& id, py::array_t<uint8_t>& out) {
            py::buffer_info buf = out.request();
            size_t size = 0;
            bool found = self.get(id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        })
        .def("contains", &cascade::LustreBackend::contains)
        .def("flush", &cascade::LustreBackend::flush);

    py::class_<cascade::AggregatedLustreBackend>(m, "AggregatedLustreBackend")
        .def(py::init<const std::string&, size_t, size_t, int>(),
             py::arg("path"),
             py::arg("max_file_size") = 256ULL*1024*1024,
             py::arg("stripe_size") = 4*1024*1024,
             py::arg("stripe_count") = 16)
        .def("put", [](cascade::AggregatedLustreBackend& self, const std::string& id, py::array_t<uint8_t>& data) {
            py::buffer_info buf = data.request();
            return self.put(id, static_cast<const uint8_t*>(buf.ptr), buf.size);
        })
        .def("get", [](cascade::AggregatedLustreBackend& self, const std::string& id, py::array_t<uint8_t>& out) {
            py::buffer_info buf = out.request();
            size_t size = 0;
            bool found = self.get(id, static_cast<uint8_t*>(buf.ptr), &size);
            return py::make_tuple(found, size);
        })
        .def("contains", &cascade::AggregatedLustreBackend::contains)
        .def("list_blocks", &cascade::AggregatedLustreBackend::list_blocks)
        .def("flush", &cascade::AggregatedLustreBackend::flush);
}
