#pragma once

#include "common.hpp"

#include <hnsw/distance.hpp>
#include <hnsw/index.hpp>
#include <hnsw/key_mapper.hpp>

#include <boost/optional.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>


struct index_t {
    virtual ~index_t() { }

    virtual void insert(const std::string &key, const vector_t &target) = 0;
    virtual void remove(const std::string &key) = 0;
    virtual std::vector<std::pair<std::string, float>> search(const vector_t &target, size_t neighbors) const = 0;
    virtual bool check() const = 0;
    virtual size_t size() const = 0;
    virtual size_t memory_footprint() const = 0;
    virtual std::string memory_footprint_info() const = 0;
    virtual size_t used_memory() const = 0;
    virtual std::string used_memory_info() const = 0;

    virtual void prepare_dataset(dataset_t &dataset) const = 0;
};


template<class Index, bool NormalizeDataset>
struct hnsw_index : index_t {
    Index wrapped;

    void insert(const std::string &key, const vector_t &target) override {
        wrapped.insert(key, target);
    }

    void remove(const std::string &key) override {
        wrapped.remove(key);
    }

    std::vector<std::pair<std::string, float>> search(const vector_t &target, size_t neighbors) const override {
        auto r = wrapped.search(target, neighbors);

        std::vector<std::pair<std::string, float>> result;
        result.reserve(r.size());

        for (auto &x: r) {
            result.emplace_back(std::move(x.key), x.distance);
        }

        return result;
    }

    bool check() const override {
        return wrapped.check();
    }

    size_t size() const override {
        return wrapped.index.nodes.size();
    }

    void prepare_dataset(dataset_t &dataset) const override {
        if (NormalizeDataset) {
            normalize(dataset);
        }
    }

    size_t memory_footprint() const override {
        size_t footprint = 0;

        for (const auto &x: wrapped.index.levels) {
            footprint += sizeof(x);
            footprint += sizeof(*x.second.begin()) * x.second.bucket_count();
        }

        footprint += sizeof(*wrapped.index.nodes.begin()) * wrapped.index.nodes.bucket_count();

        for (const auto &x: wrapped.index.nodes) {
            footprint += sizeof(*x.second.vector.begin()) * x.second.vector.capacity();
            footprint += sizeof(*x.second.layers.begin()) * x.second.layers.capacity();

            for (const auto &layer: x.second.layers) {
                footprint += sizeof(*layer.incoming.begin()) * layer.incoming.capacity();
                footprint += sizeof(*layer.outgoing.begin()) * layer.outgoing.capacity();
            }
        }

        footprint += sizeof(*wrapped.key_to_internal.begin()) * wrapped.key_to_internal.bucket_count();
        footprint += sizeof(*wrapped.internal_to_key.begin()) * wrapped.internal_to_key.bucket_count();

        for (const auto &x: wrapped.key_to_internal) {
            footprint += x.first.capacity();
        }

        for (const auto &x: wrapped.internal_to_key) {
            footprint += x.second.capacity();
        }

        return footprint;
    }

    size_t used_memory() const override {
        size_t footprint = 0;

        for (const auto &x: wrapped.index.levels) {
            footprint += sizeof(x);
            footprint += sizeof(*x.second.begin()) * x.second.size();
        }

        footprint += sizeof(*wrapped.index.nodes.begin()) * wrapped.index.nodes.size();

        for (const auto &x: wrapped.index.nodes) {
            footprint += sizeof(*x.second.vector.begin()) * x.second.vector.size();
            footprint += sizeof(*x.second.layers.begin()) * x.second.layers.size();

            for (const auto &layer: x.second.layers) {
                footprint += sizeof(*layer.incoming.begin()) * layer.incoming.size();
                footprint += sizeof(*layer.outgoing.begin()) * layer.outgoing.size();
            }
        }

        footprint += sizeof(*wrapped.key_to_internal.begin()) * wrapped.key_to_internal.size();
        footprint += sizeof(*wrapped.internal_to_key.begin()) * wrapped.internal_to_key.size();

        for (const auto &x: wrapped.key_to_internal) {
            footprint += x.first.size();
        }

        for (const auto &x: wrapped.internal_to_key) {
            footprint += x.second.size();
        }

        return footprint;
    }

    std::string memory_footprint_info() const override {
        std::string result;

        {
            size_t levels_footprint = 0;

            for (const auto &x: wrapped.index.levels) {
                levels_footprint += sizeof(x);
                levels_footprint += sizeof(*x.second.begin()) * x.second.bucket_count();
            }

            result += "levels: " + std::to_string(levels_footprint) + "; ";
        }

        result += "nodes table: " + std::to_string(sizeof(*wrapped.index.nodes.begin()) * wrapped.index.nodes.bucket_count()) + "; ";

        {
            size_t vectors_footprint = 0;

            for (const auto &x: wrapped.index.nodes) {
                vectors_footprint += sizeof(*x.second.vector.begin()) * x.second.vector.capacity();
            }

            result += "vectors: " + std::to_string(vectors_footprint) + "; ";
        }

        {
            size_t layers_vectors_footprint = 0;

            for (const auto &x: wrapped.index.nodes) {
                layers_vectors_footprint += sizeof(*x.second.layers.begin()) * x.second.layers.capacity();
            }

            result += "layers vectors: " + std::to_string(layers_vectors_footprint) + "; ";
        }

        {
            size_t incoming_links_footprint = 0;

            for (const auto &x: wrapped.index.nodes) {
                for (const auto &layer: x.second.layers) {
                    incoming_links_footprint += sizeof(*layer.incoming.begin()) * layer.incoming.capacity();
                }
            }

            result += "incoming links: " + std::to_string(incoming_links_footprint) + "; ";
        }

        {
            size_t outgoing_links_footprint = 0;

            for (const auto &x: wrapped.index.nodes) {
                for (const auto &layer: x.second.layers) {
                    outgoing_links_footprint += sizeof(*layer.outgoing.begin()) * layer.outgoing.capacity();
                }
            }

            result += "outgoing links: " + std::to_string(outgoing_links_footprint) + "; ";
        }

        result += "key_to_internal: " + std::to_string(sizeof(*wrapped.key_to_internal.begin()) * wrapped.key_to_internal.bucket_count()) + "; ";
        result += "internal_to_key: " + std::to_string(sizeof(*wrapped.internal_to_key.begin()) * wrapped.internal_to_key.bucket_count()) + "; ";

        {
            size_t keys_footprint = 0;

            for (const auto &x: wrapped.key_to_internal) {
                keys_footprint += x.first.capacity();
            }

            for (const auto &x: wrapped.internal_to_key) {
                keys_footprint += x.second.capacity();
            }

            result += "keys: " + std::to_string(keys_footprint) + "; ";
        }

        return result;
    }

    std::string used_memory_info() const override {
        std::string result;

        {
            size_t levels_footprint = 0;

            for (const auto &x: wrapped.index.levels) {
                levels_footprint += sizeof(x);
                levels_footprint += sizeof(*x.second.begin()) * x.second.size();
            }

            result += "levels: " + std::to_string(levels_footprint) + "; ";
        }

        result += "nodes table: " + std::to_string(sizeof(*wrapped.index.nodes.begin()) * wrapped.index.nodes.size()) + "; ";

        {
            size_t vectors_footprint = 0;

            for (const auto &x: wrapped.index.nodes) {
                vectors_footprint += sizeof(*x.second.vector.begin()) * x.second.vector.size();
            }

            result += "vectors: " + std::to_string(vectors_footprint) + "; ";
        }

        {
            size_t layers_vectors_footprint = 0;

            for (const auto &x: wrapped.index.nodes) {
                layers_vectors_footprint += sizeof(*x.second.layers.begin()) * x.second.layers.size();
            }

            result += "layers vectors: " + std::to_string(layers_vectors_footprint) + "; ";
        }

        {
            size_t incoming_links_footprint = 0;

            for (const auto &x: wrapped.index.nodes) {
                for (const auto &layer: x.second.layers) {
                    incoming_links_footprint += sizeof(*layer.incoming.begin()) * layer.incoming.size();
                }
            }

            result += "incoming links: " + std::to_string(incoming_links_footprint) + "; ";
        }

        {
            size_t outgoing_links_footprint = 0;

            for (const auto &x: wrapped.index.nodes) {
                for (const auto &layer: x.second.layers) {
                    outgoing_links_footprint += sizeof(*layer.outgoing.begin()) * layer.outgoing.size();
                }
            }

            result += "outgoing links: " + std::to_string(outgoing_links_footprint) + "; ";
        }

        result += "key_to_internal: " + std::to_string(sizeof(*wrapped.key_to_internal.begin()) * wrapped.key_to_internal.size()) + "; ";
        result += "internal_to_key: " + std::to_string(sizeof(*wrapped.internal_to_key.begin()) * wrapped.internal_to_key.size()) + "; ";

        {
            size_t keys_footprint = 0;

            for (const auto &x: wrapped.key_to_internal) {
                keys_footprint += x.first.size();
            }

            for (const auto &x: wrapped.internal_to_key) {
                keys_footprint += x.second.size();
            }

            result += "keys: " + std::to_string(keys_footprint) + "; ";
        }

        return result;
    }
};


inline std::unique_ptr<index_t>
make_index(std::string type,
           boost::optional<size_t> max_links,
           boost::optional<size_t> ef_construction,
           boost::optional<std::string> insert_method,
           boost::optional<std::string> remove_method)
{
    hnsw::index_options_t options;

    if (max_links) {
        options.max_links = *max_links;
    }

    if (ef_construction) {
        options.ef_construction = *ef_construction;
    }

    if (insert_method && *insert_method == "link_nearest") {
        options.insert_method = hnsw::index_options_t::insert_method_t::link_nearest;
    } else if (insert_method && *insert_method == "link_diverse") {
        options.insert_method = hnsw::index_options_t::insert_method_t::link_diverse;
    } else if (insert_method) {
        throw std::runtime_error("make_index: unknown insert method: " + *insert_method);
    }

    if (remove_method && *remove_method == "no_link") {
        options.remove_method = hnsw::index_options_t::remove_method_t::no_link;
    } else if (remove_method && *remove_method == "compensate_incomming_links") {
        options.remove_method = hnsw::index_options_t::remove_method_t::compensate_incomming_links;
    } else if (remove_method) {
        throw std::runtime_error("make_index: unknown remove method: " + *insert_method);
    }

    if (type == "dot_product") {
        using hnsw_index_t = hnsw::key_mapper<std::string, hnsw::hnsw_index<uint32_t, vector_t, hnsw::dot_product_distance_t>>;
        auto index = std::make_unique<hnsw_index<hnsw_index_t, true>>();
        index->wrapped.index.options = options;
        return std::unique_ptr<index_t>(std::move(index));
    } else if (type == "cosine") {
        using hnsw_index_t = hnsw::key_mapper<std::string, hnsw::hnsw_index<uint32_t, vector_t, hnsw::cosine_distance_t>>;
        auto index = std::make_unique<hnsw_index<hnsw_index_t, false>>();
        index->wrapped.index.options = options;
        return std::unique_ptr<index_t>(std::move(index));
    } else {
        throw std::runtime_error("make_index: unknown index type: " + type);
    }
}
