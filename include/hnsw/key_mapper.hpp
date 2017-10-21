/* Copyright 2017 Andrey Goryachev

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

#include "index.hpp"

#include "containers/hopscotch-map-1.4.0/src/hopscotch_map.h"

#include "detail/undef_hopscotch_macros.hpp"

#include <limits>
#include <random>
#include <type_traits>


namespace hnsw {


template<class Key, class Index, class Random = std::minstd_rand>
class key_mapper {
public:
    using key_t = Key;
    using internal_key_t = typename Index::key_t;
    using scalar_t = typename Index::scalar_t;
    using vector_t = typename Index::vector_t;
    using index_t = Index;
    using random_t = Random;

    static_assert(std::is_integral<internal_key_t>::value, "Cannot map on non-integral keys.");

    struct search_result_t {
        key_t key;
        scalar_t distance;
    };

    random_t random;
    index_t index;
    tsl::hopscotch_map<key_t, internal_key_t> key_to_internal;
    tsl::hopscotch_map<internal_key_t, key_t> internal_to_key;

public:
    void insert(const key_t &key, const vector_t &vector) {
        insert(key, vector_t(vector));
    }

    void insert(const key_t &key, vector_t &&vector) {
        auto internal_key = allocate_internal_key();

        if (!key_to_internal.emplace(key, internal_key).second) {
            throw std::runtime_error("key_mapper::insert: key already exists");
        }

        internal_to_key.emplace(internal_key, key);
        index.insert(internal_key, std::move(vector));
    }

    void remove(const key_t &key) {
        auto key_it = key_to_internal.find(key);

        if (key_it == key_to_internal.end()) {
            return;
        }

        index.remove(key_it->second);
        internal_to_key.erase(key_it->second);
        key_to_internal.erase(key_it);

        // Shrink hash tables when they become too sparse
        // (to reduce memory usage and ensure linear complexity for iteration).
        if (4 * internal_to_key.load_factor() < internal_to_key.max_load_factor()) {
            internal_to_key.rehash(size_t(2 * internal_to_key.size() / internal_to_key.max_load_factor()));
        }

        if (4 * key_to_internal.load_factor() < key_to_internal.max_load_factor()) {
            key_to_internal.rehash(size_t(2 * key_to_internal.size() / key_to_internal.max_load_factor()));
        }
    }

    std::vector<search_result_t> search(const vector_t &target, std::size_t nearest_neighbors) const {
        return convert_search_results(index.search(target, nearest_neighbors));
    }

    std::vector<search_result_t> search(const vector_t &target, std::size_t nearest_neighbors, std::size_t ef) const {
        return convert_search_results(index.search(target, nearest_neighbors, ef));
    }

    bool check() const {
        if (!index.check()) {
            return false;
        }

        for (const auto &key: key_to_internal) {
            auto it = internal_to_key.find(key.second);

            if (it == internal_to_key.end()) {
                return false;
            }

            if (it->second != key.first) {
                return false;
            }
        }

        for (const auto &internal: internal_to_key) {
            auto it = key_to_internal.find(internal.second);

            if (it == key_to_internal.end()) {
                return false;
            }

            if (it->second != internal.first) {
                return false;
            }

            if (index.nodes.count(internal.first) == 0) {
                return false;
            }
        }

        return true;
    }

private:
    internal_key_t allocate_internal_key() {
        std::uniform_int_distribution<internal_key_t> generator(
            std::numeric_limits<internal_key_t>::min(),
            std::numeric_limits<internal_key_t>::max()
        );

        internal_key_t new_key = generator(random);

        while (internal_to_key.count(new_key) != 0) {
            ++new_key;
        }

        return new_key;
    }

    std::vector<search_result_t>
    convert_search_results(const std::vector<typename index_t::search_result_t> &internal_result) const {
        std::vector<search_result_t> result;
        result.reserve(internal_result.size());

        for (const auto &x: internal_result) {
            result.push_back({
                internal_to_key.at(x.key),
                x.distance
            });
        }

        return result;
    }
};


}
