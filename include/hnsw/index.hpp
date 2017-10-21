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

#include "containers/flat_map.hpp"
#include "containers/hopscotch-map-1.4.0/src/hopscotch_map.h"
#include "containers/hopscotch-map-1.4.0/src/hopscotch_set.h"
#include "containers/small_set.hpp"
#include "detail/detail.hpp"
#include "prefetch.hpp"
#include "options.hpp"

#include "detail/undef_hopscotch_macros.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <map>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>


namespace hnsw {


/** Implementation of the HNSW index.
 *
 *  Key - Must be hashable with std::hash, comparable with ==, !=, and <, and copyable.
 *        For performance reasons you want it to be as lightweight as possible (uint32_t or uint64_t).
 *        If your keys are heavy, consider mapping them on more lightweight type somehow (for example, using the `key_mapper`).
 *
 *  Vector - Must be copyable, this is the only requirement.
 *           You might want to specialize the `prefetch` class for your vectors to improve performance.
 *           It's already specialized for std::vector<float> and std::vector<double> on x86 gcc.
 *
 *  Distance - Must be default-constructible. Must have a constant operator() which accepts two constant vectors
 *             and returns distance between them. Choose the return type wisely because it will be used
 *             to store distances in the index and will affect memory usage and speed.
 *             This means, if float is enough, make sure to not return double accidentally.
 *
 *  Random - Must be default-constructible and satisfy UniformRandomBitGenerator concept.
 *
 */
template<class Key,
         class Vector,
         class Distance,
         class Random = std::minstd_rand>
struct hnsw_index {
    using key_t = Key;
    using scalar_t = decltype(std::declval<Distance>()(std::declval<Vector>(), std::declval<Vector>()));
    using distance_t = Distance;
    using vector_t = Vector;
    using random_t = Random;

    struct search_result_t {
        key_t key;
        scalar_t distance;
    };

    struct node_t {
        using outgoing_links_t = flat_map<key_t, scalar_t>;
        using incoming_links_t = small_set<key_t>;

        struct layer_t {
            outgoing_links_t outgoing;
            incoming_links_t incoming;
        };

        vector_t vector;
        std::vector<layer_t> layers;
    };


    index_options_t options;
    distance_t distance;
    random_t random;

    tsl::hopscotch_map<key_t, node_t> nodes;

    // For levels order of keys is important, so it's std::map.
    std::map<size_t, tsl::hopscotch_set<key_t>> levels;

private:
    using closest_queue_t = std::priority_queue<
        std::pair<key_t, scalar_t>,
        std::vector<std::pair<key_t, scalar_t>>,
        detail::search_result_further_t
    >;

    using furthest_queue_t = std::priority_queue<
        std::pair<key_t, scalar_t>,
        std::vector<std::pair<key_t, scalar_t>>,
        detail::search_result_closer_t
    >;

public:
    void insert(const key_t &key, const vector_t &vector) {
        insert(key, vector_t(vector));
    }

    void insert(const key_t &key, vector_t &&vector) {
        if (nodes.count(key) > 0) {
            throw std::runtime_error("hnsw_index::insert: key already exists");
        }

        size_t node_level = random_level() + 1;

        auto node_it = nodes.emplace(key, node_t {
            std::move(vector),
            std::vector<typename node_t::layer_t>(node_level)
        }).first;

        for (size_t layer = 0; layer < node_level; ++layer) {
            node_it.value().layers[layer].outgoing.reserve(max_links(layer));
            node_it.value().layers[layer].incoming.reserve(max_links(layer));
        }

        if (nodes.size() == 1) {
            levels[node_level].insert(key);
            return;
        }

        key_t start = *levels.rbegin()->second.begin();

        for (size_t layer = nodes.at(start).layers.size(); layer > 0; --layer) {
            start = greedy_search(node_it->second.vector, layer - 1, start);

            if (layer <= node_level) {
                detail::priority_queue<furthest_queue_t> results;
                search_level(node_it->second.vector,
                             options.ef_construction,
                             layer - 1,
                             {start},
                             results);

                std::sort(results.c.begin(), results.c.end(), [](const auto &l, const auto &r) { return l.second < r.second; });
                set_links(key, layer - 1, results.c);

                // NOTE: Here we attempt to link all candidates to the new item.
                // The original HNSW attempts to link only with the actual neighbors.
                for (const auto &peer: results.c) {
                    try_add_link(peer.first, layer - 1, key, peer.second);
                }
            }
        }

        levels[node_level].insert(key);
    }


    void remove(const key_t &key) {
        auto node_it = nodes.find(key);

        if (node_it == nodes.end()) {
            return;
        }

        const auto &layers = node_it->second.layers;

        for (size_t layer = 0; layer < layers.size(); ++layer) {
            for (const auto &link: layers[layer].outgoing) {
                nodes.at(link.first).layers.at(layer).incoming.erase(key);
            }

            for (const auto &link: layers[layer].incoming) {
                nodes.at(link).layers.at(layer).outgoing.erase(key);
            }
        }

        if (options.remove_method != index_options_t::remove_method_t::no_link) {
            for (size_t layer = 0; layer < layers.size(); ++layer) {
                for (const auto &inverted_link: layers[layer].incoming) {
                    auto &peer_links = nodes.at(inverted_link).layers.at(layer).outgoing;
                    const key_t *new_link_ptr = nullptr;

                    if (options.insert_method == index_options_t::insert_method_t::link_nearest) {
                        new_link_ptr = select_nearest_link(inverted_link, peer_links, layers.at(layer).outgoing);
                    } else if (options.insert_method == index_options_t::insert_method_t::link_diverse) {
                        new_link_ptr = select_most_diverse_link(inverted_link, peer_links, layers.at(layer).outgoing);
                    } else {
                        assert(false);
                    }

                    if (new_link_ptr) {
                        auto new_link = *new_link_ptr;
                        auto &new_link_node = nodes.at(new_link);
                        auto d = distance(nodes.at(inverted_link).vector, new_link_node.vector);
                        peer_links.emplace(new_link, d);
                        new_link_node.layers.at(layer).incoming.insert(inverted_link);
                        try_add_link(new_link, layer, inverted_link, d);
                    }
                }
            }
        }

        auto level_it = levels.find(layers.size());

        if (level_it == levels.end()) {
            throw std::runtime_error("hnsw_index::remove: the node is not present in the levels index");
        }

        level_it->second.erase(key);

        // Shrink the hash table when it becomes too sparse
        // (to reduce memory usage and ensure linear complexity for iteration).
        if (4 * level_it->second.load_factor() < level_it->second.max_load_factor()) {
            level_it->second.rehash(size_t(2 * level_it->second.size() / level_it->second.max_load_factor()));
        }

        if (level_it->second.empty()) {
            levels.erase(level_it);
        }

        nodes.erase(node_it);

        if (4 * nodes.load_factor() < nodes.max_load_factor()) {
            nodes.rehash(size_t(2 * nodes.size() / nodes.max_load_factor()));
        }
    }


    std::vector<search_result_t> search(const vector_t &target, size_t nearest_neighbors) const {
        return search(target, nearest_neighbors, 100 + nearest_neighbors);
    }


    std::vector<search_result_t> search(const vector_t &target, size_t nearest_neighbors, size_t ef) const {
        if (nodes.empty()) {
            return {};
        }

        key_t start = *levels.rbegin()->second.begin();

        for (size_t layer = nodes.at(start).layers.size(); layer > 0; --layer) {
            start = greedy_search(target, layer - 1, start);
        }

        detail::priority_queue<furthest_queue_t> results;
        search_level(target, std::max(nearest_neighbors, ef), 0, {start}, results);

        size_t results_to_return = std::min(results.size(), nearest_neighbors);

        std::partial_sort(
            results.c.begin(),
            results.c.begin() + results_to_return,
            results.c.end(),
            [](const auto &l, const auto &r) {
                return l.second < r.second;
            }
        );

        std::vector<search_result_t> results_vector;
        results_vector.reserve(results_to_return);

        for (size_t i = 0; i < results_to_return; ++i) {
            results_vector.push_back({results.c[i].first, results.c[i].second});
        }

        return results_vector;
    }


    // Check whether the index satisfies its invariants.
    bool check() const {
        if (nodes.empty()) {
            return levels.empty();
        }

        for (const auto &node: nodes) {
            auto level_it = levels.find(node.second.layers.size());

            if (level_it == levels.end()) {
                return false;
            }

            if (level_it->second.count(node.first) == 0) {
                return false;
            }

            for (size_t layer = 0; layer < node.second.layers.size(); ++layer) {
                const auto &links = node.second.layers[layer].outgoing;

                // Self-links are not allowed.
                if (links.count(node.first) > 0) {
                    return false;
                }

                for (const auto &link: links) {
                    auto peer_node_it = nodes.find(link.first);

                    if (peer_node_it == nodes.end()) {
                        return false;
                    }

                    if (layer >= peer_node_it->second.layers.size()) {
                        return false;
                    }

                    if (peer_node_it->second.layers.at(layer).incoming.count(node.first) == 0) {
                        return false;
                    }
                }

                for (const auto &link: node.second.layers[layer].incoming) {
                    auto peer_node_it = nodes.find(link);

                    if (peer_node_it == nodes.end()) {
                        return false;
                    }

                    if (layer >= peer_node_it->second.layers.size()) {
                        return false;
                    }

                    if (peer_node_it->second.layers.at(layer).outgoing.count(node.first) == 0) {
                        return false;
                    }
                }
            }
        }

        for (const auto &level: levels) {
            for (const auto &key: level.second) {
                auto node_it = nodes.find(key);

                if (node_it == nodes.end()) {
                    return false;
                }

                if (level.first != node_it->second.layers.size()) {
                    return false;
                }
            }
        }

        return true;
    }


private:
    size_t max_links(size_t level) const {
        return (level == 0) ? (2 * options.max_links) : options.max_links;
    }


    size_t random_level() {
        // I avoid use of uniform_real_distribution to control how many times random() is called.
        // This makes inserts reproducible across standard libraries.

        // NOTE: This works correctly for standard random engines because their value_type is required to be unsigned.
        auto sample = random() - random_t::min();
        auto max_rand = random_t::max() - random_t::min();

        // If max_rand is too large, decrease it so that it can be represented by double.
        if (max_rand > 1048576) {
            sample /= max_rand / 1048576;
            max_rand /= max_rand / 1048576;
        }

        double x = std::min(1.0, std::max(0.0, double(sample) / double(max_rand)));
        return static_cast<size_t>(-std::log(x) / std::log(double(options.max_links + 1)));
    }


    void search_level(const vector_t &target,
                      size_t results_number,
                      size_t layer,
                      const std::vector<key_t> &start_from,
                      furthest_queue_t &results) const
    {
        tsl::hopscotch_set<key_t> visited_nodes;
        visited_nodes.reserve(5 * max_links(layer) * results_number);
        visited_nodes.insert(start_from.begin(), start_from.end());

        detail::priority_queue<closest_queue_t> search_front;

        for (const auto &key: start_from) {
            auto d = distance(target, nodes.at(key).vector);
            results.push({key, d});
            search_front.push({key, d});
        }

        while (results.size() > results_number) {
            results.pop();
        }

        for (size_t hop = 0; !search_front.empty() && search_front.top().second <= results.top().second && hop < nodes.size(); ++hop) {
            const auto &node = nodes.at(search_front.top().first);
            search_front.pop();

            const auto &links = node.layers.at(layer).outgoing;

            for (auto it = links.rbegin(); it != links.rend(); ++it) {
                if (visited_nodes.count(it->first) == 0) {
                    prefetch<vector_t>::pref(nodes.at(it->first).vector);
                }
            }

            for (const auto &link: links) {
                if (visited_nodes.insert(link.first).second) {
                    auto d = distance(target, nodes.at(link.first).vector);

                    if (results.size() < results_number) {
                        results.push({link.first, d});
                        search_front.push({link.first, d});
                    } else if (d < results.top().second) {
                        results.pop();
                        results.push({link.first, d});
                        search_front.push({link.first, d});
                    }
                }
            }

            // Try to make search_front smaller, so to speed up operations on it.
            while (!search_front.empty() && search_front.c.back().second > results.top().second) {
                search_front.c.pop_back();
            }
        }
    }


    key_t greedy_search(const vector_t &target, size_t layer, const key_t &start_from) const {
        key_t result = start_from;
        scalar_t result_distance = distance(target, nodes.at(start_from).vector);

        // Just a reasonable upper limit on the number of hops to avoid infinite loops.
        for (size_t hops = 0; hops < nodes.size(); ++hops) {
            const auto &node = nodes.at(result);
            bool made_hop = false;

            const auto &links = node.layers.at(layer).outgoing;

            for (auto it = links.begin(); it != links.end(); ++it) {
                if (it + 1 != links.end()) {
                    prefetch<vector_t>::pref(nodes.at((it + 1)->first).vector);
                }

                scalar_t neighbor_distance = distance(target, nodes.at(it->first).vector);

                if (neighbor_distance < result_distance) {
                    result = it->first;
                    result_distance = neighbor_distance;
                    made_hop = true;
                }
            }

            if (!made_hop) {
                break;
            }
        }

        return result;
    }


    void try_add_link(const key_t &node,
                      size_t layer,
                      const key_t &new_link,
                      scalar_t link_distance)
    {
        auto &layer_links = nodes.at(node).layers.at(layer).outgoing;

        if (layer_links.size() < max_links(layer)) {
            layer_links.emplace(new_link, link_distance);
            nodes.at(new_link).layers.at(layer).incoming.insert(node);
            return;
        }

        if (options.insert_method == index_options_t::insert_method_t::link_nearest) {
            auto furthest_key = layer_links.begin()->first;
            auto furthest_distance = layer_links.begin()->second;

            for (auto it = layer_links.begin() + 1; it < layer_links.end(); ++it) {
                if (it->first == new_link) {
                    return;
                }

                if (it->second > furthest_distance) {
                    furthest_key = it->first;
                    furthest_distance = it->second;
                }
            }

            if (link_distance < furthest_distance) {
                layer_links.erase(furthest_key);
                nodes.at(furthest_key).layers.at(layer).incoming.erase(node);
                layer_links.emplace(new_link, link_distance);
                nodes.at(new_link).layers.at(layer).incoming.insert(node);
            }

            return;
        }

        std::vector<std::pair<key_t, scalar_t>> sorted_links(layer_links.begin(), layer_links.end());

        std::sort(sorted_links.begin(),
                  sorted_links.end(),
                  [](const auto &l, const auto &r) { return l.second < r.second; });

        if (link_distance >= sorted_links.back().second) {
            return;
        }

        bool insert = true;
        size_t replace_index = sorted_links.size() - 1;
        const auto &new_link_vector = nodes.at(new_link).vector;

        for (const auto &link: sorted_links) {
            if (link.first == new_link) {
                insert = false;
                break;
            }
        }

        if (insert) {
            for (size_t i = 0; i < sorted_links.size(); ++i) {
                if (i + 1 < sorted_links.size()) {
                    prefetch<vector_t>::pref(nodes.at(sorted_links[i + 1].first).vector);
                }

                if (link_distance >= sorted_links[i].second) {
                    if (link_distance > distance(new_link_vector, nodes.at(sorted_links[i].first).vector)) {
                        insert = false;
                        break;
                    }
                } else if (replace_index > i) {
                    if (sorted_links[i].second > distance(new_link_vector, nodes.at(sorted_links[i].first).vector)) {
                        replace_index = i;
                    }
                }
            }
        }

        if (insert) {
            nodes.at(sorted_links.at(replace_index).first).layers.at(layer).incoming.erase(node);
            nodes.at(new_link).layers.at(layer).incoming.insert(node);
            layer_links.erase(sorted_links.at(replace_index).first);
            layer_links.emplace(new_link, link_distance);
        }
    }


    // new_links_set - *sorted by distance to the node* sequence of unique elements
    void set_links(const key_t &node,
                   size_t layer,
                   const std::vector<std::pair<key_t, scalar_t>> &new_links_set)
    {
        size_t need_links = max_links(layer);
        std::vector<std::pair<key_t, scalar_t>> new_links;
        new_links.reserve(need_links);

        if (options.insert_method == index_options_t::insert_method_t::link_nearest) {
            new_links.assign(
                new_links_set.begin(),
                new_links_set.begin() + std::min(new_links_set.size(), need_links)
            );
        } else {
            select_diverse_links(max_links(layer), new_links_set, new_links);
        }

        auto &outgoing_links = nodes.at(node).layers.at(layer).outgoing;

        for (const auto &link: outgoing_links) {
            nodes.at(link.first).layers.at(layer).incoming.erase(node);
        }

        std::sort(new_links.begin(), new_links.end(), [](const auto &l, const auto &r) { return l.first < r.first; });
        outgoing_links.assign_ordered_unique(new_links.begin(), new_links.end());

        for (const auto &key: new_links) {
            nodes.at(key.first).layers.at(layer).incoming.insert(node);
        }
    }


    void select_diverse_links(size_t links_number,
                              const std::vector<std::pair<key_t, scalar_t>> &candidates,
                              std::vector<std::pair<key_t, scalar_t>> &result) const
    {
        std::vector<const vector_t *> links_vectors;
        links_vectors.reserve(links_number);

        std::vector<std::pair<key_t, scalar_t>> rejected;
        rejected.reserve(links_number);

        for (const auto &candidate: candidates) {
            if (result.size() >= links_number) {
                break;
            }

            const auto &candidate_vector = nodes.at(candidate.first).vector;
            bool reject = false;

            for (const auto &link_vector: links_vectors) {
                if (distance(candidate_vector, *link_vector) < candidate.second) {
                    reject = true;
                    break;
                }
            }

            if (reject) {
                if (rejected.size() < links_number) {
                    rejected.push_back(candidate);
                }
            } else {
                result.push_back(candidate);
                links_vectors.push_back(&candidate_vector);
            }
        }

        for (const auto &link: rejected) {
            if (result.size() >= links_number) {
                break;
            }

            result.push_back(link);
        }
    }


    const key_t *select_nearest_link(const key_t &link_to,
                                     const typename node_t::outgoing_links_t &existing_links,
                                     const typename node_t::outgoing_links_t &candidates) const
    {
        auto closest_key_it = candidates.end();
        scalar_t min_distance = 0;

        for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            if (it->first != link_to && existing_links.count(it->first) == 0) {
                auto d = distance(nodes.at(it->first).vector, nodes.at(link_to).vector);

                if (closest_key_it == candidates.end() || d < min_distance) {
                    closest_key_it = it;
                    min_distance = d;
                }
            }
        }

        if (closest_key_it == candidates.end()) {
            return nullptr;
        } else {
            return &closest_key_it->first;
        }
    }


    const key_t *select_most_diverse_link(const key_t &link_to,
                                          const typename node_t::outgoing_links_t &existing_links,
                                          const typename node_t::outgoing_links_t &candidates) const
    {
        std::vector<std::pair<const key_t *, scalar_t>> filtered;
        filtered.reserve(candidates.size());

        for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            if (it->first != link_to && existing_links.count(it->first) == 0) {
                filtered.push_back({
                    &it->first,
                    distance(nodes.at(link_to).vector, nodes.at(it->first).vector)
                });
            }
        }

        std::sort(filtered.begin(),
                  filtered.end(),
                  [](const auto &l, const auto &r) { return l.second < r.second; });

        auto to_insert = filtered.end();

        for (auto it = existing_links.rbegin(); it != existing_links.rend(); ++it) {
            prefetch<vector_t>::pref(nodes.at(it->first).vector);
        }

        for (auto candidate_it = filtered.begin(); candidate_it != filtered.end(); ++candidate_it) {
            bool good = true;

            for (auto existing_link = existing_links.begin(); existing_link < existing_links.end(); ++existing_link) {
                auto d = distance(nodes.at(existing_link->first).vector,
                                  nodes.at(*candidate_it->first).vector);

                if (d < candidate_it->second) {
                    good = false;
                    break;
                }
            }

            if (good) {
                to_insert = candidate_it;
                break;
            }
        }

        if (to_insert == filtered.end() && !filtered.empty()) {
            to_insert = filtered.begin();
        }

        if (to_insert == filtered.end()) {
            return nullptr;
        } else {
            return to_insert->first;
        }
    }
};


}
