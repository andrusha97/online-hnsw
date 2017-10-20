#pragma once

#include "common.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <type_traits>


inline std::vector<std::string> find_knn_fullscan(const dataset_t &haystack, const vector_t &needle, size_t n) {
    std::priority_queue<std::pair<float, std::string>> knn;

    for (const auto &item: haystack) {
        auto distance = hnsw::dot_product_distance_t()(item.second, needle);

        if (knn.size() < n || distance < knn.top().first) {
            knn.push({distance, item.first});
        }

        if (knn.size() > n) {
            knn.pop();
        }
    }

    std::vector<std::string> result(knn.size());

    for (size_t i = knn.size(); i > 0; --i) {
        result[i - 1] = knn.top().second;
        knn.pop();
    }

    return result;
}


inline std::map<std::string, std::vector<std::string>>
compute_reference(const dataset_t &data, const dataset_t &control, size_t n, size_t threads) {
    std::map<std::string, std::vector<std::string>> reference;
    std::mutex reference_mutex;

    std::vector<std::thread> workers;
    size_t batch_size = std::max<size_t>(1, control.size() / threads);

    for (size_t batch_start = 0; batch_start < control.size(); batch_start += batch_size) {
        workers.push_back(std::thread([&, batch_start]() {
            std::vector<std::pair<std::string, std::vector<std::string>>> local_reference;
            for (size_t i = batch_start; i < batch_start + batch_size && i < control.size(); ++i) {
                local_reference.emplace_back(
                    control.at(i).first,
                    find_knn_fullscan(data, control.at(i).second, n)
                );
            }

            std::lock_guard<std::mutex> lock(reference_mutex);
            reference.insert(local_reference.begin(), local_reference.end());
        }));
    }

    for (auto &worker: workers) {
        worker.join();
    }

    return reference;
}


template<class It1, class It2>
size_t intersection_size(It1 begin1, It1 end1, It2 begin2, It2 end2) {
    std::set<typename std::decay<decltype(*begin1)>::type> set1(begin1, end1);

    return (size_t)std::count_if(begin2, end2, [&](const auto &x) { return set1.count(x) > 0; });
}


inline void assess_index(const index_t &index, const dataset_t &vectors, const dataset_t &control, size_t threads) {
    auto reference = compute_reference(vectors, control, 10, threads);

    size_t rank1 = 0;
    size_t rank10 = 0;
    size_t rank10_base = 0;

    for (const auto &item: control) {
        auto neighbor = index.search(item.second, 1);
        const auto &ref_neighbors = reference.at(item.first);

        if (!neighbor.empty() && neighbor.front().key == ref_neighbors.front()) {
            ++rank1;
        }

        auto neighbors10 = index.search(item.second, 10);
        std::vector<std::string> neighbors10_keys;

        for (const auto &n: neighbors10) {
            neighbors10_keys.push_back(n.key);
        }

        rank10 += intersection_size(ref_neighbors.begin(),
                                    ref_neighbors.begin() + std::min(10, int(ref_neighbors.size())),
                                    neighbors10_keys.begin(),
                                    neighbors10_keys.end());

        rank10_base += ref_neighbors.size();
    }


    LOG << "Rank 1: " << rank1 << "/" << control.size() << " = " << double(rank1) / double(control.size())
        << ", rank 10: " << rank10 << "/" << rank10_base << " = " << double(rank10) / double(rank10_base);
}
