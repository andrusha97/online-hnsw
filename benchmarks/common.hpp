#pragma once

#include "log.hpp"

#include <hnsw/distance.hpp>
#include <hnsw/index.hpp>
#include <hnsw/key_mapper.hpp>

#include <boost/optional.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>


using random_t = std::minstd_rand;

using vector_t = std::vector<float>;
using dataset_t = std::vector<std::pair<std::string, vector_t>>;

using index_t = hnsw::key_mapper<std::string, hnsw::hnsw_index<uint32_t, vector_t, hnsw::dot_product_distance_t>>;


inline void shuffle(dataset_t &vectors, random_t &random) {
    std::shuffle(vectors.begin(), vectors.end(), random);
}


inline void normalize(dataset_t &vectors) {
    for (auto &v: vectors) {
        float coef = 1.0f / std::sqrt(hnsw::detail::dot_product(v.second.data(), v.second.data(), v.second.size()));

        for (auto &x: v.second) {
            x *= coef;
        }
    }
}


inline size_t get_control_size(const dataset_t &vectors, boost::optional<size_t> size) {
    if (size) {
        return *size;
    } else {
        return std::min(vectors.size(), std::max<size_t>(1, vectors.size() / 100));
    }
}

inline void split_dataset(dataset_t &main, dataset_t &control, size_t control_size) {
    control.assign(main.begin(), main.begin() + control_size);

    dataset_t new_main(main.begin() + control_size, main.end());
    main = std::move(new_main);
}
