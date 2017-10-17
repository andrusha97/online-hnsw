#pragma once

#include "detail/dot_product.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace hnsw {


struct cosine_distance_t {
    template<class Vector>
    auto operator()(const Vector &one, const Vector &another) const {
        if (one.size() != another.size()) {
            throw std::runtime_error("cosine_distance_t: vectors sizes do not match");
        }

        using result_type = decltype(*one.data());

        result_type product = detail::dot_product(one.data(), another.data(), one.size())
                            / std::sqrt(detail::dot_product(one.data(), one.data(), one.size()))
                            / std::sqrt(detail::dot_product(another.data(), another.data(), one.size()));

        return std::max(result_type(0), result_type(result_type(1.0) - product));
    }
};


// Normalize your vectors before putting them in the index or searching, when using this distance.
struct dot_product_distance_t {
    template<class Vector>
    auto operator()(const Vector &one, const Vector &another) const {
        if (one.size() != another.size()) {
            throw std::runtime_error("dot_product_distance_t: vectors sizes do not match");
        }

        using result_type = decltype(*one.data());

        result_type product = detail::dot_product(one.data(), another.data(), one.size());

        return std::max(result_type(0), result_type(result_type(1.0) - product));
    }
};


}
