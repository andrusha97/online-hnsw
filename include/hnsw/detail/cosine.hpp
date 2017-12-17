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

#include "cosine_sse2.hpp"

#include <algorithm>
#include <cmath>
#include <limits>


namespace hnsw { namespace detail {


template<class T>
T cosine(const T *one, const T *another, std::size_t size) {
    T sum = 0;
    T sum_one = 0;
    T sum_another = 0;

    for (std::size_t i = 0; i < size; ++i) {
        sum += one[i] * another[i];
        sum_one += one[i] * one[i];
        sum_another += another[i] * another[i];
    }

    if (sum_one < 2 * std::numeric_limits<T>::min()) {
        if (sum_another < 2 * std::numeric_limits<T>::min()) {
            return T(1.0);
        } else {
            return T(0.0);
        }
    }

    return std::max(T(-1.0), std::min(T(1.0), sum / std::sqrt(sum_one) / std::sqrt(sum_another)));
}


#if defined(HNSW_HAVE_SSE2)

inline float cosine(const float *pVect1, const float *pVect2, std::size_t qty) {
    return cosine_sse2(pVect1, pVect2, qty);
}


inline double cosine(const double *pVect1, const double *pVect2, std::size_t qty) {
    return cosine_sse2(pVect1, pVect2, qty);
}

#endif


}}


#ifdef HNSW_HAVE_SSE2
#undef HNSW_HAVE_SSE2
#endif
