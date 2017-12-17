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

#include "l2_dist_avx.hpp"
#include "l2_dist_sse2.hpp"


namespace hnsw { namespace detail {


template<class T>
T l2sqr_dist(const T *one, const T *another, std::size_t size) {
    T sum = 0;

    for (std::size_t i = 0; i < size; ++i) {
        auto diff = one[i] - another[i];
        sum += diff * diff;
    }

    return sum;
}


#if defined(HNSW_HAVE_AVX)

inline float l2sqr_dist(const float *pVect1, const float *pVect2, std::size_t qty) {
    return l2sqr_dist_avx(pVect1, pVect2, qty);
}

#elif defined(HNSW_HAVE_SSE2)

inline float l2sqr_dist(const float *pVect1, const float *pVect2, std::size_t qty) {
    return l2sqr_dist_sse2(pVect1, pVect2, qty);
}

#endif


#if defined(HNSW_HAVE_SSE2)

inline double l2sqr_dist(const double *pVect1, const double *pVect2, std::size_t qty) {
    return l2sqr_dist_sse2(pVect1, pVect2, qty);
}

#endif


}}


#ifdef HNSW_HAVE_AVX
#undef HNSW_HAVE_AVX
#endif

#ifdef HNSW_HAVE_SSE2
#undef HNSW_HAVE_SSE2
#endif
