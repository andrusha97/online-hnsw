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

#include "dot_product_avx.hpp"
#include "dot_product_sse2.hpp"


namespace hnsw { namespace detail {


template<class T>
T dot_product(const T *one, const T *another, std::size_t size) {
    T sum = 0;

    for (std::size_t i = 0; i < size; ++i) {
        sum += one[i] * another[i];
    }

    return sum;
}


#if defined(HNSW_HAVE_AVX)

inline float dot_product(const float *pVect1, const float *pVect2, std::size_t qty) {
    return dot_product_avx(pVect1, pVect2, qty);
}

#elif defined(HNSW_HAVE_SSE2)

inline float dot_product(const float *pVect1, const float *pVect2, std::size_t qty) {
    return dot_product_sse2(pVect1, pVect2, qty);
}

#endif


#if defined(HNSW_HAVE_SSE2)

inline double dot_product(const double *pVect1, const double *pVect2, std::size_t qty) {
    return dot_product_sse2(pVect1, pVect2, qty);
}

#endif


}}


#ifdef HNSW_HAVE_AVX
#undef HNSW_HAVE_AVX
#endif

#ifdef HNSW_HAVE_SSE2
#undef HNSW_HAVE_SSE2
#endif
