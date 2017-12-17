/**
 * This file contains code from the Non-metric Space Library
 *
 * Authors: Bilegsaikhan Naidan (https://github.com/bileg), Leonid Boytsov (http://boytsov.info).
 * With contributions from Lawrence Cayton (http://lcayton.com/) and others.
 *
 * For the complete list of contributors and further details see:
 * https://github.com/searchivarius/NonMetricSpaceLib
 *
 * Copyright (c) 2014
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#pragma once

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))

#include <x86intrin.h>

#ifdef __AVX__

#include <cstddef>

#define HNSW_HAVE_AVX

namespace hnsw { namespace detail {


inline float l2sqr_dist_avx(const float *pVect1, const float *pVect2, std::size_t qty) {
    static_assert(sizeof(float) == 4, "Cannot use SIMD instructions with non-32-bit floats.");

    std::size_t qty4  = qty/4;
    std::size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + 16 * qty16;
    const float* pEnd2 = pVect1 + 4  * qty4;
    const float* pEnd3 = pVect1 + qty;

    __m256  diff, v1, v2;
    __m256  sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1)
    {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

    }

    __m128  v1_128, v2_128, diff_128;
    __m128  sum_128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0),
                                 _mm256_extractf128_ps(sum, 1));

    while (pVect1 < pEnd2) {
        v1_128   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2_128   = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff_128 = _mm_sub_ps(v1_128, v2_128);
        sum_128  = _mm_add_ps(sum_128, _mm_mul_ps(diff_128, diff_128));
    }

    float __attribute__((aligned(16))) TmpRes[4];

    _mm_store_ps(TmpRes, sum_128);
    float res= TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    while (pVect1 < pEnd3) {
        float diff = *pVect1++ - *pVect2++;
        res += diff * diff;
    }

    return (res);
}


}} // namespace hnsw::detail

#endif // __AVX__
#endif
