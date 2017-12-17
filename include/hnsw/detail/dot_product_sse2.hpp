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

#ifdef __SSE2__

#include <cstddef>

#define HNSW_HAVE_SSE2

namespace hnsw { namespace detail {


inline float dot_product_sse2(const float *pVect1, const float *pVect2, std::size_t qty) {
    static_assert(sizeof(float) == 4, "Cannot use SIMD instructions with non-32-bit floats.");

    std::size_t qty16  = qty/16;
    std::size_t qty4  = qty/4;

    const float* pEnd1 = pVect1 + 16  * qty16;
    const float* pEnd2 = pVect1 + 4  * qty4;
    const float* pEnd3 = pVect1 + qty;

    __m128  v1, v2;
    __m128  sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum  = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum  = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum  = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum  = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum  = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
    }

    float __attribute__((aligned(16))) TmpRes[4];

    _mm_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    while (pVect1 < pEnd3) {
        res += (*pVect1) * (*pVect2);
        ++pVect1; ++pVect2;
    }

    return res;
}


inline double dot_product_sse2(const double *pVect1, const double *pVect2, std::size_t qty) {
    static_assert(sizeof(double) == 8, "Cannot use SIMD instructions with non-64-bit doubles.");

    std::size_t qty8 = qty/8;
    std::size_t qty2 = qty/2;

    const double* pEnd1 = pVect1 + 8 * qty8;
    const double* pEnd2 = pVect1 + 2 * qty2;
    const double* pEnd3 = pVect1 + qty;

    __m128d  v1, v2;
    __m128d  sum = _mm_set1_pd(0);

    while (pVect1 < pEnd1) {
        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum  = _mm_add_pd(sum, _mm_mul_pd(v1, v2));

        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum  = _mm_add_pd(sum, _mm_mul_pd(v1, v2));

        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum  = _mm_add_pd(sum, _mm_mul_pd(v1, v2));

        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum  = _mm_add_pd(sum, _mm_mul_pd(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum  = _mm_add_pd(sum, _mm_mul_pd(v1, v2));
    }

    double __attribute__((aligned(16))) TmpRes[2];

    _mm_store_pd(TmpRes, sum);
    double res= TmpRes[0] + TmpRes[1];

    while (pVect1 < pEnd3) {
        res += (*pVect1) * (*pVect2);
        ++pVect1; ++pVect2;
    }

    return res;
}


}} // namespace hnsw::detail

#endif // __SSE2__
#endif
