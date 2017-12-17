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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#define HNSW_HAVE_SSE2

namespace hnsw { namespace detail {


inline float cosine_sse2(const float *pVect1, const float *pVect2, std::size_t qty) {
    static_assert(sizeof(float) == 4, "Cannot use SIMD instructions with non-32-bit floats.");

    std::size_t qty16  = qty/16;
    std::size_t qty4  = qty/4;

    const float* pEnd1 = pVect1 + 16  * qty16;
    const float* pEnd2 = pVect1 + 4  * qty4;
    const float* pEnd3 = pVect1 + qty;

    __m128  v1, v2;
    __m128  sum_prod = _mm_set1_ps(0);
    __m128  sum_square1 = sum_prod;
    __m128  sum_square2 = sum_prod;

    while (pVect1 < pEnd1) {
        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod  = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        sum_square1  = _mm_add_ps(sum_square1, _mm_mul_ps(v1, v1));
        sum_square2  = _mm_add_ps(sum_square2, _mm_mul_ps(v2, v2));

        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod  = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        sum_square1  = _mm_add_ps(sum_square1, _mm_mul_ps(v1, v1));
        sum_square2  = _mm_add_ps(sum_square2, _mm_mul_ps(v2, v2));

        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod  = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        sum_square1  = _mm_add_ps(sum_square1, _mm_mul_ps(v1, v1));
        sum_square2  = _mm_add_ps(sum_square2, _mm_mul_ps(v2, v2));

        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod  = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        sum_square1  = _mm_add_ps(sum_square1, _mm_mul_ps(v1, v1));
        sum_square2  = _mm_add_ps(sum_square2, _mm_mul_ps(v2, v2));
    }

    while (pVect1 < pEnd2) {
        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod  = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        sum_square1  = _mm_add_ps(sum_square1, _mm_mul_ps(v1, v1));
        sum_square2  = _mm_add_ps(sum_square2, _mm_mul_ps(v2, v2));
    }

    float __attribute__((aligned(16))) TmpResProd[4];
    float __attribute__((aligned(16))) TmpResSquare1[4];
    float __attribute__((aligned(16))) TmpResSquare2[4];

    _mm_store_ps(TmpResProd, sum_prod);
    float sum = TmpResProd[0] + TmpResProd[1] + TmpResProd[2] + TmpResProd[3];
    _mm_store_ps(TmpResSquare1, sum_square1);
    float norm1 = TmpResSquare1[0] + TmpResSquare1[1] + TmpResSquare1[2] + TmpResSquare1[3];
    _mm_store_ps(TmpResSquare2, sum_square2);
    float norm2 = TmpResSquare2[0] + TmpResSquare2[1] + TmpResSquare2[2] + TmpResSquare2[3];

    while (pVect1 < pEnd3) {
        sum += (*pVect1) * (*pVect2);
        norm1 += (*pVect1) * (*pVect1);
        norm2 += (*pVect2) * (*pVect2);

        ++pVect1; ++pVect2;
    }

    const float eps = std::numeric_limits<float>::min() * 2;

    if (norm1 < eps) { /*
                        * This shouldn't normally happen for this space, but
                        * if it does, we don't want to get NANs
                        */
      if (norm2 < eps) return 1.0f;
      return 0.0f;
    }
    /*
     * Sometimes due to rounding errors, we get values > 1 or < -1.
     * This throws off other functions that use scalar product, e.g., acos
     */
    return std::max(-1.0f, std::min(1.0f, sum / std::sqrt(norm1) / std::sqrt(norm2)));
}


inline double cosine_sse2(const double *pVect1, const double *pVect2, std::size_t qty) {
    static_assert(sizeof(double) == 8, "Cannot use SIMD instructions with non-64-bit doubles.");

    std::size_t qty8 = qty/8;
    std::size_t qty2 = qty/2;

    const double* pEnd1 = pVect1 + 8 * qty8;
    const double* pEnd2 = pVect1 + 2 * qty2;
    const double* pEnd3 = pVect1 + qty;

    __m128d  v1, v2;
    __m128d  sum_prod = _mm_set1_pd(0);
    __m128d  sum_square1 = sum_prod;
    __m128d  sum_square2 = sum_prod;

    while (pVect1 < pEnd1) {
        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum_prod  = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));
        sum_square1 = _mm_add_pd(sum_square1, _mm_mul_pd(v1, v1));
        sum_square2 = _mm_add_pd(sum_square2, _mm_mul_pd(v2, v2));

        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum_prod  = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));
        sum_square1 = _mm_add_pd(sum_square1, _mm_mul_pd(v1, v1));
        sum_square2 = _mm_add_pd(sum_square2, _mm_mul_pd(v2, v2));

        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum_prod  = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));
        sum_square1 = _mm_add_pd(sum_square1, _mm_mul_pd(v1, v1));
        sum_square2 = _mm_add_pd(sum_square2, _mm_mul_pd(v2, v2));

        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum_prod  = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));
        sum_square1 = _mm_add_pd(sum_square1, _mm_mul_pd(v1, v1));
        sum_square2 = _mm_add_pd(sum_square2, _mm_mul_pd(v2, v2));

    }

    while (pVect1 < pEnd2) {
        v1   = _mm_loadu_pd(pVect1); pVect1 += 2;
        v2   = _mm_loadu_pd(pVect2); pVect2 += 2;
        sum_prod  = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));
        sum_square1 = _mm_add_pd(sum_square1, _mm_mul_pd(v1, v1));
        sum_square2 = _mm_add_pd(sum_square2, _mm_mul_pd(v2, v2));

    }

    double __attribute__((aligned(16))) TmpResProd[2];
    double __attribute__((aligned(16))) TmpResSquare1[2];
    double __attribute__((aligned(16))) TmpResSquare2[2];

    _mm_store_pd(TmpResProd, sum_prod);
    double sum = TmpResProd[0] + TmpResProd[1];

    _mm_store_pd(TmpResSquare1, sum_square1);
    double norm1 = TmpResSquare1[0] + TmpResSquare1[1];

    _mm_store_pd(TmpResSquare2, sum_square2);
    double norm2 = TmpResSquare2[0] + TmpResSquare2[1];

    while (pVect1 < pEnd3) {
        sum += (*pVect1) * (*pVect2);
        norm1 += (*pVect1) * (*pVect1);
        norm2 += (*pVect2) * (*pVect2);

        ++pVect1; ++pVect2;
    }

    const double eps = std::numeric_limits<double>::min() * 2;

    if (norm1 < eps) { /*
                        * This shouldn't normally happen for this space, but
                        * if it does, we don't want to get NANs
                        */
      if (norm2 < eps) return 1;
      return 0;
    }
    /*
     * Sometimes due to rounding errors, we get values > 1 or < -1.
     * This throws off other functions that use scalar product, e.g., acos
     */
    return std::max(-1.0, std::min(1.0, sum / std::sqrt(norm1) / std::sqrt(norm2)));
}


}} // namespace hnsw::detail

#endif // __SSE2__
#endif
