#pragma once

#include <vector>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif


namespace hnsw {


template<class T, class = void>
struct prefetch {
    static void pref(const T &) {
        // Do nothing by default.
    }
};


#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
template<>
struct prefetch<std::vector<float>, void> {
    static void pref(const std::vector<float> &v) {
        _mm_prefetch(v.data(), _MM_HINT_T0);
    }
};

template<>
struct prefetch<std::vector<double>, void> {
    static void pref(const std::vector<double> &v) {
        _mm_prefetch(v.data(), _MM_HINT_T0);
    }
};
#endif


}
