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
