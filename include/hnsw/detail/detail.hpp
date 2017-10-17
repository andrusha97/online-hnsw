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

#include <algorithm>
#include <functional>
#include <queue>


namespace hnsw { namespace detail {


struct search_result_closer_t {
    template<class T>
    bool operator()(const T &l, const T &r) const {
        return l.second < r.second;
    }
};


struct search_result_further_t {
    template<class T>
    bool operator()(const T &l, const T &r) const {
        return l.second > r.second;
    }
};


template<class Base>
class priority_queue : public Base {
public:
    using base_type = Base;
    using Base::c;
};


}}
