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
