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
#include <cstddef>
#include <utility>
#include <vector>


namespace hnsw {


template<class Key, class Value>
class flat_map {
public:
    using size_type = std::size_t;
    using key_type = Key;
    using mapped_type = Value;
    using value_type = std::pair<key_type, mapped_type>;
    using container_type = std::vector<value_type>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

private:
    struct compare_t {
        bool operator()(const key_type &l, const key_type &r) const {
            return l < r;
        }

        bool operator()(const value_type &l, const key_type &r) const {
            return l.first < r;
        }

        bool operator()(const key_type &l, const value_type &r) const {
            return l < r.first;
        }

        bool operator()(const value_type &l, const value_type &r) const {
            return l.first < r.first;
        }
    };

public:
    const_iterator cbegin() const {
        return m_values.cbegin();
    }

    const_iterator cend() const {
        return m_values.cend();
    }

    const_iterator begin() const {
        return m_values.begin();
    }

    const_iterator end() const {
        return m_values.end();
    }

    iterator begin() {
        return m_values.begin();
    }

    iterator end() {
        return m_values.end();
    }

    const_reverse_iterator crbegin() const {
        return m_values.crbegin();
    }

    const_reverse_iterator crend() const {
        return m_values.crend();
    }

    const_reverse_iterator rbegin() const {
        return m_values.rbegin();
    }

    const_reverse_iterator rend() const {
        return m_values.rend();
    }

    reverse_iterator rbegin() {
        return m_values.rbegin();
    }

    reverse_iterator rend() {
        return m_values.rend();
    }

    bool empty() const {
        return m_values.empty();
    }

    size_type size() const {
        return m_values.size();
    }

    size_type capacity() const {
        return m_values.capacity();
    }

    size_type count(const key_type &k) const {
        auto range = std::equal_range(m_values.begin(), m_values.end(), k, compare_t());
        return range.second - range.first;
    }

    bool has(const key_type &k) const {
        return std::binary_search(m_values.begin(), m_values.end(), k, compare_t());
    }

    template<class It>
    void assign_ordered_unique(It begin, It end) {
        clear();
        m_values.assign(begin, end);
    }

    std::pair<iterator, bool> insert(value_type &&new_value) {
        compare_t compare;
        auto it = std::lower_bound(m_values.begin(), m_values.end(), new_value, compare);

        if (it != m_values.end() && !compare(*it, new_value) && !compare(new_value, *it)) {
            return {it, false};
        }

        auto new_item_pos = it - m_values.begin();
        m_values.insert(it, std::move(new_value));

        return {m_values.begin() + new_item_pos, true};
    }

    std::pair<iterator, bool> insert(const value_type &new_value) {
        return insert(value_type(new_value));
    }

    template<class... Args>
    std::pair<iterator, bool> emplace(Args &&... args) {
        return insert(value_type(std::forward<Args>(args)...));
    }

    size_type erase(const key_type &k) {
        auto range = std::equal_range(m_values.begin(), m_values.end(), k, compare_t());
        size_type result = size_type(range.second - range.first);

        if (range.first != range.second) {
            m_values.erase(range.first, range.second);
        }

        return result;
    }

    void clear() {
        m_values.clear();
    }

    void reserve(size_type capacity) {
        m_values.reserve(capacity);
    }

private:
    container_type m_values;
};


}
