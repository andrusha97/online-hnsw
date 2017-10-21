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


template<class T>
class small_set {
public:
    using size_type = std::size_t;
    using value_type = T;
    using container_type = std::vector<value_type>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

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

    size_type count(const value_type &v) const {
        return std::count(m_values.begin(), m_values.end(), v);
    }

    template<class It>
    void assign_unique(It begin, It end) {
        clear();
        m_values.assign(begin, end);
    }

    std::pair<iterator, bool> insert(value_type &&new_value) {
        auto it = std::find(m_values.begin(), m_values.end(), new_value);
        if (it != m_values.end()) {
            return {it, false};
        }

        if (m_values.capacity() == m_values.size()) {
            m_values.reserve(m_values.size() + 1);
        }

        m_values.push_back(std::move(new_value));
        return {m_values.begin() + m_values.size() - 1, true};
    }

    std::pair<iterator, bool> insert(const value_type &new_value) {
        return insert(value_type(new_value));
    }

    template<class... Args>
    std::pair<iterator, bool> emplace(Args &&... args) {
        return insert(value_type(std::forward<Args>(args)...));
    }

    size_type erase(const value_type &v) {
        if (m_values.empty()) {
            return 0;
        }

        size_type result = 0;
        auto l = std::find(m_values.begin(), m_values.end(), v);
        auto r = m_values.begin() + m_values.size() - 1;

        while (l <= r) {
            ++result;
            std::swap(*l, *r);
            --r;
            l = std::find(l + 1, m_values.end(), v);
        }

        m_values.resize(m_values.size() - result);
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
