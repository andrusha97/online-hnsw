#pragma once

#include "common.hpp"

#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


class vectors_reader_t {
public:
    explicit vectors_reader_t(const std::string &path) {
        m_input.open(path);

        if (!m_input) {
            throw std::runtime_error("vectors_reader_t: failed to open the input file");
        }
    }

    bool read(std::string &name, vector_t &vector) {
        if (!m_input) {
            return false;
        }

        std::string line;
        std::getline(m_input, line, '\n');

        if (line.empty()) {
            return false;
        }

        std::vector<std::string> parts;
        boost::algorithm::split(parts, line, [](const auto &x) { return x == ' '; }, boost::algorithm::token_compress_on);

        if (parts.empty()) {
            throw std::runtime_error("vectors_reader_t::read: failed to parse a line");
        }

        name = parts[0];

        vector.clear();
        vector.reserve(parts.size() - 1);

        for (auto it = parts.begin() + 1; it != parts.end(); ++it) {
            vector.push_back(boost::lexical_cast<float>(*it));
        }

        return true;
    }

private:
    std::ifstream m_input;
};


inline dataset_t read_vectors(const std::string &path) {
    vectors_reader_t reader(path);

    dataset_t result;

    std::pair<std::string, vector_t> item;
    while (reader.read(item.first, item.second)) {
        result.push_back(item);
    }

    return result;
}
