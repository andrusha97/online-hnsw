#pragma once

#include <cstddef>


namespace hnsw {


struct index_options_t {
    std::size_t max_links = 32;
    std::size_t ef_construction = 200;

    enum class insert_method_t {
        link_nearest,
        link_diverse
    };

    insert_method_t insert_method = insert_method_t::link_diverse;

    enum class remove_method_t {
        no_link,
        compensate_incomming_links
    };

    remove_method_t remove_method = remove_method_t::compensate_incomming_links;
};


}
