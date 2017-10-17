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
