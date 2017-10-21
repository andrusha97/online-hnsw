#include <catch.hpp>

#include <hnsw/distance.hpp>
#include <hnsw/index.hpp>

#include <random>
#include <vector>


namespace {

template<class Random>
std::vector<float> random_vector(size_t size, Random &engine) {
    std::uniform_real_distribution<float> generator(0.0, 1.0);
    std::vector<float> result(size);

    for (auto &v: result) {
        v = generator(engine);
    }

    return result;
}

}


TEST_CASE("hnsw with cosine distance compiles") {
    using index_t = hnsw::hnsw_index<std::string, std::vector<float>, hnsw::cosine_distance_t>;

    index_t index;
    std::minstd_rand random;

    index.insert("aaa", random_vector(100, random));
    index.insert("bbb", random_vector(100, random));
    index.insert("def", random_vector(100, random));
    index.insert("fgh", random_vector(100, random));
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 4);
    REQUIRE(!index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);

    index.remove("bbb");
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 3);
    REQUIRE(!index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);

    index.insert("123", random_vector(100, random));
    index.insert("456", random_vector(100, random));
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 5);
    REQUIRE(!index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);

    index.remove("fgh");
    index.remove("def");
    index.remove("456");
    index.remove("aaa");
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 1);
    REQUIRE(!index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);

    index.remove("123");
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 0);
    REQUIRE(index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);
}


TEST_CASE("hnsw with dot product distance compiles") {
    using index_t = hnsw::hnsw_index<std::string, std::vector<float>, hnsw::dot_product_distance_t>;

    index_t index;
    std::minstd_rand random;

    index.insert("aaa", random_vector(100, random));
    index.insert("bbb", random_vector(100, random));
    index.insert("def", random_vector(100, random));
    index.insert("fgh", random_vector(100, random));
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 4);
    REQUIRE(!index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);

    index.remove("bbb");
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 3);
    REQUIRE(!index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);

    index.insert("123", random_vector(100, random));
    index.insert("456", random_vector(100, random));
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 5);
    REQUIRE(!index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);

    index.remove("fgh");
    index.remove("def");
    index.remove("456");
    index.remove("aaa");
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 1);
    REQUIRE(!index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);

    index.remove("123");
    REQUIRE(index.check());
    REQUIRE(index.nodes.size() == 0);
    REQUIRE(index.levels.empty());

    static_cast<const index_t &>(index).search(random_vector(100, random), 10);
}
