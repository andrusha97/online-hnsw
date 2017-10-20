#include "common.hpp"
#include "evaluate.hpp"
#include "input.hpp"

#include <hnsw/distance.hpp>
#include <hnsw/index.hpp>

#include <boost/optional.hpp>
#include <boost/program_options.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>


void measure_search_speed(const index_t &index, const dataset_t &control, size_t threads) {
    std::vector<std::thread> workers;
    std::atomic<size_t> processed {0};

    for (size_t i = 0; i < threads; ++i) {
        workers.push_back(std::thread([&]() {
            auto control_item = processed++;

            if (control_item < control.size()) {
                index.search(control[control_item].second, 10);
            }
        }));
    }

    while (processed.load() < control.size()) {
        LOG << "Performed " << processed.load() << " searches.";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for (auto &w: workers) {
        w.join();
    }
}


int main(int argc, const char *argv[]) {
    namespace po = boost::program_options;

    uint32_t seed = 0;
    std::string input_file;
    boost::optional<size_t> control_size;
    size_t threads = std::thread::hardware_concurrency();

    po::options_description description("Available options");
    description.add_options()
        ("help,h", "print help message")
        ("seed", po::value<uint32_t>(), "seed for the random numbers generator")
        ("threads", po::value<size_t>(), "how many threads to use")
        ("input-file", po::value<std::string>(), "file with vectors")
        ("control-size", po::value<size_t>(), "how many vectors to take out of the input to measure search accuracy");

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, description), vm);
        po::notify(vm);

        if (vm.count("help") > 0) {
            std::cout << description << std::endl;
            return 0;
        }

        if (vm.count("seed") > 0) {
            seed = vm["seed"].as<uint32_t>();
        }

        if (vm.count("threads") > 0) {
            threads = vm["threads"].as<size_t>();
        }

        if (vm.count("control-size") > 0) {
            control_size = vm["control-size"].as<size_t>();
        }

        if (vm.count("input-file") > 0) {
            input_file = vm["input-file"].as<std::string>();
        } else {
            throw std::runtime_error("please specify a file with input");
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << description << std::endl;
        return 1;
    }

    random_t random(seed);
    auto dataset = read_vectors(input_file);
    normalize(dataset);
    shuffle(dataset, random);

    dataset_t control;
    split_dataset(dataset, control, get_control_size(dataset, control_size));

    LOG << "Building the index...";

    index_t index;

    for (const auto &x: dataset) {
        index.insert(x.first, x.second);

        if (index.index.nodes.size() % 10000 == 0) {
            LOG << "Inserted " << index.index.nodes.size() << " vectors.";
        }
    }

    LOG << "Done. Index contains " << index.index.nodes.size() << " elements.";

    assess_index(index, dataset, control, threads);

    LOG << "Measuring search performance...";
    measure_search_speed(index, control, threads);

    shuffle(dataset, random);

    LOG << "Removing items. Index contains " << index.index.nodes.size() << " elements.";

    for (const auto &x: dataset) {
        index.remove(x.first);

        if (index.index.nodes.size() % 10000 == 0) {
            LOG << index.index.nodes.size() << " vectors are left.";
        }
    }

    LOG << "Done.";

    return 0;
}
