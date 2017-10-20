#include "common.hpp"
#include "input.hpp"

#include <getRSS.h>

#include <boost/program_options.hpp>


void do_test(index_t &index, std::string input_file) {
    vectors_reader_t reader(input_file);

    LOG << "Building the index...";

    std::pair<std::string, vector_t> item;
    while (reader.read(item.first, item.second)) {
        index.insert(item.first, item.second);

        if (index.size() % 10000 == 0) {
            LOG << "Inserted " << index.size() << " vectors.";
        }
    }

    LOG << "Done. Index contains " << index.size() << " elements.";
    LOG << "RSS: " << getCurrentRSS();
}


int main(int argc, const char *argv[]) {
    namespace po = boost::program_options;

    std::string input_file;

    std::string index_type = "dot_product";
    boost::optional<size_t> max_links;
    boost::optional<size_t> ef_construction;
    boost::optional<std::string> insert_method;
    boost::optional<std::string> remove_method;

    po::options_description description("Available options");
    description.add_options()
        ("help,h", "print help message")
        ("index_type", po::value<std::string>(), "type of index (supported options: dot_product, cosine)")
        ("max_links", po::value<size_t>(), "index_options_t::max_links")
        ("ef_construction", po::value<size_t>(), "index_options_t::ef_construction")
        ("insert_method", po::value<std::string>(), "index_options_t::insert_method")
        ("remove_method", po::value<std::string>(), "index_options_t::remove_method")
        ("input-file", po::value<std::string>(), "file with vectors");

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, description), vm);
        po::notify(vm);

        if (vm.count("help") > 0) {
            std::cout << description << std::endl;
            return 0;
        }

        if (vm.count("index-type") > 0) {
            index_type = vm["index-type"].as<std::string>();
        }

        if (vm.count("max-links") > 0) {
            max_links = vm["max-links"].as<size_t>();
        }

        if (vm.count("ef-construction") > 0) {
            ef_construction = vm["ef-construction"].as<size_t>();
        }

        if (vm.count("insert-method") > 0) {
            insert_method = vm["insert-method"].as<std::string>();
        }

        if (vm.count("remove-method") > 0) {
            remove_method = vm["remove-method"].as<std::string>();
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


    do_test(
        *make_index(index_type, max_links, ef_construction, insert_method, remove_method),
        input_file
    );

    return 0;
}
