#include "common.hpp"
#include "input.hpp"

#include <getRSS.h>

#include <boost/program_options.hpp>


void do_test(std::string input_file) {
    vectors_reader_t reader(input_file);

    LOG << "Building the index...";

    index_t index;

    std::pair<std::string, vector_t> item;
    while (reader.read(item.first, item.second)) {
        index.insert(item.first, item.second);

        if (index.index.nodes.size() % 10000 == 0) {
            LOG << "Inserted " << index.index.nodes.size() << " vectors.";
        }
    }

    LOG << "Done. Index contains " << index.index.nodes.size() << " elements.";
    LOG << "RSS: " << getCurrentRSS();
}


int main(int argc, const char *argv[]) {
    namespace po = boost::program_options;

    std::string input_file;

    po::options_description description("Available options");
    description.add_options()
        ("help,h", "print help message")
        ("input-file", po::value<std::string>(), "file with vectors");

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, description), vm);
        po::notify(vm);

        if (vm.count("help") > 0) {
            std::cout << description << std::endl;
            return 0;
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


    do_test(input_file);

    return 0;
}
