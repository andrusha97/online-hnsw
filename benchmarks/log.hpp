#pragma once

#include <boost/date_time/posix_time/posix_time.hpp>

#include <iostream>
#include <mutex>


extern std::mutex log_mutex;


struct log_mutex_holder_t {
    std::unique_lock<std::mutex> lock {log_mutex};
    std::ostream *out = &std::cerr;

    ~log_mutex_holder_t() {
        *out << std::endl;
    }
};


inline std::string current_time() {
    return to_simple_string(boost::posix_time::second_clock::local_time());
}


#define LOG (*log_mutex_holder_t().out) << current_time() << ": "
