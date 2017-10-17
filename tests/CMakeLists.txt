ADD_EXECUTABLE(hnsw-unittests
    it_compiles.cpp
    main.cpp
)

TARGET_INCLUDE_DIRECTORIES(hnsw-unittests BEFORE PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

TARGET_INCLUDE_DIRECTORIES(hnsw-unittests SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/foreign/catch-1.10.0
)

TARGET_COMPILE_OPTIONS(hnsw-unittests PRIVATE -std=c++14 -pedantic -pedantic-errors -Wall -Wextra -Werror)


ADD_TEST(NAME hnsw-unittests COMMAND hnsw-unittests)

ADD_CUSTOM_TARGET(check
    hnsw-unittests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
