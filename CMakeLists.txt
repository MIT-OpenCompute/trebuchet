cmake_minimum_required(VERSION 3.10)
project(lessbasednn C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/tensor.c
    src/ops.c
    src/autograd.c
    src/layer.c
    src/network.c
    src/optimizer.c
)

# Create library
add_library(lessbasednn ${SOURCES})
target_link_libraries(lessbasednn m)

# Enable testing
enable_testing()

# Unit tests
set(TEST_SOURCES
    tests/unit/test_tensor.c
    tests/unit/test_ops.c
    tests/unit/test_autograd.c
    tests/unit/test_layer.c
    tests/unit/test_network.c
    tests/unit/test_optimizer.c
)

# Create individual test executables
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} lessbasednn m)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Full tests
add_executable(xor tests/full/xor.c)
target_link_libraries(xor lessbasednn m)
# add_executable(mnist tests/full/mnist.c)
# target_link_libraries(mnist lessbasednn m)