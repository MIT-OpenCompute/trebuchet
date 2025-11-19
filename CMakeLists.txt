cmake_minimum_required(VERSION 3.10)
project(basednn C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")

# Backend options
option(ENABLE_WGPU "Enable WGPU GPU acceleration backend" OFF)

# Include directories
include_directories(core/include)

# Source files
set(SOURCES
    core/src/tensor.c
    core/src/ops.c
    core/src/layer.c
    core/src/network.c
    core/src/optimizer.c
    core/src/registry.c
)

# Create library
add_library(basednn ${SOURCES})
target_link_libraries(basednn m)

# WGPU backend (cross-platform GPU acceleration)
if(ENABLE_WGPU)
    set(WGPU_SOURCES
        backend/wgpu/wgpu_backend.c
        backend/wgpu/wgpu_ops.c
    )
    
    target_sources(basednn PRIVATE ${WGPU_SOURCES})
    target_include_directories(basednn PRIVATE backend/wgpu)
    
    # Find WGPU library
    if(DEFINED ENV{WGPU_DIR})
        set(WGPU_DIR $ENV{WGPU_DIR})
    else()
        set(WGPU_DIR "$ENV{HOME}/.wgpu-native")
    endif()
    
    if(EXISTS "${WGPU_DIR}/include/webgpu.h" OR EXISTS "${WGPU_DIR}/include/webgpu/webgpu.h")
        target_include_directories(basednn PRIVATE 
            ${WGPU_DIR}/include
        )
        
        # Link WGPU library
        find_library(WGPU_LIB wgpu_native PATHS ${WGPU_DIR}/lib REQUIRED)
        target_link_libraries(basednn ${WGPU_LIB})
        
        # macOS Metal framework for wgpu-native Metal backend
        if(APPLE)
            target_link_libraries(basednn 
                "-framework Metal" 
                "-framework CoreGraphics" 
                "-framework Foundation" 
                "-framework QuartzCore" 
                "-framework IOKit" 
                "-framework IOSurface"
            )
        endif()
        
        target_compile_definitions(basednn PRIVATE HAS_WGPU)
        message(STATUS "WGPU GPU acceleration enabled (${WGPU_DIR})")
    else()
        message(FATAL_ERROR "WGPU enabled but library not found at ${WGPU_DIR}")
    endif()
endif()

# Enable testing
enable_testing()

# Unit tests
set(TEST_SOURCES
    core/tests/unit/test_tensor.c
    core/tests/unit/test_ops.c
    core/tests/unit/test_registry.c
    core/tests/unit/test_layer.c
    core/tests/unit/test_network.c
    core/tests/unit/test_optimizer.c
)

# Create individual test executables
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} basednn m)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Full tests
add_executable(xor core/tests/full/xor.c)
target_link_libraries(xor basednn m)
add_executable(mnist core/tests/full/mnist.c)
target_link_libraries(mnist basednn m)

# Enable GPU support for mnist if WGPU is available
if(ENABLE_WGPU)
    target_include_directories(mnist PRIVATE ${WGPU_DIR}/include backend/wgpu)
    target_compile_definitions(mnist PRIVATE HAS_WGPU)
endif()

# GPU unit tests
if(ENABLE_WGPU)
    set(WGPU_TEST_SOURCES
        backend/wgpu/tests/test_wgpu_add.c
        backend/wgpu/tests/test_wgpu_matmul.c
        backend/wgpu/tests/test_wgpu_performance.c
    )
    
    foreach(test_src ${WGPU_TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_include_directories(${test_name} PRIVATE ${WGPU_DIR}/include backend/wgpu)
        target_link_libraries(${test_name} basednn m)
        target_compile_definitions(${test_name} PRIVATE HAS_WGPU)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Examples (if they exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/custom_tensor_op.c")
    add_executable(custom_tensor_op examples/custom_tensor_op.c)
    target_link_libraries(custom_tensor_op basednn m)
endif()

# Custom compiler rules to allow relaxed static inline in C99
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")