cmake_minimum_required(VERSION 3.26)

set(PROJECT_NAME ICP_IMPLEMENTATION)
project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUITED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")

include(Dependency.cmake)
add_subdirectory(modules)

set(CLOUD_PATH_1 ${CMAKE_CURRENT_SOURCE_DIR}/resources/0000000240.bin)
set(CLOUD_PATH_2 ${CMAKE_CURRENT_SOURCE_DIR}/resources/0000000250.bin)
set(CLOUD_PATH_3 ${CMAKE_CURRENT_SOURCE_DIR}/resources/0000000260.bin)

# ICP example
add_executable(icp_example examples/icp_example.cpp)
if(NOT ${DEP_LIST} STREQUAL "")
    add_dependencies(icp_example 
        ${DEP_LIST}
    )
endif()
target_include_directories(icp_example PUBLIC
    ${DEP_INCLUDE_DIR}
)
target_link_directories(icp_example PRIVATE
    ${DEP_LIBS_DIR}
)
target_link_libraries(icp_example PUBLIC
    ${DEP_LIBS}
    ICP
)
target_compile_definitions(icp_example PRIVATE
    SOURCE_CLOUD_PATH="${CLOUD_PATH_1}"
    TARGET_CLOUD_PATH="${CLOUD_PATH_2}"
)

# point to plane ICP example
add_executable(icp_plane_example examples/icp_plane_example.cpp)
if(NOT ${DEP_LIST} STREQUAL "")
    add_dependencies(icp_plane_example 
        ${DEP_LIST}
    )
endif()
target_include_directories(icp_plane_example PUBLIC
    ${DEP_INCLUDE_DIR}
)
target_link_directories(icp_plane_example PRIVATE
    ${DEP_LIBS_DIR}
)
target_link_libraries(icp_plane_example PUBLIC
    ${DEP_LIBS}
    ICP
)
target_compile_definitions(icp_plane_example PRIVATE
    SOURCE_CLOUD_PATH="${CLOUD_PATH_1}"
    TARGET_CLOUD_PATH="${CLOUD_PATH_2}"
)

# GICP example
add_executable(gicp_example examples/gicp_example.cpp)
if(NOT ${DEP_LIST} STREQUAL "")
    add_dependencies(gicp_example 
        ${DEP_LIST}
    )
endif()
target_include_directories(gicp_example PUBLIC
    ${DEP_INCLUDE_DIR}
)
target_link_directories(gicp_example PRIVATE
    ${DEP_LIBS_DIR}
)
target_link_libraries(gicp_example PUBLIC
    ${DEP_LIBS}
    ICP
)
target_compile_definitions(gicp_example PRIVATE
        SOURCE_CLOUD_PATH="${CLOUD_PATH_1}"
    TARGET_CLOUD_PATH="${CLOUD_PATH_3}"
)