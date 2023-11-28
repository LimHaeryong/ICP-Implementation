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
#add_subdirectory(modules)

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
)

set(SOURCE_CLOUD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/room_scan1.pcd)
set(TARGET_CLOUD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/room_scan2.pcd)
target_compile_definitions(icp_example PRIVATE
    SOURCE_CLOUD_PATH="${SOURCE_CLOUD_PATH}"
    TARGET_CLOUD_PATH="${TARGET_CLOUD_PATH}"
)