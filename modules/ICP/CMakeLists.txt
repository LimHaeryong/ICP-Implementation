cmake_minimum_required(VERSION 3.26)

set(LIBRARY_NAME ICP)
project(${LIBRARY_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUITED ON)

add_library(${LIBRARY_NAME}
    src/icp.cpp
)
target_include_directories(${LIBRARY_NAME} PUBLIC
    include
)
# target_link_libraries(${LIBRARY_NAME} PRIVATE

# )
