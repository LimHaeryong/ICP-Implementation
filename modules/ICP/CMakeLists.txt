cmake_minimum_required(VERSION 3.26)

set(LIBRARY_NAME ICP)
project(${LIBRARY_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUITED ON)

add_library(${LIBRARY_NAME} 
    src/icp_base.cpp    
    src/icp.cpp
    src/icp_plane.cpp
    src/gicp.cpp
    src/ceres_optimizer.cpp
)
if(NOT ${DEP_LIST} STREQUAL "")
    add_dependencies(${LIBRARY_NAME}  
        ${DEP_LIST}
    )
endif()
target_include_directories(${LIBRARY_NAME} PUBLIC
    ${DEP_INCLUDE_DIR}
    include
)
target_link_directories(${LIBRARY_NAME} PRIVATE
    ${DEP_LIBS_DIR}
)
target_link_libraries(${LIBRARY_NAME} PUBLIC
    ${DEP_LIBS}
)
target_compile_options(${LIBRARY_NAME} PRIVATE
    ${OpenMP_CXX_FLAGS}
)
