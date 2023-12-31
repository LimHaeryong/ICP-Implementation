include(ExternalProject)

set(DEP_INSTALL_DIR ${PROJECT_BINARY_DIR}/install)
set(DEP_INCLUDE_DIR ${DEP_INSTALL_DIR}/include)
set(DEP_LIBS_DIR ${DEP_INSTALL_DIR}/lib)

find_package(OpenMP REQUIRED)
set(DEP_LIBS ${DEP_LIBS} OpenMP::OpenMP_CXX)

find_package(Ceres REQUIRED HINTS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver/lib/cmake/Ceres)
set(DEP_LIBS ${DEP_LIBS} Ceres::ceres)

find_package(Open3D REQUIRED HINTS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Open3d/lib/cmake/Open3D)
set(DEP_LIBS ${DEP_LIBS} Open3D::Open3D)

# spdlog
ExternalProject_Add(
    dep-spdlog
    GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
    GIT_TAG "v1.x"
    GIT_SHALLOW 1
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${DEP_INSTALL_DIR}
    TEST_COMMAND ""
)
set(DEP_LIBS ${DEP_LIBS} spdlog)
set(DEP_LIST ${DEP_LIST} dep-spdlog)


