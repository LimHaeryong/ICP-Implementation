#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>
#include <open3d/Open3D.h>

#include "ICP/icp.hpp"

void visualizeRegistration(const open3d::geometry::PointCloud &source,
                           const open3d::geometry::PointCloud &target,
                           const Eigen::Matrix4d &trans)
{
    auto source_transformed = std::make_shared<open3d::geometry::PointCloud>();
    auto source_copy = std::make_shared<open3d::geometry::PointCloud>();
    auto target_copy = std::make_shared<open3d::geometry::PointCloud>();
    *source_transformed = source;
    *source_copy = source;
    *target_copy = target;
    source_transformed->PaintUniformColor({1, 0, 0});
    source_copy->PaintUniformColor({0, 0, 1});
    target_copy->PaintUniformColor({0, 1, 0});
    source_transformed->Transform(trans);
    // open3d::visualization::DrawGeometries({source_transformed, target_copy},
    //                                       "ICP result", 1024, 768);

    auto visualizer = std::make_unique<open3d::visualization::Visualizer>();
    visualizer->CreateVisualizerWindow("ICP result", 1024, 768);
    visualizer->GetRenderOption().SetPointSize(1.0);
    visualizer->GetRenderOption().background_color_ = {0, 0, 0};
    visualizer->AddGeometry(source_transformed);
    visualizer->AddGeometry(target_copy);
    
    visualizer->Run();
}

int main(int argc, char *argv[])
{
    spdlog::info("icp example starts");

    std::string source_path, target_path;
    if (argc == 3)
    {
        source_path = argv[1];
        target_path = argv[2];
    }
    else
    {
        source_path = SOURCE_CLOUD_PATH;
        target_path = TARGET_CLOUD_PATH;
    }

    auto source = open3d::io::CreatePointCloudFromFile(source_path);
    auto target = open3d::io::CreatePointCloudFromFile(target_path);
    if (source->IsEmpty() || target->IsEmpty())
    {
        spdlog::warn("unable to load source or target files.");
        return 1;
    }

    double voxel_size = 0.2;
    int iteration = 200;
    Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();

    auto source_down = source->VoxelDownSample(voxel_size);
    auto target_down = target->VoxelDownSample(voxel_size);

    auto reg_result = open3d::pipelines::registration::RegistrationICP(
        *source_down, *target_down, 100.0, Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, iteration));

    trans = reg_result.transformation_;

    visualizeRegistration(*source, *target, trans);

    ICP icp;
    icp.align(*source_down, *target_down);
    auto trans2 = icp.getResultTransform();

    visualizeRegistration(*source, *target, trans2);

    return 0;
}