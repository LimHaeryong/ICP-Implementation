#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
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

    auto visualizer = std::make_unique<open3d::visualization::Visualizer>();
    visualizer->CreateVisualizerWindow("ICP result", 1024, 768);
    visualizer->GetRenderOption().SetPointSize(1.0);
    visualizer->GetRenderOption().background_color_ = {0, 0, 0};
    visualizer->AddGeometry(source_transformed);
    visualizer->AddGeometry(target_copy);

    visualizer->Run();
}

std::shared_ptr<open3d::geometry::PointCloud> readBinFile(const std::string &file_path)
{
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
        return cloud;

    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::size_t point_size = sizeof(float) * 4;
    std::size_t num_points = file_size / point_size;

    cloud->points_.resize(num_points);
    Eigen::Vector4f kitti_point;

    for (std::size_t i = 0; i < num_points; ++i)
    {
        if (!file.read(reinterpret_cast<char *>(kitti_point.data()), point_size))
            break;
        cloud->points_[i] = kitti_point.head<3>().cast<double>();
    }

    return cloud;
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

    std::shared_ptr<open3d::geometry::PointCloud> source, target;
    if (source_path.substr(source_path.size() - 4) == ".bin")
        source = readBinFile(source_path);
    else
        source = open3d::io::CreatePointCloudFromFile(source_path);

    if (target_path.substr(target_path.size() - 4) == ".bin")
        target = readBinFile(target_path);
    else
        target = open3d::io::CreatePointCloudFromFile(target_path);

    if (source->IsEmpty() || target->IsEmpty())
    {
        spdlog::warn("unable to load source or target files.");
        return 1;
    }

    double voxel_size = 0.2;
    int iteration = 30;
    Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
    double max_correspondence_dist = 10.0;

    auto source_down = source->VoxelDownSample(voxel_size);
    auto target_down = target->VoxelDownSample(voxel_size);

    auto t_start = std::chrono::high_resolution_clock::now();
    auto reg_result = open3d::pipelines::registration::RegistrationICP(
        *source_down, *target_down, max_correspondence_dist, Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, iteration));
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    trans = reg_result.transformation_;
    spdlog::info("Open3D ICP elapsed time : {}ms", duration);
    spdlog::info("trans = \n{}", trans);
    visualizeRegistration(*source, *target, trans);

    target_down->EstimateNormals();

    ICP icp;
    icp.setIteration(iteration);
    icp.setMaxCorrespondenceDist(max_correspondence_dist);
    t_start = std::chrono::high_resolution_clock::now();
    icp.align(*source_down, *target_down);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    auto trans2 = icp.getResultTransform();
    spdlog::info("My ICP elapsed time : {}ms", duration);
    spdlog::info("trans = \n{}", trans2);
    visualizeRegistration(*source, *target, trans2);

    return 0;
}