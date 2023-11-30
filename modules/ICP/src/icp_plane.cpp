#include <algorithm>

#include <spdlog/spdlog.h>
#include <omp.h>

#include "ICP/icp_plane.hpp"

void ICP_PLANE::align(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    if(!target_cloud.HasNormals())
    {
        spdlog::warn("point to plane ICP needs normal points in the target pointcloud.");
        return;
    }

    total_transform_ = Eigen::Matrix4d::Identity();
    PointCloud tmp_cloud = source_cloud;
    tree_ = std::make_shared<KDTree>(target_cloud);
    correspondence_set_.reserve(source_cloud.points_.size());

    int64_t t_corr = 0, t_comp = 0, t_trans = 0;

    for (int i = 0; i < max_iteration_; ++i)
    {
        spdlog::info("iter : {}", i);

        auto t_0 = std::chrono::high_resolution_clock::now();
        correspondenceMatching(tmp_cloud);
        auto t_1 = std::chrono::high_resolution_clock::now();
        Eigen::Matrix4d transform = computeTransform(tmp_cloud, target_cloud);
        auto t_2 = std::chrono::high_resolution_clock::now();
        this->total_transform_ *= transform;
        if (euclidean_error_ < euclidean_error_ ||
            transform.isApprox(Eigen::Matrix4d::Identity(), transformation_epsilon_))
        {
            converged_ = true;
            break;
        }
        tmp_cloud.Transform(transform);
        auto t_3 = std::chrono::high_resolution_clock::now();

        t_corr += std::chrono::duration_cast<std::chrono::microseconds>(t_1 - t_0).count();
        t_comp += std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count();
        t_trans += std::chrono::duration_cast<std::chrono::microseconds>(t_3 - t_2).count();
    }

    spdlog::info("correspondence elapsed time : {} micro seconds", t_corr);
    spdlog::info("compute transform elapsed time : {} micro seconds", t_comp);
    spdlog::info("transform/check elapsed time : {} micro seconds", t_trans);
}

Eigen::Matrix4d ICP_PLANE::computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    optimizer_->clear();
    Eigen::Vector3d rotation = Eigen::Vector3d::Zero();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    const auto& source_points = source_cloud.points_;
    const auto& target_points = target_cloud.points_;
    const auto& target_norms = target_cloud.normals_;
    for(std::size_t i = 0; i < correspondence_set_.size(); ++i)
    {
        auto [source_idx, target_idx] = correspondence_set_[i];
        optimizer_->addPointToPlaneResidual(source_points[source_idx], target_points[target_idx], target_norms[target_idx], rotation, translation);
    }
    optimizer_->solve();
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd angle_axis(rotation.norm(), rotation.normalized());
    transform.block<3, 3>(0, 0) = angle_axis.toRotationMatrix();
    transform.block<3, 1>(0, 3) = translation;

    return transform;
}