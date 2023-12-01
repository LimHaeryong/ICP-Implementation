#include <algorithm>

#include <Eigen/Dense>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
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
    if(solverType_ == Linear)
        return computeTransformLinearSolver(source_cloud, target_cloud);
    else
        return computeTransformNonlinearSolver(source_cloud, target_cloud);
}

Eigen::Matrix4d ICP_PLANE::computeTransformNonlinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud)
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

Eigen::Matrix4d ICP_PLANE::computeTransformLinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    Eigen::Matrix6d AtA = Eigen::Matrix6d::Zero();
    Eigen::Vector6d Atb = Eigen::Vector6d::Zero();

    int num_corr = correspondence_set_.size();
    Eigen::Matrix<double, Eigen::Dynamic, 6> A(num_corr, 6);
    Eigen::VectorXd b(num_corr);

    #pragma omp parallel for
    for(int i = 0; i < num_corr; ++i)
    {
        const auto& s_point = source_cloud.points_[correspondence_set_[i].first];
        const auto& t_point = target_cloud.points_[correspondence_set_[i].second];
        const auto& t_norm = target_cloud.normals_[correspondence_set_[i].second];
        A.block<1, 3>(i, 0) = s_point.cross(t_norm);
        A.block<1, 3>(i, 3) = t_norm;

        b(i) = t_norm.dot((t_point - s_point));
    }

    #pragma omp parallel for
    for(int i = 0; i < 6; ++i)
    {
        Atb(i) = A.col(i).dot(b);
        for(int j = i; j < 6; ++j)
        {
            AtA(i, j) = A.col(i).dot(A.col(j));
            AtA(j, i) = AtA(i, j);
        }
    }

    Eigen::Vector6d x_opt = AtA.inverse() * Atb;

    Eigen::AngleAxisd rot_x(x_opt(0), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rot_y(x_opt(1), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rot_z(x_opt(2), Eigen::Vector3d::UnitZ());

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = (rot_z * rot_y * rot_x).toRotationMatrix();
    transform.block<3, 1>(0, 3) = x_opt.tail(3);

    return transform;
}