#include <algorithm>

#include <Eigen/Dense>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <omp.h>

#include "ICP/icp_plane.hpp"
#include "ICP/utils.hpp"

bool ICP_PLANE::checkValidity(PointCloud &source_cloud, PointCloud &target_cloud)
{
    if (source_cloud.IsEmpty() || target_cloud.IsEmpty())
    {
        spdlog::warn("source cloud or target cloud are empty!");
        return false;
    }

    if (!target_cloud.HasNormals())
    {
        spdlog::warn("point to plane ICP needs normal points in the target pointcloud.");
        return false;
    }

    return true;
}

Eigen::Matrix4d ICP_PLANE::computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    if (solverType_ == Linear)
        return computeTransformLinearSolver(source_cloud, target_cloud);
    else
        return computeTransformNonlinearSolver(source_cloud, target_cloud);
}

Eigen::Matrix4d ICP_PLANE::computeTransformNonlinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    optimizer_->clear();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    const auto &source_points = source_cloud.points_;
    const auto &target_points = target_cloud.points_;
    const auto &target_norms = target_cloud.normals_;
    for (std::size_t i = 0; i < correspondence_set_.size(); ++i)
    {
        auto [source_idx, target_idx] = correspondence_set_[i];
        optimizer_->addPointToPlaneResidual(source_points[source_idx], target_points[target_idx], target_norms[target_idx], rotation, translation);
    }
    optimizer_->solve();

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = rotation.toRotationMatrix();
    transform.block<3, 1>(0, 3) = translation;

    return transform;
}

std::pair<Eigen::Matrix<double, 1, 6>, Eigen::Vector<double, 1>> ICP_PLANE::compute_Ai_and_bi(const Eigen::Vector3d &pi, const Eigen::Vector3d &qi, const Eigen::Vector3d& q_norm_i)
{
    Eigen::Matrix<double, 1, 6> Ai;
    Eigen::Vector<double, 1> bi;

    Ai.block<1, 3>(0, 0) = q_norm_i.transpose();
    Ai.block<1, 3>(0, 3) = pi.cross(q_norm_i).transpose();

    bi = (pi - qi).transpose() * q_norm_i;
    return std::make_pair(Ai, bi);
}

Eigen::Matrix4d ICP_PLANE::computeTransformLinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    Eigen::Matrix<double, 6, 6> C;
    Eigen::Vector<double, 6> d;
    C.setZero();
    d.setZero();

    int num_corr = correspondence_set_.size();

#pragma omp parallel
    {
        Eigen::Matrix<double, 6, 6> C_private;
        Eigen::Vector<double, 6> d_private;
        C_private.setZero();
        d_private.setZero();
#pragma omp for nowait
        for (int i = 0; i < num_corr; ++i)
        {
            const auto& p = source_cloud.points_[correspondence_set_[i].first];
            const auto& q = target_cloud.points_[correspondence_set_[i].second];
            const auto& q_norm = target_cloud.normals_[correspondence_set_[i].second];
            auto [Ai, bi] = compute_Ai_and_bi(p, q, q_norm);
            //double wi = HuberLoss(20.0, bi.squaredNorm());
            double wi = 1.0;
            C_private += wi * Ai.transpose() * Ai;
            d_private += wi * Ai.transpose() * bi;
        }
#pragma omp critical
        {
            C += C_private;
            d += d_private;
        }
    }

    Eigen::Vector<double, 6> x_opt = C.ldlt().solve(-d);
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = createRotationMatrix(x_opt.tail(3));
    transform.block<3, 1>(0, 3) = x_opt.head(3);
    return transform;
}