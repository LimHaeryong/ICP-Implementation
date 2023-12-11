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
    switch (solver_type_)
    {
    case SolverType::LeastSquares:
        return computeTransformLeastSquares(source_cloud, target_cloud);
    case SolverType::LeastSquaresUsingCeres:
        return computeTransformLeastSquaresUsingCeres(source_cloud, target_cloud);
    default:
        return computeTransformLeastSquares(source_cloud, target_cloud);
    }
}

Eigen::Matrix4d ICP_PLANE::computeTransformLeastSquares(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    Eigen::Matrix<double, 6, 6> JTJ;
    Eigen::Vector<double, 6> JTr;
    JTJ.setZero();
    JTr.setZero();

    int num_corr = correspondence_set_.size();

#pragma omp parallel
    {
        Eigen::Matrix<double, 6, 6> JTJ_private;
        Eigen::Vector<double, 6> JTr_private;
        JTJ_private.setZero();
        JTr_private.setZero();
#pragma omp for nowait
        for (int i = 0; i < num_corr; ++i)
        {
            const auto &p = source_cloud.points_[correspondence_set_[i].first];
            const auto &q = target_cloud.points_[correspondence_set_[i].second];
            const auto &q_norm = target_cloud.normals_[correspondence_set_[i].second];
            auto [JTJi, JTri] = compute_JTJ_and_JTr(p, q, q_norm);
            double wi = 1.0;
            JTJ_private += wi * JTJi;
            JTr_private += wi * JTri;
        }
#pragma omp critical
        {
            JTJ += JTJ_private;
            JTr += JTr_private;
        }
    }

    Eigen::Vector<double, 6> x_opt = JTJ.ldlt().solve(-JTr);
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = createRotationMatrix(x_opt.tail(3));
    transform.block<3, 1>(0, 3) = x_opt.head(3);
    return transform;
}

std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Vector<double, 6>> ICP_PLANE::compute_JTJ_and_JTr(const Eigen::Vector3d &p, const Eigen::Vector3d &q, const Eigen::Vector3d &q_norm)
{
    Eigen::Matrix<double, 6, 1> JT;
    Eigen::Vector<double, 1> r;
    JT.block<3, 1>(0, 0) = q_norm;
    JT.block<3, 1>(3, 0) = p.cross(q_norm);
    r = (p - q).transpose() * q_norm;
    return std::make_pair(JT * JT.transpose(), JT * r);
}

Eigen::Matrix4d ICP_PLANE::computeTransformLeastSquaresUsingCeres(const PointCloud &source_cloud, const PointCloud &target_cloud)
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