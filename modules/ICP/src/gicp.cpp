#include <algorithm>

#include <Eigen/Dense>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <omp.h>

#include "ICP/gicp.hpp"

void GICP::align(PointCloud &source_cloud, PointCloud &target_cloud)
{
    if ((!source_cloud.HasNormals() && !source_cloud.HasCovariances()) || (!target_cloud.HasNormals() && !target_cloud.HasCovariances()))
    {
        spdlog::warn("GICP needs normals and covariances in both pointclouds.");
        return;
    }

    if (!source_cloud.HasCovariances())
        computeCovariancesFromNormals(source_cloud);

    if (!target_cloud.HasCovariances())
        computeCovariancesFromNormals(target_cloud);

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
        if (euclidean_error_ < euclidean_fitness_epsilon_ ||
            transform.block<3, 1>(0, 3).norm() < transformation_epsilon_)
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

Eigen::Matrix4d GICP::computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    if (solverType_ == Linear)
        return computeTransformLinearSolver(source_cloud, target_cloud);
    else
        return computeTransformNonlinearSolver(source_cloud, target_cloud);
}

Eigen::Matrix4d GICP::computeTransformNonlinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    optimizer_->clear();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    const auto &source_points = source_cloud.points_;
    const auto &source_covs = source_cloud.covariances_;
    const auto &target_points = target_cloud.points_;
    const auto &target_covs = target_cloud.covariances_;
    for (std::size_t i = 0; i < correspondence_set_.size(); ++i)
    {
        auto [source_idx, target_idx] = correspondence_set_[i];
        optimizer_->addGICPResidual(source_points[source_idx], source_covs[source_idx], target_points[target_idx], target_covs[target_idx], rotation, translation);
    }
    optimizer_->solve();

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = rotation.toRotationMatrix();
    transform.block<3, 1>(0, 3) = translation;

    return transform;
}

Eigen::Matrix4d GICP::computeTransformLinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    Eigen::Matrix6d AtA = Eigen::Matrix6d::Zero();
    Eigen::Vector6d Atb = Eigen::Vector6d::Zero();

    int num_corr = correspondence_set_.size();
    Eigen::Matrix<double, Eigen::Dynamic, 6> A(num_corr, 6);
    Eigen::VectorXd b(num_corr);

#pragma omp parallel for
    for (int i = 0; i < num_corr; ++i)
    {
        const auto &s_point = source_cloud.points_[correspondence_set_[i].first];
        const auto &t_point = target_cloud.points_[correspondence_set_[i].second];
        const auto &t_norm = target_cloud.normals_[correspondence_set_[i].second];
        A.block<1, 3>(i, 0) = s_point.cross(t_norm);
        A.block<1, 3>(i, 3) = t_norm;

        b(i) = t_norm.dot((t_point - s_point));
    }

#pragma omp parallel for
    for (int i = 0; i < 6; ++i)
    {
        Atb(i) = A.col(i).dot(b);
        for (int j = i; j < 6; ++j)
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

void GICP::computeCovariancesFromNormals(PointCloud &cloud)
{
    int num_points = cloud.normals_.size();
    const Eigen::Matrix3d Cov = Eigen::Vector3d(cov_epsilon_, 1.0, 1.0).asDiagonal();
    cloud.covariances_.resize(num_points);
#pragma omp parallel for
    for (int i = 0; i < num_points; ++i)
    {
        Eigen::Matrix3d R = getRotationFromNormal(cloud.normals_[i]);
        cloud.covariances_[i] = R * Cov * R.transpose();
    }
}

Eigen::Matrix3d GICP::getRotationFromNormal(const Eigen::Vector3d &normal)
{
    Eigen::Vector3d e1{1.0, 0.0, 0.0};

    Eigen::Vector3d v = e1.cross(normal);
    double cos = e1.dot(normal);
    if (1.0 + cos < 1e-3)
        return Eigen::Matrix3d::Identity();

    Eigen::Matrix3d v_skew;
    v_skew << 0.0, -v(2), v(1),
        v(2), 0.0, -v(0),
        -v(1), v(0), 0.0;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + v_skew + (1.0 / (1.0 + cos)) * (v_skew * v_skew);

    return R;
}