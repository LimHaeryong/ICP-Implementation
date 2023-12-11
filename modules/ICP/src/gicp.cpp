#include <algorithm>

#include <Eigen/Dense>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <omp.h>

#include "ICP/gicp.hpp"
#include "ICP/utils.hpp"

bool GICP::checkValidity(PointCloud &source_cloud, PointCloud &target_cloud)
{
    if (source_cloud.IsEmpty() || target_cloud.IsEmpty())
    {
        spdlog::warn("source cloud or target cloud are empty!");
        return false;
    }

    if (!source_cloud.HasCovariances())
    {
        if (source_cloud.HasNormals())
        {
            spdlog::info("compute source cloud covariances from normals");
            computeCovariancesFromNormals(source_cloud);
        }
        else
        {
            spdlog::warn("source cloud needs normals or covariances");
            return false;
        }
    }

    if (!target_cloud.HasCovariances())
    {
        if (target_cloud.HasNormals())
        {
            spdlog::info("compute target cloud covariances from normals");
            computeCovariancesFromNormals(target_cloud);
        }
        else
        {
            spdlog::warn("target cloud needs normals or covariances");
            return false;
        }
    }

    return true;
}

Eigen::Matrix4d GICP::computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud)
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

Eigen::Matrix4d GICP::computeTransformLeastSquaresUsingCeres(const PointCloud &source_cloud, const PointCloud &target_cloud)
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

Eigen::Matrix4d GICP::computeTransformLeastSquares(const PointCloud &source_cloud, const PointCloud &target_cloud)
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
            const auto &p_cov = source_cloud.covariances_[correspondence_set_[i].first];
            const auto &q = target_cloud.points_[correspondence_set_[i].second];
            const auto &q_cov = target_cloud.covariances_[correspondence_set_[i].second];
            auto [JTJi, JTri] = compute_JTJ_and_JTr(p, p_cov, q, q_cov);
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

std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Vector<double, 6>> GICP::compute_JTJ_and_JTr(const Eigen::Vector3d &p, const Eigen::Matrix3d &p_cov,
                                                                                           const Eigen::Vector3d &q, const Eigen::Matrix3d &q_cov)
{
    Eigen::Matrix<double, 6, 6> JTJ;
    Eigen::Vector<double, 6> JTr;

    Eigen::Matrix<double, 3, 6> J;
    J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    J.block<3, 3>(0, 3) = -skewSymmetric(p);

    Eigen::Matrix3d C_inv = (p_cov + q_cov).inverse();
    double error = std::sqrt(((p - q).transpose() * C_inv * (p - q)).coeff(0));

    JTJ = 1.0 / error * J.transpose() * C_inv * J;
    JTr = J.transpose() * C_inv * (p - q);
    return std::make_pair(JTJ, JTr);
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