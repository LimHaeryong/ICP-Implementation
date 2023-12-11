#include <algorithm>

#include <spdlog/spdlog.h>
#include <omp.h>

#include "ICP/icp.hpp"
#include "ICP/utils.hpp"

#pragma omp declare reduction(matrix_reduction : Eigen::MatrixXd : omp_out += omp_in) initializer(omp_priv = Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))
#pragma omp declare reduction(vec3d_reduction : Eigen::Vector3d : omp_out += omp_in) initializer(omp_priv = Eigen::Vector3d::Zero())

bool ICP::checkValidity(PointCloud &source_cloud, PointCloud &target_cloud)
{
    if (source_cloud.IsEmpty() || target_cloud.IsEmpty())
    {
        spdlog::warn("source cloud or target cloud are empty!");
        return false;
    }

    return true;
}

Eigen::Matrix4d ICP::computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    switch (solver_type_)
    {
    case SolverType::SVD:
        return computeTransformSVD(source_cloud, target_cloud);
    case SolverType::LeastSquares:
        return computeTransformLeastSquares(source_cloud, target_cloud);
    default:
        return computeTransformLeastSquares(source_cloud, target_cloud);
    }
}

std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Vector<double, 6>> ICP::compute_JTJ_and_JTr(const Eigen::Vector3d &p, const Eigen::Vector3d &q)
{
    Eigen::Matrix<double, 3, 6> J;
    Eigen::Vector3d r;

    J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    J.block<3, 3>(0, 3) = -1.0 * skewSymmetric(p);

    r = p - q;

    return std::make_pair(J.transpose() * J, J.transpose() * r);
}

Eigen::Matrix4d ICP::computeTransformLeastSquares(const PointCloud &source_cloud, const PointCloud &target_cloud)
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
            auto [JTJi, JTri] = compute_JTJ_and_JTr(p, q);
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

Eigen::Matrix4d ICP::computeTransformSVD(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    int num_corr = correspondence_set_.size();

    Eigen::MatrixXd X(3, num_corr);
    Eigen::MatrixXd Y(3, num_corr);
    Eigen::Vector3d P = Eigen::Vector3d::Zero();
    Eigen::Vector3d Q = Eigen::Vector3d::Zero();

    for (int i = 0; i < num_corr; ++i)
    {
        P += source_cloud.points_[correspondence_set_[i].first];
        Q += target_cloud.points_[correspondence_set_[i].second];
    }

    P /= static_cast<double>(num_corr);
    Q /= static_cast<double>(num_corr);

#pragma omp parallel for
    for (int i = 0; i < num_corr; ++i)
    {
        X.col(i) = source_cloud.points_[correspondence_set_[i].first];
        Y.col(i) = target_cloud.points_[correspondence_set_[i].second];
    }

    Eigen::Matrix3d S = X * Y.transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    D(2, 2) = (svd.matrixV() * svd.matrixU().transpose()).determinant();

    Eigen::Matrix3d R = svd.matrixV() * D * svd.matrixU().transpose();
    Eigen::Vector3d t = Q - R * P;
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = R;
    transform.block<3, 1>(0, 3) = t;

    return transform;
}