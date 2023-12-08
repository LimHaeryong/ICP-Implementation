#include <algorithm>

#include <spdlog/spdlog.h>
#include <omp.h>

#include "ICP/icp.hpp"
#include "ICP/utils.hpp"

bool ICP::checkValidity(PointCloud &source_cloud, PointCloud &target_cloud)
{
    if (source_cloud.IsEmpty() || target_cloud.IsEmpty())
    {
        spdlog::warn("source cloud or target cloud are empty!");
        return false;
    }

    return true;
}

std::pair<Eigen::Matrix<double, 3, 6>, Eigen::Vector3d> compute_Ai_and_bi(const Eigen::Vector3d &pi, const Eigen::Vector3d &qi)
{
    Eigen::Matrix<double, 3, 6> Ai;
    Eigen::Vector3d bi;

    Ai.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Ai.block<3, 3>(0, 3) = -1.0 * skewSymmetric(pi);

    bi = pi - qi;

    return std::make_pair(Ai, bi);
}

Eigen::Matrix4d ICP::computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud)
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
            auto [Ai, bi] = compute_Ai_and_bi(p, q);
            //double wi = HuberLoss(2.0, bi.squaredNorm());
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