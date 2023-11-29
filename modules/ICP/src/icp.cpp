#include <algorithm>

#include <spdlog/spdlog.h>

#include "ICP/icp.hpp"

void ICP::align(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
    total_transform_ = Eigen::Matrix4d::Identity();
    PointCloud tmp_cloud = source_cloud;
    tree_ = std::make_shared<KDTree>(target_cloud);
    correspondence_set_.reserve(source_cloud.points_.size());

    for (int i = 0; i < max_iteration_; ++i)
    {
        correspondenceMatching(tmp_cloud);
        Eigen::Matrix4d transform = computeTransform(tmp_cloud, target_cloud);
        this->total_transform_ *= transform;
        if (euclidean_error_ < euclidean_error_ ||
            transform.isApprox(Eigen::Matrix4d::Identity(), transformation_epsilon_))
        {
            converged_ = true;
            break;
        }
        tmp_cloud.Transform(transform);
    }
}

void ICP::correspondenceMatching(const PointCloud &tmp_cloud)
{
    correspondence_set_.clear();
    euclidean_error_ = 0.0;
    auto &points = tmp_cloud.points_;
    std::vector<int> indices(1);
    std::vector<double> distances2(1);
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
    {
        if (tree_->SearchHybrid(points[i], max_correspondence_distance_, 1, indices, distances2) > 0)
        {
            correspondence_set_.emplace_back(i, indices[0]);
            euclidean_error_ += distances2[0];
        }
    }
}

Eigen::Matrix4d ICP::computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud)
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
    Eigen::Vector3d t = Q - R * t;
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = R;
    transform.block<3, 1>(0, 3) = t;

    return transform;
}