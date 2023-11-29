#include <algorithm>

#include <spdlog/spdlog.h>
#include <omp.h>

#include "ICP/icp.hpp"

#pragma omp declare reduction(matrix_reduction : Eigen::MatrixXd : omp_out += omp_in) initializer(omp_priv = Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))
#pragma omp declare reduction(vec3d_reduction : Eigen::Vector3d : omp_out += omp_in) initializer(omp_priv = Eigen::Vector3d::Zero())

void ICP::align(const PointCloud &source_cloud, const PointCloud &target_cloud)
{
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

void ICP::correspondenceMatching(const PointCloud &tmp_cloud)
{
    correspondence_set_.clear();
    euclidean_error_ = 0.0;
    auto &points = tmp_cloud.points_;
    
    #pragma omp parallel
    {
        double euclidean_error_private = 0.0;
        std::vector<std::pair<int, int>> correspondence_set_private;
        std::vector<int> indices(1);
        std::vector<double> distances2(1);
        #pragma omp for nowait
        for(int i = 0; i < static_cast<int>(points.size()); ++i)
        {
            if (tree_->SearchHybrid(points[i], max_corres_dist_, 1, indices, distances2) > 0)
            {
                correspondence_set_private.emplace_back(i, indices[0]);
                euclidean_error_private += distances2[0];
            }
        }

        #pragma omp critical
        {
            euclidean_error_ += euclidean_error_private;
            for(std::size_t i = 0; i < correspondence_set_private.size(); ++i)
                correspondence_set_.push_back(correspondence_set_private[i]);
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