#include <algorithm>

#include <spdlog/spdlog.h>
#include <omp.h>

#include "ICP/icp_base.hpp"

void ICP_BASE::align(PointCloud &source_cloud, PointCloud &target_cloud)
{
    if (!checkValidity(source_cloud, target_cloud))
    {
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
        total_transform_ = transform * total_transform_;
        if (convergenceCheck(transform))
        {
            t_corr += std::chrono::duration_cast<std::chrono::microseconds>(t_1 - t_0).count();
            t_comp += std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count();
            spdlog::info("ICP converged! iter = {}", i + 1);
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

void ICP_BASE::correspondenceMatching(const PointCloud &tmp_cloud)
{
    correspondence_set_.clear();
    matching_rmse_prev_ = matching_rmse_;
    matching_rmse_ = 0.0;
    auto &points = tmp_cloud.points_;

#pragma omp parallel
    {
        double euclidean_error_private = 0.0;
        std::vector<std::pair<int, int>> correspondence_set_private;
        std::vector<int> indices(1);
        std::vector<double> distances2(1);
#pragma omp for nowait
        for (int i = 0; i < static_cast<int>(points.size()); ++i)
        {
            if (tree_->SearchHybrid(points[i], max_corres_dist_, 1, indices, distances2) > 0)
            {
                correspondence_set_private.emplace_back(i, indices[0]);
                euclidean_error_private += distances2[0];
            }
        }

#pragma omp critical
        {
            matching_rmse_ += euclidean_error_private;
            for (std::size_t i = 0; i < correspondence_set_private.size(); ++i)
                correspondence_set_.push_back(correspondence_set_private[i]);
        }
    }

    matching_rmse_ = std::sqrt(matching_rmse_ / static_cast<double>(correspondence_set_.size()));
}

bool ICP_BASE::convergenceCheck(const Eigen::Matrix4d &transform_iter) const
{
    double relative_matching_rmse = std::abs(matching_rmse_ - matching_rmse_prev_);
    if (relative_matching_rmse > relative_matching_rmse_threshold_)
    {
        return false;
    }

    double cos_theta = 0.5 * (transform_iter.trace() - 2.0);
    if (cos_theta < cos_theta_threshold_)
    {
        return false;
    }

    double trans_sq = transform_iter.block<3, 1>(0, 3).squaredNorm();
    if (trans_sq > translation_threshold_)
    {
        return false;
    }

    return true;
}