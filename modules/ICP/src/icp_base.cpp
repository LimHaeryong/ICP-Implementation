#include <algorithm>

#include <spdlog/spdlog.h>
#include <omp.h>

#include "ICP/icp_base.hpp"

void ICP_BASE::correspondenceMatching(const PointCloud &tmp_cloud)
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
            euclidean_error_ += euclidean_error_private;
            for (std::size_t i = 0; i < correspondence_set_private.size(); ++i)
                correspondence_set_.push_back(correspondence_set_private[i]);
        }
    }

    euclidean_error_ = std::sqrt(euclidean_error_ / static_cast<double>(correspondence_set_.size()));
}
