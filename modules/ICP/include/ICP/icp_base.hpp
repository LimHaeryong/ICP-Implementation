#ifndef _ICP_ICP_BASE_HPP_
#define _ICP_ICP_BASE_HPP_

#include <memory>

#include <open3d/Open3D.h>
#include <Eigen/Dense>

#include "ICP/ceres_optimizer.hpp"

class ICP_BASE
{
public:
    using PointCloud = typename open3d::geometry::PointCloud;
    using PointCloudPtr = typename std::shared_ptr<PointCloud>;
    using KDTree = typename open3d::geometry::KDTreeFlann;
    using KDTreePtr = typename std::shared_ptr<KDTree>;

    enum class SolverType
    {
        SVD,
        LeastSquares,
        LeastSquaresUsingCeres
    };

    ICP_BASE() {}

    void align(PointCloud &source_cloud, PointCloud &target_cloud);

    void setIteration(int iteration) { max_iteration_ = iteration; }
    void setMaxCorrespondenceDist(double dist) { max_corres_dist_ = dist; }
    
    // not converged if abs(current rmse of corresponded points - prev) > threshold
    void setRelativeMatchingRmseThreshold(double threshold) { relative_matching_rmse_threshold_ = threshold; }
    
    // not converged if squared norm of translation > threshold
    void setTranslationThreshold(double threshold) { translation_threshold_ = threshold; }
    
    // not converged if cos(theta) < threshold 
    void setRotationThreshold(double threshold) { cos_theta_threshold_ = threshold; }

    Eigen::Matrix4d getResultTransform() const { return total_transform_; }
    bool hasConverged() const { return converged_; }

protected:
    void correspondenceMatching(const PointCloud &tmp_cloud);
    virtual bool checkValidity(PointCloud &source_cloud, PointCloud &target_cloud) = 0;
    virtual Eigen::Matrix4d computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud) = 0;
    bool convergenceCheck(const Eigen::Matrix4d& transform_iter) const;

    SolverType solver_type_;
    std::unique_ptr<CeresOptimizer> optimizer_;
    KDTreePtr tree_ = nullptr;
    Eigen::Matrix4d total_transform_ = Eigen::Matrix4d::Identity();
    std::vector<std::pair<int, int>> correspondence_set_;
    int max_iteration_ = 30;
    bool converged_ = false;
    double max_corres_dist_ = 10.0;
    double relative_matching_rmse_threshold_ = 1e-6;
    double translation_threshold_ = 1e-6;
    double cos_theta_threshold_ = 1.0 - 1e-5;
    double matching_rmse_ = std::numeric_limits<double>::max();
    double matching_rmse_prev_ = std::numeric_limits<double>::max();
};

#endif // _ICP_ICP_BASE_HPP_