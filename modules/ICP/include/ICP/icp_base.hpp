#ifndef _ICP_ICP_BASE_HPP_
#define _ICP_ICP_BASE_HPP_

#include <memory>

#include <open3d/Open3D.h>
#include <Eigen/Dense>

class ICP_BASE
{
public:
    using PointCloud = typename open3d::geometry::PointCloud;
    using PointCloudPtr = typename std::shared_ptr<PointCloud>;
    using KDTree = typename open3d::geometry::KDTreeFlann;
    using KDTreePtr = typename std::shared_ptr<KDTree>;

    ICP_BASE() {}

    virtual void align(const PointCloud& source_cloud, const PointCloud& target_cloud) = 0;

    void setIteration(int iteration) { max_iteration_ = iteration; }
    void setMaxCorrespondenceDist(double dist) { max_corres_dist_ = dist; }
    void setEuclideanFitnessEpsilon(double epsilon) { euclidean_fitness_epsilon_ = epsilon; }
    void setTransformationEpsilon(double epsilon) { transformation_epsilon_ = epsilon; }

    Eigen::Matrix4d getResultTransform() const { return total_transform_; }
    bool hasConverged() const { return converged_; }

protected: 
    void correspondenceMatching(const PointCloud &tmp_cloud);

    KDTreePtr tree_ = nullptr;
    Eigen::Matrix4d total_transform_ = Eigen::Matrix4d::Identity();
    std::vector<std::pair<int, int>> correspondence_set_;
    int max_iteration_ = 30;
    bool converged_ = false;
    double max_corres_dist_ = 10.0;
    double euclidean_fitness_epsilon_ = 1e-6;
    double transformation_epsilon_ = 1e-6;
    double euclidean_error_ = 0.0;
    
};

#endif // _ICP_ICP_BASE_HPP_