#ifndef _ICP_ICP_PLANE_HPP_
#define _ICP_ICP_PLANE_HPP_

#include "ICP/icp_base.hpp"
#include "ICP/ceres_optimizer.hpp"

class ICP_PLANE : public ICP_BASE
{
public:
    ICP_PLANE()
        : optimizer_(std::make_unique<CeresOptimizer>(CeresOptimizer::Type::PointToPlane))
    {
    }

    void align(const PointCloud &source_cloud, const PointCloud &target_cloud) override;

private:
    std::unique_ptr<CeresOptimizer> optimizer_;
    Eigen::Matrix4d computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud);
};

#endif // _ICP_ICP_PLANE_HPP_