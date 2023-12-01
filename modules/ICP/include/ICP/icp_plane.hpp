#ifndef _ICP_ICP_PLANE_HPP_
#define _ICP_ICP_PLANE_HPP_

#include "ICP/icp_base.hpp"
#include "ICP/ceres_optimizer.hpp"

class ICP_PLANE : public ICP_BASE
{
public:
    enum SolverType
    {
        Linear,
        NonLinear
    };

    ICP_PLANE(SolverType solverType = SolverType::Linear)
        : solverType_(solverType)
    {
        if(solverType == SolverType::NonLinear)
            optimizer_ = std::make_unique<CeresOptimizer>(CeresOptimizer::Type::PointToPlane);
    }

    void align(const PointCloud &source_cloud, const PointCloud &target_cloud) override;

private:
    SolverType solverType_;
    std::unique_ptr<CeresOptimizer> optimizer_;
    Eigen::Matrix4d computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud);
    Eigen::Matrix4d computeTransformNonlinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud);
    Eigen::Matrix4d computeTransformLinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud);
};

#endif // _ICP_ICP_PLANE_HPP_