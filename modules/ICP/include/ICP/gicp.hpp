#ifndef _ICP_GICP_HPP_
#define _ICP_GICP_HPP_

#include "ICP/icp_base.hpp"
#include "ICP/ceres_optimizer.hpp"

class GICP : public ICP_BASE
{
public:
    GICP(SolverType solverType = SolverType::NonLinear)
        : solverType_(solverType)
    {
        if (solverType == SolverType::NonLinear)
            optimizer_ = std::make_unique<CeresOptimizer>(CeresOptimizer::Type::GICP);
    }

private:
    bool checkValidity(PointCloud &source_cloud, PointCloud &target_cloud) override;
    Eigen::Matrix4d computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud) override;

    SolverType solverType_;
    std::unique_ptr<CeresOptimizer> optimizer_;

    double cov_epsilon_ = 1e-3;

    Eigen::Matrix4d computeTransformNonlinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud);
    Eigen::Matrix4d computeTransformLinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud);

    void computeCovariancesFromNormals(PointCloud &cloud);
    Eigen::Matrix3d getRotationFromNormal(const Eigen::Vector3d &normal);
};

#endif // _ICP_GICP_HPP_