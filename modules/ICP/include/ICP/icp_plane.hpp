#ifndef _ICP_ICP_PLANE_HPP_
#define _ICP_ICP_PLANE_HPP_

#include "ICP/icp_base.hpp"
#include "ICP/ceres_optimizer.hpp"

class ICP_PLANE : public ICP_BASE
{
public:
    

    ICP_PLANE(SolverType solverType = SolverType::Linear)
        : solverType_(solverType)
    {
        if (solverType == SolverType::NonLinear)
            optimizer_ = std::make_unique<CeresOptimizer>(CeresOptimizer::Type::PointToPlane);
    }

private:
    bool checkValidity(PointCloud &source_cloud, PointCloud &target_cloud) override;
    Eigen::Matrix4d computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud) override;
    std::pair<Eigen::Matrix<double, 1, 6>, Eigen::Vector<double, 1>> compute_Ai_and_bi(const Eigen::Vector3d &pi, const Eigen::Vector3d &qi, const Eigen::Vector3d& q_norm_i);

    SolverType solverType_;
    std::unique_ptr<CeresOptimizer> optimizer_;

    Eigen::Matrix4d computeTransformNonlinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud);
    Eigen::Matrix4d computeTransformLinearSolver(const PointCloud &source_cloud, const PointCloud &target_cloud);
};

#endif // _ICP_ICP_PLANE_HPP_