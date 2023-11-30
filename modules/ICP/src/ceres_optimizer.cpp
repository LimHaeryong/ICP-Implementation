#include "ICP/ceres_optimizer.hpp"

void CeresOptimizer::addPointToPlaneResidual(const Eigen::Vector3d &p_source, const Eigen::Vector3d &p_target, const Eigen::Vector3d &norm_target, Eigen::Vector3d &rotation, Eigen::Vector3d &translation)
{
    auto cost_functor = PointToPlaneError::create(p_source, p_target, norm_target);
    problem_->AddResidualBlock(cost_functor, nullptr, rotation.data(), translation.data());
}