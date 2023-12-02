#include "ICP/ceres_optimizer.hpp"

void CeresOptimizer::addPointToPlaneResidual(const Eigen::Vector3d &p_source, const Eigen::Vector3d &p_target, const Eigen::Vector3d &norm_target, Eigen::Quaterniond &rotation, Eigen::Vector3d &translation)
{
    auto cost_function = PointToPlaneError::create(p_source, p_target, norm_target);
    problem_->AddResidualBlock(cost_function, loss_function_, rotation.coeffs().data(), translation.data());
    problem_->SetManifold(rotation.coeffs().data(), quaternion_manifold_);
}

void CeresOptimizer::addGICPResidual(const Eigen::Vector3d &p_source, const Eigen::Matrix3d &cov_source, const Eigen::Vector3d &p_target, const Eigen::Matrix3d &cov_target, Eigen::Quaterniond &rotation, Eigen::Vector3d &translation)
{
    auto cost_function = GICPError::create(p_source, cov_source, p_target, cov_target);
    problem_->AddResidualBlock(cost_function, loss_function_, rotation.coeffs().data(), translation.data());
    problem_->SetManifold(rotation.coeffs().data(), quaternion_manifold_);
}