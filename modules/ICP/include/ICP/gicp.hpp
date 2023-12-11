#ifndef _ICP_GICP_HPP_
#define _ICP_GICP_HPP_

#include "ICP/icp_base.hpp"
#include "ICP/ceres_optimizer.hpp"

class GICP : public ICP_BASE
{
public:
    GICP(SolverType solver_type = SolverType::LeastSquares)
    {
        solver_type_ = solver_type;

        if (solver_type == SolverType::LeastSquaresUsingCeres)
            optimizer_ = std::make_unique<CeresOptimizer>(CeresOptimizer::Type::GICP);
        else if (solver_type == SolverType::SVD)
        {
            spdlog::warn("GICP has no SVD solver. use LeastSquares solver");
            solver_type = SolverType::LeastSquares;
        }
    }

private:
    double cov_epsilon_ = 5e-3;

    bool checkValidity(PointCloud &source_cloud, PointCloud &target_cloud) override;
    Eigen::Matrix4d computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud) override;
    Eigen::Matrix4d computeTransformLeastSquares(const PointCloud &source_cloud, const PointCloud &target_cloud);
    Eigen::Matrix4d computeTransformLeastSquaresUsingCeres(const PointCloud &source_cloud, const PointCloud &target_cloud);
    std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Vector<double, 6>> compute_JTJ_and_JTr(const Eigen::Vector3d &p, const Eigen::Matrix3d &p_cov,
                                                                                         const Eigen::Vector3d &q, const Eigen::Matrix3d &q_cov);
    void computeCovariancesFromNormals(PointCloud &cloud);
    Eigen::Matrix3d getRotationFromNormal(const Eigen::Vector3d &normal);
};

#endif // _ICP_GICP_HPP_