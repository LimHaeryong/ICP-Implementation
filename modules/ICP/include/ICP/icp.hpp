#ifndef _ICP_ICP_HPP_
#define _ICP_ICP_HPP_

#include "ICP/icp_base.hpp"

class ICP : public ICP_BASE
{
public:
    ICP(SolverType solver_type = SolverType::LeastSquares)
    {
        solver_type_ = solver_type;
        
        if(solver_type == SolverType::LeastSquaresUsingCeres)
        {
            spdlog::warn("ICP has no Ceres-Solver. use LeastSquares solver");
            solver_type = SolverType::LeastSquares;
        }
    }

private:
    bool checkValidity(PointCloud &source_cloud, PointCloud &target_cloud) override;
    Eigen::Matrix4d computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud) override;
    Eigen::Matrix4d computeTransformSVD(const PointCloud &source_cloud, const PointCloud &target_cloud);
    Eigen::Matrix4d computeTransformLeastSquares(const PointCloud &source_cloud, const PointCloud &target_cloud);
    std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Vector<double, 6>> compute_JTJ_and_JTr(const Eigen::Vector3d &p, const Eigen::Vector3d &q);
};

#endif // _ICP_ICP_HPP_