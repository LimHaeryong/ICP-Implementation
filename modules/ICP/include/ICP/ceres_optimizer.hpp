#ifndef _ICP_CERES_OPTIMIZER_HPP_
#define _ICP_CERES_OPTIMIZER_HPP_

#include <memory>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>

class CeresOptimizer
{
public:
    enum Type
    {
        PointToPlane,
        GICP
    };

    CeresOptimizer(Type optimizer_type)
        : optimizer_type_(optimizer_type), problem_(std::make_unique<ceres::Problem>())
    {
        options_.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options_.num_threads = 1;
        options_.minimizer_progress_to_stdout = false;
    }

    virtual ~CeresOptimizer() {}

    void clear() 
    { 
        problem_.reset(new ceres::Problem()); 
    }

    void solve() 
    {
        ceres::Solver::Summary summary;
        ceres::Solve(options_, problem_.get(), &summary); 
    }
    void addPointToPlaneResidual(const Eigen::Vector3d &p_source, const Eigen::Vector3d &p_target, const Eigen::Vector3d &norm_target, Eigen::Vector3d &rotation, Eigen::Vector3d &translation);

private:
    Type optimizer_type_;
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Options options_;
    
};

struct PointToPlaneError
{
    PointToPlaneError(const Eigen::Vector3d &p_source, const Eigen::Vector3d &p_target, const Eigen::Vector3d &norm_target)
        : p_source_(p_source), p_target_(p_target), norm_target_(norm_target)
    {
    }

    template <typename T>
    bool operator()(const T *const rotation, const T *const translation, T *residuals) const
    {
        // const Eigen::Matrix<T, 3, 1> p_source = p_source_.cast<T>();
        // const Eigen::Matrix<T, 3, 1> p_target = p_target_.cast<T>();
        // const Eigen::Matrix<T, 3, 1> norm_target = norm_target_.cast<T>();

        // Eigen::Matrix<T, 3, 1> p_diff;
        // ceres::AngleAxisRotatePoint(rotation, p_source.data(), p_diff.data());
        // for (int i = 0; i < 3; ++i)
        //     p_diff[i] += translation[i];
        // p_diff -= p_target;
        // residuals[0] = ceres::DotProduct(norm_target.data(), p_diff.data());

        T p_source[3];
        T p_target[3];
        T norm_target[3];
        T p_diff[3];
        for(int i = 0; i < 3; ++i)
        {
            p_source[i] = static_cast<T>(p_source_(i));
            p_target[i] = static_cast<T>(p_target_(i));
            norm_target[i] = static_cast<T>(norm_target_(i));
        }

        ceres::AngleAxisRotatePoint(rotation, p_source, p_diff);
        for(int i = 0; i < 3; ++i)
            p_diff[i] += translation[i] - p_target[i];
        
        residuals[0] = ceres::DotProduct(norm_target, p_diff);

        return true;
    }

    static ceres::CostFunction *create(const Eigen::Vector3d &p_source, const Eigen::Vector3d &p_target, const Eigen::Vector3d &norm_target)
    {
        return (new ceres::AutoDiffCostFunction<PointToPlaneError, 1, 3, 3>(
            new PointToPlaneError(p_source, p_target, norm_target)));
    }
    const Eigen::Vector3d p_source_;
    const Eigen::Vector3d p_target_;
    const Eigen::Vector3d norm_target_;
};

#endif // _ICP_CERES_OPTIMIZER_HPP_