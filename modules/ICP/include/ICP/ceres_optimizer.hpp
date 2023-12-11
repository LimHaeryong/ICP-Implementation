#ifndef _ICP_CERES_OPTIMIZER_HPP_
#define _ICP_CERES_OPTIMIZER_HPP_

#include <omp.h>
#include <memory>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>

class CeresOptimizer
{
public:
    enum class Type
    {
        PointToPlane,
        GICP
    };

    CeresOptimizer(Type optimizer_type)
        : optimizer_type_(optimizer_type), problem_(std::make_unique<ceres::Problem>())
    {

        options_.num_threads = omp_get_max_threads();
        options_.minimizer_progress_to_stdout = false;
        options_.logging_type = ceres::SILENT;

        switch (optimizer_type_)
        {
        case Type::PointToPlane:
            quaternion_manifold_ = new ceres::EigenQuaternionManifold();
            loss_function_ = new ceres::HuberLoss(1.0);
            options_.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;
            options_.max_num_iterations = 1;
            options_.parameter_tolerance = 1e-8;
            options_.gradient_tolerance = 1e-8;
            options_.function_tolerance = 1e-4;
            break;
        case Type::GICP:
            quaternion_manifold_ = new ceres::EigenQuaternionManifold();
            loss_function_ = new ceres::HuberLoss(1.0);
            options_.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;
            options_.max_num_iterations = 1;
            options_.parameter_tolerance = 1e-8;
            options_.gradient_tolerance = 1e-8;
            options_.function_tolerance = 1e-4;
            break;
        }
    }

    virtual ~CeresOptimizer()
    {
        if (quaternion_manifold_ != nullptr)
            delete quaternion_manifold_;

        if (loss_function_ != nullptr)
            delete loss_function_;
    }

    void clear()
    {
        problem_.reset(new ceres::Problem());
        switch (optimizer_type_)
        {
        case Type::PointToPlane:
            quaternion_manifold_ = new ceres::EigenQuaternionManifold();
            loss_function_ = new ceres::HuberLoss(1.0);
            break;
        case Type::GICP:
            quaternion_manifold_ = new ceres::EigenQuaternionManifold();
            loss_function_ = new ceres::HuberLoss(1.0);
            break;
        }
    }

    void solve()
    {
        ceres::Solver::Summary summary;
        ceres::Solve(options_, problem_.get(), &summary);
    }
    void addPointToPlaneResidual(const Eigen::Vector3d &p_source, const Eigen::Vector3d &p_target, const Eigen::Vector3d &norm_target, Eigen::Quaterniond &rotation, Eigen::Vector3d &translation);
    void addGICPResidual(const Eigen::Vector3d &p_source, const Eigen::Matrix3d &cov_source, const Eigen::Vector3d &p_target, const Eigen::Matrix3d &cov_target, Eigen::Quaterniond &rotation, Eigen::Vector3d &translation);

private:
    Type optimizer_type_;
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Options options_;
    ceres::Manifold *quaternion_manifold_ = nullptr;
    ceres::LossFunction *loss_function_ = nullptr;
};

struct PointToPlaneError
{
    PointToPlaneError(const Eigen::Vector3d &p_source, const Eigen::Vector3d &p_target, const Eigen::Vector3d &norm_target)
        : p_source_(p_source), p_target_(p_target), norm_target_(norm_target)
    {
    }

    template <typename T>
    bool operator()(const T *const rotation_ptr, const T *const translation_ptr, T *residuals) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> rotation(rotation_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(translation_ptr);

        Eigen::Matrix<T, 3, 1> p_diff = rotation * p_source_.cast<T>() + translation - p_target_.cast<T>();
        residuals[0] = p_diff.dot(norm_target_.cast<T>());
        return true;
    }

    static ceres::CostFunction *create(const Eigen::Vector3d &p_source, const Eigen::Vector3d &p_target, const Eigen::Vector3d &norm_target)
    {
        return (new ceres::AutoDiffCostFunction<PointToPlaneError, 1, 4, 3>(
            new PointToPlaneError(p_source, p_target, norm_target)));
    }
    const Eigen::Vector3d p_source_;
    const Eigen::Vector3d p_target_;
    const Eigen::Vector3d norm_target_;
};

struct GICPError
{
    GICPError(const Eigen::Vector3d &p_source, const Eigen::Matrix3d &cov_source,
              const Eigen::Vector3d &p_target, const Eigen::Matrix3d &cov_target)
        : p_source_(p_source), cov_source_(cov_source), p_target_(p_target), cov_target_(cov_target)
    {
    }

    template <typename T>
    bool operator()(const T *const rotation_ptr, const T *const translation_ptr, T *residuals) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> rotation(rotation_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(translation_ptr);
        Eigen::Matrix<T, 3, 1> p_diff = rotation * p_source_.cast<T>() + translation - p_target_.cast<T>();
        Eigen::Matrix<T, 3, 3> cov = cov_target_.cast<T>() + rotation * cov_source_.cast<T>() * rotation.conjugate();
        residuals[0] = p_diff.transpose() * cov.inverse() * p_diff;
        return true;
    }

    static ceres::CostFunction *create(const Eigen::Vector3d &p_source, const Eigen::Matrix3d &cov_source,
                                       const Eigen::Vector3d &p_target, const Eigen::Matrix3d &cov_target)
    {
        return (new ceres::AutoDiffCostFunction<GICPError, 1, 4, 3>(
            new GICPError(p_source, cov_source, p_target, cov_target)));
    }

    const Eigen::Vector3d p_source_;
    const Eigen::Matrix3d cov_source_;
    const Eigen::Vector3d p_target_;
    const Eigen::Matrix3d cov_target_;
};

#endif // _ICP_CERES_OPTIMIZER_HPP_