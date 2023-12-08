#ifndef _ICP_UTILS_HPP_
#define _ICP_UTILS_HPP_

#include <Eigen/Dense>

Eigen::Matrix3d createRotationMatrix(const Eigen::Vector3d &r)
{
    Eigen::AngleAxisd rotationX(r[0], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rotationY(r[1], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rotationZ(r[2], Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d rotationMatrix = (rotationZ * rotationY * rotationX).toRotationMatrix();

    return rotationMatrix;
}

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &vec)
{
    Eigen::Matrix3d skew;
    skew << 0.0, -vec[2], vec[1],
        vec[2], 0.0, -vec[0],
        -vec[1], vec[0], 0.0;
    return skew;
}

double GMLoss(double k, double residual_sq)
{
    return k / std::pow(k + residual_sq, 2.0);
}

double HuberLoss(double k, double residual_sq)
{
    if(residual_sq < k * k)
        return 0.5 * residual_sq;
    else
        return k * (std::sqrt(residual_sq) - 0.5 * k);
}

#endif // _ICP_UTILS_HPP_