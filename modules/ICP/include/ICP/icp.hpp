#ifndef _ICP_ICP_HPP_
#define _ICP_ICP_HPP_

#include "ICP/icp_base.hpp"

class ICP : public ICP_BASE
{
public:
    ICP() {}
private:
    bool checkValidity(PointCloud &source_cloud, PointCloud &target_cloud) override;
    Eigen::Matrix4d computeTransform(const PointCloud &source_cloud, const PointCloud &target_cloud) override;
};

#endif // _ICP_ICP_HPP_