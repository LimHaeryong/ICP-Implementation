#ifndef _ICP_ICP_HPP_
#define _ICP_ICP_HPP_

#include "ICP/icp_base.hpp"

class ICP : public ICP_BASE
{
public:

    ICP() {}
    void align(const PointCloud& source_cloud, const PointCloud& target_cloud) override;

private:
    void correspondenceMatching(const PointCloud& tmp_cloud);
    Eigen::Matrix4d computeTransform(const PointCloud& source_cloud, const PointCloud& target_cloud);

    
};

#endif // _ICP_ICP_HPP_