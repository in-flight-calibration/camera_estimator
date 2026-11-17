#pragma once

#include <rclcpp/rclcpp.hpp>

namespace camera_estimator
{

class CameraEstimator : public rclcpp::Node
{
public:
    CameraEstimator() : rclcpp::Node("camera_estimator")
    {
    }
};

}
