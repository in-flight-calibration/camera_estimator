#pragma once

#include <Eigen/Geometry>
#include <ceres/rotation.h>
#include <ifc_msgs/msg/stream_telemetry.hpp>

#include <array>
#include <cmath>
#include <vector>

namespace camera_estimator
{

struct HorizonError
{
    Eigen::Vector3d gravity;
    std::array<Eigen::Vector2d, 2> points;
    Eigen::Vector2d scale;

    template <typename T>
    bool operator()(const T* rotation, const T* focal_length, const T* principal_point, const T*, T* residual) const
    {
        const std::array<T, 3> body {T(gravity.x()), T(gravity.y()), T(gravity.z())};
        const std::array<T, 3> inverse {-rotation[0], -rotation[1], -rotation[2]};
        std::array<T, 3> camera;
        ceres::AngleAxisRotatePoint(inverse.data(), body.data(), camera.data());

        const T a = camera[1] / focal_length[0];
        const T b = camera[2] / focal_length[1];
        const T d = camera[0] - a * principal_point[0] - b * principal_point[1];
        const T u = a * T(scale.x());
        const T v = -b * T(scale.y());
        const T norm = ceres::sqrt(u * u + v * v);

        for (int i = 0; i < 2; ++i)
        {
            residual[i] = (u * T(points[i].x()) + v * T(points[i].y()) + d) / norm;
        }

        return true;
    }
};

inline std::vector<HorizonError> horizonErrors(const std::vector<ifc_msgs::msg::StreamTelemetry>& messages)
{
    std::vector<HorizonError> errors;

    for (const auto& message : messages)
    {
        const auto& horizon = message.horizon;
        const auto& attitude = message.autopilot_telemetry.attitude;
        const Eigen::Vector4d q(attitude[0], attitude[1], attitude[2], attitude[3]);

        if (!(horizon.confidence > 0.0f) || !std::isfinite(horizon.theta) || !std::isfinite(horizon.rho)
            || horizon.image_timestamp != message.stream_info.image_timestamp || !q.allFinite() || q.norm() < 0.5)
        {
            continue;
        }

        const Eigen::Quaterniond orientation = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z()).normalized();
        const double width = message.stream_info.size[0];
        const double height = message.stream_info.size[1];
        const double half_diagonal = 0.5 * std::hypot(width, height);
        const Eigen::Vector2d normal(-std::sin(horizon.theta), std::cos(horizon.theta));
        const Eigen::Vector2d direction(std::cos(horizon.theta), std::sin(horizon.theta));
        const Eigen::Vector2d foot = horizon.rho * normal;

        HorizonError error;
        error.gravity = orientation.conjugate() * Eigen::Vector3d::UnitZ();
        error.points = {foot - 0.5 * direction, foot + 0.5 * direction};
        error.scale = {2.0 * half_diagonal / width, 2.0 * half_diagonal / height};
        errors.push_back(error);
    }

    return errors;
}

}
