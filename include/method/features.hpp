#pragma once

#include <camera_model.hpp>
#include <ifc_tools/processing/terrain_grid.hpp>

#include <Eigen/Geometry>
#include <ifc_msgs/msg/autopilot_telemetry.hpp>
#include <ifc_msgs/msg/stream_telemetry.hpp>

#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace camera_estimator
{

struct FeatureError
{
    std::array<Eigen::Matrix3d, 2> attitude;
    std::array<Eigen::Vector3d, 2> centre;
    Eigen::Vector2d x1;
    Eigen::Vector2d x2;
    double depth;

    template <typename T>
    bool operator()(const T* rotation, const T* focal_length, const T* principal_point, const T* distortion, T* residual) const
    {
        const auto model = CameraModel<T>::fromBlocks(rotation, focal_length, principal_point, distortion);
        const auto direction = model.ray(attitude[0], x1);

        if (!direction)
        {
            return false;
        }

        const auto x = model.project(model.optical(attitude[1], centre[1], centre[0].cast<T>() + T(depth) * *direction));

        if (!x)
        {
            return false;
        }

        residual[0] = x->x() - T(x2.x());
        residual[1] = x->y() - T(x2.y());
        return true;
    }
};

struct FeatureParameters
{
    Eigen::Vector3d offset = Eigen::Vector3d::Zero();
    double max_range = 3000.0;
};

inline std::optional<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> pose(
    const ifc_msgs::msg::AutopilotTelemetry& telemetry, const Eigen::Vector3d& offset)
{
    const Eigen::Vector4d q(telemetry.attitude[0], telemetry.attitude[1], telemetry.attitude[2], telemetry.attitude[3]);
    const Eigen::Vector3d position(telemetry.position[0], telemetry.position[1], telemetry.position[2]);

    if (!q.allFinite() || q.norm() < 0.5 || !position.allFinite())
    {
        return std::nullopt;
    }

    const Eigen::Matrix3d attitude = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z()).normalized().toRotationMatrix();
    return std::make_pair(attitude, Eigen::Vector3d(position + attitude * offset));
}

inline std::vector<FeatureError> featureErrors(
    const std::vector<ifc_msgs::msg::StreamTelemetry>& messages,
    const ifc_tools::TerrainGrid& grid,
    const CameraModel<double>& model,
    const FeatureParameters& parameters)
{
    std::vector<FeatureError> errors;
    const ifc_tools::TerrainOrigin& origin = grid.origin();

    for (const auto& message : messages)
    {
        const auto& telemetry = message.autopilot_telemetry;

        if (telemetry.ref_lat != origin.ref_lat() || telemetry.ref_lon != origin.ref_lon() || telemetry.ref_alt != origin.ref_alt())
        {
            continue;
        }

        const auto first = pose(message.prev_autopilot_telemetry, parameters.offset);
        const auto second = pose(telemetry, parameters.offset);

        if (!first || !second)
        {
            continue;
        }

        FeatureError error;
        error.attitude = {first->first, second->first};
        error.centre = {first->second, second->second};

        for (const auto& match : message.features.matches)
        {
            if (!match.inlier)
            {
                continue;
            }

            error.x1 = {match.x1, match.y1};
            error.x2 = {match.x2, match.y2};
            const auto direction = model.ray(error.attitude[0], error.x1);

            if (!direction)
            {
                continue;
            }

            const auto hit = grid.raycast(error.centre[0], *direction);

            if (!hit)
            {
                continue;
            }

            const double range = (*hit - error.centre[0]).norm();

            const Eigen::Vector3d camera = model.optical(error.attitude[1], error.centre[1], *hit);

            if (range > parameters.max_range || camera.z() <= 0.0 || !model.project(camera))
            {
                continue;
            }

            error.depth = range / direction->norm();
            errors.push_back(error);
        }
    }

    return errors;
}

}
