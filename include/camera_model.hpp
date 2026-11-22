#pragma once

#include <Eigen/Geometry>
#include <ceres/rotation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace camera_estimator
{

template <typename T>
using Vector2 = Eigen::Matrix<T, 2, 1>;

template <typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
class CameraModel
{
public:
    Vector3<T> rotation = Vector3<T>::Zero();
    Vector2<T> focal_length = Vector2<T>::Ones();
    Vector2<T> principal_point = Vector2<T>::Zero();
    T distortion = T(0.0);

    static CameraModel initial(double hfov, const Eigen::Vector3d& roll_pitch_yaw, double width, double height)
    {
        const double fx = 1.0 / std::tan(hfov / 2.0);
        const Eigen::AngleAxisd mount(
            Eigen::AngleAxisd(roll_pitch_yaw.z(), Eigen::Vector3d::UnitZ())
            * Eigen::AngleAxisd(roll_pitch_yaw.y(), Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(roll_pitch_yaw.x(), Eigen::Vector3d::UnitX()));

        CameraModel model;
        model.rotation = (mount.angle() * mount.axis()).cast<T>();
        model.focal_length = Vector2<T>(T(fx), T(fx * width / height));
        return model;
    }

    static CameraModel fromBlocks(const T* rotation, const T* focal_length, const T* principal_point, const T* distortion)
    {
        CameraModel model;
        model.rotation = Vector3<T>(rotation[0], rotation[1], rotation[2]);
        model.focal_length = Vector2<T>(focal_length[0], focal_length[1]);
        model.principal_point = Vector2<T>(principal_point[0], principal_point[1]);
        model.distortion = distortion[0];
        return model;
    }

    std::optional<Vector3<T>> ray(const Eigen::Matrix3d& attitude, const Eigen::Vector2d& x) const
    {
        const T u = (T(x.x()) - principal_point.x()) / focal_length.x();
        const T v = (T(x.y()) - principal_point.y()) / focal_length.y();
        const T denominator = T(1.0) + distortion * (u * u + v * v);

        if (!(denominator > T(0.0)))
        {
            return std::nullopt;
        }

        const std::array<T, 3> camera {T(1.0), u / denominator, v / denominator};
        Vector3<T> body;
        ceres::AngleAxisRotatePoint(rotation.data(), camera.data(), body.data());
        return Vector3<T>(attitude.cast<T>() * body);
    }

    Vector3<T> optical(const Eigen::Matrix3d& attitude, const Eigen::Vector3d& centre, const Vector3<T>& point) const
    {
        const Vector3<T> body = attitude.transpose().cast<T>() * (point - centre.cast<T>());
        const std::array<T, 3> inverse {-rotation.x(), -rotation.y(), -rotation.z()};

        Vector3<T> camera;
        ceres::AngleAxisRotatePoint(inverse.data(), body.data(), camera.data());
        return {camera.y(), camera.z(), camera.x()};
    }

    std::optional<Vector2<T>> project(const Vector3<T>& camera) const
    {
        using std::sqrt;

        const T u = camera.x() / camera.z();
        const T v = camera.y() / camera.z();
        const T discriminant = T(1.0) - T(4.0) * distortion * (u * u + v * v);

        if (!(discriminant > T(0.0)))
        {
            return std::nullopt;
        }

        const T scale = T(2.0) / (T(1.0) + sqrt(discriminant));
        return Vector2<T>(focal_length.x() * scale * u + principal_point.x(), focal_length.y() * scale * v + principal_point.y());
    }

    Eigen::Quaterniond quaternion() const
    {
        const double angle = rotation.norm();

        if (angle < 1e-12)
        {
            return Eigen::Quaterniond::Identity();
        }

        return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation / angle));
    }

    double hfov() const
    {
        return 2.0 * std::atan(1.0 / focal_length.x());
    }

    Eigen::Vector3d rollPitchYaw() const
    {
        const Eigen::Matrix3d r = quaternion().toRotationMatrix();
        return {std::atan2(r(2, 1), r(2, 2)), -std::asin(std::clamp(r(2, 0), -1.0, 1.0)), std::atan2(r(1, 0), r(0, 0))};
    }
};

}
