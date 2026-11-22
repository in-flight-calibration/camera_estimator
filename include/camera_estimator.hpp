#pragma once

#include <camera_model.hpp>
#include <fusion.hpp>
#include <ifc_tools/receiver/terrain_receiver.hpp>
#include <method/features.hpp>
#include <method/horizon.hpp>

#include <Eigen/Core>
#include <ifc_msgs/msg/camera_info.hpp>
#include <ifc_msgs/msg/status.hpp>
#include <ifc_msgs/msg/stream_telemetry.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace camera_estimator
{

class CameraEstimator : public rclcpp::Node
{
public:
    CameraEstimator()
        : rclcpp::Node("camera_estimator")
    {
        constexpr double degree = M_PI / 180.0;

        const auto offset = declare_parameter<std::vector<double>>("camera.offset", {0.0, 0.0, 0.0});
        const auto roll_pitch_yaw = declare_parameter<std::vector<double>>("camera_estimator.initial.roll_pitch_yaw_deg", {0.0, 0.0, 0.0});

        if (offset.size() != 3 || roll_pitch_yaw.size() != 3)
        {
            throw std::invalid_argument("camera.offset and camera_estimator.initial.roll_pitch_yaw_deg must have 3 elements");
        }

        _features.offset = {offset[0], offset[1], offset[2]};
        _features.max_range = declare_parameter<double>("camera_estimator.max_range_m", 3000.0);

        _initial_hfov = declare_parameter<double>("camera_estimator.initial.hfov_deg", 60.0) * degree;
        _initial_roll_pitch_yaw = Eigen::Vector3d(roll_pitch_yaw[0], roll_pitch_yaw[1], roll_pitch_yaw[2]) * degree;
        _initial_sd << Eigen::Vector3d::Constant(declare_parameter<double>("camera_estimator.initial.rotation_sd_deg", 10.0) * degree),
            Eigen::Vector2d::Constant(declare_parameter<double>("camera_estimator.initial.focal_length_sd", 0.5)),
            Eigen::Vector2d::Constant(declare_parameter<double>("camera_estimator.initial.principal_point_sd", 0.05)),
            declare_parameter<double>("camera_estimator.initial.distortion_sd", 0.1);

        _valid_sd << Eigen::Vector3d::Constant(declare_parameter<double>("camera_estimator.valid_sd.rotation_deg", 0.3) * degree),
            Eigen::Vector2d::Constant(declare_parameter<double>("camera_estimator.valid_sd.focal_length", 0.01)),
            Eigen::Vector2d::Constant(declare_parameter<double>("camera_estimator.valid_sd.principal_point", 0.01)),
            declare_parameter<double>("camera_estimator.valid_sd.distortion", 0.01);

        _batch_size = static_cast<std::size_t>(declare_parameter<int>("camera_estimator.batch_size", 100));
        _horizon_batches = declare_parameter<int>("camera_estimator.horizon_batches", 3);
        _passes = declare_parameter<int>("camera_estimator.passes", 3);
        _debug = declare_parameter<bool>("camera_estimator.debug", false);

        FusionParameters fusion;
        fusion.huber_k = declare_parameter<double>("camera_estimator.huber_k", 2.0);
        fusion.process_noise = declare_parameter<double>("camera_estimator.process_noise", 1e-7);
        fusion.sd_inflation = declare_parameter<double>("camera_estimator.sd_inflation", 3.0);
        _fusion = Fusion(fusion);

        _terrain = std::make_shared<ifc_tools::TerrainReceiver>(
            "camera_estimator_terrain",
            declare_parameter<std::string>("terrain_model.topic", "/terrain_model/terrain"));

        _publisher = create_publisher<ifc_msgs::msg::CameraInfo>(
            declare_parameter<std::string>("camera_estimator.topic", "/camera_estimator/camera_info"),
            rclcpp::ServicesQoS());

        _receive_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        _update_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions options;
        options.callback_group = _receive_group;
        _subscription = create_subscription<ifc_msgs::msg::StreamTelemetry>(
            declare_parameter<std::string>("stream_telemetry.topic", "/stream_telemetry/stream_telemetry"),
            rclcpp::ServicesQoS(),
            [this](const ifc_msgs::msg::StreamTelemetry& message) { receive(message); },
            options);

        _timer = create_wall_timer(
            std::chrono::milliseconds(declare_parameter<int>("camera_estimator.update_ms", 1000)),
            [this]() { update(); },
            _update_group);
    }

    std::vector<rclcpp::Node::SharedPtr> children() const
    {
        return {_terrain};
    }

private:
    FeatureParameters _features;
    double _initial_hfov = 0.0;
    Eigen::Vector3d _initial_roll_pitch_yaw;
    Vector8 _initial_sd;
    Vector8 _valid_sd;
    std::size_t _batch_size = 0;
    int _horizon_batches = 0;
    int _passes = 0;
    bool _debug = false;

    std::mutex _mutex;
    std::vector<ifc_msgs::msg::StreamTelemetry> _buffer;
    bool _reset = false;

    std::optional<Estimate> _estimate;
    rclcpp::Time _time;
    std::uint8_t _stage = ifc_msgs::msg::CameraInfo::STAGE_INITIAL_INVALID;
    int _horizon_count = 0;

    Fusion _fusion;
    std::shared_ptr<ifc_tools::TerrainReceiver> _terrain;
    rclcpp::CallbackGroup::SharedPtr _receive_group;
    rclcpp::CallbackGroup::SharedPtr _update_group;
    rclcpp::Publisher<ifc_msgs::msg::CameraInfo>::SharedPtr _publisher;
    rclcpp::Subscription<ifc_msgs::msg::StreamTelemetry>::SharedPtr _subscription;
    rclcpp::TimerBase::SharedPtr _timer;

    void receive(const ifc_msgs::msg::StreamTelemetry& message)
    {
        std::scoped_lock lock(_mutex);

        if (message.stream_info.reset)
        {
            _buffer.clear();
            _reset = true;
        }

        if (message.complete)
        {
            _buffer.push_back(message);
        }
    }

    void update()
    {
        std::vector<ifc_msgs::msg::StreamTelemetry> batch;
        bool reset = false;

        {
            std::scoped_lock lock(_mutex);

            if (_buffer.size() >= _batch_size)
            {
                batch.swap(_buffer);
                reset = std::exchange(_reset, false);
            }
        }

        if (!batch.empty())
        {
            if (reset || !_estimate)
            {
                initialize(batch.front());
            }

            const rclcpp::Time time(batch.back().stream_info.image_timestamp);
            _estimate = _fusion.predict(*_estimate, (time - _time).seconds());
            _time = time;

            const auto begin = std::chrono::steady_clock::now();
            const std::size_t residuals = _horizon_count < _horizon_batches ? updateHorizon(batch) : updateFeatures(batch);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

            if (_debug)
            {
                log(batch.size(), residuals, seconds);
            }
        }

        publish();
    }

    void initialize(const ifc_msgs::msg::StreamTelemetry& message)
    {
        const auto& size = message.stream_info.size;
        Estimate estimate;
        estimate.model = CameraModel<double>::initial(_initial_hfov, _initial_roll_pitch_yaw, size[0], size[1]);
        estimate.covariance = _initial_sd.cwiseAbs2().asDiagonal();
        _estimate = estimate;
        _stage = ifc_msgs::msg::CameraInfo::STAGE_INITIAL_GUESS;
        _horizon_count = 0;
        _time = rclcpp::Time(message.stream_info.image_timestamp);
    }

    std::size_t updateHorizon(const std::vector<ifc_msgs::msg::StreamTelemetry>& batch)
    {
        ++_horizon_count;
        const auto errors = horizonErrors(batch);
        const auto estimate = _fusion.update(*_estimate, errors, _estimate->model);

        if (estimate)
        {
            _estimate = estimate;
            _stage = ifc_msgs::msg::CameraInfo::STAGE_ROUGH_ESTIMATION;
        }

        return errors.size();
    }

    std::size_t updateFeatures(const std::vector<ifc_msgs::msg::StreamTelemetry>& batch)
    {
        const auto grid = _terrain->grid();

        if (!grid)
        {
            return 0;
        }

        std::optional<Estimate> estimate;
        CameraModel<double> model = _estimate->model;
        std::size_t residuals = 0;

        for (int pass = 0; pass < _passes; ++pass)
        {
            const auto errors = featureErrors(batch, *grid, model, _features);
            const auto result = _fusion.update(*_estimate, errors, model);

            if (!result)
            {
                return errors.size();
            }

            estimate = result;
            model = result->model;
            residuals = errors.size();
        }

        _estimate = estimate;
        _stage = ifc_msgs::msg::CameraInfo::STAGE_FINE_ESTIMATION;
        return residuals;
    }

    bool valid() const
    {
        return _estimate && (_estimate->sd().array() <= _valid_sd.array()).all();
    }

    void publish()
    {
        constexpr float nan = std::numeric_limits<float>::quiet_NaN();

        ifc_msgs::msg::CameraInfo message;
        message.header.stamp = now();
        message.status.value = valid() ? ifc_msgs::msg::Status::VALID : ifc_msgs::msg::Status::INVALID;
        message.stage = _stage;
        message.rotation.fill(nan);
        message.rotation_sd.fill(nan);
        message.focal_length.fill(nan);
        message.focal_length_sd.fill(nan);
        message.principal_point.fill(nan);
        message.principal_point_sd.fill(nan);
        message.distortion = nan;
        message.distortion_sd = nan;

        if (_estimate)
        {
            const Eigen::Quaterniond q = _estimate->model.quaternion();
            const Vector8 x = _estimate->vector();
            const Vector8 sd = _estimate->sd();
            message.rotation = {static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()), static_cast<float>(q.w())};

            for (int i = 0; i < 3; ++i)
            {
                message.rotation_sd[i] = static_cast<float>(sd[i]);
            }

            for (int i = 0; i < 2; ++i)
            {
                message.focal_length[i] = static_cast<float>(x[3 + i]);
                message.focal_length_sd[i] = static_cast<float>(sd[3 + i]);
                message.principal_point[i] = static_cast<float>(x[5 + i]);
                message.principal_point_sd[i] = static_cast<float>(sd[5 + i]);
            }

            message.distortion = static_cast<float>(x[7]);
            message.distortion_sd = static_cast<float>(sd[7]);
        }

        _publisher->publish(message);
    }

    void log(std::size_t frames, std::size_t residuals, double seconds) const
    {
        constexpr double degree = M_PI / 180.0;
        const CameraModel<double>& model = _estimate->model;
        const Eigen::Vector3d rpy = model.rollPitchYaw() / degree;
        const Vector8 sd = _estimate->sd();
        RCLCPP_INFO(get_logger(),
            "%s stage %u frames %zu residuals %zu | HFOV %.2f deg | roll/pitch/yaw %+.2f/%+.2f/%+.2f deg (sd %.2f/%.2f/%.2f) "
            "| c %+.4f/%+.4f (sd %.4f/%.4f) | f/f %.4f (sd %.4f) | lambda %+.4f (sd %.4f) | %.2f s",
            valid() ? "valid" : "invalid", _stage, frames, residuals,
            model.hfov() / degree, rpy.x(), rpy.y(), rpy.z(), sd[0] / degree, sd[1] / degree, sd[2] / degree,
            model.principal_point.x(), model.principal_point.y(), sd[5], sd[6],
            model.focal_length.y() / model.focal_length.x(), sd[3], model.distortion, sd[7], seconds);
    }
};

}
