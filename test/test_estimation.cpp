#include <camera_model.hpp>
#include <fusion.hpp>
#include <ifc_tools/processing/terrain_grid.hpp>
#include <method/features.hpp>
#include <method/horizon.hpp>

#include <Eigen/Geometry>
#include <ifc_msgs/msg/stream_telemetry.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace
{

constexpr double width = 1280.0;
constexpr double height = 720.0;
constexpr double degree = M_PI / 180.0;

bool check(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAILED: %s\n", message);
    }

    return condition;
}

ifc_tools::TerrainGrid terrain()
{
    const ifc_tools::TerrainAxis axis(-3000.0f, 10.0f, 3000.0f);
    Eigen::MatrixXd z(axis.size(), axis.size());

    for (Eigen::Index i = 0; i < z.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < z.cols(); ++j)
        {
            const double x = axis.start() + static_cast<double>(i) * axis.step();
            const double y = axis.start() + static_cast<double>(j) * axis.step();
            const double hill = 60.0 * std::exp(-((x - 300.0) * (x - 300.0) + (y + 200.0) * (y + 200.0)) / (2.0 * 400.0 * 400.0));
            z(i, j) = -(0.02 * x - 0.01 * y + hill);
        }
    }

    return ifc_tools::TerrainGrid(ifc_tools::TerrainOrigin(52.0, 21.0, 100.0), axis, axis, z);
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> pose(double t)
{
    const double radius = 500.0;
    const double rate = 25.0 / radius;
    const double heading = rate * t + M_PI / 2.0;
    const double roll = 0.3 + 0.4 * std::sin(2.0 * t);
    const double pitch = -0.1 + 0.2 * std::sin(1.7 * t + 1.0);
    const double yaw = heading + 0.15 * std::sin(2.3 * t);

    const Eigen::Matrix3d attitude = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const Eigen::Vector3d position(radius * std::cos(rate * t), radius * std::sin(rate * t), -180.0 + 10.0 * std::sin(0.4 * t));
    return {attitude, position};
}

std::optional<Eigen::Vector2d> project(const camera_estimator::CameraModel<double>& model, const Eigen::Matrix3d& attitude, const Eigen::Vector3d& centre, const Eigen::Vector3d& point)
{
    const Eigen::Vector3d camera = model.optical(attitude, centre, point);

    if (camera.z() <= 0.0)
    {
        return std::nullopt;
    }

    const auto x = model.project(camera);

    if (!x || x->cwiseAbs().maxCoeff() > 1.0)
    {
        return std::nullopt;
    }

    return x;
}

void fill(ifc_msgs::msg::AutopilotTelemetry& telemetry, const Eigen::Matrix3d& attitude, const Eigen::Vector3d& position, const ifc_tools::TerrainOrigin& origin)
{
    const Eigen::Quaterniond q(attitude);
    telemetry.attitude = {static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()), static_cast<float>(q.w())};
    telemetry.position = {static_cast<float>(position.x()), static_cast<float>(position.y()), static_cast<float>(position.z())};
    telemetry.ref_lat = origin.ref_lat();
    telemetry.ref_lon = origin.ref_lon();
    telemetry.ref_alt = origin.ref_alt();
}

void horizon(ifc_msgs::msg::Horizon& horizon, const camera_estimator::CameraModel<double>& truth, const Eigen::Matrix3d& attitude, const Eigen::Vector3d& centre, std::mt19937& rng)
{
    const double half_diagonal = 0.5 * std::hypot(width, height);
    const Eigen::Vector3d forward = attitude * truth.quaternion() * Eigen::Vector3d::UnitX();
    const double azimuth = std::atan2(forward.y(), forward.x());
    std::array<Eigen::Vector2d, 2> points;

    for (int i = 0; i < 2; ++i)
    {
        const double a = azimuth + (i == 0 ? -0.2 : 0.2);
        const Eigen::Vector3d far = centre + 1e7 * Eigen::Vector3d(std::cos(a), std::sin(a), 0.0);
        const Eigen::Vector2d x = truth.project(truth.optical(attitude, centre, far)).value_or(Eigen::Vector2d::Zero());
        points[i] = {x.x() * 0.5 * width / half_diagonal, -x.y() * 0.5 * height / half_diagonal};
    }

    const Eigen::Vector2d direction = (points[1] - points[0]).normalized();
    double theta = std::atan2(direction.y(), direction.x());
    double rho = -points[0].x() * std::sin(theta) + points[0].y() * std::cos(theta);

    if (std::abs(theta) > M_PI / 2.0)
    {
        theta -= std::copysign(M_PI, theta);
        rho = -rho;
    }

    std::normal_distribution<double> theta_noise(0.0, 0.5 * degree);
    std::normal_distribution<double> rho_noise(0.0, 0.005);
    horizon.theta = static_cast<float>(theta + theta_noise(rng));
    horizon.rho = static_cast<float>(rho + rho_noise(rng));
    horizon.confidence = 1.0f;
}

std::vector<ifc_msgs::msg::StreamTelemetry> messages(const camera_estimator::CameraModel<double>& truth, const ifc_tools::TerrainGrid& grid)
{
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> uniform(-0.95, 0.95);
    std::normal_distribution<double> noise(0.0, 0.0015);
    std::vector<ifc_msgs::msg::StreamTelemetry> messages;

    for (int k = 0; k < 900; ++k)
    {
        const double t = 0.25 * k;
        const auto [attitude1, centre1] = pose(t);
        const auto [attitude2, centre2] = pose(t + 1.0 / 30.0);

        ifc_msgs::msg::StreamTelemetry message;
        message.complete = true;
        message.stream_info.size = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
        message.stream_info.image_timestamp.sec = k + 1;
        message.horizon.image_timestamp = message.stream_info.image_timestamp;
        fill(message.prev_autopilot_telemetry, attitude1, centre1, grid.origin());
        fill(message.autopilot_telemetry, attitude2, centre2, grid.origin());
        horizon(message.horizon, truth, attitude2, centre2, rng);

        for (int i = 0; i < 40; ++i)
        {
            const Eigen::Vector2d x1(uniform(rng), uniform(rng));
            const auto direction = truth.ray(attitude1, x1);

            if (!direction)
            {
                continue;
            }

            const auto hit = grid.raycast(centre1, *direction);

            if (!hit)
            {
                continue;
            }

            const auto x2 = project(truth, attitude2, centre2, *hit);

            if (!x2)
            {
                continue;
            }

            ifc_msgs::msg::Match match;
            match.x1 = static_cast<float>(x1.x() + noise(rng));
            match.y1 = static_cast<float>(x1.y() + noise(rng));
            match.x2 = static_cast<float>(x2->x() + noise(rng));
            match.y2 = static_cast<float>(x2->y() + noise(rng));
            match.inlier = true;
            message.features.matches.push_back(match);
        }

        messages.push_back(message);
    }

    return messages;
}

void print(const char* name, const camera_estimator::Estimate& estimate)
{
    const camera_estimator::CameraModel<double>& model = estimate.model;
    const Eigen::Vector3d rpy = model.rollPitchYaw() / degree;
    const camera_estimator::Vector8 sd = estimate.sd();
    std::printf("%s: HFOV %.3f roll/pitch/yaw %+.3f/%+.3f/%+.3f (sd %.3f/%.3f/%.3f) c %+.4f/%+.4f lambda %+.4f (sd %.4f)\n",
        name, model.hfov() / degree, rpy.x(), rpy.y(), rpy.z(), sd[0] / degree, sd[1] / degree, sd[2] / degree,
        model.principal_point.x(), model.principal_point.y(), model.distortion, sd[7]);
}

std::optional<camera_estimator::Estimate> estimate(const camera_estimator::CameraModel<double>& truth)
{
    const auto grid = terrain();
    const auto all = messages(truth, grid);
    const camera_estimator::Fusion fusion;
    const camera_estimator::FeatureParameters parameters;

    camera_estimator::Estimate estimate;
    estimate.model = camera_estimator::CameraModel<double>::initial(60.0 * degree, Eigen::Vector3d::Zero(), width, height);
    camera_estimator::Vector8 sd;
    sd << Eigen::Vector3d::Constant(10.0 * degree), Eigen::Vector2d::Constant(0.5), Eigen::Vector2d::Constant(0.05), 0.1;
    estimate.covariance = sd.cwiseAbs2().asDiagonal();

    for (std::size_t begin = 0; begin < all.size(); begin += 100)
    {
        const std::vector<ifc_msgs::msg::StreamTelemetry> batch(all.begin() + static_cast<std::ptrdiff_t>(begin), all.begin() + static_cast<std::ptrdiff_t>(begin + 100));
        estimate = fusion.predict(estimate, 25.0);

        if (begin < 300)
        {
            const auto result = fusion.update(estimate, camera_estimator::horizonErrors(batch), estimate.model);

            if (!check(result.has_value(), "horizon update"))
            {
                return std::nullopt;
            }

            estimate = *result;
            print("horizon ", estimate);
            continue;
        }

        camera_estimator::CameraModel<double> model = estimate.model;
        std::optional<camera_estimator::Estimate> result;

        for (int pass = 0; pass < 3; ++pass)
        {
            result = fusion.update(estimate, camera_estimator::featureErrors(batch, grid, model, parameters), model);

            if (!check(result.has_value(), "feature update"))
            {
                return std::nullopt;
            }

            model = result->model;
        }

        estimate = *result;
        print("features", estimate);
    }

    return estimate;
}

}

int main()
{
    camera_estimator::CameraModel<double> truth = camera_estimator::CameraModel<double>::initial(65.0 * degree, Eigen::Vector3d(2.0, -5.0, -1.0) * degree, width, height);
    truth.principal_point = {0.01, -0.02};
    bool ok = true;

    const auto pinhole = estimate(truth);
    ok &= check(pinhole.has_value(), "pinhole estimate");

    if (pinhole)
    {
        const Eigen::Vector3d error = (pinhole->model.rollPitchYaw() - truth.rollPitchYaw()) / degree;
        ok &= check(std::abs(pinhole->model.hfov() - truth.hfov()) / degree < 0.2, "HFOV within 0.2 deg");
        ok &= check(error.cwiseAbs().maxCoeff() < 0.2, "mount within 0.2 deg");
        ok &= check((pinhole->model.principal_point - truth.principal_point).cwiseAbs().maxCoeff() < 0.005, "principal point within 0.005");
        ok &= check(std::abs(pinhole->model.distortion) < 0.005, "no distortion within 0.005");
        ok &= check(pinhole->sd().allFinite(), "finite sd");
    }

    truth.distortion = -0.08;
    const auto distorted = estimate(truth);
    ok &= check(distorted.has_value(), "distorted estimate");

    if (distorted)
    {
        ok &= check(std::abs(distorted->model.hfov() - truth.hfov()) / degree < 0.2, "distorted HFOV within 0.2 deg");
        ok &= check(std::abs(distorted->model.distortion - truth.distortion) < 0.005, "distortion within 0.005");
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
