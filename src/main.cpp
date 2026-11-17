#include <camera_estimator.hpp>

#include <memory>

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<camera_estimator::CameraEstimator>();

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
