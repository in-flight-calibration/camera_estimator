cmake_minimum_required(VERSION 3.20)
project(camera_estimator)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ifc_msgs REQUIRED)
find_package(ifc_tools REQUIRED)
find_package(rclcpp REQUIRED)

function(add_camera_executable name source)
    add_executable(${name} ${source})

    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(${name} PRIVATE
        Ceres::ceres
        Eigen3::Eigen
        ifc_msgs::ifc_msgs__rosidl_typesupport_cpp
        ifc_tools::ifc_tools
        rclcpp::rclcpp
    )
endfunction()

add_camera_executable(camera_estimator src/main.cpp)

install(TARGETS camera_estimator
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
    enable_testing()
    add_camera_executable(camera_estimator_estimation test/test_estimation.cpp)
    add_test(NAME camera_estimator_estimation COMMAND camera_estimator_estimation)
endif()

ament_export_dependencies(Ceres Eigen3 ifc_msgs ifc_tools rclcpp)

ament_package()
