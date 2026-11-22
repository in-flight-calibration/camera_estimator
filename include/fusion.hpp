#pragma once

#include <camera_model.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <thread>
#include <vector>

namespace camera_estimator
{

using Vector8 = Eigen::Matrix<double, 8, 1>;
using Matrix8 = Eigen::Matrix<double, 8, 8>;

struct Estimate
{
    CameraModel<double> model;
    Matrix8 covariance = Matrix8::Identity();

    Vector8 vector() const
    {
        Vector8 x;
        x << model.rotation, model.focal_length, model.principal_point, model.distortion;
        return x;
    }

    Vector8 sd() const
    {
        return covariance.diagonal().cwiseSqrt();
    }
};

struct FusionParameters
{
    double huber_k = 2.0;
    double process_noise = 1e-7;
    double sd_inflation = 3.0;
};

class Fusion
{
public:
    explicit Fusion(const FusionParameters& parameters = {})
        : _parameters(parameters)
    {
    }

    Estimate predict(Estimate estimate, double seconds) const
    {
        estimate.covariance += Matrix8::Identity() * (_parameters.process_noise * seconds);
        return estimate;
    }

    template <typename Error>
    std::optional<Estimate> update(const Estimate& prior, const std::vector<Error>& errors, const CameraModel<double>& start) const
    {
        const double sd = robustSd(errors, start) * _parameters.sd_inflation;

        if (!(sd > 0.0))
        {
            return std::nullopt;
        }

        Estimate posterior;
        posterior.model = start;
        CameraModel<double>& model = posterior.model;

        ceres::Problem::Options problem_options;
        problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        ceres::Problem problem(problem_options);
        ceres::HuberLoss loss(_parameters.huber_k);

        for (const Error& error : errors)
        {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<Whitened<Error>, 2, 3, 2, 2, 1>(new Whitened<Error> {error, 1.0 / sd}),
                &loss,
                model.rotation.data(),
                model.focal_length.data(),
                model.principal_point.data(),
                &model.distortion);
        }

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PriorError, 8, 3, 2, 2, 1>(new PriorError(prior)),
            nullptr,
            model.rotation.data(),
            model.focal_length.data(),
            model.principal_point.data(),
            &model.distortion);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        options.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (!summary.IsSolutionUsable())
        {
            return std::nullopt;
        }

        ceres::Covariance::Options covariance_options;
        covariance_options.algorithm_type = ceres::DENSE_SVD;
        covariance_options.num_threads = options.num_threads;
        ceres::Covariance covariance(covariance_options);
        const std::vector<const double*> blocks {model.rotation.data(), model.focal_length.data(), model.principal_point.data(), &model.distortion};

        if (!covariance.Compute(blocks, &problem) || !covariance.GetCovarianceMatrix(blocks, posterior.covariance.data()))
        {
            return std::nullopt;
        }

        if (!posterior.vector().allFinite() || !posterior.covariance.allFinite())
        {
            return std::nullopt;
        }

        return posterior;
    }

private:
    FusionParameters _parameters;

    template <typename Error>
    struct Whitened
    {
        Error error;
        double weight;

        template <typename T>
        bool operator()(const T* rotation, const T* focal_length, const T* principal_point, const T* distortion, T* residual) const
        {
            if (!error(rotation, focal_length, principal_point, distortion, residual))
            {
                return false;
            }

            residual[0] *= T(weight);
            residual[1] *= T(weight);
            return true;
        }
    };

    struct PriorError
    {
        explicit PriorError(const Estimate& prior)
            : mean(prior.vector()), sqrt_information(Matrix8(prior.covariance.inverse()).llt().matrixU())
        {
        }

        template <typename T>
        bool operator()(const T* rotation, const T* focal_length, const T* principal_point, const T* distortion, T* residual) const
        {
            Eigen::Matrix<T, 8, 1> x;
            x << rotation[0], rotation[1], rotation[2], focal_length[0], focal_length[1], principal_point[0], principal_point[1], distortion[0];
            Eigen::Map<Eigen::Matrix<T, 8, 1>> r(residual);
            r = sqrt_information.cast<T>() * (x - mean.cast<T>());
            return true;
        }

        Vector8 mean;
        Matrix8 sqrt_information;
    };

    template <typename Error>
    static double robustSd(const std::vector<Error>& errors, const CameraModel<double>& model)
    {
        std::vector<double> residuals;
        residuals.reserve(2 * errors.size());

        for (const Error& error : errors)
        {
            double residual[2];

            if (!error(model.rotation.data(), model.focal_length.data(), model.principal_point.data(), &model.distortion, residual))
            {
                continue;
            }

            residuals.push_back(std::abs(residual[0]));
            residuals.push_back(std::abs(residual[1]));
        }

        if (residuals.size() < 2 * 8)
        {
            return 0.0;
        }

        const auto middle = residuals.begin() + static_cast<std::ptrdiff_t>(residuals.size() / 2);
        std::nth_element(residuals.begin(), middle, residuals.end());
        return 1.4826 * *middle;
    }
};

}
