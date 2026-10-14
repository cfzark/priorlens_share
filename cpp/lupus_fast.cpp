#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

inline double softplus(double value) {
    if (value > 0.0) {
        return value + std::log1p(std::exp(-value));
    }
    return std::log1p(std::exp(value));
}

inline double logistic(double value) {
    if (value >= 0.0) {
        const double z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(value);
    return z / (1.0 + z);
}

template <typename Levels, typename Totals, typename Cases>
double log_posterior_unnormalized(
    double beta0,
    double beta1,
    ssize_t row,
    const Levels &levels,
    const Totals &totals,
    const Cases &cases,
    double loc0,
    double loc1,
    double scale0,
    double scale1,
    double df
) {
    double value = 0.0;
    for (ssize_t j = 0; j < levels.shape(0); ++j) {
        const double eta = beta0 + beta1 * levels(j);
        value += cases(row, j) * eta - totals(row, j) * softplus(eta);
    }
    const double z0 = (beta0 - loc0) / scale0;
    const double z1 = (beta1 - loc1) / scale1;
    value -= 0.5 * (df + 1.0) * std::log1p(z0 * z0 / df);
    value -= 0.5 * (df + 1.0) * std::log1p(z1 * z1 / df);
    return value;
}

template <typename Levels, typename Totals, typename Cases>
void derivatives(
    double beta0,
    double beta1,
    ssize_t row,
    const Levels &levels,
    const Totals &totals,
    const Cases &cases,
    double loc0,
    double loc1,
    double scale0,
    double scale1,
    double df,
    double &g0,
    double &g1,
    double &h00,
    double &h01,
    double &h11
) {
    g0 = 0.0;
    g1 = 0.0;
    h00 = 0.0;
    h01 = 0.0;
    h11 = 0.0;
    for (ssize_t j = 0; j < levels.shape(0); ++j) {
        const double x = levels(j);
        const double probability = logistic(beta0 + beta1 * x);
        const double residual = cases(row, j) - totals(row, j) * probability;
        const double weight = totals(row, j) * probability * (1.0 - probability);
        g0 += residual;
        g1 += residual * x;
        h00 += weight;
        h01 += weight * x;
        h11 += weight * x * x;
    }
    const double delta0 = beta0 - loc0;
    const double delta1 = beta1 - loc1;
    const double a0 = df * scale0 * scale0;
    const double a1 = df * scale1 * scale1;
    g0 -= (df + 1.0) * delta0 / (a0 + delta0 * delta0);
    g1 -= (df + 1.0) * delta1 / (a1 + delta1 * delta1);
    h00 += (df + 1.0) * (a0 - delta0 * delta0)
        / ((a0 + delta0 * delta0) * (a0 + delta0 * delta0));
    h11 += (df + 1.0) * (a1 - delta1 * delta1)
        / ((a1 + delta1 * delta1) * (a1 + delta1 * delta1));
}

ssize_t parameter_row(ssize_t rows, ssize_t batch, ssize_t index, const char *name) {
    if (rows == 1) {
        return 0;
    }
    if (rows == batch) {
        return index;
    }
    throw std::invalid_argument(std::string(name) + " must have one row or batch rows");
}

}  // namespace

py::array_t<double> batch_count_loglik(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<double, py::array::c_style | py::array::forcecast> levels_array,
    py::array_t<long long, py::array::c_style | py::array::forcecast> totals_array,
    py::array_t<long long, py::array::c_style | py::array::forcecast> cases_array
) {
    if (points.ndim() != 2 || points.shape(1) != 2) {
        throw std::invalid_argument("points must have shape (n_points, 2)");
    }
    if (levels_array.ndim() != 1 || totals_array.ndim() != 2 || cases_array.ndim() != 2) {
        throw std::invalid_argument("levels must be 1D and count arrays must be 2D");
    }
    if (totals_array.shape(0) != cases_array.shape(0)
        || totals_array.shape(1) != cases_array.shape(1)
        || totals_array.shape(1) != levels_array.shape(0)) {
        throw std::invalid_argument("incompatible levels/totals/cases shapes");
    }
    const ssize_t batch = totals_array.shape(0);
    const ssize_t n_points = points.shape(0);
    py::array_t<double> output({batch, n_points});
    const auto point = points.unchecked<2>();
    const auto levels = levels_array.unchecked<1>();
    const auto totals = totals_array.unchecked<2>();
    const auto cases = cases_array.unchecked<2>();
    auto result = output.mutable_unchecked<2>();
    {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static) if(batch > 1)
    for (ssize_t row = 0; row < batch; ++row) {
        for (ssize_t i = 0; i < n_points; ++i) {
            double value = 0.0;
            for (ssize_t j = 0; j < levels.shape(0); ++j) {
                const double eta = point(i, 0) + point(i, 1) * levels(j);
                value += cases(row, j) * eta - totals(row, j) * softplus(eta);
            }
            result(row, i) = value;
        }
    }
    }
    return output;
}

py::array_t<double> batch_count_logposterior(
    py::array_t<double, py::array::c_style | py::array::forcecast> points_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> levels_array,
    py::array_t<long long, py::array::c_style | py::array::forcecast> totals_array,
    py::array_t<long long, py::array::c_style | py::array::forcecast> cases_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> locations_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> scales_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> dfs_array
) {
    if (points_array.ndim() != 3 || points_array.shape(2) != 2) {
        throw std::invalid_argument("points must have shape (batch, n_points, 2)");
    }
    if (levels_array.ndim() != 1 || totals_array.ndim() != 2 || cases_array.ndim() != 2) {
        throw std::invalid_argument("levels must be 1D and count arrays must be 2D");
    }
    const ssize_t batch = points_array.shape(0);
    const ssize_t n_points = points_array.shape(1);
    const ssize_t cells = levels_array.shape(0);
    if (totals_array.shape(0) != batch || cases_array.shape(0) != batch
        || totals_array.shape(1) != cells || cases_array.shape(1) != cells) {
        throw std::invalid_argument("incompatible points/levels/count shapes");
    }
    if (locations_array.ndim() != 2 || locations_array.shape(1) != 2
        || scales_array.ndim() != 2 || scales_array.shape(1) != 2
        || dfs_array.ndim() != 1) {
        throw std::invalid_argument("invalid prior parameter shapes");
    }
    // Validate broadcast rows before releasing the GIL/entering OpenMP.
    parameter_row(locations_array.shape(0), batch, 0, "locations");
    parameter_row(scales_array.shape(0), batch, 0, "scales");
    parameter_row(dfs_array.shape(0), batch, 0, "dfs");
    py::array_t<double> output({batch, n_points});
    const auto points = points_array.unchecked<3>();
    const auto levels = levels_array.unchecked<1>();
    const auto totals = totals_array.unchecked<2>();
    const auto cases = cases_array.unchecked<2>();
    const auto locations = locations_array.unchecked<2>();
    const auto scales = scales_array.unchecked<2>();
    const auto dfs = dfs_array.unchecked<1>();
    auto result = output.mutable_unchecked<2>();
    {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static) if(batch > 1)
    for (ssize_t row = 0; row < batch; ++row) {
        const ssize_t prior_row = locations.shape(0) == 1 ? 0 : row;
        const ssize_t scale_row = scales.shape(0) == 1 ? 0 : row;
        const ssize_t df_row = dfs.shape(0) == 1 ? 0 : row;
        const double df = dfs(df_row);
        const double constant =
            std::lgamma(0.5 * (df + 1.0))
            - std::lgamma(0.5 * df)
            - 0.5 * std::log(df * 3.14159265358979323846);
        for (ssize_t i = 0; i < n_points; ++i) {
            const double beta0 = points(row, i, 0);
            const double beta1 = points(row, i, 1);
            double value = 0.0;
            for (ssize_t j = 0; j < cells; ++j) {
                const double eta = beta0 + beta1 * levels(j);
                value += cases(row, j) * eta - totals(row, j) * softplus(eta);
            }
            const double z0 = (beta0 - locations(prior_row, 0)) / scales(scale_row, 0);
            const double z1 = (beta1 - locations(prior_row, 1)) / scales(scale_row, 1);
            value += 2.0 * constant;
            value -= std::log(scales(scale_row, 0)) + std::log(scales(scale_row, 1));
            value -= 0.5 * (df + 1.0) * std::log1p(z0 * z0 / df);
            value -= 0.5 * (df + 1.0) * std::log1p(z1 * z1 / df);
            result(row, i) = value;
        }
    }
    }
    return output;
}

py::dict batch_newton_modes(
    py::array_t<double, py::array::c_style | py::array::forcecast> levels_array,
    py::array_t<long long, py::array::c_style | py::array::forcecast> totals_array,
    py::array_t<long long, py::array::c_style | py::array::forcecast> cases_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> locations_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> scales_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> dfs_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> starts_array,
    int max_iterations,
    double tolerance
) {
    if (levels_array.ndim() != 1 || totals_array.ndim() != 2 || cases_array.ndim() != 2) {
        throw std::invalid_argument("levels must be 1D and count arrays must be 2D");
    }
    const ssize_t batch = totals_array.shape(0);
    const ssize_t cells = totals_array.shape(1);
    if (cases_array.shape(0) != batch || cases_array.shape(1) != cells
        || levels_array.shape(0) != cells) {
        throw std::invalid_argument("incompatible levels/totals/cases shapes");
    }
    if (locations_array.ndim() != 2 || locations_array.shape(1) != 2
        || scales_array.ndim() != 2 || scales_array.shape(1) != 2
        || dfs_array.ndim() != 1
        || starts_array.ndim() != 2 || starts_array.shape(0) != batch
        || starts_array.shape(1) != 2) {
        throw std::invalid_argument("invalid prior/start shapes");
    }
    py::array_t<double> modes({batch, static_cast<ssize_t>(2)});
    py::array_t<double> negative_hessians({batch, static_cast<ssize_t>(2), static_cast<ssize_t>(2)});
    py::array_t<bool> converged(batch);
    py::array_t<int> iterations(batch);
    const auto levels = levels_array.unchecked<1>();
    const auto totals = totals_array.unchecked<2>();
    const auto cases = cases_array.unchecked<2>();
    const auto locations = locations_array.unchecked<2>();
    const auto scales = scales_array.unchecked<2>();
    const auto dfs = dfs_array.unchecked<1>();
    const auto starts = starts_array.unchecked<2>();
    auto mode_output = modes.mutable_unchecked<2>();
    auto hessian_output = negative_hessians.mutable_unchecked<3>();
    auto converged_output = converged.mutable_unchecked<1>();
    auto iteration_output = iterations.mutable_unchecked<1>();
    {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 16) if(batch > 1)
    for (ssize_t row = 0; row < batch; ++row) {
        const ssize_t prior_row = parameter_row(locations.shape(0), batch, row, "locations");
        const ssize_t scale_row = parameter_row(scales.shape(0), batch, row, "scales");
        const ssize_t df_row = parameter_row(dfs.shape(0), batch, row, "dfs");
        const double loc0 = locations(prior_row, 0);
        const double loc1 = locations(prior_row, 1);
        const double scale0 = scales(scale_row, 0);
        const double scale1 = scales(scale_row, 1);
        const double df = dfs(df_row);
        double beta0 = starts(row, 0);
        double beta1 = starts(row, 1);
        bool ok = false;
        int used = 0;
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            used = iteration + 1;
            double g0, g1, h00, h01, h11;
            derivatives(
                beta0, beta1, row, levels, totals, cases,
                loc0, loc1, scale0, scale1, df,
                g0, g1, h00, h01, h11
            );
            if (std::max(std::abs(g0), std::abs(g1)) < tolerance) {
                ok = true;
                break;
            }
            double ridge = 0.0;
            double determinant = h00 * h11 - h01 * h01;
            for (int attempt = 0; attempt < 16
                 && (h00 + ridge <= 0.0 || h11 + ridge <= 0.0
                     || determinant <= 1e-14); ++attempt) {
                ridge = ridge == 0.0 ? 1e-8 : ridge * 10.0;
                determinant = (h00 + ridge) * (h11 + ridge) - h01 * h01;
            }
            const double a = h00 + ridge;
            const double d = h11 + ridge;
            determinant = a * d - h01 * h01;
            if (!(determinant > 0.0) || !std::isfinite(determinant)) {
                break;
            }
            double step0 = (d * g0 - h01 * g1) / determinant;
            double step1 = (-h01 * g0 + a * g1) / determinant;
            const double step_norm = std::hypot(step0, step1);
            if (step_norm > 10.0) {
                step0 *= 10.0 / step_norm;
                step1 *= 10.0 / step_norm;
            }
            const double current = log_posterior_unnormalized(
                beta0, beta1, row, levels, totals, cases,
                loc0, loc1, scale0, scale1, df
            );
            double multiplier = 1.0;
            bool accepted = false;
            for (int line = 0; line < 30; ++line) {
                const double candidate0 = beta0 + multiplier * step0;
                const double candidate1 = beta1 + multiplier * step1;
                const double candidate = log_posterior_unnormalized(
                    candidate0, candidate1, row, levels, totals, cases,
                    loc0, loc1, scale0, scale1, df
                );
                if (std::isfinite(candidate) && candidate >= current) {
                    beta0 = candidate0;
                    beta1 = candidate1;
                    accepted = true;
                    if (multiplier * std::max(std::abs(step0), std::abs(step1))
                        < tolerance * (1.0 + std::max(std::abs(beta0), std::abs(beta1)))) {
                        ok = true;
                    }
                    break;
                }
                multiplier *= 0.5;
            }
            if (!accepted || ok) {
                break;
            }
        }
        double g0, g1, h00, h01, h11;
        derivatives(
            beta0, beta1, row, levels, totals, cases,
            loc0, loc1, scale0, scale1, df,
            g0, g1, h00, h01, h11
        );
        if (std::max(std::abs(g0), std::abs(g1)) < 1e-7) {
            ok = true;
        }
        mode_output(row, 0) = beta0;
        mode_output(row, 1) = beta1;
        hessian_output(row, 0, 0) = h00;
        hessian_output(row, 0, 1) = h01;
        hessian_output(row, 1, 0) = h01;
        hessian_output(row, 1, 1) = h11;
        converged_output(row) = ok;
        iteration_output(row) = used;
    }
    }
    py::dict output;
    output["modes"] = std::move(modes);
    output["negative_hessians"] = std::move(negative_hessians);
    output["converged"] = std::move(converged);
    output["iterations"] = std::move(iterations);
    return output;
}

PYBIND11_MODULE(_lupus_fast, module) {
    module.doc() = "Low-footprint OpenMP kernels for the lupus logistic fast engine";
    module.def("batch_count_loglik", &batch_count_loglik);
    module.def("batch_count_logposterior", &batch_count_logposterior);
    module.def("batch_newton_modes", &batch_newton_modes);
    module.def("backend_info", []() {
        py::dict info;
#ifdef _OPENMP
        info["openmp"] = true;
        info["max_threads"] = omp_get_max_threads();
#else
        info["openmp"] = false;
        info["max_threads"] = 1;
#endif
        info["cpp_standard"] = 17;
        return info;
    });
}
