#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h> // NVIDIA-grade OpenMP support

namespace py = pybind11;

// ============================================================
// AMIE-APO C++ Portfolio Optimization Kernels
// High-performance solvers for production-grade trading systems
// ============================================================

// Mean-Variance objective: -(w'mu - lambda/2 * w'Sigma*w)
double mean_variance_objective(
    const std::vector<double>& weights,
    const std::vector<double>& expected_returns,
    const std::vector<std::vector<double>>& cov_matrix,
    double risk_aversion
) {
    int n = weights.size();
    double port_return = 0.0;
    for (int i = 0; i < n; i++) {
        port_return += weights[i] * expected_returns[i];
    }

    double port_risk = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            port_risk += weights[i] * cov_matrix[i][j] * weights[j];
        }
    }

    return -(port_return - 0.5 * risk_aversion * port_risk);
}

// CVaR calculation
double calculate_cvar(
    py::array_t<double> returns_array,
    const std::vector<double>& weights,
    double alpha
) {
    auto buf = returns_array.unchecked<2>();
    int n_samples = buf.shape(0);
    int n_assets = buf.shape(1);

    std::vector<double> port_returns(n_samples);
    for (int i = 0; i < n_samples; i++) {
        double r = 0.0;
        for (int j = 0; j < n_assets; j++) {
            r += buf(i, j) * weights[j];
        }
        port_returns[i] = r;
    }

    std::sort(port_returns.begin(), port_returns.end());
    int cutoff = std::max(1, (int)(n_samples * alpha));

    double cvar = 0.0;
    for (int i = 0; i < cutoff; i++) {
        cvar += port_returns[i];
    }
    return cvar / cutoff;
}

// Risk Parity objective
double risk_parity_objective(
    const std::vector<double>& weights,
    const std::vector<std::vector<double>>& cov_matrix
) {
    int n = weights.size();

    // Portfolio variance
    double port_var = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            port_var += weights[i] * cov_matrix[i][j] * weights[j];

    // Marginal risk contributions
    std::vector<double> mrc(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mrc[i] += cov_matrix[i][j] * weights[j];

    // Risk contributions
    double target = port_var / n;
    double obj = 0.0;
    for (int i = 0; i < n; i++) {
        double rc = weights[i] * mrc[i];
        obj += (rc - target) * (rc - target);
    }
    return obj;
}

// Black-Litterman posterior returns
std::vector<double> black_litterman_posterior(
    double tau,
    const std::vector<std::vector<double>>& cov_matrix,
    const std::vector<double>& market_weights,
    const std::vector<std::vector<double>>& P,
    const std::vector<double>& Q
) {
    int n = cov_matrix.size();
    int k = P.size();

    // Prior: pi = tau * Sigma * w_mkt
    std::vector<double> pi(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            pi[i] += tau * cov_matrix[i][j] * market_weights[j];

    // Simplified posterior adjustment
    std::vector<double> adjustment(n, 0.0);
    for (int v = 0; v < k; v++) {
        double residual = Q[v];
        for (int j = 0; j < n; j++)
            residual -= P[v][j] * pi[j];
        for (int j = 0; j < n; j++)
            adjustment[j] += P[v][j] * residual * 0.5;
    }

    std::vector<double> posterior(n);
    for (int i = 0; i < n; i++)
        posterior[i] = pi[i] + adjustment[i];

    return posterior;
}

inline double fast_rand(unsigned long& seed) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(seed >> 33)) / (double)(1ULL << 31) - 1.0;
}

// Monte Carlo VaR/CVaR engine (ultra-fast C++ paths)
py::dict monte_carlo_stress(
    double start_price, double mu, double vol,
    int n_paths, int n_steps
) {
    std::vector<double> final_prices(n_paths);
    double dt = 1.0 / 252.0;

    // Parallelize Monte Carlo paths using OpenMP
    #pragma omp parallel for
    for (int p = 0; p < n_paths; p++) {
        double price = start_price;
        // Optimization: Use thread-local seed for random generator
        unsigned long local_seed = 42 + p + omp_get_thread_num(); 

        for (int s = 0; s < n_steps; s++) {
            double z = fast_rand(local_seed);
            double ret = mu * dt + vol * std::sqrt(dt) * z;
            price *= std::exp(ret);
        }
        final_prices[p] = price;
    }

    // Compute returns
    std::vector<double> returns(n_paths);
    for (int i = 0; i < n_paths; i++)
        returns[i] = (final_prices[i] - start_price) / start_price;

    std::sort(returns.begin(), returns.end());
    int cutoff_99 = std::max(1, (int)(n_paths * 0.01));

    double var_99 = returns[cutoff_99];
    double cvar_99 = 0.0;
    for (int i = 0; i < cutoff_99; i++)
        cvar_99 += returns[i];
    cvar_99 /= cutoff_99;

    double mean_price = 0.0;
    for (auto& p : final_prices) mean_price += p;
    mean_price /= n_paths;

    py::dict result;
    result["var_99"] = var_99;
    result["cvar_99"] = cvar_99;
    result["mean_price"] = mean_price;
    result["n_paths"] = n_paths;
    return result;
}

PYBIND11_MODULE(cpp_solver, m) {
    m.doc() = "AMIE-APO C++ Portfolio Optimization Kernels";

    m.def("mean_variance_objective", &mean_variance_objective,
        "Computes Mean-Variance objective value",
        py::arg("weights"), py::arg("expected_returns"),
        py::arg("cov_matrix"), py::arg("risk_aversion"));

    m.def("calculate_cvar", &calculate_cvar,
        "Computes CVaR at given alpha",
        py::arg("returns"), py::arg("weights"), py::arg("alpha"));

    m.def("risk_parity_objective", &risk_parity_objective,
        "Computes Risk Parity objective",
        py::arg("weights"), py::arg("cov_matrix"));

    m.def("black_litterman_posterior", &black_litterman_posterior,
        "Computes Black-Litterman posterior returns",
        py::arg("tau"), py::arg("cov_matrix"), py::arg("market_weights"),
        py::arg("P"), py::arg("Q"));

    m.def("monte_carlo_stress", &monte_carlo_stress,
        "Ultra-fast Monte Carlo stress testing",
        py::arg("start_price"), py::arg("mu"), py::arg("vol"),
        py::arg("n_paths"), py::arg("n_steps"));
}
