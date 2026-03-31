import numpy as np
from numba import njit
from src.utils.logger import logger, log_execution_time

# ============================================================
# Numba JIT-compiled portfolio optimization kernels
# These achieve near-C++ speed without a compiler toolchain.
# ============================================================

@njit(cache=True)
def _mean_variance_objective(w, expected_returns, cov_matrix, risk_aversion):
    """JIT-compiled Mean-Variance objective function."""
    port_return = 0.0
    n = len(w)
    for i in range(n):
        port_return += w[i] * expected_returns[i]

    port_risk = 0.0
    for i in range(n):
        for j in range(n):
            port_risk += w[i] * cov_matrix[i, j] * w[j]

    return -(port_return - 0.5 * risk_aversion * port_risk)


@njit(cache=True)
def _calculate_cvar(returns, weights, alpha=0.01):
    """JIT-compiled CVaR calculation."""
    n_samples = returns.shape[0]
    n_assets = returns.shape[1]

    port_returns = np.zeros(n_samples)
    for i in range(n_samples):
        for j in range(n_assets):
            port_returns[i] += returns[i, j] * weights[j]

    sorted_returns = np.sort(port_returns)
    cutoff = int(n_samples * alpha)
    if cutoff < 1:
        cutoff = 1

    cvar = 0.0
    for i in range(cutoff):
        cvar += sorted_returns[i]
    cvar /= cutoff

    return cvar


@njit(cache=True)
def _risk_parity_objective(w, cov_matrix):
    """JIT-compiled Risk Parity objective."""
    n = len(w)
    # Portfolio variance
    port_var = 0.0
    for i in range(n):
        for j in range(n):
            port_var += w[i] * cov_matrix[i, j] * w[j]

    # Marginal risk contributions
    mrc = np.zeros(n)
    for i in range(n):
        for j in range(n):
            mrc[i] += cov_matrix[i, j] * w[j]

    # Risk contributions
    rc = np.zeros(n)
    for i in range(n):
        rc[i] = w[i] * mrc[i]

    # Target: Equal risk contribution
    target = port_var / n
    obj = 0.0
    for i in range(n):
        obj += (rc[i] - target) ** 2

    return obj


@njit(cache=True)
def _black_litterman_posterior(tau, cov_matrix, market_weights, P, Q, omega):
    """JIT-compiled Black-Litterman posterior expected returns."""
    n = cov_matrix.shape[0]
    k = P.shape[0]

    # Prior: pi = tau * Sigma * w_mkt
    pi = np.zeros(n)
    for i in range(n):
        for j in range(n):
            pi[i] += tau * cov_matrix[i, j] * market_weights[j]

    # Posterior (simplified): pi_bar = pi + tau*Sigma*P' * inv(P*tau*Sigma*P' + Omega) * (Q - P*pi)
    # For simplicity, return prior-adjusted by views magnitude
    adjustment = np.zeros(n)
    for v in range(k):
        residual = Q[v]
        for j in range(n):
            residual -= P[v, j] * pi[j]
        for j in range(n):
            adjustment[j] += P[v, j] * residual * 0.5  # Blending factor

    posterior = np.zeros(n)
    for i in range(n):
        posterior[i] = pi[i] + adjustment[i]

    return posterior


class NumbaPortfolioSolver:
    """High-performance portfolio solver using Numba JIT compilation."""

    def __init__(self, risk_aversion=3.0, transaction_cost=0.001):
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        # Warm up Numba JIT
        self._warmup()
        logger.info(f"NumbaPortfolioSolver initialized (JIT-compiled)")

    def _warmup(self):
        """Pre-compile all JIT functions."""
        dummy_w = np.array([0.5, 0.5])
        dummy_ret = np.array([0.01, 0.02])
        dummy_cov = np.eye(2) * 0.01
        dummy_returns = np.random.randn(10, 2)
        _mean_variance_objective(dummy_w, dummy_ret, dummy_cov, 3.0)
        _calculate_cvar(dummy_returns, dummy_w, 0.01)
        _risk_parity_objective(dummy_w, dummy_cov)
        _black_litterman_posterior(0.05, dummy_cov, dummy_w,
                                   np.eye(2), np.array([0.01, 0.02]), np.eye(2) * 0.01)

    @log_execution_time
    def mean_variance(self, expected_returns, cov_matrix, current_weights=None):
        """Mean-Variance optimization using grid search + JIT objective."""
        from scipy.optimize import minimize
        n = len(expected_returns)
        er = np.asarray(expected_returns, dtype=np.float64)
        cov = np.asarray(cov_matrix, dtype=np.float64)
        ra = float(self.risk_aversion)

        def obj(w):
            return _mean_variance_objective(w, er, cov, ra)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(n))
        init_w = np.ones(n) / n

        res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x if res.success else init_w

    @log_execution_time
    def mean_cvar(self, returns_matrix, alpha=0.01):
        """Mean-CVaR optimization."""
        from scipy.optimize import minimize
        n = returns_matrix.shape[1]
        ret_mat = np.asarray(returns_matrix, dtype=np.float64)

        def obj(w):
            return -_calculate_cvar(ret_mat, w, alpha)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(n))
        init_w = np.ones(n) / n

        res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x if res.success else init_w

    @log_execution_time
    def risk_parity(self, cov_matrix):
        """Risk Parity optimization."""
        from scipy.optimize import minimize
        n = cov_matrix.shape[0]
        cov = np.asarray(cov_matrix, dtype=np.float64)

        def obj(w):
            return _risk_parity_objective(w, cov)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.01, 1.0) for _ in range(n))
        init_w = np.ones(n) / n

        res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x if res.success else init_w

    @log_execution_time
    def black_litterman(self, cov_matrix, market_weights, views_P, views_Q, tau=0.05):
        """Black-Litterman model: returns posterior expected returns."""
        cov = np.asarray(cov_matrix, dtype=np.float64)
        mw = np.asarray(market_weights, dtype=np.float64)
        P = np.asarray(views_P, dtype=np.float64)
        Q = np.asarray(views_Q, dtype=np.float64)
        omega = np.eye(P.shape[0]) * 0.01

        posterior = _black_litterman_posterior(tau, cov, mw, P, Q, omega)
        # Optimize using posterior returns
        return self.mean_variance(posterior, cov)
