import numpy as np
import pandas as pd

def _pp_regression(y, regression):
    """
    Build regressors and run OLS of y_t on y_{t-1} (+ deterministic terms).
    regression: 'nc' (zero mean), 'c' (single mean), 'ct' (trend)
    Returns dict with rho_hat, se_rho, resid, sigma2, T, k, ylag (demeaned/detrended y_{t-1} info not needed separately)
    """
    y = np.asarray(y, dtype=float)
    y_dep = y[1:]
    y_lag = y[:-1]
    T = len(y_dep)

    if regression == 'nc':
        X = y_lag.reshape(-1, 1)
    elif regression == 'c':
        X = np.column_stack([np.ones(T), y_lag])
    elif regression == 'ct':
        trend = np.arange(1, T + 1, dtype=float)
        X = np.column_stack([np.ones(T), trend, y_lag])
    else:
        raise ValueError("regression must be 'nc', 'c', or 'ct'")

    k = X.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y_dep
    resid = y_dep - X @ beta
    ssr = resid @ resid
    sigma2 = ssr / (T - k)          # OLS residual variance
    rho_hat = beta[-1]              # coefficient on y_{t-1} is always last column
    se_rho = np.sqrt(sigma2 * XtX_inv[-1, -1])

    return dict(rho_hat=rho_hat, se_rho=se_rho, resid=resid, sigma2=sigma2, T=T, k=k)


def _long_run_var(resid, lag):
    """Newey-West / Bartlett-kernel long run variance estimate."""
    T = len(resid)
    gamma0 = np.sum(resid ** 2) / T
    lam2 = gamma0
    for j in range(1, lag + 1):
        gamma_j = np.sum(resid[j:] * resid[:-j]) / T
        w = 1 - j / (lag + 1)
        lam2 += 2 * w * gamma_j
    return gamma0, lam2


def pp_statistics(y, regression, lag):
    """Compute Phillips-Perron Z_rho and Z_tau statistics for one spec."""
    reg = _pp_regression(y, regression)
    rho_hat, se_rho, resid, sigma2, T = (
        reg['rho_hat'], reg['se_rho'], reg['resid'], reg['sigma2'], reg['T']
    )
    gamma0, lam2 = _long_run_var(resid, lag)

    t_rho = (rho_hat - 1) / se_rho

    Z_rho = T * (rho_hat - 1) - 0.5 * (lam2 - gamma0) * (T ** 2 * se_rho ** 2 / sigma2)

    Z_tau = np.sqrt(gamma0 / lam2) * t_rho - \
            (lam2 - gamma0) / (2 * np.sqrt(lam2) * np.sqrt(sigma2)) * (T * se_rho)

    return Z_rho, Z_tau


def _simulate_null_distribution(T, regression, lag, n_reps=2000, seed=None):
    """
    Simulate the null (unit-root random walk) distribution of Z_rho and Z_tau
    for a series of the SAME length / spec as the data, so p-values are
    finite-sample-adjusted rather than relying on fixed asymptotic tables.
    """
    rng = np.random.default_rng(seed)
    n_obs = T + 1  # need T+1 points to produce T (y_t, y_{t-1}) pairs after differencing
    rhos = np.empty(n_reps)
    taus = np.empty(n_reps)
    for b in range(n_reps):
        e = rng.standard_normal(n_obs)
        y_sim = np.cumsum(e)  # random walk under H0
        rhos[b], taus[b] = pp_statistics(y_sim, regression, lag)
    return rhos, taus


def _pvalue_from_dist(stat, sim_dist):
    """Left-tail empirical p-value: P(sim <= stat)."""
    return float(np.mean(sim_dist <= stat))


def phillips_perron_table(residuals, lags=(0, 1, 2), regressions=('nc', 'c', 'ct'),
                           n_reps=2000, seed=42):
    """
    Run PP test for all combinations of regression type x lag, returning a
    DataFrame with Rho, Pr<Rho, Tau, Pr<Tau -- mirroring SAS-style output.
    """
    y = np.asarray(residuals, dtype=float)
    T = len(y) - 1  # number of (y_t, y_{t-1}) observations used in regression

    label_map = {'nc': 'Zero Mean', 'c': 'Single Mean', 'ct': 'Trend'}
    rows = []
    # cache simulated null distributions per (regression, lag) so we don't
    # re-simulate identical specs
    cache = {}
    for reg in regressions:
        for lag in lags:
            Z_rho, Z_tau = pp_statistics(y, reg, lag)
            key = (reg, lag)
            if key not in cache:
                cache[key] = _simulate_null_distribution(
                    T, reg, lag, n_reps=n_reps, seed=seed + hash(key) % 1000
                )
            rho_dist, tau_dist = cache[key]
            pr_rho = _pvalue_from_dist(Z_rho, rho_dist)
            pr_tau = _pvalue_from_dist(Z_tau, tau_dist)
            rows.append({
                'Type': label_map[reg],
                'Lags': lag,
                'Rho': round(Z_rho, 4),
                'Pr<Rho': round(pr_rho, 4),
                'Tau': round(Z_tau, 4),
                'Pr<Tau': round(pr_tau, 4),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # quick sanity test: stationary AR(1) residuals -> should reject unit root (small p-values)
    n = 300
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = 0.5 * resid[t - 1] + rng.standard_normal()
    table = phillips_perron_table(resid, n_reps=1500)
    print(table.to_string(index=False))
