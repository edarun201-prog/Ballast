"""Weekly returns and the minimum-variance QP behind the app.

Kept free of Streamlit so it can be imported and checked on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

# Weekly bars -> 52 periods a year.  See weekly_returns() for why not 252.
PERIODS_PER_YEAR = 52

DISPLAY_NAMES = {
    "2330": "2330 台積電",
    "2412": "2412 中華電",
    "1216": "1216 統一",
    "2881": "2881 富邦金",
    "2884": "2884 玉山金",
    "SPY": "SPY 美股大盤",
    "TLT": "TLT 美債 20Y+",
    "GLD": "GLD 黃金",
}


def load_prices(path: str = "prices.csv") -> pd.DataFrame:
    px = pd.read_csv(path, index_col=0, parse_dates=True)
    return px.sort_index()


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample to Friday closes *before* differencing.

    Taipei closes about thirteen hours ahead of New York, so a daily return
    series lines up Taiwan's Tuesday with New York's Monday.  The mismatch
    shows up as a badly understated cross-market correlation -- which would
    make the optimiser think the two blocks diversify each other far better
    than they do.  Weekly bars put both markets inside the same window.
    """
    weekly = prices.resample("W-FRI").last().dropna(how="any")
    return weekly.pct_change().dropna(how="any")


def annualise(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mean vector and covariance matrix."""
    mu = returns.mean() * PERIODS_PER_YEAR
    sigma = returns.cov() * PERIODS_PER_YEAR
    return mu, sigma


def cap_floor(n: int) -> float:
    """Smallest per-asset cap that can still hold n weights summing to 1.

    Below 1/n the cap and the budget constraint contradict each other and the
    problem is infeasible for *every* target return -- so the UI must never
    offer a cap under this.
    """
    return 1.0 / n


def greedy_weights(mu: pd.Series, cap: float) -> pd.Series:
    """The highest-return feasible portfolio: fill the best names up to the cap."""
    weights = pd.Series(0.0, index=mu.index)
    remaining = 1.0
    for name in mu.sort_values(ascending=False, kind="stable").index:
        take = min(cap, remaining)
        weights[name] = take
        remaining -= take
        if remaining <= 1e-12:
            break
    return weights


def max_feasible_return(mu: pd.Series, cap: float) -> float:
    """Best annual return reachable under sum(w)=1, 0<=w<=cap."""
    w = greedy_weights(mu, cap)
    return float(mu.to_numpy() @ w.to_numpy())


def _top_is_unique(mu: pd.Series, cap: float) -> bool:
    """False when names tie across the greedy cut-off.

    With a tie the maximum-return portfolio is not unique and the tied names
    should be mixed by variance -- work only the solver can do.
    """
    held = greedy_weights(mu, cap) > 0
    if held.all():
        return True
    worst_held = mu[held].min()
    return bool(mu[~held].max() < worst_held - 1e-12)


@dataclass
class Portfolio:
    weights: pd.Series
    ret: float
    vol: float
    status: str

    @property
    def ok(self) -> bool:
        return self.status in ("optimal", "optimal_inaccurate")


def solve(
    mu: pd.Series,
    sigma: pd.DataFrame,
    cap: float,
    target: float | None = None,
) -> Portfolio | None:
    """Minimise w'Sw subject to the four constraints.

        sum(w) == 1          fully invested
        w >= 0               no shorting
        w <= cap             single-name limit
        mu'w >= target       return floor (skipped when target is None)
    """
    n = len(mu)
    floor = cap_floor(n)
    if cap < floor - 1e-12:
        return None  # cap and budget contradict; nothing to solve

    if cap <= floor + 1e-9:
        # The feasible set has collapsed to the single point w = 1/n.  Handing
        # a degenerate problem to the solver only earns an "inaccurate" warning
        # for an answer that is already known exactly.
        weights = pd.Series(np.full(n, floor), index=mu.index)
        if target is not None and float(mu.to_numpy() @ weights.to_numpy()) < target - 1e-9:
            return None
        return _measure(weights, mu, sigma, "optimal")

    if target is not None and _top_is_unique(mu, cap):
        top = greedy_weights(mu, cap)
        top_ret = float(mu.to_numpy() @ top.to_numpy())
        if target > top_ret + 1e-9:
            return None                      # unreachable, whatever the variance
        if target >= top_ret - 1e-9:
            # The return floor pins the solution to a single vertex.  Handing
            # that to the solver earns an "inaccurate" warning for an answer
            # already known exactly.
            return _measure(top, mu, sigma, "optimal")

    w = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == 1, w <= cap]
    if target is not None:
        constraints.append(mu.to_numpy() @ w >= target)

    # psd_wrap: a sample covariance is PSD in theory but can pick up a tiny
    # negative eigenvalue numerically, which cvxpy would reject outright.
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma.to_numpy())))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
    except cp.error.SolverError:
        problem.solve(solver=cp.SCS)

    if w.value is None:
        return None

    weights = pd.Series(np.clip(w.value, 0.0, None), index=mu.index)
    return _measure(weights / weights.sum(), mu, sigma, problem.status)


def _measure(
    weights: pd.Series, mu: pd.Series, sigma: pd.DataFrame, status: str
) -> Portfolio:
    variance = float(weights.to_numpy() @ sigma.to_numpy() @ weights.to_numpy())
    return Portfolio(
        weights=weights,
        ret=float(mu.to_numpy() @ weights.to_numpy()),
        vol=float(np.sqrt(max(variance, 0.0))),
        status=status,
    )


def efficient_frontier(
    mu: pd.Series,
    sigma: pd.DataFrame,
    cap: float,
    n_points: int = 40,
) -> pd.DataFrame:
    """Sweep the return floor from the global-min-variance point to the top."""
    gmv = solve(mu, sigma, cap)
    if gmv is None:
        return pd.DataFrame(columns=["ret", "vol"])

    lo = gmv.ret
    hi = max_feasible_return(mu, cap)
    if hi - lo < 1e-9:
        # cap == 1/n: equal weights is the only feasible portfolio.
        return pd.DataFrame([{"ret": gmv.ret, "vol": gmv.vol}])

    rows = []
    for target in np.linspace(lo, hi, n_points):
        p = solve(mu, sigma, cap, target=target)
        if p is not None and p.ok:
            rows.append({"ret": p.ret, "vol": p.vol})
    return pd.DataFrame(rows)


# --- diagnostics ----------------------------------------------------------
# Everything below describes a portfolio that has already been chosen; none of
# it feeds back into the optimisation.


def risk_contributions(weights: pd.Series, sigma: pd.DataFrame) -> pd.Series:
    """Each holding's share of total portfolio variance, summing to 1.

    RC_i = w_i * (Sigma w)_i / (w' Sigma w).  Worth showing next to the weights:
    a minimum-variance portfolio routinely puts a big weight on a quiet asset
    that then carries almost none of the risk, and the two bars make that
    visible in a way the weights alone never do.
    """
    w = weights.to_numpy()
    marginal = sigma.to_numpy() @ w
    variance = float(w @ marginal)
    if variance <= 0:
        return pd.Series(0.0, index=weights.index)
    return pd.Series(w * marginal / variance, index=weights.index)


def diversification_ratio(weights: pd.Series, sigma: pd.DataFrame) -> float:
    """Weighted average volatility divided by portfolio volatility.

    1.0 means the holdings move as one and nothing was gained by splitting the
    money up; higher means the correlations are doing work.
    """
    vols = np.sqrt(np.diag(sigma.to_numpy()))
    w = weights.to_numpy()
    port_vol = float(np.sqrt(max(w @ sigma.to_numpy() @ w, 0.0)))
    if port_vol <= 0:
        return float("nan")
    return float((w @ vols) / port_vol)


def effective_holdings(weights: pd.Series) -> float:
    """1 / sum(w^2): how many equally-sized positions this behaves like.

    Eight names with one at 90% is not an eight-name portfolio, and this says so.
    """
    squared = float((weights.to_numpy() ** 2).sum())
    return float(1.0 / squared) if squared > 0 else float("nan")


def portfolio_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Weekly returns of holding these weights, rebalanced every week."""
    return returns[weights.index] @ weights.to_numpy()


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough fall of the cumulative series (a negative number)."""
    curve = (1.0 + returns).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def sharpe(ret: float, vol: float, risk_free: float) -> float:
    """Excess return per unit of volatility."""
    return float("nan") if vol <= 0 else (ret - risk_free) / vol
