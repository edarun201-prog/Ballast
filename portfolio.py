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


def max_feasible_return(mu: pd.Series, cap: float) -> float:
    """Best annual return reachable under sum(w)=1, 0<=w<=cap.

    Greedy: pour the budget into the highest-mu names until the cap stops you.
    """
    remaining = 1.0
    total = 0.0
    for r in sorted(mu.to_numpy(), reverse=True):
        take = min(cap, remaining)
        total += take * r
        remaining -= take
        if remaining <= 1e-12:
            break
    return float(total)


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
    if cap < cap_floor(n) - 1e-12:
        return None  # cap and budget contradict; nothing to solve

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
    weights = weights / weights.sum()
    variance = float(weights.to_numpy() @ sigma.to_numpy() @ weights.to_numpy())
    return Portfolio(
        weights=weights,
        ret=float(mu.to_numpy() @ weights.to_numpy()),
        vol=float(np.sqrt(max(variance, 0.0))),
        status=problem.status,
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
        # Back off the last hair so the top end does not fail on rounding.
        p = solve(mu, sigma, cap, target=min(target, hi - 1e-10))
        if p is not None and p.ok:
            rows.append({"ret": p.ret, "vol": p.vol})
    return pd.DataFrame(rows)
