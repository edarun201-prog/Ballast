"""Smoke checks for the optimiser. Run: python test_portfolio.py"""

from __future__ import annotations

import numpy as np
import pandas as pd

import portfolio as pf


def fake_prices(n_assets=8, n_days=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    drift = np.linspace(0.0002, 0.0007, n_assets)
    steps = rng.normal(drift, 0.012, size=(n_days, n_assets))
    px = 100 * np.exp(np.cumsum(steps, axis=0))
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(px, index=idx, columns=cols)


def main() -> int:
    px = fake_prices()
    weekly = pf.weekly_returns(px)
    assert 0.18 < len(weekly) / len(px) < 0.24, "weekly bars should be ~1/5 of daily"
    print(f"ok  weekly resample: {len(px)} daily -> {len(weekly)} weekly")

    mu, sigma = pf.annualise(weekly)
    assert pf.PERIODS_PER_YEAR == 52
    print(f"ok  annualised on 52 periods; mu range {mu.min():.2%}..{mu.max():.2%}")

    n = len(mu)
    floor = pf.cap_floor(n)
    assert abs(floor - 1 / n) < 1e-12

    # Below 1/n every target is infeasible -- the reason the slider is clamped.
    assert pf.solve(mu, sigma, floor - 0.01) is None
    print(f"ok  cap below 1/n ({floor:.3f}) refused")

    # At exactly 1/n equal weights is the only feasible point.
    eq = pf.solve(mu, sigma, floor)
    assert eq is not None and np.allclose(eq.weights.to_numpy(), floor, atol=1e-4)
    print("ok  cap == 1/n gives equal weights")

    cap = 0.30
    gmv = pf.solve(mu, sigma, cap)
    assert gmv is not None and gmv.ok
    w = gmv.weights.to_numpy()
    assert abs(w.sum() - 1) < 1e-6, "budget"
    assert w.min() > -1e-8, "no shorting"
    assert w.max() <= cap + 1e-6, "cap"
    print(f"ok  gmv @ c={cap:.0%}: ret {gmv.ret:.2%} vol {gmv.vol:.2%}")

    hi = pf.max_feasible_return(mu, cap)
    target = gmv.ret + (hi - gmv.ret) * 0.5
    tilted = pf.solve(mu, sigma, cap, target=target)
    assert tilted is not None and tilted.ok
    assert tilted.ret >= target - 1e-6, "return floor honoured"
    assert tilted.vol >= gmv.vol - 1e-9, "tilting away from gmv cannot lower vol"
    print(f"ok  target {target:.2%} -> ret {tilted.ret:.2%} vol {tilted.vol:.2%}")

    assert pf.solve(mu, sigma, cap, target=hi + 0.05) is None
    print("ok  unreachable target refused")

    curve = pf.efficient_frontier(mu, sigma, cap, n_points=15)
    assert len(curve) >= 10
    assert curve["ret"].is_monotonic_increasing
    assert curve["vol"].is_monotonic_increasing
    print(f"ok  frontier: {len(curve)} points, both axes monotone")

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
