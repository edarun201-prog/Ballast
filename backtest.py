"""Walk-forward backtest: re-optimise on a trailing window, hold out of sample.

The optimiser in portfolio.py is fitted on the whole history, so the weights it
returns have already seen every week they would be judged on.  Any performance
figure taken from that is circular.  This module answers the honest question
instead: if you had only ever used data available on the day, and re-solved as
you went, what would you have held and how would it have done?
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pathlib import Path

import portfolio as pf

NAMES = {"0050.TW": "0050 台灣50"}

# Resolved against this file, never the process cwd: Streamlit is launched from
# wherever the user happens to be, and a benchmark that silently fails to load
# just quietly drops a row from the comparison table.
BENCH_PATH = Path(__file__).resolve().parent / "benchmarks.csv"

LOOKBACK = 156  # weeks of history each solve is allowed to see (~3 years)
STEP = 4        # weeks held before re-solving (~monthly)


@dataclass
class Track:
    name: str
    returns: pd.Series          # out-of-sample weekly returns
    weights: pd.DataFrame | None = None   # weights held, per rebalance

    @property
    def curve(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


def summarise(track: Track, risk_free: float = 0.015) -> dict:
    r = track.returns
    years = len(r) / pf.PERIODS_PER_YEAR
    total = float((1.0 + r).prod())
    cagr = total ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    vol = float(r.std() * np.sqrt(pf.PERIODS_PER_YEAR))
    row = {
        "累積報酬": total - 1.0,
        "年化報酬": cagr,
        "年化波動": vol,
        "Sharpe": pf.sharpe(cagr, vol, risk_free),
        "最大回撤": pf.max_drawdown(r),
    }
    if track.weights is not None and len(track.weights) > 1:
        # One-way turnover per rebalance: half the L1 move between weight sets.
        moves = track.weights.diff().abs().sum(axis=1) / 2.0
        row["每次換手"] = float(moves.iloc[1:].mean())
    return row


def walk_forward(
    prices: pd.DataFrame,
    cap: float,
    target: float | None = None,
    lookback: int = LOOKBACK,
    step: int = STEP,
) -> Track:
    """Solve on weeks [i-lookback, i), hold weeks [i, i+step). Never overlaps."""
    weekly = pf.weekly_returns(prices)
    chunks, held_weights, stamps = [], [], []

    for i in range(lookback, len(weekly), step):
        window = weekly.iloc[i - lookback : i]      # strictly before the hold
        future = weekly.iloc[i : i + step]
        if future.empty:
            break

        mu, sigma = pf.annualise(window)
        wanted = target
        if wanted is not None:
            # The window's feasible band moves around; a fixed target would
            # simply be infeasible in some of them.  Clamp rather than skip,
            # so the track has no survivorship gaps.
            floor_ = pf.solve(mu, sigma, cap)
            if floor_ is None:
                continue
            wanted = min(max(target, floor_.ret), pf.max_feasible_return(mu, cap))

        chosen = pf.solve(mu, sigma, cap, target=wanted)
        if chosen is None or not chosen.ok:
            continue

        chunks.append(pf.portfolio_returns(chosen.weights, future))
        held_weights.append(chosen.weights)
        stamps.append(future.index[0])

    if not chunks:
        return Track("Ballast", pd.Series(dtype=float))
    return Track(
        "Ballast walk-forward",
        pd.concat(chunks).sort_index(),
        pd.DataFrame(held_weights, index=stamps),
    )


def equal_weight(prices: pd.DataFrame, lookback: int = LOOKBACK) -> Track:
    """1/n over the same out-of-sample stretch, rebalanced weekly."""
    weekly = pf.weekly_returns(prices).iloc[lookback:]
    n = weekly.shape[1]
    w = pd.Series(1.0 / n, index=weekly.columns)
    return Track("等權重", pf.portfolio_returns(w, weekly))


def buy_and_hold(prices: pd.DataFrame, column: str, lookback: int = LOOKBACK) -> Track:
    weekly = pf.weekly_returns(prices).iloc[lookback:]
    return Track(pf.DISPLAY_NAMES.get(column, column), weekly[column])


def load_benchmarks(path: str | Path = BENCH_PATH) -> pd.DataFrame | None:
    """External yardsticks, if fetch_data.py has written them."""
    file = Path(path)
    if not file.exists():
        return None
    return pd.read_csv(file, index_col=0, parse_dates=True).sort_index()


def benchmark_tracks(
    prices: pd.DataFrame, lookback: int = LOOKBACK, path: str | Path = BENCH_PATH
) -> list[Track]:
    """One buy-and-hold track per benchmark, over the same out-of-sample weeks."""
    bench = load_benchmarks(path)
    if bench is None or bench.empty:
        return []
    aligned = bench.reindex(prices.index).ffill().dropna(how="any")
    weekly = pf.weekly_returns(aligned).iloc[lookback:]
    return [
        Track(NAMES.get(col, col), weekly[col]) for col in weekly.columns
    ]


def compare(
    prices: pd.DataFrame,
    cap: float = 0.30,
    target: float | None = None,
    benchmarks: tuple[str, ...] = ("SPY", "2330"),
    **kw,
) -> tuple[list[Track], pd.DataFrame]:
    lookback = kw.get("lookback", LOOKBACK)
    tracks = [walk_forward(prices, cap, target, **kw), equal_weight(prices, lookback)]
    tracks += benchmark_tracks(prices, lookback)
    tracks += [buy_and_hold(prices, c, lookback) for c in benchmarks if c in prices.columns]
    table = pd.DataFrame(
        {t.name: summarise(t) for t in tracks}
    ).T
    return tracks, table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prices", default="prices.csv")
    ap.add_argument("--cap", type=float, default=0.30)
    ap.add_argument("--target", type=float, default=None,
                    help="Annual return floor, e.g. 0.12. Omit for min-variance.")
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--step", type=int, default=STEP)
    args = ap.parse_args(argv)

    prices = pf.load_prices(args.prices)
    tracks, table = compare(
        prices, args.cap, args.target, lookback=args.lookback, step=args.step
    )
    first = tracks[0]
    print(f"樣本外 {first.returns.index.min():%Y-%m-%d} ~ "
          f"{first.returns.index.max():%Y-%m-%d}  "
          f"{len(first.returns)} 週，{len(first.weights)} 次再平衡")
    print(f"每次求解只看前 {args.lookback} 週，持有 {args.step} 週\n")
    fmt = {c: "{:.2%}" for c in table.columns if c != "Sharpe"}
    fmt["Sharpe"] = "{:.2f}"
    print(table.to_string(formatters={c: f.format for c, f in fmt.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
