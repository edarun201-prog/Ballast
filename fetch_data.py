"""Download the Ballast universe and write a single TWD-denominated price table.

Five Taiwan listings come from FinMind, three US ETFs and the USD/TWD rate come
from yfinance.  The US closes are multiplied by the same day's FX rate so every
column is in TWD, then the two calendars are inner-joined so only days on which
both markets traded survive.

    python fetch_data.py                      # 2018-01-01 .. today -> prices.csv
    python fetch_data.py --start 2015-01-01 --out prices.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import pandas as pd

# Taiwan listings (FinMind stock_id -> display name, kept for the UI).
TW_STOCKS = {
    "2330": "台積電",
    "2412": "中華電",
    "1216": "統一",
    "2881": "富邦金",
    "2884": "玉山金",
}
US_ETFS = ["SPY", "TLT", "GLD"]
FX_TICKER = "TWD=X"  # USD -> TWD
DEFAULT_START = "2018-01-01"
DEFAULT_OUT = "prices.csv"


def fetch_tw(start: str, end: str, token: str | None = None) -> pd.DataFrame:
    """Daily closes for the Taiwan names, one column per stock_id."""
    from FinMind.data import DataLoader

    loader = DataLoader()
    if token:
        loader.login_by_token(api_token=token)

    columns = {}
    for stock_id in TW_STOCKS:
        df = loader.taiwan_stock_daily(
            stock_id=stock_id, start_date=start, end_date=end
        )
        if df is None or df.empty:
            raise RuntimeError(
                f"FinMind returned nothing for {stock_id}. "
                "Without a token the free tier is rate-limited to a few hundred "
                "calls per hour; set FINMIND_TOKEN and retry."
            )
        s = df.assign(date=pd.to_datetime(df["date"])).set_index("date")["close"]
        s = s[~s.index.duplicated(keep="last")].astype(float)
        columns[stock_id] = s
        print(f"  {stock_id}: {len(s)} rows {s.index.min():%Y-%m-%d}..{s.index.max():%Y-%m-%d}")

    return pd.DataFrame(columns).sort_index()


def fetch_dividends(start: str, end: str, token: str | None = None) -> pd.DataFrame:
    """Cash and stock dividends per share, one row per ex-dividend date.

    Cash and stock distributions carry their own ex-dates in the source and are
    kept apart here, even though in practice they almost always fall together.
    """
    from FinMind.data import DataLoader

    loader = DataLoader()
    if token:
        loader.login_by_token(api_token=token)

    rows = []
    for stock_id in TW_STOCKS:
        df = loader.taiwan_stock_dividend(
            stock_id=stock_id, start_date=start, end_date=end
        )
        if df is None or df.empty:
            print(f"  {stock_id}: no dividend records")
            continue
        for _, r in df.iterrows():
            cash = float(r["CashEarningsDistribution"]) + float(r["CashStatutorySurplus"])
            stock = float(r["StockEarningsDistribution"]) + float(r["StockStatutorySurplus"])
            for ex_date, amount, kind in (
                (r["CashExDividendTradingDate"], cash, "cash"),
                (r["StockExDividendTradingDate"], stock, "stock"),
            ):
                if not ex_date or amount <= 0:
                    continue  # blank ex-date or a nil distribution
                rows.append(
                    {
                        "stock_id": stock_id,
                        "ex_date": pd.to_datetime(ex_date),
                        "kind": kind,
                        "amount": amount,
                    }
                )
        n = sum(1 for x in rows if x["stock_id"] == stock_id)
        print(f"  {stock_id}: {n} distributions")

    return pd.DataFrame(rows, columns=["stock_id", "ex_date", "kind", "amount"])


def adjust_for_dividends(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust the Taiwan columns so history is comparable with today.

    FinMind's taiwan_stock_daily is the raw traded price: on an ex-dividend day
    it simply drops by the distribution, which in a return series looks like a
    loss even though the holder received the cash.  yfinance already hands back
    adjusted closes for the US side, so leaving Taiwan raw would bias one half
    of the universe and not the other.

    On an ex-date the last cum-dividend close P becomes worth

        (P - cash) / (1 + stock / 10)

    to a holder -- cash leaves the price, and a stock dividend quoted in NTD of
    par splits each share into 1 + stock/10 of them.  Everything before that
    date is scaled by the resulting ratio, compounded back through time.
    """
    if events.empty:
        return prices

    out = prices.copy()
    for stock_id, group in events.groupby("stock_id"):
        if stock_id not in out.columns:
            continue
        factors = pd.Series(1.0, index=out.index)
        for _, ev in group.sort_values("ex_date").iterrows():
            before = out.index[out.index < ev["ex_date"]]
            if len(before) == 0:
                continue  # distribution predates the price history
            prev = before[-1]
            close = float(prices.loc[prev, stock_id])
            if close <= 0:
                continue
            if ev["kind"] == "cash":
                adjusted = close - ev["amount"]
            else:
                adjusted = close / (1.0 + ev["amount"] / 10.0)
            if adjusted <= 0:
                continue
            factors.loc[:prev] *= adjusted / close
        out[stock_id] = out[stock_id] * factors
    return out


def fetch_us(start: str, end: str) -> pd.DataFrame:
    """Adjusted closes for the ETFs plus the USD/TWD rate, in USD."""
    import yfinance as yf

    raw = yf.download(
        US_ETFS + [FX_TICKER],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned nothing; check the network.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    missing = [t for t in US_ETFS + [FX_TICKER] if t not in close.columns]
    if missing:
        raise RuntimeError(f"yfinance is missing columns: {missing}")

    for ticker in US_ETFS + [FX_TICKER]:
        s = close[ticker].dropna()
        print(f"  {ticker}: {len(s)} rows {s.index.min():%Y-%m-%d}..{s.index.max():%Y-%m-%d}")
    return close.sort_index()


def build_prices(tw: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    """Convert the US columns to TWD and align both calendars."""
    fx = us[FX_TICKER].ffill()  # FX quotes go quiet on US holidays
    usd_in_twd = us[US_ETFS].mul(fx, axis=0)

    # Inner join: a day only counts if both markets actually traded.
    prices = tw.join(usd_in_twd, how="inner")

    # FinMind writes 0.0 for suspended sessions; drop those rows outright.
    prices = prices.mask(prices <= 0).dropna(how="any")
    return prices.astype(float).sort_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--no-adjust",
        action="store_true",
        help="Keep Taiwan prices raw (unadjusted for dividends).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("FINMIND_TOKEN"),
        help="FinMind API token (defaults to $FINMIND_TOKEN).",
    )
    args = parser.parse_args(argv)

    print(f"Taiwan (FinMind) {args.start}..{args.end}")
    tw = fetch_tw(args.start, args.end, args.token)
    if not args.no_adjust:
        print("Dividends (FinMind)")
        events = fetch_dividends(args.start, args.end, args.token)
        tw = adjust_for_dividends(tw, events)

    print(f"US + FX (yfinance) {args.start}..{args.end}")
    us = fetch_us(args.start, args.end)

    prices = build_prices(tw, us)
    if prices.empty:
        raise RuntimeError("No overlapping trading days survived the join.")

    prices.index.name = "date"
    # Four decimals is far past any real quote; full float repr just makes
    # the file big and every diff unreadable.
    prices.to_csv(args.out, float_format="%.4f")
    print(
        f"\nwrote {args.out}: {len(prices)} rows x {prices.shape[1]} columns, "
        f"{prices.index.min():%Y-%m-%d}..{prices.index.max():%Y-%m-%d}"
    )
    dropped = len(tw) - len(prices)
    print(f"({dropped} Taiwan sessions dropped by the join / zero filter)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
