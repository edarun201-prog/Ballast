"""Ballast - minimum-variance portfolio optimiser over a TW + US universe."""

from __future__ import annotations

import math
import os
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import portfolio as pf

PRICES = Path(__file__).parent / "prices.csv"

# Validated two-colour set (dataviz six-checks, light and dark).
SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
ACCENT = {"light": "#eb6834", "dark": "#d95926"}
INK = {"light": "#52514e", "dark": "#c3c2b7"}       # secondary text
SURFACE = {"light": "#ffffff", "dark": "#0e1117"}   # Streamlit page background


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def finmind_token() -> str | None:
    """Token from Streamlit secrets, falling back to the environment."""
    try:
        token = st.secrets.get("FINMIND_TOKEN")
        if token:
            return str(token)
    except Exception:
        pass  # no secrets.toml at all
    return os.environ.get("FINMIND_TOKEN")


def refresh_prices(start: str = "2019-01-01") -> str | None:
    """Rebuild prices.csv from the live sources; return an error, or None.

    The new table is written to a temporary file and only moved into place
    once it is complete.  A failed refresh therefore leaves the committed
    snapshot untouched -- this is a demo as much as a tool, and one bad
    network call must never be able to leave it showing an error page.
    """
    import fetch_data as fd

    tmp = PRICES.with_suffix(".csv.tmp")
    try:
        end = date.today().isoformat()
        prices = fd.build_prices(
            fd.fetch_tw(start, end, finmind_token()), fd.fetch_us(start, end)
        )
        if prices.empty:
            return "兩邊沒有重疊的交易日。"
        prices.index.name = "date"
        prices.to_csv(tmp, float_format="%.4f")
        tmp.replace(PRICES)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        tmp.unlink(missing_ok=True)


def data_fingerprint() -> tuple[float, int]:
    """Identify the current prices.csv, so a regenerated file busts the cache."""
    stat = PRICES.stat()
    return (stat.st_mtime, stat.st_size)


# The fingerprint is a real (hashed) argument, not an underscore-prefixed one:
# Streamlit leaves those out of the cache key, which is exactly the bug this
# guards against -- a zero-key cache never notices the file changing underneath
# a running app.
@st.cache_data(show_spinner=False)
def load(fingerprint: tuple[float, int]):
    prices = pf.load_prices(PRICES)
    weekly = pf.weekly_returns(prices)
    mu, sigma = pf.annualise(weekly)
    return prices, weekly, mu, sigma


@st.cache_data(show_spinner=False)
def frontier(mu: pd.Series, sigma: pd.DataFrame, cap: float) -> pd.DataFrame:
    return pf.efficient_frontier(mu, sigma, cap)


def weights_chart(weights: pd.Series, color: str, ink: str) -> alt.Chart:
    df = pd.DataFrame(
        {
            "資產": [pf.DISPLAY_NAMES.get(k, k) for k in weights.index],
            "權重": weights.to_numpy(),
        }
    ).sort_values("權重", ascending=False)

    base = alt.Chart(df).encode(
        y=alt.Y("資產:N", sort="-x", title=None),
        x=alt.X(
            "權重:Q",
            title="權重",
            axis=alt.Axis(format=".0%", grid=True, tickCount=5),
            scale=alt.Scale(domain=[0, max(0.05, float(df["權重"].max()) * 1.18)]),
        ),
        tooltip=[
            alt.Tooltip("資產:N"),
            alt.Tooltip("權重:Q", format=".2%"),
        ],
    )
    bars = base.mark_bar(color=color, cornerRadiusEnd=4, height=16)
    # Direct labels: eight rows is few enough to label every one.
    labels = base.mark_text(align="left", dx=6, fontSize=12, color=ink).encode(
        text=alt.Text("權重:Q", format=".1%")
    )
    return (bars + labels).properties(width="container", height=max(220, 34 * len(df)))


def frontier_chart(
    curve: pd.DataFrame, point: pf.Portfolio, color: str, accent: str,
    ink: str, surface: str,
):
    line = (
        alt.Chart(curve)
        .mark_line(color=color, strokeWidth=2)
        .encode(
            x=alt.X(
                "vol:Q",
                title="年化波動度",
                axis=alt.Axis(format=".0%", tickCount=6),
                scale=alt.Scale(zero=False, nice=True),
            ),
            y=alt.Y(
                "ret:Q",
                title="預期年化報酬",
                axis=alt.Axis(format=".0%", tickCount=6),
                scale=alt.Scale(zero=False, nice=True),
            ),
            tooltip=[
                alt.Tooltip("vol:Q", title="波動", format=".2%"),
                alt.Tooltip("ret:Q", title="報酬", format=".2%"),
            ],
        )
    )
    here = pd.DataFrame([{"vol": point.vol, "ret": point.ret, "標記": "目前組合"}])
    dot = (
        alt.Chart(here)
        .mark_point(
            size=110, filled=True, color=accent, stroke=surface, strokeWidth=2
        )
        .encode(
            x="vol:Q",
            y="ret:Q",
            tooltip=[
                alt.Tooltip("標記:N", title=""),
                alt.Tooltip("vol:Q", title="波動", format=".2%"),
                alt.Tooltip("ret:Q", title="報酬", format=".2%"),
            ],
        )
    )
    tag = (
        alt.Chart(here)
        .mark_text(align="left", dx=12, dy=-2, fontSize=12, color=ink)
        .encode(x="vol:Q", y="ret:Q", text="標記:N")
    )
    return (line + dot + tag).properties(width="container", height=380)


st.set_page_config(page_title="Ballast", page_icon="⚖️", layout="wide")
st.title("⚖️ Ballast")
st.caption(
    "台股五檔 + 美股三檔 ETF 的最小變異數投組。價格全部換算成台幣，"
    "報酬率以週為單位計算。"
)

if not PRICES.exists():
    st.error("找不到 `prices.csv`。先執行 `python fetch_data.py` 抓取資料。")
    st.stop()

prices, weekly, mu, sigma = load(data_fingerprint())

meta, action = st.columns([4, 1], vertical_alignment="bottom")
with meta:
    st.caption(
        f"資料 {prices.index.min():%Y-%m-%d} ~ {prices.index.max():%Y-%m-%d}　"
        f"{len(prices)} 個共同交易日、{len(weekly)} 週"
    )
with action:
    if st.button("更新資料", help="重新抓 FinMind 與 yfinance，失敗則保留現有資料。"):
        with st.spinner("抓取中…"):
            error = refresh_prices()
        if error:
            st.warning(f"更新失敗，沿用現有資料。{error}")
        else:
            st.rerun()

n = len(mu)
mode = theme()
color, accent = SERIES[mode], ACCENT[mode]
ink, surface = INK[mode], SURFACE[mode]

# --- controls -------------------------------------------------------------
# The cap floor is 1/n: below it the cap and sum(w)==1 contradict each other
# and every target return is infeasible, so the slider never goes there.
cap_min_pct = math.ceil(100.0 / n * 10) / 10

left, right = st.columns(2)
with left:
    cap_pct = st.slider(
        "單一標的上限 c",
        min_value=cap_min_pct,
        max_value=100.0,
        value=max(30.0, cap_min_pct),
        step=0.5,
        format="%.1f%%",
        help=f"下限 {cap_min_pct:.1f}% = 1/{n}。再低就湊不滿 100%，必定無解。",
    )
cap = cap_pct / 100.0

gmv = pf.solve(mu, sigma, cap)
if gmv is None or not gmv.ok:
    st.error("在這個上限下求解失敗，請提高 c。")
    st.stop()

lo = math.floor(gmv.ret * 1000) / 10          # 下限：全域最小變異數組合的報酬
hi = math.floor(pf.max_feasible_return(mu, cap) * 1000) / 10   # 上限：貪婪填滿

with right:
    if hi - lo < 0.05:
        # c == 1/n: equal weights is the only feasible portfolio, so there is
        # no range to slide over.  A slider with min_value == max_value is a
        # degenerate range -- show the fixed number instead of rendering one.
        st.metric("目標年化報酬", f"{lo:.1f}%", help="c 剛好等於 1/n，只有等權重一種解。")
        target_pct = lo
    else:
        target_pct = st.slider(
            "目標年化報酬（下限）",
            min_value=lo,
            max_value=hi,
            value=lo + (hi - lo) * 0.3,
            step=0.1,
            format="%.1f%%",
            help=f"可行區間 {lo:.1f}% ~ {hi:.1f}%，隨 c 改變。",
        )

result = pf.solve(mu, sigma, cap, target=target_pct / 100.0)
if result is None or not result.ok:
    st.warning("這組條件無解，請放寬上限或降低目標報酬。")
    st.stop()

# --- results --------------------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("預期年化報酬", f"{result.ret:.2%}")
m2.metric("年化波動度", f"{result.vol:.2%}")
m3.metric("報酬 / 風險", f"{result.ret / result.vol:.2f}" if result.vol else "-")

chart_left, chart_right = st.columns([1, 1.15])
with chart_left:
    st.subheader("最佳權重")
    st.altair_chart(weights_chart(result.weights, color, ink))
with chart_right:
    st.subheader("效率前緣")
    curve = frontier(mu, sigma, cap)
    if curve.empty:
        st.info("目前的上限下只有單一可行解。")
    else:
        st.altair_chart(
            frontier_chart(curve, result, color, accent, ink, surface)
        )

with st.expander("為什麼用週報酬，不用日報酬"):
    daily = prices.pct_change().dropna(how="any")
    tw = [c for c in prices.columns if c.isdigit()]
    us = [c for c in prices.columns if not c.isdigit()]
    cross_d = daily.corr().loc[tw, us].to_numpy().mean()
    cross_w = weekly.corr().loc[tw, us].to_numpy().mean()
    st.markdown(
        f"""
台北收盤比紐約早約十三個小時，日資料會把台股的今天配到美股的昨天，
跨市場相關性因此被嚴重低估——最佳化會誤以為兩邊分散效果比實際好很多。

| 資料頻率 | 台股 × 美股平均相關係數 | 年化倍數 |
|---|---|---|
| 日 | {cross_d:.3f} | 252 |
| **週（本站採用）** | **{cross_w:.3f}** | **52** |

樣本：{prices.index.min():%Y-%m-%d} ~ {prices.index.max():%Y-%m-%d}，
{len(prices)} 個共同交易日、{len(weekly)} 週。
"""
    )

with st.expander("資料表"):
    table = pd.DataFrame(
        {
            "資產": [pf.DISPLAY_NAMES.get(k, k) for k in mu.index],
            "權重": result.weights.to_numpy(),
            "年化報酬": mu.to_numpy(),
            "年化波動": sigma.to_numpy().diagonal() ** 0.5,
        }
    ).sort_values("權重", ascending=False)
    st.dataframe(
        table.style.format(
            {"權重": "{:.2%}", "年化報酬": "{:.2%}", "年化波動": "{:.2%}"}
        ),
        hide_index=True,
    )
    st.caption(
        "台股為**未還原權值**收盤價，年化報酬被低估約當現金殖利率"
        "（高殖利率的 2881、2884、2412 受影響最大）；美股 ETF 已還原配息。"
        "兩側處理不一致，目標報酬拉越高、權重被推向美股的偏誤越明顯。"
        "詳見 README 的「已知偏誤」。"
    )
