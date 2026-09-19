"""Ballast - minimum-variance portfolio optimiser over a TW + US universe."""

from __future__ import annotations

import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import portfolio as pf

PRICES = Path(__file__).parent / "prices.csv"

# Validated two-colour set (dataviz six-checks, light and dark).
SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
ACCENT = {"light": "#eb6834", "dark": "#d95926"}


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


@st.cache_data(show_spinner=False)
def load():
    prices = pf.load_prices(PRICES)
    weekly = pf.weekly_returns(prices)
    mu, sigma = pf.annualise(weekly)
    return prices, weekly, mu, sigma


@st.cache_data(show_spinner=False)
def frontier(mu: pd.Series, sigma: pd.DataFrame, cap: float) -> pd.DataFrame:
    return pf.efficient_frontier(mu, sigma, cap)


def weights_chart(weights: pd.Series, color: str) -> alt.Chart:
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
    labels = base.mark_text(align="left", dx=6, fontSize=12).encode(
        text=alt.Text("權重:Q", format=".1%")
    )
    return (bars + labels).properties(height=max(220, 34 * len(df)))


def frontier_chart(curve: pd.DataFrame, point: pf.Portfolio, color: str, accent: str):
    line = (
        alt.Chart(curve)
        .mark_line(color=color, strokeWidth=2)
        .encode(
            x=alt.X(
                "vol:Q",
                title="年化波動度",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(zero=False, nice=True),
            ),
            y=alt.Y(
                "ret:Q",
                title="預期年化報酬",
                axis=alt.Axis(format=".0%"),
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
            size=110, filled=True, color=accent, stroke="white", strokeWidth=2
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
        .mark_text(align="left", dx=12, dy=-2, fontSize=12, color=accent)
        .encode(x="vol:Q", y="ret:Q", text="標記:N")
    )
    return (line + dot + tag).properties(height=380)


st.set_page_config(page_title="Ballast", page_icon="⚖️", layout="wide")
st.title("⚖️ Ballast")
st.caption(
    "台股五檔 + 美股三檔 ETF 的最小變異數投組。價格全部換算成台幣，"
    "報酬率以週為單位計算。"
)

if not PRICES.exists():
    st.error("找不到 `prices.csv`。先執行 `python fetch_data.py` 抓取資料。")
    st.stop()

prices, weekly, mu, sigma = load()
n = len(mu)
mode = theme()
color, accent = SERIES[mode], ACCENT[mode]

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
        st.slider("目標年化報酬", min_value=lo, max_value=lo, value=lo, disabled=True,
                  format="%.1f%%", help="c 剛好等於 1/n，只有等權重一種解。")
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
    st.altair_chart(weights_chart(result.weights, color), use_container_width=True)
with chart_right:
    st.subheader("效率前緣")
    curve = frontier(mu, sigma, cap)
    if curve.empty:
        st.info("目前的上限下只有單一可行解。")
    else:
        st.altair_chart(
            frontier_chart(curve, result, color, accent), use_container_width=True
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
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "台股為**未還原權值**收盤價，年化報酬被低估約當現金殖利率"
        "（高殖利率的 2881、2884、2412 受影響最大）；美股 ETF 已還原配息。"
        "兩側處理不一致，目標報酬拉越高、權重被推向美股的偏誤越明顯。"
        "詳見 README 的「已知偏誤」。"
    )
