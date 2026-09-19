"""Ballast - minimum-variance portfolio optimiser over a TW + US universe."""

from __future__ import annotations

import math
import os
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import backtest as bt
import portfolio as pf

PRICES = Path(__file__).parent / "prices.csv"

# Validated two-colour set (dataviz six-checks, light and dark).
SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
ACCENT = {"light": "#eb6834", "dark": "#d95926"}
INK = {"light": "#52514e", "dark": "#c3c2b7"}       # secondary text
# Diverging pair for correlation: warm/cool poles with a GREY midpoint, so
# "uncorrelated" reads as nothing rather than as a third colour.
# The dark midpoint is the terminal's own border step rather than the warm grey
# the reference palette ships: on a cool blue-black panel a warm neutral reads
# as dirt, and "no correlation" should sink into the panel, not sit on it.
DIVERGING = {"light": ["#2a78d6", "#f0efec", "#e34948"],
             "dark": ["#3987e5", "#2a2e39", "#e66767"]}
SURFACE = {"light": "#ffffff", "dark": "#1b1f2b"}   # panel the charts sit on
GRID = {"light": "#e8e8e5", "dark": "#2a2e39"}      # recessive gridlines
# Four equity curves: the categorical theme's first four slots, in order.
# Validated on the adjacent pairlist against the panel in both modes.
LINES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
         "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"]}


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
def backtest(fingerprint, cap: float, target: float | None, lookback: int, step: int):
    tracks, table = bt.compare(
        pf.load_prices(PRICES), cap, target, lookback=lookback, step=step
    )
    return tracks, table


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


def terminal(chart, ink: str, grid: str):
    """Recessive axes and gridlines, applied at the top level of every chart.

    Trading screens keep the chrome quiet and let the marks carry the page:
    no axis domain lines, hairline grid, labels in muted ink.  Numbers use
    tabular figures so columns of digits line up instead of shimmering.
    """
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            grid=True,
            gridColor=grid,
            gridWidth=1,
            domain=False,
            tickColor=grid,
            tickSize=4,
            labelColor=ink,
            titleColor=ink,
            labelFontSize=11,
            titleFontSize=11,
            labelFontWeight=400,
            titleFontWeight=400,
        )
        .configure_legend(
            labelColor=ink, titleColor=ink, labelFontSize=11, titleFontSize=11,
            symbolStrokeWidth=0,
        )
        .configure_text(font="ui-monospace, SFMono-Regular, Menlo, monospace")
    )


def risk_chart(
    weights: pd.Series, contributions: pd.Series, color: str, accent: str
) -> alt.Chart:
    """Weight against risk contribution, one pair of bars per holding."""
    names = [pf.DISPLAY_NAMES.get(k, k) for k in weights.index]
    order = [n for _, n in sorted(zip(weights.to_numpy(), names), reverse=True)]
    df = pd.concat(
        [
            pd.DataFrame({"資產": names, "值": weights.to_numpy(), "類別": "權重"}),
            pd.DataFrame(
                {"資產": names, "值": contributions.to_numpy(), "類別": "風險貢獻"}
            ),
        ]
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3, height=9)
        .encode(
            y=alt.Y("資產:N", sort=order, title=None),
            yOffset=alt.YOffset("類別:N", sort=["權重", "風險貢獻"]),
            x=alt.X("值:Q", title=None, axis=alt.Axis(format=".0%", tickCount=6)),
            # Two series, so a legend is not optional.
            color=alt.Color(
                "類別:N",
                scale=alt.Scale(
                    domain=["權重", "風險貢獻"], range=[color, accent]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("資產:N"),
                alt.Tooltip("類別:N", title=""),
                alt.Tooltip("值:Q", format=".2%"),
            ],
        )
        .properties(width="container", height=max(240, 38 * len(weights)))
    )


def equity_chart(tracks, ramp: list[str]) -> alt.Chart:
    """Growth of 1 for each track, indexed from the first out-of-sample week."""
    frames = []
    for track in tracks:
        curve = track.curve
        frames.append(
            pd.DataFrame({"日期": curve.index, "成長": curve.to_numpy(),
                          "組合": track.name})
        )
    df = pd.concat(frames)
    names = [t.name for t in tracks]
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("日期:T", title=None),
            y=alt.Y("成長:Q", title="成長倍數（起點 = 1）",
                    scale=alt.Scale(zero=False), axis=alt.Axis(format=".1f")),
            color=alt.Color(
                "組合:N",
                scale=alt.Scale(domain=names, range=ramp[: len(names)]),
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
            tooltip=[alt.Tooltip("日期:T"), alt.Tooltip("組合:N", title=""),
                     alt.Tooltip("成長:Q", format=".3f")],
        )
        .properties(width="container", height=360)
    )


def correlation_chart(corr: pd.DataFrame, ramp: list[str], ink: str) -> alt.Chart:
    """Correlation matrix as a heatmap on a diverging ramp centred at zero."""
    long = corr.stack().rename("r").reset_index()
    long.columns = ["A", "B", "r"]
    for col in ("A", "B"):
        long[col] = [pf.DISPLAY_NAMES.get(k, k) for k in long[col]]
    labels = [pf.DISPLAY_NAMES.get(k, k) for k in corr.index]

    base = alt.Chart(long).encode(
        x=alt.X("A:N", sort=labels, title=None,
                axis=alt.Axis(labelAngle=-45, labelLimit=120)),
        y=alt.Y("B:N", sort=labels, title=None, axis=alt.Axis(labelLimit=120)),
    )
    cells = base.mark_rect(stroke=None).encode(
        color=alt.Color(
            "r:Q",
            # Domain pinned symmetrically so zero always lands on the neutral.
            # Interpolating in Lab rather than RGB: straight RGB from the grey
            # midpoint to the blue pole passes through a desaturated teal, and
            # a diverging ramp must not invent a third hue.
            scale=alt.Scale(
                range=ramp, domain=[-1, 0, 1], type="linear", interpolate="lab"
            ),
            legend=alt.Legend(title="相關係數", format=".1f", orient="right"),
        ),
        tooltip=[alt.Tooltip("A:N", title=""), alt.Tooltip("B:N", title=""),
                 alt.Tooltip("r:Q", format=".3f", title="相關係數")],
    )
    # Every cell labelled: eight by eight is small enough, and the numbers are
    # the point -- the colour is only there to make the blocks pop out.
    text = base.mark_text(fontSize=10, color=ink).encode(
        text=alt.Text("r:Q", format=".2f")
    )
    return (cells + text).properties(width="container", height=340)


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
st.markdown(
    """
    <style>
      /* Numbers read as a column of figures, not as prose: tabular digits stop
         them shimmering when a slider moves and a 1 replaces an 8. */
      [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
      [data-testid="stDataFrame"], .stSlider [data-testid="stTickBar"] {
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1;
      }
      /* Readings sit in their own panel, the way a terminal boxes its quotes. */
      [data-testid="stMetric"] {
        background: #1b1f2b;
        border: 1px solid #2a2e39;
        border-radius: 4px;
        padding: 10px 12px;
      }
      [data-testid="stMetricLabel"] p {
        font-size: 0.72rem;
        letter-spacing: .06em;
        color: #787b86;
        text-transform: uppercase;
      }
      [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 500; }
      /* Charts get the same panel treatment so the page reads as tiles. */
      [data-testid="stVegaLiteChart"] {
        background: #1b1f2b;
        border: 1px solid #2a2e39;
        border-radius: 4px;
        padding: 8px 10px 4px;
      }
      /* Tabs as a terminal's section strip: a hairline rule, no pill chrome. */
      .stTabs [data-baseweb="tab-list"] {
        gap: 1.4rem;
        border-bottom: 1px solid #2a2e39;
      }
      .stTabs [data-baseweb="tab"] {
        padding: 6px 0;
        font-size: 0.86rem;
        letter-spacing: .02em;
      }
      h1 { letter-spacing: -.02em; }
      h3, h4, h5, h6 { color: #d1d4dc; letter-spacing: .01em; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
ink, surface, grid = INK[mode], SURFACE[mode], GRID[mode]

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
    st.altair_chart(terminal(weights_chart(result.weights, color, ink), ink, grid))
with chart_right:
    st.subheader("效率前緣")
    curve = frontier(mu, sigma, cap)
    if curve.empty:
        st.info("目前的上限下只有單一可行解。")
    else:
        st.altair_chart(
            terminal(frontier_chart(curve, result, color, accent, ink, surface), ink, grid)
        )

plain, advanced, back, why, data = st.tabs(
    ["白話說明", "進階指標", "回測", "為什麼用週報酬", "資料表"]
)

with plain:
    st.markdown(
        """
#### 這個頁面在做什麼

把錢分散到八個標的上，找出「在你要求的報酬之下，**波動最小**」的那一種分法。

為什麼要分散？因為不同資產不會同時漲跌。台積電跌的時候黃金可能在漲，兩個擺在
一起，整體的起伏會比單押任何一個都小。這個頁面做的就是用數學把「怎麼分最穩」算出來。

#### 兩個滑桿在控制什麼

**單一標的上限 c**　不准把超過這個比例的錢押在同一個標的上。

拉低就是強迫分散，但可選的組合變少；拉高允許集中，數學上波動可以壓得更低，
但押錯一檔就傷得重。

**目標年化報酬**　你至少要賺多少。這是個**下限**，不是預測。

拉高就必須買進更多高報酬、也更會跳的標的，波動一定跟著上去。
天下沒有白吃的午餐——那條曲線就是這句話的數字版。

#### 兩張圖怎麼看

**最佳權重**　每個標的該放多少錢，加起來剛好 100%。

**效率前緣**　那條曲線是「每個報酬水準下，最低能做到多少波動」。橘點是你現在的
設定落在哪裡。往左下走是穩但賺得少，往右上走是賺得多但顛。

曲線的**左上方是空的**——那裡代表「高報酬又低波動」，在這組資料下不存在。
這不是畫圖偷懶，是數學上真的到不了。

#### 三個數字的意思

- **預期年化報酬**：用過去的平均推估出來的，不是保證。
- **年化波動度**：漲跌的劇烈程度。8% 大致表示多數年份的報酬會落在「平均加減 8%」的範圍內。
- **報酬 / 風險**：每扛一單位波動換到多少報酬，越高越划算。

#### 最該記得的一句話

這裡所有「預期報酬」都是拿**歷史平均**當估計值。過去七年台積電漲很多，
不代表未來會。這個模型對報酬的估計極度敏感，換一段期間、換一組標的，
結果就會不一樣。

它示範的是最佳化的方法，不是投資建議。
"""
    )

with advanced:
    rf = st.number_input(
        "無風險利率（年化）", value=1.5, step=0.1, format="%.1f",
        help="用來算 Sharpe。台灣一年期定存大致在這個區間。",
    ) / 100.0

    series = pf.portfolio_returns(result.weights, weekly)
    rc = pf.risk_contributions(result.weights, sigma)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Sharpe", f"{pf.sharpe(result.ret, result.vol, rf):.2f}",
              help=f"（年化報酬 − {rf:.1%}）÷ 年化波動")
    a2.metric("最大回撤", f"{pf.max_drawdown(series):.1%}",
              help="樣本內：權重是看過這段歷史才選出來的，實際交易不會這麼好看。")
    a3.metric("分散比率", f"{pf.diversification_ratio(result.weights, sigma):.2f}",
              help="加權平均波動 ÷ 投組波動。1.00 表示分散完全沒有幫助。")
    a4.metric("有效持股數", f"{pf.effective_holdings(result.weights):.2f} / {n}",
              help="1 / Σw²。持有八檔但集中在少數幾檔時，這個數字會遠小於 8。")

    st.markdown("###### 權重 ≠ 風險貢獻")
    st.caption(
        "風險貢獻是 wᵢ(Σw)ᵢ / w'Σw，加總為 1。最小變異數組合常把大筆資金押在"
        "波動低的標的上，那檔卻只扛了一小部分風險——兩條棒子擺在一起才看得出來。"
    )
    st.altair_chart(terminal(risk_chart(result.weights, rc, color, accent), ink, grid))

    st.markdown("###### 相關係數矩陣（週報酬）")
    st.caption(
        "左上 5×5 是台股彼此，右下 3×3 是美股 ETF 彼此，交叉的區塊就是跨市場。"
        "跨市場那塊明顯比兩個對角區塊淡，這正是分散效果的來源。"
    )
    st.altair_chart(
        terminal(correlation_chart(weekly.corr(), DIVERGING[mode], ink), ink, grid)
    )

with back:
    st.caption(
        "上面那組權重是用**全部**歷史算出來的，拿同一段歷史評分等於自己考自己。"
        "這裡改問另一個問題：如果當年只看得到當下為止的資料、每隔一段時間重新求解一次，"
        "實際會拿到什麼。每次求解只看前面的視窗，持有期完全在視窗之外，沒有偷看未來。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        lookback = st.select_slider("求解視窗（週）", [104, 156, 208], value=156)
    with c2:
        step = st.select_slider("持有期（週）", [1, 2, 4, 13], value=4)
    with c3:
        use_target = st.checkbox(
            "套用目標報酬", value=False,
            help="不勾就是純最小變異數；勾了會沿用上面滑桿的目標，並在每個視窗夾進可行範圍。",
        )

    with st.spinner("逐期重新求解…"):
        tracks, table = backtest(
            data_fingerprint(), cap,
            target_pct / 100.0 if use_target else None, lookback, step,
        )

    oos = tracks[0].returns
    if oos.empty:
        st.warning("這組設定的資料不足以跑出樣本外區間，請縮短求解視窗。")
    else:
        st.caption(
            f"樣本外 {oos.index.min():%Y-%m-%d} ~ {oos.index.max():%Y-%m-%d}，"
            f"{len(oos)} 週、{len(tracks[0].weights)} 次再平衡"
        )
        st.altair_chart(terminal(equity_chart(tracks, LINES[mode]), ink, grid))

        show = table.copy()
        show.index.name = "組合"
        st.dataframe(
            show.reset_index().style.format(
                {c: "{:.2%}" for c in show.columns if c != "Sharpe"}
                | {"Sharpe": "{:.2f}"}
            ),
            hide_index=True,
        )

        st.markdown(
            """
###### 怎麼讀這張表

這個模型**最小化變異數**，它沒有在最佳化報酬，也沒有在最佳化 Sharpe。
所以要看它有沒有做到本分，看的是波動和回撤那兩欄——不是報酬那欄。

等權重的 Sharpe 通常會贏。這不是實作出錯，是
[DeMiguel, Garlappi & Uppal (2009)](https://doi.org/10.1093/rfs/hhm075)
那個著名結論在這組資料上重現：用歷史平均估期望報酬的誤差太大，
1/N 這種完全不估計的做法反而難以擊敗。

**換手率那欄要一起看。** 目標報酬拉越高，最佳化越要去追前一個視窗裡剛好表現好的
標的，換手就越兇。而這張表的所有數字**都沒有扣交易成本**——台股賣出還有 0.3% 證交稅。
一個換手 11% 的策略和一個換手 2% 的策略，帳面報酬不能直接比。

樣本外只有四年多，而且那段期間股市多頭。低波動策略在多頭裡本來就會落後。
"""
        )

with why:
    daily = prices.pct_change().dropna(how="any")
    tw = [c for c in prices.columns if c.isdigit()]
    us = [c for c in prices.columns if not c.isdigit()]
    st.markdown(
        f"""
台北收盤比紐約早約十三個小時，日資料會把台股的今天配到美股的昨天，
跨市場相關性因此被嚴重低估——最佳化會誤以為兩邊分散效果比實際好很多。

| 配對 | 日資料 | 週資料 |
|---|---|---|
| 台股 × 美股（平均） | **{daily.corr().loc[tw, us].to_numpy().mean():+.3f}** | **{weekly.corr().loc[tw, us].to_numpy().mean():+.3f}** |
| 2330 × SPY | {daily.corr().loc['2330', 'SPY']:+.3f} | {weekly.corr().loc['2330', 'SPY']:+.3f} |
| 台股內部（平均） | {daily.corr().loc[tw, tw].to_numpy().mean():+.3f} | {weekly.corr().loc[tw, tw].to_numpy().mean():+.3f} |

關鍵在第三列：**台股內部幾乎沒變**，因為那些標的同時開收盤，沒有錯位問題。
只有跨市場的配對會暴增——這是時差的指紋，不是「換週資料所有相關性都會上升」。

年化倍數因此是 52，不是 252。
樣本 {prices.index.min():%Y-%m-%d} ~ {prices.index.max():%Y-%m-%d}，
{len(prices)} 個共同交易日、{len(weekly)} 週。
"""
    )

with data:
    table = pd.DataFrame(
        {
            "資產": [pf.DISPLAY_NAMES.get(k, k) for k in mu.index],
            "權重": result.weights.to_numpy(),
            "風險貢獻": pf.risk_contributions(result.weights, sigma).to_numpy(),
            "年化報酬": mu.to_numpy(),
            "年化波動": sigma.to_numpy().diagonal() ** 0.5,
        }
    ).sort_values("權重", ascending=False)
    st.dataframe(
        table.style.format(
            {
                "權重": "{:.2%}",
                "風險貢獻": "{:.2%}",
                "年化報酬": "{:.2%}",
                "年化波動": "{:.2%}",
            }
        ),
        hide_index=True,
    )
    st.caption(
        "台股為**未還原權值**收盤價，年化報酬被低估約當現金殖利率"
        "（高殖利率的 2881、2884、2412 受影響最大）；美股 ETF 已還原配息。"
        "兩側處理不一致，目標報酬拉越高、權重被推向美股的偏誤越明顯。"
        "詳見 README 的「已知偏誤」。"
    )
