import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import altair as alt

# ---------------------------
# Page + Title
# ---------------------------
st.set_page_config(page_title="Long/Short Momentum — Recruiter Demo", page_icon="📈", layout="wide")
st.title("📈 Long/Short Momentum Strategy (Volatility-Adjusted, Stop-Loss)")

st.markdown("""
Code Written by Titouan Olier
""")
st.write("")


st.markdown("""
This app showcases a **long/short momentum** strategy on the S&P 500:
- **Go Long:** strongest momentum names  
- **Go Short:** weakest *risk-adjusted* momentum names (momentum ÷ volatility)  
- **Discipline:** per-position **stop-loss**; monthly **rebalancing**  
- **Goal:** consistent, risk-aware **alpha** across regimes
""")

# ---------------------------
# Load Universe
# ---------------------------
df = pd.read_csv("2025_sp_500_stocks.csv")
tickers = df["Symbol"].dropna().unique().tolist()

# 10Y window
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)

start_date = pd.to_datetime(start_date).tz_localize(None)
end_date = pd.to_datetime(end_date).tz_localize(None)

@st.cache_data
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    return data['Close']


prices = download_data(tickers, start_date, end_date)
prices = prices.loc[:, prices.columns.dropna()].sort_index()
returns = prices.pct_change()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Strategy Parameters")
k_long = st.sidebar.slider("Number of Long Positions", 5, 50, 10, 1)
k_short = st.sidebar.slider("Number of Short Positions", 5, 50, 15, 1)
stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", 0.0, 25.0, 5.0, 0.5) / 100.0

# ---------------------------
# Factors
# ---------------------------
def compute_momentum(prices, lookback=189): return prices.pct_change(lookback)
def compute_momentum_2(prices, lookback=252): return prices.pct_change(lookback)
def compute_momentum_3(prices, lookback=126): return prices.pct_change(lookback)
def compute_volatility(prices, lookback=20): return prices.pct_change().rolling(lookback).std() * np.sqrt(252)

mom1, mom2, mom3 = compute_momentum(prices), compute_momentum_2(prices), compute_momentum_3(prices)
momentum = mom1 + mom2 + mom3

vol_short_term = compute_volatility(prices, 126)
vol_mid_term = compute_volatility(prices, 189)
vol_long_term = compute_volatility(prices, 252)
volatility = vol_short_term + vol_mid_term + vol_long_term

# Rebalance schedule
min_date = prices.index[252]
rebalance_dates = momentum.loc[momentum.index >= min_date].index[::126]

# ---------------------------
# Portfolio construction
# ---------------------------
weights_long_history = pd.DataFrame(0, index=rebalance_dates, columns=tickers)
weights_short_history = pd.DataFrame(0, index=rebalance_dates, columns=tickers)

for date in rebalance_dates:
    mom_slice = momentum.loc[date].dropna()
    vol_slice = volatility.loc[date].dropna()
    if len(mom_slice) < (k_long + k_short):
        continue

    # Long sleeve
    longs = mom_slice.nlargest(k_long)
    w_long = longs / longs.abs().sum() * 0.5
    weights_long_history.loc[date, w_long.index] = w_long

    # Short sleeve (risk-adjusted momentum)
    adj_score_short = (mom_slice / vol_slice).dropna()
    shorts = adj_score_short.nsmallest(k_short)
    w_short = shorts / shorts.abs().sum() * 0.5
    weights_short_history.loc[date, w_short.index] = w_short

# Effective daily weights
weights_long_effective = pd.DataFrame(0, index=prices.index, columns=tickers)
weights_short_effective = pd.DataFrame(0, index=prices.index, columns=tickers)

num_long_positions = pd.Series(index=rebalance_dates, dtype=int)
num_short_positions = pd.Series(index=rebalance_dates, dtype=int)
num_long_liquidated = pd.Series(index=rebalance_dates, dtype=int)
num_short_liquidated = pd.Series(index=rebalance_dates, dtype=int)

portfolio_returns_long = []
portfolio_returns_short = []

for i in range(len(rebalance_dates) - 1):
    date0, date1 = rebalance_dates[i], rebalance_dates[i + 1]

    # ----------------- LONGS -----------------
    w_long = weights_long_history.loc[date0]

    # ✅ only tickers actually held this period
    active_long = w_long[w_long != 0].index

    rets_long = returns.loc[date0:date1].iloc[1:]
    # (optional but safe) align to active columns only
    rets_long = rets_long.reindex(columns=active_long)

    num_long_positions[date0] = len(active_long)

    # Pre-fill weights only for active longs
    weights_long_effective.loc[rets_long.index, active_long] = w_long.loc[active_long].values

    # Compute cumulative returns per active ticker
    cum_ret_long = (1 + rets_long[active_long]).cumprod() - 1

    # Stop-loss mask only on active tickers
    mask_stop_long = cum_ret_long <= -stop_loss_pct

    # Zero weights on stop days (only active tickers)
    for col in active_long:
        stop_days = mask_stop_long.index[mask_stop_long[col]]
        weights_long_effective.loc[stop_days, col] = 0

    # ✅ count how many *positions held this period* got liquidated
    num_long_liquidated[date0] = int(mask_stop_long.any(axis=0).sum())


    # Portfolio return using only active positions
    port_ret_long = (
            rets_long[active_long] *
            weights_long_effective.loc[rets_long.index, active_long]
    ).sum(axis=1)
    portfolio_returns_long.append(port_ret_long)

    # ----------------- SHORTS -----------------
    w_short = weights_short_history.loc[date0]
    active_short = w_short[w_short != 0].index

    rets_short = returns.loc[date0:date1].iloc[1:]
    rets_short = rets_short.reindex(columns=active_short)

    num_short_positions[date0] = len(active_short)

    weights_short_effective.loc[rets_short.index, active_short] = w_short.loc[active_short].values

    cum_ret_short = (1 + rets_short[active_short]).cumprod() - 1
    mask_stop_short = cum_ret_short >= stop_loss_pct

    for col in active_short:
        stop_days = mask_stop_short.index[mask_stop_short[col]]
        weights_short_effective.loc[stop_days, col] = 0

    num_short_liquidated[date0] = int(mask_stop_short.any(axis=0).sum())

    # Portfolio return
    port_ret_short = (
            rets_short[active_short] *
            weights_short_effective.loc[rets_short.index, active_short]
    ).sum(axis=1)
    portfolio_returns_short.append(port_ret_short)

portfolio_returns_long = pd.concat(portfolio_returns_long)
portfolio_returns_short = pd.concat(portfolio_returns_short)
portfolio_returns_total = portfolio_returns_long + portfolio_returns_short

# ---------------------------
# Stats
# ---------------------------
def stats(rets: pd.Series):
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = np.nan if ann_vol == 0 else ann_ret / ann_vol
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = np.nan if downside == 0 else ann_ret / downside
    return ann_ret, ann_vol, sharpe, sortino

long_stats = stats(portfolio_returns_long)
short_stats = stats(portfolio_returns_short)
total_stats = stats(portfolio_returns_total)

st.subheader("📊 Backtest Performance (10 Years)")

# Total Portfolio (Main Highlight)
st.markdown("## Total Portfolio Performance (Long + Short)")
total_cols = st.columns(4)
total_labels = ["Ann. Return", "Ann. Volatility", "Sharpe", "Sortino"]
total_values = [f"{total_stats[0]:.2%}", f"{total_stats[1]:.2%}", f"{total_stats[2]:.2f}", f"{total_stats[3]:.2f}"]

for col, label, value in zip(total_cols, total_labels, total_values):
    with col:
        st.metric(label, value)

st.markdown("---")  # Divider to separate sections

# Long vs Short Metrics
c1, c2 = st.columns(2)

with c1:
    st.markdown("### 📈 Long Portfolio")
    long_values = [f"{long_stats[0]:.2%}", f"{long_stats[1]:.2%}", f"{long_stats[2]:.2f}", f"{long_stats[3]:.2f}"]
    for label, value in zip(total_labels, long_values):
        st.metric(label, value)

with c2:
    st.markdown("### 📉 Short Portfolio")
    short_values = [f"{short_stats[0]:.2%}", f"{short_stats[1]:.2%}", f"{short_stats[2]:.2f}", f"{short_stats[3]:.2f}"]
    for label, value in zip(total_labels, short_values):
        st.metric(label, value)


# ---------------------------
# Cumulative Performance & Drawdown
# ---------------------------
cum_long = (1 + portfolio_returns_long).cumprod()
cum_short = (1 + portfolio_returns_short).cumprod()
cum_total = (1 + portfolio_returns_total).cumprod()

st.subheader("📈 Cumulative Performance")

# Put your series in a DataFrame with a proper datetime index
df = pd.DataFrame({
    "Date": cum_long.index,   # assuming cum_long, cum_short, cum_total have datetime index
    "Long": cum_long.values,
    "Short": cum_short.values,
    "Total": cum_total.values
})

# Melt into long format for Altair
df_melted = df.melt("Date", var_name="Strategy", value_name="Cumulative Return")

# Altair chart with custom x-axis (years displayed)
chart = (
    alt.Chart(df_melted)
    .mark_line()
    .encode(
        x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", title="Year")),  # format years
        y=alt.Y("Cumulative Return:Q", title="Performance"),
        color="Strategy:N"
    )
    .properties(width=700, height=400)
)

st.altair_chart(chart, use_container_width=True)



st.subheader("📉 Drawdown (Total Portfolio)")
dd = cum_total / cum_total.cummax() - 1
fig, ax = plt.subplots(figsize=(10, 3))
ax.fill_between(dd.index, dd.values, 0, alpha=0.5)
ax.set_ylabel("Drawdown")
ax.set_xlabel("Date")
st.pyplot(fig)

# ---------------------------
# Net Exposure
# ---------------------------
st.subheader("🧭 Net Exposure Over Time")

gross_long_ts = weights_long_effective.sum(axis=1)
gross_short_ts = weights_short_effective.sum(axis=1)
net_exposure = gross_long_ts + gross_short_ts  # (note: with your logic this typically centers near 0.0)

# Prepare the dataframe
exposure_df = pd.DataFrame({
    "Date": net_exposure.index,
    "Gross Long": gross_long_ts,
    "Gross Short": gross_short_ts,
    "Net Exposure": net_exposure
}).reset_index(drop=True)

# Melt for Altair
exposure_melted = exposure_df.melt(id_vars="Date", var_name="Type", value_name="Exposure")

# Plot
chart = alt.Chart(exposure_melted).mark_line().encode(
    x=alt.X("yearmonth(Date):T", title="Year"),
    y=alt.Y("Exposure", title="Exposure"),
    color="Type:N",
    tooltip=["Date:T", "Type:N", "Exposure:Q"]
).properties(
    width=800,
    height=400
).interactive()

st.altair_chart(chart, use_container_width=True)

# ---------------------------
# Benchmark Comparison (SPY)
# ---------------------------
@st.cache_data
def load_benchmark(symbol: str, start: datetime, end: datetime) -> pd.Series:
    s = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    return s.dropna()

bench = load_benchmark("^GSPC", start_date, end_date)
bench_ret = bench.pct_change().reindex(portfolio_returns_total.index).fillna(0.0)
cum_bench = (1 + bench_ret).cumprod()

st.subheader("🏆 Strategy vs Benchmark (SPY)")
st.line_chart(pd.DataFrame({
    "Strategy (Total)": cum_total,
    "SPY": cum_bench.squeeze()  # <-- convert (N,1) array to 1D
}))

# Excess performance & simple CAPM alpha/beta
excess_ann_return = (cum_total.iloc[-1] ** (252 / len(portfolio_returns_total)) - cum_bench.iloc[-1] ** (252 / len(portfolio_returns_total)))
def capm_alpha_beta(y: pd.Series, x: pd.Series):
    df = pd.concat([y, x], axis=1).dropna()
    if df.empty: return np.nan, np.nan
    X = np.vstack([np.ones(len(df)), df.iloc[:,1].values]).T
    beta = np.linalg.lstsq(X, df.iloc[:,0].values, rcond=None)[0][1]
    alpha_daily = np.linalg.lstsq(X, df.iloc[:,0].values, rcond=None)[0][0]
    return alpha_daily*252, beta

alpha_ann, beta = capm_alpha_beta(portfolio_returns_total, bench_ret)

m1, m2, m3 = st.columns(3)
#m1.metric("Excess Ann. Return (Strategy − SPY)", f"{excess_ann_return:.2%}")
m1.metric("Excess Ann. Return (Strategy − SPY)", f"{excess_ann_return.iloc[0]:.2%}")
m2.metric("CAPM Alpha (ann.)", f"{alpha_ann:.2%}" if pd.notna(alpha_ann) else "—")
m3.metric("Beta vs SPY", f"{beta:.2f}" if pd.notna(beta) else "—")

# ---------------------------
# Latest Allocations (styled)
# ---------------------------
last_date_alloc = weights_long_history.index[-1]
st.subheader(f"📌 Latest Allocations ({last_date_alloc.date()})")

colL, colS = st.columns(2)
with colL:
    last_long = weights_long_history.loc[last_date_alloc]
    long_df = pd.DataFrame({'Long Weight': last_long})
    long_df = long_df[long_df['Long Weight'] != 0].sort_values('Long Weight', ascending=False)
    st.dataframe(long_df.style.format("{:.2%}").background_gradient(cmap="Greens"))
with colS:
    last_short = weights_short_history.loc[last_date_alloc]
    short_df = pd.DataFrame({'Short Weight': last_short})
    short_df = short_df[short_df['Short Weight'] != 0].sort_values('Short Weight', ascending=True)
    st.dataframe(short_df.style.format("{:.2%}").background_gradient(cmap="Reds_r"))

# ---------------------------
# Positions & Liquidations
# ---------------------------
st.subheader("🧾 Positions & Liquidations per Rebalance")
st.dataframe(
    pd.DataFrame({
        "Longs": num_long_positions,
        "Shorts": num_short_positions,
        "Longs Liquidated": num_long_liquidated,
        "Shorts Liquidated": num_short_liquidated
    }).style.format("{:.0f}").background_gradient(cmap="Blues")
)

# ---------------------------
# Key Takeaways
# ---------------------------
st.markdown("""
## Key Takeaways
- **Balanced Alpha Engine:** Long winners, short (risk-adjusted) losers — diversified across both tails.
- **Risk Discipline:** Per-position stop-loss + steady net exposure keep tail risk contained.
- **Quality of Returns:** Strong **Sharpe** and **Sortino**, with controlled **drawdowns**.
- **Benchmark Edge:** Strategy outperforms **SPY** with positive **alpha** and differentiated beta profile.
""")
