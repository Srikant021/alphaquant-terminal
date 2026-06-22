import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import timedelta
import concurrent.futures

warnings.filterwarnings('ignore')

# ========================= CONFIG & THEME =========================
st.set_page_config(
    page_title="Quant ML Master Dashboard v5.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHART_THEME = {
    "template": "plotly_dark",
    "primary": "#67e8f9",     # Cyan
    "secondary": "#fbbf24",   # Amber
    "bullish": "#22c55e",     # Green
    "bearish": "#ef4444",     # Red
    "neutral": "white",
    "watermark": "gray"
}

FNO_INDICES = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Finnifty": "NIFTY_FIN_SERVICE.NS",
    "Midcap Nifty": "^NSEMDCP50",
    "Nifty Next 50": "^NSMIDCP",
    "Sensex": "^BSESN"
}

CRYPTO_ASSETS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Ripple (XRP)": "XRP-USD"
}

# ========================= DATA & UTILITIES =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return None

def get_vix_data(asset_class, ticker, period="1y"):
    """Fetches real VIX for Equities, or generates a Synthetic 30D VIX for Crypto."""
    if asset_class == "Indian Equities":
        return fetch_data("^INDIAVIX", period=period)
    else:
        # Generate Synthetic VIX for Crypto (30D Realized Volatility)
        data = fetch_data(ticker, period="2y") # Fetch extra for rolling window
        if data is None: return None
        ret = np.log(data['Close'] / data['Close'].shift(1))
        synth_vix = ret.rolling(30).std() * np.sqrt(365) * 100
        vix_df = data[['Close']].copy()
        vix_df['Close'] = synth_vix
        vix_df = vix_df.dropna()
        
        days_map = {"5d": 5, "6mo": 180, "1y": 365}
        days = days_map.get(period, 365)
        return vix_df.tail(days)

def get_scalar(series):
    if isinstance(series, pd.Series):
        if series.empty: return 0.0
        val = series.iloc[-1]
    else:
        val = series
    if hasattr(val, 'item'): return float(val.item())
    return float(val)

def add_watermark(fig):
    fig.add_annotation(
        text="Quant ML Master Dashboard v5.0", xref="paper", yref="paper",
        x=0.99, y=0.01, showarrow=False, font=dict(size=10, color=CHART_THEME["watermark"]), opacity=0.6
    )

# ========================= 1. IVR & IVP =========================
def calc_volatility_metrics(close_series):
    current_iv = get_scalar(close_series)
    high_52w = get_scalar(close_series.max())
    low_52w = get_scalar(close_series.min())
    denom = (high_52w - low_52w)
    ivr = ((current_iv - low_52w) / denom * 100) if denom != 0 else 0.0
    ivp = (close_series[close_series < current_iv].count() / len(close_series)) * 100
    regime = "HIGH VOLATILITY - Net Short Premium" if ivr > 50 else "LOW VOLATILITY - Net Long Premium"
    return current_iv, high_52w, low_52w, ivr, ivp, regime

def render_volatility_metrics(asset_class, ticker):
    vix_name = "India VIX" if asset_class == "Indian Equities" else "Synthetic IV (30D HV)"
    data = get_vix_data(asset_class, ticker, period="1y")
    if data is None: return st.warning(f"{vix_name} data unavailable.")
    
    close = data['Close']
    current_iv, high_52w, low_52w, ivr, ivp, regime = calc_volatility_metrics(close)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode='lines', name=vix_name, line=dict(color=CHART_THEME["primary"], width=2.5)))
    fig.add_hline(y=high_52w, line_dash="dash", line_color=CHART_THEME["bearish"], annotation_text=f"52W High: {high_52w:.1f}")
    fig.add_hline(y=low_52w, line_dash="dash", line_color=CHART_THEME["bullish"], annotation_text=f"52W Low: {low_52w:.1f}")
    fig.add_hline(y=current_iv, line_color=CHART_THEME["neutral"], line_width=3, annotation_text=f"Current: {current_iv:.1f}")
    fig.update_layout(title=f"Macro Volatility Rank & Percentile ({vix_name})", template=CHART_THEME["template"], height=650, hovermode="x unified")
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("IV Rank (IVR)", f"{ivr:.1f}%")
    c2.metric("IV Percentile (IVP)", f"{ivp:.1f}%")
    c3.metric("Regime", regime)

# ========================= 2. Expected Move =========================
def calc_expected_move(spot, current_vix, trading_days):
    daily_vol = (current_vix / 100) * np.sqrt(1 / trading_days)
    return spot * daily_vol, daily_vol

def render_expected_move(selected_name, ticker, asset_class, currency, trading_days):
    asset_data = fetch_data(ticker, period="1mo")
    vix = get_vix_data(asset_class, ticker, period="5d")
    if asset_data is None or vix is None: return st.warning(f"Data fetch failed for Expected Move ({selected_name}).")

    spot = get_scalar(asset_data['Close'])
    current_vix = get_scalar(vix['Close'])
    exp_move, daily_vol = calc_expected_move(spot, current_vix, trading_days)
    recent = asset_data.tail(20)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent['Close'], mode='lines+markers', name=selected_name, line=dict(color=CHART_THEME["primary"], width=3)))
    tomorrow = recent.index[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(x=[recent.index[-1], tomorrow], y=[spot, spot], mode='lines', name='Spot', line=dict(color=CHART_THEME["neutral"], dash='dash')))
    fig.add_trace(go.Scatter(x=[tomorrow], y=[spot + exp_move], mode='markers', name='Upper (+1σ)', marker=dict(color=CHART_THEME["bullish"], size=18, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=[tomorrow], y=[spot - exp_move], mode='markers', name='Lower (-1σ)', marker=dict(color=CHART_THEME["bearish"], size=18, symbol='triangle-down')))
    fig.update_layout(title=f"{selected_name} Daily Expected Move (1σ)", template=CHART_THEME["template"], height=600)
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Spot Price", f"{currency}{spot:,.2f}")
    c2.metric("Vol Benchmark", f"{current_vix:.2f}")
    c3.metric("Expected Move", f"± {currency}{exp_move:,.1f} ({daily_vol*100:.2f}%)")

# ========================= 3. Correlation Divergence =========================
def calc_divergence(d1_close, d2_close, name1, name2):
    df1 = d1_close.to_frame(name=name1)
    df2 = d2_close.to_frame(name=name2)
    data = pd.merge(df1, df2, left_index=True, right_index=True).dropna()
    normalized = (data / data.iloc[0]) * 100
    log_returns = np.log(data / data.shift(1)).dropna()
    rolling_corr = log_returns[name1].rolling(20).corr(log_returns[name2]).dropna()
    current_corr = float(rolling_corr.iloc[-1])
    regime = "HIGH CORRELATION" if current_corr > 0.80 else ("SEVERE DIVERGENCE" if current_corr < 0.50 else "MODERATE DIVERGENCE")
    return data, normalized, rolling_corr, current_corr, regime

def render_index_divergence(div1, div2, name1, name2, currency):
    d1 = fetch_data(div1, period="1y")
    d2 = fetch_data(div2, period="1y")
    if d1 is None or d2 is None: return st.warning("Divergence data unavailable.")

    data, normalized, rolling_corr, current_corr, regime = calc_divergence(d1['Close'], d2['Close'], name1, name2)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Normalized Prices", "20-Day Rolling Correlation"), row_heights=[0.65, 0.35], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name1], name=name1, line=dict(color=CHART_THEME["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name2], name=name2, line=dict(color=CHART_THEME["secondary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name='Correlation', line=dict(color=CHART_THEME["neutral"])), row=2, col=1)
    fig.update_layout(title=f"{name1} vs {name2} Correlation", template=CHART_THEME["template"], height=750)
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(name1, f"{currency}{float(data[name1].iloc[-1]):,.2f}")
    c2.metric(name2, f"{currency}{float(data[name2].iloc[-1]):,.2f}")
    c3.metric("20D Correlation", f"{current_corr:.3f}", delta=regime)

# ========================= 4. Volatility Cone =========================
def calc_vol_cone(close_series, trading_days):
    returns = np.log(close_series / close_series.shift(1)).dropna()
    windows = [10, 20, 30, 60, 90, 120, 180, trading_days]
    stats = []
    for w in windows:
        vol = returns.rolling(w).std() * np.sqrt(trading_days) * 100
        stats.append({
            'window': w, 'max': float(vol.max()), 'min': float(vol.min()),
            'median': float(vol.median()), 'current': float(vol.iloc[-1])
        })
    return pd.DataFrame(stats)

def render_volatility_cone(selected_name, ticker, trading_days):
    data = fetch_data(ticker, period="max")
    if data is None: return st.warning("Data unavailable for Volatility Cone.")

    df_stats = calc_vol_cone(data['Close'], trading_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['max'], name='Max Vol', mode='lines+markers', line=dict(color=CHART_THEME["bearish"])))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['min'], name='Min Vol', mode='lines+markers', line=dict(color=CHART_THEME["bullish"])))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['median'], name='Median', mode='lines+markers', line=dict(color=CHART_THEME["neutral"], dash='dash')))
    fig.add_trace(go.Scatter(x=df_stats['window'], y=df_stats['current'], name='Current', mode='lines+markers', line=dict(color=CHART_THEME["secondary"], width=4)))
    fig.update_layout(title=f"Volatility Cone - {selected_name}", template=CHART_THEME["template"], height=650)
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

# ========================= 5. Volatility Risk Premium =========================
def calc_vrp(close_series, vix_close, trading_days):
    hv = np.log(close_series / close_series.shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100
    df = pd.merge(vix_close.to_frame('VIX'), hv.to_frame('HV'), left_index=True, right_index=True).dropna()
    df['VRP'] = df['VIX'] - df['HV']
    current_vrp = float(df['VRP'].iloc[-1])
    regime = "POSITIVE VRP (Good for Sellers)" if current_vrp > 0 else "NEGATIVE VRP (Good for Buyers)"
    return df, current_vrp, regime

def render_vrp(selected_name, ticker, asset_class, trading_days):
    main_data = fetch_data(ticker, period="6mo")
    vix = get_vix_data(asset_class, ticker, period="6mo")
    if main_data is None or vix is None: return st.warning("Data unavailable for VRP.")

    df, current_vrp, regime = calc_vrp(main_data['Close'], vix['Close'], trading_days)
    vix_label = 'Implied Vol (VIX)' if asset_class == "Indian Equities" else 'Synthetic IV (30D HV)'

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name=vix_label, line=dict(color=CHART_THEME["primary"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HV'], name='Realized Vol (20D)', line=dict(color=CHART_THEME["secondary"])), row=1, col=1)
    colors = np.where(df['VRP'] > 0, CHART_THEME["bullish"], CHART_THEME["bearish"])
    fig.add_trace(go.Bar(x=df.index, y=df['VRP'], name='VRP', marker_color=colors), row=2, col=1)
    fig.update_layout(title=f"Volatility Risk Premium (VRP) - {selected_name}", template=CHART_THEME["template"], height=750)
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Vol Benchmark", f"{float(df['VIX'].iloc[-1]):.2f}")
    c2.metric("HV (20D)", f"{float(df['HV'].iloc[-1]):.2f}")
    c3.metric("VRP", f"{current_vrp:+.2f}%", regime)

# ========================= 6. Hurst Exponent =========================
def calculate_hurst(ts):
    if len(ts) < 20: return np.nan
    lags = range(2, 20)
    reg_val = [np.std(ts.values[lag:] - ts.values[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(reg_val), 1)
    return poly[0]

def render_hurst_regime(selected_name, ticker):
    data = fetch_data(ticker, period="1y")
    if data is None: return st.warning("Data unavailable for Hurst Exponent.")

    close = data['Close']
    log_prices = np.log(close)
    hurst_series = log_prices.rolling(window=60).apply(calculate_hurst, raw=False)
    df = pd.DataFrame({'Close': close, 'Hurst': hurst_series}).dropna()
    current_hurst = float(df['Hurst'].iloc[-1])
    regime = "MEAN REVERTING" if current_hurst < 0.45 else ("TRENDING" if current_hurst > 0.55 else "RANDOM WALK")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color=CHART_THEME["neutral"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Hurst'], name='Hurst', line=dict(color=CHART_THEME["primary"])), row=2, col=1)
    fig.update_layout(title=f"Hurst Exponent - Market Regime ({selected_name})", template=CHART_THEME["template"], height=750)
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Hurst Value", f"{current_hurst:.3f}")
    c2.metric("Regime", regime)

# ========================= 7. Liquidity Sweeps =========================
def calc_liquidity_sweeps(df):
    window = 20
    df['Prev_High'] = df['High'].rolling(window).max().shift(1)
    df['Prev_Low'] = df['Low'].rolling(window).min().shift(1)
    df['Supply_Sweep'] = (df['High'] > df['Prev_High']) & (df['Close'] < df['Prev_High'])
    df['Demand_Sweep'] = (df['Low'] < df['Prev_Low']) & (df['Close'] > df['Prev_Low'])
    return df

def render_liquidity_sweep(selected_name, ticker):
    data = fetch_data(ticker, period="30d", interval="15m")
    if data is None: return st.warning("Intraday data lookup failed.")

    df = calc_liquidity_sweeps(data.copy())
    plot_df = df.tail(80).reset_index()

    fig = go.Figure(data=[go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='Candlestick')])
    supply_idx = plot_df[plot_df['Supply_Sweep']].index
    demand_idx = plot_df[plot_df['Demand_Sweep']].index

    if not supply_idx.empty:
        fig.add_trace(go.Scatter(x=supply_idx, y=plot_df.loc[supply_idx, 'High']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=14, color=CHART_THEME["bearish"]), name='Supply Sweep'))
    if not demand_idx.empty:
        fig.add_trace(go.Scatter(x=demand_idx, y=plot_df.loc[demand_idx, 'Low']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=14, color=CHART_THEME["bullish"]), name='Demand Sweep'))
    
    fig.update_layout(title=f"Intraday Liquidity Sweeps (15m) - {selected_name}", template=CHART_THEME["template"], height=650, xaxis_rangeslider_visible=False)
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    last_regime = "SUPPLY SWEEP" if df['Supply_Sweep'].iloc[-1] else "DEMAND SWEEP" if df['Demand_Sweep'].iloc[-1] else "PRICE DISCOVERY"
    st.metric("Latest Regime", last_regime)

# ========================= 8. Advanced Volatility (Yang-Zhang) =========================
def calc_yang_zhang(o, h, l, c, trading_days):
    N = len(o)
    if N == 0: return 0, 0.0, 0.0
    log_ho = np.log(o / c.shift(1))
    vol_o = log_ho.std() ** 2
    log_co = np.log(c / o)
    vol_c = log_co.std() ** 2
    rs = (np.log(h/o) * np.log(h/c)) + (np.log(l/o) * np.log(l/c))
    vol_rs = rs.mean()
    k = 0.34 / (1.34 + (N + 1) / (N - 1)) if N > 1 else 0
    yz_var = vol_o + k * vol_c + (1 - k) * vol_rs
    yz_vol = np.sqrt(yz_var) * np.sqrt(trading_days) * 100
    c2c_vol = np.log(c / c.shift(1)).std() * np.sqrt(trading_days) * 100
    return N, float(yz_vol), float(c2c_vol)

def render_advanced_volatility(selected_name, ticker, trading_days):
    data = fetch_data(ticker, period="1y")
    if data is None: return st.warning("Data unavailable.")

    df = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    n, yz, c2c = calc_yang_zhang(df['Open'], df['High'], df['Low'], df['Close'], trading_days)
    
    st.subheader(f"Yang-Zhang vs Close-to-Close ({selected_name})")
    st.caption("Yang-Zhang is superior as it accounts for overnight gaps and intraday trend.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Trading Days", n)
    c2.metric("Yang-Zhang Vol (Gap+Intraday)", f"{yz:.2f}%")
    c3.metric("Close-to-Close Vol", f"{c2c:.2f}%", delta=f"{yz - c2c:+.2f}% (Hidden Vol)", delta_color="inverse")

# ========================= 9. Market Synthesis =========================
def render_market_synthesis(selected_name, ticker, asset_class, div1, div2, div1_name, div2_name, currency, trading_days):
    st.subheader(f"🧠 Quantitative Synthesis: {selected_name}")
    
    with st.spinner("Aggregating multi-timeframe risk models (Parallel Fetching)..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_vix = executor.submit(get_vix_data, asset_class, ticker, "6mo")
            future_daily = executor.submit(fetch_data, ticker, "1y")
            future_intra = executor.submit(fetch_data, ticker, "30d", "15m")
            future_div1 = executor.submit(fetch_data, div1, "1y")
            future_div2 = executor.submit(fetch_data, div2, "1y")

            vix_data = future_vix.result()
            daily_data = future_daily.result()
            intra_data = future_intra.result()
            d1_data = future_div1.result()
            d2_data = future_div2.result()

        if any(d is None for d in [vix_data, daily_data, intra_data, d1_data, d2_data]):
            st.error("Incomplete data for full market synthesis. Please check asset selection.")
            return

        current_vix = get_scalar(vix_data['Close'])
        _, _, _, ivr, _, _ = calc_volatility_metrics(vix_data['Close'])
        _, vrp_val, _ = calc_vrp(daily_data['Close'].tail(126), vix_data['Close'].tail(126), trading_days)
        
        log_prices = np.log(daily_data['Close'])
        hurst_series = log_prices.rolling(window=60).apply(calculate_hurst, raw=False).dropna()
        hurst_val = float(hurst_series.iloc[-1])
        hurst_regime = "MEAN REVERTING" if hurst_val < 0.45 else ("TRENDING" if hurst_val > 0.55 else "RANDOM WALK")

        liq_df = calc_liquidity_sweeps(intra_data.copy())
        liq_regime = "SUPPLY SWEEP (Resistance)" if liq_df['Supply_Sweep'].iloc[-1] else ("DEMAND SWEEP (Support)" if liq_df['Demand_Sweep'].iloc[-1] else "PRICE DISCOVERY")

        _, _, _, corr_val, div_regime = calc_divergence(d1_data['Close'], d2_data['Close'], div1_name, div2_name)

    # --- SYNTHESIS LOGIC ---
    bias = 0 
    if hurst_regime == "TRENDING": bias += 1
    if hurst_regime == "MEAN REVERTING": bias -= 1
    if "DEMAND" in liq_regime: bias += 1
    if "SUPPLY" in liq_regime: bias -= 1
    if corr_val < 0.50: bias -= 1
    
    macro_state = "Bullish / Trending Focus" if bias > 0 else ("Bearish / Mean Reversion Focus" if bias < 0 else "Neutral / Choppy")
    option_strategy = "Net Short Premium (Credit Spreads/Iron Condors)" if ivr > 50 and vrp_val > 0 else "Net Long Premium (Debit Spreads/Directional)"

    st.info(f"**Actionable Insight:** The market is currently in a **{macro_state}** state. Based on implied vs. realized volatility pricing, the optimal options approach favors **{option_strategy}**.")

    st.markdown("### 📊 Market Variables Matrix")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Volatility & Pricing**")
        st.metric("IV Rank", f"{ivr:.1f}%", "High Premium" if ivr > 50 else "Low Premium", delta_color="inverse")
        st.metric("VRP Edge", f"{vrp_val:+.2f}%", "Sellers Edge" if vrp_val > 0 else "Buyers Edge", delta_color="normal")
    with col2:
        st.markdown("**2. Market Behavior**")
        st.metric("Hurst Exponent", f"{hurst_val:.3f}", hurst_regime, delta_color="off")
        st.metric("Intraday Liquidity", "Latest Bias", liq_regime, delta_color="off")
    with col3:
        st.markdown("**3. Systemic Risk**")
        st.metric(f"{div1_name}/{div2_name} Corr", f"{corr_val:.2f}", div_regime, delta_color="inverse" if corr_val < 0.50 else "normal")

    # Radar Chart
    categories = ['IV Rank', 'Trend (Hurst)', 'Correlation', 'VRP Premium']
    norm_ivr = min(ivr / 100, 1.0)
    norm_hurst = min(hurst_val, 1.0)
    norm_corr = max(0, min(corr_val, 1.0))
    norm_vrp = max(0, min((vrp_val + 5) / 10, 1.0)) 

    fig = go.Figure(data=go.Scatterpolar(r=[norm_ivr, norm_hurst, norm_corr, norm_vrp], theta=categories, fill='toself', line=dict(color=CHART_THEME["primary"])))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), showlegend=False, template=CHART_THEME["template"], title="Current Regime Profile (Normalized)", height=400)
    add_watermark(fig)
    st.plotly_chart(fig, use_container_width=True)

    # --- DYNAMIC TEXT GENERATOR ---
    st.markdown("### 📝 AI Daily Market Analysis")
    
    div_text = f"The {div1_name} and {div2_name} are moving together, confirming the broader trend." if corr_val > 0.5 else f"The {div2_name} is acting independently from {div1_name}, breaking systemic correlation. This historically leads to choppy, unpredictable price action."
    vix_name = "India VIX" if asset_class == "Indian Equities" else "Synthetic Volatility Index"
    
    vol_text = f"The {vix_name} is currently sitting at {current_vix:.2f}, placing the Volatility Rank (IVR) at {ivr:.1f}%. "
    if vrp_val > 0:
        vol_text += f"Because the Volatility benchmark is higher than short-term Realized Volatility, the Volatility Risk Premium (VRP) is positive (+{vrp_val:.2f}%). The market is overpricing risk, giving sellers a mathematical edge."
    else:
        vol_text += f"Because short-term Realized Volatility is actually higher than the Volatility benchmark, the Volatility Risk Premium (VRP) is negative ({vrp_val:.2f}%). The market is fundamentally underpricing risk, making options/leverage cheap for buyers."

    hurst_text = f"The Hurst Exponent is registering at {hurst_val:.3f}, indicating a {hurst_regime} regime. "
    if hurst_regime == "TRENDING": hurst_text += "Breakouts are likely to succeed; you should follow the momentum."
    elif hurst_regime == "MEAN REVERTING": hurst_text += "Breakouts are highly likely to fail; you should be fading the extremes at support/resistance levels."
    else: hurst_text += "The market is currently a random walk; wait for clearer structural signals."

    sweep_text = f"Intraday micro-structure shows a {liq_regime}. "
    if "SUPPLY" in liq_regime: sweep_text += "Institutional 'smart money' is actively selling into rallies and trapping retail breakout buyers at the highs."
    elif "DEMAND" in liq_regime: sweep_text += "Institutional 'smart money' is actively absorbing panic selling at the lows to drive the market up."
    else: sweep_text += "No major institutional liquidity traps have been triggered in the immediate short-term."

    st.info(f"""
    **Systemic Risk:** The {div1_name}/{div2_name} correlation is currently {corr_val:.2f}. {div_text}\n
    **Volatility & Pricing:** {vol_text}\n
    **Market Behavior:** {hurst_text}\n
    **Intraday Flow:** {sweep_text}\n
    **Conclusion:** Combining these factors, the overarching quantitative bias is **{macro_state}**, and the optimal mathematical approach to derivatives today is **{option_strategy}**.
    """)

# ========================= MAIN APP ROUTER =========================
def main():
    st.title("⚡ Quant ML Master Dashboard v5.0")
    st.markdown("**Multi-Asset Analytics • Institutional Risk Tools**")

    with st.sidebar:
        st.header("Asset Selection")
        asset_class = st.radio("Asset Class", ["Indian Equities", "Crypto"])
        
        if asset_class == "Indian Equities":
            ASSET_DICT = FNO_INDICES
            div1, div2 = "^NSEI", "^NSEBANK"
            div1_name, div2_name = "Nifty 50", "Bank Nifty"
            currency = "₹"
            trading_days = 252
        else:
            ASSET_DICT = CRYPTO_ASSETS
            div1, div2 = "BTC-USD", "ETH-USD"
            div1_name, div2_name = "Bitcoin", "Ethereum"
            currency = "$"
            trading_days = 365 # Crypto trades 24/7
            
        selected_name = st.selectbox("Primary Asset", options=list(ASSET_DICT.keys()), index=0)
        ticker = ASSET_DICT[selected_name]
        
        st.header("Navigation")
        tool = st.radio("Select Tool:", [
            "9. Market Synthesis (Overview)",
            "1. IVR & IVP", "2. Expected Move", "3. Index Divergence",
            "4. Volatility Cone", "5. Volatility Risk Premium",
            "6. Hurst Regime", "7. Liquidity Sweeps", "8. Advanced Volatility (Yang-Zhang)"
        ])

    if tool == "9. Market Synthesis (Overview)": render_market_synthesis(selected_name, ticker, asset_class, div1, div2, div1_name, div2_name, currency, trading_days)
    elif tool == "1. IVR & IVP": render_volatility_metrics(asset_class, ticker)
    elif tool == "2. Expected Move": render_expected_move(selected_name, ticker, asset_class, currency, trading_days)
    elif tool == "3. Index Divergence": render_index_divergence(div1, div2, div1_name, div2_name, currency)
    elif tool == "4. Volatility Cone": render_volatility_cone(selected_name, ticker, trading_days)
    elif tool == "5. Volatility Risk Premium": render_vrp(selected_name, ticker, asset_class, trading_days)
    elif tool == "6. Hurst Regime": render_hurst_regime(selected_name, ticker)
    elif tool == "7. Liquidity Sweeps": render_liquidity_sweep(selected_name, ticker)
    elif tool == "8. Advanced Volatility (Yang-Zhang)": render_advanced_volatility(selected_name, ticker, trading_days)

    st.divider()
    st.caption("v5.0 • Multi-Asset Expansion • Synthetic Crypto IV • Dynamic Daily AI Reporting")

if __name__ == "__main__":
    main()