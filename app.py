# ==============================================================================
# ALADDIN QUANT TERMINAL v30.0 — COMPLETE INSTITUTIONAL ENGINE
# ==============================================================================

import os
import re
import time
import yaml
import logging
import threading
import warnings
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
import concurrent.futures

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Fyers API
try:
    from fyers_apiv3.FyersWebsocket import data_ws
    from fyers_apiv3 import fyersModel
    HAS_FYERS = True
except ImportError:
    HAS_FYERS = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================== CONFIGURATION ==============================
class Config:
    DEFAULTS = {
        'thresholds': {
            'ivr_high': 50.0,
            'hurst_trend': 0.55,
            'hurst_mean_revert': 0.45,
            'corr_high': 0.80,
            'corr_divergence': 0.50,
            'vrp_sellers': 0.0,
        },
        'retry': {
            'max_attempts': 3,
            'backoff_factor': 2.0,
        }
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.DEFAULTS.copy()
        self._load_yaml(config_path)
        self._apply_env_overrides()

    def _load_yaml(self, path: str) -> None:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    user_config = yaml.safe_load(f)
                if user_config: self._deep_update(self.config, user_config)
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")

    def _apply_env_overrides(self) -> None:
        for section, params in self.config.items():
            if not isinstance(params, dict): continue
            for key, val in params.items():
                env_key = f"ALADDIN_{section.upper()}_{key.upper()}"
                if env_key in os.environ:
                    try:
                        default_type = type(self.DEFAULTS.get(section, {}).get(key, val))
                        self.config[section][key] = default_type(os.environ[env_key])
                    except Exception as e:
                        logger.warning(f"Could not apply env {env_key}: {e}")

    def _deep_update(self, d: dict, u: dict) -> None:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def get(self, section: str, key: str) -> Any:
        return self.config.get(section, {}).get(key)

CONFIG = Config()

# ============================== CONSTANTS & THEMES ==============================
CHART_THEME = {
    "template": "plotly_dark",
    "primary": "#00FFFF",
    "secondary": "#FFA500",
    "bullish": "#00FF00",
    "bearish": "#FF3333",
    "neutral": "#FFFFFF",
    "accent": "#A78BFA",
    "watermark": "#334155"
}

INDIAN_ASSETS = {
    "Nifty 50 (Index)": "^NSEI",
    "Bank Nifty (Index)": "^NSEBANK",
    "FinNifty (Index)": "NIFTY_FIN_SERVICE.NS",
    "Nifty IT (Index)": "^CNXIT",
    "Nifty Auto (Index)": "^CNXAUTO",
    "Nifty Metal (Index)": "^CNXMETAL",
    "Sensex (Index)": "^BSESN",
    "Reliance Ind (Stock)": "RELIANCE.NS",
    "HDFC Bank (Stock)": "HDFCBANK.NS",
    "Infosys (Stock)": "INFY.NS",
    "TCS (Stock)": "TCS.NS",
    "ICICI Bank (Stock)": "ICICIBANK.NS"
}

CRYPTO_ASSETS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD"
}

FYERS_SYMBOLS = {
    "NSE:NIFTY50-INDEX": "^NSEI",
    "NSE:NIFTYBANK-INDEX": "^NSEBANK",
    "NSE:FINNIFTY-INDEX": "NIFTY_FIN_SERVICE.NS",
    "NSE:CNXIT-INDEX": "^CNXIT",
    "NSE:CNXAUTO-INDEX": "^CNXAUTO",
    "NSE:CNXMETAL-INDEX": "^CNXMETAL",
    "BSE:SENSEX-INDEX": "^BSESN",
    "NSE:RELIANCE-EQ": "RELIANCE.NS",
    "NSE:HDFCBANK-EQ": "HDFCBANK.NS",
    "NSE:INFY-EQ": "INFY.NS",
    "NSE:TCS-EQ": "TCS.NS",
    "NSE:ICICIBANK-EQ": "ICICIBANK.NS"
}

# ============================== FYERS WEBSOCKET ENGINE ==============================
def onmessage(message):
    for tick in message:
        if 'symbol' in tick:
            fyers_symbol = tick['symbol']
            if fyers_symbol in FYERS_SYMBOLS:
                ticker_symbol = FYERS_SYMBOLS[fyers_symbol]
                st.session_state[f"live_tick_{ticker_symbol}"] = tick
                if f"live_history_{ticker_symbol}" not in st.session_state:
                    st.session_state[f"live_history_{ticker_symbol}"] = []
                st.session_state[f"live_history_{ticker_symbol}"].append({
                    'Time': datetime.now(),
                    'Close': tick.get('ltp', 0),
                    'High': tick.get('high_price', tick.get('ltp', 0)),
                    'Low': tick.get('low_price', tick.get('ltp', 0)),
                    'Volume': tick.get('vol_traded_today', 0)
                })
                st.session_state[f"live_history_{ticker_symbol}"] = st.session_state[f"live_history_{ticker_symbol}"][-500:]

def onerror(message):
    logger.error(f"Fyers WebSocket Error: {message}")

def onclose(message):
    logger.info("Fyers WebSocket Connection Closed")

def onopen():
    symbols = list(FYERS_SYMBOLS.keys())
    if 'fyers_ws' in st.session_state:
        st.session_state.fyers_ws.subscribe(symbol=symbols, data_type="SymbolUpdate")

def start_fyers_websocket(client_id: str, access_token: str):
    ws_token = f"{client_id}:{access_token}"
    fyers_ws = data_ws.FyersDataSocket(
        access_token=ws_token,
        log_path="",
        litemode=False,
        write_to_file=False,
        reconnect=True,
        on_connect=onopen,
        on_close=onclose,
        on_error=onerror,
        on_message=onmessage
    )
    st.session_state.fyers_ws = fyers_ws
    fyers_ws.connect()

# ============================== QUANTITATIVE MATHEMATICS ==============================
def get_pivots(high: float, low: float, close: float) -> Tuple[float, float, float]:
    pivot = (high + low + close) / 3.0
    r1 = (2.0 * pivot) - low
    s1 = (2.0 * pivot) - high
    return pivot, r1, s1

def frac_diff_series(series: pd.Series, d: float = 0.4, window: int = 20) -> pd.Series:
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights)[::-1]
    
    diff = np.convolve(series, weights, mode='valid')
    padded = np.empty_like(series)
    padded[:] = np.nan
    padded[window-1:] = diff
    return pd.Series(padded, index=series.index)

def calculate_hurst(ts: pd.Series) -> float:
    if len(ts) < 20:
        return np.nan
    lags = range(2, 20)
    tau = [lag for lag in lags]
    ts_arr = ts.values
    reg = [np.std(ts_arr[lag:] - ts_arr[:-lag]) for lag in lags]
    try:
        poly = np.polyfit(np.log(tau), np.log(reg), 1)
        return poly[0]
    except Exception:
        return np.nan

def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
    if T <= 0.0 or sigma <= 0.0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T) / 100.0

    if option_type.lower() == 'call':
        delta = si.norm.cdf(d1)
        theta = (- (S * sigma * si.norm.pdf(d1)) / (2.0 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365.0
    else:
        delta = si.norm.cdf(d1) - 1.0
        theta = (- (S * sigma * si.norm.pdf(d1)) / (2.0 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365.0

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

# ============================== DATA FETCHERS & NSE INGESTION ==============================
def markdown_to_html(text: str) -> str:
    return re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)

def safe_get_scalar(series: Union[pd.Series, float, int, None], default: float = 0.0) -> float:
    if series is None: return default
    if isinstance(series, pd.Series):
        if series.empty: return default
        val = series.iloc[-1]
    else: val = series
    try:
        if hasattr(val, 'item'): val = val.item()
        val = float(val)
    except (ValueError, TypeError): val = default
    if np.isnan(val) or np.isinf(val): val = default
    return val

def add_watermark(fig: go.Figure) -> None:
    fig.add_annotation(
        text="ALADDIN QUANT TERMINAL v30.0",
        xref="paper", yref="paper", x=0.99, y=0.01,
        showarrow=False, font=dict(size=9, color=CHART_THEME["watermark"]), opacity=0.5
    )

def section_header(number: str, title: str, icon: str = "◈") -> None:
    st.markdown(f"""
