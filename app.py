"""
FINANCIAL ANALYSIS SUITE WITH ML & CRYPTO
Professional Trading Dashboard with Quantitative Tools + ML Predictions
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import json
from pathlib import Path

# PyQt5 GUI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QLabel, QPushButton, 
                             QMessageBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

warnings.filterwarnings('ignore')

# ==================== ML PREDICTION ENGINE ====================
class VolatilityMLPredictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=100)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, price_data, lookback=30):
        if len(price_data) < lookback + 5:
            return None, None
            
        returns = np.log(price_data / price_data.shift(1)).dropna()
        # Clean infinite values which break StandardScaler
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        features, targets = [], []
        
        for i in range(len(returns) - lookback - 5):
            window = returns.iloc[i:i+lookback].values
            mean_ret = np.mean(window)
            std_ret = np.std(window)
            
            features.append([
                std_ret,                                      
                mean_ret,                                     
                np.max(window),                               
                np.min(window),                               
                std_ret / (abs(mean_ret) + 1e-8),             
                np.abs(window).mean()                         
            ])
            targets.append(np.std(returns.iloc[i+lookback:i+lookback+5].values))
        
        return np.nan_to_num(np.array(features)), np.array(targets)
    
    def train(self, price_data):
        X, y = self.prepare_features(price_data)
        if X is None or len(X) == 0:
            return False
            
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
            except Exception as e:
                print(f"Failed to train {name}: {e}")
        
        self.is_trained = True
        return True
    
    def predict(self, price_data):
        if not self.is_trained: return None
            
        X, _ = self.prepare_features(price_data)
        if X is None or len(X) == 0: return None
            
        X_scaled = self.scaler.transform(X[-1:])
        
        predictions = []
        for model in self.models.values():
            try:
                predictions.append(model.predict(X_scaled)[0])
            except:
                continue
        
        return np.mean(predictions) if predictions else None

class TrendMLPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, price_data, lookback=20):
        if len(price_data) < lookback + 5: return None, None
            
        df = pd.DataFrame(price_data)
        df.columns = ['close']
        
        df['returns'] = df['close'].pct_change()
        df['sma20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        df['vol'] = df['returns'].rolling(20).std()
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        features, targets = [], []
        for i in range(len(df) - lookback - 5):
            f = [
                (df['close'].iloc[i+lookback] / df['close'].iloc[i]) - 1,  
                (df['sma20'].iloc[i+lookback] / df['close'].iloc[i+lookback]) - 1,
                df['rsi'].iloc[i+lookback] / 50,
                df['macd'].iloc[i+lookback],
                df['vol'].iloc[i+lookback]
            ]
            features.append(f)
            targets.append((df['close'].iloc[i+lookback+5] / df['close'].iloc[i+lookback]) - 1)
        
        return np.nan_to_num(np.array(features)), np.array(targets)
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(prices, fast=12, slow=26):
        return prices.ewm(span=fast).mean() - prices.ewm(span=slow).mean()
    
    def train(self, price_data):
        X, y = self.prepare_features(price_data)
        if X is None or len(X) == 0: return False
            
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return True
    
    def predict(self, price_data):
        if not self.is_trained: return None
        X, _ = self.prepare_features(price_data)
        if X is None or len(X) == 0: return None
        X_scaled = self.scaler.transform(X[-1:])
        return self.model.predict(X_scaled)[0]

# ==================== DATA FETCHING WORKER ====================
class DataFetcher(QThread):
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def run(self):
        try:
            print("Fetching market & crypto data in background...")
            tickers = {
                'nifty': '^NSEI',
                'vix': '^INDIAVIX',
                'bank': '^NSEBANK',
                'btc': 'BTC-USD',
                'eth': 'ETH-USD'
            }
            results = {}
            for name, ticker in tickers.items():
                data = yf.download(ticker, period="1y", progress=False)
                if data.empty:
                    raise Exception(f"No data for {ticker}")
                results[name] = data
            
            self.data_ready.emit(results)
        except Exception as e:
            self.error_occurred.emit(f"Error fetching data: {str(e)}")

# ==================== ANALYSIS MODULES ====================
class FinancialAnalyzer:
    @staticmethod
    def calculate_ivr_ivp(vix_data):
        close_prices = vix_data['Close'].squeeze()
        current_iv = float(close_prices.iloc[-1])
        high_52w = float(close_prices.max())
        low_52w = float(close_prices.min())
        
        ivr = ((current_iv - low_52w) / (high_52w - low_52w + 1e-8)) * 100
        days_below = (close_prices < current_iv).sum()
        ivp = (days_below / len(close_prices)) * 100
        
        return {'current_iv': current_iv, 'ivr': ivr, 'ivp': ivp, 'high_52w': high_52w, 'low_52w': low_52w}
    
    @staticmethod
    def calculate_expected_move(nifty_close, vix_value):
        spot_price = float(nifty_close.iloc[-1])
        daily_volatility = (vix_value / 100) * np.sqrt(1/365)
        expected_move = spot_price * daily_volatility
        return {'spot_price': spot_price, 'expected_move': expected_move, 'upper_bound': spot_price + expected_move, 'lower_bound': spot_price - expected_move}
    
    @staticmethod
    def calculate_parkinson(high, low):
        log_hl = np.log(high / low).dropna()
        if log_hl.empty: return 0.0
        constant = 1 / (4 * len(log_hl) * np.log(2))
        parkinson_var = constant * (log_hl ** 2).sum()
        return float(np.sqrt(parkinson_var) * np.sqrt(252) * 100)

# ==================== MAIN GUI APPLICATION ====================
class FinancialDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIFTY & Crypto Analysis Suite")
        self.setGeometry(100, 100, 1400, 900)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; color: #c9d1d9; }
            QTabBar::tab { background-color: #161b22; color: #8b949e; padding: 8px 20px; border: 1px solid #30363d; }
            QTabBar::tab:selected { background-color: #0d1117; color: #58a6ff; border-bottom: 2px solid #58a6ff; }
            QPushButton { background-color: #238636; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; }
            QPushButton:hover { background-color: #2ea043; }
            QLabel { color: #c9d1d9; }
            QTableWidget { background-color: #0d1117; color: #c9d1d9; gridline-color: #30363d; }
        """)
        
        self.data_dict = {}
        self.analyzer = FinancialAnalyzer()
        self.vol_predictor = VolatilityMLPredictor()
        self.trend_predictor = TrendMLPredictor()
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.create_dashboard_tab()
        self.create_ivr_ivp_tab()
        self.create_expected_move_tab()
        self.create_correlation_tab()
        self.create_volatility_cone_tab()
        self.create_vrp_tab()
        self.create_hurst_tab()
        self.create_ml_prediction_tab()
        self.create_summary_tab()
        
        # Background data fetch to prevent UI freeze
        self.load_all_data()
        
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_all_data)
        self.refresh_timer.start(300000) 
    
    def create_dashboard_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        header = QLabel("Financial Analysis Suite (NIFTY & Crypto)")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setStyleSheet("color: #58a6ff;")
        layout.addWidget(header)
        
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #ffa657; font-size: 14px;")
        status_layout.addWidget(self.status_label)
        
        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.clicked.connect(self.load_all_data)
        status_layout.addWidget(refresh_btn)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        metrics_layout = QHBoxLayout()
        self.metric_labels = {}
        metrics = ['Nifty Price', 'Bank Nifty', 'Bitcoin', 'Ethereum', 'VIX', 'Expected Move']
        colors = ['#58a6ff', '#85e89d', '#f7931a', '#627eea', '#ffa657', '#ff7b72']
        
        for metric, color in zip(metrics, colors):
            label = QLabel(f"{metric}\n--")
            label.setStyleSheet(f"color: {color}; font-size: 14px; padding: 15px; border: 1px solid #30363d; border-radius: 6px; font-weight:bold;")
            metrics_layout.addWidget(label)
            self.metric_labels[metric] = label
        
        layout.addLayout(metrics_layout)
        layout.addStretch()
        widget.setLayout(layout)
        self.tabs.addTab(widget, "📊 Dashboard")
        
    def create_ivr_ivp_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.ivr_chart = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.ivr_chart)
        self.ivr_info = QLabel()
        self.ivr_info.setStyleSheet("background-color: #161b22; padding: 15px; border-radius: 6px;")
        layout.addWidget(self.ivr_info)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "📈 IVR & IVP")

    def create_expected_move_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.exp_move_chart = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.exp_move_chart)
        self.exp_move_info = QLabel()
        self.exp_move_info.setStyleSheet("background-color: #161b22; padding: 15px; border-radius: 6px;")
        layout.addWidget(self.exp_move_info)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "🎯 Expected Move")

    def create_correlation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        title = QLabel("Multi-Asset Correlation (Nifty, BankNifty, BTC, ETH)")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #58a6ff;")
        layout.addWidget(title)
        
        self.corr_chart = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.corr_chart)
        self.corr_info = QLabel()
        self.corr_info.setStyleSheet("background-color: #161b22; padding: 15px; border-radius: 6px;")
        layout.addWidget(self.corr_info)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "🔗 Correlation")

    def create_volatility_cone_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.vol_cone_chart = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.vol_cone_chart)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "📊 Volatility Cone")

    def create_vrp_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.vrp_chart = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.vrp_chart)
        self.vrp_info = QLabel()
        self.vrp_info.setStyleSheet("background-color: #161b22; padding: 15px; border-radius: 6px;")
        layout.addWidget(self.vrp_info)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "⚡ VRP")

    def create_hurst_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.hurst_chart = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.hurst_chart)
        self.hurst_info = QLabel()
        self.hurst_info.setStyleSheet("background-color: #161b22; padding: 15px; border-radius: 6px;")
        layout.addWidget(self.hurst_info)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "🌊 Hurst Exponent")

    def create_ml_prediction_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.ml_info = QLabel()
        self.ml_info.setStyleSheet("background-color: #161b22; padding: 15px; border-radius: 6px;")
        layout.addWidget(self.ml_info)
        self.ml_chart = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.ml_chart)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "🤖 ML Predictions")

    def create_summary_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.setStyleSheet("background-color: #0d1117; color: #c9d1d9; gridline-color: #30363d;")
        layout.addWidget(self.summary_table)
        
        export_btn = QPushButton("📥 Export Report to JSON")
        export_btn.clicked.connect(self.export_report)
        layout.addWidget(export_btn)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "📋 Summary")

    def load_all_data(self):
        self.status_label.setText("Fetching market & crypto data in background...")
        self.status_label.setStyleSheet("color: #ffa657;")
        
        # Start background thread to prevent UI freezing
        self.fetcher = DataFetcher()
        self.fetcher.data_ready.connect(self.on_data_fetched)
        self.fetcher.error_occurred.connect(self.on_fetch_error)
        self.fetcher.start()
        
    def on_fetch_error(self, err_msg):
        self.status_label.setText(f"❌ {err_msg}")
        self.status_label.setStyleSheet("color: #ff7b72;")

    def on_data_fetched(self, data):
        self.data_dict = data
        try:
            self.update_all_charts()
            self.update_summary()
            self.status_label.setText(f"✅ Data loaded at {datetime.now().strftime('%H:%M:%S')}")
            self.status_label.setStyleSheet("color: #85e89d;")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.on_fetch_error(f"Error processing data: {e}")

    def update_all_charts(self):
        nifty = self.data_dict['nifty']['Close'].squeeze()
        vix = self.data_dict['vix']
        
        ivr_data = self.analyzer.calculate_ivr_ivp(vix)
        self.plot_ivr_ivp(ivr_data)
        
        exp_move = self.analyzer.calculate_expected_move(nifty, ivr_data['current_iv'])
        self.plot_expected_move(exp_move)
        
        self.plot_correlation()
        self.plot_vrp()
        
        self.train_and_plot_ml()
        self.update_key_metrics(ivr_data, exp_move)

    def update_key_metrics(self, ivr_data, exp_move):
        nifty_px = float(self.data_dict['nifty']['Close'].iloc[-1])
        bank_px = float(self.data_dict['bank']['Close'].iloc[-1])
        btc_px = float(self.data_dict['btc']['Close'].iloc[-1])
        eth_px = float(self.data_dict['eth']['Close'].iloc[-1])
        
        self.metric_labels['Nifty Price'].setText(f"Nifty\n{nifty_px:,.0f}")
        self.metric_labels['Bank Nifty'].setText(f"Bank Nifty\n{bank_px:,.0f}")
        self.metric_labels['Bitcoin'].setText(f"BTC-USD\n${btc_px:,.2f}")
        self.metric_labels['Ethereum'].setText(f"ETH-USD\n${eth_px:,.2f}")
        self.metric_labels['VIX'].setText(f"VIX\n{ivr_data['current_iv']:.1f}")
        self.metric_labels['Expected Move'].setText(f"Nifty Exp Move\n±{exp_move['expected_move']:.0f}")

    def plot_ivr_ivp(self, data):
        self.ivr_chart.axes.clear()
        vix_close = self.data_dict['vix']['Close'].tail(252).squeeze()
        x = range(len(vix_close))
        self.ivr_chart.axes.plot(x, vix_close.values, color='#00ffff', linewidth=2)
        self.ivr_chart.axes.axhline(data['current_iv'], color='white')
        self.ivr_chart.axes.set_title('VIX (1Y)', color='#c9d1d9')
        self.ivr_chart.axes.set_facecolor('#0d1117')
        self.ivr_chart.draw()
        self.ivr_info.setText(f"<b>Current VIX:</b> {data['current_iv']:.2f} | <b>IVR:</b> {data['ivr']:.1f}%")

    def plot_expected_move(self, data):
        self.exp_move_chart.axes.clear()
        nifty_close = self.data_dict['nifty']['Close'].tail(15).squeeze()
        x = range(len(nifty_close))
        self.exp_move_chart.axes.plot(x, nifty_close.values, color='#00ffff', marker='o')
        self.exp_move_chart.axes.scatter([len(x)], [data['upper_bound']], color='#85e89d')
        self.exp_move_chart.axes.scatter([len(x)], [data['lower_bound']], color='#ff7b72')
        self.exp_move_chart.axes.set_facecolor('#0d1117')
        self.exp_move_chart.draw()
        self.exp_move_info.setText(f"<b>Target Range:</b> {data['lower_bound']:.0f} to {data['upper_bound']:.0f}")

    def plot_correlation(self):
        self.corr_chart.axes.clear()
        
        # ALIGN ALL DATA BY DATE TO FIX VALUE ERROR
        df = pd.concat([
            self.data_dict['nifty']['Close'],
            self.data_dict['bank']['Close'],
            self.data_dict['btc']['Close'],
            self.data_dict['eth']['Close']
        ], axis=1).dropna()
        df.columns = ['Nifty', 'BankNifty', 'BTC', 'ETH']
        df = df.tail(252)

        df_norm = (df / df.iloc[0]) * 100
        x = range(len(df_norm))
        
        self.corr_chart.axes.plot(x, df_norm['Nifty'].values, color='#00ffff', label='Nifty')
        self.corr_chart.axes.plot(x, df_norm['BankNifty'].values, color='#85e89d', label='Bank Nifty')
        self.corr_chart.axes.plot(x, df_norm['BTC'].values, color='#f7931a', label='Bitcoin')
        self.corr_chart.axes.plot(x, df_norm['ETH'].values, color='#627eea', label='Ethereum')
        
        self.corr_chart.axes.legend(facecolor='#161b22', edgecolor='#30363d')
        self.corr_chart.axes.set_title('Normalized Performance (1Y)', color='#c9d1d9')
        self.corr_chart.axes.set_facecolor('#0d1117')
        self.corr_chart.draw()
        
        corr_matrix = df.pct_change().corr()
        btc_nifty_corr = corr_matrix.loc['Nifty', 'BTC']
        self.corr_info.setText(f"<b>Nifty to Bitcoin Correlation:</b> {btc_nifty_corr:.2f}")

    def plot_vrp(self):
        self.vrp_chart.axes.clear()
        log_returns = np.log(self.data_dict['nifty']['Close'] / self.data_dict['nifty']['Close'].shift(1))
        hv = log_returns.rolling(window=20).std() * np.sqrt(252) * 100
        vix = self.data_dict['vix']['Close'].squeeze()
        
        df = pd.DataFrame({'vix': vix, 'hv': hv}).dropna().tail(180)
        x = range(len(df))
        self.vrp_chart.axes.plot(x, df['vix'].values, color='#00ffff', label='VIX')
        self.vrp_chart.axes.plot(x, df['hv'].values, color='#ffa657', label='HV')
        self.vrp_chart.axes.set_facecolor('#0d1117')
        self.vrp_chart.draw()

    def train_and_plot_ml(self):
        try:
            nifty = self.data_dict['nifty']['Close']
            self.vol_predictor.train(nifty)
            self.trend_predictor.train(nifty)
            
            vol_pred = self.vol_predictor.predict(nifty)
            trend_pred = self.trend_predictor.predict(nifty)
            
            self.ml_chart.axes.clear()
            categories = ['Vol Prediction', '5-day Trend']
            values = [vol_pred * 100 if vol_pred else 0, (trend_pred * 100) if trend_pred else 0]
            colors = ['#85e89d' if v > 0 else '#ff7b72' for v in values]
            
            self.ml_chart.axes.bar(categories, values, color=colors, alpha=0.8, width=0.6)
            self.ml_chart.axes.set_facecolor('#0d1117')
            self.ml_chart.draw()
            self.ml_info.setText(f"<b>Model Trained!</b> Next Volatility: {values[0]:.2f}%, Expected 5-Day Move: {values[1]:+.2f}%")
        except Exception as e:
            self.ml_info.setText(f"ML Error: {e}")

    def update_summary(self):
        metrics = {}
        for asset, data in self.data_dict.items():
            if data.empty: continue
            current = float(data['Close'].iloc[-1])
            # Fix IndexError: Use iloc[0] for the oldest loaded day instead of iloc[-252]
            ytd = float(data['Close'].iloc[0]) 
            ret = ((current / ytd) - 1) * 100
            metrics[f'{asset.upper()} Current Price'] = f"{current:,.2f}"
            metrics[f'{asset.upper()} 1-Year Return'] = f"{ret:+.2f}%"

        self.summary_table.setRowCount(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            self.summary_table.setItem(i, 0, QTableWidgetItem(key))
            self.summary_table.setItem(i, 1, QTableWidgetItem(value))
        self.summary_table.resizeColumnsToContents()

    def export_report(self):
        try:
            report = { 'timestamp': datetime.now().isoformat() }
            for asset, data in self.data_dict.items():
                report[asset] = {
                    'price': float(data['Close'].iloc[-1]),
                    'high_52w': float(data['Close'].max())
                }
            
            # Save locally instead of hardcoded linux path
            report_path = Path.cwd() / 'analysis_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            QMessageBox.information(self, "Success", f"Report exported to {report_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0d1117')
        self.axes = self.fig.add_subplot(111)
        self.axes.tick_params(colors='#c9d1d9')
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dashboard = FinancialDashboard()
    dashboard.show()
    sys.exit(app.exec_())