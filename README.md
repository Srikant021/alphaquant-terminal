# AlphaQuant Terminal - Modular Python Quantitative Trading Dashboard

A refactored and modularized version of the AlphaQuant Terminal with:
- ✅ Version-pinned dependencies
- ✅ Configuration management (.env)
- ✅ Modular architecture (500+ lines → modules)
- ✅ Real Fyers API integration for options data
- ✅ Comprehensive unit tests for Hurst exponent
- ✅ Logging framework
- ✅ Improved error handling

## 📂 Project Structure

```
alphaquant-terminal/
├── alphaquant/              # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── logger.py           # Logging setup
│   ├── data.py            # Data fetching (yfinance)
│   ├── indicators.py      # Technical indicators
│   ├── ml.py              # ML models
│   ├── charts.py          # Visualization functions
│   └── fyers_integration.py # Real options data
├── tests/
│   ├── conftest.py        # Pytest fixtures
│   └── test_hurst.py      # Hurst exponent tests
├── app.py                 # Streamlit main app (~250 lines)
├── .env.example           # Configuration template
├── requirements.txt       # Version-pinned dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/Srikant021/alphaquant-terminal.git
cd alphaquant-terminal

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

### 2. Configure

Edit `.env`:

```env
DEFAULT_MARKET=Crypto
RSI_PERIOD=14
BOLLINGER_PERIOD=20

# Optional: Fyers API for real options data
FYERS_API_ENABLED=false
FYERS_API_KEY=your_key_here
FYERS_API_SECRET=your_secret_here
FYERS_CLIENT_ID=your_client_id_here
```

### 3. Run

```bash
# Start Streamlit app
streamlit run app.py

# Run tests
pytest tests/ -v --cov=alphaquant
```

## 📊 Features

### Technical Indicators
- RSI (Relative Strength Index)
- Bollinger Bands
- MACD
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)
- Hurst Exponent (market regime detection)
- Liquidity Sweeps (supply/demand level analysis)

### Advanced Metrics
- IV Rank & IV Percentile
- Parkinson Volatility
- Volatility Cone
- Expected Move Calculator

### Machine Learning
- Random Forest classifier for price prediction
- Probability calibration
- Feature importance analysis
- Explainable predictions with factor breakdown
- Rolling performance metrics (accuracy, precision, recall)

### Options Analysis
- Open Interest profiles (synthetic + Fyers real data)
- Max Pain calculation
- Greeks analysis (when using Fyers)

## 🔧 Configuration

All settings in `.env` (see `.env.example` for all options):

```python
# Indicators
RSI_PERIOD=14
BOLLINGER_PERIOD=20
BOLLINGER_STD_DEV=2.0
ATR_PERIOD=14

# ML Model
ML_N_ESTIMATORS=150
ML_MAX_DEPTH=6
ML_CALIBRATE=true

# Hurst Exponent
HURST_MIN_DATA=100
HURST_TREND_THRESHOLD=0.58
HURST_MEAN_REV_THRESHOLD=0.42

# Data
DATA_FETCH_RETRY_ATTEMPTS=3
DATA_CACHE_TTL_LIVE=300
DATA_CACHE_TTL_HISTORY=3600

# Fyers API (optional)
FYERS_API_ENABLED=false
FYERS_API_KEY=...
FYERS_API_SECRET=...
```

## 📦 Dependencies

All pinned to specific versions in `requirements.txt`:

```
streamlit==1.28.0
scikit-learn==1.3.2
yfinance==0.2.32
pandas==2.1.3
numpy==1.26.2
plotly==5.18.0
python-dotenv==1.0.0
fyers-apiv3==2.1.13
pytest==7.4.3
```

## 🧪 Testing

Comprehensive test suite for Hurst exponent:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=alphaquant --cov-report=html

# Run specific test
pytest tests/test_hurst.py::TestHurstExponent::test_hurst_trending_market -v
```

### Test Coverage

- ✅ Hurst calculation with various data points
- ✅ Trending market detection
- ✅ Mean-reverting market detection
- ✅ Random walk identification
- ✅ Insufficient data handling
- ✅ NaN value handling
- ✅ Confidence scoring accuracy
- ✅ Return type validation
- ✅ Reproducibility tests

## 🏗️ Modular Architecture

### Benefits of Refactoring

| Aspect | Before | After |
|--------|--------|-------|
| **Main file** | 1500 lines | 250 lines |
| **Maintainability** | Monolithic | Modular |
| **Testing** | No tests | 16+ unit tests |
| **Configuration** | Hard-coded | .env based |
| **Reusability** | Limited | High |
| **Error handling** | Generic | Specific |
| **Logging** | Basic | Comprehensive |

### Module Breakdown

**alphaquant/config.py** (80 lines)
- Centralized settings
- Environment variable loading
- Configuration validation

**alphaquant/logger.py** (50 lines)
- Rotating file logging
- Console logging
- Log formatting

**alphaquant/data.py** (150 lines)
- yfinance wrapper
- Retry logic with backoff
- Caching decorators
- Error handling

**alphaquant/indicators.py** (400 lines)
- Technical indicators
- Hurst exponent
- Liquidity sweep detection
- Volatility metrics

**alphaquant/ml.py** (200 lines)
- Random Forest training
- Feature engineering
- Explainability functions
- Model serialization

**alphaquant/charts.py** (300 lines)
- Price charts
- Volatility visualization
- Hurst trend charts
- Plotly styling

**alphaquant/fyers_integration.py** (100 lines)
- Fyers API wrapper
- Option chain fetching
- Max pain calculation

**app.py** (250 lines)
- Streamlit UI
- Data flow orchestration
- Sidebar controls
- Metric displays

## 🔌 Fyers API Integration

Real options data instead of synthetic:

```python
from alphaquant.fyers_integration import get_fyers_client

# Initialize
fyers = get_fyers_client()

# Get option chain
chain = fyers.get_option_chain("NIFTY50", expiry="21 DEC 2023")

# Get max pain
max_pain = fyers.get_max_pain("NIFTY50", expiry="21 DEC 2023")

# Get OI by strike
oi_data = fyers.get_open_interest_by_strike("BANKNIFTY")
```

### Setup Fyers API

1. Register at https://fyers.in/
2. Get API credentials (key, secret, client ID)
3. Add to `.env`:
   ```env
   FYERS_API_ENABLED=true
   FYERS_API_KEY=your_key
   FYERS_API_SECRET=your_secret
   FYERS_CLIENT_ID=your_client_id
   ```
4. Uncomment Fyers code in app.py

## 📈 Usage Example

```python
# Import modules
from alphaquant.data import fetch_data, get_live_price
from alphaquant.indicators import hurst_exponent, compute_rsi
from alphaquant.ml import train_ml_model, build_ml_features

# Fetch data
data = fetch_data("BTC-USD", period="1y", interval="1d")
live_price = get_live_price("BTC-USD")

# Calculate Hurst
h, interpretation, confidence = hurst_exponent(data['Close'])
print(f"Hurst: {h:.3f} ({interpretation}) - Confidence: {confidence}")

# Train ML model
model, scaler, accuracy, cols = train_ml_model("BTC-USD")
print(f"Model accuracy: {accuracy:.2f}%")

# Build features
features = build_ml_features(data)
print(features.head())
```

## 🎯 Next Steps

- [ ] Add backtesting framework (backtrader)
- [ ] Implement position sizing strategies
- [ ] Add stop-loss/take-profit helpers
- [ ] Connect to live brokers (Fyers, Zerodha)
- [ ] Add alert system (email, Telegram)
- [ ] Create web API (FastAPI)
- [ ] Add Docker support
- [ ] Implement hyperparameter tuning

## 📝 License

MIT License - see LICENSE file

## 👨‍💻 Author

AlphaQuant Team

## 📞 Support

Issues & PRs welcome on GitHub!

---

**Made with ❤️ for quantitative traders**
