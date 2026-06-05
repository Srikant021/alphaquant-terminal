"""
pytest configuration and fixtures.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_price_series():
    """Generate sample price series for testing."""
    np.random.seed(42)
    prices = np.cumprod(1 + np.random.randn(300) * 0.01)
    return pd.Series(prices)


@pytest.fixture
def trending_series():
    """Generate trending price series."""
    np.random.seed(42)
    trend = np.linspace(100, 150, 300)
    noise = np.random.randn(300) * 2
    return pd.Series(trend + noise)


@pytest.fixture
def mean_reverting_series():
    """Generate mean-reverting price series."""
    np.random.seed(42)
    mean = 100
    prices = [mean]
    for _ in range(299):
        deviation = prices[-1] - mean
        next_price = prices[-1] - 0.2 * deviation + np.random.randn() * 2
        prices.append(max(next_price, 50))
    return pd.Series(prices)
