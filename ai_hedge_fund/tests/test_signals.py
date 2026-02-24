"""
Basic unit tests for signal pipeline.
Run: pytest tests/
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
import datetime


def make_fake_ohlcv(n=200) -> pd.DataFrame:
    dates  = pd.date_range("2023-01-01", periods=n, freq="B")
    close  = 150 + np.cumsum(np.random.randn(n) * 0.5)
    open_  = close * (1 + np.random.randn(n) * 0.001)
    high   = np.maximum(open_, close) * (1 + abs(np.random.randn(n)) * 0.003)
    low    = np.minimum(open_, close) * (1 - abs(np.random.randn(n)) * 0.003)
    vol    = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=dates)


class TestFeatures:
    def test_build_features_shape(self):
        from src.signals.features import build_features
        df = make_fake_ohlcv(200)
        features = build_features(df)
        assert len(features) == len(df)
        assert features.shape[1] > 20, "Expected many features"

    def test_no_lookahead_bias(self):
        from src.signals.features import build_features
        df = make_fake_ohlcv(200)
        f1 = build_features(df.iloc[:100])
        f2 = build_features(df)
        # First 100 rows should be identical (no future data leakage)
        common_cols = [c for c in f1.columns if c in f2.columns]
        for col in common_cols[:5]:
            pd.testing.assert_series_equal(
                f1[col].dropna().iloc[-10:],
                f2[col].iloc[:100].dropna().iloc[-10:],
                check_names=False,
                rtol=1e-3
            )


class TestQuantSignal:
    def test_returns_valid_signal(self):
        from src.signals.features import build_features
        from src.signals.quant_signal import compute_quant_signal
        df = make_fake_ohlcv(200)
        features = build_features(df)
        signal = compute_quant_signal(features, "TEST")
        assert signal.direction in [-1.0, 0.0, 1.0]
        assert 0.0 <= signal.conviction <= 1.0

    def test_insufficient_data_returns_flat(self):
        from src.signals.features import build_features
        from src.signals.quant_signal import compute_quant_signal
        df = make_fake_ohlcv(10)
        features = build_features(df)
        signal = compute_quant_signal(features, "TEST")
        assert signal.direction == 0.0


class TestPositionSizer:
    def test_kelly_size_basic(self):
        from src.risk.position_sizer import kelly_size
        result = kelly_size(
            portfolio_value=100_000,
            win_rate=0.55,
            avg_win_pct=0.04,
            avg_loss_pct=0.02,
            price=150.0
        )
        assert result.shares > 0
        assert result.notional <= 100_000
        assert result.method in ("kelly", "fixed_fractional")

    def test_options_size_basic(self):
        from src.risk.position_sizer import options_size
        result = options_size(
            portfolio_value=100_000,
            max_loss_per_contract=200.0
        )
        assert result.contracts >= 1
        assert result.risk_dollars <= 100_000 * 0.05  # within 5%


class TestSentimentParsing:
    def test_parse_valid_json(self):
        from src.signals.sentiment import SentimentEngine
        engine = SentimentEngine.__new__(SentimentEngine)
        raw = '{"score": 0.7, "confidence": 0.8, "reasoning": "Strong earnings beat.", "catalysts": ["EPS beat"], "timeframe": "short_term"}'
        result = engine._parse_response(raw)
        assert result.score == 0.7
        assert result.confidence == 0.8
        assert "Strong earnings" in result.reasoning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
