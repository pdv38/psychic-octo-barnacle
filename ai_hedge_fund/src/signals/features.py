"""
Feature Engineering
────────────────────
Builds the feature matrix from raw OHLCV data.
Used by both the ML model and quant signal generators.
"""
import numpy as np
import pandas as pd
from loguru import logger


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given an OHLCV DataFrame (columns: open, high, low, close, volume),
    returns a DataFrame of engineered features ready for the ML pipeline.
    
    All features are computed without look-ahead bias (use only past data).
    """
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Trend Features ────────────────────────────────────────────────────────
    for n in [5, 10, 20, 50, 200]:
        f[f"sma_{n}"] = c.rolling(n).mean()
        f[f"price_vs_sma_{n}"] = (c / f[f"sma_{n}"]) - 1

    f["ema_12"] = c.ewm(span=12).mean()
    f["ema_26"] = c.ewm(span=26).mean()
    f["macd"]   = f["ema_12"] - f["ema_26"]
    f["macd_signal"] = f["macd"].ewm(span=9).mean()
    f["macd_hist"]   = f["macd"] - f["macd_signal"]

    # ── Momentum ──────────────────────────────────────────────────────────────
    for n in [1, 5, 10, 21, 63]:
        f[f"return_{n}d"] = c.pct_change(n)

    # RSI
    for n in [7, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(n).mean()
        loss  = (-delta.clip(upper=0)).rolling(n).mean()
        rs    = gain / loss.replace(0, np.nan)
        f[f"rsi_{n}"] = 100 - (100 / (1 + rs))

    # Rate of Change
    f["roc_10"] = c.pct_change(10) * 100
    f["roc_20"] = c.pct_change(20) * 100

    # ── Volatility ────────────────────────────────────────────────────────────
    ret = c.pct_change()
    for n in [5, 10, 21]:
        f[f"realized_vol_{n}d"] = ret.rolling(n).std() * np.sqrt(252)

    # ATR (Average True Range)
    tr = pd.DataFrame({
        "hl":  h - l,
        "hc":  (h - c.shift()).abs(),
        "lc":  (l - c.shift()).abs(),
    }).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean()
    f["atr_pct"] = f["atr_14"] / c  # normalized ATR

    # Bollinger Bands
    bb_mid  = c.rolling(20).mean()
    bb_std  = c.rolling(20).std()
    bb_up   = bb_mid + 2 * bb_std
    bb_dn   = bb_mid - 2 * bb_std
    f["bb_pct"]   = (c - bb_dn) / (bb_up - bb_dn).replace(0, np.nan)
    f["bb_width"] = (bb_up - bb_dn) / bb_mid

    # ── Volume ───────────────────────────────────────────────────────────────
    f["volume_sma_20"] = v.rolling(20).mean()
    f["volume_ratio"]  = v / f["volume_sma_20"]
    f["volume_z"]      = (v - v.rolling(20).mean()) / v.rolling(20).std()

    # On-Balance Volume
    obv = (np.sign(c.diff()) * v).cumsum()
    f["obv_slope"] = obv.diff(5) / obv.abs().rolling(5).mean().replace(0, np.nan)

    # ── Market Microstructure ─────────────────────────────────────────────────
    f["close_location"] = (c - l) / (h - l).replace(0, np.nan)  # 0=low, 1=high
    f["gap"]             = (df["open"] / c.shift() - 1)          # overnight gap

    # ── Cross-asset (regime proxy) ────────────────────────────────────────────
    # Computed externally and merged in if available; placeholder here
    f["spy_return_5d"] = np.nan   # filled by signal engine

    # Drop NaN rows (from rolling windows)
    f = f.replace([np.inf, -np.inf], np.nan)

    return f


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes NaN-heavy columns)."""
    return [c for c in df.columns if df[c].notna().sum() > 50]
