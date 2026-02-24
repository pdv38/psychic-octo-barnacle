"""
Model Training Script
──────────────────────
Run this ONCE before live trading to train and save the LightGBM signal model.
Can be re-run weekly to retrain on fresh data.

Usage:
    python notebooks/train_model.py
    python notebooks/train_model.py --symbols SPY QQQ AAPL MSFT TSLA --lookback 500
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import UNIVERSE
from src.data.market_data import market_data
from src.signals.features import build_features
from src.signals.ml_signal import ml_engine


def train(symbols: list[str], lookback_days: int = 500, forward_days: int = 5):
    """
    Collect historical data for all symbols, build features,
    and train a single ensemble LightGBM model.
    """
    logger.info(f"Training on {len(symbols)} symbols, {lookback_days} days lookback")

    all_features = []
    all_prices   = []

    for sym in symbols:
        logger.info(f"  Fetching {sym}...")
        bars = market_data.get_daily_bars([sym], lookback_days=lookback_days)
        df   = bars.get(sym)

        if df is None or len(df) < 100:
            logger.warning(f"  {sym}: insufficient data, skipping")
            continue

        features = build_features(df)
        prices   = df["close"]

        # Align index
        common_idx = features.index.intersection(prices.index)
        features   = features.loc[common_idx]
        prices     = prices.loc[common_idx]

        # Add symbol as a feature (label-encoded)
        features["symbol_hash"] = hash(sym) % 1000

        all_features.append(features)
        all_prices.append(prices)
        logger.info(f"  {sym}: {len(features)} rows, {len(features.columns)} features")

    if not all_features:
        logger.error("No data collected. Aborting.")
        return

    # Concatenate all symbols
    X_all = pd.concat(all_features, axis=0).sort_index()
    y_all = pd.concat(all_prices,   axis=0).sort_index()

    # Remove inf/nan
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Total training rows: {len(X_all)} | Features: {X_all.shape[1]}")

    # Train
    metrics = ml_engine.train(X_all, y_all, forward_days=forward_days)
    logger.info(f"Training complete: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",  nargs="+", default=UNIVERSE)
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--forward",  type=int, default=5)
    args = parser.parse_args()

    train(args.symbols, args.lookback, args.forward)
