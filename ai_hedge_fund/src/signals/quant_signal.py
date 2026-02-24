"""
Quantitative Signal Generator
──────────────────────────────
Rule-based signals derived from technical analysis and volatility regime.
These are interpretable, auditable, and explainable to your mentor/TD Bank.

Outputs a signal in [-1, 0, 1] and a conviction float in [0, 1].
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger


@dataclass
class QuantSignal:
    direction: float   # -1 short, 0 flat, +1 long
    conviction: float  # 0–1
    components: dict   # sub-signal breakdown for logging


def compute_quant_signal(features: pd.DataFrame, symbol: str) -> QuantSignal:
    """
    Combines multiple quant sub-signals into a composite directional signal.
    
    Sub-signals:
        1. Trend alignment (price vs SMA stack)
        2. Momentum (MACD + RSI)
        3. Mean reversion (Bollinger Band extremes)
        4. Volume confirmation
        5. Volatility regime (avoid high-vol uncertainty)
    """
    if features.empty or len(features) < 50:
        return QuantSignal(0.0, 0.0, {})

    row = features.iloc[-1]

    components = {}
    scores = []

    # ── 1. Trend Alignment ────────────────────────────────────────────────────
    # Bullish if price > SMA20 > SMA50; bearish if reverse
    trend_score = 0.0
    try:
        p_vs_20  = row.get("price_vs_sma_20", 0)
        p_vs_50  = row.get("price_vs_sma_50", 0)
        p_vs_200 = row.get("price_vs_sma_200", 0)
        trend_score = np.sign(p_vs_20) * 0.4 + np.sign(p_vs_50) * 0.35 + np.sign(p_vs_200) * 0.25
        components["trend"] = round(trend_score, 3)
        scores.append(trend_score)
    except Exception:
        pass

    # ── 2. MACD Momentum ──────────────────────────────────────────────────────
    macd_score = 0.0
    try:
        macd_hist = row.get("macd_hist", 0)
        prev_hist = features["macd_hist"].iloc[-2] if len(features) > 2 else 0
        macd_score = np.clip(macd_hist / (abs(macd_hist) + 1e-6), -1, 1)
        # Bonus: histogram expanding in direction
        if macd_hist * prev_hist > 0 and abs(macd_hist) > abs(prev_hist):
            macd_score *= 1.2
        macd_score = np.clip(macd_score, -1, 1)
        components["macd"] = round(macd_score, 3)
        scores.append(macd_score * 0.8)
    except Exception:
        pass

    # ── 3. RSI Momentum / Mean Reversion ─────────────────────────────────────
    rsi_score = 0.0
    try:
        rsi = row.get("rsi_14", 50)
        if rsi > 70:
            rsi_score = -0.8    # overbought → lean bearish
        elif rsi > 60:
            rsi_score = 0.3
        elif rsi < 30:
            rsi_score = 0.8     # oversold → lean bullish
        elif rsi < 40:
            rsi_score = -0.3
        else:
            rsi_score = (rsi - 50) / 50.0  # neutral zone
        components["rsi"] = round(rsi_score, 3)
        scores.append(rsi_score * 0.6)
    except Exception:
        pass

    # ── 4. Bollinger Band Position ────────────────────────────────────────────
    bb_score = 0.0
    try:
        bb_pct = row.get("bb_pct", 0.5)
        # Mean-reversion: near lower band → bullish, near upper → bearish
        bb_score = -(bb_pct - 0.5) * 2   # maps [0,1] → [+1,-1]
        bb_score = np.clip(bb_score, -1, 1)
        components["bollinger"] = round(bb_score, 3)
        scores.append(bb_score * 0.5)
    except Exception:
        pass

    # ── 5. Volume Confirmation ────────────────────────────────────────────────
    vol_confirm = 1.0
    try:
        vol_ratio = row.get("volume_ratio", 1.0)
        trend_dir = np.sign(trend_score) if trend_score != 0 else 0
        if vol_ratio > 1.5 and trend_dir != 0:
            vol_confirm = 1.3   # high volume in trend direction = boost
        elif vol_ratio < 0.5:
            vol_confirm = 0.7   # low volume = less conviction
        components["volume_confirm"] = round(vol_confirm, 3)
    except Exception:
        pass

    # ── 6. Volatility Regime Filter ───────────────────────────────────────────
    vol_filter = 1.0
    try:
        atr_pct = row.get("atr_pct", 0.02)
        if atr_pct > 0.04:      # extremely volatile: reduce sizing
            vol_filter = 0.5
            components["vol_regime"] = "HIGH"
        elif atr_pct < 0.01:    # very calm: full conviction
            vol_filter = 1.1
            components["vol_regime"] = "LOW"
        else:
            components["vol_regime"] = "NORMAL"
    except Exception:
        pass

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not scores:
        return QuantSignal(0.0, 0.0, components)

    raw_score  = np.mean(scores) * vol_confirm * vol_filter
    direction  = float(np.sign(raw_score))
    conviction = float(np.clip(abs(raw_score), 0, 1))

    # Require minimum conviction to avoid noise trades
    if conviction < 0.25:
        direction = 0.0
        conviction = 0.0

    components["raw_score"] = round(raw_score, 4)

    logger.debug(f"{symbol} QUANT signal: dir={direction:+.0f} conv={conviction:.2f} | {components}")
    return QuantSignal(direction, conviction, components)
