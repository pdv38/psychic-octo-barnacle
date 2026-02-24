"""
Signal Ensemble
────────────────
Combines Quant + ML + Sentiment signals into a final trading decision.

Each signal source has a configurable weight.
Final output: TradeSignal with direction, conviction, and full audit log.
"""
import datetime
from dataclasses import dataclass, field
from loguru import logger

from config.settings import (
    QUANT_WEIGHT, ML_WEIGHT, SENTIMENT_WEIGHT, SIGNAL_THRESHOLD
)
from src.signals.quant_signal import compute_quant_signal, QuantSignal
from src.signals.ml_signal import ml_engine, MLSignal
from src.signals.sentiment import sentiment_engine, SentimentSignal
from src.signals.features import build_features
from src.data.market_data import market_data


@dataclass
class TradeSignal:
    symbol: str
    timestamp: datetime.datetime
    direction: float          # -1, 0, +1
    conviction: float         # 0–1 combined
    action: str               # "BUY" | "SELL" | "FLAT"
    asset_type: str           # "equity" | "option"

    # Sub-signals for audit
    quant: QuantSignal = None
    ml: MLSignal = None
    sentiment: SentimentSignal = None

    # Options-specific
    option_strategy: str = ""  # e.g. "bull_call_spread", "sell_put"
    option_details: dict = field(default_factory=dict)

    def to_log_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action": self.action,
            "direction": self.direction,
            "conviction": round(self.conviction, 4),
            "quant_dir": self.quant.direction if self.quant else None,
            "quant_conv": self.quant.conviction if self.quant else None,
            "ml_dir": self.ml.direction if self.ml else None,
            "ml_conf": self.ml.confidence if self.ml else None,
            "sentiment_score": self.sentiment.score if self.sentiment else None,
            "sentiment_reasoning": self.sentiment.reasoning if self.sentiment else "",
            "option_strategy": self.option_strategy,
        }


class EnsembleSignalEngine:
    """
    Master signal aggregator.
    
    For each symbol:
    1. Build features from recent price history
    2. Run Quant signal
    3. Run ML signal
    4. Run LLM sentiment
    5. Weighted combination → final direction + conviction
    6. Decide equity vs options strategy based on IV rank
    """

    def generate(self, symbol: str) -> TradeSignal:
        """Full signal pipeline for one symbol."""
        now = datetime.datetime.now()

        try:
            # ── Fetch Data ─────────────────────────────────────────────────────
            bars = market_data.get_daily_bars([symbol], lookback_days=300)
            df = bars.get(symbol)
            if df is None or len(df) < 60:
                logger.warning(f"{symbol}: insufficient data ({len(df) if df is not None else 0} bars)")
                return self._flat_signal(symbol, now)

            # ── Features ───────────────────────────────────────────────────────
            features = build_features(df)

            # ── Sub-signals ────────────────────────────────────────────────────
            q_signal = compute_quant_signal(features, symbol)
            m_signal = ml_engine.predict(features, symbol)
            headlines = market_data.get_news(symbol, max_items=8)
            s_signal  = sentiment_engine.analyze(symbol, headlines)

            # ── Weighted Ensemble ──────────────────────────────────────────────
            # Normalize each to [-1, +1] × conviction
            q_contrib = q_signal.direction * q_signal.conviction * QUANT_WEIGHT
            m_contrib = m_signal.direction * m_signal.confidence * ML_WEIGHT
            s_contrib = s_signal.score    * s_signal.confidence  * SENTIMENT_WEIGHT

            raw = q_contrib + m_contrib + s_contrib
            direction  = 1.0 if raw > 0 else (-1.0 if raw < 0 else 0.0)
            conviction = min(abs(raw), 1.0)

            # Require all sub-signals to roughly agree for high conviction
            # (avoids cases where one signal is extremely strong and overrides others)
            directions = [
                q_signal.direction,
                m_signal.direction,
                float(1 if s_signal.score > 0.1 else (-1 if s_signal.score < -0.1 else 0))
            ]
            agreement  = sum(1 for d in directions if d == direction)
            if agreement < 2:
                conviction *= 0.6   # reduce conviction on disagreement

            # Kill switch: below threshold → flat
            if conviction < SIGNAL_THRESHOLD:
                direction  = 0.0
                conviction = 0.0

            action = "BUY" if direction > 0 else ("SELL" if direction < 0 else "FLAT")

            # ── Options vs Equity Routing ──────────────────────────────────────
            iv_rank = market_data.get_iv_rank(symbol)
            asset_type, option_strategy, option_details = self._route_instrument(
                symbol, direction, conviction, iv_rank
            )

            signal = TradeSignal(
                symbol          = symbol,
                timestamp       = now,
                direction       = direction,
                conviction      = conviction,
                action          = action,
                asset_type      = asset_type,
                quant           = q_signal,
                ml              = m_signal,
                sentiment       = s_signal,
                option_strategy = option_strategy,
                option_details  = option_details,
            )

            logger.info(
                f"[SIGNAL] {symbol:6s} {action:4s} | "
                f"conv={conviction:.2f} iv_rank={iv_rank:.0f} "
                f"type={asset_type} strat={option_strategy or 'equity'}"
            )
            return signal

        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return self._flat_signal(symbol, now)

    def _route_instrument(
        self, symbol: str, direction: float, conviction: float, iv_rank: float
    ) -> tuple[str, str, dict]:
        """
        Decide whether to trade equity or options, and which options strategy.
        
        IV Rank logic:
        - High IV (>70): SELL premium (collect theta) — credit spreads, cash-secured puts
        - Low IV (<30):  BUY options (cheap vol)     — debit spreads
        - Mid IV:        Equity position
        
        Direction:
        - Bullish + High IV → sell cash-secured put / bull put spread
        - Bearish + High IV → sell covered call / bear call spread  
        - Bullish + Low IV  → buy call / bull call spread
        - Bearish + Low IV  → buy put / bear put spread
        """
        from config.settings import OPTIONS_UNIVERSE

        if symbol not in OPTIONS_UNIVERSE or direction == 0:
            return "equity", "", {}

        if iv_rank > 70:
            # Sell premium — volatility is rich
            if direction > 0:
                return "option", "bull_put_spread", {"iv_rank": iv_rank, "type": "credit"}
            else:
                return "option", "bear_call_spread", {"iv_rank": iv_rank, "type": "credit"}
        elif iv_rank < 30:
            # Buy options — volatility is cheap
            if direction > 0:
                return "option", "bull_call_spread", {"iv_rank": iv_rank, "type": "debit"}
            else:
                return "option", "bear_put_spread", {"iv_rank": iv_rank, "type": "debit"}
        else:
            # Mid IV — trade equity, not options
            return "equity", "", {"iv_rank": iv_rank}

    def _flat_signal(self, symbol: str, ts: datetime.datetime) -> TradeSignal:
        return TradeSignal(
            symbol=symbol, timestamp=ts, direction=0.0,
            conviction=0.0, action="FLAT", asset_type="equity"
        )

    def run_universe(self, symbols: list[str]) -> list[TradeSignal]:
        """Generate signals for the full universe. Returns only actionable signals."""
        signals = []
        for sym in symbols:
            sig = self.generate(sym)
            signals.append(sig)
        return signals


# Singleton
ensemble = EnsembleSignalEngine()
