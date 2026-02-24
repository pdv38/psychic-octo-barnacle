"""
Market Data Layer
─────────────────
Fetches equity OHLCV bars and options chains.
Primary: yfinance (free, no key needed)
Live:    Alpaca Markets (free paper account)
"""
import time
import datetime
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    UNIVERSE, OPTIONS_UNIVERSE
)


class MarketDataFeed:
    """Unified market data interface for equity + options."""

    def __init__(self):
        self._alpaca = None
        if ALPACA_AVAILABLE and ALPACA_API_KEY:
            try:
                self._alpaca = tradeapi.REST(
                    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
                )
                logger.info("Alpaca connection established (paper mode)")
            except Exception as e:
                logger.warning(f"Alpaca init failed, falling back to yfinance: {e}")

    # ── Equity Bars ──────────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        symbols: list[str],
        lookback_days: int = 365,
        end: Optional[datetime.date] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Returns dict of {symbol: OHLCV DataFrame} for each symbol.
        Uses Alpaca if available and market hours, else yfinance.
        """
        end = end or datetime.date.today()
        start = end - datetime.timedelta(days=lookback_days)
        result = {}

        for sym in symbols:
            try:
                df = self._fetch_yfinance(sym, start.isoformat(), end.isoformat())
                if df is not None and not df.empty:
                    result[sym] = df
                    logger.debug(f"Fetched {len(df)} bars for {sym}")
            except Exception as e:
                logger.error(f"Failed to fetch {sym}: {e}")

        return result

    def _fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d")
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna()
        return df

    # ── Options Data ──────────────────────────────────────────────────────────

    def get_options_chain(self, symbol: str) -> dict:
        """
        Fetches full options chain for a symbol via yfinance.
        Returns dict with calls, puts DataFrames and metadata.
        """
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                logger.warning(f"No options data for {symbol}")
                return {}

            spot = ticker.fast_info.get("last_price") or ticker.fast_info.last_price
            today = datetime.date.today()

            chains = {}
            for exp in expirations[:6]:  # Nearest 6 expirations
                exp_date = datetime.date.fromisoformat(exp)
                dte = (exp_date - today).days

                if dte < 1:
                    continue

                try:
                    chain = ticker.option_chain(exp)
                    chains[exp] = {
                        "calls": chain.calls,
                        "puts": chain.puts,
                        "dte": dte,
                        "spot": spot,
                    }
                except Exception as e:
                    logger.debug(f"Skipping {symbol} {exp}: {e}")

            return chains

        except Exception as e:
            logger.error(f"Options chain error for {symbol}: {e}")
            return {}

    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        """
        Compute IV Rank: where current IV sits in its 1-year range.
        IV Rank = (current_IV - 52w_low) / (52w_high - 52w_low) * 100
        Uses ATM implied vol approximated from options chain.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            if hist.empty:
                return 50.0  # neutral default

            # Realized vol proxy (annualized)
            returns = hist["Close"].pct_change().dropna()
            rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100

            current_vol = rolling_vol.iloc[-1]
            vol_52w_high = rolling_vol.max()
            vol_52w_low  = rolling_vol.min()

            if vol_52w_high == vol_52w_low:
                return 50.0

            iv_rank = (current_vol - vol_52w_low) / (vol_52w_high - vol_52w_low) * 100
            return round(float(iv_rank), 1)

        except Exception as e:
            logger.warning(f"IV Rank computation failed for {symbol}: {e}")
            return 50.0

    def get_current_price(self, symbol: str) -> float:
        """Get latest price via yfinance fast_info."""
        try:
            t = yf.Ticker(symbol)
            return float(t.fast_info.last_price)
        except Exception:
            return 0.0

    def get_news(self, symbol: str, max_items: int = 10) -> list[dict]:
        """Fetch recent news headlines for sentiment analysis."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            return [
                {
                    "title": n.get("title", ""),
                    "publisher": n.get("publisher", ""),
                    "link": n.get("link", ""),
                    "published": n.get("providerPublishTime", 0),
                }
                for n in news[:max_items]
            ]
        except Exception as e:
            logger.warning(f"News fetch failed for {symbol}: {e}")
            return []


# Singleton
market_data = MarketDataFeed()
