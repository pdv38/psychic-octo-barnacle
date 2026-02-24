"""
Central configuration for the AI Hedge Fund.
All tuneable parameters live here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Broker ──────────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
PAPER_MODE        = os.getenv("PAPER_MODE", "true").lower() == "true"

# ── Universe ─────────────────────────────────────────────────────────────────
UNIVERSE = os.getenv(
    "UNIVERSE",
    "SPY,QQQ,AAPL,MSFT,TSLA,AMZN,NVDA,META"
).split(",")

# Options universe — liquid names with tight spreads
OPTIONS_UNIVERSE = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]

# ── Risk Parameters ──────────────────────────────────────────────────────────
MAX_RISK_PCT_PER_TRADE = float(os.getenv("MAX_RISK_PCT", "0.02"))   # 2% of portfolio per trade
MAX_PORTFOLIO_RISK     = float(os.getenv("MAX_PORTFOLIO_RISK", "0.10"))  # 10% total VaR limit
MAX_DAILY_DRAWDOWN     = 0.03   # Kill switch: halt if down 3% on day
MAX_OPEN_POSITIONS     = 5
KELLY_FRACTION         = 0.25   # Use 25% Kelly (conservative)

# ── Options Config ───────────────────────────────────────────────────────────
OPTIONS_DTE_MIN        = 7      # Min days-to-expiry
OPTIONS_DTE_MAX        = 45     # Max days-to-expiry (theta sweet spot)
OPTIONS_DELTA_TARGET   = 0.30   # Target delta for directional plays
IV_RANK_SELL_THRESHOLD = 70     # Sell premium when IV rank > 70
IV_RANK_BUY_THRESHOLD  = 30     # Buy options when IV rank < 30
MIN_OPEN_INTEREST      = 500    # Liquidity filter

# ── Signal Config ────────────────────────────────────────────────────────────
SIGNAL_LOOKBACK_DAYS   = 252    # 1 year for feature engineering
SIGNAL_THRESHOLD       = 0.55   # Min model confidence to trade
SENTIMENT_WEIGHT       = 0.25   # LLM sentiment weight in ensemble
QUANT_WEIGHT           = 0.40   # Technical/quant signal weight
ML_WEIGHT              = 0.35   # ML model weight in ensemble

# ── Backtrader ────────────────────────────────────────────────────────────────
INITIAL_CASH           = 100_000.0
COMMISSION             = 0.001   # 0.1% equity; options handled separately
SLIPPAGE               = 0.001

# ── Schedule (ET) ────────────────────────────────────────────────────────────
MARKET_OPEN_ET  = "09:30"
MARKET_CLOSE_ET = "16:00"
SIGNAL_RUN_TIME = "09:15"   # Pre-market signal computation

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER    = "openai"   # "openai" or "anthropic"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NEWS_API_KEY    = os.getenv("NEWS_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
import pathlib
BASE_DIR    = pathlib.Path(__file__).parent.parent
LOG_DIR     = BASE_DIR / "logs"
REPORT_DIR  = BASE_DIR / "reports"
DATA_DIR    = BASE_DIR / "data_cache"

for d in [LOG_DIR, REPORT_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)
