# 🤖 AI-Native Hedge Fund

> Autonomous options & equity paper trader · Runs on GitHub Actions · 9:30–16:00 ET

**Stack:** Python · Backtrader · QuantLib · LightGBM · LLM Sentiment · Alpaca Paper API

Live dashboard → [your-name.github.io/ai-hedge-fund](https://your-name.github.io/ai-hedge-fund)

---

## Architecture

```
Market Data (yfinance / Alpaca)
        ↓
Feature Engineering (OHLCV → 30+ technical features)
        ↓
┌──────────────────────────────────────────────────┐
│  Signal Ensemble                                 │
│   ├── Quant Signal    (40%)  — rule-based        │
│   ├── ML Signal       (35%)  — LightGBM          │
│   └── LLM Sentiment   (25%)  — GPT/Claude        │
└──────────────────────────────────────────────────┘
        ↓
Risk Engine (QuantLib VaR + Kelly sizing)
        ↓
Options Router (IV Rank → spread strategy selection)
        ↓
Backtrader Broker (paper execution + stop/TP management)
        ↓
Daily HTML Report → GitHub Pages
```

## Quick Start

### 1. Clone & configure
```bash
git clone https://github.com/your-name/ai-hedge-fund
cd ai-hedge-fund
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the ML model (one-time)
```bash
python notebooks/train_model.py
```

### 4. Test locally
```bash
python main.py --force --symbols SPY QQQ AAPL
```

### 5. Push to GitHub — bot runs automatically
The GitHub Actions workflow runs Mon–Fri and publishes reports to GitHub Pages.

---

## GitHub Secrets Required

Add these in `Settings → Secrets → Actions`:

| Secret | Where to get |
|--------|-------------|
| `ALPACA_API_KEY` | [alpaca.markets](https://alpaca.markets) (free paper account) |
| `ALPACA_SECRET_KEY` | Same |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) (optional) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) (optional) |
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org) (optional) |

Only `ALPACA_*` are required. LLM keys are optional — system runs with neutral sentiment if absent.

## Enable GitHub Pages

`Settings → Pages → Source: Deploy from branch → Branch: main → Folder: /docs`

---

## Signal Logic

### Quant Signal (40% weight)
Rule-based combination of:
- SMA trend alignment (5/20/50/200)
- MACD histogram direction + expansion
- RSI overbought/oversold zones
- Bollinger Band mean reversion
- Volume confirmation

### ML Signal (35% weight)
LightGBM binary classifier:
- **Label:** positive 5-day forward return (1) vs negative (0)
- **Features:** 30+ technical indicators
- **Validation:** Walk-forward cross-validation (no lookahead)
- **Training:** Run `notebooks/train_model.py` weekly to retrain

### LLM Sentiment (25% weight)
GPT-4o-mini or Claude Haiku:
- Fetches last 8 news headlines per symbol
- Classifies: bullish/bearish score + confidence + reasoning
- Reasoning logged per trade — great for explainability demos

---

## Options Strategy Routing

| IV Rank | Direction | Strategy |
|---------|-----------|----------|
| > 70 (rich vol) | Bullish | Bull Put Spread (credit) |
| > 70 (rich vol) | Bearish | Bear Call Spread (credit) |
| < 30 (cheap vol) | Bullish | Bull Call Spread (debit) |
| < 30 (cheap vol) | Bearish | Bear Put Spread (debit) |
| 30–70 | Either | Equity position |

Options are sized by **max loss per contract** via Kelly Criterion.  
QuantLib Black-Scholes-Merton engine computes theoretical prices + all Greeks (Δ, Γ, Θ, ν).

---

## Risk Management

- **Position sizing:** Kelly Criterion × 0.25 (conservative fraction)
- **Stop-loss:** 2× ATR below entry (dynamic)
- **Take-profit:** 4× ATR above entry
- **Daily kill switch:** Halt all trading if daily drawdown > 3%
- **Max open positions:** 5 simultaneous
- **Portfolio VaR cap:** 10% 95-confidence 1-day VaR

---

## Project Structure

```
ai_hedge_fund/
├── main.py                        # Entry point
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py                # All tuneable parameters
├── src/
│   ├── data/
│   │   └── market_data.py         # yfinance + Alpaca feeds
│   ├── signals/
│   │   ├── features.py            # Feature engineering (30+ indicators)
│   │   ├── quant_signal.py        # Rule-based quant signal
│   │   ├── ml_signal.py           # LightGBM ML signal
│   │   ├── sentiment.py           # LLM news sentiment
│   │   └── ensemble.py            # Signal combiner + options router
│   ├── risk/
│   │   ├── options_pricer.py      # QuantLib BSM + spread builders
│   │   └── position_sizer.py      # Kelly + VaR position sizing
│   ├── execution/
│   │   ├── strategy.py            # Backtrader Strategy class
│   │   └── runner.py              # Session orchestrator
│   └── reporting/
│       └── report.py              # HTML report generator
├── notebooks/
│   └── train_model.py             # ML model training script
├── tests/
│   └── test_signals.py            # Unit tests
├── docs/                          # GitHub Pages output
├── logs/                          # Trade journals
└── .github/
    └── workflows/
        └── trade.yml              # GitHub Actions schedule
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Disclaimer

This is a paper-trading educational project. No real money is at risk. Past simulated performance does not guarantee future results.
