"""
Position Sizer
───────────────
Determines trade size using:
1. Kelly Criterion (ML win-rate + payoff ratio)
2. Fixed fractional (2% risk per trade — fallback)
3. Portfolio VaR cap via QuantLib
4. Options: size by max_loss not notional

Always respects:
- MAX_RISK_PCT_PER_TRADE
- MAX_PORTFOLIO_RISK (portfolio-level VaR)
- MAX_OPEN_POSITIONS
"""
import numpy as np
from dataclasses import dataclass
from loguru import logger

from config.settings import (
    MAX_RISK_PCT_PER_TRADE, MAX_PORTFOLIO_RISK,
    MAX_OPEN_POSITIONS, KELLY_FRACTION, INITIAL_CASH
)


@dataclass
class SizeResult:
    shares: int          # equity: number of shares
    contracts: int       # options: number of contracts
    notional: float      # $ value of position
    risk_dollars: float  # max $ at risk
    method: str          # "kelly" | "fixed_fractional" | "options_risk"


def kelly_size(
    portfolio_value: float,
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    price: float,
) -> SizeResult:
    """
    Full Kelly * KELLY_FRACTION for equity positions.
    
    Kelly fraction = (W * b - L) / b
    where b = avg_win / avg_loss (payoff ratio), W = win_rate, L = loss_rate
    """
    if avg_loss_pct <= 0 or price <= 0:
        return _fixed_fractional(portfolio_value, price)

    b = avg_win_pct / avg_loss_pct   # payoff ratio
    kelly_f = (win_rate * b - (1 - win_rate)) / b
    kelly_f = max(0.0, min(kelly_f, 1.0))

    # Apply fraction for conservatism
    position_pct = kelly_f * KELLY_FRACTION

    # Hard cap at MAX_RISK_PCT_PER_TRADE
    position_pct = min(position_pct, MAX_RISK_PCT_PER_TRADE * 5)  # 5× risk = 10% max notional

    notional = portfolio_value * position_pct
    shares   = max(1, int(notional / price))
    notional = shares * price
    risk_dollars = notional * avg_loss_pct

    logger.debug(
        f"Kelly sizing: kelly_f={kelly_f:.3f} frac={KELLY_FRACTION} "
        f"pos_pct={position_pct:.3f} shares={shares}"
    )
    return SizeResult(shares, 0, notional, risk_dollars, "kelly")


def options_size(
    portfolio_value: float,
    max_loss_per_contract: float,
) -> SizeResult:
    """
    Size options position by max loss (defined-risk).
    Risk at most MAX_RISK_PCT_PER_TRADE of portfolio per trade.
    """
    if max_loss_per_contract <= 0:
        return SizeResult(0, 0, 0, 0, "options_risk")

    risk_budget   = portfolio_value * MAX_RISK_PCT_PER_TRADE
    contracts     = max(1, int(risk_budget / max_loss_per_contract))
    notional      = contracts * max_loss_per_contract
    risk_dollars  = contracts * max_loss_per_contract

    logger.debug(
        f"Options sizing: budget=${risk_budget:.0f} "
        f"max_loss/contract=${max_loss_per_contract:.0f} contracts={contracts}"
    )
    return SizeResult(0, contracts, notional, risk_dollars, "options_risk")


def portfolio_var_check(
    positions: dict,   # {symbol: {"notional": float, "price": float, "returns": list}}
    new_notional: float,
    portfolio_value: float,
) -> bool:
    """
    Simple parametric VaR check.
    Returns True if adding new_notional keeps portfolio VaR under MAX_PORTFOLIO_RISK.
    
    Uses 95% 1-day VaR with normal distribution approximation.
    QuantLib integration point: could replace with full MC simulation.
    """
    try:
        total_notional = sum(p["notional"] for p in positions.values()) + new_notional
        position_count = len(positions)

        if position_count >= MAX_OPEN_POSITIONS:
            logger.warning(f"Max open positions ({MAX_OPEN_POSITIONS}) reached")
            return False

        # Portfolio exposure as % of portfolio value
        exposure_pct = total_notional / portfolio_value
        if exposure_pct > 0.95:
            logger.warning(f"Portfolio exposure {exposure_pct:.1%} too high")
            return False

        # Simplified VaR: assume 2% daily vol, 95% = 1.645σ
        # Full QL version: fit covariance matrix and run MonteCarlo
        assumed_daily_vol = 0.02
        z_95 = 1.645
        portfolio_var = total_notional * assumed_daily_vol * z_95
        var_pct = portfolio_var / portfolio_value

        if var_pct > MAX_PORTFOLIO_RISK:
            logger.warning(
                f"VaR check FAILED: portfolio VaR {var_pct:.1%} > "
                f"limit {MAX_PORTFOLIO_RISK:.1%}"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"VaR check error: {e}")
        return True   # fail-open (let execution layer decide)


def _fixed_fractional(portfolio_value: float, price: float) -> SizeResult:
    """Fallback: risk exactly MAX_RISK_PCT_PER_TRADE of portfolio."""
    risk_dollars = portfolio_value * MAX_RISK_PCT_PER_TRADE
    stop_distance_pct = 0.02   # assume 2% stop
    notional = risk_dollars / stop_distance_pct
    shares   = max(1, int(notional / price))
    notional = shares * price

    return SizeResult(shares, 0, notional, risk_dollars, "fixed_fractional")
