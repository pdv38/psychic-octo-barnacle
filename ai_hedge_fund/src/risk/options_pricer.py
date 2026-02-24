"""
Options Pricer — QuantLib Black-Scholes-Merton
───────────────────────────────────────────────
Uses QuantLib to price options and compute Greeks.
Also handles:
- Optimal strike selection given a target delta
- Spread legs construction
- Max loss / max gain calculation for position sizing
"""
import datetime
import numpy as np
from loguru import logger
from dataclasses import dataclass

try:
    import QuantLib as ql
    QL_AVAILABLE = True
except ImportError:
    QL_AVAILABLE = False
    logger.warning("QuantLib not installed. Options pricing disabled.")


@dataclass
class OptionLeg:
    symbol: str
    option_type: str       # "call" | "put"
    strike: float
    expiry: datetime.date
    dte: int
    price: float           # theoretical price (BSM)
    delta: float
    gamma: float
    theta: float           # per day
    vega: float
    iv: float              # implied volatility used


@dataclass
class OptionsPosition:
    strategy: str          # "bull_call_spread", etc.
    legs: list[OptionLeg]
    max_profit: float
    max_loss: float
    breakeven: float
    net_premium: float     # positive = credit, negative = debit


def price_option(
    spot: float,
    strike: float,
    expiry: datetime.date,
    vol: float,
    risk_free_rate: float = 0.053,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> OptionLeg | None:
    """
    Price a single European option using QuantLib BSM engine.
    Returns OptionLeg with price + all Greeks.
    """
    if not QL_AVAILABLE:
        return None

    try:
        today = datetime.date.today()
        dte   = (expiry - today).days
        if dte <= 0:
            return None

        # QL date setup
        calculation_date = ql.Date(today.day, today.month, today.year)
        expiry_date      = ql.Date(expiry.day, expiry.month, expiry.year)
        ql.Settings.instance().evaluationDate = calculation_date

        calendar  = ql.UnitedStates(ql.UnitedStates.NYSE)
        day_count = ql.Actual365Fixed()

        # Market handles
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        flat_ts     = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, risk_free_rate, day_count)
        )
        div_ts      = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, dividend_yield, day_count)
        )
        vol_ts      = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(calculation_date, calendar, vol, day_count)
        )

        bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_ts, flat_ts, vol_ts)

        # Option
        ql_type = ql.Option.Call if option_type == "call" else ql.Option.Put
        payoff  = ql.PlainVanillaPayoff(ql_type, strike)
        exercise = ql.EuropeanExercise(expiry_date)
        option   = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

        price = option.NPV()
        delta = option.delta()
        gamma = option.gamma()
        theta = option.theta() / 365.0   # annualized → daily
        vega  = option.vega() / 100.0    # per 1% vol move

        return OptionLeg(
            symbol=f"?{strike}{option_type[0].upper()}{expiry.strftime('%y%m%d')}",
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            dte=dte,
            price=round(price, 4),
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 4),
            vega=round(vega, 4),
            iv=vol,
        )

    except Exception as e:
        logger.error(f"QuantLib pricing error: {e}")
        return None


def build_bull_call_spread(
    spot: float, vol: float, expiry: datetime.date,
    target_delta: float = 0.35, width_pct: float = 0.03
) -> OptionsPosition | None:
    """
    Bull call spread: buy lower strike call, sell upper strike call.
    Max profit = width - net debit. Max loss = net debit.
    """
    long_strike  = round(spot * (1 + 0.01), 0)   # slightly OTM
    short_strike = round(spot * (1 + 0.01 + width_pct), 0)

    long_leg  = price_option(spot, long_strike,  expiry, vol, option_type="call")
    short_leg = price_option(spot, short_strike, expiry, vol, option_type="call")

    if not long_leg or not short_leg:
        return None

    net_debit   = long_leg.price - short_leg.price
    max_profit  = (short_strike - long_strike) - net_debit
    max_loss    = net_debit
    breakeven   = long_strike + net_debit

    return OptionsPosition(
        strategy="bull_call_spread",
        legs=[long_leg, short_leg],
        max_profit=round(max_profit * 100, 2),   # per contract (×100)
        max_loss=round(max_loss * 100, 2),
        breakeven=round(breakeven, 2),
        net_premium=round(-net_debit * 100, 2),  # negative = debit
    )


def build_bull_put_spread(
    spot: float, vol: float, expiry: datetime.date, width_pct: float = 0.03
) -> OptionsPosition | None:
    """
    Bull put spread (credit spread): sell higher strike put, buy lower strike put.
    Max profit = net credit. Max loss = width - credit.
    """
    short_strike = round(spot * 0.98, 0)   # OTM put
    long_strike  = round(spot * (0.98 - width_pct), 0)

    short_leg = price_option(spot, short_strike, expiry, vol, option_type="put")
    long_leg  = price_option(spot, long_strike,  expiry, vol, option_type="put")

    if not short_leg or not long_leg:
        return None

    net_credit = short_leg.price - long_leg.price
    max_profit = net_credit
    max_loss   = (short_strike - long_strike) - net_credit
    breakeven  = short_strike - net_credit

    return OptionsPosition(
        strategy="bull_put_spread",
        legs=[short_leg, long_leg],
        max_profit=round(max_profit * 100, 2),
        max_loss=round(max_loss * 100, 2),
        breakeven=round(breakeven, 2),
        net_premium=round(net_credit * 100, 2),  # positive = credit
    )


def build_bear_call_spread(
    spot: float, vol: float, expiry: datetime.date, width_pct: float = 0.03
) -> OptionsPosition | None:
    """
    Bear call spread (credit spread): sell lower strike call, buy higher strike call.
    """
    short_strike = round(spot * 1.02, 0)
    long_strike  = round(spot * (1.02 + width_pct), 0)

    short_leg = price_option(spot, short_strike, expiry, vol, option_type="call")
    long_leg  = price_option(spot, long_strike,  expiry, vol, option_type="call")

    if not short_leg or not long_leg:
        return None

    net_credit = short_leg.price - long_leg.price
    max_profit = net_credit
    max_loss   = (long_strike - short_strike) - net_credit
    breakeven  = short_strike + net_credit

    return OptionsPosition(
        strategy="bear_call_spread",
        legs=[short_leg, long_leg],
        max_profit=round(max_profit * 100, 2),
        max_loss=round(max_loss * 100, 2),
        breakeven=round(breakeven, 2),
        net_premium=round(net_credit * 100, 2),
    )


def build_bear_put_spread(
    spot: float, vol: float, expiry: datetime.date, width_pct: float = 0.03
) -> OptionsPosition | None:
    """
    Bear put spread (debit): buy higher strike put, sell lower strike put.
    """
    long_strike  = round(spot * 0.99, 0)
    short_strike = round(spot * (0.99 - width_pct), 0)

    long_leg  = price_option(spot, long_strike,  expiry, vol, option_type="put")
    short_leg = price_option(spot, short_strike, expiry, vol, option_type="put")

    if not long_leg or not short_leg:
        return None

    net_debit  = long_leg.price - short_leg.price
    max_profit = (long_strike - short_strike) - net_debit
    max_loss   = net_debit
    breakeven  = long_strike - net_debit

    return OptionsPosition(
        strategy="bear_put_spread",
        legs=[long_leg, short_leg],
        max_profit=round(max_profit * 100, 2),
        max_loss=round(max_loss * 100, 2),
        breakeven=round(breakeven, 2),
        net_premium=round(-net_debit * 100, 2),
    )


STRATEGY_BUILDERS = {
    "bull_call_spread": build_bull_call_spread,
    "bull_put_spread":  build_bull_put_spread,
    "bear_call_spread": build_bear_call_spread,
    "bear_put_spread":  build_bear_put_spread,
}
