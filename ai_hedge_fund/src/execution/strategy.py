"""
Backtrader Strategy — AI Hedge Fund Master Strategy
─────────────────────────────────────────────────────
Orchestrates the full trading loop inside Backtrader's framework.

At each daily bar:
1. Check market hours & kill switches
2. Pull pre-computed signals from ensemble
3. Run position sizing (risk engine)
4. Submit equity or options orders
5. Log everything to trade journal
"""
import datetime
import csv
import json
import backtrader as bt
import backtrader.indicators as btind
from loguru import logger

from config.settings import (
    MAX_DAILY_DRAWDOWN, MAX_OPEN_POSITIONS,
    INITIAL_CASH, REPORT_DIR, LOG_DIR
)
from src.signals.ensemble import ensemble, TradeSignal
from src.risk.position_sizer import kelly_size, options_size, portfolio_var_check
from src.risk.options_pricer import STRATEGY_BUILDERS


class AIHedgeFundStrategy(bt.Strategy):
    """
    Main Backtrader strategy class.
    Runs once per daily bar for each symbol in the universe.
    """

    params = dict(
        universe=None,        # list of symbol strings
        stop_loss_atr=2.0,    # stop = 2× ATR below entry
        take_profit_atr=4.0,  # take-profit = 4× ATR above entry
        verbose=True,
    )

    def __init__(self):
        self.signal_cache    = {}       # {symbol: TradeSignal}
        self.open_positions  = {}       # {symbol: {"entry": price, "stop": price, "size": int}}
        self.daily_pnl_start = None
        self.trade_log       = []
        self.order_map       = {}       # {order_ref: symbol}

        # ATR for each data feed
        self.atrs = {}
        for i, d in enumerate(self.datas):
            self.atrs[d._name] = btind.ATR(d, period=14)

        # Trade journal CSV
        self._journal_path = LOG_DIR / "trade_journal.csv"
        self._journal_headers_written = self._journal_path.exists()

        logger.info("AIHedgeFundStrategy initialized")

    def start(self):
        self.daily_pnl_start = self.broker.getvalue()
        logger.info(f"Strategy started | Portfolio: ${self.broker.getvalue():,.2f}")

    def next(self):
        now = datetime.datetime.now()
        portfolio_value = self.broker.getvalue()

        # ── Daily Kill Switch ──────────────────────────────────────────────────
        daily_return = (portfolio_value / self.daily_pnl_start) - 1 if self.daily_pnl_start else 0
        if daily_return < -MAX_DAILY_DRAWDOWN:
            logger.warning(
                f"KILL SWITCH: daily drawdown {daily_return:.2%} exceeded "
                f"{MAX_DAILY_DRAWDOWN:.2%}. No new trades today."
            )
            return

        # ── Process each symbol ────────────────────────────────────────────────
        for data in self.datas:
            symbol = data._name
            if len(data) < 60:
                continue

            current_price = data.close[0]
            if current_price <= 0:
                continue

            # Check stop-loss / take-profit for open positions
            if symbol in self.open_positions:
                self._check_exits(data, symbol)
                continue

            # Don't open more positions than the cap
            if len(self.open_positions) >= MAX_OPEN_POSITIONS:
                continue

            # ── Get Signal ─────────────────────────────────────────────────────
            signal = self.signal_cache.get(symbol)
            if signal is None or signal.action == "FLAT":
                continue

            # ── Position Sizing ────────────────────────────────────────────────
            atr = float(self.atrs[symbol][0]) if symbol in self.atrs else current_price * 0.02

            if signal.asset_type == "equity":
                size_result = kelly_size(
                    portfolio_value=portfolio_value,
                    win_rate=0.55,          # conservative assumption
                    avg_win_pct=0.04,
                    avg_loss_pct=atr / current_price,
                    price=current_price,
                )
                shares = size_result.shares

                # VaR gate
                if not portfolio_var_check({}, shares * current_price, portfolio_value):
                    continue

                if shares < 1:
                    continue

                stop  = current_price - self.p.stop_loss_atr   * atr
                tp    = current_price + self.p.take_profit_atr  * atr

                if signal.action == "BUY":
                    order = self.buy(data=data, size=shares)
                elif signal.action == "SELL":
                    order = self.sell(data=data, size=shares)
                    stop  = current_price + self.p.stop_loss_atr * atr   # stop above for short
                else:
                    continue

                self.order_map[order.ref] = symbol
                self.open_positions[symbol] = {
                    "entry": current_price,
                    "stop":  stop,
                    "take_profit": tp,
                    "size":  shares,
                    "direction": signal.action,
                    "atr":   atr,
                }

                self._log_trade("OPEN", symbol, signal.action, current_price, shares, signal)
                logger.info(
                    f"[ORDER] {signal.action} {shares}x {symbol} @ ${current_price:.2f} "
                    f"stop=${stop:.2f} tp=${tp:.2f}"
                )

            elif signal.asset_type == "option":
                self._handle_option_signal(signal, portfolio_value, current_price)

    def _check_exits(self, data, symbol: str):
        """Manage stop-loss and take-profit for open positions."""
        pos = self.open_positions[symbol]
        current_price = data.close[0]
        direction = pos.get("direction", "BUY")

        hit_stop = (direction == "BUY"  and current_price <= pos["stop"]) or \
                   (direction == "SELL" and current_price >= pos["stop"])
        hit_tp   = (direction == "BUY"  and current_price >= pos["take_profit"]) or \
                   (direction == "SELL" and current_price <= pos["take_profit"])

        if hit_stop or hit_tp:
            reason = "STOP_LOSS" if hit_stop else "TAKE_PROFIT"
            size   = pos["size"]

            if direction == "BUY":
                self.sell(data=data, size=size)
            else:
                self.buy(data=data, size=size)

            pnl = (current_price - pos["entry"]) * size * (1 if direction == "BUY" else -1)
            logger.info(
                f"[CLOSE] {symbol} {reason} @ ${current_price:.2f} | "
                f"entry=${pos['entry']:.2f} P&L=${pnl:.2f}"
            )
            self._log_trade("CLOSE", symbol, reason, current_price, size, pnl=pnl)
            del self.open_positions[symbol]

    def _handle_option_signal(self, signal: TradeSignal, portfolio_value: float, spot: float):
        """
        Log the options strategy recommendation.
        In paper mode with Backtrader, we simulate equity exposure equivalent.
        Full Alpaca options API can replace this in production.
        """
        import datetime

        strategy_name = signal.option_strategy
        builder_fn    = STRATEGY_BUILDERS.get(strategy_name)

        if not builder_fn:
            return

        # Pick nearest valid expiry (21-45 DTE)
        today  = datetime.date.today()
        expiry = today + datetime.timedelta(days=30)

        # Approximate vol from recent price history
        vol = 0.25  # default 25% IV; replace with market_data.get_iv_rank()-derived vol

        position = builder_fn(spot, vol, expiry)
        if not position:
            return

        size_result = options_size(portfolio_value, position.max_loss)
        contracts   = size_result.contracts

        if contracts < 1:
            return

        logger.info(
            f"[OPTIONS] {strategy_name.upper()} {signal.symbol} | "
            f"{contracts} contracts | "
            f"max_profit=${position.max_profit * contracts:.0f} "
            f"max_loss=${position.max_loss * contracts:.0f} | "
            f"breakeven=${position.breakeven:.2f}"
        )

        self._log_trade(
            "OPTION_OPEN", signal.symbol, strategy_name,
            spot, contracts, signal,
            extra={
                "strategy": strategy_name,
                "legs":    [{"strike": l.strike, "type": l.option_type, "price": l.price, "delta": l.delta}
                            for l in position.legs],
                "max_profit": position.max_profit * contracts,
                "max_loss":   position.max_loss * contracts,
                "net_premium": position.net_premium * contracts,
            }
        )

    def _log_trade(
        self, event: str, symbol: str, action: str, price: float,
        size: int | float, signal: TradeSignal = None, pnl: float = None, extra: dict = None
    ):
        """Write trade to CSV journal."""
        import json

        row = {
            "timestamp":     datetime.datetime.now().isoformat(),
            "event":         event,
            "symbol":        symbol,
            "action":        action,
            "price":         round(price, 4),
            "size":          size,
            "pnl":           round(pnl, 2) if pnl else "",
            "portfolio_val": round(self.broker.getvalue(), 2),
            "conviction":    round(signal.conviction, 4) if signal else "",
            "quant_dir":     signal.quant.direction if signal and signal.quant else "",
            "ml_dir":        signal.ml.direction if signal and signal.ml else "",
            "sentiment":     round(signal.sentiment.score, 3) if signal and signal.sentiment else "",
            "sentiment_reason": signal.sentiment.reasoning if signal and signal.sentiment else "",
            "extra":         json.dumps(extra) if extra else "",
        }

        self.trade_log.append(row)

        with open(self._journal_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._journal_headers_written:
                writer.writeheader()
                self._journal_headers_written = True
            writer.writerow(row)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            symbol = self.order_map.get(order.ref, "?")
            logger.info(
                f"[FILL] {symbol} | size={order.executed.size} "
                f"price=${order.executed.price:.2f} comm=${order.executed.comm:.2f}"
            )

    def stop(self):
        final = self.broker.getvalue()
        ret   = (final / INITIAL_CASH - 1) * 100
        logger.info(f"Strategy complete | Final value: ${final:,.2f} | Return: {ret:.2f}%")


def inject_signals(signals: list[TradeSignal], strategy_instance):
    """
    Called before each bar to pre-load signals into the strategy's cache.
    Decouples signal generation (I/O heavy) from Backtrader's tight loop.
    """
    strategy_instance.signal_cache = {s.symbol: s for s in signals}
