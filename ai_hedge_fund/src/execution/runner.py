"""
Execution Runner
─────────────────
Entry point for a live paper-trading session.
1. Fetch signals for the universe (pre-market)
2. Build Backtrader data feeds
3. Run strategy for the day
4. Generate & publish report
"""
import datetime
import pytz
import backtrader as bt
from loguru import logger

from config.settings import (
    UNIVERSE, INITIAL_CASH, COMMISSION, SLIPPAGE,
    PAPER_MODE, LOG_DIR, REPORT_DIR
)
from src.data.market_data import market_data
from src.signals.ensemble import ensemble
from src.execution.strategy import AIHedgeFundStrategy, inject_signals
from src.reporting.report import generate_daily_report


ET = pytz.timezone("America/New_York")


def is_market_hours() -> bool:
    now = datetime.datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= now <= close_time


def run_session(
    mode: str = "paper",
    force: bool = False,
    symbols: list = None,
):
    """
    Run one full trading session.
    
    Args:
        mode:    "paper" (default) | "backtest"
        force:   Run even outside market hours (for testing)
        symbols: Override universe
    """
    logger.info(f"=== AI Hedge Fund Session Start | mode={mode} ===")

    if not force and not is_market_hours():
        logger.warning("Outside market hours. Use --force to override.")
        return

    symbols = symbols or UNIVERSE
    logger.info(f"Universe: {symbols}")

    # ── 1. Pre-market signal generation ───────────────────────────────────────
    logger.info("Generating signals for universe...")
    signals = ensemble.run_universe(symbols)
    actionable = [s for s in signals if s.action != "FLAT"]
    logger.info(
        f"Signals: {len(actionable)}/{len(signals)} actionable | "
        + ", ".join(f"{s.symbol}:{s.action}({s.conviction:.2f})" for s in actionable)
    )

    # ── 2. Fetch price data for Backtrader ────────────────────────────────────
    logger.info("Fetching OHLCV data...")
    bars = market_data.get_daily_bars(symbols, lookback_days=300)

    # ── 3. Build Cerebro ──────────────────────────────────────────────────────
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(perc=SLIPPAGE)

    # Add data feeds
    feeds_added = 0
    for sym in symbols:
        df = bars.get(sym)
        if df is None or df.empty:
            logger.warning(f"No data for {sym}, skipping")
            continue

        data_feed = bt.feeds.PandasData(
            dataname=df,
            name=sym,
            open="open", high="high", low="low", close="close", volume="volume",
        )
        cerebro.adddata(data_feed)
        feeds_added += 1

    if feeds_added == 0:
        logger.error("No data feeds available. Aborting.")
        return

    # ── 4. Add strategy with pre-loaded signals ────────────────────────────────
    cerebro.addstrategy(AIHedgeFundStrategy, universe=symbols)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name="sharpe",   riskfreerate=0.053)
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns,       _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # ── 5. Run ─────────────────────────────────────────────────────────────────
    logger.info(f"Running Backtrader with {feeds_added} symbols...")
    results = cerebro.run()
    strat = results[0]

    # Inject signals into strategy (called after init, before first next())
    inject_signals(signals, strat)

    # ── 6. Extract analytics ───────────────────────────────────────────────────
    try:
        sharpe   = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
        drawdown = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0)
        ret_data = strat.analyzers.returns.get_analysis()
        trade_an = strat.analyzers.trades.get_analysis()
    except Exception as e:
        logger.warning(f"Analyzer extraction error: {e}")
        sharpe = drawdown = ret_data = trade_an = None

    final_value = cerebro.broker.getvalue()
    total_return = (final_value / INITIAL_CASH - 1) * 100

    logger.info(f"Session complete | Final: ${final_value:,.2f} | Return: {total_return:.2f}%")
    if sharpe:
        logger.info(f"Sharpe: {sharpe:.3f} | Max DD: {drawdown:.2f}%")

    # ── 7. Generate report ─────────────────────────────────────────────────────
    report_path = generate_daily_report(
        signals=signals,
        trade_log=strat.trade_log,
        final_value=final_value,
        initial_value=INITIAL_CASH,
        sharpe=sharpe,
        drawdown=drawdown,
    )
    logger.info(f"Report saved: {report_path}")

    return {
        "final_value": final_value,
        "total_return": total_return,
        "sharpe": sharpe,
        "drawdown": drawdown,
        "signals": [s.to_log_dict() for s in signals],
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Hedge Fund — Paper Trader")
    parser.add_argument("--mode",    default="paper",  choices=["paper", "backtest"])
    parser.add_argument("--force",   action="store_true", help="Run outside market hours")
    parser.add_argument("--symbols", nargs="+", default=None)
    args = parser.parse_args()

    run_session(mode=args.mode, force=args.force, symbols=args.symbols)
