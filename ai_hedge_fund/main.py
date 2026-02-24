"""
AI Hedge Fund — Main Entry Point
──────────────────────────────────
Usage:
    python main.py                    # paper trade (market hours only)
    python main.py --mode paper       # explicit paper mode
    python main.py --force            # run outside market hours (dev/test)
    python main.py --symbols SPY QQQ  # override universe
"""
import sys
import pathlib

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from loguru import logger
from config.settings import LOG_DIR
from src.execution.runner import run_session

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add(LOG_DIR / "trading_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="1 day")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Hedge Fund — Paper Trader")
    parser.add_argument("--mode",    default="paper", choices=["paper", "backtest"])
    parser.add_argument("--force",   action="store_true", help="Run outside market hours")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override universe")
    args = parser.parse_args()

    result = run_session(
        mode=args.mode,
        force=args.force,
        symbols=args.symbols,
    )

    if result:
        print(f"\n✅ Session complete")
        print(f"   Portfolio: ${result['final_value']:,.2f}")
        print(f"   Return:    {result['total_return']:+.2f}%")
        print(f"   Report:    {result['report_path']}")
