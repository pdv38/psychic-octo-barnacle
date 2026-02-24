"""
Daily Report Generator
───────────────────────
Generates an HTML performance report after each session.
Published to docs/ for GitHub Pages.
"""
import datetime
import json
import pathlib
from loguru import logger
from jinja2 import Template

from config.settings import REPORT_DIR, BASE_DIR


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Hedge Fund — Daily Report {{ date }}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
  :root { --bg:#08090d; --card:#12141c; --border:#1e2130; --accent:#00e5a0; --text:#e8eaf0; --muted:#6b7280; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Syne',sans-serif; padding:40px 24px; max-width:960px; margin:0 auto; }
  h1 { font-size:2rem; font-weight:800; }
  h1 span { color:var(--accent); }
  .meta { font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--muted); margin-top:8px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; margin:32px 0; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }
  .card-label { font-family:'Space Mono',monospace; font-size:0.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
  .card-value { font-size:1.6rem; font-weight:800; }
  .pos { color:#00e5a0; } .neg { color:#ff4757; } .neu { color:#e8eaf0; }
  table { width:100%; border-collapse:collapse; margin-top:24px; }
  th { font-family:'Space Mono',monospace; font-size:0.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; padding:8px 12px; text-align:left; border-bottom:1px solid var(--border); }
  td { font-family:'Space Mono',monospace; font-size:0.7rem; padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.04); }
  tr:hover td { background:rgba(255,255,255,0.02); }
  .section-title { font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--muted); letter-spacing:0.15em; text-transform:uppercase; margin:32px 0 12px; display:flex; align-items:center; gap:10px; }
  .section-title::after { content:''; flex:1; height:1px; background:var(--border); }
  .badge { display:inline-block; padding:3px 8px; border-radius:4px; font-family:'Space Mono',monospace; font-size:0.6rem; border:1px solid; }
  .badge-buy  { background:rgba(0,229,160,0.1); border-color:rgba(0,229,160,0.3); color:#00e5a0; }
  .badge-sell { background:rgba(255,71,87,0.1);  border-color:rgba(255,71,87,0.3);  color:#ff4757; }
  .badge-flat { background:rgba(107,114,128,0.1); border-color:rgba(107,114,128,0.3); color:#6b7280; }
  footer { margin-top:48px; font-family:'Space Mono',monospace; font-size:0.6rem; color:var(--muted); border-top:1px solid var(--border); padding-top:16px; }
</style>
</head>
<body>
  <h1>AI <span>Hedge Fund</span></h1>
  <div class="meta">Daily Report · {{ date }} · Paper Trading Mode</div>

  <div class="grid">
    <div class="card">
      <div class="card-label">Portfolio Value</div>
      <div class="card-value neu">${{ "{:,.0f}".format(final_value) }}</div>
    </div>
    <div class="card">
      <div class="card-label">Daily Return</div>
      <div class="card-value {{ 'pos' if total_return >= 0 else 'neg' }}">{{ "{:+.2f}".format(total_return) }}%</div>
    </div>
    <div class="card">
      <div class="card-label">Sharpe Ratio</div>
      <div class="card-value {{ 'pos' if sharpe and sharpe > 1 else 'neu' }}">{{ "{:.2f}".format(sharpe) if sharpe else "N/A" }}</div>
    </div>
    <div class="card">
      <div class="card-label">Max Drawdown</div>
      <div class="card-value {{ 'neg' if drawdown and drawdown > 0 else 'neu' }}">{{ "{:.2f}".format(drawdown) if drawdown else "0.00" }}%</div>
    </div>
    <div class="card">
      <div class="card-label">Signals Generated</div>
      <div class="card-value neu">{{ signals | length }}</div>
    </div>
    <div class="card">
      <div class="card-label">Trades Executed</div>
      <div class="card-value neu">{{ trades | length }}</div>
    </div>
  </div>

  <div class="section-title">Signal Summary</div>
  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Action</th><th>Conviction</th><th>Quant</th><th>ML</th><th>Sentiment</th><th>Instrument</th><th>Reasoning</th>
      </tr>
    </thead>
    <tbody>
      {% for s in signals %}
      <tr>
        <td><strong>{{ s.symbol }}</strong></td>
        <td><span class="badge badge-{{ s.action.lower() }}">{{ s.action }}</span></td>
        <td>{{ "{:.2f}".format(s.conviction) }}</td>
        <td>{{ s.quant_dir if s.quant_dir is not none else "—" }}</td>
        <td>{{ s.ml_dir if s.ml_dir is not none else "—" }}</td>
        <td>{{ "{:+.2f}".format(s.sentiment_score) if s.sentiment_score else "—" }}</td>
        <td>{{ s.option_strategy if s.option_strategy else "equity" }}</td>
        <td style="font-size:0.58rem; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{{ s.sentiment_reasoning }}">{{ s.sentiment_reasoning[:80] + "…" if s.sentiment_reasoning and s.sentiment_reasoning|length > 80 else (s.sentiment_reasoning or "—") }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  {% if trades %}
  <div class="section-title">Trade Journal</div>
  <table>
    <thead>
      <tr><th>Time</th><th>Symbol</th><th>Event</th><th>Price</th><th>Size</th><th>P&L</th></tr>
    </thead>
    <tbody>
      {% for t in trades %}
      <tr>
        <td>{{ t.timestamp[:19] }}</td>
        <td>{{ t.symbol }}</td>
        <td>{{ t.event }} / {{ t.action }}</td>
        <td>${{ t.price }}</td>
        <td>{{ t.size }}</td>
        <td class="{{ 'pos' if t.pnl and t.pnl|float > 0 else ('neg' if t.pnl and t.pnl|float < 0 else '') }}">{{ "${:,.2f}".format(t.pnl|float) if t.pnl else "—" }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  <footer>
    AI Hedge Fund · Paper Trading · Built with Backtrader + QuantLib + LightGBM · Generated {{ date }}
  </footer>
</body>
</html>"""


def generate_daily_report(
    signals: list,
    trade_log: list,
    final_value: float,
    initial_value: float,
    sharpe: float = None,
    drawdown: float = None,
) -> pathlib.Path:
    """Generate and save the daily HTML report."""
    today = datetime.date.today().isoformat()
    total_return = (final_value / initial_value - 1) * 100

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        date=today,
        final_value=final_value,
        total_return=total_return,
        sharpe=sharpe,
        drawdown=drawdown,
        signals=[s.to_log_dict() for s in signals],
        trades=trade_log,
    )

    # Save to reports/
    report_path = REPORT_DIR / f"report_{today}.html"
    report_path.write_text(html)

    # Also write to docs/ for GitHub Pages
    docs_dir = BASE_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "index.html").write_text(html)
    (docs_dir / f"report_{today}.html").write_text(html)

    # Write summary JSON for CI badge / monitoring
    summary = {
        "date":         today,
        "final_value":  round(final_value, 2),
        "total_return": round(total_return, 4),
        "sharpe":       round(sharpe, 4) if sharpe else None,
        "drawdown":     round(drawdown, 4) if drawdown else None,
        "n_signals":    len(signals),
        "n_trades":     len(trade_log),
    }
    (docs_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(f"Daily report written to {report_path}")
    return report_path
