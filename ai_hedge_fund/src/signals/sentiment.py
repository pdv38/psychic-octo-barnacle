"""
LLM Sentiment Signal
─────────────────────
Uses GPT-4o-mini or Claude Haiku to classify news headlines → sentiment score.
Logs the full chain-of-thought reasoning (great for mentor demos).

Output: float in [-1, +1] and reasoning string.
"""
import json
from dataclasses import dataclass
from loguru import logger

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from config.settings import (
    LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY
)


SENTIMENT_PROMPT = """You are a quantitative analyst at a hedge fund evaluating news sentiment for trading.

Analyze the following news headlines for {symbol} and return a JSON object with:
1. "score": float from -1.0 (very bearish) to +1.0 (very bullish), 0.0 = neutral
2. "confidence": float from 0.0 to 1.0 (how certain you are)
3. "reasoning": 2-3 sentence explanation of the key factors
4. "catalysts": list of up to 3 key bullish/bearish catalysts found
5. "timeframe": "immediate" | "short_term" | "long_term" (expected impact horizon)

Be rigorous. Consider: earnings surprises, macro headwinds, sector rotation, regulatory risk,
management changes, competitive dynamics. Ignore clickbait.

Headlines:
{headlines}

Respond ONLY with valid JSON. No other text."""


@dataclass
class SentimentSignal:
    score: float       # -1 to +1
    confidence: float  # 0–1
    reasoning: str
    catalysts: list
    timeframe: str
    raw_response: str


class SentimentEngine:
    """LLM-powered news sentiment for equity signal generation."""

    def __init__(self):
        self._openai_client  = None
        self._anthropic_client = None
        self._setup_client()

    def _setup_client(self):
        if LLM_PROVIDER == "openai" and OPENAI_AVAILABLE and OPENAI_API_KEY:
            self._openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info("Sentiment engine: OpenAI GPT-4o-mini")
        elif LLM_PROVIDER == "anthropic" and ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
            self._anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Sentiment engine: Anthropic Claude Haiku")
        else:
            logger.warning(
                "No LLM API key found. Sentiment signal will be neutral (0). "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"
            )

    def analyze(self, symbol: str, headlines: list[dict]) -> SentimentSignal:
        """Classify news headlines for a symbol. Returns SentimentSignal."""
        null_signal = SentimentSignal(0.0, 0.0, "No LLM available", [], "immediate", "")

        if not headlines:
            return SentimentSignal(0.0, 0.0, "No headlines", [], "immediate", "")

        if self._openai_client is None and self._anthropic_client is None:
            return null_signal

        # Format headlines
        headline_text = "\n".join([
            f"- [{h.get('publisher','?')}] {h.get('title','')}"
            for h in headlines[:8]
        ])

        prompt = SENTIMENT_PROMPT.format(symbol=symbol, headlines=headline_text)

        try:
            raw = self._call_llm(prompt)
            return self._parse_response(raw)
        except Exception as e:
            logger.error(f"Sentiment LLM error for {symbol}: {e}")
            return null_signal

    def _call_llm(self, prompt: str) -> str:
        if self._openai_client:
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return resp.choices[0].message.content

        elif self._anthropic_client:
            resp = self._anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        raise RuntimeError("No LLM client available")

    def _parse_response(self, raw: str) -> SentimentSignal:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        data = json.loads(text.strip())
        return SentimentSignal(
            score      = float(data.get("score", 0.0)),
            confidence = float(data.get("confidence", 0.5)),
            reasoning  = data.get("reasoning", ""),
            catalysts  = data.get("catalysts", []),
            timeframe  = data.get("timeframe", "short_term"),
            raw_response = raw,
        )


# Singleton
sentiment_engine = SentimentEngine()
