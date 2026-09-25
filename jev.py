"""Minimal client for TypeSafe JEV via OpenRouter's decisions endpoint."""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import meter

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env = Path(__file__).with_name(".env")
    for line in env.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip().lower() in ("apikey", "openrouter_api_key"):
            return value.strip().strip('"').strip("'")
    raise RuntimeError("OpenRouter API key not found in env or .env")


def decide(state: str, questions: dict, model: str = MODEL) -> dict:
    """Send raw questions to JEV and return the full response."""
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.load(resp)
            meter.record("jev", result.get("usage"))
            return result
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524) or attempt == 4:
                raise RuntimeError(f"JEV request failed ({e.code}): {e.read().decode()}") from e
        except OSError:  # timeouts, DNS failures, connection resets
            if attempt == 4:
                raise
        time.sleep(min(30, 2 ** (attempt + 1)))


def noul(state: str, instructions: str) -> float:
    """Yes/no question. Returns a value in [0, 1] (1 = true)."""
    answer = decide(state, {"q": {"type": "noul", "instructions": instructions}})
    return answer["answers"]["q"]["noul"]


def choice(state: str, instructions: str, criteria: dict[str, str]) -> dict:
    """Multiple choice. Returns {"choice", "probabilities", "confidence"}."""
    answer = decide(
        state,
        {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}},
    )
    return answer["answers"]["q"]


def score(state: str, instructions: str, criteria: list[str]) -> dict:
    """Ordinal scale over criteria indices.

    Returns {"score", "probabilities", "confidence", "legend"}; "score" is the
    probability-weighted mean (float), e.g. 1.98.
    """
    answer = decide(
        state,
        {"q": {"type": "score", "instructions": instructions, "criteria": criteria}},
    )
    return answer["answers"]["q"]
