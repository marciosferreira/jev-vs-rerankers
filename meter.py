"""Per-request cost and token meter.

Wrap code in `with metered() as m:`; every OpenRouter call made in that context (same thread)
adds its real cost (usage.cost reported by OpenRouter) and token counts to `m`. Calls made
outside a metered block (e.g. the judge) are not recorded.
"""

import contextvars
from contextlib import contextmanager

_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar("meter", default=None)

KINDS = ("embed", "jev", "llm")


def new_meter() -> dict:
    return {
        "cost": {k: 0.0 for k in KINDS},
        "calls": {k: 0 for k in KINDS},
        "tokens_in": {k: 0 for k in KINDS},
        "tokens_out": {k: 0 for k in KINDS},
        "missing_cost": 0,  # calls whose response had no usage.cost
    }


@contextmanager
def metered():
    m = new_meter()
    token = _current.set(m)
    try:
        yield m
    finally:
        _current.reset(token)


def bind(fn):
    """Wrap fn so it records into the current meter even when run in another thread
    (worker threads do not inherit context variables)."""
    m = _current.get()

    def wrapper(*args, **kwargs):
        token = _current.set(m)
        try:
            return fn(*args, **kwargs)
        finally:
            _current.reset(token)

    return wrapper


def record(kind: str, usage: dict | None) -> None:
    m = _current.get()
    if m is None:
        return
    usage = usage or {}
    m["calls"][kind] += 1
    m["tokens_in"][kind] += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    m["tokens_out"][kind] += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    if usage.get("cost") is None:
        m["missing_cost"] += 1
    else:
        m["cost"][kind] += usage["cost"]


def total(m: dict) -> float:
    return sum(m["cost"].values())
