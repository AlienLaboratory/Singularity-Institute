"""Cost calculator for Claude Opus 5 tokens.

Prices (USD per 1M tokens), verified 2026-10-10:
  https://platform.claude.com/docs/en/about-claude/pricing
"""

from __future__ import annotations

import math
from typing import Union

# --- Verified Claude Opus 5 prices (USD per million tokens) ---
INPUT_PRICE = 5.00
OUTPUT_PRICE = 25.00
CACHE_READ_PRICE = 0.50
CACHE_WRITE_5MIN_PRICE = 6.25
CACHE_WRITE_1HR_PRICE = 10.00

_MILLION = 1_000_000


def _validate_tokens(value, name: str) -> int:
    """Ensure *value* is a finite, non-negative number and return it as int."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int or float, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float, got {type(value).__name__}")
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    if math.isinf(value):
        raise ValueError(f"{name} must not be infinite")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def compute_cost(
    input_tokens: Union[int, float],
    output_tokens: Union[int, float],
    cache_read_tokens: Union[int, float] = 0,
    cache_write_tokens: Union[int, float] = 0,
) -> float:
    """Compute total cost in USD using the **5-minute** cache-write rate.

    Parameters
    ----------
    input_tokens : int | float
    output_tokens : int | float
    cache_read_tokens : int | float  (default 0)
    cache_write_tokens : int | float (default 0, 5-minute rate)

    Returns
    -------
    float  Total cost in USD.
    """
    i = _validate_tokens(input_tokens, "input_tokens")
    o = _validate_tokens(output_tokens, "output_tokens")
    cr = _validate_tokens(cache_read_tokens, "cache_read_tokens")
    cw = _validate_tokens(cache_write_tokens, "cache_write_tokens")

    cost = (
        i * INPUT_PRICE
        + o * OUTPUT_PRICE
        + cr * CACHE_READ_PRICE
        + cw * CACHE_WRITE_5MIN_PRICE
    ) / _MILLION
    return cost


def compute_cost_1hr_cache_write(
    input_tokens: Union[int, float],
    output_tokens: Union[int, float],
    cache_read_tokens: Union[int, float] = 0,
    cache_write_tokens: Union[int, float] = 0,
) -> float:
    """Compute total cost in USD using the **1-hour** cache-write rate.

    Same signature as compute_cost but cache_write_tokens is priced at
    CACHE_WRITE_1HR_PRICE instead of the 5-minute rate.
    """
    i = _validate_tokens(input_tokens, "input_tokens")
    o = _validate_tokens(output_tokens, "output_tokens")
    cr = _validate_tokens(cache_read_tokens, "cache_read_tokens")
    cw = _validate_tokens(cache_write_tokens, "cache_write_tokens")

    cost = (
        i * INPUT_PRICE
        + o * OUTPUT_PRICE
        + cr * CACHE_READ_PRICE
        + cw * CACHE_WRITE_1HR_PRICE
    ) / _MILLION
    return cost
