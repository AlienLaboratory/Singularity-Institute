"""In-memory cost accounting for a single research task.

This module builds on :mod:`cost` (which prices Claude Opus 5 tokens in USD)
and adds an independent, configurable estimate of GPU electricity cost in UAH.
It is strictly in-memory: it does not touch the database, agent, orchestrator,
configuration, or Web UI, and it does not persist anything.

Two clearly separated currency domains are produced and never mixed:

* **API-equivalent cost -> USD.**  What the same input/output token counts
  *would* cost on Anthropic's Opus 5 API (a billing-equivalence figure, not the
  money a local Ollama run actually spent).
* **GPU electricity cost -> UAH.**  An *estimate* derived from an assumed GPU
  power draw, the elapsed time, and a configurable UAH/kWh tariff.

Design rules implemented here:

* Missing token counts produce an **unavailable** cost (``None`` with a clear
  reason); a fabricated ``0`` is never returned for missing data.
* Negative, non-numeric, NaN, and infinite numeric inputs are **rejected**.
* GPU power is treated as an **assumption** (representative sustained draw),
  *not* as measured consumption.
* Anthropic cache token counts are **not invented**; the Ollama runtime reports
  prompt and generated token counts only, so those two are the only token
  inputs used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cost

__all__ = [
    "CostResult",
    "TaskAccounting",
    "calculate_api_cost_usd",
    "calculate_electricity_cost_uah",
    "DEFAULT_GPU_WATTS",
    "DEFAULT_TARIFF_UAH_PER_KWH",
]

# ---------------------------------------------------------------------------
# Configurable electricity defaults (UAH domain)
# ---------------------------------------------------------------------------
# ASSUMPTION, NOT MEASUREMENT: a representative sustained power draw (watts) for
# a single GPU under load.  This is an engineering estimate used only to
# approximate electricity cost; it is NOT a reading from a power meter and it
# is NOT per-task measured consumption.  Override it at call time if you know
# your actual draw.
DEFAULT_GPU_WATTS = 350.0

# Default electricity tariff in UAH per kWh (configurable at call time).
DEFAULT_TARIFF_UAH_PER_KWH = 4.32

# 1 kWh expressed in watt-seconds (joules): 1000 W * 3600 s = 3,600,000 W*s.
_WATT_SECONDS_PER_KWH = 3_600_000.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_quantity(name: str, value) -> float:
    """Validate a numeric quantity.

    Rejects ``bool``, non-numeric types, ``NaN``, ``inf``, and negative
    values; returns the value as ``float`` on success.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be an int or float, got {type(value).__name__}"
        )
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    if math.isinf(value):
        raise ValueError(f"{name} must not be infinite")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return float(value)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CostResult:
    """Outcome of a single cost calculation.

    value     -- the cost in ``currency``, or ``None`` when unavailable.
    currency  -- ``"USD"`` (API-equivalent) or ``"UAH"`` (electricity).
    available -- ``True`` when ``value`` is a real, computed number.
    reason    -- human-readable explanation for why the cost is unavailable.
    """

    value: Optional[float]
    currency: str
    available: bool
    reason: Optional[str] = None

    def as_usd(self) -> Optional[float]:
        """Return ``value`` only if this result is in USD."""
        if self.currency != "USD":
            raise ValueError(f"{self.currency} result cannot be read as USD")
        return self.value

    def as_uah(self) -> Optional[float]:
        """Return ``value`` only if this result is in UAH."""
        if self.currency != "UAH":
            raise ValueError(f"{self.currency} result cannot be read as UAH")
        return self.value


# ---------------------------------------------------------------------------
# Accounting data structure for a single task
# ---------------------------------------------------------------------------
@dataclass
class TaskAccounting:
    """Accounting record for a **single task** (in-memory only).

    Raw inputs
    ----------
    input_tokens    : actual prompt tokens, or ``None`` if the runtime did not
                      report them.
    output_tokens   : actual generated tokens, or ``None`` if not reported.
    elapsed_seconds : elapsed seconds for the task, or ``None`` if not measured.
                      ``0`` is a valid value (a task that consumed no time
                      legitimately costs no electricity).

    Computed (populated by :meth:`summarize`)
    -----------------------------------------
    api_cost_usd             : Opus 5-equivalent API cost in **USD**, or ``None``.
    api_cost_reason          : reason string when ``api_cost_usd`` is ``None``.
    electricity_cost_uah     : estimated GPU electricity cost in **UAH**, or ``None``.
    electricity_cost_reason  : reason string when ``electricity_cost_uah`` is ``None``.

    The USD and UAH figures live on separate fields and are never interchanged.
    """

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    elapsed_seconds: Optional[float] = None

    api_cost_usd: Optional[float] = None
    api_cost_reason: Optional[str] = None
    electricity_cost_uah: Optional[float] = None
    electricity_cost_reason: Optional[str] = None

    def summarize(
        self,
        gpu_watts: float = DEFAULT_GPU_WATTS,
        tariff_uah_per_kwh: float = DEFAULT_TARIFF_UAH_PER_KWH,
    ) -> "TaskAccounting":
        """Compute and store both the USD API-equivalent cost and the UAH
        electricity estimate for this task, then return ``self``."""
        api = calculate_api_cost_usd(self.input_tokens, self.output_tokens)
        self.api_cost_usd = api.value
        self.api_cost_reason = api.reason if not api.available else None

        elec = calculate_electricity_cost_uah(
            self.elapsed_seconds, gpu_watts, tariff_uah_per_kwh
        )
        self.electricity_cost_uah = elec.value
        self.electricity_cost_reason = elec.reason if not elec.available else None
        return self


# ---------------------------------------------------------------------------
# Cost calculations
# ---------------------------------------------------------------------------
def calculate_api_cost_usd(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> CostResult:
    """Compute the Opus 5-equivalent API cost in **USD**.

    Uses only the *actual* prompt (``input_tokens``) and generated
    (``output_tokens``) token counts that the Ollama runtime reports.  Anthropic
    cache token counts are deliberately **not** invented: :func:`cost.compute_cost`
    is called with its cache parameters left at their zero defaults.

    Missing token counts
    ---------------------
    If either token count is ``None``, the cost is **unavailable**: a
    ``CostResult`` with ``value=None`` and ``available=False`` (plus a clear
    reason) is returned.  A fabricated ``0`` is never produced.

    Invalid token counts
    --------------------
    Negative / non-numeric / NaN / infinite values are rejected by
    :func:`cost.compute_cost` (which raises), matching :mod:`cost`'s behaviour.

    Returns
    -------
    CostResult
        A USD result.
    """
    missing = []
    if input_tokens is None:
        missing.append("input_tokens")
    if output_tokens is None:
        missing.append("output_tokens")
    if missing:
        return CostResult(
            value=None,
            currency="USD",
            available=False,
            reason="unavailable: missing " + " and ".join(missing),
        )

    # Delegate to cost.compute_cost, which validates and prices the tokens.
    value = cost.compute_cost(input_tokens, output_tokens)
    return CostResult(value=value, currency="USD", available=True)


def calculate_electricity_cost_uah(
    elapsed_seconds: Optional[float],
    gpu_watts: float = DEFAULT_GPU_WATTS,
    tariff_uah_per_kwh: float = DEFAULT_TARIFF_UAH_PER_KWH,
) -> CostResult:
    """Estimate the GPU electricity cost for a task in **UAH**.

    Formula
    -------
        gpu_watts * elapsed_seconds / 3,600,000 * (UAH per kWh)

    (``watts * seconds`` is watt-seconds = joules; dividing by 3,600,000
    converts to kWh; multiplying by the UAH/kWh tariff yields UAH.)

    Parameters
    ----------
    elapsed_seconds : Optional[float]
        Elapsed time in seconds.  ``0`` is valid and yields ``0.0`` UAH.
        ``None`` means the time was not measured -> cost unavailable.
    gpu_watts : float, optional
        GPU power draw in watts.  An **assumption**, not a measured value
        (see :data:`DEFAULT_GPU_WATTS`).
    tariff_uah_per_kwh : float, optional
        Electricity tariff in UAH per kWh.

    Returns
    -------
    CostResult
        A UAH result.

    Raises
    ------
    TypeError / ValueError
        If any of ``gpu_watts``, ``tariff_uah_per_kwh``, or
        ``elapsed_seconds`` is negative, non-numeric, NaN, or infinite.
    """
    if elapsed_seconds is None:
        return CostResult(
            value=None,
            currency="UAH",
            available=False,
            reason="unavailable: elapsed_seconds not provided",
        )

    watts = _validate_quantity("gpu_watts", gpu_watts)
    tariff = _validate_quantity("tariff_uah_per_kwh", tariff_uah_per_kwh)
    seconds = _validate_quantity("elapsed_seconds", elapsed_seconds)

    uah = (watts * seconds) / _WATT_SECONDS_PER_KWH * tariff
    return CostResult(value=uah, currency="UAH", available=True)
