"""
Focused tests for accounting.py (in-memory task accounting).

Covers:
    - normal API-equivalent (USD) cost calculation
    - normal GPU electricity (UAH) cost calculation
    - zero elapsed time -> 0.0 UAH (valid), and missing elapsed -> unavailable
    - configurable GPU power (watts) and electricity tariff (UAH/kWh)
    - missing token metadata -> unavailable USD cost (None + reason), never 0
    - invalid inputs (negative, non-numeric, NaN, infinite) are rejected
    - currency units are kept distinct (USD vs UAH)

Run from the institute root:
    python -m py_compile accounting.py tests/test_accounting.py
    python -m unittest tests.test_accounting -v
"""

import math
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import accounting
from accounting import (
    CostResult,
    TaskAccounting,
    calculate_api_cost_usd,
    calculate_electricity_cost_uah,
    DEFAULT_GPU_WATTS,
    DEFAULT_TARIFF_UAH_PER_KWH,
)


class TestApiCostUsd(unittest.TestCase):
    def test_normal_calculation(self):
        # 10,000 input tokens, 4,000 output tokens at Opus 5 rates.
        # 10k * $5/M + 4k * $25/M = $0.05 + $0.10 = $0.15
        res = calculate_api_cost_usd(10_000, 4_000)
        self.assertEqual(res.currency, "USD")
        self.assertTrue(res.available)
        self.assertIsNotNone(res.value)
        self.assertAlmostEqual(res.value, 0.15, places=12)
        # as_usd() is the sanctioned USD accessor
        self.assertAlmostEqual(res.as_usd(), 0.15, places=12)

    def test_zero_tokens_is_available_zero(self):
        # Zero tokens is a *valid* count (a task that used no tokens), so it
        # yields a real, available $0.0 -- this is distinct from missing data.
        res = calculate_api_cost_usd(0, 0)
        self.assertTrue(res.available)
        self.assertEqual(res.value, 0.0)

    def test_missing_input_tokens_unavailable(self):
        res = calculate_api_cost_usd(None, 4_000)
        self.assertEqual(res.currency, "USD")
        self.assertFalse(res.available)
        self.assertIsNone(res.value)
        self.assertIsNotNone(res.reason)
        self.assertIn("input_tokens", res.reason)

    def test_missing_output_tokens_unavailable(self):
        res = calculate_api_cost_usd(10_000, None)
        self.assertFalse(res.available)
        self.assertIsNone(res.value)
        self.assertIn("output_tokens", res.reason)

    def test_missing_both_tokens_unavailable(self):
        res = calculate_api_cost_usd(None, None)
        self.assertFalse(res.available)
        self.assertIsNone(res.value)
        self.assertIn("input_tokens", res.reason)
        self.assertIn("output_tokens", res.reason)

    def test_missing_token_cost_is_not_fabricated_zero(self):
        # Regression guard: unavailable must be None, never a fabricated 0.0.
        res = calculate_api_cost_usd(None, 4_000)
        self.assertFalse(res.available)
        self.assertIsNone(res.value)


class TestElectricityCostUah(unittest.TestCase):
    def test_normal_calculation_with_defaults(self):
        # 350 W for 3,600 s (1 hour) = 350 kWh? No: 350 W * 3600 s / 3.6e6 =
        # 0.35 kWh. * 4.32 UAH/kWh = 1.512 UAH.
        res = calculate_electricity_cost_uah(3600.0)
        self.assertEqual(res.currency, "UAH")
        self.assertTrue(res.available)
        self.assertAlmostEqual(res.value, 1.512, places=9)
        self.assertAlmostEqual(res.as_uah(), 1.512, places=9)

    def test_zero_elapsed_time_is_available_zero(self):
        # 0 seconds is valid -> 0.0 UAH (a task that consumed no time).
        res = calculate_electricity_cost_uah(0.0)
        self.assertTrue(res.available)
        self.assertEqual(res.value, 0.0)

    def test_missing_elapsed_is_unavailable(self):
        res = calculate_electricity_cost_uah(None)
        self.assertEqual(res.currency, "UAH")
        self.assertFalse(res.available)
        self.assertIsNone(res.value)
        self.assertIsNotNone(res.reason)
        self.assertIn("elapsed_seconds", res.reason)

    def test_configurable_power_and_tariff(self):
        # 100 W * 3600 s / 3.6e6 = 0.1 kWh * 10 UAH/kWh = 1.0 UAH
        res = calculate_electricity_cost_uah(3600.0, 100.0, 10.0)
        self.assertAlmostEqual(res.value, 1.0, places=12)

    def test_configurable_defaults_values(self):
        # Verify the module exposes the required configurable defaults.
        self.assertEqual(DEFAULT_GPU_WATTS, 350.0)
        self.assertEqual(DEFAULT_TARIFF_UAH_PER_KWH, 4.32)

    def test_whole_kwh(self):
        # 1000 W for 3600 s = exactly 1 kWh * 2.0 UAH/kWh = 2.0 UAH
        res = calculate_electricity_cost_uah(3600.0, 1000.0, 2.0)
        self.assertAlmostEqual(res.value, 2.0, places=12)


class TestInvalidInputs(unittest.TestCase):
    def test_negative_input_tokens_rejected(self):
        with self.assertRaises(ValueError):
            calculate_api_cost_usd(-1, 100)

    def test_negative_output_tokens_rejected(self):
        with self.assertRaises(ValueError):
            calculate_api_cost_usd(100, -1)

    def test_nan_input_tokens_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            calculate_api_cost_usd(math.nan, 100)

    def test_infinite_input_tokens_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            calculate_api_cost_usd(math.inf, 100)

    def test_non_numeric_input_tokens_rejected(self):
        with self.assertRaises(TypeError):
            calculate_api_cost_usd("100", 100)

    def test_bool_input_tokens_rejected(self):
        with self.assertRaises(TypeError):
            calculate_api_cost_usd(True, 100)

    def test_negative_elapsed_rejected(self):
        with self.assertRaises(ValueError):
            calculate_electricity_cost_uah(-1.0)

    def test_nan_elapsed_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            calculate_electricity_cost_uah(math.nan)

    def test_infinite_elapsed_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            calculate_electricity_cost_uah(math.inf)

    def test_non_numeric_elapsed_rejected(self):
        with self.assertRaises(TypeError):
            calculate_electricity_cost_uah("3600")

    def test_negative_gpu_watts_rejected(self):
        with self.assertRaises(ValueError):
            calculate_electricity_cost_uah(3600.0, -350.0)

    def test_non_numeric_gpu_watts_rejected(self):
        with self.assertRaises(TypeError):
            calculate_electricity_cost_uah(3600.0, "350")

    def test_nan_tariff_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            calculate_electricity_cost_uah(3600.0, 350.0, math.nan)

    def test_negative_tariff_rejected(self):
        with self.assertRaises(ValueError):
            calculate_electricity_cost_uah(3600.0, 350.0, -4.32)


class TestCurrencySeparation(unittest.TestCase):
    def test_api_result_is_usd_only(self):
        res = calculate_api_cost_usd(10_000, 4_000)
        self.assertEqual(res.currency, "USD")
        self.assertAlmostEqual(res.as_usd(), 0.15, places=12)
        with self.assertRaises(ValueError):
            res.as_uah()

    def test_electricity_result_is_uah_only(self):
        res = calculate_electricity_cost_uah(3600.0)
        self.assertEqual(res.currency, "UAH")
        self.assertAlmostEqual(res.as_uah(), 1.512, places=9)
        with self.assertRaises(ValueError):
            res.as_usd()


class TestTaskAccounting(unittest.TestCase):
    def test_summarize_populates_both_costs(self):
        ta = TaskAccounting(input_tokens=10_000, output_tokens=4_000,
                            elapsed_seconds=3600.0)
        ta.summarize()
        self.assertAlmostEqual(ta.api_cost_usd, 0.15, places=12)
        self.assertAlmostEqual(ta.electricity_cost_uah, 1.512, places=9)
        # Reasons are cleared for available values
        self.assertIsNone(ta.api_cost_reason)
        self.assertIsNone(ta.electricity_cost_reason)

    def test_summarize_with_missing_data(self):
        ta = TaskAccounting(input_tokens=None, output_tokens=4_000,
                            elapsed_seconds=None)
        ta.summarize()
        self.assertIsNone(ta.api_cost_usd)
        self.assertIsNotNone(ta.api_cost_reason)
        self.assertIn("input_tokens", ta.api_cost_reason)
        self.assertIsNone(ta.electricity_cost_uah)
        self.assertIsNotNone(ta.electricity_cost_reason)
        self.assertIn("elapsed_seconds", ta.electricity_cost_reason)

    def test_default_construction_has_no_costs(self):
        ta = TaskAccounting()
        self.assertIsNone(ta.api_cost_usd)
        self.assertIsNone(ta.electricity_cost_uah)


if __name__ == "__main__":
    unittest.main(verbosity=2)
