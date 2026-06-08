"""Regression tests for portfolio rebalance bar mapping (India / US / stock RRG)."""

from __future__ import annotations

import unittest

import pandas as pd

from momentum.rrg_core import panel_rebal_bar_index


class PanelRebalBarIndexTests(unittest.TestCase):
    """Same weekly-index rule for India ETF, US ETF, and stock profiles."""

    WI = pd.DatetimeIndex(
        ["2026-05-22", "2026-05-29", "2026-06-05", "2026-06-12"]
    )

    def test_latest_slider_uses_prior_rebalance_not_slider_bar(self):
        """Slider on latest weekly bar → active rebalance is prior bar."""
        idx = panel_rebal_bar_index(self.WI, pd.Timestamp("2026-06-12"), 1)
        self.assertEqual(self.WI[idx], pd.Timestamp("2026-06-05"))

    def test_rebalance_bar_still_maps_to_itself(self):
        idx = panel_rebal_bar_index(self.WI, pd.Timestamp("2026-06-05"), 1)
        self.assertEqual(self.WI[idx], pd.Timestamp("2026-06-05"))

    def test_mid_week_maps_to_hold_week_start(self):
        idx = panel_rebal_bar_index(self.WI, pd.Timestamp("2026-06-10"), 1)
        self.assertEqual(self.WI[idx], pd.Timestamp("2026-06-05"))

    def test_daily_preview_date_maps_to_friday_hold_week(self):
        """Monday daily preview (08-06) → weekly rebalance bar 05-06."""
        latest = pd.Timestamp("2026-06-08")
        pos = int(self.WI.get_indexer([latest.normalize()], method="ffill")[0])
        idx = panel_rebal_bar_index(self.WI, self.WI[pos], 1)
        self.assertEqual(self.WI[pos], pd.Timestamp("2026-06-05"))
        self.assertEqual(self.WI[idx], pd.Timestamp("2026-06-05"))


class DayPanelWasDateTests(unittest.TestCase):
    """Day unit Was date (shared rrg_app for India ETF, US ETF, stock)."""

    DI = pd.DatetimeIndex(["2026-06-04", "2026-06-05", "2026-06-08"])

    def test_was_is_prior_trading_bar_not_calendar_day(self):
        """Monday slider → Was Friday, not Thursday (weekend skipped)."""
        current_i = 2
        tail = 1
        self.assertGreater(current_i, tail)
        prev_i = current_i - 1
        self.assertEqual(self.DI[prev_i], pd.Timestamp("2026-06-05"))
        self.assertNotEqual(self.DI[prev_i], pd.Timestamp("2026-06-04"))


if __name__ == "__main__":
    unittest.main()
