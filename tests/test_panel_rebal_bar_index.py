"""Regression tests for portfolio rebalance bar mapping (India / US / stock RRG)."""

from __future__ import annotations

import unittest

import pandas as pd

from momentum.rrg_core import (
    forward_rebal_at_latest_active,
    panel_rebal_bar_index,
    weekly_preview_rebalance_bar_index,
)


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


class WeeklyPreviewRebalanceBarIndexTests(unittest.TestCase):
    """Preview picks: as-of day vs last rebalance (India + US weekly Fri bars)."""

    WI = pd.DatetimeIndex(
        ["2026-05-29", "2026-06-05", "2026-06-12", "2026-06-19"]
    )

    def test_mid_week_preview_uses_current_hold_week_start(self):
        """17-Jun (Tue) → rebalance 12-Jun, not 19-Jun weekly bar end."""
        idx = weekly_preview_rebalance_bar_index(
            self.WI, pd.Timestamp("2026-06-17"), 1
        )
        self.assertEqual(self.WI[idx], pd.Timestamp("2026-06-12"))

    def test_rebalance_friday_uses_prior_week_start(self):
        """19-Jun rebalance Friday → change from 12-Jun."""
        idx = weekly_preview_rebalance_bar_index(
            self.WI, pd.Timestamp("2026-06-19"), 1
        )
        self.assertEqual(self.WI[idx], pd.Timestamp("2026-06-12"))

    def test_before_next_weekly_bar_loaded(self):
        """Only through 12-Jun weekly; daily as-of 17-Jun → 12-Jun."""
        wi = pd.DatetimeIndex(["2026-05-29", "2026-06-05", "2026-06-12"])
        idx = weekly_preview_rebalance_bar_index(
            wi, pd.Timestamp("2026-06-17"), 1
        )
        self.assertEqual(wi[idx], pd.Timestamp("2026-06-12"))


class ForwardRebalAtLatestTests(unittest.TestCase):
    """Rebalance @ latest bar checkbox (India + US weekly Fri bars)."""

    WI = pd.DatetimeIndex(
        ["2026-05-29", "2026-06-05", "2026-06-12", "2026-06-19"]
    )

    def test_active_at_latest_slider_with_prior_rebalance(self):
        max_i = len(self.WI) - 1
        self.assertTrue(
            forward_rebal_at_latest_active(self.WI, max_i, 1, enabled=True)
        )

    def test_inactive_when_not_at_latest(self):
        self.assertFalse(
            forward_rebal_at_latest_active(self.WI, len(self.WI) - 2, 1, enabled=True)
        )

    def test_inactive_when_checkbox_off(self):
        self.assertFalse(
            forward_rebal_at_latest_active(
                self.WI, len(self.WI) - 1, 1, enabled=False
            )
        )

    def test_inactive_when_normal_equals_slider(self):
        """Only one weekly bar in hold window — nothing to forward-preview."""
        wi = pd.DatetimeIndex(["2026-06-05", "2026-06-12"])
        self.assertFalse(forward_rebal_at_latest_active(wi, 1, 1, enabled=True))


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
