# SPDX-FileCopyrightText: 2026 Tim Laverty
# SPDX-License-Identifier: MIT

"""Checks for timing, incomplete data, and the forward-return calendar."""
from datetime import datetime, timezone
import unittest

import numpy as np
import pandas as pd
from run_market_bottom import calculate, completed_session_cutoff, forward_metrics


class MarketBottomChecks(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2023-01-02', periods=400, name='date')
        self.prices = pd.DataFrame({
            'spx': np.arange(400, dtype=float) + 1000,
            'vix': 20., 'rsp': 100., 'spy': 400.,
        }, index=self.dates)
        self.spread = pd.Series(4., index=self.dates)

    def test_same_day_credit_observation_cannot_affect_score(self):
        original = calculate(self.prices, self.spread)
        changed = self.spread.copy()
        changed.iloc[350] = 50.
        result = calculate(self.prices, changed)
        pd.testing.assert_frame_equal(original.iloc[:351], result.iloc[:351])
        self.assertEqual(result.credit_observation_date.iloc[350], self.dates[349])
        self.assertEqual(result.hy_spread.iloc[351], 50.)

    def test_future_prices_cannot_affect_earlier_scores(self):
        complete = calculate(self.prices, self.spread)
        truncated = calculate(self.prices.iloc[:350], self.spread.iloc[:350])
        pd.testing.assert_frame_equal(complete.iloc[:350], truncated)

    def test_relative_breadth_also_requires_an_actual_rsp_recovery(self):
        scenarios = [
            ('both fall, RSP falls less', np.linspace(100., 80., 25),
             np.linspace(400., 240., 25), False),
            ('RSP recovers and outperforms', np.linspace(100., 120., 25),
             np.full(25, 400.), True),
            ('flat RSP equals its average while SPY falls', np.full(25, 100.),
             np.linspace(400., 320., 25), False),
        ]
        for name, rsp, spy, expected in scenarios:
            with self.subTest(name=name):
                prices = self.prices.copy()
                prices.loc[self.dates[-25:], 'rsp'] = rsp
                prices.loc[self.dates[-25:], 'spy'] = spy
                result = calculate(prices, self.spread)
                row = result.iloc[-1]
                # All scenarios pass the old relative-strength checks.
                self.assertGreater(row.breadth_ratio, row.breadth_mean_20)
                self.assertGreater(row.breadth_ratio, row.breadth_ratio_5_sessions_ago)
                self.assertTrue(row.score_available)
                self.assertEqual(bool(row.breadth_recovery), expected)
                self.assertEqual(row.score, int(expected))

    def test_missing_credit_does_not_become_zero_score_or_shorten_calendar(self):
        result = calculate(self.prices, self.spread.iloc[300:])
        self.assertEqual(len(result), len(self.prices))
        self.assertTrue(result.score.iloc[:311].isna().all())
        self.assertTrue(result.score.iloc[311:].notna().all())

    def test_forward_returns_enter_next_close_and_censor_incomplete_window(self):
        ft = forward_metrics(self.prices)
        self.assertAlmostEqual(ft.ret_21d.iloc[0], 1022 / 1001 - 1)
        self.assertEqual(ft.maxdd_21d.iloc[0], 0.)
        self.assertEqual(ft.ret_21d.notna().sum(), 378)
        self.assertTrue(ft.ret_21d.iloc[-22:].isna().all())
        falling = self.prices.copy()
        falling['spx'] = 1400 - np.arange(400)
        losses = forward_metrics(falling)
        self.assertAlmostEqual(losses.maxdd_21d.iloc[0], 1378 / 1399 - 1)

    def test_live_us_session_is_excluded(self):
        # 10:00 UTC on Nov 20 is before the US market opens.
        cutoff = completed_session_cutoff(datetime(2026, 11, 20, 10, tzinfo=timezone.utc))
        self.assertEqual(cutoff, pd.Timestamp('2026-11-19'))
        after_close = completed_session_cutoff(datetime(2026, 11, 20, 22, tzinfo=timezone.utc))
        self.assertEqual(after_close, pd.Timestamp('2026-11-20'))


if __name__ == '__main__':
    unittest.main()
