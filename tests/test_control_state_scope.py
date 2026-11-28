"""Check independent state-rate denominators and complete pre-policy rankings."""

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from analysis_data import county_panel
from test_analysis_missingness import arcos_fixture, population_fixture, mortality_fixture

spec = importlib.util.spec_from_file_location("control_states", ROOT / "code/04_control_states.py")
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


class ControlScopeTests(unittest.TestCase):
    def test_state_rates_use_each_outcomes_own_reporting_population(self):
        county = county_panel(arcos_fixture(), population_fixture(), mortality_fixture())
        state = controls.state_series(county).set_index(["state", "year"])
        fl = state.loc[("FL", 2006)]
        self.assertAlmostEqual(fl.mme_per_capita, 140 / 320_000)
        self.assertAlmostEqual(fl.deaths_per_100k, 12 / 350_000 * 1e5)
        self.assertEqual(fl.n_transactions, 11)
        self.assertEqual(fl.n_missing_mme, 8)
        self.assertEqual(fl.n_county_years, 5)
        self.assertTrue(pd.isna(state.loc[("WA", 2006), "mme_per_capita"]))
        self.assertTrue(pd.isna(state.loc[("WA", 2006), "deaths_per_100k"]))

    def test_wholly_missing_state_mme_is_unavailable_with_counters_retained(self):
        county = county_panel(arcos_fixture(), population_fixture(), mortality_fixture())
        county = county.loc[county.fips.isin(["12003", "12111"])]
        state = controls.state_series(county).set_index(["state", "year"])
        self.assertTrue(pd.isna(state.loc[("FL", 2006), "mme_per_capita"]))
        self.assertEqual(state.loc[("FL", 2006), "n_transactions"], 6)
        self.assertEqual(state.loc[("FL", 2006), "n_missing_mme"], 6)

    def test_candidate_with_missing_pre_year_is_not_ranked(self):
        panel = pd.DataFrame([
            (state, year, float(year - 2000))
            for state in ["FL", "CA", "PA"]
            for year in range(2006, 2010)
            if not (state == "PA" and year == 2007)
        ], columns=["state", "year", "mme_per_capita"])
        gap = controls.trend_distance(panel, "mme_per_capita", "FL", 2010)
        self.assertEqual(gap.index.tolist(), ["CA"])
        self.assertAlmostEqual(gap.loc["CA"], 0)

    def test_year_missing_from_every_state_cannot_silently_shorten_preperiod(self):
        panel = pd.DataFrame([
            (state, year, 1.) for state in ["FL", "CA"] for year in (2006, 2008, 2009)
        ], columns=["state", "year", "mme_per_capita"])
        with self.assertRaisesRegex(ValueError, "complete, nonzero"):
            controls.trend_distance(panel, "mme_per_capita", "FL", 2010)

    def test_zero_preperiod_mean_cannot_produce_a_valid_match(self):
        panel = pd.DataFrame([
            (state, year, 1. if state == "FL" else 0.)
            for state in ["FL", "CA"] for year in range(2006, 2010)
        ], columns=["state", "year", "mme_per_capita"])
        gap = controls.trend_distance(panel, "mme_per_capita", "FL", 2010)
        self.assertTrue(gap.empty)


if __name__ == "__main__":
    unittest.main()
