"""Regression cases for missing MME and independent outcome samples."""

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from analysis_data import county_panel, coverage_summary, shipments

spec = importlib.util.spec_from_file_location("panel_builder", ROOT / "code/05_build_analysis_panel.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def population_fixture():
    return pd.DataFrame([
        (fips, county, year, baseline if year == 2006 else 300_000)
        for fips, county, baseline in [
            ("12001", "Alpha County", 150_000),
            ("12003", "Beta County", 160_000),
            ("12005", "Gamma County", 170_000),
            ("12007", "Delta County", 180_000),
            ("12009", "Small County", 149_999),
            ("12111", "St. Lucie County", 200_000),
            ("53001", "Washington County", 200_000),
            ("06001", "Control County", 200_000),
        ]
        for year in (2006, 2007)
    ], columns=["fips", "county", "year", "population"])


def arcos_fixture():
    return pd.DataFrame([
        ("FL", "ALPHA", 2006, "OXYCODONE", "RETAIL PHARMACY", 100.0, 2, 0),
        ("FL", "BETA", 2006, "OXYCODONE", "RETAIL PHARMACY", None, 3, 3),
        ("FL", "GAMMA", 2006, "OXYCODONE", "RETAIL PHARMACY", 40.0, 2, 1),
        ("FL", "GAMMA", 2006, "HYDROCODONE", "RETAIL PHARMACY", None, 1, 1),
        ("FL", "SMALL", 2006, "OXYCODONE", "RETAIL PHARMACY", 999.0, 1, 0),
        ("FL", "SAINT LUCIE", 2006, "OXYCODONE", "RETAIL PHARMACY", None, 1, 1),
        ("FL", "ST. LUCIE", 2006, "OXYCODONE", "RETAIL PHARMACY", None, 2, 2),
    ], columns=["buyer_state", "buyer_county", "year", "drug_name", "buyer_bus_act",
                "opioid_mme", "n_transactions", "n_missing_mme"])


def mortality_fixture():
    return pd.DataFrame([
        ("12003", 2006, None),  # explicit record, D1 unavailable
        ("12005", 2006, 0),     # explicit numeric zero is valid
        ("12007", 2006, 12),    # deaths without shipment observations
    ], columns=["county_fips", "year", "d1_deaths"])


class MissingnessTests(unittest.TestCase):
    def test_mme_totals_and_counters_survive_both_aggregation_steps(self):
        data = shipments(arcos_fixture(), population_fixture()).set_index("fips")
        self.assertTrue(pd.isna(data.loc["12003", "opioid_mme"]))
        self.assertTrue(pd.isna(data.loc["12111", "opioid_mme"]))
        self.assertEqual(data.loc["12111", "n_missing_mme"], 3)
        self.assertEqual(data.loc["12005", "opioid_mme"], 40)
        self.assertEqual(data.loc["12005", "n_transactions"], 3)
        self.assertEqual(data.loc["12005", "n_missing_mme"], 2)

    def test_independent_outcomes_and_coverage_before_exclusions(self):
        panel = county_panel(arcos_fixture(), population_fixture(), mortality_fixture())
        year = panel.loc[panel.year.eq(2006)].set_index("fips")
        self.assertNotIn("12009", set(panel.fips))
        self.assertEqual(year.loc["12001", "opioid_mme"], 100)
        self.assertFalse(year.loc["12001", "has_mortality_record"])
        self.assertTrue(pd.isna(year.loc["12001", "overdose_deaths"]))
        self.assertTrue(year.loc["12003", "has_mortality_record"])
        self.assertTrue(pd.isna(year.loc["12003", "overdose_deaths"]))
        self.assertEqual(year.loc["12005", "overdose_deaths"], 0)
        self.assertFalse(year.loc["12005", "mme_complete"])
        self.assertFalse(year.loc["12007", "has_shipments"])
        self.assertTrue(pd.isna(year.loc["12007", "opioid_mme"]))
        self.assertEqual(year.loc["12007", "overdose_deaths"], 12)
        summary = coverage_summary(panel, ["state", "year"]).set_index(["state", "year"])
        fl = summary.loc[("FL", 2006)]
        self.assertEqual(fl.n_county_years, 5)
        self.assertEqual(fl.n_deaths_reported, 2)
        self.assertEqual(fl.deaths_reporting_share, 0.4)
        self.assertEqual(fl.n_mme_reported, 2)
        self.assertEqual(fl.n_mme_complete, 1)
        self.assertEqual(fl.n_mme_partial, 1)
        self.assertEqual(fl.n_missing_mme, 8)
        # A state-year without any shipments has unknown transaction counts, not 0.
        self.assertTrue(pd.isna(summary.loc[("WA", 2006), "n_transactions"]))

    def test_custom_controls_use_the_same_scope_and_independent_samples(self):
        panel = builder.build_panel(arcos_fixture(), population_fixture(), mortality_fixture(),
                                    controls={"FL": ["CA"], "WA": ["CA"]})
        self.assertEqual(set(panel.loc[panel.comparison.eq("FL"), "state"]), {"FL", "CA"})
        self.assertEqual(set(panel.loc[panel.comparison.eq("WA"), "state"]), {"WA", "CA"})
        self.assertTrue(panel.population_2006.ge(150_000).all())
        self.assertEqual(len(panel.loc[panel.comparison.eq("FL") & panel.year.eq(2006)]), 6)

    def test_missing_d1_column_fails_instead_of_using_old_total(self):
        with self.assertRaisesRegex(ValueError, "regenerate mortality"):
            county_panel(arcos_fixture(), population_fixture(),
                         mortality_fixture().rename(columns={"d1_deaths": "drug_overdose_deaths"}))


if __name__ == "__main__":
    unittest.main()
