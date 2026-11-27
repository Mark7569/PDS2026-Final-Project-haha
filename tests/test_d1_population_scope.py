import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mortality_clean = load_module("mortality_clean", Path("code/02_mortality_clean.py"))
panel_builder = load_module("panel_builder", Path("code/05_build_analysis_panel.py"))


class MortalityD1Tests(unittest.TestCase):
    def test_d1_counts_are_separate_and_unreported_values_remain_null(self):
        source = (
            "County Code\tYear\tDrug/Alcohol Induced Cause Code\tDeaths\n"
            "1001\t2010\tD1\t11\n"
            "1001\t2010\tD2\tMissing\n"
            "1003\t2010\tD2\t12\n"
            "1005\t2010\tD1\tMissing\n"
            "1007\t2010\tD1\t0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "mortality.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("Underlying Cause of Death, 2010.txt", source)
            result = mortality_clean.clean_archive(archive_path).set_index("county_fips")

        self.assertEqual(result.loc["01001", "d1_deaths"], 11)
        self.assertTrue(pd.isna(result.loc["01001", "drug_overdose_deaths"]))
        self.assertTrue(pd.isna(result.loc["01003", "d1_deaths"]))
        self.assertTrue(pd.isna(result.loc["01005", "d1_deaths"]))
        self.assertEqual(result.loc["01007", "d1_deaths"], 0)


class PopulationScopeTests(unittest.TestCase):
    def test_panel_uses_fixed_2006_population_cutoff_and_d1_outcome(self):
        population = pd.DataFrame(
            [
                (fips, county, year, pop)
                for fips, county, first_population in [
                    ("12001", "Alpha County", 160_000),
                    ("12003", "Beta County", 140_000),
                    ("53001", "Gamma County", 200_000),
                    ("06001", "Delta County", 180_000),
                ]
                for year, pop in [(2006, first_population), (2007, first_population + 1000)]
            ],
            columns=["fips", "county", "year", "population"],
        )
        names = {
            "12001": ("FL", "Alpha County"),
            "12003": ("FL", "Beta County"),
            "53001": ("WA", "Gamma County"),
            "06001": ("CA", "Delta County"),
        }
        arcos = pd.DataFrame(
            [
                {
                    "buyer_state": names[fips][0],
                    "buyer_county": names[fips][1],
                    "year": year,
                    "drug_name": "OXYCODONE",
                    "buyer_bus_act": "RETAIL PHARMACY",
                    "opioid_mme": 100.0,
                    "n_transactions": 1,
                    "n_missing_mme": 0,
                }
                for fips in names
                for year in (2006, 2007)
            ]
        )
        mortality = pd.DataFrame(
            [
                {
                    "county_fips": fips,
                    "year": year,
                    "d1_deaths": 12,
                    "drug_overdose_deaths": 99,
                }
                for fips in names
                for year in (2006, 2007)
            ]
        )

        panel = panel_builder.build_panel(arcos, population, mortality)

        self.assertNotIn("12003", set(panel["fips"]))
        self.assertTrue((panel["population_2006"] >= 150_000).all())
        self.assertTrue(panel["overdose_deaths"].eq(12).all())


if __name__ == "__main__":
    unittest.main()
