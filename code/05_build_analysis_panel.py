"""Build the county-year analysis panel for the pre-post and difference-in-difference models.

Joins three cleaned datasets on county FIPS and year with inner joins:

- ARCOS opioid shipments (end buyers only, no buprenorphine), matched from ARCOS
  county names to Census FIPS codes
- Census county population
- CDC drug overdose deaths

Then stacks one comparison per treated state: Florida with its controls, and
Washington with its controls. A control state can appear in both comparisons, so
the `comparison` column identifies which one a row belongs to.

Usage:
    python code/05_build_analysis_panel.py \
        --arcos data/intermediate/arcos_county_year_drug_buyer.parquet \
        --population data/intermediate/population_county_year.parquet \
        --mortality data/intermediate/mortality_county_year.parquet \
        --output data/intermediate/analysis_panel.parquet
"""

import argparse
import re
import unicodedata

import pandas as pd

YEARS = range(2006, 2016)  # years covered by both population and mortality data
POLICY_YEAR = {"FL": 2010, "WA": 2012}
# From code/04_control_states.py (pre-trend ranking, excluding states with
# concurrent pain clinic laws or binding prescribing rules).
CONTROLS = {
    "FL": ["PA", "CA", "MI", "SC", "CO"],
    "WA": ["UT", "CT", "OR", "MA", "CA"],
}
# Same buyer and drug definition as the control-state ranking.
END_BUYERS = {
    "RETAIL PHARMACY",
    "CHAIN PHARMACY",
    "PRACTITIONER",
    "PRACTITIONER-DW/30",
    "PRACTITIONER-DW/100",
    "PRACTITIONER-DW/275",
    "MLP-NURSE PRACTITIONER",
    "MLP-PHYSICIAN ASSISTANT",
}
EXCLUDED_DRUGS = {"BUPRENORPHINE"}
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}
STATE_ABBR = {v: k for k, v in STATE_FIPS.items()}


def name_key(name):
    """Reduce a county name to a matching key: 'St. Lucie County' and 'SAINT LUCIE' -> 'STLUCIE'."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper().strip()
    name = re.sub(r"\bSAINTE\b", "STE", name)
    name = re.sub(r"\bSAINT\b", "ST", name)
    name = re.sub(r"\s+(COUNTY|PARISH)$", "", name)
    return re.sub(r"[^A-Z]", "", name)


def match_fips(arcos_names, population):
    """Map each ARCOS (state, county name) to a Census county FIPS code.

    Virginia's independent cities are named 'X city' by the Census but sometimes
    just 'X' in ARCOS, so an unmatched name is retried with 'CITY' appended.
    """
    census = population[["fips", "county"]].drop_duplicates()
    census = census.assign(state_fips=census["fips"].str[:2], key=census["county"].map(name_key))
    census = census.drop_duplicates(["state_fips", "key", "fips"])
    if census.duplicated(["state_fips", "key"]).any():
        raise ValueError("Two Census counties in one state share a matching key")
    lookup = census.set_index(["state_fips", "key"])["fips"]

    def find(state, county):
        state_fips = STATE_FIPS.get(state)
        key = name_key(county)
        for candidate in (key, key + "CITY"):
            if (state_fips, candidate) in lookup.index:
                return lookup[(state_fips, candidate)]
        return pd.NA

    return arcos_names.assign(fips=[find(s, c) for s, c in zip(arcos_names["buyer_state"], arcos_names["buyer_county"])])


def shipments(arcos, population):
    keep = (
        arcos["buyer_bus_act"].isin(END_BUYERS)
        & ~arcos["drug_name"].isin(EXCLUDED_DRUGS)
        & arcos["year"].isin(YEARS)
    )
    totals = arcos[keep].groupby(["buyer_state", "buyer_county", "year"])["opioid_mme"].sum().reset_index()
    names = match_fips(totals[["buyer_state", "buyer_county"]].drop_duplicates(), population)
    totals = totals.merge(names, on=["buyer_state", "buyer_county"], how="inner", validate="many_to_one")
    # Territories and Alaska are not in the population data, so they never match.
    in_states = totals["buyer_state"].isin(set(STATE_FIPS) - {"AK"})
    unmatched = totals[totals["fips"].isna() & in_states]
    print(f"ARCOS county names in the 49 states and DC not matched to a FIPS code: "
          f"{unmatched[['buyer_state', 'buyer_county']].drop_duplicates().to_records(index=False).tolist()} "
          f"({unmatched['opioid_mme'].sum() / totals.loc[in_states, 'opioid_mme'].sum():.4%} of MME)")
    totals = totals.dropna(subset=["fips"])
    # Two ARCOS spellings can map to one county; add them together.
    return totals.groupby(["fips", "year"], as_index=False)["opioid_mme"].sum()


def build_panel(arcos, population, mortality, deaths_column):
    population = population[population["year"].isin(YEARS)]
    mme = shipments(arcos, population)
    deaths = mortality.rename(columns={"county_fips": "fips", deaths_column: "overdose_deaths"})
    deaths = deaths.loc[deaths["year"].isin(YEARS), ["fips", "year", "overdose_deaths"]]

    panel = population[["fips", "county", "year", "population"]]
    sizes = {"population": len(panel)}
    panel = panel.merge(mme, on=["fips", "year"], how="inner", validate="one_to_one")
    sizes["with shipments"] = len(panel)
    panel = panel.merge(deaths, on=["fips", "year"], how="inner", validate="one_to_one")
    sizes["with mortality records"] = len(panel)
    print("County-years after each inner join: " + ", ".join(f"{k} {v:,}" for k, v in sizes.items()))

    panel["state"] = panel["fips"].str[:2].map(STATE_ABBR)
    panel["mme_per_capita"] = panel["opioid_mme"] / panel["population"]
    panel["deaths_per_100k"] = panel["overdose_deaths"] / panel["population"] * 1e5
    # Fixed 2006 population, for a scope condition that does not change over time.
    pop_2006 = population.loc[population["year"] == 2006, ["fips", "population"]]
    panel = panel.merge(pop_2006.rename(columns={"population": "population_2006"}), on="fips", how="inner")

    comparisons = []
    for treated, policy_year in POLICY_YEAR.items():
        rows = panel[panel["state"].isin([treated] + CONTROLS[treated])].copy()
        rows["comparison"] = treated
        rows["treated"] = rows["state"] == treated
        rows["policy_year"] = policy_year
        rows["post"] = rows["year"] >= policy_year
        rows["years_from_policy"] = rows["year"] - policy_year
        comparisons.append(rows)
    columns = [
        "comparison", "state", "fips", "county", "year", "policy_year", "years_from_policy",
        "treated", "post", "population", "population_2006", "opioid_mme", "mme_per_capita",
        "overdose_deaths", "deaths_per_100k",
    ]
    return pd.concat(comparisons, ignore_index=True)[columns].sort_values(["comparison", "state", "fips", "year"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arcos", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--mortality", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--deaths-column",
        default="drug_overdose_deaths",
        help="mortality column used as the outcome (switch to the D1 column once it exists)",
    )
    args = parser.parse_args()

    panel = build_panel(
        pd.read_parquet(args.arcos),
        pd.read_parquet(args.population),
        pd.read_parquet(args.mortality),
        args.deaths_column,
    )
    panel.to_parquet(args.output, index=False)

    print(f"\nWrote {len(panel):,} rows to {args.output}")
    summary = panel.groupby(["comparison", "state"]).agg(
        counties=("fips", "nunique"),
        county_years=("fips", "size"),
        deaths_reported=("overdose_deaths", lambda d: d.notna().mean()),
    )
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
