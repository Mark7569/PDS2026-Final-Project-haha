"""Shared county sample and MME aggregation for control selection and analysis."""

import re
import unicodedata

import pandas as pd

YEARS = range(2006, 2016)  # years covered by both population and mortality data
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
MIN_POPULATION_2006 = 150_000
MME_COLUMNS = ["opioid_mme", "n_transactions", "n_missing_mme"]


def summarize_mme(rows, keys):
    """Preserve observed subtotals, all-missing totals, and transaction counts."""
    return rows.groupby(keys, as_index=False)[MME_COLUMNS].sum(min_count=1)


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
    totals = summarize_mme(arcos.loc[keep], ["buyer_state", "buyer_county", "year"])
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
    return summarize_mme(totals, ["fips", "year"])


def scope_population(population, min_population_2006=MIN_POPULATION_2006, states=None):
    """Apply a fixed baseline cutoff before consulting outcome availability."""
    if min_population_2006 < 0:
        raise ValueError("Minimum 2006 population cannot be negative")
    population = population.loc[population["year"].isin(YEARS)].copy()
    population["state"] = population["fips"].str[:2].map(STATE_ABBR)
    allowed = set(STATE_FIPS) - {"AK", "DC"} if states is None else set(states)
    population = population.loc[population["state"].isin(allowed)].copy()
    baseline = population.loc[population["year"].eq(2006), ["fips", "population"]]
    population = population.merge(
        baseline.rename(columns={"population": "population_2006"}),
        on="fips", how="left", validate="many_to_one",
    )
    missing = sorted(population.loc[population["population_2006"].isna(), "fips"].unique())
    if missing:
        raise ValueError(f"Analysis counties lack a 2006 population baseline: {missing}")
    return population.loc[population["population_2006"].ge(min_population_2006)].copy()


def county_panel(arcos, population, mortality, deaths_column="d1_deaths",
                 min_population_2006=MIN_POPULATION_2006, states=None):
    """Keep every in-scope population row; missing outcomes stay unavailable."""
    if deaths_column not in mortality:
        raise ValueError(f"Mortality column {deaths_column!r} is absent; regenerate mortality with the D1 cleaner")
    panel = scope_population(population, min_population_2006, states)
    mme = shipments(arcos, population.loc[population["year"].isin(YEARS)])
    panel = panel.merge(mme, on=["fips", "year"], how="left",
                        validate="one_to_one", indicator="shipment_join")
    panel["has_shipments"] = panel.pop("shipment_join").eq("both")
    deaths = mortality.rename(columns={"county_fips": "fips"})
    deaths = deaths.loc[deaths["year"].isin(YEARS), ["fips", "year", deaths_column]]
    deaths = deaths.rename(columns={deaths_column: "overdose_deaths"})
    panel = panel.merge(deaths, on=["fips", "year"], how="left",
                        validate="one_to_one", indicator="mortality_join")
    panel["has_mortality_record"] = panel.pop("mortality_join").eq("both")
    panel["mme_complete"] = panel["has_shipments"] & panel["n_missing_mme"].eq(0).fillna(False)
    panel["mme_per_capita"] = panel["opioid_mme"] / panel["population"]
    panel["deaths_per_100k"] = panel["overdose_deaths"] / panel["population"] * 1e5
    return panel


def coverage_summary(panel, group_keys):
    """Report availability before dropping missing mortality or shipment outcomes."""
    rows = panel.assign(
        mme_reported=panel["opioid_mme"].notna(),
        mme_partial=panel["opioid_mme"].notna() & ~panel["mme_complete"],
        deaths_reported=panel["overdose_deaths"].notna(),
    )
    grouped = rows.groupby(group_keys, observed=True)
    summary = grouped.agg(
        n_county_years=("fips", "size"),
        n_counties=("fips", "nunique"),
        n_shipment_records=("has_shipments", "sum"),
        n_mme_reported=("mme_reported", "sum"),
        n_mme_complete=("mme_complete", "sum"),
        n_mme_partial=("mme_partial", "sum"),
        n_mortality_records=("has_mortality_record", "sum"),
        n_deaths_reported=("deaths_reported", "sum"),
    )
    for column in ["n_transactions", "n_missing_mme"]:
        summary[column] = grouped[column].sum(min_count=1)
    summary["mme_reporting_share"] = summary["n_mme_reported"] / summary["n_county_years"]
    summary["deaths_reporting_share"] = summary["n_deaths_reported"] / summary["n_county_years"]
    return summary.reset_index()
