"""Clean Census county populations for 2006-2015, excluding Alaska.

Inputs remain unchanged. County names/codes retain each source's geography;
cross-vintage differences are reported for review, not automatically reconciled.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def clean_population(path, years):
    """Read one Census vintage and return one row per county-year."""
    data = pd.read_csv(path, dtype="string", encoding="latin1")

    population_columns = [f"POPESTIMATE{year}" for year in years]

    # SUMLEV means summary level. 010 for the nation; 040 for states; 050 for conties
    required = ["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME"]

    missing = set(required + population_columns) - set(data.columns)

    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    # The older file with data from 2006 to 2009 stores codes without leading zeros.
    for column, width in [("SUMLEV", 3), ("STATE", 2), ("COUNTY", 3)]:
        data[column] = data[column].str.strip().str.zfill(width)

    # 050 denotes counties/county equivalents; state FIPS 02 is Alaska.
    data = data.loc[data["SUMLEV"].eq("050") & data["STATE"].ne("02")].copy()

    if data.empty:
        raise ValueError(f"{path.name}: no county records retained")

    data["fips"] = (
        data["STATE"] + data["COUNTY"]
    )  # e.g. state: 01 + county: 001 -> 01001

    for column in ["STNAME", "CTYNAME"]:
        data[column] = data[column].str.strip().replace("", pd.NA)
    data = data.rename(columns={"STNAME": "state", "CTYNAME": "county"})

    long = data.melt(
        id_vars=["fips", "state", "county"],
        value_vars=population_columns,
        var_name="year",
        value_name="population",
    )
    long["year"] = long["year"].str.removeprefix("POPESTIMATE").astype("int64")
    population = pd.to_numeric(long["population"], errors="raise")

    if population.isna().any() or (population <= 0).any():
        raise ValueError(f"{path.name}: population must be nonmissing and positive")
    if (population % 1 != 0).any():
        raise ValueError(f"{path.name}: noninteger population")
    long["population"] = population.astype("int64")
    return long


def validate(data):
    """Reject invalid identifiers, duplicate keys and unexpected coverage."""
    if data[["fips", "state", "county", "year"]].isna().any().any():
        raise ValueError("Missing identifiers")
    if not data["fips"].str.fullmatch(r"\d{5}").all():
        raise ValueError("FIPS must contain exactly five digits")
    if data.duplicated(["fips", "year"]).any():
        raise ValueError("Duplicate county-year keys")
    if set(data["year"]) != set(range(2006, 2016)):
        raise ValueError("Expected years 2006-2015")
    if data["fips"].str.startswith("02").any():
        raise ValueError("Alaska was not excluded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    early = clean_population(args.input_dir / "co-est00int-tot.csv", range(2006, 2010))
    late = clean_population(
        args.input_dir / "co-est2019-alldata.csv", range(2010, 2016)
    )
    population = pd.concat([early, late], ignore_index=True)
    population = population.sort_values(["fips", "year"]).reset_index(drop=True)
    validate(population)

    # Keep the year since a timeless name-to-code mapping would conceal changes.
    crosswalk = population[["fips", "state", "county", "year"]].copy()
    early_ids, late_ids = set(early["fips"]), set(late["fips"])
    early_only = early_ids - late_ids
    late_only = late_ids - early_ids
    geography_review = crosswalk.loc[
        crosswalk["fips"].isin(early_only | late_only)
    ].copy()

    # Also report changed names for codes present in both sources.
    names = crosswalk[["fips", "state", "county"]].drop_duplicates()
    changed_name_ids = (
        names.loc[names.duplicated("fips", keep=False), "fips"].unique().tolist()
    )

    report = {
        "rows": len(population),
        "counties_by_year": {
            str(year): int(count)
            for year, count in population.groupby("year")["fips"].nunique().items()
        },
        "fips_only_in_2006_2009": sorted(early_only),
        "fips_only_in_2010_2015": sorted(late_only),
        "fips_with_multiple_names": sorted(changed_name_ids),
        "state_format": "Full Census state names, not postal abbreviations",
        "population_definition": "July 1 resident population estimate",
        "geography_note": (
            "Source vintages may use different county geographies. Review code "
            "and name changes before merging shipments or mortality. This output "
            "does not harmonize county boundaries or fill absent county-years."
        ),
        "validation": "Passed identifiers, unique keys, years, and population checks",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    population.to_parquet(
        args.output_dir / "population_county_year.parquet", index=False
    )
    crosswalk.to_parquet(
        args.output_dir / "county_crosswalk_by_year.parquet", index=False
    )
    geography_review.to_csv(args.output_dir / "geography_review.csv", index=False)
    (args.output_dir / "population_quality_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
