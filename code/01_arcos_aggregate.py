"""Aggregate ARCOS opioid shipments to county-year totals.

Streams the Washington Post ARCOS bulk file straight out of the zip archive
(about 245 GB unzipped), so the archive never has to be extracted. Keeps all
states, so control states can be chosen later without a second pass.

Output: one row per buyer state, buyer county, year, drug and buyer type, with total
morphine milligram equivalents (MME) shipped, the number of transactions, and the
number of transactions with missing MME. A group whose MME is missing for every
transaction has a missing total rather than zero.

Sales with a missing buyer state or county, or a transaction date that is not a
valid date in the ARCOS coverage years, are left out of the main output and
summarized in a separate `<output>_excluded.csv` file.

Usage:
    python 01_arcos_aggregate.py ~/pds-data/arcos/arcos_all.zip arcos_county_year.parquet
    python 01_arcos_aggregate.py ~/pds-data/arcos/arcos_all.zip sample.parquet --nrows 500000
"""

import argparse
import io
import time
import zipfile

import pandas as pd

# Columns we need, as named in the ARCOS data dictionary. Matching is
# case-insensitive because the bulk file's header may differ in case.
NEEDED = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "TRANSACTION_DATE",
    "TRANSACTION_CODE",
    "CALC_BASE_WT_IN_GM",
    "MME_Conversion_Factor",
    "DRUG_NAME",
    "BUYER_BUS_ACT",
    "MME",  # precomputed in the bulk file; used to cross-check our calculation
]
# Kept by drug and buyer type so the team can later exclude drugs (for example
# buprenorphine, mainly used to treat opioid use disorder) or buyers (for example
# narcotic treatment programs dispensing methadone) without another full pass.
GROUP_KEYS = ["buyer_state", "buyer_county", "year", "drug_name", "buyer_bus_act"]
# Years covered by the Washington Post bulk release; dates outside are invalid.
VALID_YEARS = range(2006, 2020)
SUM_COLUMNS = ["opioid_mme", "n_transactions", "n_missing_mme"]


def open_member(zip_path):
    """Return the zip archive and the name of its single data file."""
    archive = zipfile.ZipFile(zip_path)
    members = [m for m in archive.infolist() if not m.is_dir()]
    data = max(members, key=lambda m: m.file_size)
    print(f"Reading {data.filename} ({data.file_size / 1e9:.1f} GB unzipped)")
    return archive, data.filename


def detect_layout(archive, member):
    """Find the delimiter and map our column names to the file's header."""
    with archive.open(member) as raw:
        header = io.TextIOWrapper(raw, encoding="utf-8", errors="replace").readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    columns = [c.strip().strip('"') for c in header.rstrip("\n").split(sep)]
    lookup = {c.lower(): c for c in columns}
    missing = [c for c in NEEDED if c.lower() not in lookup]
    if missing:
        raise KeyError(f"Columns not found in file header: {missing}\nHeader: {columns}")
    return sep, {lookup[c.lower()]: c for c in NEEDED}


def clean_text(values):
    """Strip and upper-case text, keeping missing and blank values as missing."""
    cleaned = values.str.strip().str.upper()
    return cleaned.mask(cleaned == "")


def parse_year(dates):
    """Return the year of each valid TRANSACTION_DATE, or missing if invalid.

    Accepts YYYY-MM-DD (the bulk file's format) and MMDDYYYY (the older API
    format). A date must be a real calendar date in VALID_YEARS.
    """
    s = dates.str.strip()
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    eight_digits = parsed.isna() & s.str.fullmatch(r"\d{8}").fillna(False)
    parsed[eight_digits] = pd.to_datetime(s[eight_digits], format="%m%d%Y", errors="coerce")
    year = parsed.dt.year.astype("Int64")
    return year.where(year.isin(VALID_YEARS))


def summarize(rows, keys):
    """Total MME by group, keeping a group missing when all of its MME is missing."""
    grouped = rows.groupby(keys, dropna=False)
    return pd.DataFrame(
        {
            "opioid_mme": grouped["opioid_mme"].sum(min_count=1),
            "n_transactions": grouped.size(),
            "n_missing_mme": grouped["mme_missing"].sum(),
        }
    ).reset_index()


def aggregate_chunk(chunk, rename, stats):
    chunk = chunk.rename(columns=rename)
    stats["rows"] += len(chunk)
    stats["drugs"].update(chunk["DRUG_NAME"].dropna().unique())

    # The Post's data is limited to sales ("S"); keep the filter in case the
    # bulk file includes other transaction types.
    sales = chunk[chunk["TRANSACTION_CODE"].str.strip() == "S"].copy()
    stats["non_sale_rows"] += len(chunk) - len(sales)

    weight = pd.to_numeric(sales["CALC_BASE_WT_IN_GM"], errors="coerce")
    factor = pd.to_numeric(sales["MME_Conversion_Factor"], errors="coerce")
    sales["opioid_mme"] = weight * 1000 * factor  # grams -> mg, then to MME
    sales["mme_missing"] = sales["opioid_mme"].isna()
    stats["missing_mme_rows"] += int(sales["mme_missing"].sum())
    file_mme = pd.to_numeric(sales["MME"], errors="coerce")
    stats["mme_mismatch_rows"] += int(((sales["opioid_mme"] - file_mme).abs() > 0.01 * file_mme.abs() + 1e-6).sum())

    sales["buyer_state"] = clean_text(sales["BUYER_STATE"])
    sales["buyer_county"] = clean_text(sales["BUYER_COUNTY"])
    sales["drug_name"] = clean_text(sales["DRUG_NAME"])
    sales["buyer_bus_act"] = clean_text(sales["BUYER_BUS_ACT"])
    sales["year"] = parse_year(sales["TRANSACTION_DATE"])

    # Quarantine sales we cannot place in a county-year: they would otherwise
    # become groups keyed on missing values and break the FIPS mapping.
    reason = pd.Series(pd.NA, index=sales.index, dtype="string")
    reason = reason.mask(sales["year"].isna(), "invalid_date")
    reason = reason.mask(sales["buyer_county"].isna(), "missing_county")
    reason = reason.mask(sales["buyer_state"].isna(), "missing_state")
    sales["exclusion_reason"] = reason
    excluded = sales[reason.notna()]
    kept = sales[reason.isna()]

    return summarize(kept, GROUP_KEYS), summarize(excluded, ["exclusion_reason", "buyer_state"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("zip_path")
    parser.add_argument("output")
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--nrows", type=int, default=None, help="read only the first N rows, for testing")
    args = parser.parse_args()

    archive, member = open_member(args.zip_path)
    sep, rename = detect_layout(archive, member)
    stats = {
        "rows": 0,
        "non_sale_rows": 0,
        "missing_mme_rows": 0,
        "mme_mismatch_rows": 0,
        "drugs": set(),
    }

    partials = []
    excluded_partials = []
    start = time.time()
    with archive.open(member) as raw:
        reader = pd.read_csv(
            raw,
            sep=sep,
            usecols=list(rename),
            dtype=str,
            chunksize=args.chunksize,
            nrows=args.nrows,
            encoding_errors="replace",
        )
        for i, chunk in enumerate(reader, start=1):
            kept, excluded = aggregate_chunk(chunk, rename, stats)
            partials.append(kept)
            excluded_partials.append(excluded)
            print(f"chunk {i}: {stats['rows']:,} rows read, {time.time() - start:,.0f}s elapsed", flush=True)

    result = (
        pd.concat(partials)
        .groupby(GROUP_KEYS, dropna=False)[SUM_COLUMNS]
        .sum(min_count=1)
        .reset_index()
        .sort_values(GROUP_KEYS)
    )
    result.to_parquet(args.output, index=False)

    excluded = (
        pd.concat(excluded_partials)
        .groupby(["exclusion_reason", "buyer_state"], dropna=False)[SUM_COLUMNS]
        .sum(min_count=1)
        .reset_index()
    )
    excluded_path = str(args.output).removesuffix(".parquet") + "_excluded.csv"
    excluded.to_csv(excluded_path, index=False)

    print(f"\nWrote {len(result):,} county-year rows to {args.output}")
    print(f"Rows read: {stats['rows']:,}")
    print(f"Non-sale rows dropped: {stats['non_sale_rows']:,}")
    print(f"Rows with missing MME: {stats['missing_mme_rows']:,}")
    print(f"Rows where our MME differs from the file's MME column by >1%: {stats['mme_mismatch_rows']:,}")
    print(f"Groups with some missing MME: {int((result['n_missing_mme'] > 0).sum()):,}")
    print(f"Groups with all MME missing (total left missing): {int(result['opioid_mme'].isna().sum()):,}")
    print(f"Sales excluded, by reason (details in {excluded_path}):")
    print(excluded.groupby("exclusion_reason")["n_transactions"].sum().to_string())
    print(f"Drugs present: {sorted(stats['drugs'])}")
    print(f"Years: {sorted(result['year'].unique().tolist())}")


if __name__ == "__main__":
    main()
