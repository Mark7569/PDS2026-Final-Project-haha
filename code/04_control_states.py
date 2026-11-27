"""Rank candidate control states by how closely their pre-policy trends match.

For each treated state (Florida 2010, Washington 2012), compares every candidate
state's pre-policy trend in two outcomes:

- opioid shipments per capita (MME per resident, end buyers only, no buprenorphine)
- D1 unintentional drug-poisoning deaths per 100,000 in reporting counties

Both outcomes use counties with 2006 population >= 150,000. Each rate uses
the population of counties reporting that outcome, independently of the other.

Trends are compared on a common scale: each state's yearly values divided by its
own pre-period mean, so states with different levels but parallel trends score as
similar. The score is the root mean squared difference from the treated state.

Usage:
    python code/04_control_states.py \
        --arcos data/intermediate/arcos_county_year_drug_buyer.parquet \
        --population data/intermediate/population_county_year.parquet \
        --mortality data/intermediate/mortality_county_year.parquet \
        --output-dir outputs
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_data import MIN_POPULATION_2006, county_panel, coverage_summary

TREATED = {"FL": 2010, "WA": 2012}
FIRST_YEAR = 2006
# Alaska is dropped per the project instructions (county boundaries change in 2010).
# DC is not a state.
NOT_CANDIDATES = {"AK", "DC"}
# State opioid prescribing rules with the force of law and pain clinic laws, with
# the year each took effect. Sources:
#   [CDC] CDC Public Health Law Program, "Menu of Pain Management Clinic Regulation"
#         (laws through Sept 28, 2012; footnote adds AL, GA and IN by July 2013).
#   [NPHL] Network for Public Health Law (Corey Davis), "Appendix B: State-by-State
#         Summary of Opioid Prescribing Regulations and Guidelines" (hosted by the
#         Arizona Department of Health Services).
# Only years 2006 to 2015 matter here. Advisory guidelines are not listed.
STATE_POLICIES = {
    "AL": [(2013, "pain clinic law [CDC]; prescribing rule Ala. Admin. Code 540-X-4 [NPHL]")],
    "GA": [(2013, "pain clinic law [CDC]; Ga. Comp. R. & Regs. 360-3-.06 [NPHL]")],
    "IN": [(2013, "pain clinic law [CDC]"), (2014, "844 Ind. Admin. Code 5-6-3 [NPHL]")],
    "KY": [(2012, "pain clinic law, KRS 218A.175 [CDC]"), (2013, "201 KAR 9:260 [NPHL]")],
    "LA": [(2005, "pain clinic law, La. R.S. 40:2198.11 [CDC]"), (2008, "pain clinic rules [CDC]")],
    "ME": [(2012, "Board of Licensure in Medicine rule, 02-373 Me. Code R. 21 [NPHL]")],
    "MS": [(2012, "pain clinic rules, 30-17 Miss. Code R. 2640 [CDC, NPHL]")],
    "NJ": [(2011, "N.J. Admin. Code 13:35-7.6 [NPHL]")],
    "NM": [(2014, "N.M. Code R. 16.10.14.10 [NPHL]")],
    "NY": [(2013, "I-STOP, N.Y. Pub. Health 3334-a(2) [NPHL]")],
    "OH": [(2011, "pain clinic law, Ohio Rev. Code 4731.054 [CDC]")],
    "OK": [(2010, "Okla. Stat. tit. 63, 2-302(M) [NPHL]")],
    "RI": [(2014, "R.I. Admin. Code 31-2-6:3.0 [NPHL]")],
    "TN": [(2011, "pain clinic law, Tenn. Code 63-1-301 [CDC]"), (2013, "Tenn. Code 53-11-308 [NPHL]")],
    "TX": [(2007, "22 Tex. Admin. Code 170.3 [NPHL]"), (2009, "pain clinic law, Tex. Occ. Code 168 [CDC]"),
           (2010, "pain clinic rules, 22 Tex. Admin. Code 195 [CDC]")],
    "VT": [(2014, "Vt. Stat. tit. 18, 4289(d) [NPHL]")],
    "WV": [(2012, "pain clinic law, W. Va. Code 16-5H [CDC]"), (2014, "W. Va. Code 60A-9-5a [NPHL]")],
}
# A candidate is excluded if any of its policies took effect in this window around
# the treated state's policy year: three pre-policy years, three post-policy years
# (the policy year counts as post), plus one more year in case of anticipation.
WINDOW_BEFORE, WINDOW_AFTER = 3, 3


def concurrent_policies(state, policy_year):
    lo, hi = policy_year - WINDOW_BEFORE, policy_year + WINDOW_AFTER
    return "; ".join(f"{y}: {d}" for y, d in STATE_POLICIES.get(state, []) if lo <= y <= hi)


STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
OUTCOMES = {
    "mme_per_capita": "Opioid shipments (MME per resident)",
    "deaths_per_100k": "D1 deaths per 100,000 (reporting counties)",
}


def state_series(counties):
    """Aggregate independently reporting counties; retain all-missing state-years.

    The population denominator for each outcome matches its numerator's counties.
    MME counters describe the entire scoped sample, including all-missing totals.
    """
    panel = coverage_summary(counties, ["state", "year"])
    for count, outcome, scale in [
        ("opioid_mme", "mme_per_capita", 1),
        ("overdose_deaths", "deaths_per_100k", 1e5),
    ]:
        observed = counties.loc[counties[count].notna()]
        totals = observed.groupby(["state", "year"])[[count, "population"]].sum(min_count=1)
        rate = (totals[count] / totals["population"] * scale).rename(outcome).reset_index()
        panel = panel.merge(rate, on=["state", "year"], how="left", validate="one_to_one")
    return panel


def trend_distance(panel, outcome, treated, policy_year):
    """Require every expected pre-policy year and a nonzero normalizing mean."""
    pre = panel.loc[panel["year"].between(FIRST_YEAR, policy_year - 1)]
    wide = pre.pivot(index="year", columns="state", values=outcome)
    wide = wide.reindex(range(FIRST_YEAR, policy_year)).dropna(axis=1)
    wide = wide.loc[:, wide.mean().ne(0)]
    if treated not in wide:
        raise ValueError(f"{treated} lacks a complete, nonzero pre-policy series for {outcome}")
    normalized = wide / wide.mean()
    gap = normalized.sub(normalized[treated], axis=0)
    return (gap.pow(2).mean() ** 0.5).drop(treated).rename(f"{outcome}_gap")


def rank_candidates(panel, treated, policy_year):
    candidates = set(STATE_NAMES) - NOT_CANDIDATES - set(TREATED)
    scores = pd.concat([trend_distance(panel, o, treated, policy_year) for o in OUTCOMES], axis=1)
    # Include states without an eligible county or a complete pre-period series
    # in the audit ranking, with unavailable scores rather than silently omitting them.
    scores = scores.reindex(sorted(candidates))
    scores["mean_gap"] = scores.mean(axis=1, skipna=False)
    scores["concurrent_policy"] = [concurrent_policies(s, policy_year) for s in scores.index]
    return scores.rename_axis("state").reset_index().sort_values(
        ["mean_gap", "state"], na_position="last",
    ).set_index("state")


def plot_trends(panel, treated, policy_year, controls, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (outcome, label) in zip(axes, OUTCOMES.items()):
        data = panel.dropna(subset=[outcome])
        data = data[data["year"] <= 2015]
        control_mean = data[data["state"].isin(controls)].groupby("year")[outcome].mean()
        for state in controls:
            s = data[data["state"] == state]
            ax.plot(s["year"], s[outcome], color="0.75", linewidth=1)
        ax.plot(control_mean.index, control_mean.values, color="0.3", linewidth=2, label="Control average")
        t = data[data["state"] == treated]
        ax.plot(t["year"], t[outcome], color="#c0392b", linewidth=2.5, label=STATE_NAMES[treated])
        ax.axvline(policy_year - 0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Year")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{STATE_NAMES[treated]} vs. controls ({', '.join(controls)}); grey lines are individual controls")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arcos", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--mortality", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-controls", type=int, default=5)
    parser.add_argument("--min-population-2006", type=int, default=MIN_POPULATION_2006)
    parser.add_argument("--deaths-column", default="d1_deaths")
    args = parser.parse_args()
    if args.n_controls < 1:
        parser.error("--n-controls must be at least 1")
    counties = county_panel(
        pd.read_parquet(args.arcos), pd.read_parquet(args.population),
        pd.read_parquet(args.mortality), args.deaths_column, args.min_population_2006,
    )
    panel = state_series(counties)
    if args.deaths_column != "d1_deaths":
        OUTCOMES["deaths_per_100k"] = f"{args.deaths_column} per 100,000 (reporting counties)"

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    audit = panel.assign(deaths_column=args.deaths_column,
                         min_population_2006=args.min_population_2006)
    audit.to_csv(out / "control_state_year_coverage.csv", index=False)
    selections = {}
    for treated, policy_year in TREATED.items():
        ranking = rank_candidates(panel, treated, policy_year)
        eligible = ranking.loc[ranking["concurrent_policy"].eq("") & ranking["mean_gap"].notna()]
        if len(eligible) < args.n_controls:
            raise ValueError(f"Only {len(eligible)} eligible controls for {treated}; requested {args.n_controls}")
        controls = eligible.index[:args.n_controls].tolist()
        selections[treated] = controls
        ranking["selected"] = ranking.index.isin(controls)
        ranking["deaths_column"] = args.deaths_column
        ranking["min_population_2006"] = args.min_population_2006
        ranking.to_csv(out / f"control_ranking_{treated.lower()}.csv")
        plot_trends(panel, treated, policy_year, controls, out / "figures" / f"pretrends_{treated.lower()}.png")
        print(f"\n{STATE_NAMES[treated]} (policy {policy_year}), best matches first:")
        print(ranking.head(12).round(3).to_string())
        print(f"Suggested controls: {controls}")
    (out / "selected_controls.json").write_text(json.dumps(selections, indent=2) + "\n")
    (out / "control_selection_metadata.json").write_text(json.dumps({
        "deaths_column": args.deaths_column,
        "min_population_2006": args.min_population_2006,
        "first_pre_policy_year": FIRST_YEAR,
        "n_controls": args.n_controls,
        "rate_denominator": "population of scoped counties reporting the same outcome",
        "inputs": {
            name: {"file": Path(path).name,
                   "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for name, path in {"arcos": args.arcos, "population": args.population,
                               "mortality": args.mortality}.items()
        },
        "selections": selections,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
