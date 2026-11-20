"""Rank candidate control states by how closely their pre-policy trends match.

For each treated state (Florida 2010, Washington 2012), compares every candidate
state's pre-policy trend in two outcomes:

- opioid shipments per capita (MME per resident, end buyers only, no buprenorphine)
- drug overdose deaths per 100,000 residents, among counties with a reported count

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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TREATED = {"FL": 2010, "WA": 2012}
FIRST_YEAR = 2006
# Buyers who dispense to patients. Manufacturers and distributors only move stock
# between businesses; hospitals, mail-order and treatment programs are left out
# because their shipments do not reflect local prescribing.
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
EXCLUDED_DRUGS = {"BUPRENORPHINE"}  # mainly used to treat opioid use disorder
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
OUTCOMES = {
    "mme_per_capita": "Opioid shipments (MME per resident)",
    "deaths_per_100k": "Overdose deaths per 100,000 (reporting counties)",
}


def shipments_per_capita(arcos, population):
    """State-year MME per resident from end-buyer shipments."""
    keep = arcos["buyer_bus_act"].isin(END_BUYERS) & ~arcos["drug_name"].isin(EXCLUDED_DRUGS)
    mme = arcos[keep].groupby(["buyer_state", "year"])["opioid_mme"].sum().rename("mme")
    mme = mme.reset_index().rename(columns={"buyer_state": "state"})
    pop = population.assign(state=population["fips"].str[:2].map({v: k for k, v in STATE_FIPS.items()}))
    pop = pop.groupby(["state", "year"])["population"].sum().reset_index()
    out = mme.merge(pop, on=["state", "year"], how="inner")
    return out.assign(mme_per_capita=out["mme"] / out["population"])[["state", "year", "mme_per_capita"]]


def deaths_per_100k(mortality, population):
    """State-year overdose deaths per 100,000, using only counties with a reported count.

    Counts under 10 are suppressed, so this rate describes reporting (mostly larger)
    counties, which matches the project's scope condition.
    """
    reported = mortality.dropna(subset=["drug_overdose_deaths"]).rename(columns={"county_fips": "fips"})
    merged = reported.merge(population[["fips", "year", "population"]], on=["fips", "year"], how="inner")
    merged["state"] = merged["fips"].str[:2].map({v: k for k, v in STATE_FIPS.items()})
    totals = merged.groupby(["state", "year"])[["drug_overdose_deaths", "population"]].sum().reset_index()
    totals["deaths_per_100k"] = totals["drug_overdose_deaths"] / totals["population"] * 1e5
    return totals[["state", "year", "deaths_per_100k"]]


def trend_distance(panel, outcome, treated, policy_year):
    """RMS gap between each state's normalized pre-period trend and the treated state's."""
    pre = panel[(panel["year"] >= FIRST_YEAR) & (panel["year"] < policy_year)].dropna(subset=[outcome])
    wide = pre.pivot(index="year", columns="state", values=outcome)
    wide = wide.dropna(axis=1)  # require every pre-period year
    normalized = wide / wide.mean()
    gap = normalized.sub(normalized[treated], axis=0)
    return (gap.pow(2).mean() ** 0.5).drop(treated).rename(f"{outcome}_gap")


def rank_candidates(panel, treated, policy_year):
    candidates = set(STATE_NAMES) - NOT_CANDIDATES - set(TREATED)
    scores = pd.concat([trend_distance(panel, o, treated, policy_year) for o in OUTCOMES], axis=1)
    scores = scores[scores.index.isin(candidates)]
    scores["mean_gap"] = scores.mean(axis=1, skipna=False)
    scores["concurrent_policy"] = [concurrent_policies(s, policy_year) for s in scores.index]
    return scores.sort_values("mean_gap")


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
    args = parser.parse_args()

    population = pd.read_parquet(args.population)
    panel = shipments_per_capita(pd.read_parquet(args.arcos), population).merge(
        deaths_per_100k(pd.read_parquet(args.mortality), population), on=["state", "year"], how="outer"
    )

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for treated, policy_year in TREATED.items():
        ranking = rank_candidates(panel, treated, policy_year)
        ranking.to_csv(out / f"control_ranking_{treated.lower()}.csv")
        eligible = ranking[(ranking["concurrent_policy"] == "") & ranking["mean_gap"].notna()]
        controls = eligible.index[: args.n_controls].tolist()
        plot_trends(panel, treated, policy_year, controls, out / "figures" / f"pretrends_{treated.lower()}.png")
        print(f"\n{STATE_NAMES[treated]} (policy {policy_year}), best matches first:")
        print(ranking.head(12).round(3).to_string())
        print(f"Suggested controls: {controls}")


if __name__ == "__main__":
    main()
