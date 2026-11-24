"""Pre-post and difference-in-difference plots and regressions for FL and WA.

Reads the analysis panel from 05_build_analysis_panel.py. For each treated state
(Florida 2010, Washington 2012) and each outcome (opioid shipments per capita,
overdose deaths per 100,000):

- Pre-post plot: the treated state's counties, with separate linear fits before
  and after the policy and 95% confidence bands.
- Difference-in-difference plot: the same fits for the treated state and its
  control states.
- Regressions from Appendix A of the project instructions, with county fixed
  effects and standard errors clustered by county:
    (1) levels only:  y ~ county FE + post + post:treated
    (2) levels and trends:  y ~ county FE + t + t:treated + post + post:treated
                                + post:t + post:t:treated
  where t is years from the policy (0 in the policy year).

The unit is the county-year, and each county counts equally. Overdose death
models use only county-years with a reported count.

Usage:
    python code/06_analysis.py data/intermediate/analysis_panel.parquet --output-dir outputs
    python code/06_analysis.py panel.parquet --output-dir outputs --min-population-2006 100000
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

STATE_NAMES = {"FL": "Florida", "WA": "Washington"}
OUTCOMES = {
    "mme_per_capita": "Opioid shipments (MME per resident)",
    "deaths_per_100k": "Overdose deaths per 100,000 residents",
}
TREATED_COLOR, CONTROL_COLOR = "#c0392b", "#2c3e50"


def fit_line(rows, outcome):
    """Linear fit of outcome on years from policy, with a 95% band clustered by county."""
    model = smf.ols(f"{outcome} ~ years_from_policy", data=rows).fit(
        cov_type="cluster", cov_kwds={"groups": rows["fips"].astype("category").cat.codes}
    )
    grid = pd.DataFrame({"years_from_policy": sorted(rows["years_from_policy"].unique())})
    band = model.get_prediction(grid).summary_frame(alpha=0.05)
    return grid["years_from_policy"], band["mean"], band["mean_ci_lower"], band["mean_ci_upper"]


def draw_group(ax, rows, outcome, color, label):
    means = rows.groupby("years_from_policy")[outcome].mean()
    ax.scatter(means.index, means.values, color=color, s=18, zorder=3)
    for period, segment in rows.groupby("post"):
        if segment["years_from_policy"].nunique() < 2:
            continue
        x, fit, lower, upper = fit_line(segment, outcome)
        ax.plot(x, fit, color=color, linewidth=2, label=label if not period else None)
        ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)


def plot(data, treated, outcome, controls, path, include_controls):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    t = data[data["treated"]]
    draw_group(ax, t, outcome, TREATED_COLOR, f"{STATE_NAMES[treated]} counties")
    if include_controls:
        draw_group(ax, data[~data["treated"]], outcome, CONTROL_COLOR, f"Control counties ({', '.join(controls)})")
    ax.axvline(-0.5, color="black", linestyle="--", linewidth=1)
    ax.text(-0.45, ax.get_ylim()[1], " policy", va="top", fontsize=9)
    ax.set_xlabel("Years from policy change")
    ax.set_ylabel(OUTCOMES[outcome])
    kind = "Difference-in-difference" if include_controls else "Pre-post"
    ax.set_title(f"{kind}: {STATE_NAMES[treated]}, {OUTCOMES[outcome].split(' (')[0].lower()}", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def regressions(data, treated, outcome):
    """Fit Appendix A specifications (1) and (2) with county fixed effects."""
    rows = data.assign(
        t=data["years_from_policy"],
        post=data["post"].astype(int),
        treated=data["treated"].astype(int),
    )
    cluster = {"groups": rows["fips"].astype("category").cat.codes}
    levels = smf.ols(f"{outcome} ~ C(fips) + post + post:treated", data=rows).fit(cov_type="cluster", cov_kwds=cluster)
    trends = smf.ols(
        f"{outcome} ~ C(fips) + t + t:treated + post + post:treated + post:t + post:t:treated", data=rows
    ).fit(cov_type="cluster", cov_kwds=cluster)

    def row(label, result, expression):
        test = result.t_test(expression)
        low, high = test.conf_int(alpha=0.05)[0]
        return {
            "comparison": treated,
            "outcome": outcome,
            "estimate": label,
            "coef": float(test.effect[0]),
            "se": float(test.sd[0][0]),
            "ci_low": low,
            "ci_high": high,
            "p_value": float(test.pvalue),
            "n_county_years": int(result.nobs),
            "n_counties": rows["fips"].nunique(),
        }

    return [
        row("Spec 1: DiD in levels (post x treated)", levels, "post:treated = 0"),
        row("Spec 2: DiD in intercept at policy year (beta5)", trends, "post:treated = 0"),
        row("Spec 2: DiD in yearly slope (beta7)", trends, "post:t:treated = 0"),
        row("Spec 2: pre-period slope gap, treated minus control (beta3)", trends, "t:treated = 0"),
        # Treated-minus-control gap at t=2 is b1 + 2*b3 + b5 + 2*b7 and at t=-1 is
        # b1 - b3, so the change is 3*b3 + b5 + 2*b7. (Appendix A writes b3 for the
        # last term, but the pre-period slope gap applies over all three years.)
        row(
            "Spec 2: DiD, 2 years after vs 1 year before",
            trends,
            "2*post:t:treated + post:treated + 3*t:treated = 0",
        ),
        # The same comparison net of the pre-existing slope gap: how far the treated
        # state ends up from where its pre-policy trend relative to controls pointed.
        row("Spec 2: deviation from pre-trend at 2 years after (beta5 + 2*beta7)", trends,
            "post:treated + 2*post:t:treated = 0"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("panel")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--years-before", type=int, default=3)
    parser.add_argument("--years-after", type=int, default=3, help="years after the policy year to include")
    parser.add_argument(
        "--min-population-2006", type=int, default=0, help="scope condition: keep counties at or above this size"
    )
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    panel = panel[
        panel["years_from_policy"].between(-args.years_before, args.years_after)
        & (panel["population_2006"] >= args.min_population_2006)
    ]
    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    results = []
    for treated, data in panel.groupby("comparison"):
        controls = sorted(data.loc[~data["treated"], "state"].unique())
        for outcome in OUTCOMES:
            rows = data.dropna(subset=[outcome])
            short = "mme" if outcome == "mme_per_capita" else "deaths"
            plot(rows, treated, outcome, controls, out / "figures" / f"prepost_{treated.lower()}_{short}.png", False)
            plot(rows, treated, outcome, controls, out / "figures" / f"did_{treated.lower()}_{short}.png", True)
            results.extend(regressions(rows, treated, outcome))

    table = pd.DataFrame(results)
    table.to_csv(out / "regression_results.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
