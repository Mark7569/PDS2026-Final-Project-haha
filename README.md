# Opioid Prescribing Policies: Florida and Washington

Practical Data Science team project (Xuechun Zhao, Tsaone Tapologo, Mark Li).

We estimate how Florida's 2010 and Washington's 2012 opioid prescribing regulations affected opioid shipments per capita and drug overdose deaths per 100,000 residents, using pre-post and difference-in-difference comparisons at the county-year level.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Pipeline scripts, numbered in the order they run |
| `data/intermediate/` | Cleaned county-year datasets (parquet, tracked with git-lfs) |
| `outputs/figures/` | Figures for the report |
| `report/` | Stakeholder report |

## Source data

Source files are not stored in this repository. Download them to a local folder outside the repo and pass their paths to the scripts.

| Source | Link |
|---|---|
| ARCOS opioid shipments, 2006 to 2019 (Washington Post, about 17.7 GB zipped) | https://gfx-data.news-engineering.aws.wapo.pub/ne-static/arcos/v2/bulk/arcos_all.zip |
| ARCOS data dictionary | https://github.com/wpinvestigative/arcos-api/blob/master/data/data_dictionary.csv |
| US Vital Statistics mortality, 2003 to 2015 (course-provided) | https://www.dropbox.com/s/kad4dwebr88l3ud/US_VitalStatistics.zip?dl=0 |
| Census county population, 2000–2010 (use 2006–2009) | https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv |
| Census county population, 2010–2019 (use 2010–2015) | https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv |

## Setup

Install [git-lfs](https://git-lfs.com) before cloning or pulling, so parquet files download correctly:

```bash
git lfs install
```

Install the Python dependencies:

```bash
python -m pip install pandas pyarrow matplotlib statsmodels
```

## Population cleaning

Place these original Census files in a local folder outside the repository:

- `co-est00int-tot.csv`
- `co-est2019-alldata.csv`

Run the following command from the repository root, replacing `/path/to/population` with your source-data folder:

```bash
python code/03_population_clean.py --input-dir "/path/to/population" --output-dir data/intermediate
```

## Mortality cleaning

Clean the course Vital Statistics ZIP into county-year drug-overdose death counts:

```bash
python code/02_mortality_clean.py ~/pds-data/US_VitalStatistics.zip \
  data/intermediate/mortality_county_year.parquet
```

The output is keyed by five-digit county FIPS and year. It sums CDC cause codes
D1-D4 (all drug-poisoning intents; not opioid-specific), preserves explicit
`Missing` counts and county-years with no D1-D4 rows as unavailable. The source
omits both suppressed and zero counts, so the cleaner does not infer zeros.
It also retains D1 deaths separately in `d1_deaths` (unintentional poisonings);
an explicit `Missing` value or absent D1 row remains null, not zero.
`reported_overdose_categories` counts reported numeric D1-D4 categories, and
`deaths_complete` is true only when all four are reported. A death total with
fewer than four reported categories is partial and must not be treated as a
complete total. The cleaner emits county-years present in the source; a county-
year absent entirely from the source will not appear. Build the in-scope sample
from the population table, then left-join shipments and mortality independently.
Missing mortality records must not remove valid shipment observations or become
zero deaths. The primary outcome is D1, unintentional drug poisoning; it is not
an opioid-specific death count.

## Analysis panel and coverage

The primary sample uses counties with **2006 population >= 150,000**, held fixed
across 2006–2015. County FIPS without a 2006 baseline cannot satisfy the cutoff;
they are excluded and listed in the run log rather than assigned a later-year
population. `analysis_data.py` supplies the same geographic matching,
population scope, and MME aggregation to control selection and panel building.
Both aggregation stages use `sum(min_count=1)`: a wholly missing MME total stays
missing. Observed subtotals with some missing transactions remain numeric;
`n_transactions`, `n_missing_mme`, and `mme_complete` identify their completeness.
A county-year without any shipment record is unavailable, never assumed zero.

```bash
python code/05_build_analysis_panel.py \
  --arcos data/intermediate/arcos_county_year_drug_buyer.parquet \
  --population data/intermediate/population_county_year.parquet \
  --mortality data/intermediate/mortality_county_year.parquet \
  --output data/intermediate/analysis_panel.parquet --coverage-dir outputs
```

The default control lists use the regenerated D1 / 150,000-baseline rankings:
Florida: **CA, PA, CO, IA, MI**; Washington: **CT, OR, KS, MA, CA**. Pass
`--controls-file outputs/selected_controls.json` to use a fresh selection.
Alternative scopes use `--min-population-2006`; the mortality outcome uses
`--deaths-column`. Rerun control selection before building an alternative sample.

`coverage_by_group_year.csv` and `coverage_by_state_year.csv` use **every in-scope
population county-year** as their denominator, before any outcome is dropped.
They include shipment/mortality record counts, numeric MME/D1 counts, partial MME
counts, transaction counters, and reporting shares. `in_analysis_window` marks
the default -3 through +3 policy-year window; window arguments can change it.
The analysis script drops only the missing outcome used by each model, so MME
and mortality regressions can have different samples. Reporting counties can
change over time; inspect the coverage tables before interpreting trends.

```bash
python -m unittest discover -s tests -v
```

## Control-state selection

```bash
python code/04_control_states.py \
  --arcos data/intermediate/arcos_county_year_drug_buyer.parquet \
  --population data/intermediate/population_county_year.parquet \
  --mortality data/intermediate/mortality_county_year.parquet \
  --output-dir outputs --deaths-column d1_deaths --min-population-2006 150000

python code/05_build_analysis_panel.py \
  --arcos data/intermediate/arcos_county_year_drug_buyer.parquet \
  --population data/intermediate/population_county_year.parquet \
  --mortality data/intermediate/mortality_county_year.parquet \
  --controls-file outputs/selected_controls.json \
  --output data/intermediate/analysis_panel.parquet --coverage-dir outputs
```

Control selection uses the same county matching, buyer/drug restrictions and
fixed population cutoff as the panel. State rates divide each observed subtotal
by the population of counties reporting that outcome. Mortality missingness
does not restrict MME. `control_state_year_coverage.csv` retains counters and
availability before exclusions; it does not assume missing counts are zero.

For each outcome, the ranking compares state trends normalized by their own
pre-policy mean, using all years from 2006 through the year before the policy.
The score averages the two root-mean-square gaps. A missing pre-year or zero
pre-period mean makes that outcome score unavailable. The top five states with
both scores and no concurrent policy in the existing policy table are selected.
Unavailable candidates remain visible in the CSV rankings. Input file hashes,
parameters, and selections are saved in `control_selection_metadata.json`.

The ranking chooses the closest available matches; it does not establish
parallel trends. Florida's shipment trend and Washington's pre-2012 decline
still need discussion. The existing policy table does not systematically cover
mandatory PDMP-use laws. D1 reporting counties may also change over time.

PR #5 builds on the shared sample fixes in PR #4, which incorporates the D1 and
population-cutoff changes from PR #6. Review/merge #4 before #5 so the latter's
remaining diff isolates control selection and regenerated outputs.

## Workflow

- Work on a feature branch and merge into `main` through a reviewed pull request.
- Write code as `.py` scripts, not notebooks.
- Never modify source data; write cleaned outputs to `data/intermediate/`.
