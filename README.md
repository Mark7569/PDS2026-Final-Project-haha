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

Install the Python dependencies for population cleaning:

```bash
python -m pip install pandas pyarrow
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
from the population table, then inner-join mortality as specified in the
analysis plan. Report and inspect county-years dropped by the join; do not treat
absent mortality records as zero.

The primary analysis uses counties with 2006 population of at least 150,000.
Among Florida, Washington, and the selected control states, D1 was reported for
90.4% of county-years at a 100,000 cutoff (190 counties), 96.1% at 150,000
(153 counties), and 98.6% at 250,000 (105 counties). The 150,000 cutoff keeps
at least four counties in each selected state while substantially reducing
unreported D1 counts; 129 of the 153 in-scope counties have D1 reported in all
ten years. Missing D1 values remain unavailable. Both the panel builder and
analysis script accept `--min-population-2006` to rerun the analysis with
alternative thresholds.

## Workflow

- Work on a feature branch and merge into `main` through a reviewed pull request.
- Write code as `.py` scripts, not notebooks.
- Never modify source data; write cleaned outputs to `data/intermediate/`.
