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

## Workflow

- Work on a feature branch and merge into `main` through a reviewed pull request.
- Write code as `.py` scripts, not notebooks.
- Never modify source data; write cleaned outputs to `data/intermediate/`.



