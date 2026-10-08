# TellCo User Analytics

User overview, engagement, experience and satisfaction analytics for TellCo's xDR data: an installable
Python package (`tellco`), a Streamlit dashboard, a versioned feature store and MLflow model tracking.
Prepared by Srishti Sharma, NextHikes IT Solutions.

**Headline result:** one week of session data, with no revenue, cost or churn data, supports a recommendation
to *continue due diligence*, not a buy or reject decision. The satisfaction score is a behavioural proxy, not
measured satisfaction, and the regression that predicts it recovers a formula rather than forecasting behaviour.

## Data

The raw xDR export (`telcom_data*.xlsx`) is **not included** in this repository because it contains customer
identifiers (MSISDN). Place your copy in `data/`, or upload it from the dashboard sidebar. Customer-level
output tables are also not committed.

On first load the dashboard writes a pickle cache next to the Excel file (`telcom_data*.cache.pkl`). It holds
the raw rows, so it is git-ignored like the export itself.

## Quick start

Requires Python 3.11 or newer (developed and tested with Python 3.14). `requirements.txt` lists the exact
versions used.

```bash
pip install -r requirements.txt                    # optional: reproduce the exact tested environment
pip install -e ".[dashboard]"                      # install the package and dashboard extras
pip install -e ".[notebook]"                       # optional: extras needed to re-run the analysis notebook
tellco-run --data data/telcom_data.xlsx            # clean, cluster, score, train, track, build feature store
streamlit run app.py                               # open the dashboard
```

The dashboard looks for `telcom_data*.xlsx` in the project folder or in `data/`. If it finds none, it asks
you to upload the file.

To export the scored table to MySQL as well, create the database first (for example
`CREATE DATABASE tellco;`), then run:

```bash
tellco-run --data data/telcom_data.xlsx --mysql USER:PASSWORD@localhost:3306/tellco
```

Anything typed on the command line can appear in your shell history, so avoid using a real password on a
shared machine. The dashboard has an equivalent export form on the Export & Tracking page.

To browse tracked model runs:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

| Module | Purpose |
|---|---|
| `tellco.pipeline` | Cleaning, per-customer aggregation, outlier treatment, k-means clustering, engagement / experience / satisfaction scores, robustness analyses |
| `tellco.modeling` | Regression models, loss convergence, permutation importance, SHAP, model save and load |
| `tellco.tracking` | MLflow run logging (parameters, metrics, loss steps, artifacts) |
| `tellco.feature_store` | Versioned parquet snapshots with a JSON registry |
| `tellco.db` | Export to MySQL (or any SQLAlchemy database) with a verification `SELECT` |
| `tellco.cli` | The `tellco-run` command |

## Repository contents

| Path | Contents |
|---|---|
| `app.py` | Streamlit dashboard (11 sections, including robustness checks and a what-if scorer) |
| `tellco/` | Installable analysis package |
| `notebooks/` | Analysis notebook and MySQL export script |
| `figures/` | Figures produced by the notebook |
| `docs/` | Slide deck (20 slides or fewer), written report, problem statement, `dashboard_screenshot.png`, `mysql_select_screenshot.png`, MLflow screenshots |
| `feature_store/registry.json` | Feature-store registry (data snapshots are not committed) |
| `outputs/` | Run log, model comparison, loss curve and saved models (the MLflow database and `mysql_export.json` are git-ignored) |
| `project.json` | Project metadata read by the dashboard (`github_repo`) |
| `requirements.txt` | Exact tested versions |

## Notes on the data

* `Dur. (ms)` is in **seconds** (it matches End − Start); `Dur. (ms).1` is the true millisecond value and is dropped.
* Rows without `MSISDN/Number` cannot be attributed to a customer and are dropped (0.71% of rows and traffic).
* `Total DL (Bytes)` excludes `Other DL`; `Total UL (Bytes)` includes `Other UL`, so application shares are shares of
  the seven counters, not of total traffic.
* The data cover one week (sessions ending 24–30 April 2019), not a month.
* TCP retransmission and RTT are heavily imputed; throughput is the one network measure that is almost fully observed.
* The brief's outlier rule (replace with the mean of non-outliers) changes 15.0% of customers who carry 29.9% of raw
  traffic. Treated results are primary; untreated figures are shown alongside and should be used for heavy-user
  and concentration questions.
* The satisfaction score is a behavioural proxy (distance-based), not measured satisfaction.

## Licence

MIT. See `LICENSE`.