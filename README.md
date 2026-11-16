# DataPrep (dataprep-next)

**Fast exploratory data analysis and data cleaning for pandas, built for notebooks.**

`dataprep-next` is a maintained continuation of [DataPrep](https://github.com/sfu-db/dataprep),
whose last release was in 2024 and which no longer installs alongside current pandas, NumPy
and bokeh. The import name is unchanged, so existing code keeps working:

```python
from dataprep.eda import plot, create_report
from dataprep.clean import clean_email
```

- **One line per question:** `plot(df)` gives an interactive overview of every column,
  `create_report(df)` a full profiling report, `summarize(df)` the same statistics as a
  DataFrame, and `plot(df, target="y")` the columns most associated with a target.
- **Runs offline:** every plot and report is a single self-contained HTML document. Nothing is
  fetched from the internet, so it works on closed networks with only a PyPI mirror.
- **Works in VS Code notebooks** as well as Jupyter, JupyterLab, Colab and plain scripts.
- **Fast:** 2-16x faster than the original on the benchmarks below.
- **251 cleaning functions** for emails, phone numbers, dates, addresses, countries, URLs,
  IP addresses, currencies, financial codes and national identifiers from 88 countries,
  including US Social Security, employer and taxpayer numbers.

## Installation

Requires Python 3.10 or newer.

```bash
pip install dataprep-next
```

Optional extras:

| Extra | Adds |
|---|---|
| `dataprep-next[dask]` | Dask DataFrames as input |
| `dataprep-next[gui]` | `clean_df_gui()`, a point-and-click cleaning app (Flask) |
| `dataprep-next[nlp]` | stemming and lemmatizing in word statistics and `clean_text` (NLTK) |
| `dataprep-next[duplication]` | the interactive `clean_duplication()` tool |
| `dataprep-next[notebook]` | widget progress bars in notebooks |
| `dataprep-next[all]` | all of the above |

### On a machine without internet access

If the machine can reach an internal PyPI mirror, `pip install dataprep-next` is all you need.
To install from files instead, collect the wheels on a connected machine with the same
operating system and Python version, copy the folder across, and install from it.
`pip wheel` (rather than `pip download`) also builds a wheel for any dependency that is only
published as source code, so the offline install needs no build tools:

```bash
pip wheel "dataprep-next[all]" -w wheels/
```

```bash
pip install --no-index --find-links wheels/ "dataprep-next[all]"
```

Only two features use the network, and both are opt-in: currency conversion in
`clean_currency(..., target_currency=...)`, which fetches exchange rates, and
`set_bokeh_resources("cdn")`.

## Exploratory data analysis

```python
from dataprep.datasets import load_dataset
from dataprep.eda import create_report, plot, plot_correlation, plot_diff, plot_missing

df = load_dataset("titanic")

plot(df)                        # every column at a glance, with data-quality insights
plot(df, "Age")                 # one column in depth: histogram, KDE, Q-Q, box plot, stats
plot(df, "Age", "Sex")          # how two columns relate
plot_correlation(df)            # Pearson, Spearman and Kendall correlation matrices
plot_missing(df)                # where values are missing, and how they co-occur
plot_diff([df[df.Survived == 1], df[df.Survived == 0]])  # compare 2-5 DataFrames
create_report(df)               # all of the above in one report
```

With one datetime, one numerical and one categorical column, `plot(df, x, y, z)` draws the
number aggregated per category over time, e.g. `plot(sales, "order_date", "revenue", "region")`.

In a notebook the result displays when it is the last expression of a cell. Elsewhere, or to
keep a result:

```python
report = create_report(df, title="Titanic")
report.save("titanic.html")     # a standalone file you can e-mail or archive
report.show_browser()           # open in the system browser
plot(df, "Age").show()          # display now, e.g. inside a loop
```

Customize any chart with `config`, choose which charts to draw with `display`, and override
the detected column types with `dtype`:

```python
plot(df, "Fare", config={"hist.bins": 20, "hist.yscale": "log"})
plot(df, "Fare", display=["Histogram", "Box Plot"])
plot(df, "Pclass", dtype={"Pclass": "nominal"})
```

Every chart has a **?** button that lists the `config` keys that change it.

### Statistics as a DataFrame

`summarize(df)` returns the statistics and data-quality insights behind `plot(df)` as a
DataFrame with one row per column, for filtering, saving, comparing runs or checking in tests:

```python
from dataprep.eda import summarize

summary = summarize(df)
summary.loc[summary["missing_pct"] > 5, ["missing", "missing_pct"]]   # Age and Cabin
summary.loc["Fare", ["mean", "median", "max", "insights"]]            # ... "Skewed"
summary.attrs["dataset"]                                              # rows, duplicates, ...

assert summary.loc["PassengerId", "distinct"] == len(df)              # e.g. in a pipeline test
```

The columns of a summary are always the same (count, missing, distinct, top value, mean,
quantiles, skewness, zeros, text lengths, date range, insights, ...); statistics that do not
apply to a column are NaN.

### Which columns relate to a target?

```python
from dataprep.eda import rank_features

plot(df, target="Survived")         # ranking chart and the distribution of the target
rank_features(df, "Survived")       # the ranking as a DataFrame
```

Every column gets a score from 0 (no association) to 1 (determines the target): the Spearman
correlation for two numerical columns, the correlation ratio for a numerical and a categorical
column, and Cramér's V for two categorical columns. The last two are bias-corrected, so ID-like
columns do not rank high by chance. Columns that look like identifiers, scores high enough to
suggest target leakage, and imbalanced classes are flagged.

### Large tables

Every EDA function takes `sample=`, a number of rows or a fraction, and then works on a random
sample of the rows. The sample is the same on every run and the output says that it shows a
sample:

```python
create_report(big_df, sample=200_000)
plot(big_df, "price", sample=0.1)
```

### Column types

Each column is given a type that decides which statistics and charts it gets:

| Type | Typical columns |
|---|---|
| Continuous | numbers with 10 or more distinct values |
| SmallCardNum | numbers with fewer than 10 distinct values, plotted as categories |
| Nominal | strings, booleans, `category`, mixed objects |
| DateTime | `datetime64` columns, with or without a time zone |
| GeoGraphy | strings that are mostly country names or codes, drawn on a world map |
| GeoPoint | strings that are mostly `"latitude, longitude"` pairs |

Data that is already in memory as pandas is used directly. Dask DataFrames (with the `dask`
extra), Polars DataFrames, PyArrow tables and anything else with a `to_pandas()` method are
converted to pandas first, so they need to fit in memory.

## Data cleaning

Every column cleaner takes a DataFrame and a column name and returns a new DataFrame with a
`<column>_clean` column. The general-purpose cleaners (email, phone, date, country, URL, IP,
address, coordinates, currency, column headers and `clean_df`) also print a short summary of
what they changed:

```python
import pandas as pd
from dataprep.clean import clean_country, clean_email, clean_phone, validate_email

df = pd.DataFrame({
    "email": ["Jane.Doe@Example.COM", "not an email"],
    "phone": ["555.234.5678", "(555) 234 5678 x99"],
    "country": ["Canada", "DEU"],
})

df = clean_email(df, "email")                       # invalid values become NaN
df = clean_phone(df, "phone", output_format="e164")
df = clean_country(df, "country", output_format="alpha-2")
validate_email(df["email"])                         # True / False per value
```

Options shared by the column cleaners: `inplace=True` replaces the column instead of adding
one, `errors="ignore"` keeps unparseable values instead of setting them to NaN,
`errors="raise"` stops at the first one, and `report=False` turns off the printed summary
where there is one.

| Area | Functions |
|---|---|
| Whole tables | `clean_df`, `clean_headers`, `clean_duplication`, `clean_ml`, `clean_df_gui` |
| Text and contacts | `clean_text`, `clean_email`, `clean_phone`, `clean_url`, `clean_address` |
| Places and time | `clean_country`, `clean_lat_long`, `clean_date` |
| Technical values | `clean_ip`, `clean_json`, `clean_currency` |
| Financial and product codes | `clean_iban`, `clean_bic`, `clean_isin`, `clean_cusip`, `clean_lei`, `clean_figi`, `clean_cfi`, `clean_isbn`, `clean_issn`, `clean_isrc`, `clean_isni`, `clean_ean`, `clean_gs1_128`, `clean_imei`, `clean_mac`, `clean_iso6346` (shipping containers), `clean_iso11649` (payment references), `clean_vatin`, ... |
| National identifiers | 208 functions named `clean_<country code>_<identifier>` for 88 countries and the EU, e.g. `clean_us_ssn`, `clean_us_ein`, `clean_us_rtn`, `clean_ca_sin`, `clean_gb_nhs`, `clean_de_vat`, `clean_br_cpf`, `clean_in_gstin`, `clean_se_personnummer`, `clean_za_idnr`. |

Every `clean_*` function has a matching `validate_*` function.

`clean_df_gui(df)` starts a local cleaning app. In VS Code it opens in your browser; in Jupyter
it is embedded in the cell output. The returned handle gives you the cleaned DataFrame
(`gui.df`) and the Python code that reproduces your steps (`gui.code`).

## Using DataPrep in VS Code

Plots and reports work in the VS Code notebook editor and the interactive window with no
extra setup:

- Each output is drawn in its own frame, so several plots in one notebook do not interfere
  with each other, and charts keep a white background on dark themes.
- Frames resize to their content, and the mouse wheel scrolls the notebook even when the
  pointer is over a chart.
- Progress bars appear as plain text bars. Install the `notebook` extra for widget bars.

Each output embeds the charting library (about 1.5 MB) so that it works offline and still
renders when the notebook is reopened. If size matters more than offline use, call
`dataprep.eda.set_bokeh_resources("cdn")` at the top of the notebook.

## Performance

Measured on an Apple M-series laptop, best of two runs, against the original DataPrep 0.4.6 on
its own 2022 stack (pandas 1.5, dask 2022.12, bokeh 2.4):

| Call | Original | dataprep-next |
|---|---|---|
| 81 columns x 1,460 rows: `plot(df)` | 5.2 s | 1.2 s |
| 81 columns x 1,460 rows: `create_report(df)` | 12.0 s | 5.1 s |
| 17 columns x 1,000,000 rows: `plot(df)` | 6.4 s | 1.0 s |
| 17 columns x 1,000,000 rows: `plot_correlation(df)` | 10.8 s | 2.4 s |
| 17 columns x 1,000,000 rows: `plot_missing(df)` | 3.7 s | 0.2 s |
| 17 columns x 1,000,000 rows: `create_report(df)` | 17.4 s | 4.8 s |

`import dataprep.clean` takes 0.02 s, down from 3.7 s. Per-column work runs on a thread pool;
set its size with `dataprep.eda.set_num_workers(n)` or the `DATAPREP_NUM_WORKERS` environment
variable. For tables with many millions of rows, `sample=` (see above) is the quickest way to
a first look.

## Changes from the original DataPrep

- Works with Python 3.10-3.13, pandas 2.2-3.x, NumPy 1.26-2.x, bokeh 3.4+ and pydantic 2.
- EDA computes with pandas and NumPy in memory instead of Dask, which made it both simpler and
  faster. Dask is now an optional extra.
- Removed: `dataprep.connector` (it downloaded API configurations from GitHub at run time),
  `create_db_report` and `dataprep.lineage`.
- Plots and reports no longer load anything from the internet (the original fetched a web font).
- Plot outputs are isolated in frames, which fixes broken layouts when several plots share a
  notebook and blank outputs in VS Code.
- New: `summarize()`, target analysis (`plot(df, target=...)`, `rank_features()`), the
  `sample=` option, and 81 more identifier cleaners, including the US ones.
- Clearer error messages for empty data, unknown columns and unsupported column combinations.
- Many bug fixes; see [CHANGELOG.md](CHANGELOG.md).

## Development

```bash
git clone https://github.com/MLResearch42/dataprofiler.git
cd dataprofiler
uv venv && uv pip install -e ".[dev]"
pytest -n auto
```

## License and credits

MIT licensed. DataPrep was created by the
[SFU Database Systems Lab](https://github.com/sfu-db); this fork keeps their copyright notice
and builds on their work.
