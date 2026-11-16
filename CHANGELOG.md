# Changelog

## 0.5.0 (unreleased): first release as `dataprep-next`

The first release of the maintained fork of [DataPrep](https://github.com/sfu-db/dataprep)
0.4.6. Install with `pip install dataprep-next`; the import name is still `dataprep`.

### New features

- `dataprep.eda.summarize(df)`: the statistics and data-quality insights of every column as a
  pandas DataFrame with a fixed set of columns, plus table-level figures in
  `summary.attrs["dataset"]`. Takes the same `dtype` and insight `config` as `plot`.
- Target analysis: `plot(df, target="y")` ranks every column by its association with a target
  column and shows the distribution of the target; `rank_features(df, "y")` returns the
  ranking as a DataFrame and `summarize(df, target="y")` adds it to the summary. Scores are
  the Spearman correlation, the bias-corrected correlation ratio or the bias-corrected
  Cramér's V, depending on the column types. ID-like columns, possible target leakage and
  imbalanced classes are flagged.
- `sample=` on `plot`, `plot_correlation`, `plot_missing`, `plot_diff`, `create_report`,
  `create_diff_report`, `summarize` and `rank_features`: work on a reproducible random sample
  of the rows, given as a number of rows or a fraction. Outputs say that they show a sample.
  Dask input is sampled before it is loaded into memory.
- 81 more cleaners built on `python-stdnum`, each with a `validate_*` function: 74 national
  identifiers (among them the US SSN, EIN, ITIN, ATIN, PTIN, TIN and bank routing numbers, and
  identifiers from Sweden, Turkey, Russia, Ukraine, Slovenia, Slovakia, South Africa, Singapore,
  Thailand, Pakistan, Indonesia and more) and 7 international codes: CFI, GS1-128, ISNI,
  ISO 6346 container codes, ISO 11649 creditor references, ISRC and MAC addresses. Personal
  numbers that encode a birth date or gender offer them as output formats.

### Requirements

- Python 3.10-3.13 (was 3.8-3.12).
- pandas 2.2 or newer, including pandas 3 (was pandas 1.x only).
- NumPy 1.26 or newer, including NumPy 2.
- bokeh 3.4 or newer (was bokeh 2.x only).
- pydantic 2 (was pydantic 1).
- Packaging moved from Poetry to a standard `pyproject.toml` built with hatchling.
- The core install needs only pre-built wheels. Optional extras: `dask`, `duplication`, `gui`,
  `nlp`, `notebook` and `all`. Dask, Flask, NLTK, ipywidgets and Metaphone are no longer required.

### Removed

- `dataprep.connector`. It downloaded API configurations from GitHub at run time, so it could
  not work on closed networks, and most of the APIs it targeted have changed.
- `dataprep.eda.create_db_report` and `dataprep.datasets.load_db` / `get_db_names`.
- `dataprep.lineage`.
- The bundled jQuery and pagination plugin used by plot outputs.

### Exploratory data analysis

- **Computation runs on pandas and NumPy in memory instead of Dask.** Every function is 2-16x
  faster (see the benchmarks in the README). Dask DataFrames are still accepted when the
  `dask` extra is installed; they are converted to pandas, so the data must fit in memory.
- Polars DataFrames, PyArrow tables and other objects with `to_pandas()` are accepted.
- Per-column work runs on a thread pool: `dataprep.eda.set_num_workers(n)` or the
  `DATAPREP_NUM_WORKERS` environment variable control its size.
- Rendering many charts is linear in the number of columns instead of quadratic (a bokeh
  serialization bottleneck): `plot(df)` on an 81-column table went from 5.2 s to 1.2 s.
- Notebook output is drawn in an isolated frame. This fixes blank or broken output in the
  VS Code notebook editor, styles leaking between outputs, and unreadable charts on dark
  themes. Frames resize to their content and pass mouse-wheel scrolling on to the notebook.
- Nothing is loaded from the internet: the web font the report fetched from Google Fonts and
  the dataprep.ai link were removed. `set_bokeh_resources("cdn")` opts into smaller outputs
  that need internet access to view.
- Outputs only embed the parts of BokehJS they use (1.4 MB instead of 4.2 MB per output).
- Plot objects gained `to_html()`; `save()` accepts `pathlib.Path`. `Report.save()` returns the
  path it wrote. Both have an informative `repr`.
- The two copies of the type system (`dtypes` and `dtypes_v2`) were merged; `dtypes_v2` remains
  as an alias. Types can be given as strings, classes or instances, and unknown type names
  raise a clear `ValueError`.
- Clear errors for empty DataFrames, unknown columns, a column given twice, too few numerical
  columns for `plot_correlation`, and invalid `plot(df, x, y, z)` combinations.
- `plot(df, x, y, z)` accepts the datetime, numerical and categorical columns in any order.
- Fixed: `plot(df, x, y)` failed for a datetime column paired with a country or lat/long column.
- Fixed: numbers written with a decimal comma ("12,5") were detected as latitude/longitude pairs.
- Fixed: the lower whisker of the box plot in `plot_missing(df, x, y)` was computed from Q3.
- Fixed: `plot_diff` mixed up colours, row counts and baselines when the DataFrames did not
  share the same columns; the stats table showed a wrong "Average Row Size in Memory"; a debug
  `print` of the statistics was removed.
- Fixed: correlation insights modified the correlation matrix in place and ignored undefined
  (NaN) correlations inconsistently.
- Fixed: a column without any values was reported as having "all distinct values".
- `dataprep.datasets` no longer needs Dask, so `load_dataset` works on a core install.
- Reports have an explicit white background, so they stay readable in browsers set to dark mode.
- Pagination in `plot(df)` (tables with 18 or more columns) was rewritten without jQuery.
- Removed the dead `outlier` module and the call to `output_notebook()` on import.

### Data cleaning

- `import dataprep.clean` takes 0.02 s instead of 3.7 s: functions are imported on first use,
  and optional packages (Flask, NLTK, Dask) only when a function needs them.
- The 134 existing national-identifier cleaners share one implementation built on `python-stdnum`
  (about 22,000 lines removed) while keeping every public name, signature and docstring.
- pandas DataFrames are cleaned with pandas directly instead of being converted to Dask.
- Works with pandas' new default string dtype and with `pd.NA`.
- Fixed: `clean_au_abn` returned an empty tuple instead of a DataFrame.
- Fixed: 154 `validate_*` functions used `DataFrame.applymap`, which pandas 3 removed.
- `clean_currency` currency conversion fails with a clear message and a timeout instead of
  hanging when the exchange-rate service cannot be reached.
- `clean_df_gui` works with Flask 3 and flask-cors 6, picks a free port, opens the system
  browser from VS Code notebooks and embeds the app in Jupyter. It returns a handle with the
  cleaned DataFrame (`gui.df`), the code for the cleaning steps (`gui.code`) and `stop()`.
- `pytz` was replaced by the standard library's `zoneinfo`.

### Tests and tooling

- About 2,900 tests (up from 185 test functions), including a test for every `clean_*` /
  `validate_*` pair and a check that Clean and EDA run with Dask, Flask and NLTK unavailable.
- CI on GitHub Actions for Python 3.10-3.13 on Linux, macOS and Windows, against the newest
  and the oldest supported dependency versions; releases publish to PyPI with trusted
  publishing.
- Linting with ruff replaces pylint, pytype, pyright and black.
