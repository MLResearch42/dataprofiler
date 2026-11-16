**********************
DataPrep Documentation
**********************

**Version**: |release|

DataPrep lets you explore and clean pandas DataFrames with a few lines of code.
This is the maintained continuation of the original
`DataPrep <https://github.com/sfu-db/dataprep>`_ project, published on PyPI as
``dataprep-next``. The import name is unchanged: ``import dataprep``.

You can use DataPrep to:

- Explore your data with interactive plots and profiling reports (:ref:`userguide/eda`)
- Get the same statistics as a DataFrame, rank columns against a target, and profile
  large tables from a sample (:ref:`userguide/eda/summarize`)
- Clean and standardize column values, including national identifiers from 88 countries
  (:ref:`userguide/clean`)

Every plot and report is a self-contained HTML document: nothing is loaded from
the internet, so DataPrep works on closed networks and inside Jupyter, JupyterLab
and the VS Code notebook editor.

.. toctree::
    :maxdepth: 2
    :hidden:

    installation
    user_guide/user_guide
    api_reference/dataprep
    contribute
    acknowledgement
