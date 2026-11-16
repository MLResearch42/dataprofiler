.. _installation:

============
Installation
============

DataPrep needs Python 3.10 or newer. Install it from PyPI:

::

    pip install dataprep-next

and import it as ``dataprep``:

.. code-block:: python

    from dataprep.eda import plot, create_report
    from dataprep.clean import clean_email

Optional extras
===============

The core install covers ``plot``, ``plot_correlation``, ``plot_missing``,
``plot_diff``, ``create_report``, ``create_diff_report`` and every ``clean_*``
function except ``clean_duplication()`` on pandas DataFrames. Extras add the rest:

* ``pip install "dataprep-next[dask]"``: accept Dask DataFrames as input
* ``pip install "dataprep-next[gui]"``: ``clean_df_gui()``, the point-and-click cleaning app
* ``pip install "dataprep-next[nlp]"``: stemming and lemmatizing (NLTK)
* ``pip install "dataprep-next[duplication]"``: the interactive ``clean_duplication()`` tool
* ``pip install "dataprep-next[notebook]"``: widget progress bars in notebooks
* ``pip install "dataprep-next[all]"``: everything above

Lemmatizing additionally needs NLTK's ``wordnet`` corpus, which NLTK downloads
separately. Stemming needs no extra data.

Installing on a machine without internet access
===============================================

DataPrep never contacts the internet at import time or while plotting. If your
machine can only reach an internal PyPI mirror, ``pip install dataprep-next``
works as long as the mirror carries its dependencies. To install from files
instead, collect the wheels on a connected machine with the same operating
system and Python version. ``pip wheel`` (rather than ``pip download``) also
builds a wheel for any dependency that is only published as source code:

::

    pip wheel "dataprep-next[all]" -w wheels/

copy the ``wheels`` folder across, and install from it:

::

    pip install --no-index --find-links wheels/ "dataprep-next[all]"

The only features that use the network are ``clean_currency(..., target_currency=...)``,
which fetches exchange rates, and ``set_bokeh_resources("cdn")``, which you
have to opt into.

Supported environments
======================

* Jupyter Notebook and JupyterLab
* The VS Code notebook editor and interactive window
* Google Colab and Kaggle notebooks
* Plain Python scripts: use ``.save("file.html")`` or ``.show_browser()``
