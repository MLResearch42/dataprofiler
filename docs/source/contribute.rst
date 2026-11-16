.. _contribute:

============
Contributing
============

Set up a development environment with `uv <https://docs.astral.sh/uv/>`_ (pip works too):

::

    git clone https://github.com/MLResearch42/dataprofiler.git
    cd dataprofiler
    uv venv && uv pip install -e ".[dev]"

Run the checks that CI runs:

::

    ruff check dataprep
    pytest -n auto

Build the package:

::

    uv build

Guidelines
==========

* Every public function needs a NumPy-style docstring with a runnable example.
* New behaviour needs a test in ``dataprep/tests``.
* Nothing may contact the network at import time or during normal use.
* Keep ``import dataprep.clean`` fast: import heavy or optional packages inside
  the function that needs them.
