#!/usr/bin/env python3
"""Obsolete: batch Bayesian fitting moved to scripts/data_pipeline/.

Phase 4 (Plan 04-01) replaced the PyEM/PyMC pipeline with NumPyro NUTS in
``nn4psych.bayesian.reduced_bayesian``. The canonical batch fitting entry
points are:

- Human data: ``scripts/data_pipeline/09_fit_human_data.py`` (Plan 04-03)
- RNN data:   ``scripts/data_pipeline/10_fit_rnn_data.py``   (Plan 04-04b)

This stub remains so cluster scripts that historically referenced
``cluster/batch_fit_bayesian.py`` fail loudly rather than silently importing
removed PyEM functions.
"""
from __future__ import annotations

import sys


def main() -> int:
    """Print obsolete notice to stderr and exit with code 2."""
    sys.stderr.write(
        "cluster/batch_fit_bayesian.py is obsolete (BAYES-01, Plan 04-01).\n"
        "Use scripts/data_pipeline/09_fit_human_data.py (humans, Plan 04-03)\n"
        "or scripts/data_pipeline/10_fit_rnn_data.py (RNN, Plan 04-04b).\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
