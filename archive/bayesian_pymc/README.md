# Archived Bayesian Models (PyMC / PyEM)

These files were the original Bayesian model implementations using PyMC and a custom PyEM framework.
They have been superseded by the NumPyro implementation at `src/nn4psych/bayesian/numpyro_models.py`.

## Files

- `bayesian_models.py` — PyMC-based MCMC model (requires pymc, pytensor)
- `pyem_models.py` — Custom PyEM-based MLE/MAP estimation (pure numpy/scipy)
- `fit_bayesian_pymc.py` — Fitting script that uses bayesian_models.py
- `fit_bayesian_pyem.py` — Fitting script that uses pyem_models.py

## Why Archived

- NumPyro/JAX is faster and composable with the rest of the JAX-based analysis pipeline
- PyMC and PyTensor add heavy dependencies that conflict with JAX
- The PyEM approach provides only point estimates; NumPyro provides full posteriors

## Restoring

These files are preserved in git history. To restore:

```
git show HEAD:bayesian/bayesian_models.py > bayesian_models.py
git show HEAD:bayesian/pyem_models.py > pyem_models.py
```

## BAYES-01 (Phase 4)

These files are intentionally archived. They are NOT importable from the main
`nn4psych.bayesian` package. Verified by Plan 04-01:

- The Phase 4 canonical Bayesian implementation lives in
  `src/nn4psych/bayesian/reduced_bayesian.py`.
- PyEM (`pyem_models.py`, `bayesian_models.py`) and PyMC
  (`fit_bayesian_pymc.py`) implementations here are kept only for git-history
  reference and to support reading legacy fit JSONs.
- Re-importing from this directory into `src/nn4psych/` is a regression and
  will fail BAYES-01 traceability in REQUIREMENTS.md.
