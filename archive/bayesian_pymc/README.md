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
