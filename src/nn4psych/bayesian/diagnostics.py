"""MCMC convergence diagnostics, retry helper, and per-fit JSON summary.

Phase 4 reusable module for MCMC diagnostics via ArviZ. Used by:
- 04-02: parameter recovery validation
- 04-03: human schizophrenia fits (per-subject x per-condition)
- 04-04b: RNN cohort fits (per-seed x per-subject-replay)

References
----------
Nassar, M. R., et al. (2021). PMC8041039.
    Parameter recovery via NumPyro NUTS with ArviZ diagnostics.

Notes
-----
ArviZ FutureWarnings are suppressed at module level (upstream deprecations).
ArviZ 0.23.4 uses ``stat_focus`` kwarg in ``az.summary`` (not ``kind``).
See M3 fix documentation in 04-02-SUMMARY.md for the chosen code path.
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

import numpy as np
import arviz as az
from numpyro.infer import MCMC, NUTS


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


def to_jsonable(obj: object) -> object:
    """Recursively convert numpy scalars and arrays to Python builtins.

    Required before ``json.dumps`` — NumPy scalars/arrays are not JSON
    serializable by default (lesson from STATE.md decision 02-03).

    Parameters
    ----------
    obj : object
        Arbitrary nested structure (dict, list, numpy scalar/array,
        Python builtin).

    Returns
    -------
    object
        JSON-serializable equivalent (dicts, lists, float, int, str, None).

    Examples
    --------
    >>> import numpy as np
    >>> to_jsonable({'a': np.float32(1.5), 'b': np.array([1, 2])})
    {'a': 1.5, 'b': [1, 2]}
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Diagnostic extraction
# ---------------------------------------------------------------------------


def run_diagnostics(
    mcmc: MCMC,
    var_names: list[str] | None = None,
) -> dict:
    """Extract R-hat, ESS_bulk, and divergence count from a fitted MCMC.

    Wraps ArviZ ``az.from_numpyro``, ``az.rhat``, and ``az.ess`` to compute
    per-run convergence diagnostics. Divergences are documented but NOT
    used as a gate (Phase 4 CONTEXT.md 2026-04-29 decision).

    Parameters
    ----------
    mcmc : MCMC
        A NumPyro MCMC object that has already been run (``mcmc.run``
        called). Must include ``extra_fields=('diverging',)`` to get
        divergence count; if not, a ``RuntimeWarning`` is emitted and
        ``n_divergences`` is set to -1.
    var_names : list[str] or None
        Optional list of variable names to restrict diagnostics to.
        Default None (all variables in posterior).

    Returns
    -------
    dict
        Keys:
        - ``rhat_max`` (float): maximum R-hat across all variables.
        - ``ess_min`` (float): minimum ESS_bulk across all variables.
        - ``n_divergences`` (int): total divergent transitions, or -1 if
          ``extra_fields=('diverging',)`` was not passed to ``mcmc.run``.
        - ``passes_rhat`` (bool): ``rhat_max <= 1.01``.
        - ``passes_ess`` (bool): ``ess_min >= 400``.
        - ``passes_gate`` (bool): both R-hat and ESS gates pass.

    Notes
    -----
    ``passes_divergences`` is intentionally NOT returned — divergences are
    documented per fit, not used as a convergence gate (Phase 4 CONTEXT.md
    2026-04-29 relaxation of original ROADMAP SC-3/SC-4 "zero divergences").

    Divergence count of -1 signals "not measured" (MCMC was run without
    ``extra_fields``). Downstream JSON encodes -1 as ``null`` via
    ``to_jsonable``.
    """
    idata = az.from_numpyro(mcmc)

    # Restrict to requested var_names if specified
    rhat_ds = az.rhat(idata, var_names=var_names)
    ess_ds = az.ess(idata, method="bulk", var_names=var_names)

    # Compute max R-hat across all data variables (handles multi-dim arrays)
    rhat_values = []
    for v in rhat_ds.data_vars:
        arr = np.asarray(rhat_ds[v].values)
        rhat_values.append(float(arr.max()))
    rhat_max = max(rhat_values) if rhat_values else float("nan")

    # Compute min ESS_bulk across all data variables
    ess_values = []
    for v in ess_ds.data_vars:
        arr = np.asarray(ess_ds[v].values)
        ess_values.append(float(arr.min()))
    ess_min = min(ess_values) if ess_values else float("nan")

    # Divergences: extra_fields=('diverging',) required at mcmc.run time
    extra_fields = mcmc.get_extra_fields()
    div_field = extra_fields.get("diverging") if extra_fields else None
    if div_field is None:
        warnings.warn(
            "MCMC was run without extra_fields=('diverging',); "
            "divergence count unavailable",
            RuntimeWarning,
            stacklevel=2,
        )
        n_divergences = -1
    else:
        n_divergences = int(div_field.sum())

    passes_rhat = rhat_max <= 1.01
    passes_ess = ess_min >= 400.0

    return {
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "n_divergences": n_divergences,
        "passes_rhat": passes_rhat,
        "passes_ess": passes_ess,
        "passes_gate": passes_rhat and passes_ess,
    }


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def fit_with_retry(
    model_fn: object,
    model_kwargs: dict,
    *,
    seed: int,
    num_warmup_first: int = 2000,
    num_warmup_retry: int = 4000,
    num_samples: int = 2000,
    num_chains: int = 4,
    target_accept_first: float = 0.95,
    target_accept_retry: float = 0.99,
    max_tree_depth: int = 10,
) -> tuple[MCMC, str, list[dict]]:
    """Fit MCMC with optional retry if convergence gates fail.

    Implements the RESEARCH.md Section 5 retry logic: first attempt uses
    conservative defaults (2000 warmup, target_accept=0.95); if R-hat or
    ESS gates fail, retries once with 2x warmup (4000) and
    target_accept=0.99. Both attempts are logged in the returned
    ``attempts`` list.

    Parameters
    ----------
    model_fn : callable
        NumPyro model function (e.g., ``reduced_bayesian_model``).
    model_kwargs : dict
        Keyword arguments passed to ``mcmc.run``. Must include the data
        (e.g., ``bag_positions``, ``bucket_positions``, ``context``).
        ``extra_fields=('diverging',)`` is appended automatically.
    seed : int
        Base random seed. Second attempt uses ``seed + 1`` to avoid
        identical random initialization.
    num_warmup_first : int
        NUTS warmup steps for first attempt (default: 2000).
    num_warmup_retry : int
        NUTS warmup steps for retry attempt (default: 4000).
    num_samples : int
        Posterior samples per chain for both attempts (default: 2000).
    num_chains : int
        Number of NUTS chains (default: 4).
    target_accept_first : float
        NUTS target acceptance probability for first attempt (default: 0.95).
    target_accept_retry : float
        NUTS target acceptance probability for retry (default: 0.99).
    max_tree_depth : int
        Maximum NUTS tree depth (default: 10).

    Returns
    -------
    mcmc : MCMC
        The MCMC object from the last attempt run (pass or fail).
    status : str
        ``'PASS'`` if any attempt met both R-hat and ESS gates;
        ``'FAILED'`` if all attempts failed.
    attempts : list[dict]
        Per-attempt diagnostic records. Each record contains:
        ``{'attempt': int, 'num_warmup': int, 'target_accept_prob': float,
        'rhat_max': float, 'ess_min': float, 'n_divergences': int,
        'passes_rhat': bool, 'passes_ess': bool, 'passes_gate': bool}``.

    Notes
    -----
    If status is ``'FAILED'``, the returned MCMC is from the last (retry)
    attempt. Downstream callers should inspect ``attempts`` to understand
    why both attempts failed. FAILED fits are documented in per-fit JSON;
    they do NOT raise exceptions (caller decides how to handle).
    """
    import jax

    attempts: list[dict] = []

    schedule = [
        (num_warmup_first, target_accept_first),
        (num_warmup_retry, target_accept_retry),
    ]

    for attempt_idx, (n_warmup, target_accept) in enumerate(schedule):
        kernel = NUTS(
            model_fn,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=n_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
        )
        rng_key = jax.random.PRNGKey(seed + attempt_idx)
        mcmc.run(rng_key, **model_kwargs, extra_fields=("diverging",))

        diag = run_diagnostics(mcmc)
        record: dict = {
            "attempt": attempt_idx + 1,
            "num_warmup": n_warmup,
            "target_accept_prob": target_accept,
            **diag,
        }
        attempts.append(record)

        if diag["passes_gate"]:
            return mcmc, "PASS", attempts

    return mcmc, "FAILED", attempts


# ---------------------------------------------------------------------------
# Per-fit JSON summary
# ---------------------------------------------------------------------------


def make_fit_summary(
    mcmc: MCMC,
    *,
    status: str,
    attempts: list[dict],
    var_names: list[str],
    **meta: object,
) -> dict:
    """Build a trimmed per-fit JSON summary (~10 KB) from a fitted MCMC.

    Aggregates per-parameter posterior statistics (mean, median, SD,
    95% HDI, R-hat, ESS_bulk) and fit metadata into a JSON-serializable
    dict. Designed to be written to disk via ``json.dump`` for Phase 5
    ingestion.

    Parameters
    ----------
    mcmc : MCMC
        Fitted NumPyro MCMC object.
    status : str
        Fit status, typically from ``fit_with_retry``: ``'PASS'`` or
        ``'FAILED'``.
    attempts : list[dict]
        Per-attempt diagnostic records from ``fit_with_retry``.
    var_names : list[str]
        Parameter names to summarize (e.g.,
        ``['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']``).
    **meta : object
        Arbitrary metadata to embed at the top level (e.g.
        ``subject_id='sub01'``, ``condition='changepoint'``,
        ``seed=42``).

    Returns
    -------
    dict
        JSON-serializable dict (all numpy types converted via
        ``to_jsonable``). Top-level keys:
        - All ``meta`` keys passed as kwargs.
        - ``status``: str.
        - ``attempts``: list[dict].
        - ``params``: dict mapping param name to per-param stats.
        - ``n_divergences``: int or None (-1 encoded as null).
        - ``n_chains``: int.
        - ``n_samples_per_chain``: int.

        Per-param stat keys: ``mean``, ``median``, ``sd``,
        ``hdi_2.5``, ``hdi_97.5``, ``rhat``, ``ess_bulk``.

    Notes
    -----
    M3 fix (ArviZ kwarg): ArviZ 0.23.4 uses ``stat_focus='stats'`` /
    ``stat_focus='diagnostics'`` in ``az.summary``, NOT ``kind=``. Using
    ``kind=`` raises ``TypeError`` at runtime. This function tries
    ``stat_focus`` first, then falls back to a single-call approach
    slicing the DataFrame by column name if ``stat_focus`` is not
    accepted (future-proofing for ArviZ version changes).

    All numpy scalars are recursively converted to Python builtins via
    ``to_jsonable`` before return (STATE.md lesson 02-03: cast numpy
    int64/float32 before json.dump).
    """
    idata = az.from_numpyro(mcmc)

    # M3 fix: Try stat_focus (ArviZ 0.23.4 pattern), fall back to single call
    param_stats: dict[str, dict] = {}
    try:
        summary_stats = az.summary(
            idata,
            var_names=var_names,
            stat_focus="stats",
            round_to="none",
        )
        summary_diag = az.summary(
            idata,
            var_names=var_names,
            stat_focus="diagnostics",
            round_to="none",
        )

        for param in var_names:
            row_stats = summary_stats.loc[param]
            row_diag = summary_diag.loc[param]
            hdi = az.hdi(idata, hdi_prob=0.95, var_names=[param])
            posterior_arr = np.asarray(idata.posterior[param])
            param_stats[param] = {
                "mean": float(row_stats["mean"]),
                "median": float(np.median(posterior_arr)),
                "sd": float(row_stats["sd"]),
                "hdi_2.5": float(np.asarray(hdi[param].values)[0]),
                "hdi_97.5": float(np.asarray(hdi[param].values)[1]),
                "rhat": float(row_diag["r_hat"]),
                "ess_bulk": float(row_diag["ess_bulk"]),
            }

    except TypeError:
        # Fallback: single az.summary call, slice by column names
        # Used when stat_focus is not accepted (older/newer ArviZ versions)
        summary_all = az.summary(
            idata,
            var_names=var_names,
            round_to="none",
        )
        for param in var_names:
            row = summary_all.loc[param]
            hdi = az.hdi(idata, hdi_prob=0.95, var_names=[param])
            posterior_arr = np.asarray(idata.posterior[param])
            param_stats[param] = {
                "mean": float(row["mean"]),
                "median": float(np.median(posterior_arr)),
                "sd": float(row["sd"]),
                "hdi_2.5": float(np.asarray(hdi[param].values)[0]),
                "hdi_97.5": float(np.asarray(hdi[param].values)[1]),
                "rhat": float(row["r_hat"]),
                "ess_bulk": float(row["ess_bulk"]),
            }

    # Divergences from extra_fields
    extra_fields = mcmc.get_extra_fields()
    div_field = extra_fields.get("diverging") if extra_fields else None
    n_divergences: int | None = (
        int(div_field.sum()) if div_field is not None else None
    )

    result = {
        **meta,
        "status": status,
        "attempts": attempts,
        "params": param_stats,
        "n_divergences": n_divergences,
        "n_chains": mcmc.num_chains,
        "n_samples_per_chain": mcmc.num_samples,
    }

    return to_jsonable(result)
