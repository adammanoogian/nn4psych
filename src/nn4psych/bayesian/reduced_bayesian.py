"""Reduced Bayesian Observer (RBO) model for the helicopter-bag task.

Implements the Nassar 2010+2021 Reduced Bayesian Observer in NumPyro/JAX.
This is the canonical Phase 4 forward model.

References
----------
Nassar, M. R., Wilson, R. C., Heasly, B., & Gold, J. I. (2010).
    An approximately Bayesian delta-rule model explains the dynamics of
    belief updating in a changing environment.
    Journal of Neuroscience, 30(37), 12366-12378.

Nassar, M. R., et al. (2021). PMC8041039. Reduced Bayesian observer for
    changepoint/oddball conditions. Equations referenced:
    - Omega (changepoint probability): Eq. 4
    - tau (relative uncertainty):      Eq. 5
    - alpha (learning rate):           Eqs. 2-3
    - update (mean estimate):          Eqs. 6-7

Notes
-----
Priors below are weakly-informative defaults until verified against
Nassar 2021 supplement (see Plan 04-03 Task 1 for Brain2021Code data
fetch which gates supplement access).

Math-symbol naming (H, LW, UU, tau, omega, alpha) is intentionally
contained within this module. Descriptive names are used at the API
boundary per project three-layer naming convention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np
from jax.scipy.stats import norm as jax_norm
from jax.scipy.stats import uniform as jax_uniform
from numpyro.infer import MCMC, NUTS, Predictive

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SIGMA_N: float = 20.0
"""Bag placement noise (SD), in screen-space units (range 0-300).

Confirmed from Nassar 2021 PMC text. This is a FIXED generative constant,
not a free parameter.
"""

BAG_RANGE: float = 300.0
"""Uniform support for the bag position (screen-space range 0-300)."""


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


def compute_rbo_forward(
    params: dict[str, jax.Array],
    pred_errors: jax.Array,
    context: str,
    sigma_N: float = SIGMA_N,
    new_block: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute Reduced Bayesian Observer forward pass given parameters.

    Core forward model implementing Equations 2-7 from Nassar 2010/2021.
    JAX-traceable: uses ``jax.lax.scan`` + ``jax.lax.cond`` with the
    Phase 1 (01-03) pattern — ``is_changepoint`` computed OUTSIDE
    ``step_fn`` so it is a closed-over, tracer-compatible constant.

    Parameters
    ----------
    params : dict[str, jax.Array]
        Free parameters with keys: ``'H'``, ``'LW'``, ``'UU'``.
        ``'sigma_motor'`` and ``'sigma_LR'`` are not needed for the
        forward pass (they enter the likelihood only).
    pred_errors : jax.Array
        Prediction errors ``delta_t = bag_t - bucket_t``, shape ``(n,)``.
    context : str
        ``'changepoint'`` or ``'oddball'``.
    sigma_N : float
        Bag placement SD (fixed constant from paper, default 20.0).
    new_block : jax.Array or None
        Boolean array of shape ``(n,)`` marking the first trial of a new
        block.  When ``new_block[t]`` is True, ``tau`` is reset to
        ``tau_0=0.5`` before computing trial ``t`` (matches MATLAB
        ``getTrialVarsFromPEs_cannon.m:117-118``: ``if newBlock(i)
        errBased_RU(i)=initRU``).  If ``None``, the entire sequence is
        treated as one continuous run (no resets), suitable for synthetic
        recovery and within-block analyses.

    Returns
    -------
    learning_rate : jax.Array
        Per-trial learning rates ``alpha_t``, shape ``(n,)``.
    normative_update : jax.Array
        Predicted updates ``alpha_t * delta_t``, shape ``(n,)``.
    omega : jax.Array
        Per-trial changepoint/oddball probabilities ``Omega_t``, shape ``(n,)``.
    tau : jax.Array
        Relative uncertainty (includes initial ``tau_0``), shape ``(n+1,)``.

    Notes
    -----
    Equation references (Nassar 2010/2021):
    - Eq. 4: ``Omega_t`` — changepoint probability
    - Eq. 5: ``tau`` update — full predictive-variance-weighted form
    - Eqs. 2-3: ``alpha_t`` — learning rate (CP vs OB branch)
    - Eq. 6-7: Likelihood (computed in ``reduced_bayesian_model``)
    """
    H = params["H"]
    LW = params["LW"]
    UU = params["UU"]

    n_trials = pred_errors.shape[0]

    # Initial relative uncertainty: matches MATLAB frugFun5.m line 41
    # (R(1) = 1 → tau_0 = 1/(R+1) = 0.5). UU does NOT enter the init —
    # using 0.5/UU produces tau_0 > 1 for UU < 0.5, breaking sqrt(1-tau).
    tau_0 = 0.5

    # Compute is_changepoint OUTSIDE step_fn: closed-over, tracer-safe.
    # See Phase 1 lesson 01-03: jnp.bool_(context == 'changepoint') must
    # be evaluated at Python trace time (not inside jax.lax.scan).
    is_changepoint = jnp.bool_(context == "changepoint")

    # Default new_block to all-False if not provided.  This preserves
    # backward-compatible single-run behavior (used by synthetic recovery
    # and the matlab parity test).
    if new_block is None:
        new_block_arr = jnp.zeros(n_trials, dtype=jnp.bool_)
    else:
        new_block_arr = jnp.asarray(new_block, dtype=jnp.bool_)

    def step_fn(
        carry: jax.Array, t: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        """Single-trial update (scanned over all trials).

        Variable correspondence with MATLAB frugFun5.m:
            tau_prev  <->  yInt = 1/(R+1)   (relative weight; tau in [0,1])
            totSig    <->  MATLAB totSig = sigmaE / sqrt(1 - tau)
            omega_t   <->  pCha
            lr_t      <->  Alph = yInt + pCha*(1-yInt)  [CP]
                            or  yInt + pCha*(-yInt)      [OB]
        """
        tau_prev_raw = carry
        # MATLAB getTrialVarsFromPEs_cannon.m:117-118 — reset RU at every
        # newBlock boundary.  Use lax.cond for tracer-safe branching.
        tau_prev_raw = jax.lax.cond(
            new_block_arr[t],
            lambda _: jnp.asarray(tau_0),
            lambda _: tau_prev_raw,
            operand=None,
        )
        # Clip tau to (eps, 1-eps) for numerical stability of sqrt(1-tau)
        # and downstream variance terms. Math-required range is (0, 1).
        tau_prev = jnp.clip(tau_prev_raw, 1e-6, 1.0 - 1e-6)

        delta = pred_errors[t]

        # ---------------------------------------------------------------
        # EQUATION 4: Changepoint/Oddball Probability (Omega_t)
        # ---------------------------------------------------------------
        # Uniform component: MATLAB uses d = ones(300)/300, so changLike
        # = 1/300 for all trials (bag positions are always in [0, 300]).
        # This is a CONSTANT, not a density evaluated at delta.
        # Previous impl incorrectly used uniform.pdf(delta, 0, 300) which
        # returns 0 for delta < 0, causing omega=0 when bag < bucket.
        # Fix 04-03a: U_val = constant (1/BAG_RANGE)^LW.
        U_log = LW * jnp.log(1.0 / BAG_RANGE)  # = LW * log(1/300)

        # Normal component: MATLAB totSig = sigmaE / sqrt(1 - tau).
        # This is the predictive SD that includes both observation noise
        # (sigmaE) and process uncertainty (sigmaU = sigmaE*sqrt(tau/(1-tau))):
        #   totSig^2 = sigmaE^2 + sigmaU^2 = sigmaE^2 / (1 - tau)
        # Previous impl used sigma_N / tau (incorrect; off by sqrt factor).
        # Fix 04-03a: use sigma_N / sqrt(1 - tau) to match MATLAB totSig.
        tot_sig = sigma_N / jnp.sqrt(1.0 - tau_prev + 1e-10)
        N_log = LW * jax_norm.logpdf(delta, loc=0.0, scale=tot_sig)

        # Log-space change ratio matching MATLAB line 99:
        #   changeRatio = exp(LW*log(changLike/pI) + log(H/(1-H)))
        # Note: pI in MATLAB also uses the truncated-normal normalization
        # factor (normcdf(300,B,totSig) - normcdf(0,B,totSig)).  However,
        # since we operate in pred_error space (delta = bag - bucket), the
        # truncation correction cancels between U and N when bag ∈ [0,300].
        # Accepted deviation: truncation correction on N_log not applied
        # here (see 04-03a-SUMMARY.md for analysis).
        log_change_ratio = (
            U_log - N_log
            + jnp.log(H / (1.0 - H + 1e-10))
        )
        omega_t = jax.nn.sigmoid(log_change_ratio)

        # ---------------------------------------------------------------
        # EQUATION 5: Relative Uncertainty (tau_t)
        # Derived from MATLAB R-update (second-moment form, trueRun=0):
        #   ss = pCha*(sigmaE^2/1) + pNoCha*(sigmaE^2/(R+1))
        #          + pCha*pNoCha*(-(1-tau)*delta)^2
        #   R[i+1] = sigmaE^2 / ss
        #   tau[i+1] = 1 / (R[i+1] + 1) = ss / (ss + sigmaE^2)
        #
        # Note: MATLAB residual (B+yInt*Delta - data) = -(1-tau)*delta.
        # ---------------------------------------------------------------
        pNoCha = 1.0 - omega_t
        sigma_N_sq = sigma_N ** 2
        # CP second-moment ss (MATLAB frugFun5.m lines 120-121)
        residual_sq_cp = ((1.0 - tau_prev) * delta) ** 2
        ss_cp = (
            omega_t * sigma_N_sq
            + pNoCha * (sigma_N_sq * tau_prev)   # sigma_N^2/(R+1) = sigma_N^2*tau
            + omega_t * pNoCha * residual_sq_cp
        )
        # OB second-moment ss (MATLAB frugFun5_uniformOddballs.m lines 120-122)
        # 1st term: sigmaE^2/R = sigmaE^2*tau/(1-tau)
        # residual: yInt*Delta = tau*delta
        residual_sq_ob = (tau_prev * delta) ** 2
        ss_ob = (
            omega_t * (sigma_N_sq * tau_prev / (1.0 - tau_prev + 1e-10))
            + pNoCha * (sigma_N_sq * tau_prev)
            + omega_t * pNoCha * residual_sq_ob
        )
        # Select ss based on context (closed-over constant is_changepoint)
        ss = jax.lax.cond(
            is_changepoint,
            lambda _: ss_cp,
            lambda _: ss_ob,
            operand=None,
        )
        # MATLAB getTrialVarsFromPEs_cannon.m:184 — divide ss by ud (= UU)
        # BEFORE computing RU.  UU = exp(log_UU) ≥ 1 by construction, so
        # ss/UU ≤ ss and the resulting RU = (ss/UU) / (ss/UU + nVar) is
        # mathematically guaranteed in [0, 1].  This makes the post-hoc
        # tau_next clip unnecessary and keeps the gradient w.r.t. UU
        # well-defined across the entire prior support.
        ss_after_uu = ss / (UU + 1e-40)
        this_tau = ss_after_uu / (ss_after_uu + sigma_N_sq + 1e-40)
        tau_next = this_tau

        # ---------------------------------------------------------------
        # EQUATIONS 2 & 3: Learning Rate (alpha_t)
        # jax.lax.cond for JAX-compatible branching inside scan.
        # Python if/else would silently produce dead code inside scan.
        #   CP (frugFun5.m line 113):    Alph = yInt + pCha*(1-yInt)
        #   OB (frugFun5_uo.m line 109): Alph = yInt + pCha*(-yInt)
        # ---------------------------------------------------------------
        lr_t = jax.lax.cond(
            is_changepoint,
            lambda _: omega_t + tau_prev - (omega_t * tau_prev),  # Eq. 2: CP
            lambda _: tau_prev - (omega_t * tau_prev),  # Eq. 3: OB
            operand=None,
        )

        # ---------------------------------------------------------------
        # EQUATION 1: Normative Update
        # ---------------------------------------------------------------
        norm_update_t = lr_t * delta

        return tau_next, (lr_t, norm_update_t, omega_t, tau_next)

    # Run scan over all trials
    _, (learning_rate, normative_update, omega, tau_scan) = jax.lax.scan(
        step_fn, tau_0, jnp.arange(n_trials)
    )

    # Prepend initial tau so tau has shape (n_trials + 1,)
    tau = jnp.concatenate([jnp.array([tau_0]), tau_scan])

    return learning_rate, normative_update, omega, tau


# ---------------------------------------------------------------------------
# NumPyro probabilistic model
# ---------------------------------------------------------------------------


def reduced_bayesian_model(
    bucket_positions: jax.Array | None = None,
    bag_positions: jax.Array | None = None,
    context: str = "changepoint",
    sigma_N: float = SIGMA_N,
    new_block: jax.Array | None = None,
) -> None:
    """NumPyro model for the Nassar 2010+2021 Reduced Bayesian Observer.

    Defines the full generative model with paper-informed priors and
    the observation likelihood (Eqs. 6-7). Use with MCMC for posterior
    inference.

    Parameters
    ----------
    bucket_positions : jax.Array, optional
        Observed bucket positions (agent actions), shape ``(n_trials,)``.
    bag_positions : jax.Array, optional
        Observed bag landing positions, shape ``(n_trials,)``.
    context : str
        ``'changepoint'`` or ``'oddball'``.
    sigma_N : float
        Bag placement SD (fixed from paper, default 20.0).
    new_block : jax.Array or None
        Boolean array marking the first trial of each block; when True,
        relative uncertainty is reset.  See ``compute_rbo_forward``.

    Notes
    -----
    Prior choices match Nassar 2021 ``fitFrugFunSchiz.m:127-130`` exactly
    for the parameters the paper specifies (``H``, ``LW``, ``log_UU``).
    For ``sigma_motor`` (varInt) and ``sigma_LR`` (varSlope), the paper
    uses unbounded fmincon with no prior; we substitute weakly-informative
    HalfNormal priors that NUTS can sample efficiently — these are not
    constrained by paper specification because the paper used MAP not MCMC.
    """
    # -------------------------------------------------------------------
    # PRIORS — paper-faithful where the paper specifies them
    # -------------------------------------------------------------------

    # PAPER-VERIFIED: Nassar 2021 fitFrugFunSchiz.m:127.  Beta(1.1, 19)
    # has mean ≈ 0.055 and concentrates mass at low hazard rates while
    # leaving a long tail toward high values.  Picks up high-hazard
    # subjects without overweighting them a priori.
    H = numpyro.sample("H", dist.Beta(1.1, 19))

    # PAPER-VERIFIED: Nassar 2021 fitFrugFunSchiz.m:128.  Beta(2, 1) is
    # right-skewed, peaking at 1; favors near-Bayes-optimal LW values.
    LW = numpyro.sample("LW", dist.Beta(2, 1))

    # PAPER-VERIFIED prior: Nassar 2021 fitFrugFunSchiz.m:130 uses
    #   tParams(4) ~ Normal(0, 5) truncated to [0, 10]
    # where tParams(4) is "the log of the divisor" (line 42).  The actual
    # divisor used in getTrialVarsFromPEs_cannon.m:184 is exp(tParams(4)),
    # which is ≥ 1 by construction.  We sample log_UU and expose UU as a
    # deterministic so downstream code (synthetic data generator, recovery
    # report) can use the divisor value directly.
    log_UU = numpyro.sample(
        "log_UU",
        dist.TruncatedNormal(0.0, 5.0, low=0.0, high=10.0),
    )
    UU = numpyro.deterministic("UU", jnp.exp(log_UU))

    sigma_motor = numpyro.sample(
        "sigma_motor",
        dist.HalfNormal(10.0),
        # FALLBACK pending Nassar 2021 supplement.
        # Rationale: HalfNormal(10.0) is weakly regularizing in screen
        # units (0-300). Typical motor noise for this task is 5-30 units.
        # Scale chosen to avoid pinning near 0 while staying in the range
        # of physically plausible motor variability.
    )

    sigma_LR = numpyro.sample(
        "sigma_LR",
        dist.HalfNormal(1.0),
        # FALLBACK pending Nassar 2021 supplement.
        # Rationale: HalfNormal(1.0) keeps the multiplicative variance
        # slope small and positive. sigma_LR scales update variance
        # proportionally to update magnitude (Eq. 7).
    )

    # -------------------------------------------------------------------
    # LIKELIHOOD (only when observed data is provided)
    # -------------------------------------------------------------------
    if bucket_positions is not None and bag_positions is not None:
        n_trials = bucket_positions.shape[0]

        # Prediction errors: bag_t - bucket_t
        pred_errors = bag_positions - bucket_positions

        params = {
            "H": H,
            "LW": LW,
            "UU": UU,
            "sigma_motor": sigma_motor,
            "sigma_LR": sigma_LR,
        }

        learning_rate, normative_update, omega, tau = compute_rbo_forward(
            params, pred_errors, context, sigma_N=sigma_N, new_block=new_block
        )

        # EQUATION 6-7: Observed bucket updates
        # bucket_update_t = diff(bucket_positions), prepending first position
        # so the first update is treated as starting from the same position.
        bucket_update = jnp.diff(bucket_positions, prepend=bucket_positions[0])

        # EQUATION 7: Update variance
        sigma_update = sigma_motor + jnp.abs(normative_update) * sigma_LR

        # EQUATION 6: Update likelihood
        with numpyro.plate("trials", n_trials):
            numpyro.sample(
                "bucket_update",
                dist.Normal(normative_update, sigma_update),
                obs=bucket_update,
            )

        # Store derived quantities for posterior predictive checks
        # (used by 04-02 diagnostics and 04-03 summary outputs)
        numpyro.deterministic("learning_rate", learning_rate)
        numpyro.deterministic("normative_update", normative_update)
        numpyro.deterministic("omega", omega)
        numpyro.deterministic("tau", tau[:-1])  # align length with n_trials


# ---------------------------------------------------------------------------
# MCMC entry point
# ---------------------------------------------------------------------------


def run_mcmc(
    bag_positions: np.ndarray | jax.Array,
    bucket_positions: np.ndarray | jax.Array,
    context: str,
    *,
    num_warmup: int = 2000,
    num_samples: int = 2000,
    num_chains: int = 4,
    target_accept_prob: float = 0.95,
    max_tree_depth: int = 10,
    seed: int = 42,
    progress_bar: bool = False,
) -> MCMC:
    """Run NUTS MCMC for the Reduced Bayesian Observer.

    Phase 4 canonical defaults: 4 chains x 2000 warmup x 2000 draws,
    target_accept_prob=0.95, extra_fields=('diverging',).

    Parameters
    ----------
    bag_positions : array-like
        Bag landing positions, shape ``(n_trials,)``.
    bucket_positions : array-like
        Bucket positions (agent actions), shape ``(n_trials,)``.
    context : str
        ``'changepoint'`` or ``'oddball'``.
    num_warmup : int
        NUTS warmup/adaptation steps (default: 2000).
    num_samples : int
        Posterior samples per chain (default: 2000).
    num_chains : int
        Number of parallel MCMC chains (default: 4).
    target_accept_prob : float
        NUTS target acceptance probability (default: 0.95).
    max_tree_depth : int
        Maximum NUTS tree depth (default: 10).
    seed : int
        Random seed for reproducibility (default: 42).
    progress_bar : bool
        Whether to display a progress bar (default: False).

    Returns
    -------
    MCMC
        NumPyro MCMC object with posterior samples accessible via
        ``mcmc.get_samples()``. Divergence counts accessible via
        ``mcmc.get_extra_fields()['diverging'].sum()``.

    Notes
    -----
    ``extra_fields=('diverging',)`` is passed to ``mcmc.run()`` — this is
    REQUIRED for 04-02 diagnostics module to extract divergence counts via
    ``mcmc.get_extra_fields()['diverging']``.

    Four chains require ``XLA_FLAGS=--xla_force_host_platform_device_count=4``
    set before any JAX import. This is handled in ``nn4psych.bayesian.__init__``
    (see RESEARCH.md Pitfall 3).
    """
    bag_jax = jnp.array(bag_positions)
    bucket_jax = jnp.array(bucket_positions)

    kernel = NUTS(
        reduced_bayesian_model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        bucket_positions=bucket_jax,
        bag_positions=bag_jax,
        context=context,
        extra_fields=("diverging",),
    )

    return mcmc


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def prior_sampler(
    model_fn: object,
    num_samples: int,
    rng_key: jax.Array,
) -> dict[str, jax.Array]:
    """Draw prior samples from a NumPyro model.

    Wraps ``numpyro.infer.Predictive`` with no ``posterior_samples``
    argument (which causes it to sample from the prior).

    Parameters
    ----------
    model_fn : callable
        NumPyro model function (e.g., ``reduced_bayesian_model``).
    num_samples : int
        Number of prior samples to draw.
    rng_key : jax.Array
        JAX random key.

    Returns
    -------
    dict[str, jax.Array]
        Prior samples keyed by parameter name. Each value has shape
        ``(num_samples,)`` for scalar parameters.

    Examples
    --------
    >>> key = jax.random.PRNGKey(0)
    >>> samples = prior_sampler(reduced_bayesian_model, 50, key)
    >>> samples['H'].shape
    (50,)
    """
    predictive = Predictive(model_fn, num_samples=num_samples)
    return predictive(rng_key)


def simulate_synthetic_data(
    params: dict[str, float],
    n_trials: int,
    hazard: float,
    context: str,
    seed: int,
) -> tuple[jax.Array, jax.Array]:
    """Generate a synthetic Nassar-paradigm trial sequence.

    Produces ``(bag_positions, bucket_positions)`` by simulating the
    generative process and running the participant forward model.

    For parameter recovery in 04-02: draw ``params`` from the prior
    using ``prior_sampler``, call this function to generate data, then
    fit MCMC to verify recovery.

    Parameters
    ----------
    params : dict[str, float]
        Model parameters: ``'H'``, ``'LW'``, ``'UU'``, ``'sigma_motor'``,
        ``'sigma_LR'``.
    n_trials : int
        Number of trials to simulate.
    hazard : float
        True hazard rate for the generative process.
    context : str
        ``'changepoint'`` or ``'oddball'``.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    bag_positions : jax.Array
        Simulated bag landing positions, shape ``(n_trials,)``.
    bucket_positions : jax.Array
        Simulated bucket positions (agent behavior), shape ``(n_trials,)``.

    Notes
    -----
    Generative process:
    - **changepoint**: Helicopter at ``heli_0 ~ Uniform(0, 300)``.
      Each trial: ``heli`` jumps with prob ``hazard`` to new
      ``Uniform(0, 300)``; bag ``~ Normal(heli, sigma_N)``.
    - **oddball**: Drift process, ``heli_t = heli_{t-1} + N(0, 7.5)``.
      Occasional bags from ``Uniform(0, BAG_RANGE)`` with prob ``hazard``.

    Bucket positions are generated by running the participant forward
    model: given prediction errors, compute normative updates, then
    add ``Normal(0, sigma_motor + |update| * sigma_LR)`` noise.
    """
    rng = np.random.default_rng(seed)

    # --- Generative model: bag positions ---
    bag_positions_np = np.zeros(n_trials)

    if context == "changepoint":
        heli = rng.uniform(0, BAG_RANGE)
        for t in range(n_trials):
            if rng.random() < hazard:
                heli = rng.uniform(0, BAG_RANGE)
            bag_positions_np[t] = rng.normal(heli, SIGMA_N)
    else:  # oddball
        heli = rng.uniform(0, BAG_RANGE)
        drift_sd = 7.5
        for t in range(n_trials):
            heli = heli + rng.normal(0, drift_sd)
            heli = float(np.clip(heli, 0, BAG_RANGE))
            if rng.random() < hazard:
                bag_positions_np[t] = rng.uniform(0, BAG_RANGE)
            else:
                bag_positions_np[t] = rng.normal(heli, SIGMA_N)

    # Clip bag positions to valid range
    bag_positions_np = np.clip(bag_positions_np, 0, BAG_RANGE)

    # --- Participant forward model: bucket positions ---
    # Bucket starts at the midpoint of the range
    bucket_positions_np = np.zeros(n_trials)
    bucket_positions_np[0] = BAG_RANGE / 2.0

    for t in range(1, n_trials):
        pred_err_t = bag_positions_np[t - 1] - bucket_positions_np[t - 1]
        pred_errors_so_far = jnp.array([pred_err_t])
        lr, norm_upd, _, _ = compute_rbo_forward(
            {k: jnp.array(float(v)) for k, v in params.items()},
            pred_errors_so_far,
            context,
        )
        motor_noise = rng.normal(
            0,
            float(params["sigma_motor"])
            + abs(float(norm_upd[0])) * float(params["sigma_LR"]),
        )
        bucket_positions_np[t] = float(
            np.clip(
                bucket_positions_np[t - 1] + float(norm_upd[0]) + motor_noise,
                0,
                BAG_RANGE,
            )
        )

    return jnp.array(bag_positions_np), jnp.array(bucket_positions_np)


def assert_jax_devices(expected: int = 4) -> None:
    """Assert that at least ``expected`` JAX CPU devices are available.

    Verifies that ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
    was set before any JAX import (required for 4-chain NUTS parallelism).

    Parameters
    ----------
    expected : int
        Minimum expected number of JAX devices (default: 4).

    Raises
    ------
    RuntimeError
        If actual device count is less than ``expected``. Message includes
        expected vs actual per CLAUDE.md error convention.

    Examples
    --------
    >>> assert_jax_devices(expected=4)  # passes if XLA_FLAGS was set
    """
    actual = jax.local_device_count()
    if actual < expected:
        raise RuntimeError(
            f"Expected at least {expected} JAX devices but got {actual}; "
            f"set XLA_FLAGS=--xla_force_host_platform_device_count={expected} "
            f"before importing jax"
        )
