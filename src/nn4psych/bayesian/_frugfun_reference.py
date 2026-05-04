"""Faithful NumPy port of frugFun5.m and frugFun5_uniformOddballs.m.

This module is a line-by-line translation of the Nassar lab MATLAB code for
cross-validation against the JAX/NumPyro ``compute_rbo_forward`` implementation.
It is NOT used in production fits — the underscore prefix marks it as internal
validation code.

MATLAB sources (in data/raw/nassar2021/Brain2021Code/functionCodes/):
  frugFun5.m                -- CP (changepoint) version
  frugFun5_uniformOddballs.m -- OB (oddball) version

Key design differences between CP and OB variants:
  - CP: alpha = yInt + pCha*(1-yInt)   [increases with pCha]
  - OB: alpha = yInt + pCha*(-yInt)    [decreases with pCha, = yInt*(1-pCha)]
  - CP R-update: 1st term = sigmaE^2/1, residual = (B+yInt*Delta - data)^2
  - OB R-update: 1st term = sigmaE^2/R, residual = (yInt*Delta)^2

References
----------
Nassar, M. R., et al. (2021). Dissociable forms of uncertainty-driven
    representational change across the human cortex. *Nature Neuroscience*.
    PMC8041039.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def frugfun5_reference(
    data: np.ndarray,
    Hazard: float,
    noise: float,
    drift: float = 0.0,
    likeWeight: float = 1.0,
    trueRun: int = 0,
    initGuess: float = 150.0,
    inRun: float = 1.0,
) -> dict[str, np.ndarray]:
    """Faithful NumPy port of frugFun5.m (CP changepoint version).

    Line-by-line translation of the MATLAB ``frugFun5`` function for
    cross-validation only. Pure NumPy + scipy; not JAX-traceable.

    Parameters
    ----------
    data : np.ndarray
        Bag positions, shape ``(n,)``. Must be in [0, 300] for the task.
    Hazard : float
        Hazard rate H in [0, 1].
    noise : float
        Observation noise SD (sigmaE). Use ``SIGMA_N = 20.0`` for the
        Nassar 2021 helicopter task.
    drift : float
        Drift SD for the helicopter position. 0 for standard CP/OB task.
    likeWeight : float
        Likelihood-weighting exponent LW in [0, 1]. 1 = near-optimal;
        0 = pure delta-rule.
    trueRun : int
        0 = second-moment R-update (default); 1 = run-length mean update.
    initGuess : float
        Initial belief B(0). Default 150.0 (midpoint of [0, 300]).
    inRun : float
        Initial run-length R(0). Default 1.0.

    Returns
    -------
    dict
        Keys:

        ``'B'``
            Belief trajectory, shape ``(n+1,)``. B[0] is the initial
            belief; B[i+1] is the updated belief after observing data[i].
        ``'totSig'``
            Total predictive SD per trial, shape ``(n,)``.
        ``'R'``
            Run-length estimates, shape ``(n+1,)``. R[0] = inRun;
            R[i+1] is updated after trial i.
        ``'pCha'``
            Changepoint probability per trial, shape ``(n,)``.
        ``'sigmaU'``
            Uncertainty SD per trial (drift-adjusted), shape ``(n,)``.
        ``'alpha'``
            Learning rate per trial, shape ``(n,)``.

    Notes
    -----
    MATLAB correspondence:

    - Line 73:  ``d = ones(300,1)./300``  — uniform component constant 1/300.
    - Line 81:  ``sigmaU = sqrt((sigmaE/sqrt(R))^2 + drift^2)``
    - Line 82:  ``R[i] = noise^2 / sigmaU[i]^2``  (drift-aware recompute)
    - Line 84:  ``totSig = sqrt(sigmaE^2 + sigmaU^2)``
    - Lines 86-90: truncated-normal pI (truncated to [0, 300] around B[i])
    - Line 99:  ``changeRatio = exp(LW*log(changLike/pI) + log(H/(1-H)))``
    - Lines 110-115: belief update via yInt = 1/(R+1), alpha = yInt + pCha*(1-yInt)
    - Lines 117-127: R-update (second-moment or true run-length)
    """
    sigmaE = float(noise)
    n = len(data)

    # Clamp Hazard to [0, 1] as in MATLAB
    Hazard = float(np.clip(Hazard, 0.0, 1.0))

    # Uniform component: d = ones(300)/300 (MATLAB line 73)
    # changLike = d(data(i)) for data in [1,300], else d(1).
    # Since d is uniform at 1/300, changLike is always 1/300.
    UNIFORM_LIKE = 1.0 / 300.0

    # Allocate output arrays
    B = np.full(n + 1, np.nan)
    R = np.full(n + 1, np.nan)
    sigmaU = np.full(n, np.nan)
    totSig = np.full(n, np.nan)
    pCha = np.full(n, np.nan)
    alpha = np.full(n, np.nan)

    B[0] = float(initGuess)
    R[0] = float(inRun)

    for i in range(n):
        # --- Part 1: expected distribution (MATLAB lines 81-84) ---
        # sigmaU: uncertainty SD incorporating drift
        sigmaU[i] = np.sqrt((sigmaE / np.sqrt(R[i])) ** 2 + drift ** 2)
        # R recomputed with drift correction (MATLAB line 82)
        R[i] = noise ** 2 / sigmaU[i] ** 2
        # Total predictive SD (MATLAB line 84)
        totSig[i] = np.sqrt(sigmaE ** 2 + sigmaU[i] ** 2)

        # --- Part 2: changepoint probability (MATLAB lines 86-107) ---
        # Truncated-normal pI: bag data is in [0, 300]; normalize by
        # the fraction of the Gaussian mass inside [0, 300].
        pI = norm.pdf(data[i], B[i], totSig[i])
        normalize = (
            norm.cdf(300.0, B[i], totSig[i]) - norm.cdf(0.0, B[i], totSig[i])
        )
        pI = pI / normalize  # truncation correction (MATLAB line 90)

        # Uniform component (MATLAB lines 93-97)
        changLike = UNIFORM_LIKE  # d is uniform 1/300 for all valid data

        # Log-space change ratio (MATLAB line 99)
        change_ratio = np.exp(
            likeWeight * np.log(changLike / pI)
            + np.log(Hazard / (1.0 - Hazard))
        )

        if not np.isfinite(change_ratio):
            # MATLAB lines 101-103: pCha = 1, pNoCha = 0
            pCha[i] = 1.0
            pNoCha = 0.0
        else:
            pCha[i] = change_ratio / (change_ratio + 1.0)
            pNoCha = 1.0 - pCha[i]

        # --- Part 3: belief update (MATLAB lines 110-115) ---
        # CP version: slope = 1 - yInt  => alpha = yInt + pCha*(1-yInt)
        yInt = 1.0 / (R[i] + 1.0)
        slope = 1.0 - yInt  # CP: slope is positive
        alpha[i] = yInt + pCha[i] * slope
        Delta = data[i] - B[i]
        B[i + 1] = B[i] + alpha[i] * Delta

        # --- Part 4: R update (MATLAB lines 117-127) ---
        if trueRun == 1:
            # True run-length mean update (MATLAB line 118)
            R[i + 1] = (R[i] + 1.0) * pNoCha + pCha[i]
        else:
            # Second-moment matching (MATLAB lines 120-121)
            # Note: residual is (B[i] + yInt*Delta) - data[i]
            residual = (B[i] + yInt * Delta) - data[i]
            ss = (
                pCha[i] * (sigmaE ** 2 / 1.0)
                + pNoCha * (sigmaE ** 2 / (R[i] + 1.0))
                + pCha[i] * pNoCha * residual ** 2
            )
            R[i + 1] = sigmaE ** 2 / ss

    return {
        "B": B,
        "totSig": totSig,
        "R": R,
        "pCha": pCha,
        "sigmaU": sigmaU,
        "alpha": alpha,
    }


def frugfun5_oddball_reference(
    data: np.ndarray,
    Hazard: float,
    noise: float,
    drift: float = 0.0,
    likeWeight: float = 1.0,
    initGuess: float = 150.0,
    inRun: float = 1.0,
) -> dict[str, np.ndarray]:
    """Faithful NumPy port of frugFun5_uniformOddballs.m (OB variant).

    Parameters and return value are identical to ``frugfun5_reference``
    (no ``trueRun`` argument; the oddball variant always uses the
    second-moment R-update).

    Key differences from the CP version:

    - **alpha**: ``yInt + pCha*(-yInt) = yInt*(1-pCha)``
      (slope = -yInt; learning rate *decreases* with pCha).
    - **R-update**: 1st term uses ``sigmaE^2/R[i]`` (not ``sigmaE^2/1``);
      residual squared term is ``(yInt*Delta)^2`` (not
      ``(B+yInt*Delta - data)^2``).

    Notes
    -----
    MATLAB source: ``frugFun5_uniformOddballs.m``.
    Line 108: ``slope = (-yInt)`` vs CP line 112 ``slope = (1-yInt)``.
    Line 120-122: R-update first term and residual differ from CP.
    """
    sigmaE = float(noise)
    n = len(data)

    Hazard = float(np.clip(Hazard, 0.0, 1.0))
    UNIFORM_LIKE = 1.0 / 300.0

    B = np.full(n + 1, np.nan)
    R = np.full(n + 1, np.nan)
    sigmaU = np.full(n, np.nan)
    totSig = np.full(n, np.nan)
    pCha = np.full(n, np.nan)
    alpha = np.full(n, np.nan)

    B[0] = float(initGuess)
    R[0] = float(inRun)

    for i in range(n):
        # Part 1: expected distribution (identical to CP)
        sigmaU[i] = np.sqrt((sigmaE / np.sqrt(R[i])) ** 2 + drift ** 2)
        R[i] = noise ** 2 / sigmaU[i] ** 2
        totSig[i] = np.sqrt(sigmaE ** 2 + sigmaU[i] ** 2)

        # Part 2: changepoint probability (identical to CP)
        pI = norm.pdf(data[i], B[i], totSig[i])
        normalize = (
            norm.cdf(300.0, B[i], totSig[i]) - norm.cdf(0.0, B[i], totSig[i])
        )
        pI = pI / normalize

        changLike = UNIFORM_LIKE

        change_ratio = np.exp(
            likeWeight * np.log(changLike / pI)
            + np.log(Hazard / (1.0 - Hazard))
        )

        if not np.isfinite(change_ratio):
            pCha[i] = 1.0
            pNoCha = 0.0
        else:
            pCha[i] = change_ratio / (change_ratio + 1.0)
            pNoCha = 1.0 - pCha[i]

        # Part 3: belief update — OB variant
        # slope = -yInt (MATLAB line 108), so alpha = yInt*(1-pCha)
        yInt = 1.0 / (R[i] + 1.0)
        slope = -yInt  # OB: negative slope — pCha pulls alpha DOWN
        alpha[i] = yInt + pCha[i] * slope
        Delta = data[i] - B[i]
        B[i + 1] = B[i] + alpha[i] * Delta

        # Part 4: R update — OB second-moment form (MATLAB lines 120-127)
        # 1st term: sigmaE^2/R[i] (not /1 as in CP)
        # residual: (B[i] + yInt*Delta) - B[i] = yInt*Delta
        residual = yInt * Delta  # (MATLAB line 122: B(i)+yInt*Delta - B(i))
        ss = (
            pCha[i] * (sigmaE ** 2 / R[i])
            + pNoCha * (sigmaE ** 2 / (R[i] + 1.0))
            + pCha[i] * pNoCha * residual ** 2
        )
        R[i + 1] = sigmaE ** 2 / ss

    return {
        "B": B,
        "totSig": totSig,
        "R": R,
        "pCha": pCha,
        "sigmaU": sigmaU,
        "alpha": alpha,
    }
