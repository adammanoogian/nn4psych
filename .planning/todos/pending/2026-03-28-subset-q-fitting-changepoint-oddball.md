---
created: 2026-03-28T10:00
title: Fit Q to trial subsets around change points and oddballs
area: analysis
files:
  - src/nn4psych/analysis/circuit_inference.py
  - scripts/analysis/analyze_fixed_points.py
---

## Problem

Q is currently fitted to all trials simultaneously — a single static projection. This averages out any trial-to-trial shifts in which dimensions carry the computation. For the PIE task (change-point / oddball), the key scientific question is whether the computational subspace reorganizes differently after change points vs oddballs.

## Solution

Fit separate LatentNets on trial subsets:

1. Q_pre_cp: trials 1-10 before each change point
2. Q_post_cp: trials 1-10 after each change point
3. Q_pre_ob: trials 1-10 before each oddball
4. Q_post_ob: trials 1-10 after each oddball

Compare subspaces via principal angles: SVD(Q_A @ Q_B^T) -> cos(theta).

If Q_post_cp != Q_pre_cp but Q_post_ob ≈ Q_pre_ob, this would be a circuit-level explanation for why learning rates differ: the circuit itself reconfigures for change points but not oddballs.

Connect to fixed point analysis: do the fixed points in latent space (w_rec fixed points) shift after change points?

Connect to Bayesian model: do Nassar hazard rate / learning weight parameters predict the magnitude of subspace shift?

Depends on: PIE task training with ContinuousActorCritic, LatentNet fitting on PIE data.
Priority: v2 after multi-task comparison (ADV-02).
