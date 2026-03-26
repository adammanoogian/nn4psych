---
created: 2026-03-26T09:00
title: Multi-task latent circuit comparison across tasks and RNNs
area: analysis
files:
  - src/nn4psych/analysis/circuit_inference.py
  - src/nn4psych/models/multitask_actor_critic.py
  - scripts/analysis/analyze_fixed_points.py
---

## Problem

The current v1 pipeline fits a single latent circuit (Q, w_rec, w_in, w_out) on one task (context-DM). This doesn't answer two key scientific questions:

1. **Same RNN, different tasks:** When a MultiTaskActorCritic is trained on PIE + context-DM with shared W_hh, do both tasks use the same computational subspace? Or does the RNN partition its hidden space into task-specific circuits?

2. **Different RNNs, same task:** When two separately trained single-task RNNs solve the same task, do they discover the same latent circuit? This tests whether the circuit is a universal solution or an artifact of training.

Additionally, input-conditioned fixed points in the latent space (FP-02) would bridge the gap between the Sussillo fixed point analysis (full 64-dim) and the latent circuit analysis (8-12 dim), making the dynamical systems structure interpretable.

The ultimate goal: link latent circuit parameters (specific entries in w_rec) to Bayesian model parameters (hazard rate H, learning weight LW from Nassar 2021). This would connect circuit-level mechanisms to computational-level descriptions.

## Solution

**Axis 1 — Same RNN, different tasks:**
- Train MultiTaskActorCritic on PIE + context-DM (shared W_hh)
- Fit Q_PIE and Q_DM separately from same W_hh
- Compare via: principal angles (SVD of Q_A @ Q_B^T), cross-task reconstruction (nmse of task B through Q_A), recurrence correlation (corr of w_rec_A vs w_rec_B)

**Axis 2 — Different RNNs, same task:**
- Train N separate single-task ContinuousActorCritic on context-DM (different seeds)
- Can't directly compare Q (different W_hh spaces)
- Align hidden spaces via Procrustes on matched trial-averaged responses
- Then compare Q in the aligned space

**Fixed points in latent space (FP-02):**
- Find y* where y* = ReLU(w_rec @ y* + w_in @ u) for each task input u
- Analyze n×n Jacobian (much easier than N×N)
- Compare fixed point topology across task conditions

**Depends on:** ADV-02 (multi-task circuit), ADV-05 (MultiTaskActorCritic verification), FP-02 (input-conditioned fixed points)

**Priority:** v2 after Phase 5 completion
