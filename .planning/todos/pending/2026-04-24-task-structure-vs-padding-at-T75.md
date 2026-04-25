---
created: 2026-04-24T00:00
title: Investigate whether T=75 padding beyond task-active window adds noise to latent circuit fit
area: analysis
files:
  - src/nn4psych/analysis/circuit_inference.py
  - src/nn4psych/envs/
  - data/processed/rnn_behav/circuit_data.npz
  - output/circuit_analysis/validation_results.json
---

## Problem

03-02 benchmark produced invariant subspace corr = 0.703 (< 0.85 soft threshold) with
T=75 circuit data. ContextDecisionMaking-v0 at dt=100ms has ~15-30 steps of actual
task structure (fixation + stimulus + delay + decision), leaving ~45-60 steps of
blank/padding per trial.

LatentNet's loss is computed over all T=75 steps uniformly. If the blank periods
contain drift/decay dynamics unrelated to the task computation, the fitted w_rec
may be partly capturing those dynamics — which would degrade the invariant subspace
match (w_rec doesn't align with Q^T W_rec Q because Q was chosen to explain
task-relevant dynamics but W_rec was fitted on full-trial dynamics).

Evidence for this hypothesis:
- Per-trial R² full-space = 0.85 (single trials include noise)
- Trial-avg R² full-space = 0.98 (averaging washes out the noise → huge R² gap)
- Invariant subspace 0.70 (connectivity fit is the most noise-sensitive metric)
- Data previously at T=500 was ~5% task-relevant; now T=75 is ~20-40% task-relevant
  — partial improvement but not a full fix

Evidence against:
- ReLU dynamics during blank periods should decay cleanly; may not distort w_rec much
- Paper (Langdon & Engel) also fits full-trial data and reports R² = 0.96
- May just be that n_latent=16 is too high and over-fits

## Solution

Before (or as part of) 03-03 planning, investigate:

**Option A — Masked loss**: modify LatentNet.fit() to accept a per-timestep mask
(or per-trial task-window metadata) and compute NMSE_y only over task-active steps.
Requires NeuroGym per-trial timing info (should be extractable from trial_info).
Expected: invariant subspace corr should improve if padding is the issue.

**Option B — Shorter T regen**: regenerate circuit_data.npz with tighter max_steps
(e.g., T=30 with delay=0 or a tuned trial-end detection). Simpler but requires
re-running the ensemble.

**Option C — Condition-sliced fitting**: fit LatentNet separately to (task-active
only) vs (padding only) and quantify how much each contributes to w_rec. Diagnostic,
not a fix.

**Option D — n_latent sweep**: before blaming padding, sweep n_latent ∈ {4, 8, 12, 16}
at current data to find the rank knee. If invariant subspace improves at lower rank,
the issue is over-parameterisation not padding.

## Dependencies

- Should probably do **D first** (cheap ensemble sweep, quickly diagnoses over-fit)
- Then **A or B** depending on D's result
- Blocks/shapes 03-03 perturbation analysis design (if Q is a lossy map of connectivity,
  Q-mapped perturbations won't predict behavioral deltas cleanly)

## Priority

High — addresses the core 03-02 soft-fail and directly shapes 03-03 plan structure.
Should be surfaced during `/gsd:discuss-phase 03-03` so the chosen approach is
baked into the 03-03 plan rather than bolted on after.
