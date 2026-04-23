---
created: 2026-04-23T00:00
title: Plan cluster GPU adoption for latent circuit fitting
area: planning
files:
  - .planning/ROADMAP.md
  - .planning/phases/03-latent-circuit-inference/03-02-PLAN.md
  - .planning/todos/pending/2026-03-29-gpu-performance-bottlenecks-latentnet.md
  - cluster/logs/circuit_ensemble_54148075.out
  - src/nn4psych/analysis/latent_net.py
---

## Problem

The cluster GPU workflow (SLURM + L40S) already exists and was used for the 03-02
ensemble run, but it is not formally in the roadmap as a phase. Right now it lives
as one-off scripts in `cluster/` with log artifacts that never flow back to
`output/circuit_analysis/`. The recent run showed the GPU is being wasted:
29.4% avg util, 618 MiB of 44.4 GB memory used, 62% Python overhead, ~90 min for
100 inits of a tiny (n=8, N=64) model. For a single ensemble we could have run
this locally in similar wall time.

The question is: **when does adopting the cluster workflow pay off**, and what
needs to be in place (code + math) for that moment? Without a clear trigger, we
keep burning SLURM time on problems the cluster isn't helping with, and keep
re-deriving artifact paths by hand.

Separately: the math for the breakeven isn't written down anywhere. Per-init
wall time on CPU vs GPU at different (n, N, T, batch_size, n_inits) is unknown,
and the soft-fail invariant-subspace result at n=8 suggests we may need to
sweep (n, l_y, T, training quality) — at which point the scaling of total work
is what determines whether the cluster is necessary.

## Solution

Scope for a future phase (likely post-03-03, before 04 or as a 3.5 inserted phase):

1. **Breakeven math.** Derive per-init wall time model as a function of
   (n, N, T, batch_size, n_trials, epochs) for both CPU and GPU on this machine.
   Identify the problem-size threshold where GPU actually beats CPU *after*
   the Python-overhead fixes in `gpu-performance-bottlenecks-latentnet`.
   Include SLURM queue time and artifact round-trip as overhead terms.

2. **Adoption trigger.** Write down concrete conditions that flip local → cluster:
   number of ensembles needed (sweeps), latent rank, or whether
   subset-Q / multi-task work is on the immediate critical path.

3. **Integrate GPU into a phase.** Wire the cluster workflow so that:
   - Pipeline outputs (best model, validation JSON, diagnostics) land in
     `output/circuit_analysis/` not just `cluster/logs/`.
   - Local and cluster runs use the same entrypoint (e.g. `08_infer_latent_circuits.py`
     with a `--cluster` flag or matching SLURM wrapper) so results are reproducible.
   - SUMMARY.md can cite the cluster run as the canonical 03-02 artifact.

4. **Prerequisite check.** The GPU-bottlenecks todo should probably be done
   *before* this, because without torch.compile + reduced Cayley the cluster
   isn't actually faster. That todo is the unblocker; this todo is the
   adoption decision.

Priority: schedule after 03-02 is locked in. Relevant triggers are sweep work
(rank/l_y search for invariant-subspace fix) and the v2 subset-Q fitting
(dozens of ensembles) — either one makes this a blocker for the next
experimental cycle.

TBD: exact phase number (likely 3.5 inserted, or folded into Phase 4 setup).
