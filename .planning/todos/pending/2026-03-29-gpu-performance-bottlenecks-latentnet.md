---
created: 2026-03-29T10:00
title: Optimize LatentNet GPU performance bottlenecks
area: training
files:
  - src/nn4psych/analysis/latent_net.py
  - src/nn4psych/analysis/circuit_inference.py
---

## Problem

LatentNet ensemble fitting runs at 55s/init on L40S GPU (90 min for 100 inits). Profiling shows the GPU computes for only 0.2% of forward pass time — 99.8% is kernel launch overhead and Python loop overhead. The 55s measured vs 20.8s theoretical = 62% Python overhead.

Current bottlenecks ranked by impact:

1. **Python overhead (62% of wall time):** Python for-loop over 500 epochs, tensor indexing creates Python objects, GC between iterations, attribute assignment for Cayley q.

2. **Kernel launch overhead (38% of GPU time):** Sequential loop of 74 timesteps launches 5 CUDA kernels per step (~10μs launch overhead each). Actual compute per kernel is ~0.02μs. GPU is idle 99.6% of forward pass.

3. **Cayley transform (2%):** torch.linalg.solve on 64x64 matrix uses cuSOLVER which is optimized for large matrices. Called 4000 times per init (every gradient step).

4. **torch.stack in forward (minor):** Copies 74 tensors into contiguous memory at end of forward. Autograd-required so can't pre-allocate.

## Solution

**Priority 1 — torch.compile (expected 3-5x speedup):**
```python
@torch.compile
def forward(self, u):
    ...
```
Fuses the 5 per-timestep operations into one kernel, eliminates Python overhead in the loop. May also help with backward pass fusion. Test compatibility with Cayley transform (matrix inverse may not compile).

**Priority 2 — Reduce Cayley frequency:**
Compute `self.q = self.cayley_transform(self.a)` every 10 gradient steps instead of every step. a changes by tiny amounts each step — q update can be amortized. Expected: saves ~90μs × 9/10 steps = marginal but free.

**Priority 3 — Custom CUDA kernel (10-20x on forward/backward):**
Single CUDA kernel processes all 74 timesteps per batch element. Each CUDA thread handles one of 128 batch elements. Eliminates all kernel launch overhead. Requires writing CUDA C++ and integrating with PyTorch autograd. High effort, high reward — only if running many experiments (hyperparameter sweeps, subset fitting).

**Priority 4 — Reduce T:**
If task computation happens in T=30 window instead of T=75, forward/backward time halves. Requires verifying task structure — which timesteps contain meaningful dynamics.

**Not helpful:** Parallelizing across time (RNN is fundamentally sequential due to ReLU nonlinearity — can't use parallel scan). Larger batch sizes (128 already saturates the small 8×8 matmuls).

Priority: when running many experiments (subset Q fitting, hyperparameter sweeps). Not blocking for v1 single runs.
