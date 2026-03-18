# Domain Pitfalls

**Domain:** Computational neuroscience — RNN-RL + latent circuit inference + NumPyro/JAX Bayesian fitting
**Researched:** 2026-03-18
**Milestone context:** Adding latent circuit inference (Langdon & Engel 2025), NumPyro/JAX Bayesian fitting, and new task environments to existing PyTorch RNN-RL codebase

---

## Critical Pitfalls

Mistakes that cause rewrites or invalidate scientific conclusions.

---

### Pitfall 1: Accepting a Single Latent Circuit Solution Without Multi-Init Comparison

**What goes wrong:** Running latent circuit fitting (Q embedding + w_rec/w_in/w_out optimization) from one random initialization and treating the converged solution as the circuit mechanism.

**Why it happens:** The loss landscape for latent circuit inference has multiple local minima. The Langdon & Engel 2025 paper (verified from PMC) found that only ~10% of random initializations converge to acceptable solutions and that 200-RNN ensembles produced at least three distinct connectivity solution clusters. Researchers assume gradient descent finds the global minimum.

**Consequences:** The inferred circuit mechanism is an artifact of initialization, not the RNN's actual computation. Downstream perturbation analysis and connectivity interpretation are built on a spurious solution. The paper found cross-RNN variance in inferred latent connectivity was four times the within-RNN variance — a single fit will fall somewhere in that distribution unpredictably.

**Warning signs:**
- Fitting produces a circuit that explains activity well (r² > 0.8) but predicts incorrect behavioral outputs (z predictions don't match choices)
- Two fits from different seeds give circuits with very different w_rec structures but similar activity loss
- Correlation between QᵀW_recQ and the inferred w_rec is below 0.85

**Prevention:** Run minimum 100 random initializations, select top-10 by test-set fit quality, then use the ensemble to assess solution consistency. The invariant subspace condition QᵀW_recQ ≈ w_rec must be validated quantitatively (target r ≥ 0.89 as in the paper) after each fit, not assumed.

**Phase:** Latent circuit inference implementation phase — build the multi-init loop and validation check before any scientific interpretation.

---

### Pitfall 2: Ignoring the Invariant Subspace Constraint

**What goes wrong:** The embedding Q is optimized to minimize reconstruction loss, but the core theoretical requirement — that the subspace spanned by Q is invariant under the full RNN's recurrent dynamics — is never explicitly validated.

**Why it happens:** The loss function (Equation 7 in Langdon & Engel) combines activity reconstruction and behavioral output terms. It is possible to minimize this loss while the invariant subspace relationship QᵀW_recQ = w_rec holds only approximately or not at all.

**Consequences:** The inferred latent circuit does not actually describe the RNN's recurrent mechanism. It is a projection artifact. Perturbation experiments built on such a circuit will produce predictions that diverge from actual RNN behavior.

**Warning signs:**
- Post-fit correlation between QᵀW_recQ and inferred w_rec below 0.85
- Activity reconstruction is high but behavioral output from the latent circuit diverges from the RNN's choices on held-out trials
- The latent circuit can't reproduce the RNN's behavior when simulated forward independently

**Prevention:** After each fit, explicitly compute correlation(QᵀW_recQ, w_rec) and report it alongside reconstruction r². Treat fits where this correlation falls below a threshold (suggest 0.85) as failed. This is not reported in FixedPointFinder or generic dimensionality reduction code — it must be added explicitly.

**Phase:** Latent circuit implementation — add as an automated post-fit diagnostic, not a post-hoc check.

---

### Pitfall 3: JAX Pre-allocates GPU Memory Before PyTorch Can Use It

**What goes wrong:** JAX's XLA backend pre-allocates 75% of available GPU VRAM on first import/use. PyTorch then cannot allocate memory for the RNN training pass.

**Why it happens:** JAX's default allocator (`XLA_PYTHON_CLIENT_PREALLOCATE=true`) grabs a large fixed block on first GPU use to avoid repeated allocation overhead. Neither PyTorch nor JAX communicates memory reservations to the other. This has been a documented issue in the JAX repo (issue #15084, issue #19213).

**Consequences:** CUDA OOM errors during PyTorch training after NumPyro MCMC has run, or MCMC sampling fails because PyTorch already holds the GPU. Errors manifest as `XlaRuntimeError: INTERNAL: RET_CHECK failure` or `torch.cuda.OutOfMemoryError`, both of which are opaque.

**Warning signs:**
- Training or MCMC works in isolation but fails when run in the same Python process
- GPU VRAM shows near-full usage before the intended computation begins
- The error appears specifically when importing one framework after the other has run something

**Prevention:**
1. Set environment variables **before any import of JAX** (setting them after has no effect):
   ```python
   import os
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
   ```
2. Initialize a JAX PRNGKey (`jax.random.PRNGKey(0)`) before importing PyTorch on GPU systems — this forces JAX to claim its GPU context first, preventing the `XlaRuntimeError` from the import order conflict.
3. Design the pipeline so PyTorch training (RNN) and NumPyro MCMC run in separate Python processes, communicating via saved files. This is the safest approach for a research codebase where both must run on the same machine.

**Phase:** JAX/NumPyro setup phase — environment configuration must be the first thing in any script that uses both frameworks.

---

### Pitfall 4: JAX Tracing Breaks on Python Control Flow in NumPyro Models

**What goes wrong:** The `compute_normative_model` function in `bayesian/numpyro_models.py` uses a Python `if context == 'changepoint'` branch inside `step_fn`, which is called inside `jax.lax.scan`. This breaks JAX tracing.

**Why it happens:** `jax.lax.scan` traces the function at compile time. Python conditionals on values that are not JAX traceable (string variables like `context`) cause the branch to be fixed at trace time — meaning the model silently uses whichever branch was evaluated on first compilation for all subsequent calls, regardless of the `context` argument passed at runtime.

**Consequences:** Running the model with `context='oddball'` may silently use changepoint equations (or vice versa). The model produces wrong learning rates without any error. This is a silent correctness failure.

**Warning signs:**
- Model produces nearly identical results for 'changepoint' and 'oddball' conditions
- Learning rate distributions don't differ between conditions as expected from Nassar 2021 Figure 6
- No error thrown, but fitted parameters are implausible

**Prevention:** Replace Python `if/else` on `context` with JAX-compatible conditionals inside scan:
```python
lr_t = jnp.where(context_flag == 0,
                  omega_t + tau_prev - (omega_t * tau_prev),   # changepoint
                  tau_prev - (omega_t * tau_prev))              # oddball
```
where `context_flag` is an integer JAX array passed through the model, not a Python string. The existing `bayesian/numpyro_models.py` has this bug in the `step_fn` definition at line 149.

**Phase:** NumPyro/JAX setup phase — fix before any MCMC fitting runs.

---

### Pitfall 5: MCMC Divergences Treated as Acceptable Noise

**What goes wrong:** NumPyro/NUTS reports divergences during sampling but they are ignored because the trace plots look visually smooth and R-hat is below 1.05.

**Why it happens:** Divergences are rare enough to seem like noise. The older R-hat ≤ 1.10 threshold (still found in many tutorials) masks problems that the stricter R-hat ≤ 1.01 standard would catch. The Bayesian cognitive modeling troubleshooting literature (PMC10522800) is explicit: "Divergences must never be disregarded in cognitive models."

**Consequences:** The posterior is biased. Inferred parameters H (hazard rate), LW (likelihood weighting), and UU (uncertainty underestimation) are pulled away from their true values by high-curvature regions the sampler couldn't explore. Patient vs. control parameter comparisons will reflect sampler pathology, not biological differences.

**Warning signs:**
- `mcmc.print_summary()` reports any divergences (even 1-2 per chain)
- R-hat > 1.01 for any parameter
- Effective Sample Size (ESS) below 400 (100 × 4 chains)
- Bivariate plots show divergent transitions concentrated in parameter corners

**Prevention:**
1. Enforce zero divergence tolerance — any divergence triggers reparameterization, not just resampling
2. For H, LW, UU (all Beta-distributed): the current `Beta(2, 2)` priors are reasonable, but verify with prior predictive checks that the implied behavioral distributions are not implausible
3. If divergences concentrate near boundary values (H → 0 or H → 1), use logit-Normal priors instead of Beta for better HMC geometry
4. Use `numpyro.diagnostics.hpdi` and ArviZ `az.plot_pair(idata, divergences=True)` to visualize divergence locations

**Phase:** NumPyro model validation phase — parameter recovery simulation must be run before fitting to real data.

---

## Moderate Pitfalls

Mistakes that produce incorrect results or significant technical debt.

---

### Pitfall 6: Fixed Point Search Uses Only Zero-Input Dynamics

**What goes wrong:** The existing `FixedPointAnalyzer.find_fixed_points()` in `scripts/analysis/analyze_fixed_points.py` searches for fixed points of `h' = tanh(W_hh @ h)` — the autonomous dynamics with no input. RNN fixed points are input-dependent: the actual fixed points during task performance occur under constant task-specific inputs.

**Why it happens:** Fixed-point analysis tutorials typically show the input-free case. The `rnn_dynamics` method in the current codebase explicitly ignores `W_ih` and input biases.

**Consequences:** The fixed points found have no behavioral correlate. They may not exist anywhere in the RNN's actual trajectory during the task. The line attractor analysis and stability results are computed for a dynamical system the RNN never actually inhabits.

**Warning signs:**
- Fixed points are found but simulated trajectories (from task-conditioned initial states) don't pass through them
- The task-conditioned hidden states, when PCA-projected, show no proximity to the reported fixed points
- Fixed points are the same regardless of task condition (CP vs OB)

**Prevention:** Fix-point search must be done under representative input patterns. For the helicopter task:
- Compute the mean/representative input encoding for each task condition
- Find fixed points of `h' = tanh(W_hh @ h + W_ih @ u + b)` where `u` is the held constant at its representative value
- The Golub & Sussillo FixedPointFinder toolbox (https://github.com/mattgolub/fixed-point-finder) handles this correctly via its `inputs` argument
- Consider adapting to PyTorch using the `tripdancer0916/pytorch-fixed-point-analysis` approach

**Phase:** Fixed point analysis implementation — the current implementation needs revision before results are interpreted.

---

### Pitfall 7: Rank Selection for Latent Circuit Is Arbitrary

**What goes wrong:** The dimensionality `n` of the latent circuit (size of Q matrix) is chosen by percentage-variance-explained PCA cutoff alone, without validating that the chosen subspace is mechanistically sufficient.

**Why it happens:** PCA variance explained is the default justification. Langdon & Engel 2025 chose n=8 based on task structure (2 contexts × 2 sensory × 2 choice), not variance explained alone.

**Consequences:** Too low a rank misses circuit components and produces a circuit that cannot implement the task (behavioral output z ≠ actual choices). Too high a rank includes noise dimensions, destabilizes optimization, and produces non-unique solutions.

**Warning signs:**
- The latent circuit at chosen rank cannot match behavioral output loss below a reasonable threshold even after 100 random inits
- Increasing rank by 2 substantially improves behavioral output fit
- The PCA-selected rank does not align with obvious task structure dimensions

**Prevention:** Use task structure as the primary guide for rank selection (count distinct task variable dimensions the circuit must encode). Validate that the latent circuit at the chosen rank can achieve near-zero behavioral output loss. Report sensitivity of results to rank ±2.

**Phase:** Latent circuit implementation — rank selection is a modeling decision, not an analysis output.

---

### Pitfall 8: Hierarchical Parameter Recovery Not Validated Before Fitting Clinical Data

**What goes wrong:** The NumPyro model is fit to schizophrenia patient and control data without first verifying that the model can recover known parameters from simulated data.

**Why it happens:** Parameter recovery simulation is considered optional validation. Researchers jump to real data once the model runs without crashing.

**Consequences:** If the model cannot recover H, LW, UU from synthetic data with known ground truth, any differences found between patients and controls are uninterpretable — they may reflect model non-identifiability rather than biological differences.

**Warning signs:**
- Correlations between generating parameters and recovered parameters below 0.7 in simulation
- Systematic bias: recovered H consistently overestimates or underestimates the generating value
- Posterior distributions on real data are nearly identical to the prior (the data isn't informing inference)

**Prevention:**
1. Before fitting any real participant, generate 50 synthetic datasets with known H/LW/UU values spanning the plausible range
2. Fit the model to each synthetic dataset and compute correlation between generating and recovered values
3. Only proceed to real data if correlations are above 0.85 for all key parameters
4. The existing `validation/test_parameter_recovery.py` stub needs to be implemented — it exists but likely only tests imports

**Phase:** NumPyro model validation — must precede Nassar .mat file fitting.

---

### Pitfall 9: extract_behavior Relies on Private Environment State and Will Break

**What goes wrong:** `src/nn4psych/analysis/behavior.py:extract_behavior` calls `env._reset_state()` (private method, line 66) and `env.get_state_history()` (line 88), both of which are internal PIE_CP_OB_v2 implementation details.

**Why it happens:** The existing codebase was built incrementally. The known tech debt item "fragile extract_behavior relying on private env methods" is documented in PROJECT.md.

**Consequences:** When the PIE environment is refactored or when new NeuroGym task environments are used, `extract_behavior` silently breaks or returns wrong state arrays. The multi-task evaluation loop in `train_multitask.py` already uses different state access patterns (lines 480-488), indicating the two code paths are already diverged.

**Warning signs:**
- `extract_behavior` produces different state arrays than the training loop state collection for the same model on the same task
- AttributeError when calling extract_behavior on NeuroGym-wrapped environments
- Learning rates computed from extract_behavior output differ from learning rates computed inline during training

**Prevention:** Refactor extract_behavior to use only the public environment API: `env.reset()`, `env.step()`, `env.normalize_states()`. Record state during the rollout rather than calling `get_state_history()` after. Apply this refactoring before extending to NeuroGym tasks where the private API will certainly not exist.

**Phase:** Infrastructure cleanup phase — fix before implementing latent circuit extraction, which will depend on reliable behavior extraction.

---

### Pitfall 10: Multi-Task Training Gradient Interference Not Detected

**What goes wrong:** The multi-task training loop in `train_multitask.py` accumulates gradients from both tasks before each optimizer step (especially in `trial` and `block` interleave modes). Task-specific gradients may interfere, causing one task to improve while the other regresses — a phenomenon not visible in per-epoch mean return plots.

**Why it happens:** Multi-task RL training with a shared backbone is known to suffer from gradient interference when task objectives conflict. The existing `tests/test_multitask_actor_critic.py` and `tests/test_configs.py` are listed as untracked (untested) in git status.

**Consequences:** A trained model shows degraded performance on one task without the training diagnostics revealing it. The model checkpoint selected at epoch 99 may not generalize to both tasks. Downstream latent circuit inference on a poorly-trained multi-task model produces uninterpretable circuits.

**Warning signs:**
- Loss for one task trends upward while the other trends downward during later training epochs
- Per-task mean distances plateau at different values during test phase (hidden helicopter)
- Gradient norms for different task heads are orders of magnitude apart

**Prevention:**
1. Log per-task performance and loss at every epoch, not just every 10 epochs
2. Monitor gradient norms per task head (actor_A, actor_B) vs shared backbone during training
3. Consider gradient surgery or PCGrad as intervention if interference is detected
4. The `eval_frequency` in `MultiTaskConfig` (default 10) should be reduced to 1 during debugging

**Phase:** Training validation phase — run diagnostic training with `eval_frequency=1` before committing to a full hyperparameter sweep.

---

## Minor Pitfalls

Mistakes that cause delays or produce misleading figures.

---

### Pitfall 11: PCA for Latent Circuit Visualization Is Not the Same PCA Used for Q Initialization

**What goes wrong:** Hidden states are visualized in PCA space for figures, and separately the Q matrix is initialized from PCA components. If these are computed on different data splits or different numbers of trials, the visualization and the circuit are not aligned.

**Prevention:** Compute a single PCA on the full hidden state trajectory. Use its first-n components to initialize Q. Reuse the same PCA transform for all visualizations. Store the PCA object with the trained latent circuit.

**Phase:** Latent circuit implementation.

---

### Pitfall 12: Comparing Patient vs. Control Model Fits Without Testing for Absolute Fit Quality

**What goes wrong:** Group differences in H, LW, UU are reported even though the model fit quality (posterior predictive check against observed bucket updates) is poor for one or both groups.

**Prevention:** Before comparing parameter estimates across groups, verify posterior predictive r² > 0.7 for both groups separately. A model that fits controls well but patients poorly cannot produce valid group comparisons — the posterior for the poorly-fit group reflects prior information more than data.

**Phase:** Bayesian model fitting and comparison phase.

---

### Pitfall 13: Hardcoded input_dim=9 in Analysis Scripts Will Silently Load Wrong Model

**What goes wrong:** `scripts/analysis/analyze_fixed_points.py` line 306 hard-codes `ActorCritic(input_dim=9, ...)` regardless of the model file's actual input dimension. Multi-task models or models trained on NeuroGym tasks have different input dimensions.

**Prevention:** Infer input_dim from the checkpoint: `weight_ih = checkpoint['rnn.weight_ih_l0']; input_dim = weight_ih.shape[1]`. This is already done for hidden_dim (line 303-304) but not for input_dim.

**Phase:** Infrastructure cleanup — fix before fixed point analysis is run on multi-task or NeuroGym-trained models.

---

### Pitfall 14: MAT File Data Structure Is Hardcoded and Undocumented

**What goes wrong:** The `.mat` file loading in `scripts/data_pipeline/06_compare_with_human_data.py` uses hardcoded indexing: `data[0][0][cp][:sz_pat]` with a hardcoded `sz_pat=102`. If the Nassar data files have different structure between slidingWindowFits_model and slidingWindowFits_subjects, or if a different version of the .mat files is used, this fails silently with shape mismatches.

**Prevention:** Add explicit shape assertions and a `describe_mat_structure()` utility that prints the full nested structure of any loaded .mat file. Run this once on the actual files and document the structure in a comment at the top of the loading function.

**Phase:** Data pipeline validation — run this before any statistical comparison of model vs. human data.

---

### Pitfall 15: NumPyro `jax.lax.scan` Cannot Accept Python String Arguments as Loop Variables

**What goes wrong:** The existing `compute_normative_model` passes `context` as a plain Python string into a function that is traced by `jax.lax.scan`. Beyond the conditional branching issue (Pitfall 4), any attempt to use `context` as a dynamic value (e.g., per-trial context switching) will fail because JAX cannot trace Python strings.

**Prevention:** Represent all model variants as integer flags or use separate model functions for changepoint vs. oddball. Never pass Python strings as arguments to functions called inside `jax.lax.scan` or `jax.vmap`.

**Phase:** NumPyro model implementation — design decision to make before parameterizing per-subject fits.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|---|---|---|
| JAX/NumPyro setup | GPU memory conflict with PyTorch (Pitfall 3) | Set XLA env vars first, use separate processes |
| JAX/NumPyro setup | Python string in jax.lax.scan (Pitfalls 4, 15) | Replace with integer flags before any fitting |
| Latent circuit implementation | Multi-init required (Pitfall 1) | Build 100-init loop before any scientific claim |
| Latent circuit implementation | Invariant subspace not validated (Pitfall 2) | Add QᵀWrecQ correlation check as automated diagnostic |
| Latent circuit implementation | Rank selection arbitrary (Pitfall 7) | Use task structure to justify rank, validate behaviorally |
| Fixed point analysis | Input-free dynamics used (Pitfall 6) | Use input-conditioned fixed points from the start |
| Fixed point analysis | Hardcoded input_dim (Pitfall 13) | Infer from checkpoint before any analysis |
| NumPyro model validation | Parameter recovery skipped (Pitfall 8) | Mandatory simulation study before real data |
| NumPyro model validation | MCMC divergences ignored (Pitfall 5) | Zero-divergence policy, R-hat ≤ 1.01 |
| Behavior extraction | extract_behavior fragility (Pitfall 9) | Refactor to public API before NeuroGym use |
| Multi-task training | Gradient interference undetected (Pitfall 10) | Per-task logging at every epoch |
| Clinical data comparison | Fit quality not checked per group (Pitfall 12) | Posterior predictive check before group comparison |
| Clinical data comparison | MAT file structure undocumented (Pitfall 14) | Add structure assertions and documentation |
| Visualization | PCA spaces misaligned (Pitfall 11) | Single PCA object for initialization and visualization |

---

## Sources

- Langdon & Engel 2025, "Latent circuit inference from heterogeneous neural responses during cognitive tasks," *Nature Neuroscience* — https://pmc.ncbi.nlm.nih.gov/articles/PMC11893458/ (HIGH confidence — primary reference paper, full text reviewed)
- "Troubleshooting Bayesian cognitive models: A tutorial with matstanlib," PMC10522800 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10522800/ (HIGH confidence — methodology paper reviewed in full)
- NumPyro Getting Started documentation — https://num.pyro.ai/en/stable/getting_started.html (HIGH confidence — official docs)
- JAX GPU Memory Allocation documentation — https://docs.jax.dev/en/latest/gpu_memory_allocation.html (HIGH confidence — official docs)
- JAX RNG clash with PyTorch on GPU, jax-ml/jax issue #15084 — https://github.com/jax-ml/jax/issues/15084 (MEDIUM confidence — upstream issue tracker)
- NumPyro GPU memory issue #539 — https://github.com/pyro-ppl/numpyro/issues/539 (MEDIUM confidence — upstream issue tracker)
- Pals et al. 2024, "Inferring stochastic low-rank recurrent neural networks from neural data," NeurIPS 2024 — https://arxiv.org/abs/2406.16749 (MEDIUM confidence — preprint reviewed)
- Frontiers review on computational models in psychosis — https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2022.814111/full (MEDIUM confidence — domain review)
- Golub & Sussillo FixedPointFinder — https://github.com/mattgolub/fixed-point-finder (MEDIUM confidence — tool documentation reviewed, but tests currently broken)
- Project codebase direct inspection: `bayesian/numpyro_models.py`, `scripts/analysis/analyze_fixed_points.py`, `src/nn4psych/analysis/behavior.py`, `scripts/training/train_multitask.py` (HIGH confidence — direct code review)
