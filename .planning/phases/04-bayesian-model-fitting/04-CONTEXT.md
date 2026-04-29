# Phase 4: Bayesian Model Fitting - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the Nassar 2021 reduced Bayesian observer model in NumPyro/JAX, validate via parameter recovery on 50 synthetic datasets (≥0.85 per-parameter recovery), then fit per-subject × per-condition to human schizophrenia data and to a cohort of K=20 RNN training seeds — all with MCMC convergence diagnostics enforced (R-hat ≤ 1.01, ESS ≥ 400; divergences documented but not gating). A second exploratory model (Contextual HMM / CRP-type) is scoped as a *prototype-only* gated plan (04-05) — single human subject + posterior predictive, no full sweep.

Phase 4 produces fits and diagnostics. Phase 5 consumes those fits for the group-level schizophrenia-vs-control-vs-RNN comparison.

</domain>

<decisions>
## Implementation Decisions

### Model formulation

- **Primary model:** Reduced Bayesian observer per Nassar et al. 2010 + 2021 update (tracks expectation μ, prior precision α, change-point probability Ω). This is what 04-01..04-04 implement and validate end-to-end.
- **Exploratory secondary:** Contextual HMM / CRP-type model (Chinese Restaurant Process for context inference). Goes in a **separate gated plan 04-05**, scope = single-subject prototype fit + posterior predictive only. Not run on RNN. Not parameter-recovered. Plan 04-05 is authored only after 04-01..04-04 ship; can be deferred entirely if Reduced Bayesian fits leave no interesting residual structure.
- **Pooling:** Per-subject independent. One MCMC fit per (subject, condition) cell. No partial-pooling / hierarchical structure in Phase 4. (A hierarchical re-fit can be a Phase 5 side analysis if needed.)
- **Priors:** Match the Nassar 2021 paper. Researcher must confirm exact prior specifications from the paper's methods section; if any params are unspecified in the paper, use weakly-informative defaults that reproduce Nassar's MAP fits and document the choice.
- **Data split:** Per-subject × per-condition (CP × OB × subject) — each cell is its own MCMC fit.

### Parameter recovery design (04-02)

- **Synthetic data source:** Sample 50 parameter sets from the model priors (not grid, not Latin Hypercube). Tests prior → posterior coverage directly.
- **Synthetic trial sequences:** Match the Nassar paradigm exactly — generate CP+OB sequences with the same hazards, predictive distributions, and noise as the human task. Recovery validates the actual fitting pipeline used on human data.
- **Recovery threshold:** **Per-parameter, all must pass** (each free parameter independently must satisfy true-vs-recovered correlation r ≥ 0.85 across the 50 synthetic fits). Strict reading of ROADMAP SC-2.
- **Recovery report artifact:** Posterior mean + 95% HDI per parameter + scatter plot (true vs recovered) via ArviZ + matplotlib. Per-parameter r and r². No InferenceData NetCDFs saved for the synthetic recovery cohort (lightweight footprint).

### MCMC configuration

- **Default NUTS settings:** Conservative — 4 chains, 2000 warmup, 2000 draws per chain (8000 post-warmup samples total). target_accept=0.95.
- **Diagnostic gates (must pass):** R-hat ≤ 1.01 AND ESS_bulk ≥ 400.
- **Divergences policy:** **Document but do not gate.** This is a documented relaxation of ROADMAP SC-3/SC-4 wording ("zero divergences"). Planner should update REQUIREMENTS.md / phase Success Criteria to reflect the practical threshold once 04-02 reveals what the model's typical divergence count looks like. Rationale: zero divergences is unrealistic for the Nassar reduced model's posterior geometry; document count and continue rather than force retries.
- **Failure handler (R-hat or ESS gate fail):** Auto-retry once with 2× warmup (4000) and target_accept=0.99. If retry still fails the gates, mark fit as FAILED in the per-fit JSON and continue to the next subject. Both attempts are logged.

### Per-fit artifact storage

- **Trimmed posterior summary JSON only.** Per-fit on-disk: per-parameter posterior mean / median / 95% HDI / R-hat / ESS_bulk / divergence count + retry log. Lightweight (~10 KB per fit).
- No InferenceData NetCDFs saved per-fit. If Phase 5 group analysis needs full posteriors, that's a flag to revisit and re-run a small subset with full saves, or add InferenceData saving as a 04-05/06 extension.

### RNN fit unit (04-04)

- **RNN cohort:** K=20 RNN training seeds, fit independently. K=20 is the standard cognitive-modeling cohort size; not matched to human N.
- **RNN trial sequences:** Replay human trial sequences. For each RNN seed, run the RNN on each human subject's exact trial sequence and fit the model to the resulting RNN behavior. Strongest matched-stimuli comparison for Phase 5.
- **Per-context split:** **Pooled** across modality_context 0 and 1. One fit per RNN seed × subject-replay (mirrors per-subject human handling, which doesn't split by context within a subject's CP-or-OB fit). Note: human fits ARE split by CP/OB condition; the RNN context-pooling decision is about modality_context (the within-task feature), not about the CP/OB experimental conditions.
- **Model parity with humans:** Reduced Bayesian **only**. CHMM exploratory model stays human-only unless Reduced fits reveal a gap that motivates extending to RNN.

### Plan structure (informs gsd-planner)

- **04-01:** Archive PyEM models from prior structure; implement `src/nn4psych/bayesian/` subpackage with the Reduced Bayesian model in NumPyro/JAX.
- **04-02:** MCMC convergence diagnostics (R-hat / ESS / divergence reporting via ArviZ) + parameter recovery on 50 prior-sampled synthetic datasets (Nassar paradigm trials). Produces SC-2 evidence.
- **04-03:** Fit Reduced Bayesian to Nassar 2021 human .mat files, per-subject × per-condition. Wires `scripts/data_pipeline/09_fit_human_data.py`.
- **04-04:** Fit Reduced Bayesian to K=20 RNN seeds, replayed human sequences, pooled across modality_context. Wires `scripts/data_pipeline/10_fit_rnn_data.py`. Requires re-training K=20 seeds of the canonical RNN — flag for planner: this is RNN-side compute (cluster GPU) on top of the fitting compute.
- **04-05 (gated):** CHMM / CRP-type model prototype on a single human subject + posterior predictive. Authored only after 04-01..04-04 ship; can be deferred to v2.

### Claude's Discretion

- Exact Nassar 2021 paper priors (researcher confirms from paper; planner picks weakly-informative defaults if paper doesn't specify).
- Specific CHMM variant for 04-05 (Anderson local-MAP / Gershman-Niv non-parametric / Sanborn particle filter). Researcher surveys, planner picks based on simplicity.
- Trial-sequence noise model details (Gaussian likelihood vs alternatives) — researcher confirms from paper.
- ArviZ plot styling and exact figure layout for the recovery scatter.
- File layout under `src/nn4psych/bayesian/` (single module vs sub-modules per model).
- How to record FAILED fits in the per-fit JSON without breaking Phase 5 ingestion.
- Choice of RNN training seed strategy (fully fresh seeds vs paired-with-data seeds).

</decisions>

<specifics>
## Specific Ideas

- "Match Nassar 2021 paper" was the explicit answer for priors — researcher should fetch the paper's methods section and document the priors used; this is a hard reference, not a vague analogy.
- The CHMM secondary is a CRP-type contextual hidden Markov model in the spirit of Anderson's rational analysis / Gershman & Niv's non-parametric clustering / Collins & Frank's contextual models. The user explicitly wants to "explore this more" → researcher should survey what's available and pick the simplest that captures context inference for changepoint+oddball data.
- Replay-human-sequences for the RNN fits is the canonical approach in matched-stimulus cognitive-modeling comparisons (cf. Daw lab Two-Step replay studies). Researcher should reference standard methodology if planning encounters edge cases.
- The K=20 RNN cohort is standard for "RNN-as-cognitive-cohort" papers; if Phase 5 group comparison is underpowered at K=20, scaling up is straightforward via the planned re-training script.

</specifics>

<deferred>
## Deferred Ideas

- **Hierarchical / partial-pooling re-fit** as a Phase 5 side analysis. Not in Phase 4. Mentioned in initial discussion as "both" option for pooling but explicitly deferred when user chose "per-subject independent" for Phase 4.
- **Full InferenceData NetCDF saves per fit** — deferred unless Phase 5 group analysis needs full posteriors. Currently only trimmed JSON summaries.
- **Matched-N RNN cohort** (K = human N) — deferred. Phase 4 ships K=20; matched-N is a follow-up if Phase 5 power is weak.
- **CHMM on RNN** — deferred. Phase 4 keeps CHMM human-only as a prototype.
- **CHMM full pipeline** (parameter recovery + per-subject fitting + WAIC comparison) — deferred to v2 unless 04-05 prototype reveals strong residual structure that the Reduced Bayesian misses.
- **Updating ROADMAP SC-3/4 wording from "zero divergences" to the practical threshold** — flagged for planner to handle as part of 04-02 (REQUIREMENTS.md update on CIRC-equivalent BAYES SCs).

## Open research questions (gsd-phase-researcher work)

- Inspect Nassar 2021 .mat file structure via `describe_mat_structure()` BEFORE writing 09_fit_human_data.py (long-standing flag from STATE.md). Document subject N, condition labels, trial layout, sequence length.
- Confirm Nassar 2021 paper's exact prior specifications (methods section); document each free parameter and its prior.
- Identify whether the Nassar 2021 reduced model already exists as a reference implementation (PyMC/NumPyro/Stan ports) to validate against.
- Survey CHMM/CRP variants for changepoint+oddball context inference; pick simplest for 04-05 prototype.
- Confirm whether human .mat files include the trial-by-trial generative parameters or only outcomes (affects whether parameter recovery's "true" parameters are inferred from data or sampled fresh).
- Determine memory + wall-time budget for K=20 RNN re-trainings (cluster GPU; coordinate with cluster-side scripts established in Phase 3).

</deferred>

---

*Phase: 04-bayesian-model-fitting*
*Context gathered: 2026-04-29*
