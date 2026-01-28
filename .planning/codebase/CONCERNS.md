# Codebase Concerns

**Analysis Date:** 2026-01-28

## Tech Debt

**Hardcoded dimensions in analysis functions:**
- Issue: Multiple analysis functions hardcode model dimensions (input_dim=9, hidden_dim=64, action_dim=3) instead of accepting parameterized configs
- Files: `src/nn4psych/analysis/behavior.py` (lines 202, 207-208 in `batch_extract_behavior`)
- Impact: Cannot easily analyze models with different architectures; requires duplicating code or manually editing parameters
- Fix approach: Refactor `batch_extract_behavior()` and `get_area()` to accept `ModelConfig` objects or accept dimension parameters; use configuration system consistently

**Duplicated hyperparameter sweep values:**
- Issue: Hyperparameter sweep ranges defined in multiple places (config.py, configs.py, same values repeated)
- Files: `config.py` (lines 143-146), `src/nn4psych/training/configs.py` (lines 277-280)
- Impact: Maintaining consistency across files is error-prone; changes in one location may not propagate
- Fix approach: Define sweep values in single canonical location (prefer `src/nn4psych/training/configs.py`), import in `config.py`

**Legacy archive directory still in use:**
- Issue: `archive/` directory contains 8 duplicate implementations of ActorCritic and multiple deprecated scripts; codebase not fully cleaned after consolidation
- Files: All of `archive/` directory
- Impact: Developers may accidentally import or reference old code; confusion about canonical implementation; wasted storage
- Fix approach: Move legacy code to proper deprecation note or separate branch; leave minimal reference implementations with clear warnings

**Unused noise parameter in ActorCritic:**
- Issue: `noise` parameter accepted in `ActorCritic.__init__()` (line 65) but never used throughout the class
- Files: `src/nn4psych/models/actor_critic.py` (lines 32-34, 74)
- Impact: Confusing API; suggests feature that doesn't work; technical debt in parameter space
- Fix approach: Either implement noise injection in forward pass or remove parameter entirely

## Known Bugs

**Environment reset inconsistency in batch_extract_behavior:**
- Symptoms: `batch_extract_behavior()` loads models but always creates fresh environment instances without seeding
- Files: `src/nn4psych/analysis/behavior.py` (lines 207-208)
- Trigger: Call `batch_extract_behavior()` multiple times on same model set; results may not be reproducible
- Workaround: Manually set random seeds before calling function; consider wrapping in seed context manager

**Assumption about sequence length in forward pass:**
- Symptoms: Comment on line 148 of `actor_critic.py` says "assuming seq_len=1" but code doesn't validate this
- Files: `src/nn4psych/models/actor_critic.py` (lines 147-149)
- Trigger: Pass sequence with seq_len > 1; only last timestep's output is used (squeeze(1))
- Workaround: Always ensure input has seq_len=1; validate input shape before forward pass

**MultiTaskActorCritic padding to max_obs_dim unused:**
- Symptoms: Lines 177-179 calculate max_obs_dim and max_action_dim but they're not used by MultiTaskActorCritic (they're only used in PaddedMultiTaskActorCritic)
- Files: `src/nn4psych/models/multitask_actor_critic.py` (lines 177-179)
- Trigger: Code confusion about when padding is actually applied
- Impact: Dead code reduces clarity
- Fix approach: Move these calculations to PaddedMultiTaskActorCritic only

## Security Considerations

**No input validation on dimension parameters:**
- Risk: Users can pass mismatched task specifications (e.g., task expects 6D obs but provide 4D); errors occur at runtime rather than initialization
- Files: `src/nn4psych/models/multitask_actor_critic.py` (TaskSpec class, no validation)
- Current mitigation: Tests verify dimensions but only in test suite
- Recommendations: Add runtime validation in TaskSpec; verify encoder input dimensions match actual task data during forward pass

**No bounds checking on learning rates and discount factors:**
- Risk: Users can pass gamma > 1 or learning_rate < 0 without error; leads to divergent training
- Files: `src/nn4psych/training/configs.py` (TrainingConfig, MultiTaskTrainingConfig)
- Current mitigation: Comments in docstrings describe expected ranges
- Recommendations: Add `__post_init__` validators to TrainingConfig classes; validate gamma in [0, 1), learning_rate > 0

**Model weight file loading doesn't validate architecture:**
- Risk: Loading checkpoint into model with wrong dimensions fails silently or at unpredictable point
- Files: `src/nn4psych/analysis/behavior.py` (line 134), `batch_extract_behavior()` (line 203)
- Current mitigation: None; relies on KeyError during state_dict loading
- Recommendations: Add architecture compatibility check before loading; save model architecture with weights

## Performance Bottlenecks

**No batch processing in extract_behavior:**
- Problem: `extract_behavior()` processes one sample at a time (batch_size=1)
- Files: `src/nn4psych/analysis/behavior.py` (line 77 creates single sample)
- Cause: Designed for trial-by-trial sequential processing; no vectorization
- Current impact: Slow for large-scale evaluation (100 epochs x 200 trials = 20K forward passes)
- Improvement path: Vectorize to process multiple trials/epochs concurrently; batch environment resets

**Learning rate calculation includes redundant sorting:**
- Problem: `get_lrs_v2()` sorts by prediction error but padding destroys sort order (padding added at end)
- Files: `src/nn4psych/utils/metrics.py` (lines 111-113, then 116-128)
- Cause: Sorting then padding defeats purpose of sorting
- Impact: Minimal but indicates unclear logic flow
- Improvement path: Either keep sort and don't pad, or pad first then sort

**Normalization function called per timestep in extract_behavior:**
- Problem: `env.normalize_states(obs)` called inside inner loop (line 84)
- Files: `src/nn4psych/analysis/behavior.py` (lines 72, 84)
- Cause: States normalized after each step; no caching
- Impact: Redundant computations; not critical but adds overhead
- Improvement path: Vectorize normalization across batch

## Fragile Areas

**PIE environment dependency not specified clearly:**
- Files: `src/nn4psych/analysis/behavior.py`, `tests/test_task_compatibility.py`
- Why fragile: Code imports from `envs` but environment classes (PIE_CP_OB_v2) defined in `envs/pie_environment.py` without version/compatibility specification
- Safe modification: Always use from `envs` package import; never hardcode environment implementations; document expected interface
- Test coverage: Basic shape tests exist; missing tests for behavior consistency across environment changes

**NeuroGym wrapper dimensions are "approximate":**
- Files: `src/nn4psych/training/configs.py` (lines 667-692, comments state "obs/action dims are approximate")
- Why fragile: DawTwoStep obs_dim=4 is placeholder, actual value determined at runtime by environment
- Safe modification: Document that dimension discovery happens at env creation time; validate at startup; store actual dims after instantiation
- Test coverage: Tests in `test_task_compatibility.py` print actual dimensions but don't enforce consistency

**extract_behavior relies on private environment methods:**
- Files: `src/nn4psych/analysis/behavior.py` (line 66: `env._reset_state()`, line 88: `env.get_state_history()`)
- Why fragile: Uses underscore methods suggesting internal implementation details; will break if PIE environment changes internal API
- Safe modification: Define public interface methods for state reset and history retrieval; update PIE_CP_OB_v2 to expose these
- Test coverage: Only tested with current PIE implementation

**Reward handling inconsistency in extract_behavior:**
- Files: `src/nn4psych/analysis/behavior.py` (line 69, 85)
- Why fragile: Reward initialized to 0.0 then updated from env.step() output; no validation that reward is scalar
- Safe modification: Add assertions that reward is float/scalar; clarify whether reward should be normalized
- Test coverage: No explicit test for reward types or ranges

## Scaling Limits

**Single-task model assumes 1-hot or embedding for context:**
- Current capacity: Handles 2-10 tasks efficiently with one-hot; embedding mode not tested at scale
- Limit: Beyond ~100 tasks, one-hot context becomes inefficient; embedding mode not well tested
- Scaling path: Add curriculum learning for task addition; benchmark embedding mode at scale; consider attention-based task selection

**Training on CPU becomes very slow beyond 100 epochs:**
- Current capacity: Typical training on CPU: ~1-5 seconds per epoch (200 trials)
- Limit: 100 epochs = 100-500 seconds acceptable; 1000+ epochs becomes impractical
- Scaling path: GPU-accelerated environment simulation; distributed training; or use faster environment implementation

**Memory usage not profiled:**
- Files: No memory profiling or warnings
- Current impact: Unknown; likely fine for single models but batch analysis could use significant RAM
- Scaling limit: Unclear when batch_extract_behavior() would hit memory ceiling

## Dependencies at Risk

**neurogym as optional dependency with incomplete fallback:**
- Risk: Code gracefully handles missing neurogym but task registry references neurogym tasks (lines 666-692)
- Files: `envs/__init__.py` (lines 12-33), `src/nn4psych/training/configs.py` (lines 666-692)
- Impact: User can try to create neurogym task configs even when neurogym not installed; fails at runtime
- Migration plan: Either make neurogym required for multi-task training, or delay task registry population until import-time

**gymnasium pinned to >=0.28.0 but environment interface may change:**
- Files: `pyproject.toml` (line 33)
- Risk: Gymnasium 0.30+ may have breaking changes; no upper bound specified
- Impact: Future version could break environment interface
- Recommendation: Pin both lower and upper bounds; test against latest stable version regularly

**scipy.ndimage.uniform_filter1d used in metrics but scipy version unconstrained:**
- Files: `src/nn4psych/utils/metrics.py` (line 51)
- Risk: API may change in scipy 2.0+
- Impact: Smoothing window behavior could differ
- Recommendation: Specify scipy version bounds; add fallback implementation

## Missing Critical Features

**No configuration validation at startup:**
- Problem: ExperimentConfig accepts arbitrary hyperparameter combinations without warning
- Blocks: Cannot prevent common mistakes (e.g., gamma > 1, negative learning rate, invalid discount factors)
- Recommendation: Add `__post_init__` validators to all Config dataclasses; document valid ranges

**No model versioning or compatibility checking:**
- Problem: Saved models don't record architecture version or config
- Blocks: Loading old checkpoints into new code version may fail silently
- Recommendation: Save config alongside weights; add architecture hash to checkpoint

**No visualization utilities in core package:**
- Problem: plotting.py exists but is sparse; no standard functions for visualizing learning curves, behavior
- Files: `src/nn4psych/utils/plotting.py`
- Blocks: Analysis scripts must implement custom plotting
- Recommendation: Add standard plots (learning curves, learning rates vs PE, task performance over time)

## Test Coverage Gaps

**Untested areas:**
- What's not tested: Configuration round-trip (to_yaml/from_yaml, to_json/from_json) on all Config classes
- Files: `src/nn4psych/training/configs.py` (lines 167-205, 507-525)
- Risk: YAML/JSON serialization could silently drop fields or fail on edge cases
- Priority: Medium - configuration is critical but tests exist for one simple case only

**Untested areas:**
- What's not tested: Error handling in extract_behavior when environment returns unexpected shapes
- Files: `src/nn4psych/analysis/behavior.py` (entire extract_behavior function)
- Risk: If environment changes, extract_behavior fails with cryptic shapes error
- Priority: Medium - used in batch analysis

**Untested areas:**
- What's not tested: Multi-task training loop with interleaved tasks (only single-task forward passes tested)
- Files: `src/nn4psych/models/multitask_actor_critic.py` (no training loop tests)
- Risk: Task interleaving logic not validated; could have bugs in epoch/trial scheduling
- Priority: High - core feature for multi-task training

**Untested areas:**
- What's not tested: PaddedMultiTaskActorCritic model (alternative architecture defined but never used or tested)
- Files: `src/nn4psych/models/multitask_actor_critic.py` (lines 446-611)
- Risk: Dead code or untested feature; if used, behavior is unknown
- Priority: Low - alternative approach, not primary path

**Untested areas:**
- What's not tested: NeuroGym environment reset and trial boundaries (only basic step tests)
- Files: `tests/test_task_compatibility.py` (basic forward pass tests only)
- Risk: Trial management bugs could accumulate errors across episodes
- Priority: High - required for reliable multi-task training

---

*Concerns audit: 2026-01-28*
