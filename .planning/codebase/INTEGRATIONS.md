# External Integrations

**Analysis Date:** 2026-01-28

## APIs & External Services

**None detected.**

This codebase does not integrate with external HTTP APIs, cloud services, or remote APIs. All computation is local.

## Data Storage

**Databases:**

None. The codebase uses local file-based storage exclusively.

**File Storage:**

Local filesystem only:
- Raw data: `data/raw/` - Immutable source data (Nassar task data, behavioral records)
- Processed data: `data/processed/` - Cleaned datasets
- Intermediate: `data/intermediate/` - Temporary arrays and cache files
- Outputs: `output/` - CSV files with results, metrics, behavioral summaries
- Figures: `figures/` - PNG/PDF plot outputs from analysis scripts
- Model checkpoints: `trained_models/checkpoints/`, `trained_models/best_models/` - PyTorch .pth files

**Configuration Management:**

YAML configuration files in `src/nn4psych/configs/`:
- `default.yaml` - Baseline experiment configuration
- Loaded via `ExperimentConfig.from_yaml()` in `src/nn4psych/training/configs.py`

**Caching:**

No explicit caching layer. Outputs are written to disk for manual inspection and reuse.

## Authentication & Identity

**Auth Provider:**

None. No authentication system implemented.

**Access Control:**

None. This is a research codebase with no multi-user or permission system.

## Monitoring & Observability

**Error Tracking:**

None (no external error tracking service).

Local error handling:
- Exceptions caught and logged via Python logging (implicit; no explicit logging config found)
- Test failures tracked via pytest output

**Logs:**

Console output only:
- Training progress via `tqdm` progress bars
- Test output via pytest (see `pytest.ini` configuration with `-v --tb=short`)
- Analysis script outputs printed to stdout

No persistent log files configured.

**Metrics Collection:**

Metrics saved to CSV files:
- `output/behavioral_summary/` - Behavioral metrics per model/condition
- `output/model_performance/` - Training metrics (loss, accuracy, etc.)
- `output/parameter_exploration/` - Hyperparameter sweep results

Metrics are computed locally and stored as files; no metrics streaming service.

## CI/CD & Deployment

**Hosting:**

Not applicable. This is a research/training codebase. No deployment target configured.

Can be run on:
- Local development machine
- HPC cluster (via batch job submission)
- Cloud compute (AWS, GCP, Azure) if manually provisioned

**CI Pipeline:**

Pre-commit hooks only (GitHub/local):
- Config: `.pre-commit-config.yaml`
- Hooks run: black, flake8, isort, YAML/JSON/TOML validation
- No GitHub Actions workflow detected

Optional testing:
- `pytest` configured in `pytest.ini`
- Run manually with `pytest` command
- Tests in `tests/` and `validation/` directories

No automated CI pipeline (GitHub Actions, GitLab CI, etc.).

## Environment Configuration

**Required Environment Variables:**

None detected. The codebase does not read environment variables for configuration.

All configuration via:
1. Python code defaults
2. YAML configuration files
3. Command-line arguments (in scripts)
4. Dataclass defaults in `src/nn4psych/training/configs.py`

**Optional Environment Variables (by convention, not used):**

Could be added for:
- `CUDA_VISIBLE_DEVICES` - Control GPU access (PyTorch standard)
- `OMP_NUM_THREADS` - Control NumPy threading
- `DATA_DIR` - Override data directory (not currently implemented)

**Secrets/Credentials:**

None required. No API keys, database passwords, or credentials anywhere in codebase.

## Webhooks & Callbacks

**Incoming Webhooks:**

None. This is a standalone Python package. No web server or HTTP endpoints.

**Outgoing Webhooks:**

None. The codebase does not make outbound HTTP requests.

## Task-Specific Integrations

**NeuroGym Tasks (Optional):**

Third-party package integration (optional):
- Package: `neurogym 0.0.1+` - Neuroscience behavioral task library
- License: Open source (MIT)
- URL: https://github.com/neurogym/neurogym
- Integration point: `envs/neurogym_wrapper.py`
- Gracefully disabled if not installed (see `envs/__init__.py` try/except block)

**Supported NeuroGym Tasks:**
- `DawTwoStep-v0` - Two-step decision making (Daw et al. 2011)
- `SingleContextDecisionMaking-v0` - Context-dependent perceptual decision
- `PerceptualDecisionMaking-v0` - Random dot motion discrimination
- Others via generic wrapper

**NeuroGym Wrapper Functionality:**
- Converts neurogym trial-based interface to step-based interface
- Handles observation/reward/done wrapping
- Provides context ID for multi-task identification
- Stores task metadata (input_dim, action_dim, action_names)

**Gymnasium/Gym Interface:**

Primary environment abstraction:
- Gym 0.26+ (legacy support)
- Gymnasium 0.29+ (current standard)
- Custom environment: `envs/pie_environment.py` - `PIE_CP_OB_v2`
  - Implements Gymnasium interface (step, reset, render)
  - Predictive Inference Environment for change-point and oddball tasks

## Data Sources

**Behavioral Data:**

- Nassar et al. (2021) dataset - Change-point and oddball task performance
  - Raw data: `data/raw/` (immutable)
  - Extraction script: `extract_nassar_trial_data.py` (top-level)
  - Inspection scripts: `inspect_nassar_data.py`, `inspect_raw_nassar_data.py`

**Task Stimuli:**

Generated procedurally:
- Change-point task: Gaussian reward function, discrete state changes
- Oddball task: Rare stimulus detection
- NeuroGym tasks: Defined by neurogym package

## Output Destinations

**Analysis Outputs:**

CSV files saved to `output/`:
- Behavioral summaries: `output/behavioral_summary/collated_model_behavior.csv`
- Learning rates: `output/behavioral_summary/learning_rates_by_condition.csv`
- Trial data: `output/behavioral_summary/task_trials_long.csv`
- Hyperparameter sweeps: `output/parameter_exploration/gamma_sweep_results.csv`, etc.
- Model comparison: `output/model_performance/model_comparison_metrics.csv`

**Visualizations:**

PNG/PDF figures saved to `figures/`:
- Behavioral plots: `figures/behavioral_summary/`
- Model performance: `figures/model_performance/`
- Parameter exploration: `figures/parameter_exploration/`
- Dynamical systems: `figures/dynamical_systems/`

**Model Artifacts:**

PyTorch model weights (.pth files):
- Location: `trained_models/checkpoints/`, `trained_models/best_models/`
- JSON metadata: `trained_models/transfer/transfer_*_results.json`
- Validation results: `output/validation/nassar2021/validation_results.json`

## Dependency on External Research

**References (Not External Integrations):**

The codebase is based on published research:
- Nassar et al. (2021) - Change-point and oddball learning paradigms (inspiration for task)
- Wang et al. (2018) - RNN actor-critic foundations (model architecture)
- Daw et al. (2011) - NeuroGym DawTwoStep task

These are implemented as local code, not fetched from external sources at runtime.

---

*Integration audit: 2026-01-28*
