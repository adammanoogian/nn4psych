# Codebase Structure

**Analysis Date:** 2026-01-28

## Directory Layout

```
nn4psych/                              # Project root
├── src/                               # Source layout (PEP 420)
│   └── nn4psych/                      # Main package
│       ├── __init__.py                # Package exports (ActorCritic, PIE_CP_OB_v2)
│       ├── models/                    # Neural network architectures
│       │   ├── __init__.py
│       │   ├── actor_critic.py        # Single-task ActorCritic RNN
│       │   └── multitask_actor_critic.py  # Multi-task variant
│       ├── training/                  # Configuration management
│       │   ├── __init__.py
│       │   └── configs.py             # ModelConfig, TaskConfig, TrainingConfig, ExperimentConfig
│       ├── analysis/                  # Behavior extraction and analysis
│       │   ├── __init__.py
│       │   ├── behavior.py            # extract_behavior() function
│       │   └── hyperparams.py         # HyperparamAnalyzer class
│       ├── utils/                     # Shared utilities
│       │   ├── __init__.py
│       │   ├── io.py                  # save_model, load_model, saveload
│       │   ├── metrics.py             # get_lrs, get_lrs_v2, learning rate extraction
│       │   └── plotting.py            # plot_behavior, visualization utilities
│       └── configs/                   # Config file templates
│           └── default.yaml           # Default experiment configuration
│
├── envs/                              # Environment definitions (top-level, not src/)
│   ├── __init__.py                    # Exports PIE_CP_OB_v2
│   └── pie_environment.py             # PIE_CP_OB_v2 class (change-point, oddball)
│
├── scripts/                           # Executable analysis scripts
│   ├── training/                      # Model training entry points
│   │   ├── train_rnn_canonical.py     # Main single-task training loop
│   │   ├── train_multitask.py         # Multi-task training
│   │   ├── transfer_learning.py       # Transfer learning variant
│   │   └── examples/
│   │       └── train_example.py       # Minimal tutorial example
│   ├── data_pipeline/                 # Analysis pipeline stages
│   │   ├── 00_run_full_pipeline.py    # Orchestrate all stages
│   │   ├── 01_extract_model_behavior.py  # Run models, extract states
│   │   ├── 02_compute_learning_metrics.py # Compute learning rates
│   │   ├── 03_analyze_hyperparameter_sweeps.py
│   │   ├── 04_visualize_behavioral_summary.py
│   │   ├── 05_visualize_hyperparameter_effects.py
│   │   ├── 06_compare_with_human_data.py
│   │   ├── 07_analyze_fixed_points.py
│   │   └── extract_nassar_trials.py   # Data extraction from Nassar et al.
│   ├── analysis/                      # Research analysis scripts
│   │   ├── analyze_hyperparams_unified.py  # Hyperparameter sweep analysis
│   │   ├── analyze_rnn_refactored.py
│   │   ├── analyze_fixed_points.py
│   │   ├── nassarfig6.py              # Reproduce paper figures
│   │   ├── validate_nassar2021.py     # Model validation
│   │   ├── compare_fitted_params.py
│   │   ├── visualize_learning_rates.py
│   │   └── bayesian/                  # Bayesian model fitting
│   │       ├── bayesian_models.py
│   │       ├── fit_bayesian_pymc.py
│   │       ├── fit_bayesian_pyem.py
│   │       └── fit_bayesian_numpyro.py
│   └── fitting/                       # Model fitting utilities
│       ├── batch_fit_bayesian.py
│       └── README.md
│
├── trained_models/                    # Model weights and checkpoints
│   ├── checkpoints/                   # Training checkpoints
│   ├── best_models/                   # Best performing models
│   ├── weights/                       # Saved model weights
│   └── README.md
│
├── data/                              # Data organization
│   ├── raw/                           # Immutable source data
│   │   ├── nassar2021/                # Nassar et al. 2021 human data
│   │   └── fig2_values/               # Reference data
│   ├── processed/                     # Cleaned/preprocessed data
│   │   ├── bayesian_models/
│   │   ├── pt_rnn_context/
│   │   ├── rnn_behav/                 # Extracted model behavior
│   │   │   └── model_params_101000/
│   │   │       └── 30epochs/
│   │   └── nassar2021/
│   ├── intermediate/                  # Temporary computation arrays
│   └── README.md
│
├── output/                            # Analysis results
│   ├── behavioral_summary/            # Behavioral data CSVs
│   ├── model_performance/             # Performance metrics
│   ├── parameter_exploration/         # Hyperparameter sweep results
│   ├── processed/                     # Processed data artifacts
│   │   └── nassar2021/
│   └── validation/
│       └── nassar2021/
│
├── figures/                           # Generated plots
│   ├── behavioral_summary/            # Behavior visualizations
│   ├── model_performance/             # Performance plots
│   ├── parameter_exploration/         # Hyperparameter sweep plots
│   ├── dynamical_systems/             # RNN dynamical analysis
│   └── analysis/
│
├── notebooks/                         # Jupyter notebooks
│   ├── exploratory/                   # Experimental analysis notebooks
│   ├── tutorials/
│   └── README.md
│
├── tests/                             # Unit and integration tests
│   ├── __init__.py
│   ├── test_imports.py                # Module import validation
│   ├── test_models.py                 # ActorCritic forward pass, initialization
│   └── test_task_compatibility.py     # Environment and task validation
│
├── validation/                        # Integration test outputs
│
├── archive/                           # Legacy code (not maintained)
│   ├── v0/                            # Early JAX implementations
│   ├── v1/                            # JAX actor-critic variants
│   ├── v2/                            # PyTorch variants
│   ├── toy/                           # Minimal prototype
│   ├── legacy_scripts/
│   ├── legacy_analysis/
│   ├── normative_model/
│   └── README.md
│
├── bayesian/                          # Bayesian modeling utilities (legacy location)
│
├── docs/                              # Project documentation
│   ├── DEVELOPMENT.md                 # Development guidelines
│   ├── ANALYSIS_PIPELINE.md           # Pipeline documentation
│   ├── NASSAR_DATA_PROCESSING.md      # Data processing guide
│   ├── BAYESIAN_VALIDATION.md
│   ├── COMPLETE_ANALYSIS_PIPELINE.md
│   ├── CHANGELOG.md
│   ├── PROJECT_STATUS.md
│   ├── REORGANIZATION_VALIDATION.md
│   ├── methods/
│   │   └── bayesian_normative_model.md
│   └── reference_papers/
│
├── config.py                          # Central configuration (paths, defaults, parameters)
├── pyproject.toml                     # Package metadata and dependencies
├── pytest.ini                         # Pytest configuration
├── environment.yml                    # Conda environment specification
├── .gitignore                         # Git ignore rules
├── .pre-commit-config.yaml            # Pre-commit hooks
├── .github/                           # GitHub workflows and templates
├── .vscode/                           # VS Code settings
└── README.md                          # Project overview
```

## Directory Purposes

**src/nn4psych/ (Source Package):**
- Purpose: Core library code following PEP 517 source layout
- Contains: Models, configs, utilities, analysis functions
- Key files: `__init__.py` exports ActorCritic and PIE_CP_OB_v2 as package entrypoints

**envs/ (Top-level Environments):**
- Purpose: Task environment definitions (separated from src/ for flexibility)
- Contains: PIE_CP_OB_v2 class only
- Isolation: Not part of nn4psych package proper; imported as `from envs import PIE_CP_OB_v2`

**scripts/training/:**
- Purpose: Executable entry points for model training
- Usage: `python scripts/training/train_rnn_canonical.py --epochs 100 --gamma 0.95`
- Key file: `train_rnn_canonical.py` (main production training script)

**scripts/data_pipeline/:**
- Purpose: Sequential analysis stages (01, 02, 03, ..., 07)
- Usage: Run in order for complete analysis
- Each script is self-contained and can run independently

**scripts/analysis/:**
- Purpose: Research-specific analysis scripts
- Usage: `python scripts/analysis/validate_nassar2021.py` to validate against human data
- Not part of standard pipeline

**trained_models/:**
- Purpose: Organize saved model weights by category
- Structure: checkpoints/ for in-training saves, best_models/ for final models
- Naming: Script output determines filenames

**data/raw/ → data/processed/:**
- Purpose: Strict data organization following data science standards
- raw/: Never modified; immutable source (Nassar et al. 2021 human data)
- intermediate/: Temporary arrays during processing
- processed/: Final cleaned datasets and extracted behaviors

**output/:**
- Purpose: Analysis results (CSVs, tables)
- Subdir: behavioral_summary/ contains collated_model_behavior.csv, learning_rates_by_condition.csv, etc.

**figures/:**
- Purpose: Generated visualizations
- Subdir: Mirrors output/ structure (behavioral_summary/, model_performance/, etc.)

**archive/:**
- Purpose: Preserve legacy implementations
- Status: Not maintained; code here should not be imported by active scripts
- Reason: Codebase recently consolidated (see README.md migration table)

## Key File Locations

**Entry Points:**
- Training: `scripts/training/train_rnn_canonical.py`
- Behavior Extraction: `scripts/data_pipeline/01_extract_model_behavior.py`
- Analysis Pipeline: `scripts/data_pipeline/00_run_full_pipeline.py`
- Validation: `scripts/analysis/validate_nassar2021.py`

**Configuration:**
- Project paths/params: `config.py` (root level)
- Experiment config system: `src/nn4psych/training/configs.py`
- Default experiment: `src/nn4psych/configs/default.yaml`

**Core Logic:**
- Model: `src/nn4psych/models/actor_critic.py` (single-task), `src/nn4psych/models/multitask_actor_critic.py` (multi-task)
- Task: `envs/pie_environment.py` (PIE_CP_OB_v2 class)
- Behavior: `src/nn4psych/analysis/behavior.py` (extract_behavior function)
- Metrics: `src/nn4psych/utils/metrics.py` (get_lrs_v2 learning rate function)

**Testing:**
- Unit tests: `tests/test_models.py` (forward pass, initialization)
- Import validation: `tests/test_imports.py`
- Task tests: `tests/test_task_compatibility.py`

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `actor_critic.py`, `pie_environment.py`)
- Scripts: Descriptive with numeric prefix if ordered (e.g., `01_extract_model_behavior.py`)
- Test files: `test_*.py` (pytest discovery pattern)
- Config files: `.yaml` (YAML format), `.yml` (conda environments)

**Directories:**
- Packages: `lowercase` (e.g., `nn4psych`, `envs`)
- Stage directories: Numeric prefix for ordering (e.g., `01_`, `02_`)
- Output categories: `descriptive_underscore` (e.g., `behavioral_summary`, `model_performance`)

**Classes:**
- PascalCase: ActorCritic, PIE_CP_OB_v2, ExperimentConfig
- All classes are top-level imports or in `__init__.py`

**Functions:**
- snake_case: extract_behavior, get_lrs_v2, save_model, load_model

**Variables/Constants:**
- UPPERCASE for module-level constants: PROJECT_ROOT, PERFORMANCE_THRESHOLD, GAMMA_VALUES
- snake_case for local variables

## Where to Add New Code

**New Model Architecture:**
- Primary code: `src/nn4psych/models/my_model.py`
- Tests: `tests/test_models.py` (add test class for new model)
- Export: Add to `src/nn4psych/models/__init__.py`
- Usage: Training scripts can import via `from nn4psych.models import MyModel`

**New Analysis Function:**
- Behavior analysis: `src/nn4psych/analysis/behavior.py`
- Metrics/utilities: `src/nn4psych/utils/metrics.py` or new file `src/nn4psych/utils/my_utils.py`
- Plotting: `src/nn4psych/utils/plotting.py`
- Pipeline script: `scripts/data_pipeline/XX_my_analysis.py` (use next sequential number)

**New Task/Environment:**
- Implementation: `envs/my_environment.py`
- Export: Add to `envs/__init__.py`
- Tests: `tests/test_task_compatibility.py` (add test for new environment)
- Usage: Training scripts import via `from envs import MyEnvironment`

**New Training Variant:**
- Implementation: `scripts/training/train_my_variant.py`
- Follow pattern from `train_rnn_canonical.py` (argparse for params)
- Save models to `trained_models/` (script responsibility)

**Utilities and Helpers:**
- Shared across package: `src/nn4psych/utils/`
- Specific to analysis: `src/nn4psych/analysis/`
- Shared across scripts: Consider moving to utils layer
- One-off analysis: Keep in `scripts/analysis/`

## Special Directories

**archive/:**
- Purpose: Preserve legacy code from pre-consolidation era
- Generated: No (manually preserved)
- Committed: Yes (for reference only)
- Policy: Never delete; never import from active code

**tests/:**
- Purpose: Unit and integration tests
- Generated: No (manually written)
- Committed: Yes
- Command: `pytest` (runs all; `pytest tests/test_models.py` for specific)

**notebooks/:**
- Purpose: Exploratory and tutorial Jupyter notebooks
- Generated: No (interactive development)
- Committed: Some (key tutorials); others in .gitignore
- Usage: `jupyter notebook notebooks/exploratory/`

**output/ and figures/:**
- Purpose: Generated analysis artifacts
- Generated: Yes (by data_pipeline/ scripts)
- Committed: No (.gitignore)
- Retention: Git-ignored; regenerated by pipeline

**.planning/codebase/:**
- Purpose: GSD codebase analysis documents
- Generated: Yes (by Claude agent)
- Committed: Yes (planning artifacts)
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, etc.

---

*Structure analysis: 2026-01-28*
