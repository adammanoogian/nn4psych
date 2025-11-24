# NN4Psych Project Status & Analysis Guide

**Date:** 2025-11-21
**Status:** All imports working after reorganization fixes
**Version:** 0.2.0

---

## Validation Status

**Import Tests:** 8/8 PASSED

| Module | Status |
|--------|--------|
| nn4psych package | PASS |
| ActorCritic model | PASS |
| PIE_CP_OB_v2 Environment | PASS |
| Training configs | PASS |
| Analysis behavior | PASS |
| Utils io | PASS |
| Utils metrics | PASS |
| Config module | PASS |

**Fix Applied:** Updated `envs/__init__.py` to use relative import (`.pie_environment`)

---

## All Analysis Scripts & Their Outputs

### 1. Data Processing Pipeline
| Script | Location | Output | Description |
|--------|----------|--------|-------------|
| `00_run_full_pipeline.py` | `scripts/data_pipeline/` | N/A | Master runner for all stages |
| `01_extract_model_behavior.py` | `scripts/data_pipeline/` | `output/behavioral_summary/task_trials_long.csv`, `raw_behavior_data.pickle` | Extract behavior from trained models |
| `02_compute_learning_metrics.py` | `scripts/data_pipeline/` | `output/behavioral_summary/learning_rates_by_condition.csv`, `summary_performance_metrics.csv` | Compute learning rates |
| `03_analyze_hyperparameter_sweeps.py` | `scripts/data_pipeline/` | `output/parameter_exploration/{param}_sweep_results.csv` | Analyze hyperparameter effects |

### 2. Analysis Scripts
| Script | Location | Output | Description |
|--------|----------|--------|-------------|
| `analyze_rnn_refactored.py` | `scripts/analysis/` | Various figures | Complete RNN analysis |
| `analyze_hyperparams_unified.py` | `scripts/analysis/` | `output/parameter_exploration/` | Unified hyperparameter analysis |
| `visualize_learning_rates.py` | `scripts/analysis/` | `figures/behavioral_summary/` | Learning rate plots |
| `nassarfig6.py` | `scripts/analysis/` | `figures/behavioral_summary/nassarfig6*.png` | Reproduce Nassar et al. figures |
| `analyze_fixed_points.py` | `scripts/analysis/` | Neural dynamics plots | Fixed point analysis |

### 3. Training Scripts
| Script | Location | Output | Description |
|--------|----------|--------|-------------|
| `train_rnn_canonical.py` | `scripts/training/` | `trained_models/checkpoints/*.pth` | Full RNN training |
| `train_example.py` | `scripts/training/examples/` | Model files | Example config-based training |

### 4. Bayesian Fitting Scripts
| Script | Location | Output | Description |
|--------|----------|--------|-------------|
| `fit_bayesian_pyem.py` | `scripts/fitting/` | `output/bayesian_fits/` | PyEM framework fitting |
| `fit_bayesian_pymc.py` | `scripts/fitting/` | `output/bayesian_fits/` | PyMC Bayesian fitting |
| `bayesian_models.py` | `scripts/analysis/bayesian/` | N/A (library) | Core Bayesian models |
| `pyem_models.py` | `scripts/analysis/bayesian/` | N/A (library) | PyEM model implementations |

### 5. Master Analysis Runner
| Script | Location | Description |
|--------|----------|-------------|
| `run_complete_analysis.py` | `scripts/` | Run any combination of pipelines |

---

## Figure Locations

### `figures/behavioral_summary/` (29 files)
- `Helicopter_CP.png` - Change-point condition visualization
- `Helicopter_OB.png` - Oddball condition visualization
- `learning_rate_vs_prediction_error.png` - Main learning rate plot
- `learning_rate_by_prediction_error.png` - LR by PE (all)
- `learning_rate_by_prediction_error_change-point.png` - LR by PE (CP)
- `learning_rate_by_prediction_error_oddball.png` - LR by PE (OB)
- `learning_rate_histogram.png` - LR distribution
- `learning_rate_histogram_change-point.png` - LR distribution (CP)
- `learning_rate_histogram_oddball.png` - LR distribution (OB)
- `learning_rate_after_hazard.png` - LR after hazard events
- `lr_after_hazard.png` - Alternative hazard plot
- `states_and_learning_rate_over_trials.png` - Trial-by-trial states
- `states_and_learning_rate_over_trials_change-point.png` - States (CP)
- `states_and_learning_rate_over_trials_oddball.png` - States (OB)
- `update_by_prediction_error.png` - Update vs PE
- `update_by_prediction_error_change-point.png` - Update vs PE (CP)
- `update_by_prediction_error_oddball.png` - Update vs PE (OB)
- `interactions_line_graph.png` - Interaction effects
- `interactions_line_graph_change-point.png` - Interactions (CP)
- `interactions_line_graph_oddball.png` - Interactions (OB)
- `nassarfig6a.png` - Reproduction of Nassar Fig 6A
- `nassarfig6b.png` - Reproduction of Nassar Fig 6B

### `figures/model_performance/` (6 files)
- `pca_trajectories.mp4` - PCA animation (2D)
- `pca_trajectories_3d.mp4` - PCA animation (3D)
- `pca_trajectories_contextual.mp4` - Contextual PCA
- `pca_trajectories_train.mp4` - Training PCA
- `pca_trajectories_test_3d.mp4` - Test PCA (3D)
- `contextual_bandit.png` - Contextual bandit visualization
- `area_between_curves.png` - AUC comparison

---

## Data Locations

### `data/raw/` - Immutable source data
- `nassar2021/` - Human behavioral data (.mat files)
- `fig2_values/` - Reference values for Figure 2 (.pickle)

### `data/processed/` - Analysis outputs
- `rnn_behav/model_params_101000/` - RNN behavioral extractions
- `bayesian_models/` - Bayesian model outputs
- `pt_rnn_context/` - Context-based RNN data

### `data/intermediate/` - Temporary files
- `activity_*.npy` - Neural activity arrays
- `history_*.npy` - History arrays
- `W*_contextual.npy` - Weight matrices

### `trained_models/checkpoints/model_params_101000/`
- 100+ trained .pth model files with various hyperparameters
- Naming: `{loss}_V3_{gamma}g_{preset}rm_{rollout}bz_...pth`

---

## Quick Commands

```bash
# Activate conda environment
conda activate base

# Run full data pipeline
python scripts/data_pipeline/00_run_full_pipeline.py

# Run specific analysis
python scripts/analysis/analyze_hyperparams_unified.py --param gamma
python scripts/analysis/visualize_learning_rates.py
python scripts/analysis/nassarfig6.py

# Run Bayesian fitting
python scripts/fitting/fit_bayesian_pyem.py --n_iter 1000
python scripts/fitting/fit_bayesian_pymc.py --method mle

# Train a model
python scripts/training/train_rnn_canonical.py --gamma 0.95 --epochs 10000

# Run master analysis runner
python scripts/run_complete_analysis.py --list           # List pipelines
python scripts/run_complete_analysis.py --preset full    # Run full analysis
python scripts/run_complete_analysis.py --preset validate # Run validation
```

---

## Remaining Tasks (TODO)

### Critical (Fix Before Use)
1. [ ] **Upgrade gym to gymnasium** - Current gym is deprecated and doesn't support NumPy 2.0
   - Replace `import gym` with `import gymnasium as gym`
   - Update `envs/pie_environment.py` to use gymnasium API

### High Priority
2. [ ] **Create output directory structure** - `output/` directory is currently empty
   - Run `scripts/data_pipeline/01_extract_model_behavior.py` to generate initial data

3. [ ] **Test data pipeline end-to-end** - Run full pipeline and verify outputs
   ```bash
   python scripts/data_pipeline/00_run_full_pipeline.py
   ```

4. [ ] **Validate training scripts** - Test that training produces valid models
   ```bash
   python scripts/training/train_rnn_canonical.py --epochs 100 --gamma 0.95
   ```

### Medium Priority
5. [ ] **Update pyproject.toml** - Change gym to gymnasium in dependencies
6. [ ] **Add pytest tests** - Create unit tests in `tests/` directory
7. [ ] **Test Bayesian fitting** - Verify PyEM and PyMC scripts work
8. [ ] **Generate missing output CSVs** - Run pipeline to create behavioral summary CSVs

### Low Priority / Nice to Have
9. [ ] **Create Sphinx documentation** - Generate API docs
10. [ ] **Add GitHub Actions CI** - Automated testing on push
11. [ ] **Containerize with Docker** - For reproducibility
12. [ ] **Add model registry** - MLflow or W&B integration

---

## File Structure Summary

```
nn4psych/
├── src/nn4psych/           # Core package (WORKING)
│   ├── models/             # ActorCritic
│   ├── training/           # Configs
│   ├── analysis/           # Behavior, hyperparams
│   └── utils/              # IO, metrics, plotting
├── envs/                   # Environment (FIXED)
├── scripts/                # Analysis scripts (17 .py files)
│   ├── data_pipeline/      # Numbered pipeline (4 scripts)
│   ├── analysis/           # Analysis tools (5 scripts)
│   ├── training/           # Training (2 scripts)
│   ├── fitting/            # Bayesian (2 scripts)
│   └── run_complete_analysis.py
├── data/                   # Data storage (83 files)
├── figures/                # Generated plots (29 files)
├── trained_models/         # Model checkpoints (100+ files)
├── output/                 # Analysis outputs (EMPTY - needs generation)
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
└── validation/             # Integration tests
```

---

## Notes

- **Gym Deprecation Warning:** The gym package shows a deprecation warning. Should migrate to gymnasium.
- **NumPy 2.0:** Some packages may have compatibility issues with NumPy 2.x
- **Output Directory:** Currently empty - run data pipeline to populate
- **All core imports work correctly after fix**
