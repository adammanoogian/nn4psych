# NN4Psych Analysis Pipeline Guide

**Last Updated:** 2025-11-19
**Version:** v2
**Project:** Neural Networks for Psychological Modeling

---

## Overview

This pipeline processes trained RNN actor-critic models to extract and analyze behavioral data from predictive inference tasks. The pipeline follows standardized data analysis project conventions for reproducibility and maintainability.

---

## Project Structure

### Package Organization

The nn4psych package has been reorganized for clarity and modularity:

```
nn4psych/                      # Main RNN package (pure algorithm)
├── models/
│   └── actor_critic.py       # RNN actor-critic model
├── env/                       # Standalone environment module
│   ├── __init__.py
│   └── pie_environment.py    # PIE_CP_OB_v2 environment
├── training/
│   └── configs.py
├── analysis/
│   ├── behavior.py
│   └── hyperparams.py
├── utils/
│   ├── io.py
│   ├── metrics.py           # Includes Bayesian utility functions
│   └── plotting.py
└── __init__.py

scripts/
├── training/                 # Training scripts
├── data_pipeline/            # Numbered pipeline scripts (00-07)
├── analysis/                 # Analysis & visualization scripts
│   └── bayesian/            # Bayesian normative models
│       ├── __init__.py
│       ├── bayesian_models.py
│       └── pyem_models.py
└── fitting/                 # Model fitting scripts
    ├── fit_bayesian_pymc.py
    └── fit_bayesian_pyem.py
```

### Key Design Decisions

1. **src/ Layout**: Standard Python packaging pattern that eliminates naming confusion between project root (`nn4psych/`) and package (`src/nn4psych/`).

2. **Standalone Environments (`envs/`)**: PIE_CP_OB_v2 is at top level, importable by:
   - RNN training code
   - Bayesian model fitting scripts
   - Analysis scripts
   - External projects

3. **Bayesian Models (`scripts/analysis/bayesian/`)**: Moved out of package since they're analysis tools, not core components.

4. **Organized Storage**:
   - `trained_models/` - Model weights (checkpoints, best models)
   - `data/` - Organized by type (raw, processed, intermediate)
   - `output/` - Data files (CSV, pickle)
   - `figures/` - Plot outputs (by analysis type)
   - `notebooks/` - Organized by purpose (exploratory, tutorials)

### Importing Components

```python
# Import RNN model
from nn4psych.models import ActorCritic

# Import environment (standalone, top-level)
from envs import PIE_CP_OB_v2

# Import Bayesian models (from analysis scripts)
from scripts.analysis.bayesian.bayesian_models import BayesianModel

# Import utilities
from nn4psych.utils.io import save_model, load_model
from nn4psych.utils.metrics import get_lrs_v2
from nn4psych.utils.plotting import plot_behavior
```

---

## Pipeline Architecture

```
Stage 01: Extract Behavior → Stage 02: Compute Metrics → Stage 03: Analyze Hyperparameters
         ↓                            ↓                            ↓
    Trained Models             Learning Rates              Parameter Sweeps
    (.pth files)              & Summary Stats            & Performance Trends
                                      ↓
Stage 04: Behavioral Viz → Stage 05: Hyperparameter Viz → Stage 06: Human Comparison → Stage 07: Dynamical Systems
         ↓                            ↓                            ↓                            ↓
  figures/behavioral_summary  figures/parameter_exploration    Nassar 2021 comparison    Fixed Point Analysis
```

---

## Stage 01: Behavior Extraction

### Script
`scripts/data_pipeline/01_extract_model_behavior.py`

### Description
Loads trained model weights and runs them through the predictive inference task environments (change-point and oddball conditions) to extract behavioral data.

### Input
- Model weight files (`.pth`) from `model_params*/` directories
- Model architecture defined in `config.py`

### Output
- `output/behavioral_summary/task_trials_long.csv` - Long-format trial data
- `output/behavioral_summary/raw_behavior_data.pickle` - Raw state vectors

### Columns Created
- `model_id` - Unique model identifier
- `epoch` - Training epoch number
- `trial` - Trial number within epoch
- `condition` - Task condition (change-point or oddball)
- `bucket_position` - Agent's predicted position
- `bag_position` - True bag drop position
- `helicopter_position` - Hidden state position
- `hazard_trigger` - Binary indicator of hazard event
- `prediction_error` - Signed error (bag - bucket)
- `abs_prediction_error` - Absolute prediction error

### Run
```bash
cd /path/to/nn4psych
python scripts/data_pipeline/01_extract_model_behavior.py
```

---

## Stage 02: Compute Learning Metrics

### Script
`scripts/data_pipeline/02_compute_learning_metrics.py`

### Description
Processes trial data to compute learning rates, prediction errors, and aggregated performance metrics.

### Input
- `output/behavioral_summary/task_trials_long.csv`

### Output
- `output/behavioral_summary/learning_rates_by_condition.csv` - Learning rate observations
- `output/behavioral_summary/summary_performance_metrics.csv` - Aggregated metrics per model

### Key Computations
1. **Learning Rate**: `update / prediction_error` where:
   - `update = diff(bucket_position)`
   - Only computed when `abs_prediction_error > 20`
   - Clipped to [0, 1] range

2. **Summary Statistics**:
   - Mean, std, median of prediction errors
   - Mean, std, median of learning rates
   - Hazard event frequency

### Run
```bash
python scripts/data_pipeline/02_compute_learning_metrics.py
```

---

## Stage 03: Hyperparameter Analysis

### Script
`scripts/data_pipeline/03_analyze_hyperparameter_sweeps.py`

### Description
Analyzes model performance across different hyperparameter configurations by parsing filenames and aggregating results.

### Input
- Model files with encoded hyperparameters in filenames
- Expected format: `{perf}_{version}_{gamma}g_{preset}rm_{rollout}bz_...`

### Output
- `output/parameter_exploration/gamma_sweep_results.csv`
- `output/parameter_exploration/rollout_sweep_results.csv`
- `output/parameter_exploration/preset_sweep_results.csv`
- `output/parameter_exploration/scale_sweep_results.csv`

### Sweep Values
- **Gamma (discount)**: [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1]
- **Rollout (batch size)**: [5, 10, 20, 30, 50, 75, 100, 150, 200]
- **Preset (memory init)**: [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
- **Scale (TD scale)**: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

### Run
```bash
python scripts/data_pipeline/03_analyze_hyperparameter_sweeps.py
```

---

## Stage 04: Behavioral Visualization

### Script
`scripts/data_pipeline/04_visualize_behavioral_summary.py`

### Description
Generates comprehensive behavioral analysis figures from trained models:
- State trajectory plots (helicopter, bucket, bag positions over trials)
- Learning rate vs prediction error curves
- Learning rate histograms by condition
- Update magnitude by prediction error plots

### Input
- Trained model files (`.pth`)
- Or: Pre-extracted behavioral data from Stage 01

### Output
- `figures/behavioral_summary/Helicopter_CP.png` - Change-point state trajectories
- `figures/behavioral_summary/Helicopter_OB.png` - Oddball state trajectories
- `figures/behavioral_summary/learning_rate_by_prediction_error.png`
- `figures/behavioral_summary/learning_rate_histogram_*.png`
- `figures/behavioral_summary/update_by_prediction_error_*.png`
- `figures/behavioral_summary/states_and_learning_rate_over_trials_*.png`

### Run
```bash
python scripts/data_pipeline/04_visualize_behavioral_summary.py --epochs 30
```

---

## Stage 05: Hyperparameter Visualization

### Script
`scripts/data_pipeline/05_visualize_hyperparameter_effects.py`

### Description
Analyzes and visualizes how hyperparameters affect model learning behavior.
Computes area under learning rate curves (CP - OB difference) for each parameter configuration.

### Input
- Trained models with varied hyperparameters in `trained_models/checkpoints/`

### Output
- `figures/parameter_exploration/gamma_area_*.png` - Discount factor effects
- `figures/parameter_exploration/rollout_area_*.png` - Rollout size effects
- `figures/parameter_exploration/preset_area_*.png` - Memory reset effects
- `figures/parameter_exploration/scale_area_*.png` - TD scale effects
- `figures/parameter_exploration/all_params_area.png` - Combined comparison

### Run
```bash
python scripts/data_pipeline/05_visualize_hyperparameter_effects.py --params all --epochs 8
```

---

## Stage 06: Human Data Comparison

### Script
`scripts/data_pipeline/06_compare_with_human_data.py`

### Description
Replicates Nassar 2021 Figure 6, comparing model learning rates with human subject data.
Analyzes patients vs controls in change-point and oddball conditions.

### Input
- `data/raw/nassar2021/slidingWindowFits_model_23-Nov-2021.mat`
- `data/raw/nassar2021/slidingWindowFits_subjects_23-Nov-2021.mat`

### Output
- `figures/behavioral_summary/nassarfig6a.png` - Subject learning rates
- `figures/behavioral_summary/nassarfig6b.png` - Model learning rates
- `figures/model_performance/area_between_curves.png` - Area comparison bar chart

### Run
```bash
python scripts/data_pipeline/06_compare_with_human_data.py
```

---

## Stage 07: Dynamical Systems Analysis

### Script
`scripts/data_pipeline/07_analyze_fixed_points.py`

### Description
Analyzes the dynamical systems properties of trained RNN models to understand their computational structure:
- **Fixed Points**: Finds equilibrium states where `h_t+1 = h_t`
- **Stability Analysis**: Computes Jacobian eigenvalues to classify stable vs unstable fixed points
- **Line Attractors**: Detects continuous attractor manifolds via null space analysis
- **Trajectory Visualization**: Projects high-dimensional RNN dynamics into 2D PCA space

### Input
- Trained model files (`.pth`)
- Default: Best model from `trained_models/checkpoints/model_params_101000/`

### Output
- `figures/dynamical_systems/fixed_points_{model_name}.png` - 2D PCA projection showing:
  - RNN trajectory from random initialization
  - Fixed point locations (red stars)
  - Start and end states
  - Variance explained by PC1 and PC2

- `figures/dynamical_systems/fixed_points_{model_name}.svg` - Vector version

### Key Metrics
- **Number of fixed points**: Unique equilibrium states found
- **Stable vs unstable**: Classification based on max |eigenvalue| < 1
- **Line attractor dimension**: Null space dimension of recurrent weight matrix
- **PCA variance explained**: How much dynamics captured in 2D projection

### Run
```bash
# Use best model (auto-selected)
python scripts/data_pipeline/07_analyze_fixed_points.py

# Specify model
python scripts/data_pipeline/07_analyze_fixed_points.py --model_path path/to/model.pth

# Adjust search parameters
python scripts/data_pipeline/07_analyze_fixed_points.py --num_fps 2000 --num_steps 200
```

---

## Master Pipeline Runner

### Script
`scripts/data_pipeline/00_run_full_pipeline.py`

### Description
Executes all pipeline stages in sequence with error handling and stage selection.

### Usage
```bash
# Run entire pipeline
python scripts/data_pipeline/00_run_full_pipeline.py

# Start from specific stage
python scripts/data_pipeline/00_run_full_pipeline.py --start 2

# Run specific range
python scripts/data_pipeline/00_run_full_pipeline.py --start 2 --end 3

# List all stages
python scripts/data_pipeline/00_run_full_pipeline.py --list
```

---

## Analysis Scripts

Located in `scripts/analysis/`:

### nassarfig6.py
Standalone script for Nassar 2021 figure replication (also available as Stage 06).

**Output**: `figures/behavioral_summary/nassarfig6*.png`, `figures/model_performance/area_between_curves.png`

---

## Exploratory Notebooks

Located in `notebooks/exploratory/`:

### plot_pca.ipynb
Generates PCA trajectory visualizations of RNN hidden states.

**Output**: `figures/model_performance/pca_trajectories.mp4`

### plot_pca_helicopter.ipynb
PCA analysis of hidden states for helicopter task.

**Output**: `figures/model_performance/pca_trajectories_3d.mp4`

### plot_pca_contextual.ipynb
PCA analysis for contextual bandit condition.

**Output**: `figures/model_performance/pca_trajectories_contextual.mp4`, `figures/model_performance/contextual_bandit.png`

> **Note**: These notebooks are exploratory and not part of the main numbered pipeline. Run interactively for visualization development.

---

## Data Integration Pattern

All computed metrics are integrated into main data files rather than creating separate CSVs:

```python
# Example: Adding new metrics to existing data
df_trials = pd.read_csv(TRIALS_DATA_PATH)

# Compute new metric
df_trials['new_metric'] = compute_metric(df_trials)

# Save back to same file
df_trials.to_csv(TRIALS_DATA_PATH, index=False)
```

**Benefits**:
- Single source of truth
- No merging in downstream scripts
- Maintains data lineage

---

## Configuration

All paths and parameters are centralized in `config.py`:

```python
from config import (
    OUTPUT_DIR,
    BEHAVIORAL_SUMMARY_DIR,
    MODEL_PARAMS,
    TASK_PARAMS,
    COLUMN_NAMES,
)
```

**Key Configuration Sections**:
- `MODEL_PARAMS` - Actor-Critic architecture settings
- `TASK_PARAMS` - Environment configuration
- `TRAINING_PARAMS` - Training hyperparameters
- `COLUMN_NAMES` - Standardized column naming
- `*_VALUES` - Hyperparameter sweep ranges

---

## Output Directory Structure

```
output/
├── behavioral_summary/
│   ├── task_trials_long.csv
│   ├── learning_rates_by_condition.csv
│   ├── summary_performance_metrics.csv
│   └── raw_behavior_data.pickle
├── model_performance/
│   ├── hyperparameter_sweep_results.csv
│   └── best_performing_models.csv
└── parameter_exploration/
    ├── gamma_sweep_results.csv
    ├── rollout_sweep_results.csv
    ├── preset_sweep_results.csv
    └── scale_sweep_results.csv

figures/
├── behavioral_summary/
│   ├── learning_rate_by_prediction_error.png
│   ├── learning_rate_distribution.png
│   └── model_comparison_summary.png
├── model_performance/
└── parameter_exploration/
```

---

## Validation

Tests are located in `validation/` (for numbered tests) and `tests/` (for pytest):

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# Run with coverage
pytest --cov=nn4psych
```

---

## Reproducibility Checklist

- [ ] Set random seed in `config.py` (`DEFAULT_SEED = 42`)
- [ ] Use configuration files rather than hard-coded values
- [ ] Document model versions in output filenames
- [ ] Save experiment configurations alongside results
- [ ] Use consistent column naming from `COLUMN_NAMES`

---

## Common Issues & Solutions

### Missing Model Directory
```
No model_params directories found
```
**Solution**: Train models first or download pre-trained weights.

### Empty Output Files
```
No data extracted
```
**Solution**: Check model file format and architecture compatibility.

### Import Errors
```
ModuleNotFoundError: No module named 'nn4psych'
```
**Solution**: Install package with `pip install -e .` from project root.

---

## Next Steps

1. **Add Bayesian Model Fitting**: Implement PyMC fitting in `scripts/fitting/`
2. **Parameter Recovery**: Add validation tests for parameter recovery
3. **Group Analysis**: Compare models by clinical/behavioral groups
4. **Neural Analysis**: Add fixed-point analysis pipeline

---

## References

- Nassar et al. (2021) - Change-point and oddball learning paradigms
- Wang et al. (2018) - RNN actor-critic foundations
- Data Analysis Project Template v2.0

---

**Template Compliance**: Follows QUICK_START_TEMPLATE.md conventions for data analysis projects.
