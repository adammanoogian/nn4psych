# Architecture

**Analysis Date:** 2026-01-28

## Pattern Overview

**Overall:** Layered modular architecture with clear separation of concerns following scientific computing patterns.

**Key Characteristics:**
- Configuration-driven experiments (dataclass-based config objects)
- Single source of truth for models and utilities (no duplicate implementations)
- Gym-compatible environment abstraction for task definition
- Three-tier pipeline: Training → Behavior Extraction → Analysis
- PyTorch-based neural network models with actor-critic structure
- Support for both single-task and multi-task experiments

## Layers

**Models Layer:**
- Purpose: Define neural network architectures for reinforcement learning agents
- Location: `src/nn4psych/models/`
- Contains: ActorCritic class, MultiTaskActorCritic class
- Depends on: PyTorch (torch.nn)
- Used by: Training scripts, behavior extraction, analysis pipelines

**Task/Environment Layer:**
- Purpose: Provide gym-compatible task environments (helicopter bag positioning paradigm)
- Location: `envs/pie_environment.py`
- Contains: PIE_CP_OB_v2 environment class (change-point and oddball conditions)
- Depends on: Gymnasium, NumPy
- Used by: Training loops, behavior extraction

**Training Layer:**
- Purpose: Configuration management and parameter specification
- Location: `src/nn4psych/training/`
- Contains: ModelConfig, TaskConfig, TrainingConfig, ExperimentConfig (all dataclasses)
- Depends on: PyYAML, JSON (for serialization)
- Used by: Training scripts and experiment reproducibility

**Analysis Layer:**
- Purpose: Extract and analyze learned behaviors from trained models
- Location: `src/nn4psych/analysis/`
- Contains: extract_behavior (state extraction), hyperparameter analysis utilities
- Depends on: Models layer, metrics utilities
- Used by: Data pipeline scripts, research validation

**Utilities Layer:**
- Purpose: Shared helper functions for I/O, metrics, and visualization
- Location: `src/nn4psych/utils/`
- Contains: io.py (model save/load), metrics.py (learning rate extraction), plotting.py
- Depends on: PyTorch, NumPy, SciPy, Matplotlib
- Used by: All other layers

## Data Flow

**Training Flow:**

1. User loads/creates ExperimentConfig via `src/nn4psych/training/configs.py`
2. ActorCritic model instantiated from ModelConfig
3. PIE_CP_OB_v2 environment instantiated from TaskConfig
4. Training loop runs (in `scripts/training/` entry points):
   - Environment resets and generates observations
   - Model processes observations with RNN, outputs actions via actor, value via critic
   - Environment executes actions, returns rewards
   - Policy gradient updates applied (actor-critic algorithm)
   - Model saved at intervals (via `nn4psych.utils.io.save_model`)
5. Training produces checkpoint with model weights and metadata

**Behavior Extraction Flow:**

1. Trained model loaded from checkpoint via `load_model()`
2. `extract_behavior()` in `src/nn4psych/analysis/behavior.py` runs model on task for N epochs
3. Model hidden states collected across trials (states tuple: trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers)
4. State data pickled and saved to `data/processed/`

**Analysis Flow:**

1. Extracted state data loaded via `unpickle_state_vector()`
2. Learning rates computed per trial via `get_lrs_v2()` in `src/nn4psych/utils/metrics.py`
   - Calculates: prediction_error = |true_state - predicted_state|
   - learning_rate = |change_in_prediction| / |prediction_error|
3. Results aggregated across models and hyperparameters
4. Analysis outputs saved to `output/` and figures to `figures/`

**Data Paths:**
- Input: Raw task data in `data/raw/`
- Processing: Intermediate arrays in `data/intermediate/`
- Output: Behavioral CSVs in `output/behavioral_summary/`
- Artifacts: Figures in `figures/`, models in `trained_models/`

## Key Abstractions

**ActorCritic (src/nn4psych/models/actor_critic.py):**
- Purpose: RNN-based agent for predictive inference tasks
- Pattern: PyTorch nn.Module with shared RNN for feature extraction, separate actor/critic heads
- Constructor parameters control architecture: input_dim, hidden_dim, action_dim, gain (RNN scaling), noise, bias
- Methods:
  - `forward(x, hx)` → (actor_logits, critic_value, new_hx)
  - `get_initial_hidden()`, `reset_hidden()` for state management
  - `from_config(config)` class method for instantiation from ModelConfig

**PIE_CP_OB_v2 (envs/pie_environment.py):**
- Purpose: Gym-compatible environment for change-point (hazard-based) and oddball (random event) tasks
- Pattern: Stateful environment with reset() and step() methods
- Key state variables: bucket position (agent prediction), bag position (truth), helicopter position (hidden state when train_cond=False)
- Reward: Gaussian function centered at bag position, higher reward for closer predictions
- Context: One-hot encoded task condition (change-point or oddball) concatenated to observations

**ExperimentConfig (src/nn4psych/training/configs.py):**
- Purpose: Structured, serializable configuration for complete experiments
- Pattern: Nested dataclasses (ExperimentConfig → ModelConfig, TaskConfig, TrainingConfig)
- Serialization: YAML or JSON export/import via `to_yaml()`, `from_yaml()`, `to_json()`, `from_json()`
- Enables: Reproducible experiments, parameter sweeps, experiment documentation

**Behavior State Tuple (returned by extract_behavior):**
- Purpose: Complete record of agent performance across an epoch
- Structure: (trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers)
- Usage: Input to metrics functions for computing learning rates and prediction errors

## Entry Points

**Training Entry Points (scripts/training/):**
- `train_rnn_canonical.py`: Single-task training loop with command-line arg parsing
  - Trains ActorCritic on change-point or oddball task
  - Saves checkpoints during training
  - Command-line args: epochs, trials, gamma, rollout_size, hidden_dim, seed, etc.

- `train_multitask.py`: Multi-task training with shared RNN, task-specific heads
  - Trains on both change-point and oddball conditions in same epoch

- `examples/train_example.py`: Tutorial example
  - Demonstrates configuration-based setup
  - Minimal hyperparameters

**Analysis Entry Points (scripts/data_pipeline/):**
- `01_extract_model_behavior.py`: Loads models, extracts behavior states
- `02_compute_learning_metrics.py`: Computes learning rates from states
- `03_analyze_hyperparameter_sweeps.py`: Aggregates performance across parameters
- `04_visualize_behavioral_summary.py`: Plots behavior summaries
- `00_run_full_pipeline.py`: Orchestrates full analysis pipeline

**Validation Entry Points (scripts/analysis/):**
- `validate_nassar2021.py`: Validates model behavior against Nassar et al. 2021 human data
- `nassarfig6.py`: Reproduces Figure 6 comparisons

## Error Handling

**Strategy:** Exceptions propagate; validation at configuration/instantiation time.

**Patterns:**
- `load_model()` returns checkpoint dict if model_class is None (defensive fallback)
- `save_model()` prints success message; fails loudly on I/O errors
- `extract_behavior()` checks device and defaults to CPU if not specified
- Configuration classes use default values; no required fields
- Environment reset() can fail if trial count exceeded (raises implicitly via state checks)

## Cross-Cutting Concerns

**Logging:** Print-based logging in utility functions (e.g., "File saved: {filepath}"). No formal logging framework.

**Validation:**
- Learning rate clipping: `LR_CLIP_RANGE = (0.0, 1.0)` in `config.py`
- Performance thresholds: `PERFORMANCE_THRESHOLD = 10.0` for filtering models
- Prediction error filtering: `LR_PE_THRESHOLD = 20` for meaningful LR calculation

**Configuration:**
- Central `config.py` in project root defines all paths, parameters, column names
- `src/nn4psych/training/configs.py` provides dataclass-based runtime configuration
- Command-line args in training scripts override defaults
- YAML/JSON export enables experiment reproducibility

**Device Management:**
- Training scripts explicitly set device to 'cpu' (GPU support available but not default)
- `extract_behavior()` infers device from model parameters
- `load_model()` accepts optional device parameter, defaults to CPU

---

*Architecture analysis: 2026-01-28*
