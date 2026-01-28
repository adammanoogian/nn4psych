# Technology Stack

**Analysis Date:** 2026-01-28

## Languages

**Primary:**
- Python 3.8+ - Core application language for neural networks and scientific computing
- Supports: 3.8, 3.9, 3.10, 3.11 (as declared in `pyproject.toml`)

## Runtime

**Environment:**
- Python (CPython, recommended 3.10 as per `environment.yml`)
- PyTorch (GPU-capable, CPU fallback)
- Works with both CPU and CUDA devices (configurable via `device` in `TrainingConfig`)

**Package Manager:**
- pip (primary) - `requirements.txt` specifies core dependencies
- conda (alternative) - `environment.yml` provides Conda environment specification
- setuptools with pyproject.toml (modern packaging standard)

**Lockfile:**
- No explicit lockfile present; versions pinned in `requirements.txt` (>=constraints)
- `environment.yml` pins exact versions for reproducibility

## Frameworks

**Core ML/Neural Networks:**
- PyTorch 2.0+ - Deep learning framework for model implementation
  - Location: `src/nn4psych/models/actor_critic.py`, `src/nn4psych/models/multitask_actor_critic.py`
  - Used for: RNN, Actor-Critic architecture, gradient-based training

**Environment/Task Frameworks:**
- Gymnasium 0.29+ - Standard RL environment interface (successor to OpenAI Gym)
- gym 0.26+ - Kept for legacy compatibility
- neurogym 0.0.1+ - Neuroscience-based behavioral tasks (NeuroGym)
  - Optional dependency; provides predefined neuroscience tasks
  - Integrated via `envs/neurogym_wrapper.py` with graceful fallback

**Scientific Computing:**
- NumPy 1.24+ - Array operations and numerical computing
- SciPy 1.10+ - Advanced scientific functions (signal processing, stats)
- Pandas 2.0+ - Data manipulation and analysis
- scikit-learn 1.2+ - Machine learning utilities (metrics, preprocessing)

**Visualization:**
- Matplotlib 3.7+ - 2D plotting library
- Seaborn 0.12+ - Statistical data visualization (built on matplotlib)

**Configuration & Serialization:**
- PyYAML 6.0+ - YAML config file parsing and generation
- JSON (built-in) - JSON serialization for results/configs
  - Used in `src/nn4psych/training/configs.py` for ExperimentConfig

**Testing:**
- pytest 7.0+ - Test framework and runner
- pytest-cov 4.0+ - Code coverage tracking for pytest
- Configuration: `pytest.ini` with coverage reporting

**Code Quality:**
- black 23.0+ - Code formatter (line-length: 100)
  - Configuration in `pyproject.toml` and `.pre-commit-config.yaml`
- flake8 6.0+ - Linter (extends ignore: E203, W503)
  - Configuration in `.pre-commit-config.yaml`
- mypy 1.0+ - Static type checker
  - Configuration in `pyproject.toml`
- isort 5.13.2 - Import statement organizer (black profile, 100 char line)
  - Configured in `.pre-commit-config.yaml`

**Pre-commit Hooks:**
- pre-commit 2.13+ - Git hooks framework
  - Config: `.pre-commit-config.yaml`
  - Hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-large-files (1MB limit), check-json, check-toml, debug-statements

**Optional/Advanced Frameworks:**
- PyMC 5.0+ - Bayesian modeling (optional dependency `[bayesian]`)
  - Location: `scripts/analysis/bayesian/` contains Bayesian model implementations
- ArviZ 0.15+ - Bayesian visualization and diagnostics
  - Used with PyMC for posterior analysis
- JAX 0.4+ - Functional ML framework (legacy support, not actively maintained)
  - Original code in `archive/v0/`, `archive/v1/`

**Progress/Utilities:**
- tqdm 4.60+ - Progress bars for long-running training loops
- statsmodels 0.14+ - Statistical models and tests (via environment.yml)

## Key Dependencies

**Critical (Direct Imports in Core Code):**
- torch - All model implementations depend on PyTorch
  - `src/nn4psych/models/actor_critic.py` - RNN backbone
  - `src/nn4psych/models/multitask_actor_critic.py` - Multi-task variant
  - `src/nn4psych/training/` - Training loops
  - `src/nn4psych/utils/io.py` - Model serialization

- numpy - Scientific computing in utils and analysis
  - `src/nn4psych/utils/metrics.py` - Statistical calculations
  - `src/nn4psych/utils/plotting.py` - Data preparation
  - `src/nn4psych/analysis/behavior.py` - Behavioral analysis

- gymnasium / gym - Environment interface
  - `envs/pie_environment.py` - Implements Gymnasium-compliant PIE_CP_OB_v2
  - `envs/neurogym_wrapper.py` - Wraps neurogym tasks to Gymnasium interface

**Scientific Infrastructure:**
- scipy - Signal processing and statistics
  - `src/nn4psych/utils/metrics.py` uses scipy.ndimage (smoothing), scipy.stats
  - Model fitting and statistical tests in analysis scripts

- matplotlib - Plotting in analysis scripts
  - `src/nn4psych/utils/plotting.py` - Behavioral visualization
  - `src/nn4psych/analysis/hyperparams.py` - Hyperparameter sweep plots

- scikit-learn - Machine learning utilities
  - Model evaluation metrics in analysis workflows

- pyyaml - Configuration file handling
  - `src/nn4psych/training/configs.py` - YAML load/save in ExperimentConfig

**Optional but Integrated:**
- neurogym - Predefined behavioral tasks
  - Optional import in `envs/__init__.py` with graceful degradation
  - `envs/neurogym_wrapper.py` adapts neurogym tasks to nn4psych interface

## Configuration

**Environment Setup:**

Development with Conda:
```bash
conda env create -f environment.yml
conda activate nn4psych
```

Development with pip:
```bash
pip install -e ".[dev]"
pip install -r requirements.txt
```

**Build System:**
- setuptools with pyproject.toml
- Package layout: `src/nn4psych/` (src layout, PEP 420)
- Entry point configuration in `pyproject.toml` `[project.urls]`

**Configuration Management:**
- YAML configs in `src/nn4psych/configs/` (e.g., `default.yaml`)
- Dataclass-based config in `src/nn4psych/training/configs.py`:
  - `ModelConfig` - Architecture parameters
  - `TaskConfig` - Environment parameters
  - `TrainingConfig` - Optimizer/learning parameters
  - `ExperimentConfig` - Unified experiment specification

**Runtime Configuration:**
- Device selection: CPU or CUDA (in `TrainingConfig.device`)
- Seed management: Controlled via `TrainingConfig.seed`
- Save directory: Configurable via `TrainingConfig.save_dir`

**Development Configuration:**

Code formatting (black):
- Line length: 100 characters
- Target version: Python 3.8+
- Configuration in `pyproject.toml`

Linting (flake8):
- Max line length: 100
- Ignored rules: E203 (whitespace before colon), W503 (line break before operator)

Type checking (mypy):
- Python version: 3.8
- Settings: warn_return_any, warn_unused_configs, ignore_missing_imports

Import sorting (isort):
- Profile: black
- Line length: 100

## Platform Requirements

**Development:**
- Python 3.8 or higher
- pip or conda package manager
- Optional: CUDA toolkit (for GPU acceleration with PyTorch)
- Optional: git (for pre-commit hooks)

**Production/Training:**
- Python 3.8+
- PyTorch with CPU or GPU support
- Sufficient RAM (varies by model size and batch size; default ~2-4GB)
- Optional: CUDA 11.x or 12.x compatible GPU (for GPU training)

**Data & Output:**
- Disk space for:
  - Training data: `data/` directory (varies)
  - Model checkpoints: `trained_models/` (varies by number of epochs/models)
  - Analysis outputs: `output/` and `figures/` directories

---

*Stack analysis: 2026-01-28*
