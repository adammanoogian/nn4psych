# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CONTRIBUTING.md with comprehensive development guidelines
- CHANGELOG.md for tracking version history
- Documentation index at docs/README.md

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

## [0.2.0] - 2025-11-19

### Added
- **src/ Layout**: Adopted standard Python src/ layout for better package organization
  - Moved nn4psych package to src/nn4psych/
  - Updated pyproject.toml to use src/ directory
  - Eliminates naming confusion between project root and package

- **Standalone Environment Module**: Extracted environment to top-level
  - Created envs/ directory at project root
  - Moved PIE_CP_OB_v2 from nn4psych/env/ to envs/
  - Environment now importable by RNN training, Bayesian models, and external projects

- **Organized Model Storage**: Created trained_models/ directory structure
  - trained_models/checkpoints/ for training checkpoints
  - trained_models/best_models/ for best performing models
  - trained_models/weights/ for weight arrays
  - Moved model_params_101000/, heli_trained_rnn.pkl, weights.npy to organized locations

- **Structured Data Organization**: Reorganized data/ directory
  - data/raw/ for immutable source data
  - data/processed/ for cleaned datasets
  - data/intermediate/ for temporary computation arrays
  - Added data/README.md with management guidelines

- **Organized Notebooks**: Created notebook structure
  - notebooks/exploratory/ for EDA notebooks
  - notebooks/tutorials/ for how-to guides
  - Added notebooks/README.md with best practices

- **Comprehensive Documentation**: 7 targeted README files
  - Root README.md with complete structure overview
  - docs/ANALYSIS_PIPELINE.md for pipeline documentation
  - data/README.md for data management
  - notebooks/README.md for notebook guidelines
  - trained_models/README.md for model storage
  - scripts/fitting/README.md for fitting workflows
  - archive/README.md for legacy code guide

- **Enhanced .gitignore**: Comprehensive rules for new structure
  - Proper exclusions for trained_models/, data/intermediate/
  - .gitkeep files to preserve empty directories
  - Organized by category (Python, IDE, Data, etc.)

### Changed
- **Import Paths**: Updated all imports for new structure
  - Changed from `nn4psych.env` to `envs` (12 files updated)
  - Package imports remain unchanged due to src/ layout

- **Configuration**: Updated config.py with new directory paths
  - Added SRC_DIR, ENVIRONMENT_DIR, TRAINED_MODELS_DIR
  - Added PROCESSED_DATA_DIR, INTERMEDIATE_DATA_DIR
  - Added NOTEBOOKS_DIR

- **Bayesian Models Location**: Moved from package to scripts
  - Moved from nn4psych/models/bayesian/ to scripts/analysis/bayesian/
  - Fixed dependencies to use nn4psych.utils.metrics
  - Updated fitting scripts to import from new location

- **Plot Organization**: Consolidated output structure
  - Moved legacy /plots to standardized figures/ structure
  - Updated all scripts to output to figures/ subdirectories
  - Organized by type: behavioral_summary/, model_performance/, parameter_exploration/

### Removed
- **Cleaned Up Root**: Removed unnecessary files
  - Removed root __init__.py
  - Removed cleanRL directory (~240MB of unused code)
  - Removed empty normative_model/ directory

- **Legacy Structure**: Archived old organization patterns
  - Previous flat structure moved to archive with migration guide
  - Removed duplicate cleanRL entries from .gitignore

### Fixed
- Bayesian model import dependencies (old utils → nn4psych.utils.metrics)
- Environment import paths across entire codebase
- Hardcoded data paths in pyem_models.py (added NOTE comments)

## [0.1.0] - 2024-11-21

### Added
- Initial modular package structure
- Actor-Critic RNN model implementation
- PIE_CP_OB_v2 environment for change-point and oddball tasks
- Data pipeline scripts (00-03)
- Analysis utilities for behavior and hyperparameters
- Configuration system with YAML support
- Testing infrastructure (pytest)
- Package installation via pyproject.toml

### Changed
- Consolidated 8 duplicate ActorCritic implementations into single source
- Unified 5 analyze_hyperparams_*.py scripts into single parameterized class
- Centralized configuration in config.py

### Deprecated
- Legacy v0, v1, v2 implementations (moved to archive/)
- Old utils.py and utils_funcs.py (replaced by nn4psych/utils/)

---

## Version History Summary

- **v0.2.0** (2025-11-19): Major reorganization - src/ layout, extracted environment, organized storage
- **v0.1.0** (2024-11-21): Initial modular package release

---

## How to Update This File

When making changes:

1. Add entries to **[Unreleased]** section immediately
2. Categorize changes: Added, Changed, Deprecated, Removed, Fixed, Security
3. When releasing, move [Unreleased] to new version section
4. Follow format: `- Brief description [#issue-number if applicable]`

Example:
```markdown
### Added
- New feature for X that does Y [#123]
```
