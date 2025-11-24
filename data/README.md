# Data Directory

Organized data storage for the nn4psych project.

## Structure

```
data/
├── raw/                    # Original, immutable data
│   ├── nassar2021/        # Nassar et al. 2021 reference data
│   └── fig2_values/       # Figure 2 reference values
├── processed/              # Processed and cleaned datasets
│   ├── rnn_behav/         # RNN behavioral data
│   ├── bayesian_models/   # Bayesian model outputs
│   └── pt_rnn_context/    # PyTorch RNN context data
├── intermediate/           # Intermediate computation arrays
│   ├── activity_*.npy     # Neural activity arrays
│   ├── history_*.npy      # Training history arrays
│   ├── states_*.npy       # State arrays
│   └── W*.npy            # Weight matrices
└── README.md
```

## Data Management Principles

1. **Raw data is read-only**: Never modify files in `raw/`
2. **Processed data is reproducible**: Can be regenerated from raw data + scripts
3. **Intermediate data is temporary**: Large arrays that can be recomputed
4. **Use relative paths**: Import from `config.py` for consistency

## Usage

```python
from config import DATA_DIR, RAW_DATA_DIR

# Load raw data
raw_path = RAW_DATA_DIR / "nassar2021" / "data.mat"

# Save processed data
processed_path = DATA_DIR / "processed" / "rnn_behav" / "results.pkl"
```

## Size Guidelines

- Raw data: ~100MB (reference datasets)
- Processed data: ~500MB (cleaned + aggregated)
- Intermediate data: ~2GB (large arrays, can be gitignored)
