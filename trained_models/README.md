# Trained Models Directory

This directory contains trained RNN actor-critic model weights organized by purpose.

## Structure

```
trained_models/
├── checkpoints/       # Training checkpoints and hyperparameter sweeps
│   └── model_params_101000/
├── best_models/       # Best performing models for production use
│   └── heli_trained_rnn.pkl
├── weights/          # Extracted weight arrays
│   └── weights.npy
└── README.md
```

## Usage

### Loading Checkpoints
```python
from nn4psych.utils.io import load_model
from nn4psych.models import ActorCritic

model = load_model("trained_models/checkpoints/model.pth", ActorCritic)
```

### Loading Best Models
```python
import pickle

with open("trained_models/best_models/heli_trained_rnn.pkl", "rb") as f:
    model_data = pickle.load(f)
```

## Notes

- This directory is gitignored by default to avoid committing large model files
- Only commit model metadata or links to external storage (e.g., Hugging Face, Google Drive)
- For reproducibility, document model training parameters in commit messages or config files
