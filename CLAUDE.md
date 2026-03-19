# Project Instructions

## Local Execution Constraints

This machine has 16GB RAM. When running training or extraction scripts locally:

1. **Run training scripts ONE AT A TIME.** Never launch multiple `python scripts/training/*.py` processes in parallel. Each PyTorch process uses ~300-400MB.

2. **Use small test parameters for smoke tests:**
   - `--epochs 3 --trials 10 --maxt 30` for train_rnn_canonical.py
   - `--epochs 3 --trials 10` for train_multitask.py
   - `--epochs 3 --trials 10 --extract_epochs 1 --extract_trials 10` for train_context_dm.py

3. **Wait for each process to complete before starting the next.** Check `echo $?` for exit code.

4. **Use `scripts/run_local.py` as a wrapper** for memory-guarded execution:
   ```bash
   python scripts/run_local.py scripts/training/train_rnn_canonical.py --epochs 3 --trials 10
   ```

## Python Environment

Use the conda environment for all execution:
```
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe
```

## Matplotlib

All scripts must use `matplotlib.use('Agg')` before importing pyplot. Never call `plt.show()`.
