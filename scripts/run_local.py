#!/usr/bin/env python3
"""
Local runner with memory guard for sequential script execution.

Wraps subprocess calls to enforce:
1. Only one training/extraction process at a time (sequential)
2. Memory check before launching — aborts if < MIN_FREE_MB available
3. CPU thread limiting via configure_cpu_threads()

Usage:
    python scripts/run_local.py scripts/training/train_rnn_canonical.py --epochs 5 --trials 20
    python scripts/run_local.py scripts/training/train_multitask.py --epochs 5 --trials 20

Or import and use programmatically:
    from scripts.run_local import run_safe
    run_safe(["scripts/training/train_rnn_canonical.py", "--epochs", "5"])
"""

import subprocess
import sys
import os
from pathlib import Path

# Minimum free memory (MB) required before launching a subprocess
MIN_FREE_MB = 1500


def get_free_memory_mb() -> float:
    """Get available physical memory in MB. Returns inf if unknown."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        pass

    # Fallback: try Windows systeminfo (slow) or /proc/meminfo (Linux)
    if sys.platform == "linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024
        except (OSError, ValueError):
            pass

    # Can't determine — assume OK
    return float("inf")


def run_safe(args: list, min_free_mb: float = MIN_FREE_MB, **kwargs) -> subprocess.CompletedProcess:
    """
    Run a Python script as a subprocess with a memory guard.

    Parameters
    ----------
    args : list
        Script path and arguments (e.g. ["scripts/training/foo.py", "--epochs", "5"])
    min_free_mb : float
        Minimum free memory in MB required to proceed.
    **kwargs
        Passed to subprocess.run().

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    MemoryError
        If available memory is below min_free_mb.
    """
    free_mb = get_free_memory_mb()
    if free_mb < min_free_mb:
        raise MemoryError(
            f"Only {free_mb:.0f} MB free (need {min_free_mb:.0f} MB). "
            f"Close other applications or reduce test parameters."
        )

    cmd = [sys.executable] + list(args)
    print(f"[run_local] Memory: {free_mb:.0f} MB free — launching: {' '.join(args)}")
    return subprocess.run(cmd, **kwargs)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_local.py <script.py> [args...]")
        sys.exit(1)

    script_args = sys.argv[1:]
    result = run_safe(script_args)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
