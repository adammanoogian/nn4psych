"""
CPU thread and resource management for training.

Provides automatic detection of execution context (local vs cluster/batch)
and sets appropriate resource limits.
"""

import os
import torch

# Default thread limit for local (interactive) runs
LOCAL_THREAD_LIMIT = 4


def is_cluster_or_batch() -> bool:
    """
    Detect whether we are running on a SLURM cluster or in a batch context.

    Checks for SLURM environment variables and common batch indicators.

    Returns
    -------
    bool
        True if running on a cluster or in a non-interactive batch job.
    """
    slurm_vars = ["SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CLUSTER_NAME"]
    if any(os.environ.get(v) for v in slurm_vars):
        return True

    # PBS/Torque
    if os.environ.get("PBS_JOBID"):
        return True

    # SGE
    if os.environ.get("JOB_ID") and os.environ.get("SGE_ROOT"):
        return True

    # LSF
    if os.environ.get("LSB_JOBID"):
        return True

    return False


def configure_cpu_threads(local_limit: int = LOCAL_THREAD_LIMIT) -> int:
    """
    Set CPU thread limits based on execution context.

    On local/interactive runs, limits PyTorch to `local_limit` threads
    to avoid saturating the machine. On cluster/batch runs, leaves
    threading unconstrained (uses all allocated cores).

    Parameters
    ----------
    local_limit : int
        Maximum threads for local runs. Default is 4.

    Returns
    -------
    int
        The number of threads that were set (or 0 if unconstrained).
    """
    if is_cluster_or_batch():
        # On cluster: respect SLURM_CPUS_PER_TASK if set, otherwise no limit
        cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpus:
            n = int(cpus)
            torch.set_num_threads(n)
            torch.set_num_interop_threads(max(1, n // 2))
            return n
        return 0
    else:
        torch.set_num_threads(local_limit)
        torch.set_num_interop_threads(max(1, local_limit // 2))
        return local_limit
