"""
Standalone environment module for predictive inference tasks.

This module provides environments for training RNN models:
- PIE_CP_OB_v2: Predictive inference environment (change-point / oddball)
- NeuroGym wrappers: Adapted neurogym tasks for multi-task learning
"""

from .pie_environment import PIE_CP_OB_v2

# Import neurogym wrappers (optional, requires neurogym package)
try:
    from .neurogym_wrapper import (
        NeurogymWrapper,
        DawTwoStepWrapper,
        SingleContextDecisionMakingWrapper,
        PerceptualDecisionMakingWrapper,
        create_neurogym_env,
        get_neurogym_task_spec,
        list_available_neurogym_tasks,
        NEUROGYM_TASK_DEFAULTS,
        NEUROGYM_AVAILABLE,
    )
except ImportError:
    NEUROGYM_AVAILABLE = False
    NeurogymWrapper = None
    DawTwoStepWrapper = None
    SingleContextDecisionMakingWrapper = None
    PerceptualDecisionMakingWrapper = None
    create_neurogym_env = None
    get_neurogym_task_spec = None
    list_available_neurogym_tasks = None
    NEUROGYM_TASK_DEFAULTS = {}

__all__ = [
    "PIE_CP_OB_v2",
    "NeurogymWrapper",
    "DawTwoStepWrapper",
    "SingleContextDecisionMakingWrapper",
    "PerceptualDecisionMakingWrapper",
    "create_neurogym_env",
    "get_neurogym_task_spec",
    "list_available_neurogym_tasks",
    "NEUROGYM_TASK_DEFAULTS",
    "NEUROGYM_AVAILABLE",
]
