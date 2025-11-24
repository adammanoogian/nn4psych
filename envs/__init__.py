"""
Standalone environment module for predictive inference tasks.

This module provides the PIE_CP_OB_v2 environment that can be used
by both RNN training code and Bayesian model fitting scripts.
"""

from .pie_environment import PIE_CP_OB_v2

__all__ = ["PIE_CP_OB_v2"]
