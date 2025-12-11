"""Neural network models for predictive inference tasks."""

from nn4psych.models.actor_critic import ActorCritic
from nn4psych.models.multitask_actor_critic import (
    MultiTaskActorCritic,
    PaddedMultiTaskActorCritic,
    TaskSpec,
)

__all__ = [
    "ActorCritic",
    "MultiTaskActorCritic",
    "PaddedMultiTaskActorCritic",
    "TaskSpec",
]
