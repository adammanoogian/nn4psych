---
created: 2026-03-21T10:30
title: Explore SAC with continuous actions for cognitive tasks
area: training
files:
  - src/nn4psych/models/continuous_rnn.py
  - scripts/training/train_context_dm.py
---

## Problem

The current RL pipeline uses A2C with discrete actions (Categorical over 3 choices). Some cognitive tasks may benefit from continuous action spaces — graded confidence reports, continuous spatial positioning (e.g., the helicopter task bucket could move by a continuous displacement instead of left/right/stay), or probabilistic wagering tasks where the agent bets a continuous amount.

SAC (Soft Actor-Critic) is the standard algorithm for continuous control with maximum entropy regularization, which could produce more exploratory and potentially more human-like behavior than A2C's greedy policy gradient.

## Questions to Evaluate

1. Which cognitive tasks would benefit from continuous action spaces? (graded confidence, continuous positioning, wagering)
2. Does SAC's maximum entropy objective (alpha * H(pi)) produce behavior more similar to human exploration patterns than A2C?
3. How to adapt CleanRL's `sac_continuous_action.py` with our `ContinuousActorCritic` RNN backbone — the RNN is the feature extractor, SAC replaces the actor/critic heads and training loop
4. Does off-policy replay (SAC's replay buffer) improve sample efficiency for longer training runs?
5. Does the reparameterization trick + tanh squashing affect interpretability of the actor's decision process vs discrete Categorical?

## References

- CleanRL: https://github.com/vwxyzjn/cleanrl (sac_continuous_action.py)
- Spinning Up SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
- Haarnoja et al. 2018 "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"

## Solution

TBD — v2/exploratory. Not needed for current milestone (latent circuit + Bayesian fitting). Evaluate after Phase 5 completion.
