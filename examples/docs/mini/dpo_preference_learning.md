# Direct Preference Optimization

Direct Preference Optimization (DPO) trains a language model to prefer one
response over another without explicitly modeling a reward function. It was
introduced as a simpler alternative to reinforcement learning from human
feedback (RLHF) and has become the default fine-tuning recipe for
preference data in many open-source projects.

## The Dataset

A DPO dataset consists of triples: a prompt, a chosen response, and a
rejected response. The chosen response is one that a human labeler (or a
strong judge model) preferred. The rejected response is a plausible
alternative that was dispreferred. Quality of this data dominates every
other hyperparameter: if the two responses don't meaningfully differ along
the axis the model should learn, training does nothing useful.

## The Loss

DPO's loss function derives from the reward-model-free reformulation of
the PPO objective used in RLHF. Given a policy model and a reference
model, DPO minimizes the log-sigmoid of a margin: the difference between
the log-likelihood ratios of chosen and rejected responses, each measured
against the reference model. The temperature of this margin is controlled
by a single scalar beta, typically between 0.01 and 0.5. Higher beta means
sharper preference — the model is penalized more harshly for preferring
rejected over chosen.

## Reference Model

The reference model anchors the policy and keeps it from drifting too far
from a known-good distribution. In practice this is the supervised
fine-tuned (SFT) model that preceded DPO training. Some implementations
drop the reference model entirely for efficiency (ORPO does this), at the
cost of reduced stability.

## Practical Tips

Three tips recur across well-run DPO projects. First, run a solid SFT
pass before DPO; DPO cannot fix a model that cannot follow the format.
Second, cap response lengths, because long chosen responses tend to win
due to longer logprob sums, not real quality. Third, watch the reward
margin during training: if it collapses toward zero, the dataset is too
easy or beta is too low.

## Known Failure Modes

DPO tends to over-optimize toward the dispreferred direction, sometimes
producing responses that explicitly hedge, refuse, or "look safe" even
when a direct answer was chosen. Calibrating beta, filtering low-margin
pairs, and adding a small SFT regularization term all help. When none of
these work, the data itself is usually the cause; rechecking labeler
agreement often reveals the real issue.
