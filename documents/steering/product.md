---
inclusion: always
---

# Product — CS7641 RL Spring 2026

## Objective

Produce a reproducible, assignment-complete analysis pipeline and report for the CS7641 Spring 2026 Reinforcement Learning assignment using Blackjack-v1 and CartPole-v1. Work must support both the analysis report and the reproducibility sheet.

## Assignment Contract

Derived from:
- `documents/canvas/assignment/md/RL_Report_Spring_2026_v1-2.md`
- `documents/canvas/assignment/md/RL_Report_Spring_2026_FAQ_v2.md`

Requirements:
1. Solve **Blackjack-v1** and **CartPole-v1** with VI, PI, SARSA, and Q-Learning.
2. CartPole requires state discretization for VI/PI and tabular methods.
3. Validate **at least 2 hyperparameters** per model with a staged search protocol.
4. Average all model/config comparisons over **5 independent random seeds**; report exact seed list and total wall-clock.
5. Report mean ± variability bands (IQR) — no single-seed results.
6. Hypothesis-driven analysis — not a result dump.
7. 8-page report (IEEE Conference template, Overleaf) + reproducibility sheet.
8. Submit READ-ONLY Overleaf link + GitHub commit hash.
9. Include **mandatory citations**: Sutton & Barto 2018 (Blackjack), Barto et al. 1983 (CartPole).
10. Include **AI Use Statement** at the end of the report listing tools used and what they assisted with.

## Environment Scope

| Environment | Type | State Space | Action Space | Notes |
|---|---|---|---|---|
| `Blackjack-v1` | Discrete, stochastic | (player sum, dealer card, usable ace) | hit / stick | Inherently tabular |
| `CartPole-v1` | Continuous, deterministic | (x, ẋ, θ, θ̇) | left / right | Requires discretization |

## Algorithm Scope

| Algorithm | Type | Environments |
|---|---|---|
| Value Iteration (VI) | Model-based DP | Both |
| Policy Iteration (PI) | Model-based DP | Both |
| SARSA | Model-free, on-policy | Both |
| Q-Learning | Model-free, off-policy | Both |
| DQN + Rainbow ablation | Deep RL (**optional extra credit — up to 5 pts**) | CartPole only |

## Libraries

- `bettermdptools` — MDP utilities, VI/PI wrappers
- `pymdptoolbox` — supplemental MDP toolbox
- `gymnasium` — environment interaction
- `numpy`, `pandas` — computation and metrics
- `matplotlib` — all figures
- `tqdm` — progress bars
