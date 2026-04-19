# CS 7641 — RL Unit Cheat Sheet

> Covers: MDPs · Policy Evaluation · RL Algorithms · Epsilon-Greedy · Game Theory · Minimax · Repeated Games

---

## 1. Markov Decision Processes (MDPs)

### Core Components
| Symbol | Meaning |
|---|---|
| S | Set of states |
| A | Set of actions |
| T(s, a, s') | Transition probability: P(next state = s' \| state = s, action = a) |
| R(s, a, s') | Reward received when transitioning s → s' via action a |
| γ ∈ [0,1) | Discount factor — how much future rewards are worth today |
| π(a\|s) | Policy — probability of taking action a in state s |

### Transition Function Rules
- T(s, a, s') ≥ 0 for all s, a, s' (non-negative)
- Σ_{s'} T(s, a, s') = 1 for every (s, a) pair (must sum to exactly 1)
- In a **deterministic** MDP: T(s, a, s*) = 1 for exactly one s*, and 0 for all others
- Changing T **does** affect the optimal policy

### Bellman Expectation Equation

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} T(s,a,s') \big[ R(s,a,s') + \gamma \times V^\pi(s') \big]$$

- $V^\pi(s)$ is the expected discounted return starting from state $s$ under policy $\pi$
- It relates the value of a state to the values of its **successor states**

### Bellman Optimality Equation

$$V^*(s) = \max_a \sum_{s'} T(s,a,s') \big[ R(s,a,s') + \gamma \times V^*(s') \big]$$

---

## 2. Policy Evaluation (Bellman Update)

Used to compute $V^\pi$ for a **fixed policy**. Repeatedly apply:

$$V_{new}(s) = \sum_a \pi(a|s) \sum_{s'} T(s,a,s') \big[ R(s,a,s') + \gamma \times V_{old}(s') \big]$$

### Synchronous vs Asynchronous
| Type | Rule |
|---|---|
| Synchronous | Use ALL V_old values on the right-hand side. Compute all V_new simultaneously. |
| Asynchronous | Use the latest available value (updates propagate immediately within one sweep). |

### Worked Example Pattern (Molebert-style)
Given: $\pi(Dig) = 0.85$, $\pi(Nap) = 0.15$, $\gamma = 0.75$, all $V = 0$ initially.

**Step 1:** Compute from the terminal state backwards.

**Step 2:** For each state $s$:
$$V_{new}(s) = \pi(Dig) \times [R_{dig} + \gamma \times V_{old}(next_{dig})] + \pi(Nap) \times [R_{nap} + \gamma \times V_{old}(s)]$$

**Example —** $V(B3)$, $R_{dig}=5$, $R_{nap}=0.2$:
$$= 0.85 \times (5 + 0.75\times 0) + 0.15 \times (0.2 + 0.75\times 0) = 4.25 + 0.03 = 4.28$$

---

## 3. RL Algorithms — SARSA vs Q-Learning

### Update Rules
| Algorithm | Type | Update Formula |
|---|---|---|
| SARSA | On-policy | $Q(s,a) \leftarrow Q(s,a) + \alpha [ r + \gamma Q(s',a') - Q(s,a) ]$ |
| Q-Learning | Off-policy | $Q(s,a) \leftarrow Q(s,a) + \alpha [ r + \gamma \max_{a'} Q(s',a') - Q(s,a) ]$ |

where $a'$ in SARSA is the **actual next action taken** (including exploration).

### Key Differences

| Property | SARSA (on-policy) | Q-Learning (off-policy) |
|---|---|---|
| Learns | The policy being followed (behavior policy) | The optimal policy regardless of behavior |
| Q-values represent | Return under behavior policy (with exploration) | Return under optimal policy |
| Risk behavior | More conservative in risky areas | More aggressive (assumes optimal future) |
| Next action used | Actual action taken (ε-greedy with exploration) | Max over all possible next actions |
| Cliff-walking | Learns safer path (away from cliff edge) | Learns shortest path (near cliff edge) |

### Quick Decision Rule
- **Did the agent actually take that next action?** → SARSA
- **Did it take the best possible next action?** → Q-Learning

---

## 4. Epsilon-Greedy Action Selection

### Setup
- $\varepsilon$ = exploration rate ($0 \leq \varepsilon \leq 1$)
- $k_{total}$ = total number of actions
- $k_{greedy}$ = number of greedy actions (those with maximum Q-value)

### Probability Formulas

$$P(\text{greedy action}) = \frac{1 - \varepsilon}{k_{greedy}} + \frac{\varepsilon}{k_{total}}$$
$$P(\text{non-greedy action}) = \frac{\varepsilon}{k_{total}}$$

### Quick Reference Table (4 total actions, 2 greedy)

| $\varepsilon$ | $P$(each greedy) | $P$(each non-greedy) |
|---|---|---|
| 0.0 | 0.500 | 0.000 |
| 0.1 | 0.475 | 0.025 |
| 0.2 | 0.450 | 0.050 |
| 0.4 | 0.400 | 0.100 |
| 0.6 | 0.350 | 0.150 |
| 1.0 | 0.250 | 0.250 |

> As ε increases: greedy actions get LESS probability, non-greedy get MORE. At ε=1, all actions equally likely.

### Common Mistake
Do NOT weight by Q-value magnitudes. Once you know which actions are greedy vs non-greedy, Q-values are irrelevant to the probability formula. Only ε and counts matter.

---

## 5. Model-Free vs Model-Based RL

| Property | Model-Free (e.g. Q-Learning, SARSA) | Model-Based (e.g. Dyna-Q) |
|---|---|---|
| How it learns | Directly from real experience | Builds a model of T(s,a,s') and R, then plans |
| Needs env model? | No | Yes (learned or given) |
| Sample efficiency | Lower — needs more real interactions | Higher — can simulate from model |
| Compute cost | Lower | Higher (planning/simulation step) |
| Risk | Can't plan ahead | Poor model → poor performance |
| Model reusability | N/A | Same model can be reused for new reward functions |

> **Memory hook:** Model-free = trial-and-error chef. Model-based = chef who runs mental simulations first.

---

## 6. Game Theory — Single-Stage Games

### Minimax Strategy (Zero-Sum Games)
Huey maximises, Dewey/opponents minimise Huey's payoff.

**Step 1:** For each of Huey's options (rows), find the MINIMUM payoff across all opponent columns.

**Step 2:** Choose the row with the MAXIMUM of those minimums.

$$\text{Minimax} = \arg\max_{row} \min_{col} \text{Payoff}(row, col)$$

### Mixed Strategy Worst-Case
When mixing $p\%$ option X and $(1-p)\%$ option Y:

**Step 1:** For EACH opponent column $j$, compute the blended payoff:
$$\text{Blended}(j) = p \times \text{Payoff}(X, j) + (1-p) \times \text{Payoff}(Y, j)$$

**Step 2:** Worst-case = $\min_j \text{Blended}(j)$

> **NEVER:** $\min(X) \times p + \min(Y) \times (1-p)$ — this is wrong because opponent picks ONE scenario, not separate worst-cases per option.

### Nash Equilibrium
A strategy profile where no player can improve by unilaterally changing their strategy.

### Dominant Strategy
An action that is always best regardless of what the opponent does.

### Zero-Sum Game
One player's gain = the other's loss in EVERY outcome. If any outcome has both players losing (or both gaining), it is NOT zero-sum.

---

## 7. Repeated Games & Cooperation

### Folk Theorem
In an infinitely repeated game with discounting, **any payoff above the minimax value** can be sustained as a Nash equilibrium — provided players are patient enough (discount factor $\delta$ is sufficiently high).

$$\text{Condition for cooperation: } \delta \geq \frac{\text{Temptation gain}}{\text{Temptation gain} + \text{Punishment loss}}$$

> Heavy discounting (low δ, caring little about the future) = cooperation breaks down.
> Patient players (high δ) = cooperation sustainable.

### Common Strategies

| Strategy | Rule | Strength | Weakness |
|---|---|---|---|
| Grim Trigger | Cooperate until opponent defects; then defect forever | Strong deterrent | No forgiveness — one mistake ends cooperation permanently |
| Tit-for-Tat (TfT) | Copy opponent's last action | Simple, provokable, forgiving | Can't reform an always-defect opponent |
| Pavlov (Win-Stay, Lose-Shift) | Repeat last action if it paid off; switch if it didn't | Self-correcting | Gets trapped in switching cycle against always-defect |
| Always Defect | Defect every round | Exploits cooperators | Gets punished by TfT and Grim Trigger |

### Against an Always-Defect Opponent
| Your strategy | What happens |
|---|---|
| Tit-for-Tat | Punishes each time but Puddles never reforms — TfT defects forever |
| Pavlov | Trapped in cooperation→defect→cooperation cycle |
| Grim Trigger | Defects permanently after first defection |
| Folk Theorem | Does NOT guarantee cooperation with irrational/committed defectors |

### Subgame Perfect Equilibrium (SPE)
A strategy profile that is a Nash equilibrium in **every subgame** (including off-path subgames). Key property: **no empty threats** — every threatened punishment must be credible (actually rational to carry out).

---

## 8. Two-Stage / Extensive Form Games

### Backward Induction
To find SPE: solve the last stage first, then work backwards.

```
Step 1: Solve the final subgame → find its Nash equilibrium.
Step 2: Substitute that equilibrium payoff into the prior stage.
Step 3: Solve the prior stage given those expected payoffs.
```

### Credible Threats
A threat is credible only if the threatening player would actually benefit from carrying it out (SPE condition). Non-credible threats don't deter rational opponents.

### Zero-Sum vs Non-Zero-Sum Check
Ask: "Is there any outcome where both players gain or both lose?"
- If YES → not zero-sum
- If NO (one gains exactly what the other loses in EVERY cell) → zero-sum

---

## 9. Quick Formula Reference Card

### Bellman Policy Evaluation (1 step)
$$V_{new}(s) = \sum_a \pi(a|s) \big[ R(s,a) + \gamma \times V_{old}(\text{next\_state}(s,a)) \big]$$

### Epsilon-Greedy Probabilities
$$P(\text{greedy}) = \frac{1 - \varepsilon}{k_{greedy}} + frac{\varepsilon}{k_{total}}$$
$$P(\text{non-greedy}) = \frac{\varepsilon}{k_{total}}$$

### Minimax (pure strategy)
$$\text{Best spot} = \arg\max_i \min_j \text{Payoff}(i, j)$$

### Mixed Strategy Worst-Case
$$\text{Worst-case} = \min_j \big[ p \times \text{Payoff}(X,j) + (1-p) \times \text{Payoff}(Y,j) \big]$$

### SARSA Update
$$Q(s,a) \leftarrow Q(s,a) + \alpha \big[ r + \gamma \times Q(s', a') - Q(s,a) \big] \quad \text{← $a'$ is actual next action}$$

### Q-Learning Update
$$Q(s,a) \leftarrow Q(s,a) + \alpha \big[ r + \gamma \times \max_{a'} Q(s',a') - Q(s,a) \big]$$

---

## 10. Trap Door — Common Exam Mistakes

| Trap | Correct understanding |
|---|---|
| "Sum of transition probabilities can exceed 1" | FALSE — must sum to exactly 1 |
| "Deterministic MDP → T(s,a,s') = 1 for ALL triples" | FALSE — only 1 for the specific next state |
| "SARSA learns the optimal policy" | FALSE — SARSA learns the behavior (exploratory) policy |
| "Model-free is more sample efficient" | FALSE — model-based is more sample efficient (can simulate) |
| "Higher ε → greedy actions more likely" | FALSE — higher ε = more exploration = greedy actions less likely |
| "Heavy discounting helps cooperation" | FALSE — heavy discounting kills cooperation (future punishments worthless) |
| "Mixed strategy worst-case = p×min(X) + (1−p)×min(Y)" | FALSE — blend first per scenario, then take min across scenarios |
| "Non-zero-sum game if one player gains" | FALSE — zero-sum requires one's gain = other's loss in EVERY outcome |
| "Folk Theorem guarantees cooperation regardless of opponent" | FALSE — requires rational, responsive players; fails vs committed always-defect |
| "SPE requires mixed strategies in every subgame" | FALSE — SPE can be in pure strategies |

---

## 11. Negative Marking Strategy

Canvas multi-select scoring:
$$\text{Score} = \frac{\text{correct\_selected} - \text{wrong\_selected}}{\text{total\_correct}} \times \text{points}$$

**Rule:** Each wrong selection cancels one correct selection.

| Confidence | Action |
|---|---|
| Very confident (>90%) | Select it |
| Somewhat confident (~60%) | Skip — risk not worth it |
| Unsure (<50%) | Definitely skip |

Leaving an option blank costs you (1/n_correct × points) in missed upside.
Selecting it wrong costs you the same amount as a penalty PLUS you lose the upside.
→ Only select what you are sure about.
