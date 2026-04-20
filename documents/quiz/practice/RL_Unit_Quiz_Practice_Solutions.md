# RL Unit Quiz Practice — Solutions
## CS 7641: Machine Learning

> DO NOT DISTRIBUTE OUTSIDE OF CS7641

---

## Question 1 — Markov Decision Processes

### Part 1. Policies in Reinforcement Learning (MCMA)

**Correct Answers: A, B, D**

| Option | Verdict | Explanation |
|---|---|---|
| A | TRUE | By definition, a deterministic policy maps each state to a single action with probability 1. |
| B | TRUE | Stochastic policies assign a distribution over actions, allowing randomness. |
| C | FALSE | Some MDPs have multiple optimal policies that yield the same expected return. |
| D | TRUE | Policy evaluation involves computing the state-value or action-value functions for the policy. |
| E | FALSE | If the policy is already optimal, the improvement step will yield the same policy — not a strictly better one. |
| F | FALSE | Random policies do not maximize expected return and are not generally optimal. |

---

### Part 2. Rewards in Reinforcement Learning (MCMA)

**Correct Answers: A, B, D, E**

| Option | Verdict | Explanation |
|---|---|---|
| A | TRUE | Reward shaping can guide exploration by giving hints about progress. |
| B | TRUE | The reward function encodes the task objective. |
| C | FALSE | Noise can destabilize learning if not handled carefully. |
| D | TRUE | If the agent rarely sees non-zero reward, learning the value of actions becomes harder. |
| E | TRUE | A low discount factor makes the agent short-sighted and ignores long-term consequences. |
| F | FALSE | The return includes the sum of future **discounted** rewards, not just immediate ones. |

---

### Part 3. Bellman Update

**a) General Bellman Expectation Equation**

$$V(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [ R(s, a, s') + \gamma V(s') ]$$

Since the robot always "Go Forward", the policy is deterministic (π = 1 for the single action).

---

**b) Updated V(S3)**

$$V(S3) = R(S3 \rightarrow S4) + \gamma \times V(S4) = 4 + 0.8 \times 0 = \mathbf{4.0}$$

**Answer: V(S3) = 4.0**

---

**c) Updated V(S2)**

$$V(S2) = R(S2 \rightarrow S3) + \gamma \times V(S3) = 2 + 0.8 \times 4.0 = \mathbf{5.2}$$

**Answer: V(S2) = 5.2**

> Note: This is an **asynchronous** (in-place) update — it uses the freshly computed V(S3) = 4.0 rather than the old value of 0.

---

**d) Updated V(S1)**

$$V(S1) = R(S1 \rightarrow S2) + \gamma \times V(S2) = 1 + 0.8 \times 5.2 = \mathbf{5.16}$$

**Answer: V(S1) = 5.16**

---

**e) Expected total return starting from S1**

After this one update step, V(S1) = the expected total return from S1.

**Answer: 5.16**

---

## Question 2 — Reinforcement Learning

### Part 1. Q-Learning (MCMA)

**Correct Answers: A, C, D**

| Option | Verdict | Explanation |
|---|---|---|
| A | TRUE | Q-Learning uses max over actions in the next state, regardless of the action actually taken — that's off-policy. |
| B | FALSE | Q-Learning is model-free and learns directly from real transitions. No environment model needed. |
| C | TRUE | Large steps mean less averaging and can cause oscillations or divergence. |
| D | TRUE | Greedy-only behavior can prevent sufficient exploration, leaving many states unvisited. |
| E | FALSE | Convergence requires a **decaying** learning rate and sufficient exploration. |
| F | FALSE | Q-values represent the expected return under the **optimal** policy, not the behavior policy — because Q-Learning is off-policy. |

---

### Part 2. Greedy vs. Exploratory Strategies (MCMA)

**Correct Answers: A, C, E**

| Option | Verdict | Explanation |
|---|---|---|
| A | TRUE | By definition, the agent exploits with probability 1 − ε. |
| B | FALSE | Lower ε means **less** exploration, not more. |
| C | TRUE | Persistent exploration helps avoid local optima. |
| D | FALSE | Purely greedy policies can get stuck and fail to find optimal solutions. |
| E | TRUE | Softmax provides graded exploration based on Q-values rather than a fixed ε cutoff. |
| F | FALSE | More exploration typically **increases** performance variance due to random actions. |

---

### Part 3. Greedy Exploration for Q-Learning

**Setup:**
- $Q(\text{Heavy Irrigation}) = 4.5$
- $Q(\text{Moderate Irrigation}) = 5.0$ ← maximum
- $Q(\text{No Irrigation}) = 3.0$
- $\varepsilon = 0.2$, $N = 3$ total actions, $k_{\text{greedy}} = 1$

---

**a) Greedy action**

The greedy action has the highest Q-value:

$$Q(\text{Moderate Irrigation}) = 5.0 \rightarrow \text{Greedy action: Moderate Irrigation}$$

---

**b) Probability of greedy action (ε = 0.2)**

With only 1 greedy action out of 3:

$$P(\text{greedy}) = (1 - \varepsilon) + \frac{\varepsilon}{N} = (1 - 0.2) + \frac{0.2}{3} = 0.8 + 0.0667 \approx \mathbf{0.867}$$

**Answer: ≈ 0.867**

---

**c) Probability of Heavy Irrigation (non-greedy)**

Heavy Irrigation is not the greedy action — it only gets the random exploration share:

$$P(\text{Heavy Irrigation}) = \frac{\varepsilon}{N} = \frac{0.2}{3} \approx \mathbf{0.067}$$

**Answer: ≈ 0.067**

---

**d) Probability of greedy action with ε = 0.5**

$$P(\text{greedy}) = (1 - 0.5) + \frac{0.5}{3} = 0.5 + 0.167 \approx \mathbf{0.667}$$

**Answer: ≈ 0.667**

---

**e) Why is ε too high harmful?**

If ε is too high, the agent takes random actions too frequently and does not exploit its learned Q-values. This slows learning and prevents convergence to an optimal policy — the agent keeps exploring even when it has already found good actions.

---

## Question 3 — Game Theory

### Part 1. Pure vs. Mixed Strategies (MCMA)

**Correct Answers: A, B, D, F**

| Option | Verdict | Explanation |
|---|---|---|
| A | TRUE | A pure strategy picks one action with certainty (probability 1). |
| B | TRUE | A mixed strategy randomizes over two or more pure strategies. |
| C | FALSE | Some finite games (e.g. Matching Pennies) have **no** pure-strategy Nash equilibrium. |
| D | TRUE | Mixed strategies fill this gap — Nash's theorem guarantees equilibrium existence in finite games when mixed strategies are allowed. |
| E | FALSE | A mixed strategy can yield a better expected payoff by preventing exploitation. |
| F | TRUE | Indifference ensures each pure strategy in the support yields the same expected payoff, making mixing optimal. |

---

### Part 2. Prisoner's Dilemma (MCMA)

**Correct Answers: A, B, C, E, F**

| Option | Verdict | Explanation |
|---|---|---|
| A | TRUE | In a one-shot dilemma, defecting maximizes immediate reward regardless of the other's choice — it is the dominant strategy. |
| B | TRUE | Repeated interactions and high discount factors make cooperation sustainable. |
| C | TRUE | History-based strategies can discourage betrayal by retaliation. |
| D | FALSE | Mutual cooperation can yield higher long-term returns if future rewards matter sufficiently. |
| E | TRUE | Tit-for-Tat rewards cooperation and punishes defection, supporting mutual cooperation. |
| F | TRUE | Without valuing future payoffs (low discount factor), there is no incentive to share resources. |

---

### Part 3. Minimax

**a) Worst-case (minimum) profit per spot:**

$$\begin{aligned}
\text{Spot A:}&\ \min(500, 350, 200, 150) = \$150 \\
\text{Spot B:}&\ \min(450, 400, 300, 200) = \$200 \\
\text{Spot C:}&\ \min(600, 250, 200, 200) = \$200 \\
\text{Spot D:}&\ \min(450, 450, 450, 300) = \$300 \\
\text{Spot E:}&\ \min(550, 350, 350, 100) = \$100 \\
\text{Spot F:}&\ \min(500, 400, 400, 350) = \$350 \end{aligned}$$

**b) Minimax choice:**

$$\max\{150, 200, 200, 300, 100, 350\} = \$350 \rightarrow \text{Choose Spot F}$$

**Huey should choose Spot F**, guaranteeing a worst-case daily profit of **$350**.

---

## Question 4 — Game Theory Continued

### Part 1. Folk Theorem (MCMA)

**Correct Answers: A, B, D, F**

| Option | Verdict | Explanation |
|---|---|---|
| A | TRUE | The Folk Theorem shows repeated interaction can sustain cooperation by threat of punishment — any feasible, individually rational payoff is achievable. |
| B | TRUE | SPE strengthens Nash by ruling out non-credible threats — strategies must be Nash in every subgame, including off-path ones. |
| C | FALSE | Not all Nash equilibria are subgame perfect. Only those that hold after every possible history qualify. |
| D | TRUE | Grim Trigger's threat of eternal punishment makes defection unattractive if future payoffs are valued. |
| E | FALSE | The Folk Theorem does **not** apply to finite repeated games — backward induction unravels cooperation from the final round. |
| F | TRUE | This is the core mechanism: future punishment (loss of resource access) enforces cooperation. |

---

### Part 2. Pavlov and Zero-Sum (MCMA)

**Correct Answers: A, C, E, F**

| Option | Verdict | Explanation |
|---|---|---|
| A | TRUE | Pavlov = Win-Stay, Lose-Shift: keep a strategy if it worked, switch if exploited. |
| B | FALSE | Pavlov and Tit-for-Tat differ, especially in noisy settings. Pavlov is more forgiving — it can restore cooperation after accidental defection. |
| C | TRUE | By definition, zero-sum means total payoffs sum to zero — one's gain exactly equals another's loss. |
| D | FALSE | Some zero-sum games (like Rock-Paper-Scissors) have only mixed-strategy equilibria. |
| E | TRUE | Pavlov's forgiveness avoids endless retaliation cycles after accidental defections. |
| F | TRUE | Minimax ensures a player maximizes their worst-case payoff — the standard approach to zero-sum game equilibria. |

---

### Part 3. Tit-for-Tat, Pavlov, and Folk Theorem

**Setup — payoff table:**

| You \ Adrian | Paw (P) | Stay (S) |
|---|---|---|
| Ignore (I) | You: −2, Adrian: +2 | You: +1, Adrian: 0 |
| Guard (G) | You: 0, Adrian: −1 | You: −1, Adrian: 0 |

**3-Round Sequence:**

| Round | Adrian | You | Outcome | Your Payoff | Adrian's Payoff |
|---|---|---|---|---|---|
| 1 | Paw | Guard | Mug safe + spray | 0 | −1 |
| 2 | Paw | Ignore | Mug falls | −2 | +2 |
| 3 | Stay | Ignore | Mug safe | +1 | 0 |
| **Total** | | | | **−1** | **+1** |

---

**a) Total payoffs**

$$\text{Your total} = 0 + (-2) + 1 = \mathbf{-1}$$
$$\text{Adrian's total} = -1 + 2 + 0 = \mathbf{+1}$$

You broke even overall but Adrian ended slightly ahead because the punishment was relaxed too soon.

---

**b) Tit-for-Tat vs Pavlov — which better discourages pawing?**

**Tit-for-Tat (TfT):**
- TfT copies the opponent's last action.
- After Adrian paws in Round 1, you guarded → correct punishment.
- But you Ignored in Round 2 — the punishment broke down.
- TfT would guard again after observing pawing, but only in the next round. It does not self-correct mid-episode.

**Pavlov (Win-Stay, Lose-Shift):**
- In Round 2, you got −2 (a bad payoff) → Pavlov says switch from Ignore to Guard.
- This would re-establish punishment in Round 3, discouraging future pawing.

**Conclusion: Pavlov better discourages repeated pawing.** It reacts immediately to a bad payoff, while Tit-for-Tat only mirrors Adrian's last move and does not inherently self-correct from the owner's side.

---

**c) Does this support the Folk Theorem?**

The Folk Theorem states: in infinitely repeated games, cooperative outcomes can be sustained with credible, consistent punishment for defection.

- The spray bottle is the punishment mechanism.
- Round 1: punishment applied → Adrian got −1 (learned lesson).
- Round 2: punishment dropped → Adrian risked pawing again and gained +2.

**Conclusion:** The Folk Theorem would hold only if punishment is applied consistently. Relaxing the threat in Round 2 removed its credibility, allowing Adrian to exploit the gap. Consistent Guarding after every Paw would sustain the cooperative outcome (Adrian Stays).
