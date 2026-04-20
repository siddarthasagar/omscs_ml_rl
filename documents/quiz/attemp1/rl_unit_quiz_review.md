# RL Unit Quiz — Full Review (Attempts 1 & 2)
### CS 7641 · Spring 2026 · Siddartha Sagar Chinne

**Attempt 1 Score:** 10.33 / 33 | Submitted Apr 19 at 5:54 pm
**Attempt 2 Score:** Partial results visible | 1 attempt remaining

| Status | Questions |
|---|---|
| Correct | Q2, Q9, Q17, Q22, Q23 |
| Partial | Q7, Q8, Q13, Q14 (A2: 1.33/2), Q18, Q19 |
| Incorrect / 0 | Q1, Q3–6, Q10–12, Q15–16, Q20–21, Q24 |

---

## Negative Marking — How It Works

For multi-select questions, Canvas scores using:

$$\text{Score} = \frac{\text{correct\_selected} - \text{wrong\_selected}}{\text{total\_correct}} \times \text{points}$$

One wrong selection cancels one correct selection. **Strategy: only select options you are confident about.** Skipping an uncertain option is safer than guessing.

---

## Question Set 1 — MDPs

### Q1 — 0 / 2 (Incorrect)
**Topic:** MDP transition functions and value functions — select all true

**Real-life analogy:** Think of an MDP as a board game. At each square (state) you pick a move (action) and a dice roll determines where you go next. The transition function is the dice + board rules. Values are how much total score you expect from each square if you keep playing.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | MISSED | A transition function must assign non-negative probabilities that sum to 1 for each state-action pair | TRUE — definition of a valid probability distribution |
| 2 | Correct (selected) | The value of a state depends on transition probabilities and expected rewards | TRUE — Bellman equation directly contains $T(s,a,s')$ and $R$ |
| 3 | Wrong (selected) | If transitions are deterministic, the transition probability for each state-action pair is always 1 | FALSE — in a deterministic MDP, exactly one next state gets prob=1, all others get 0. The statement implies all $T(s,a,s')$ triples equal 1, which is impossible since they must sum to 1 |
| 4 | Wrong (selected) | The sum of transition probabilities across all possible next states can exceed one | FALSE — must sum to exactly 1, always |
| 5 | Correct (selected) | The Bellman expectation equation relates the value of a state to the values of its possible next states | TRUE — $V(s) = \sum_a \pi(a|s) \sum_{s'} T(s,a,s') [R + \gamma V(s')]$ |
| 6 | Correct to skip | Changing transition probabilities has no effect on the optimal policy | FALSE — changing $T$ reshapes what policy is best |

**Scoring check:** Correct options = $\{1, 2, 5\}$. Selected: $2\checkmark, 3\times, 4\times, 5\checkmark$ $\rightarrow$ $\frac{2 - 2}{3} \times 2 = \mathbf{0}$ ✓

**Next attempt:** Select **1, 2, 5 only**.

---

### Q2 — 2 / 2 (Correct)
**Topic:** Stochastic environments — select all that apply

All correct. Stochastic = same action can lead to different outcomes.

- **Stochastic (correct selections):** Delivery drone pushed by wind, cleaning robot with partial stain, delivery bot on icy sidewalks
- **Deterministic (correctly skipped):** Robot arm with no slippage, vacuum robot always same result, warehouse conveyor always moves forward

---

### Q3 — 0 / 1 (Incorrect — entered 4.28)
**Topic:** Compute updated V(B3)

**Real-life analogy:** Molebert is like a delivery driver who can either keep driving forward (Dig, big worm reward) or nap at the current burrow (Nap, small snack). The crucial insight: napping keeps Molebert in B3 — so the value of B3 appears on both sides of its own equation. Unlike terminal state B4 (where you just substitute 0), B3 is a non-terminal state with a self-loop that must be solved algebraically.

**Setup:**
- Tunnel: B1 → B2 → B3 → B4 (terminal)
- Rewards: B1→B2 = +1, B2→B3 = +3, B3→B4 = +5, Nap = +0.2
- Policy: π(Dig) = 0.85, π(Nap) = 0.15
- Discount: γ = 0.75
- Initial values: all = 0

**Key distinction — terminal vs non-terminal self-loop:**

| State | Type | Treatment |
|---|---|---|
| B4 | Terminal | V(B4) = 0 fixed — substitute 0 directly, no algebra |
| B3 (when Napping) | Non-terminal self-loop | V(B3) appears on both sides — must solve algebraically |

**Why the naive answer of 4.28 was wrong:** Substituting V_old(B3) = 0 for the Nap term implicitly treats B3 as if it were terminal — exactly the conceptual error the course hint points at.

**Correct approach — write the equation with V(B3) on both sides, then solve:**

$$\begin{aligned}
V(B3) &= \pi(\text{Dig}) \times [R_{dig} + \gamma \times V(B4)] \\
      &\quad + \pi(\text{Nap}) \times [R_{nap} + \gamma \times V(B3)] \\
V(B3) &= 0.85 \times (5 + 0.75 \times 0) \\
      &\quad + 0.15 \times (0.2 + 0.75 \times V(B3)) \\
V(B3) &= 0.85 \times 5.0 + 0.15 \times 0.2 + 0.15 \times 0.75 \times V(B3) \\
V(B3) &= 4.25 + 0.03 + 0.1125 \times V(B3) \\
V(B3) - 0.1125 \times V(B3) &= 4.28 \\
0.8875 \times V(B3) &= 4.28 \\
V(B3) &= \frac{4.28}{0.8875} = \mathbf{4.82}
\end{aligned}$$

**Answer: 4.82**

---

### Q4 — 0 / 1 (Unanswered)
**Topic:** Compute updated V(B2)

**Update style: Asynchronous (in-place)** — use V_new(B3) = 4.82 for the Dig term. The Nap self-loop on B2 requires the same algebraic treatment as Q3.

**Calculation:**

$$\begin{aligned}
V(B2) &= 0.85 \times [R_{dig} + \gamma \times V_{new}(B3)] \\
      &\quad + 0.15 \times [R_{nap} + \gamma \times V(B2)] \\
V(B2) &= 0.85 \times (3 + 0.75 \times 4.82) \\
      &\quad + 0.15 \times (0.2 + 0.75 \times V(B2)) \\
V(B2) &= 0.85 \times (3 + 3.615) + 0.03 + 0.1125 \times V(B2) \\
V(B2) &= 5.6228 + 0.03 + 0.1125 \times V(B2) \\
V(B2) - 0.1125 \times V(B2) &= 5.6528 \\
0.8875 \times V(B2) &= 5.6528 \\
V(B2) &= \frac{5.6528}{0.8875} = \mathbf{6.37}
\end{aligned}$$

**Answer: 6.37** *(previously stated as 5.31, then 2.58 — both wrong)*

---

### Q5 — 0 / 1 (Unanswered)
**Topic:** Compute updated V(B1)

**Using V_new(B2) = 6.37. Same algebraic treatment for the Nap self-loop.**

**Calculation:**

$$\begin{aligned}
V(B1) &= 0.85 \times [R_{dig} + \gamma \times V_{new}(B2)] \\
      &\quad + 0.15 \times [R_{nap} + \gamma \times V(B1)] \\
V(B1) &= 0.85 \times (1 + 0.75 \times 6.37) \\
      &\quad + 0.15 \times (0.2 + 0.75 \times V(B1)) \\
V(B1) &= 0.85 \times (1 + 4.7775) + 0.03 + 0.1125 \times V(B1) \\
V(B1) &= 4.9109 + 0.03 + 0.1125 \times V(B1) \\
V(B1) - 0.1125 \times V(B1) &= 4.9409 \\
0.8875 \times V(B1) &= 4.9409 \\
V(B1) &= \frac{4.9409}{0.8875} = \mathbf{5.57}
\end{aligned}$$

**Answer: 5.57** *(previously stated as 4.27, then 0.88 — both wrong)*

---

### Q6 — 0 / 1 (Unanswered)
**Topic:** Expected total worm haul starting from B1 after this update step

After one in-place algebraic Bellman sweep, $V_{\text{new}}(B1)$ is the expected discounted worm haul from B1 under the current policy:

$$V_{\text{new}}(B1) = \mathbf{5.57}$$

**Answer: 5.57** *(previously stated as 4.27, then 0.88 — both wrong)*

---

> **Summary of the Molebert insight:** The course hint says "think about how you'd treat the Nap state differently from terminal states." The difference is:
> - **Terminal state (B4):** V = 0, substitute directly — no equation to solve
> - **Non-terminal self-loop (Nap at B1/B2/B3):** V appears on both sides — rearrange and solve algebraically
>
> The general algebraic pattern for any self-looping state:
> $$\begin{aligned}
> V(s) &= \pi(\text{Dig}) \times [R_{dig} + \gamma \times V(\text{next})] \\
>      &\quad + \pi(\text{Nap}) \times [R_{nap} + \gamma \times V(s)] \\
> \\
> V(s) \times [1 - \pi(\text{Nap}) \times \gamma] &= \pi(\text{Dig}) \times [R_{dig} + \gamma \times V(\text{next})] + \pi(\text{Nap}) \times R_{nap} \\
> \\
> V(s) &= \frac{\pi(\text{Dig}) \times (R_{dig} + \gamma \times V(\text{next})) + \pi(\text{Nap}) \times R_{nap}}{1 - \pi(\text{Nap}) \times \gamma} \\
>      &= \frac{[\ldots]}{1 - 0.15 \times 0.75} \\
>      &= \frac{[\ldots]}{0.8875}
> \end{aligned}$$
> The denominator $0.8875 = 1 - 0.1125$ is the same for all three states since $\pi(\text{Nap})$ and $\gamma$ are fixed.

---

## Question Set 2 — Reinforcement Learning

### Q7 — 0.67 / 2 (Partial)
**Topic:** Model-free (Bot A) vs model-based (Bot B)

**Real-life analogy:** Model-free (Bot A) is like a chef who learns only by cooking in the real kitchen — tastes every dish, adjusts from real experience. Model-based (Bot B) also builds a mental simulation of the kitchen — can mentally rehearse new recipes without using real ingredients. The simulation costs compute but saves real-world trials.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | Correct (selected) | Bot B can plan future actions without taking real watering actions | TRUE — model-based agents simulate using their learned model |
| 2 | Correct to skip | Bot A learns by predicting effects without trying them | FALSE — Bot A is model-FREE; it learns only from actual experience |
| 3 | Correct (selected) | Bot B can suffer if its learned model is inaccurate | TRUE — bad model → bad planning → bad policy |
| 4 | Wrong (selected) | Bot A will always be more sample efficient than Bot B | FALSE — model-based is typically more sample efficient (can simulate data from the model) |
| 5 | Correct (selected) | Bot B needs more compute for planning/simulated rollouts | TRUE — planning requires computation at each step |
| 6 | Correct to skip | Bot B's model is specialized and cannot be reused | FALSE — a learned environment model can often be repurposed for different reward functions |

**Correct options:** {1, 3, 5}. Selected: 1✓ 3✓ 4✗ 5✓ → (3 − 1) / 3 × 2 = **0.67** ✓

**Next attempt:** Select **1, 3, 5 only**. Drop option 4.

---

### Q8 — 1 / 2 (Partial)
**Topic:** SARSA (Drone 1, on-policy) vs Q-Learning (Drone 2, off-policy)

**Real-life analogy:** SARSA is like a student graded on the actual choices they made — including guesses (exploration). Q-Learning grades on what the best possible answer would have been, even if they wrote something different. SARSA learns the policy it follows; Q-Learning always aims for the optimal policy.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | Correct (selected) | Drone 1's Q-values depend on the actions it actually takes, including exploration | TRUE — SARSA update uses the actual next action a' |
| 2 | MISSED | Drone 2 updates using the max Q-value of next actions, regardless of what it does next | TRUE — Q-Learning update: max_{a'} Q(s',a') |
| 3 | Correct (selected) | SARSA's on-policy updates make Drone 1 more conservative in risky areas | TRUE — SARSA accounts for probability of exploring into dangerous states |
| 4 | Correct (selected) | Drone 2's updates assume it will always act optimally in the future | TRUE — the max operator assumes future greedy behavior |
| 5 | Wrong (selected) | Drone 1's Q-values represent the return under the optimal policy | FALSE — SARSA Q-values represent the BEHAVIOR policy return (with ε-exploration). Representing optimal policy return is the property of Q-Learning, not SARSA |
| 6 | Correct to skip | Both guaranteed to learn the same final policy | FALSE — different policies, may not converge identically |

**Correct options:** {1, 2, 3, 4}. Selected: 1✓ 3✓ 4✓ 5✗ → (3 − 1) / 4 × 2 = **1.0** ✓

**Next attempt:** Select **1, 2, 3, 4**. Add option 2. Drop option 5.

---

### Q9 — 1 / 1 (Correct)
**Topic:** Which actions are greedy? (maximum Q-value)

**Context — Factory maintenance Q-values:**
- Q(Full Replacement) = 8.0 ← maximum
- Q(Partial Repair) = 7.5
- Q(Increase Cooling) = 8.0 ← maximum
- Q(Do Nothing) = 5.0
- ε = 0.2, k_total = 4, k_greedy = 2

Full Replacement and Increase Cooling share the highest Q-value of 8.0. Correctly identified. ✓

---

### Q10 — 0 / 1 (Incorrect)
**Topic:** P(Full Replacement selected under ε-greedy)

**Real-life analogy:** 80% of the time you pick one of your two favourite dishes (greedy), 20% of the time you pick randomly from the full menu of 4. Each favourite gets (0.8/2) from exploitation plus (0.2/4) from random exploration.

$$P(\text{greedy}) = \frac{1 - \varepsilon}{k_{\text{greedy}}} + \frac{\varepsilon}{k_{\text{total}}}
= \frac{0.8}{2} + \frac{0.2}{4}
= 0.40 + 0.05
= \mathbf{0.45}$$

**Your answer: 0.0561 (wrong). Correct answer: 0.45**

---

### Q11 — 0 / 1 (Incorrect)
**Topic:** P(Do Nothing chosen under ε-greedy)

Do Nothing is non-greedy ($Q = 5.0$). Only appears in the random exploration portion.

$$P(\text{non-greedy}) = \frac{\varepsilon}{k_{\text{total}}}
= \frac{0.2}{4}
= \mathbf{0.05}$$

**Your answer: 0.0351 (wrong). Correct answer: 0.05**

> Both Q10 and Q11 errors suggest a softmax-style formula was used instead of the ε-split formula. The correct formula ignores Q-value magnitudes once greedy/non-greedy is identified.

---

### Q12 — 0 / 1 (Incorrect)
**Topic:** P(Increase Cooling) when ε increases to 0.6

Increase Cooling remains greedy (Q = 8.0). Only ε changes.

$$P(\text{IC}) = \frac{1 - 0.6}{2} + \frac{0.6}{4}
= \frac{0.4}{2} + \frac{0.6}{4}
= 0.20 + 0.15
= \mathbf{0.35}$$

**Your answer: 0.1684 (wrong). Correct answer: 0.35**

> As ε increases 0.2 → 0.6, P(IC) decreases 0.45 → 0.35. More exploration = greedy actions get less weight. At ε = 0: P = 0.5. At ε = 1: P = 0.25.

---

## Question Set 3 — Game Theory

### Q13 — 0.67 / 2 (Partial)
**Topic:** Ada & Byron trading bots — repeated game, Folk Theorem

**Real-life analogy:** Two gas stations: if both price high (collude), both earn well. If one cuts price it grabs short-term market share, but the other retaliates next week. Folk Theorem: in a long-running repeated game, cooperation is stable if players are patient enough.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | Correct (selected) | This is a repeated game where the Folk Theorem applies | TRUE — indefinitely repeated game with discounting = textbook Folk Theorem |
| 2 | Correct to skip | Undercutting once always leads to maximum profit in the long run | FALSE — punishment follows, long-run profits drop |
| 3 | Correct (selected) | A credible punishment strategy can make collusion stable | TRUE — threat of future undercutting deters defection today |
| 4 | Correct to skip | Strictly zero-sum because gain of one equals loss of the other | FALSE — both can gain (collusion) or both can lose (mutual undercutting). Not zero-sum |
| 5 | MISSED | In a strictly zero-sum game, Folk Theorem does not apply | TRUE — in a purely zero-sum game there are no cooperative gains to sustain; Folk Theorem gives only minimax outcomes |
| 6 | Wrong (selected) | If both bots discount the future heavily, collusion is more likely to hold | FALSE — heavy discounting = future punishments worth LESS → defection more tempting. Cooperation needs patient players (high δ) |

**Correct options:** {1, 3, 5}. Selected: 1✓ 3✓ 6✗ missed 5 → (2 − 1) / 3 × 2 = **0.67** ✓

**Next attempt:** Select **1, 3, 5**. Drop option 6. Add option 5.

---

### Q14 — 0 / 2 (A1) → 1.33 / 2 (A2) — Still Partial
**Topic:** Huey & Dewey pirate subgame — two-stage game

**Real-life analogy:** Two business partners: stage 1 = share profits fairly or grab more. If both grab, they enter a costly legal fight (stage 2). The threat of a costly fight deters grabbing in stage 1 — backward induction in subgame perfect equilibrium (SPE).

**Attempt 2 result:** Score 1.33/2 with selection {2, 4} confirms 3 total correct options exist:

$$1.33 = \frac{2}{\text{total\_correct}} \times 2 \rightarrow \text{total\_correct} = 3$$

The missed third option is **option 1** — in a mixed strategy Nash equilibrium, a player randomises precisely to make the opponent indifferent between their own strategies. Huey mixing Share/Steal at the right probability makes Dewey's expected payoff from Sharing equal to his payoff from Stealing. That indifference is what makes the mix self-consistent and is a general property of two-stage games.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | MISSED | Huey can use a mixed strategy in the first stage to make Dewey indifferent between sharing and stealing | TRUE — in a mixed strategy Nash equilibrium, a player randomises at exactly the probability that makes the opponent indifferent. General property of two-stage games, not dependent on specific payoff values |
| 2 | Correct (selected A2) | The second-stage fight can be analyzed for subgame perfect equilibrium | TRUE — it is a proper subgame; SPE requires an equilibrium in every subgame |
| 3 | Correct to skip | A pure strategy will always yield better payoffs than mixing | FALSE — mixing can be optimal in many games |
| 4 | Correct (selected A2) | A credible threat to fight in the subgame can influence first-stage choices | TRUE — backward induction: if fighting is rational in stage 2, it deters stealing in stage 1 |
| 5 | Correct to skip | This pirate game is strictly zero-sum because one pirate's gain equals the other's loss in every outcome | FALSE — when both steal and fight, BOTH lose loot. Zero-sum requires one's gain to exactly equal the other's loss in every single outcome |
| 6 | Correct to skip | Subgame perfection requires randomizing in every subgame | FALSE — SPE can use pure strategies |

**Correct options:** {1, 2, 4}. Attempt 2 selected: 2✓ 4✓ (missed 1) → 2/3 × 2 = **1.33** ✓

**Next attempt:** Select **1, 2, 4**. Add option 1.

---

### Q15 — 0 / 2 (Incorrect)
**Topic:** Worst-case daily profit — pure minimax strategy (Spot A–F)

**Real-life analogy:** Huey picks a market stand location. Rivals will try to undercut him — so Huey wants the location where even in the worst case he still makes the most money. Minimax = pick the spot with the best worst-case.

**Payoff matrix (daily profit $):**

| Spot | R1 | R2 | R3 | R4 | Worst-case (min) |
|---|---|---|---|---|---|
| A | 950 | 800 | 450 | 200 | **200** |
| B | 900 | 700 | 600 | 400 | **400** |
| C | 1,000 | 500 | 500 | 350 | **350** |
| D | 850 | 850 | 750 | 600 | **600** |
| E | 1,200 | 700 | 400 | 250 | **250** |
| F | 1,100 | 800 | 800 | 700 | **700 ← highest** |

$$\begin{aligned}
\text{Spot A:}&\ \min(950, 800, 450, 200) = 200 \\
\text{Spot B:}&\ \min(900, 700, 600, 400) = 400 \\
\text{Spot C:}&\ \min(1000, 500, 500, 350) = 350 \\
\text{Spot D:}&\ \min(850, 850, 750, 600) = 600 \\
\text{Spot E:}&\ \min(1200, 700, 400, 250) = 250 \\
\text{Spot F:}&\ \min(1100, 800, 800, 700) = 700
\end{aligned}$$

$$\text{Minimax} = \text{Spot F}, \text{guaranteed worst-case} = \mathbf{\$700}$$

**Your answer: Spot C — wrong. Correct answer: Spot F**

---

### Q16 — 0 / 1 (Incorrect)
**Topic:** Mixed strategy (40% Spot F, 60% Spot D) — expected worst-case daily profit

$$\begin{aligned}
R_1:&\ 0.4 \times 1100 + 0.6 \times 850 = 440 + 510 = 950 \\
R_2:&\ 0.4 \times 800 + 0.6 \times 850 = 320 + 510 = 830 \\
R_3:&\ 0.4 \times 800 + 0.6 \times 750 = 320 + 450 = 770 \\
R_4:&\ 0.4 \times 700 + 0.6 \times 600 = 280 + 360 = 640
\end{aligned}$$

$$\text{Expected worst-case} = \min(950, 830, 770, 640) = \mathbf{\$640}$$

**Your answer: $3,190 — wrong. Correct answer: $640**

---

### Q17 — 1 / 1 (Correct)
**Topic:** Does mixed strategy give higher worst-case than pure minimax? → **No ✓**

Pure minimax (Spot F) guarantees $700. Mixed strategy (40% F + 60% D) only guarantees $640 — strictly worse. Correctly answered.

---

## Question Set 4 — Game Theory Continued

### Q18 — 1.2 / 2 (Partial)
**Topic:** Robot factory — Cooperate or Defect (infinitely repeated game)

**Real-life analogy:** Office shared printer: everyone could print huge jobs (defect), but if all cooperate and take turns, everyone works faster. Grim Trigger = "if you ever hog the printer, I'll never let you use it again." Subgame perfection ensures that threat is actually credible.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | Correct (selected) | Patient robots tolerate short-term losses to keep cooperation stable | TRUE — high discount factor → cooperation sustainable |
| 2 | MISSED | A robot fleet can use Grim Trigger (if any robot cuts the line, others refuse to share forever) | TRUE — valid strategy in infinitely repeated games; sustains cooperation |
| 3 | Correct to skip | Nash equilibrium means robots always share tools fairly in every round | FALSE — many Nash equilibria exist in repeated games, including always-defect |
| 4 | MISSED | Subgame perfection ensures no robot can bluff about punishing defectors | TRUE — SPE requires every threatened punishment to be a Nash equilibrium of that subgame (credible) |
| 5 | Correct (selected) | Folk Theorem implies many ways of sharing fairly can be sustained as equilibria | TRUE — any payoff above minimax can be an equilibrium for sufficiently patient players |
| 6 | Correct (selected) | Robots can use threats like blocking access to tools in future rounds | TRUE — future denial of resources is the mechanism for enforcing cooperation |

**Correct options:** {1, 2, 4, 5, 6}. Selected: 1✓ 5✓ 6✓ → 3/5 × 2 = **1.2** ✓

**Next attempt:** Select **1, 2, 4, 5, 6**. Add options 2 and 4. Avoid option 3.

---

### Q19 — 0.8 / 2 (Partial)
**Topic:** Mars rovers with Puddles (always defects)

**Real-life analogy:** Huey/Dewey/Louie = cooperative teammates. Puddles = the free-rider who always takes. TfT punishes each time but can't reform Puddles. Pavlov gets trapped in switching cycles. Grim Trigger gives up permanently. Others may need to coordinate.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | MISSED | Puddles's always-defect destabilizes cooperation assumptions for the other rovers | TRUE — TfT and Pavlov assume the opponent responds to incentives. Puddles breaks that assumption |
| 2 | MISSED | Louie's Pavlov may get trapped in constant switching against Puddles | TRUE — Cooperate → Puddles defects → Louie loses → shifts to Defect → ties/wins → shifts back → repeat endlessly |
| 3 | MISSED | TfT punishes Puddles each time, but if Puddles never cooperates, the punishment has no effect | TRUE — TfT defects in response but Puddles never changes behavior. Ineffective against a committed always-defect player |
| 4 | Correct (selected) | Dewey's Grim Trigger means once Puddles defects, Dewey will never cooperate again | TRUE — definition of Grim Trigger |
| 5 | Correct to skip | Folk Theorem guarantees all four rovers maintain stable cooperation no matter how Puddles behaves | FALSE — Folk Theorem assumes rational, responsive players. Committed always-defect breaks the incentive structure |
| 6 | Correct (selected) | The other rovers may form an implicit alliance to punish Puddles in zero-sum contests | TRUE — coordination needed to contain Puddles |

**Correct options:** {1, 2, 3, 4, 6}. Selected: 4✓ 6✓ → 2/5 × 2 = **0.8** ✓

**Next attempt:** Select **1, 2, 3, 4, 6**. Add options 1, 2, 3. Keep 4 and 6. Avoid option 5.

---

### Q20 — 0 / 1 (Incorrect)
**Topic:** Your total payoff over 3 rounds if you SWITCH to Ignore in Round 3

**Context — Pythagoruff dog game:**

| You \ Pythagoruff | Steal (S) | Stay (T) |
|---|---|---|
| Ignore (I) | You: −4, Dog: +3 | You: +2, Dog: 0 |
| Guard (G) | You: −1, Dog: −3 | You: −1, Dog: 0 |

Pythagoruff steals in every round.

$$\begin{aligned}
\text{Round 1:}&\ \text{Guard} + \text{Steal} \rightarrow \text{Your payoff} = -1 \\
\text{Round 2:}&\ \text{Guard} + \text{Steal} \rightarrow \text{Your payoff} = -1 \\
\text{Round 3:}&\ \text{Ignore} + \text{Steal} \rightarrow \text{Your payoff} = -4
\end{aligned}$$

$$\text{Total} = (-1) + (-1) + (-4) = \mathbf{-6}$$

**Your answer: 0 — wrong. Correct answer: −6**

---

### Q21 — 0 / 1 (Incorrect)
**Topic:** Pythagoruff's total payoff over 3 rounds if you switch

$$\begin{aligned}
\text{Round 1:}&\ \text{Guard} + \text{Steal} \rightarrow \text{Pythagoruff} = -3 \\
\text{Round 2:}&\ \text{Guard} + \text{Steal} \rightarrow \text{Pythagoruff} = -3 \\
\text{Round 3:}&\ \text{Ignore} + \text{Steal} \rightarrow \text{Pythagoruff} = +3
\end{aligned}$$

$$\text{Total} = (-3) + (-3) + 3 = \mathbf{-3}$$

**Your answer: −6 — wrong. Correct answer: −3**

---

### Q22 — 1 / 1 (Correct)
**Topic:** Your total payoff if you do NOT switch in Round 3 (Guard)

$$\text{Rounds 1-3: Guard + Steal each} \rightarrow -1 + -1 + -1 = \mathbf{-3}$$

---

### Q23 — 1 / 1 (Correct)
**Topic:** Pythagoruff's total payoff if you do NOT switch

$$\text{Rounds 1-3: Guard + Steal each} \rightarrow -3 + -3 + -3 = \mathbf{-9}$$

---

### Q24 — 0 / 1 (Incorrect)
**Topic:** Should you have switched strategies in Round 3?

$$\begin{aligned}
\text{Your payoff switching} &= -6 \quad \text{(Q20)} \\
\text{Your payoff not switching} &= -3 \quad \text{(Q22)} \\
\text{Difference} &= -6 - (-3) = -3 \rightarrow \text{switching is worse}
\end{aligned}$$

$$\text{Correct answer: No}$$

**Your answer: Yes — wrong. Correct answer: No**

---

## Complete Answer Key

| Q | Points | Your Answer (A1) | Correct Answer | Notes |
|---|---|---|---|---|
| Q1 | 0/2 | 2, 3, 4, 5 | **1, 2, 5** | |
| Q2 | 2/2 | Correct | Correct | ✓ |
| Q3 | 0/1 | Unanswered / 4.28 (A2) | **4.82** | Algebraic solve for self-loop |
| Q4 | 0/1 | Unanswered | **6.37** | Cascades from Q3 |
| Q5 | 0/1 | Unanswered | **5.57** | Cascades from Q4 |
| Q6 | 0/1 | Unanswered | **5.57** | = V_new(B1) |
| Q7 | 0.67/2 | 1, 3, 4, 5 | **1, 3, 5** | |
| Q8 | 1/2 | 1, 3, 4, 5 | **1, 2, 3, 4** | |
| Q9 | 1/1 | FR, IC | FR, IC | ✓ |
| Q10 | 0/1 | 0.0561 | **0.45** | |
| Q11 | 0/1 | 0.0351 | **0.05** | |
| Q12 | 0/1 | 0.1684 | **0.35** | |
| Q13 | 0.67/2 | 1, 3, 6 | **1, 3, 5** | |
| Q14 | 0/2 → 1.33/2 | 4, 5 → 2, 4 | **1, 2, 4** | A2 confirmed 3 correct options |
| Q15 | 0/2 | Spot C | **Spot F** | |
| Q16 | 0/1 | $3,190 | **$640** | |
| Q17 | 1/1 | No | No | ✓ |
| Q18 | 1.2/2 | 1, 5, 6 | **1, 2, 4, 5, 6** | |
| Q19 | 0.8/2 | 4, 6 | **1, 2, 3, 4, 6** | |
| Q20 | 0/1 | 0 | **−6** | |
| Q21 | 0/1 | −6 | **−3** | |
| Q22 | 1/1 | −3 | −3 | ✓ |
| Q23 | 1/1 | −9 | −9 | ✓ |
| Q24 | 0/1 | Yes | **No** | |

---

## Correction Log

| Question | Answer History | Final Answer | Reason |
|---|---|---|---|
| Q3 — V(B3) | 4.28 → 4.25 → **4.82** | **4.82** | Nap is a non-terminal self-loop — must solve algebraically, not substitute V_old=0 |
| Q4 — V(B2) | 2.58 → 5.31 → **6.37** | **6.37** | Cascades from Q3 algebraic correction + async update |
| Q5 — V(B1) | 0.88 → 4.27 → **5.57** | **5.57** | Cascades from Q4 algebraic correction + async update |
| Q6 — Expected haul | 0.88 → 4.27 → **5.57** | **5.57** | Follows corrected V_new(B1) |
| Q14 — Correct options | {2,4} | **{1, 2, 4}** | A2 score 1.33/2 confirmed option 1 is also correct |

---

## Strategy for Attempt 3

All 24 questions fully resolved. Targeting 33/33:

| Question | Action | Correct Answer |
|---|---|---|
| Q1 | Select 1, 2, 5 only | — |
| Q3 | Enter value | **4.82** |
| Q4 | Enter value | **6.37** |
| Q5 | Enter value | **5.57** |
| Q6 | Enter value | **5.57** |
| Q7 | Select 1, 3, 5 — drop 4 | — |
| Q8 | Select 1, 2, 3, 4 — add 2, drop 5 | — |
| Q10 | Enter value | **0.45** |
| Q11 | Enter value | **0.05** |
| Q12 | Enter value | **0.35** |
| Q13 | Select 1, 3, 5 — drop 6, add 5 | — |
| Q14 | Select 1, 2, 4 — add option 1 | — |
| Q15 | Select Spot F | **Spot F** |
| Q16 | Enter value | **$640** |
| Q18 | Select 1, 2, 4, 5, 6 — add 2 and 4 | — |
| Q19 | Select 1, 2, 3, 4, 6 — add 1, 2, 3 | — |
| Q20 | Enter value | **−6** |
| Q21 | Enter value | **−3** |
| Q24 | Select No | **No** |