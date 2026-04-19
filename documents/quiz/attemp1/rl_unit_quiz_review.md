# RL Unit Quiz — Full Review (Attempt 1)

**Score:** 10.33 / 33 | Submitted Apr 19 at 5:54 pm | 2 attempts remaining

| Status | Questions |
|---|---|
| Correct | Q2, Q9, Q17, Q22, Q23 |
| Partial | Q7, Q8, Q13, Q18, Q19 |
| Incorrect / 0 | Q1, Q3–6, Q10–12, Q14–16, Q20–21, Q24 |

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
| A | MISSED | A transition function must assign non-negative probabilities that sum to 1 for each state-action pair | TRUE — definition of a valid probability distribution |
| B | Correct (selected) | The value of a state depends on transition probabilities and expected rewards | TRUE — Bellman equation directly contains T(s,a,s') and R |
| C | Wrong (selected) | If transitions are deterministic, the transition probability for each state-action pair is always 1 | FALSE — in a deterministic MDP, exactly one next state gets prob=1, all others get 0. The statement implies all T(s,a,s') triples equal 1, which is impossible since they must sum to 1 |
| D | Wrong (selected) | The sum of transition probabilities across all possible next states can exceed one | FALSE — must sum to exactly 1, always |
| E | Correct (selected) | The Bellman expectation equation relates the value of a state to the values of its possible next states | TRUE — V(s) = $\sum_a \pi(a|s) \sum_{s'} T(s,a,s') [R + \gamma V(s')]$ |
| F | Correct to skip | Changing transition probabilities has no effect on the optimal policy | FALSE — changing T reshapes what policy is best |

**Scoring check:** Correct options = {A, B, E}. Selected: B✓ C✗ D✗ E✓ → (2 correct − 2 wrong) / 3 × 2 = **0** ✓

**Next attempt:** Select **A, B, E only**.

---

### Q2 — 2 / 2 (Correct)
**Topic:** Stochastic environments — select all that apply

All correct. Stochastic = same action can lead to different outcomes.

- **Stochastic (correct selections):** Delivery drone pushed by wind, cleaning robot with partial stain, delivery bot on icy sidewalks
- **Deterministic (correctly skipped):** Robot arm with no slippage, vacuum robot always same result, warehouse conveyor always moves forward

---

### Q3 — 0 / 1 (Unanswered)
**Topic:** Compute updated V(B3) — one Bellman policy evaluation step

**Real-life analogy:** Molebert is like a delivery driver: Dig Forward = keep driving to the next hub (big reward), Nap = take a break at current location (small snack). Policy evaluation asks: given fixed habits (85% drive, 15% nap), what is each burrow worth in expected future worms?

**Setup:**
- Tunnel: B1 → B2 → B3 → B4 (terminal)
- Rewards: B1→B2 = +1, B2→B3 = +3, B3→B4 = +5, Nap = +0.2 (stay in place)
- Policy: π(Dig) = 0.85, π(Nap) = 0.15
- Discount: γ = 0.75
- Initial values: V(B1) = V(B2) = V(B3) = V(B4) = 0

**One synchronous Bellman update** — all old values (= 0) used on the right-hand side.

**Calculation:**

$$Q(B3, \text{Dig}) = R(B3\rightarrow B4) + \gamma \times V_{old}(B4) = 5 + 0.75 \times 0 = 5.0$$
$$Q(B3, \text{Nap}) = R(\text{Nap}) + \gamma \times V_{old}(B3) = 0.2 + 0.75 \times 0 = 0.2$$

$$V_{new}(B3) = \pi(\text{Dig}) \times Q(B3,\text{Dig}) + \pi(\text{Nap}) \times Q(B3,\text{Nap}) = 0.85 \times 5.0 + 0.15 \times 0.2 = 4.25 + 0.03 = 4.28$$

**Answer: 4.28**

---

### Q4 — 0 / 1 (Unanswered)
**Topic:** Compute updated V(B2)

**Calculation:**

$$Q(B2, \text{Dig}) = 3 + 0.75 \times V_{old}(B3) = 3 + 0.75 \times 0 = 3.0$$
$$Q(B2, \text{Nap}) = 0.2 + 0.75 \times V_{old}(B2) = 0.2 + 0.75 \times 0 = 0.2$$

$$V_{new}(B2) = 0.85 \times 3.0 + 0.15 \times 0.2 = 2.55 + 0.03 = 2.58$$

**Answer: 2.58**

> **Key insight:** Even though V(B3) was just computed as 4.28, synchronous updates use the OLD value (0) everywhere. Using 4.28 would be asynchronous/in-place — a different result and a different algorithm.

---

### Q5 — 0 / 1 (Unanswered)
**Topic:** Compute updated V(B1)

**Calculation:**

$$Q(B1, \text{Dig}) = 1 + 0.75 \times V_{old}(B2) = 1 + 0.75 \times 0 = 1.0$$
$$Q(B1, \text{Nap}) = 0.2 + 0.75 \times V_{old}(B1) = 0.2 + 0.75 \times 0 = 0.2$$

$$V_{new}(B1) = 0.85 \times 1.0 + 0.15 \times 0.2 = 0.85 + 0.03 = 0.88$$

**Answer: 0.88**

---

### Q6 — 0 / 1 (Unanswered)
**Topic:** Expected total worm haul starting from B1 after this update step

After one Bellman sweep from all-zero values, V(B1) captures the one-step lookahead (immediate expected reward, since all bootstrapped values were 0). The expected worm haul = $V_{new}(B1)$.

$$V_{new}(B1) = 0.85 \times (1 + 0) + 0.15 \times (0.2 + 0) = 0.88$$

**Answer: 0.88**

> After more sweeps the value converges to the true discounted return (a larger number). After just one sweep from zero, it reflects one step of lookahead only.

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
| 6 | Correct to skip | Bot B's model is specialized and cannot be reused | FALSE — a learned environment model can often be repurposed |

**Correct options:** {1, 3, 5}. Selected: 1✓ 3✓ 4✗ 5✓ → (3 correct − 1 wrong) / 3 × 2 = **0.67** ✓

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
| 5 | Wrong (selected) | Drone 1's Q-values represent the return under the optimal policy | FALSE — SARSA Q-values represent the BEHAVIOR policy return (with ε-exploration). Q-Learning converges to optimal Q-values |
| 6 | Correct to skip | Both guaranteed to learn the same final policy | FALSE — different policies, may not converge identically |

**Correct options:** {1, 2, 3, 4}. Selected: 1✓ 3✓ 4✓ 5✗ → (3 correct − 1 wrong) / 4 × 2 = **1.0** ✓

**Next attempt:** Select **1, 2, 3, 4**. Add option 2. Drop option 5.

---

### Q9 — 1 / 1 (Correct)
**Topic:** Which actions are greedy? (maximum Q-value)

**Context — Factory maintenance Q-values:**
- Q(Full Replacement) = 8.0 ← maximum
- Q(Partial Repair) = 7.5
- Q(Increase Cooling) = 8.0 ← maximum
- Q(Do Nothing) = 5.0
- ε = 0.2

Full Replacement and Increase Cooling both share the highest Q-value of 8.0 — correctly identified as greedy. ✓

---

### Q10 — 0 / 1 (Incorrect)
**Topic:** P(Full Replacement selected under epsilon-greedy policy)

**Real-life analogy:** Epsilon-greedy is like a restaurant regular: 80% of the time they order one of their two favourites (greedy), 20% of the time they pick randomly from the full menu. With 4 items and 2 favourites, each favourite gets (0.8/2) from exploitation plus (0.2/4) from random exploration.

**Calculation:** $\varepsilon = 0.2$, $k_{greedy} = 2$, $k_{total} = 4$. Full Replacement is greedy.

$$P(\text{greedy action}) = \frac{1 - \varepsilon}{k_{greedy}} + \frac{\varepsilon}{k_{total}} = \frac{0.8}{2} + \frac{0.2}{4} = 0.40 + 0.05 = 0.45$$

**Your answer: 0.0561 (wrong). Correct answer: 0.45**

---

### Q11 — 0 / 1 (Incorrect)
**Topic:** P(Do Nothing chosen under epsilon-greedy policy)

Do Nothing is non-greedy (Q = 5.0, not maximum). It only appears in the random exploration portion.

$$P(\text{non-greedy action}) = \frac{\varepsilon}{k_{total}} = \frac{0.2}{4} = 0.05$$

**Your answer: 0.0351 (wrong). Correct answer: 0.05**

> Both Q10 and Q11 answers suggest the epsilon-greedy formula was applied incorrectly — possibly using Q-values themselves as weights (softmax-style) rather than the ε-split formula. The correct formula ignores the actual Q-value magnitudes once greedy/non-greedy has been determined.

---

### Q12 — 0 / 1 (Incorrect)
**Topic:** P(Increase Cooling) when ε increases to 0.6

Increase Cooling remains greedy (Q = 8.0, still maximum). Only ε changes.

$$P(\text{IC}) = \frac{1 - \varepsilon}{k_{greedy}} + \frac{\varepsilon}{k_{total}} = \frac{0.4}{2} + \frac{0.6}{4} = 0.20 + 0.15 = 0.35$$

**Your answer: 0.1684 (wrong). Correct answer: 0.35**

> As ε increases from 0.2 → 0.6, P(IC) decreases from 0.45 → 0.35. More exploration = greedy actions get less weight. At ε = 0 (pure greedy): P = 0.5. At ε = 1 (pure random): P = 0.25.

---

## Question Set 3 — Game Theory

### Q13 — 0.67 / 2 (Partial)
**Topic:** Ada & Byron trading bots — repeated game, Folk Theorem

**Real-life analogy:** Two gas stations across the street: if both price high (collude), both earn well. If one cuts price it grabs short-term market share, but the other retaliates next week. This is the Folk Theorem: in a long-running repeated game, cooperation can be stable if players are patient enough.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | Correct (selected) | This is a repeated game where the Folk Theorem applies | TRUE — indefinitely repeated game with discounting = textbook Folk Theorem |
| 2 | Correct to skip | Undercutting once always leads to maximum profit in the long run | FALSE — punishment follows, long-run profits drop |
| 3 | Correct (selected) | A credible punishment strategy can make collusion stable | TRUE — threat of future undercutting deters defection today |
| 4 | Correct to skip | Strictly zero-sum because gain of one equals loss of the other | FALSE — both can gain (collusion) or both can lose (mutual undercutting). Not zero-sum |
| 5 | MISSED | In a strictly zero-sum game, Folk Theorem does not apply | TRUE — in a purely zero-sum game there are no cooperative gains to sustain; Folk Theorem gives only minimax outcomes |
| 6 | Wrong (selected) | If both bots discount the future heavily, collusion is more likely to hold | FALSE — heavy discounting = future punishments worth LESS → defection more tempting. Cooperation needs patient players (high δ) |

**Correct options:** {1, 3, 5}. Selected: 1✓ 3✓ 6✗ missed 5 → (2 correct − 1 wrong) / 3 × 2 = **0.67** ✓

**Next attempt:** Select **1, 3, 5**. Drop option 6. Add option 5.

---

### Q14 — 0 / 2 (Incorrect)
**Topic:** Huey & Dewey pirate subgame — two-stage game

**Real-life analogy:** Two business partners negotiating: stage 1 = share profits fairly or grab more. If both grab, they enter a costly legal fight (stage 2). The threat of a costly fight deters grabbing in stage 1 — this is backward induction in subgame perfect equilibrium (SPE).

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | Skip | Huey can use a mixed strategy to make Dewey indifferent | Depends on specific payoff values — skip unless confident |
| 2 | MISSED | The second-stage fight can be analyzed for subgame perfect equilibrium | TRUE — it is a proper subgame; SPE requires an equilibrium in every subgame |
| 3 | Correct to skip | Pure strategy always better than mixing | FALSE — mixing can be optimal in many games |
| 4 | Correct (selected) | A credible threat to fight can influence first-stage choices | TRUE — backward induction: if fighting is rational in stage 2, it deters stealing in stage 1 |
| 5 | Wrong (selected) | This pirate game is strictly zero-sum | FALSE — when both steal and fight, BOTH lose loot. Zero-sum requires one's gain to exactly equal the other's loss everywhere |
| 6 | Correct to skip | SPE requires randomizing in every subgame | FALSE — SPE can use pure strategies |

**Correct options:** {2, 4}. Selected: 4✓ 5✗ → (1 correct − 1 wrong) / 2 × 2 = **0** ✓

**Next attempt:** Select **2 and 4 only**. Deselect 5.

---

### Q15 — 0 / 2 (Incorrect)
**Topic:** Worst-case daily profit — pure minimax strategy (Spot A–F)

**Real-life analogy:** Huey is picking a market stand location. Dewey and Louie will try to undercut him — so Huey wants the location where even in the worst case (most hostile rivals), he still makes the most money. Minimax = pick the spot with the best worst-case.

**Payoff matrix (daily profit $):**

| Spot | R1 | R2 | R3 | R4 | Worst-case (min) |
|---|---|---|---|---|---|
| A | 950 | 800 | 450 | 200 | **200** |
| B | 900 | 700 | 600 | 400 | **400** |
| C | 1000 | 500 | 500 | 350 | **350** |
| D | 850 | 850 | 750 | 600 | **600** |
| E | 1200 | 700 | 400 | 250 | **250** |
| F | 1100 | 800 | 800 | 700 | **700 ← highest** |

**Calculation:**

For each spot, find min across all rival scenarios R1–R4.
Then choose the spot with the maximum of those minimums (maximin).

- Spot A: $\min(950, 800, 450, 200) = 200$
- Spot B: $\min(900, 700, 600, 400) = 400$
- Spot C: $\min(1000, 500, 500, 350) = 350$
- Spot D: $\min(850, 850, 750, 600) = 600$
- Spot E: $\min(1200, 700, 400, 250) = 250$
- Spot F: $\min(1100, 800, 800, 700) = 700$ ← maximum worst-case

**Minimax choice = Spot F, guaranteed worst-case = $700**

**Your answer: Spot C ($350 worst-case) — wrong. Correct answer: Spot F ($700)**

---

### Q16 — 0 / 1 (Incorrect)
**Topic:** Mixed strategy (40% Spot F, 60% Spot D) — expected worst-case daily profit

**Key point:** For a mixed strategy, the adversary picks the SINGLE rival scenario that minimises the weighted combination — not separate minimums per spot. Compute the blended payoff at every scenario first, then take the overall minimum.

**Calculation:**

For each rival scenario $j$:
$$0.4 \times \text{Payoff}(F, j) + 0.6 \times \text{Payoff}(D, j)$$

- R1: $0.4 \times 1100 + 0.6 \times 850 = 440 + 510 = 950$
- R2: $0.4 \times 800 + 0.6 \times 850 = 320 + 510 = 830$
- R3: $0.4 \times 800 + 0.6 \times 750 = 320 + 450 = 770$
- R4: $0.4 \times 700 + 0.6 \times 600 = 280 + 360 = 640$ ← minimum

$$\text{Expected worst-case} = \min(950, 830, 770, 640) = \$640$$

**Your answer: $3,190 — wrong. Correct answer: $640**

> $3,190 ≈ summing all individual cells (440+510+320+510+320+450+280+360). That is not the worst-case formula — you must blend first per scenario, then take the minimum across scenarios.

---

### Q17 — 1 / 1 (Correct)
**Topic:** Does mixed strategy give higher worst-case expected profit than pure minimax?

**Answer: No ✓**

Pure minimax (Spot F alone) guarantees $700 worst-case. The mixed strategy (40% F + 60% D) only guarantees $640 worst-case — strictly worse. Mixing can sometimes improve worst-case payoffs, but only when it creates indifference for the opponent across all scenarios. Here the adversary still exploits R4 where the blend yields only $640.

---

## Question Set 4 — Game Theory Continued

### Q18 — 1.2 / 2 (Partial)
**Topic:** Robot factory — Cooperate or Defect (infinitely repeated game)

**Real-life analogy:** Office shared printer: everyone could print huge jobs (defect), but if all cooperate and take turns, everyone works faster. Grim Trigger = "if you ever hog the printer, I'll never let you use it again." Subgame perfection ensures that threat is actually credible.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | Correct (selected) | Patient robots tolerate short-term losses to keep cooperation stable | TRUE — high discount factor → cooperation sustainable |
| 2 | MISSED | A robot fleet can use Grim Trigger (if any robot cuts the line, others refuse to share forever) | TRUE — valid strategy in infinitely repeated games; can sustain cooperation |
| 3 | Correct to skip | Nash equilibrium means robots always share tools fairly in every round | FALSE — many Nash equilibria exist in repeated games, including always-defect |
| 4 | MISSED | Subgame perfection ensures no robot can bluff about punishing defectors | TRUE — SPE requires every threatened punishment to be an actual Nash equilibrium of the subgame (credible) |
| 5 | Correct (selected) | Folk Theorem implies many ways of sharing fairly can be sustained as equilibria | TRUE — any payoff above minimax can be an equilibrium for sufficiently patient players |
| 6 | Correct (selected) | Robots can use threats like blocking access to tools in future rounds | TRUE — future denial of resources is the mechanism for enforcing cooperation |

**Correct options:** {1, 2, 4, 5, 6}. Selected: 1✓ 5✓ 6✓ (3 correct, 0 wrong) → 3/5 × 2 = **1.2** ✓

**Next attempt:** Select **1, 2, 4, 5, 6**. Add options 2 and 4. Avoid option 3.

---

### Q19 — 0.8 / 2 (Partial)
**Topic:** Mars rovers with Puddles (always defects)

**Real-life analogy:** Huey/Dewey/Louie = cooperative teammates in a group project. Puddles = the person who always free-rides. Tit-for-Tat punishes each time but can't reform Puddles. Pavlov keeps flip-flopping. Grim Trigger gives up on Puddles permanently. The others may need to coordinate in competitive tasks.

| Option | Status | Statement | Verdict |
|---|---|---|---|
| 1 | MISSED | Puddles's always-defect can destabilize cooperation assumptions for the other rovers | TRUE — strategies like TfT and Pavlov assume the opponent responds to incentives. Puddles breaks that assumption |
| 2 | MISSED | Louie's Pavlov may get trapped in constant switching against Puddles | TRUE — Pavlov cycle: Cooperate → Puddles defects → Louie loses → shifts to Defect → ties/wins → shifts back to Cooperate → repeat endlessly |
| 3 | MISSED | TfT punishes Puddles each time, but if Puddles never cooperates, the punishment has no effect | TRUE — TfT defects in response, but Puddles's behavior never changes. Punishment mechanism is ineffective against a committed always-defect player |
| 4 | Correct (selected) | Dewey's Grim Trigger means once Puddles defects, Dewey will never cooperate again | TRUE — definition of Grim Trigger |
| 5 | Correct to skip | Folk Theorem guarantees all four rovers maintain stable cooperation no matter how Puddles behaves | FALSE — Folk Theorem assumes rational, responsive players. An unconditional always-defect agent breaks the incentive structure |
| 6 | Correct (selected) | The other rovers may form an implicit alliance to punish Puddles in zero-sum contests | TRUE — direct zero-sum contests require coordination to contain Puddles |

**Correct options:** {1, 2, 3, 4, 6}. Selected: 4✓ 6✓ (2 correct, 0 wrong) → 2/5 × 2 = **0.8** ✓

**Next attempt:** Select **1, 2, 3, 4, 6**. Add options 1, 2, 3. Keep 4 and 6. Avoid option 5.

---

### Q20 — 0 / 1 (Incorrect)
**Topic:** Your total payoff over 3 rounds if you SWITCH to Ignore in Round 3

**Context — Pythagoruff dog game:**

You have a mischievous dog (Pythagoruff) who steals donuts. You can Guard (spray bottle) or Ignore. Pythagoruff can Steal or Stay.

**Payoff table (per round):**

| You \ Pythagoruff | Steal (S) | Stay (T) |
|---|---|---|
| Ignore (I) | You = −4, Dog = +3 | You = +2, Dog = 0 |
| Guard (G) | You = −1, Dog = −3 | You = −1, Dog = 0 |

**Fixed sequence:** Pythagoruff steals in every round.

**Calculation — switching to Ignore in R3:**

- Round 1: You Guard + Pythagoruff Steals → Your payoff = −1
- Round 2: You Guard + Pythagoruff Steals → Your payoff = −1
- Round 3: You Ignore + Pythagoruff Steals → Your payoff = −4

Total = −1 + (−1) + (−4) = −6

**Your answer: 0 — wrong. Correct answer: −6**

> You likely computed 0 by ignoring Rounds 1 and 2, or misread the Ignore+Steal payoff. Pythagoruff always steals — switching to Ignore in R3 exposes you fully with no spray bottle deterrent.

---

### Q21 — 0 / 1 (Incorrect)
**Topic:** Pythagoruff's total payoff over 3 rounds if you switch

**Calculation — switching to Ignore in R3:**

- Round 1: You Guard + Pythagoruff Steals → Pythagoruff = −3 (gets sprayed)
- Round 2: You Guard + Pythagoruff Steals → Pythagoruff = −3 (gets sprayed)
- Round 3: You Ignore + Pythagoruff Steals → Pythagoruff = +3 (steals freely, no spray)

Total = −3 + (−3) + 3 = −3

**Your answer: −6 — wrong. Correct answer: −3**

> −6 = −3 × 2 rounds, so you applied the Guard+Steal payoff to all three rounds and missed the R3 switch. In R3 you stop guarding, so Pythagoruff benefits (+3 instead of −3).

---

### Q22 — 1 / 1 (Correct)
**Topic:** Your total payoff if you do NOT switch in Round 3 (Guard instead)

**Calculation:**

- Round 1: Guard + Steal → You = −1
- Round 2: Guard + Steal → You = −1
- Round 3: Guard + Steal → You = −1

Total = −1 + (−1) + (−1) = −3 ✓

All three rounds you Guard, Pythagoruff always Steals, consistently −1 per round. **Correctly answered: −3 ✓**

---

### Q23 — 1 / 1 (Correct)
**Topic:** Pythagoruff's total payoff if you do NOT switch

**Calculation:**

- Round 1: Guard + Steal → Pythagoruff = −3
- Round 2: Guard + Steal → Pythagoruff = −3
- Round 3: Guard + Steal → Pythagoruff = −3

Total = −3 + (−3) + (−3) = −9 ✓

Pythagoruff gets sprayed every round. **Correctly answered: −9 ✓**

---

### Q24 — 0 / 1 (Incorrect)
**Topic:** Compute the difference in total payoff between switching and not switching. Should you have switched?

**Calculation:**

$$\text{Your payoff if you switch} = -6 \text{ (Q20 correct answer)}$$
$$\text{Your payoff if you don't switch} = -3 \text{ (Q22 correct answer)}$$

$$\text{Difference} = \text{switching} - \text{not switching} = -6 - (-3) = -3$$

Switching gives you −3 less than not switching. Switching makes you worse off.

**Your answer: Yes — wrong. Correct answer: No**

> You answered Yes because you computed Q20 = 0 (wrong), making it appear that switching (0) beats not switching (−3). But the true switching payoff is −6, which is worse than −3. Keeping Guard is the better strategy — it limits Pythagoruff's gains and minimises your losses every round.

---

## Complete Answer Key

| Q | Points | Your Answer | Correct Answer | Status |
|---|---|---|---|---|
| Q1 | 0/2 | B, C, D, E | A, B, E | ✗ |
| Q2 | 2/2 | Correct | Correct | ✓ |
| Q3 | 0/1 | Unanswered | **4.28** | ✗ |
| Q4 | 0/1 | Unanswered | **2.58** | ✗ |
| Q5 | 0/1 | Unanswered | **0.88** | ✗ |
| Q6 | 0/1 | Unanswered | **0.88** | ✗ |
| Q7 | 0.67/2 | 1, 3, 4, 5 | 1, 3, 5 | Partial |
| Q8 | 1/2 | 1, 3, 4, 5 | 1, 2, 3, 4 | Partial |
| Q9 | 1/1 | FR, IC | FR, IC | ✓ |
| Q10 | 0/1 | 0.0561 | **0.45** | ✗ |
| Q11 | 0/1 | 0.0351 | **0.05** | ✗ |
| Q12 | 0/1 | 0.1684 | **0.35** | ✗ |
| Q13 | 0.67/2 | 1, 3, 6 | 1, 3, 5 | Partial |
| Q14 | 0/2 | 4, 5 | 2, 4 | ✗ |
| Q15 | 0/2 | Spot C | **Spot F** | ✗ |
| Q16 | 0/1 | $3,190 | **$640** | ✗ |
| Q17 | 1/1 | No | No | ✓ |
| Q18 | 1.2/2 | 1, 5, 6 | 1, 2, 4, 5, 6 | Partial |
| Q19 | 0.8/2 | 4, 6 | 1, 2, 3, 4, 6 | Partial |
| Q20 | 0/1 | 0 | **−6** | ✗ |
| Q21 | 0/1 | −6 | **−3** | ✗ |
| Q22 | 1/1 | −3 | −3 | ✓ |
| Q23 | 1/1 | −9 | −9 | ✓ |
| Q24 | 0/1 | Yes | **No** | ✗ |

---

## Strategy for Attempt 2

All 24 questions now fully resolved. Targeting 30+ / 33:

| Question | Action | Correct Answer |
|---|---|---|
| Q1 | Select A, B, E only — drop C and D | — |
| Q3 | Enter value | **4.28** |
| Q4 | Enter value | **2.58** |
| Q5 | Enter value | **0.88** |
| Q6 | Enter value | **0.88** |
| Q7 | Select 1, 3, 5 — drop option 4 | — |
| Q8 | Select 1, 2, 3, 4 — add 2, drop 5 | — |
| Q10 | Enter value | **0.45** |
| Q11 | Enter value | **0.05** |
| Q12 | Enter value | **0.35** |
| Q13 | Select 1, 3, 5 — drop 6, add 5 | — |
| Q14 | Select 2, 4 only — drop 5 | — |
| Q15 | Select Spot F | **Spot F** |
| Q16 | Enter value | **$640** |
| Q18 | Select 1, 2, 4, 5, 6 — add 2 and 4 | — |
| Q19 | Select 1, 2, 3, 4, 6 — add 1, 2, and 3 | — |
| Q20 | Enter value | **−6** |
| Q21 | Enter value | **−3** |
| Q24 | Select No | **No** |
