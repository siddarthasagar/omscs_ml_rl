# RL Unit Quiz Practice
## CS 7641: Machine Learning

---

## Question 1 — Markov Decision Processes

### Part 1. Policies in Reinforcement Learning (MCMA)

You are training an agent to navigate a maze with multiple goal states. The agent uses a stochastic policy that maps states to probability distributions over actions.

**Which of the following statements about policies are true? Select all that apply.**

- A. A deterministic policy always selects the same action for a given state.
- B. A stochastic policy can assign non-zero probability to multiple actions in the same state.
- C. The optimal policy is always unique for any MDP.
- D. A policy can be evaluated by computing its expected return starting from each possible state.
- E. The policy improvement theorem guarantees that a better policy can always be found after each iteration.
- F. A random policy that chooses all actions uniformly is generally optimal for large state spaces.

---

### Part 2. Rewards in Reinforcement Learning (MCMA)

An agent is learning to maximize total reward in an environment with delayed and sparse rewards.

**Which of the following statements about rewards are true? Select all that apply.**

- A. Shaping rewards can help the agent learn faster by providing intermediate signals.
- B. The reward signal directly defines what the agent should optimize for.
- C. Adding random noise to the reward always improves exploration.
- D. Sparse rewards can make it more difficult for the agent to learn an optimal policy.
- E. Discounting future rewards too heavily can cause the agent to ignore long-term consequences.
- F. The total return is defined as the sum of immediate rewards only, without any consideration of future rewards.

---

### Part 3. Bellman Update

A robot explorer travels along a straight path with 4 states in sequence:

$$S_1 \rightarrow S_2 \rightarrow S_3 \rightarrow S_4 \text{ (terminal)}$$

The robot has only one action: **"Go Forward"**.

**Rewards for each step:**
- $S_1 \rightarrow S_2$: +1
- $S_2 \rightarrow S_3$: +2
- $S_3 \rightarrow S_4$: +4

The robot follows a fixed policy: always "Go Forward".

**Discount factor:** $\gamma = 0.8$

**Initial value estimates:** $V(S_1) = V(S_2) = V(S_3) = V(S_4) = 0$

**Questions:**

a) Write the general Bellman Expectation Equation for this policy.

b) Compute the updated value for V(S3).

c) Compute the updated value for V(S2).

d) Compute the updated value for V(S1).

e) If the robot starts in S1, what is its expected total return after this one update step?
   *(Give only your final numerical answer, rounded to two decimal places.)*

---

## Question 2 — Reinforcement Learning

### Part 1. Q-Learning (MCMA)

You are training an agent using Q-Learning in a grid world with discrete states and actions. The agent uses the standard Q-Learning update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big]$$

**Which of the following statements about Q-Learning are true? Select all that apply.**

- A. Q-Learning is an off-policy method because it learns the optimal policy independently of the agent's behavior policy.
- B. Q-Learning updates require a model of the environment to simulate next states and rewards.
- C. A high learning rate (α) can make Q-values converge faster but may increase instability.
- D. If the agent always picks the greedy action, it may fail to visit all states sufficiently often.
- E. Q-Learning always converges to the optimal Q-values, even with a non-decaying learning rate and purely greedy actions.
- F. The Q-values represent the expected discounted return when following the behavior policy used during training.

---

### Part 2. Greedy vs. Exploratory Strategies (MCMA)

An agent uses an ε-greedy strategy to balance exploitation and exploration while learning an optimal policy.

**Which of the following statements about exploration are true? Select all that apply.**

- A. An ε-greedy policy chooses the best-known action with probability 1 − ε.
- B. As ε approaches zero, the agent explores more frequently.
- C. A fixed, non-zero ε ensures persistent exploration even after convergence.
- D. Greedy policies always perform better than exploratory policies during training.
- E. Softmax action selection uses a probability distribution weighted by action values instead of a fixed ε.
- F. Increasing exploration generally reduces the variance of the agent's performance.

---

### Part 3. Greedy Exploration for Q-Learning

A rice farmer uses reinforcement learning to decide how to irrigate a field during the growing season. At a certain decision point, the soil moisture is moderate. The farmer's Q-value estimates for the three actions are:

- $Q(\text{Heavy Irrigation}) = 4.5$
- $Q(\text{Moderate Irrigation}) = 5.0$
- $Q(\text{No Irrigation}) = 3.0$

The farmer uses an $\varepsilon$-greedy policy with **$\varepsilon = 0.2$**.

**Questions:**

a) Which action is the greedy action?

b) What is the probability that the system selects the greedy action under this ε-greedy policy?

c) What is the probability that the system selects Heavy Irrigation under this ε-greedy policy?

d) Suppose the farmer increases exploration by setting **ε = 0.5**. What is the new probability of choosing the greedy action?

e) Explain briefly why setting ε too high could hurt overall performance.

---

## Question 3 — Game Theory

### Part 1. Pure vs. Mixed Strategies (MCMA)

A player is involved in a finite, simultaneous-move game with two other players. The player can choose between pure strategies (choosing one action with certainty) or mixed strategies (randomizing actions with probabilities).

**Which of the following statements about pure and mixed strategies are true? Select all that apply.**

- A. A pure strategy assigns probability 1 to a single action.
- B. A mixed strategy allows a player to assign non-zero probabilities to multiple actions.
- C. In every finite game, there is always a pure-strategy Nash equilibrium.
- D. A mixed-strategy equilibrium can exist even when no pure-strategy equilibrium exists.
- E. A mixed strategy can never yield a higher expected payoff than any pure strategy.
- F. A mixed-strategy equilibrium requires that players are indifferent between the pure strategies in their support.

---

### Part 2. Prisoner's Dilemma (MCMA)

Two robots repeatedly interact in a shared environment. In each episode, they can choose to **Cooperate** (share battery power and map information) or **Defect** (withhold resources and sabotage the other).

**The rewards work as follows:**
- If both robots cooperate: each gets a moderate energy boost and better mapping accuracy.
- If one robot defects while the other cooperates: the defector gains extra energy while the cooperator loses resources.
- If both defect: both waste energy defending themselves and mapping is less accurate.

The robots learn through reinforcement learning with rewards and keep track of each other's past actions.

**Which of the following statements about this scenario are true? Select all that apply.**

- A. In a single interaction, defection is the dominant strategy for each robot.
- B. Over multiple episodes, the robots may learn to cooperate if future rewards are valued highly enough.
- C. Keeping track of the other robot's past actions can help each robot punish defection.
- D. The best policy in the repeated setting is always to defect in every episode.
- E. Strategies like tit-for-tat can emerge to support mutual cooperation over time.
- F. If the robots only care about immediate rewards, they will never choose to cooperate.

---

### Part 3. Minimax

Huey, Dewey, and Louie are three young vendors competing to sell ice cream along a long beach. Huey must choose one of 6 ice cream stand locations along the beach. After Huey picks a spot, Dewey and Louie pick their stand spots to **minimize Huey's daily profit**.

Huey wants to **maximize his minimum guaranteed profit**.

| Option | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| A | $500 | $350 | $200 | $150 |
| B | $450 | $400 | $300 | $200 |
| C | $600 | $250 | $200 | $200 |
| D | $450 | $450 | $450 | $300 |
| E | $550 | $350 | $350 | $100 |
| F | $500 | $400 | $400 | $350 |

**Questions:**

a) Compute the worst-case daily profit for each stand option.

b) Which stand should Huey choose under the minimax strategy?

---

## Question 4 — Game Theory Continued

### Part 1. Folk Theorem (MCMA)

A team of autonomous robots must repeatedly share limited charging stations while patrolling a large area. Each robot can choose to **cooperate** (wait its turn) or **defect** (jump the queue). They interact infinitely over time and monitor each other's choices.

**Which of the following statements about this scenario are true? Select all that apply.**

- A. The Folk Theorem says any payoff that is feasible and individually rational can be sustained as an equilibrium in an infinitely repeated game.
- B. A Subgame Perfect Equilibrium (SPE) requires strategies that form a Nash equilibrium in every subgame, including after any history.
- C. Any Nash equilibrium in this repeated robot scenario is automatically subgame perfect.
- D. A Grim Trigger strategy can help robots enforce cooperation by punishing queue-jumping forever.
- E. The Folk Theorem applies equally to finite robot interactions as it does to infinite ones.
- F. Threats of losing access to shared stations in the future can sustain cooperative robot behavior.

---

### Part 2. Pavlov and Zero-Sum (MCMA)

A band of pirates repeatedly divides treasure after raids. Each pirate can cooperate (split fairly) or defect (take extra). One pirate uses a **Pavlov** strategy: repeat last split if it was good, change if cheated. They compare this with zero-sum contests over the loot.

**Which of the following statements about this scenario are true? Select all that apply.**

- A. Pavlov is also known as Win-Stay, Lose-Shift.
- B. Pavlov always produces the same results as Tit-for-Tat in repeated pirate bargains.
- C. In a zero-sum pirate contest, one pirate's gain is exactly equal to another pirate's loss.
- D. Zero-sum pirate games always have at least one pure-strategy Nash equilibrium.
- E. Pavlov can sometimes maintain fair splitting under noise better than Tit-for-Tat.
- F. Minimax strategies are used to compute equilibria in zero-sum pirate contests.

---

### Part 3. Tit-for-Tat, Pavlov, and Folk Theorem

Your cat, **Adrian Schrody**, loves knocking your favorite mug off the counter at night. Now you have a spray bottle to discourage him.

**Actions per round:**
- You: **Ignore (I)** or **Guard (G)**
- Adrian: **Paw (P)** or **Stay (S)**

**Payoffs per round:**
- Ignore + Paw: Mug falls → You: −2, Adrian: +2
- Ignore + Stay: Mug safe → You: +1, Adrian: 0
- Guard + Paw: Mug safe + Spray → You: 0, Adrian: −1
- Guard + Stay: Mug safe → You: −1, Adrian: 0

**3-Round Sequence:**
1. Round 1: Adrian Paws, You Guard (Spray)
2. Round 2: Adrian Paws, You Ignore
3. Round 3: Adrian Stays, You Ignore

**Questions:**

a) Compute your total payoff and Adrian's total payoff over 3 rounds.

b) Based on the sequence, would Tit-for-Tat or Pavlov better discourage pawing? Justify numerically.

c) Does this outcome support the Folk Theorem idea that cooperation can be sustained with credible punishment?
