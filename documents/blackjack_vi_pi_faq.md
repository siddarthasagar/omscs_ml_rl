# Blackjack VI & PI: Report Writing FAQ / Talking Points

This document summarizes the key conceptual insights regarding Value Iteration (VI) and Policy Iteration (PI) on the Blackjack environment. Use these points to construct the analysis sections of your final report.

## 1. Why are the Y-axes different on the Convergence Plot?
* **Value Iteration (VI)** uses `max |ΔV|` on a log scale. VI only cares about the value function and halts solely when the maximum change across all states drops below a mathematical threshold ($\delta = 1e-06$).
* **Policy Iteration (PI)** uses `# policy changes` on a log scale. PI defines convergence as the moment the policy stabilizes (zero states flip their action).
* **The Takeaway:** By separating these metrics, the visualization perfectly illustrates the fundamental difference in how these two algorithms define "convergence." 

## 2. Why is there only one Policy Heatmap? Where is the PI heatmap?
* You do not need two heatmaps because **VI and PI achieved 100.0% policy agreement**. 
* VI and PI are just two different mathematical paths to find the exact same true optimal policy. Because Blackjack is a fully-known, finite MDP, both algorithms perfectly solved it. Plotting a second heatmap would be entirely redundant.

## 3. Why does the Y-axis (Player Hand) start at 4 for Hard Hands, but 12 for Soft Hands?
* **Hard Hand (No Usable Ace):** The absolute lowest hand you can be dealt without an Ace is a pair of 2s, giving a minimum sum of 4.
* **Soft Hand (Usable Ace):** A "Soft Hand" means you are holding an Ace that is being counted as an 11. The lowest possible card you could hold alongside it is a 1. Therefore, $11 + 1 = 12$. It is mathematically impossible to have a Soft Hand with a sum lower than 12.
* **The Takeaway:** The differing Y-axes prove that the code accurately reflects the true geometric boundaries of the Blackjack state space.

## 4. Is "Heatmap" the correct term? Does it satisfy "action distributions" in the rubric?
* The rubric explicitly asks for *"Policy heatmaps OR action distributions."* Therefore, generating a policy heatmap perfectly satisfies the assignment requirement.
* Technically, because the policy is deterministic (binary actions: Hit or Stick), it is a **"Discrete Policy Map"** or **"Decision Boundary Plot."** However, in RL literature and coursework, plotting discrete actions on a 2D state grid is universally accepted and referred to as a Policy Heatmap.

## 5. What are the key insights to mention about the Optimal Policy Heatmap?
When analyzing the optimal strategy the agent learned, make these three points:
1. **Exploiting Dealer Busts:** Against weak dealer cards (2 through 6), the agent learns to play very conservatively (Sticking on 13-16). It learns to let the dealer bust themselves.
2. **The Safe Threshold:** Across both plots, the policy universally shifts to "Stick" once the player reaches a sum of 17+, as the risk of drawing a bust card becomes statistically too high.
3. **The Usable Ace "Safety Net":** The Soft Hand policy is drastically more aggressive (Hitting on Soft 17 and sometimes Soft 18). The agent learned that the Ace acts as a safety net (it can revert to a 1), allowing it to take risks it would never take with a Hard Hand.

## 6. If both VI and PI perfectly solve the MDP, what is the trade-off?
The primary difference is the **computational work done per iteration**:
* **Value Iteration (VI)** is "lazy" per iteration. It does a simple one-step lookahead update. Because the math is lightweight, it is fast per iteration, but requires **more iterations** (e.g., 9) to propagate the values and converge.
* **Policy Iteration (PI)** is "exhaustive" per iteration. It completely evaluates the current policy (which involves solving a system of equations or heavy looping) before making a greedy update. Because it does massive amounts of math to evaluate the policy, it takes **very few iterations** (e.g., 3) to find the optimal solution.
* **The Takeaway:** In a tiny environment like Blackjack (290 states), PI's heavy math is trivially fast, making both algorithms highly effective. However, in massive real-world environments with millions of states, PI's exhaustive policy evaluation becomes computationally impossible, making VI the preferred scalable choice.

## 7. How do I defend the Hyperparameters and Baseline Constants?
You should explicitly defend your chosen constants by referencing the fact that they are mathematically conservative baselines which you empirically verified via your Hyperparameter Validation Sweep.
* **`GAMMA = 0.99` (The Discount Factor):** This defines the agent as "far-sighted," caring almost equally about winning the game at the end as intermediate steps. *Defense:* Your HP sweep tested `[0.85, 0.90, 0.95, 0.99]` and empirically proved that 0.99 yielded the highest mean evaluation return by preventing the agent from being too short-sighted.
* **`DELTA = 1e-6` (The Convergence Threshold):** A strictly conservative threshold guaranteeing that the value function is smoothed to the 6th decimal place. *Defense:* Your HP sweep tested `[1e-2, 1e-3, 1e-4, 1e-6]`. While looser deltas converge faster, 1e-6 guarantees mathematical perfection for the final policy.
* **`BJ_EVAL_EPISODES_MAIN = 1_000` (Evaluation Budget):** Blackjack is highly stochastic. *Defense:* 1,000 episodes is large enough to utilize the Law of Large Numbers to smooth out variance and give statistical confidence to the mean returns, while remaining computationally fast. (Conversely, `BJ_EVAL_EPISODES_HP = 500` was chosen for fast screening).
* **`SEEDS = [42, 43, 44, 45, 46]`:** *Defense:* This directly fulfills the assignment FAQ contract requiring ~5 independent seeds per compared model to defend against statistical flukes and ensure strict reproducibility.
