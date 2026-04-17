Here is the concise list of the core Reinforcement Learning theoretical concepts you are evaluating in this assignment:

*   **Markov Decision Processes (MDPs):** Formulating a problem using states, actions, transition probabilities ($T$), and rewards ($R$).
*   **Model-Based vs. Model-Free RL:** Having perfect mathematical knowledge of the environment (DP) versus learning strictly through trial-and-error interaction (TD Learning).
*   **State Space Discretization:** The challenge of mapping an infinite continuous environment (CartPole physics) into a finite, discrete grid so tabular algorithms can solve it.
*   **Dynamic Programming Convergence:** The algorithmic difference between propagating values incrementally backward (Value Iteration) versus globally evaluating and updating a strategy at once (Policy Iteration).
*   **On-Policy vs. Off-Policy Learning:** How an algorithm updates its expectations. **SARSA** updates based on the exact action it takes (including random ones). **Q-Learning** updates based on the theoretical best action, regardless of what it actually executed.
*   **Exploration vs. Exploitation ($\epsilon$-greedy):** The tradeoff of exploring new random actions versus exploiting known good actions to maximize reward (managed via learning rate $\alpha$ and epsilon decay $\epsilon$).
*   **Empirical Model Estimation & Sparsity:** Calculating transition probabilities via random rollouts, and handling "unseen" states using mathematical smoothing (Laplace +1).