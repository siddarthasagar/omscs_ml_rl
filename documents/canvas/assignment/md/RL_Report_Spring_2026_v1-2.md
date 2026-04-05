# Reinforcement Learning Report
**CS7641: Machine Learning — Spring 2026**

---

## 1. Assignment Weight

The assignment is worth **12% of the total points**.

Read everything below carefully as this assignment has changed term-over-term.

---

## 2. Objective

In some sense, we have spent the semester thinking about machine learning techniques for various forms of function approximation. It's now time to think about using what we've learned in order to allow an agent of some kind to act in the world more directly. This assignment asks you to consider the application of some of the techniques we've learned from reinforcement learning to make decisions.

---

## 3. Procedure

### 3.1 The Problems Given to You

You are being asked to explore Markov Decision Processes (MDPs) using a combination of dynamic programming and reinforcement learning approaches.

You will analyze and solve two predefined MDPs:

- **Blackjack (Discrete and Stochastic)** — A turn-based card game where actions affect future outcomes probabilistically. Gym Environment: `Blackjack-v1`
- **CartPole (Continuous and Deterministic)** — A physics-based balancing problem where discretization of the state space is required for dynamic programming methods. Gym Environment: `CartPole-v1`

These MDPs have distinct characteristics: Blackjack is inherently discrete and stochastic, whereas CartPole is continuous and requires state discretization. Your analysis should consider how these differences impact algorithm performance.

Solve both MDPs using:

- **Value Iteration (VI) and Policy Iteration (PI):** Compare convergence rates and assess how discretization influences results in the CartPole environment.
- **SARSA:** Implement and compare this core on-policy model-free method.
- **Q-Learning:** Implement any standard off-policy Q-learning variant (tabular with your chosen exploration strategy is fine) and compare against SARSA.

Analyze and compare results:

- How many iterations does VI vs. PI take to converge?
- Which method converges faster? Why?
- How does discretization affect the CartPole solution?
- How do SARSA and Q-Learning compare in performance (sample efficiency, stability, final return)?
- What exploration strategies did you use (e.g., ε-greedy, softmax), and how did they affect learning for each method?

### Extra Credit Opportunity (up to 5 points)

Implement one ablation/component from the Rainbow DQN framework on CartPole (e.g., Double Q-learning, Dueling networks, Prioritized replay, Noisy nets, n-step returns, or Distributional RL/C51). Briefly compare your ablated variant to a vanilla DQN baseline and explain the observed effect.

- Clearly describe how DQN differs from a tabular Q-Learning implementation.
- Discuss any challenges you encountered with function approximation.
- Compare your DQN results to your tabular methods and explain any performance differences.

This extra credit is entirely optional but can be a rewarding challenge if you choose to take it on.

Additional Reading: If using a variant of DQN, you may refer to the [2017 Rainbow DQN study](Link to study).

> **Note:** This extra credit is optional but requires additional time and computation.

**Analysis writeup is limited to 8 pages.** The page limit does include your citations. Anything past 8 pages will not be read. Please keep your analysis as concise while still covering the requirements of the assignment.

Your report must be written in **LaTeX on Overleaf**. You can create an account with your Georgia Tech email (e.g. gburdell3@gatech.edu). When submitting your report, you are required to include a **READ ONLY** link to the Overleaf Project. If a link is not provided in the report or Canvas submission comment, **5 points will be deducted** from your score. For a starting template, please use the **IEEE Conference template**.

---

### 3.2 Acceptable Libraries

The algorithms used in this assignment are straightforward enough that many students may choose to implement substantial portions themselves. At the same time, there are several Python libraries that can help with environment interaction and selected algorithmic components. You may use external libraries, but it remains your responsibility to ensure that your work is reproducible and that your analysis is your own.

- [bettermdptools (python)](https://github.com/jlm429/bettermdptools)
- [Gymnasium (python)](https://gymnasium.farama.org/)
- [PyMDPtoolbox (python)](https://pymdptoolbox.readthedocs.io/)

Students are encouraged to use libraries thoughtfully. For the core of the assignment, your emphasis should remain on understanding and analyzing Value Iteration, Policy Iteration, SARSA, and Q-Learning rather than outsourcing the full experimental pipeline to a black-box implementation.

---

## 4. Submission Details

All scored assignments are due by the time and date indicated on Canvas (Eastern Time, ET). Canvas displays times in your local time zone if your profile is set correctly; grading deadlines are enforced in ET.

All assignments are due at **11:59:00 PM ET** on the final Sunday of the unit. Submissions received by 7:59:00 AM ET the next morning are accepted without penalty.

### Late Penalty

After the grace window, the score is reduced **20 points per calendar day**. Treat 11:59 PM ET as your real deadline; allow time for upload and verification.

### What to Submit

You will submit **two PDFs:**

1. `RL_Report_{GTusername}.pdf` — your report (Overleaf).
2. `REPRO_RL_{GTusername}.pdf` — your reproducibility sheet, which must include:
   - A READ-ONLY link to your Overleaf project.
   - A GitHub commit hash (single SHA) from the final push of your code.
   - Exact run instructions to reproduce results on a standard Linux machine (environment setup, commands, data paths, and random seeds).

Additional requirements:

- Include the READ-ONLY Overleaf link in the report or Canvas submission comment. Do not send email invitations.
- Use the GT Enterprise GitHub for course-related code and GT Enterprise Overleaf for writing your report.
- Provide sufficient instructions to retrieve code and data (Canvas paths and file names are sufficient).

### Report Contents

**Brief description of the MDPs**
- Provide an overview of Blackjack and CartPole, explaining their differences in terms of state space (discrete vs. continuous), action space, and reward structure.
- Discuss why these environments are interesting to study in the context of Markov Decision Processes (MDPs).
- You must contain a **hypothesis** about your experimentation. This is open-ended as each of you will have a variety of perspectives on the features and attributes of the data that may or may not perform a certain way given the required algorithms. Whatever hypothesis you choose, you will need to back it up with experimentation and thorough discussion. It is not enough to just show results.

**Explanation of methods**
- Describe Value Iteration (VI) and Policy Iteration (PI) in detail, explaining how they work and how you applied them to the two MDPs.
- Explain how SARSA and Q-Learning function (and DQN if attempting the EC).
- For CartPole, describe the discretization strategy.

**Analysis of results**
- Compare the performance of VI vs. PI:
  - How many iterations were needed for convergence?
  - Which algorithm converged faster? Why?
  - Did they produce the same optimal policy?
- Compare SARSA:
  - How did the algorithm perform in terms of reward maximization?
  - Which exploration strategy was used, and how did it affect learning?
  - If a deep RL approach (e.g., DQN) was used, how did it compare to tabular Q-Learning?
- Discuss the effect of discretization on CartPole's solution:
  - Did different levels of discretization impact performance?
  - Were there trade-offs between computational efficiency and accuracy?

**Visualizations and Data-Driven Evidence**
- Graphs showing convergence rates for different algorithms.
- Policy heatmaps or action distributions (for Blackjack).
- Learning curves for SARSA (showing cumulative rewards over time).
- CartPole balancing performance (e.g., episode length over training iterations).

**Extra Credit Analysis (if completed)**
- Clearly describe your DQN implementation: network design, experience replay, target networks, or any variants used.
- Discuss challenges you faced with training stability, convergence, or hyperparameter tuning.
- Compare DQN performance with your tabular Q-Learning results, explaining any key differences.

**Conclusion**
- Summarize key findings.
- Discuss challenges encountered and possible improvements.
- Reflect on how different RL methods performed and what insights were gained.

> **Note on figures:** Figures should remain legible at 100% zoom. Do not try to squish figures together in specific sections where axis labels become 8pt font or less. We are looking for clear and concise demonstration of knowledge and synthesis of results. Any paper that solely has figures without formal writing will not be graded.

You may submit the assignment as many times as you wish up to the due date, but we will only consider your **last submission** for grading purposes.

---

## 5. Feedback Requests

When your assignment is scored, you will receive feedback explaining your errors and successes in some level of detail. This feedback is for your benefit, both on this assignment and for future assignments. It is considered a part of your learning goal to internalize this feedback.

If you are confused by a piece of feedback, please start a private thread on Ed and we will jump in to help clarify.

> Since this will be the last assignment of the term and too close to the end of the term, we will not conduct the Reviewer Response for this report.

---

## 6. Plagiarism and Proper Citation

The easiest way to fail this class is to plagiarize. Using the analysis, code, or graphs of others in this class is considered plagiarism. We care about your analysis: it must be original and grounded in your own experiments.

If you copy any amount of text from other students, websites, or any other source without proper attribution, that is plagiarism. Citing is required but does not permit copying large blocks of text. All citations must use a consistent style (IEEE, MLA, or APA).

We report all suspected cases of plagiarism to the Office of Student Integrity. Students who are under investigation are not allowed to drop from the course in question, and the consequences can be severe, ranging from a lowered grade to expulsion from the program.

### LLMs (Disclosure Required)

We treat AI based assistance the same way we treat collaboration with people. You may discuss ideas and seek help from classmates, colleagues, and AI tools, but **all submitted work must be your own**. The goal of reports is synthesis of analysis, not merely getting an algorithm to run.

Every submission must include an **AI Use Statement**. List the tools used and what they assisted with, and confirm that you reviewed and understood all assisted content.

**Allowed with disclosure:** brainstorming, outlining, grammar and clarity edits, code generation, code refactoring, and debugging.

**Not allowed:** submitting AI written analysis, conclusions, or figures as your own; fabricating results or citations; paraphrasing AI or prior work to evade checks.

Example Statement (at the very end of the report before References):

> *"AI Use Statement. I used ChatGPT and Visual Studio Code Copilot to brainstorm and outline sections of the report, generate and refactor small code snippets, debug an indexing issue, and edit grammar and clarity throughout. I reviewed, verified, and understand all assisted content."*

### How to Attribute & Cite

**Blackjack MDP:** Sutton & Barto (2018), *Reinforcement Learning: An Introduction* (2nd ed.), Example 5.3 "Blackjack," online book. Gym Environment: `Blackjack-v1`.

**CartPole MDP:** Barto, Sutton & Anderson (1983), "Neuronlike adaptive elements that can solve difficult learning control problems," *IEEE Transactions on Systems, Man, and Cybernetics* 13(5):834–846, doi:10.1109/TSMC.1983.6313077. Gym Environment: `CartPole-v1`.

### In-Text Citation Styles

You may use MLA, APA, or IEEE; pick one style and stay consistent across the paper (in-text and references).

Examples using the hotel paper:
- **APA:** ". . . (António, Almeida, & Nunes, 2019)." or "António et al. (2019) . . . "
- **MLA:** ". . . (António, Almeida, and Nunes 2019)." or "António, Almeida, and Nunes argue . . . "
- **IEEE:** ". . . [antonio2019hotel]." (numbered, bracketed citations like [1]; reference list ordered by first appearance)

Include the full reference entry in your bibliography. In LaTeX, use BibTeX/BibLaTeX with an appropriate style (e.g., `style=apa` or `style=ieee`). Tip: In Google Scholar, click "Cite" → "BibTeX" to copy a starter entry, then verify authors, capitalization, and DOI.

---

## 7. Version Control

- **03/07/2026** — v1 — TJL final updates for Spring 2026 and posting to class.

*Assignment description written by Theodore LaGrow.*
