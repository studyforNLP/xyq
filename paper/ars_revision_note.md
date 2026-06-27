# ARS revision note

## Paper configuration record

| Parameter | Value |
|---|---|
| Paper type | Empirical algorithm paper / conference-style journal draft |
| Field | Project scheduling, reinforcement learning, dynamic decision optimization |
| Main research question | Can rollout cost-to-go assisted Q-learning improve multi-milestone multi-skill interruptible project scheduling compared with heuristic dispatching, rollout heuristic and plain Q-learning? |
| Output language | English revised manuscript and Simplified Chinese manuscript |
| Citation status | References are not fabricated; citation placeholders are retained for later verification |
| Main evidence | J10-J60 comparison experiments, convergence curves, runtime analysis |
| Main limitation | RQL improves solution quality but has high training cost and does not fully converge on J60 under the current budget |

## Central thesis

RQL is effective because it combines learned action preferences with rollout-based forward evaluation: Q-learning supplies reusable state-action preferences, while cost-to-go rollout reduces the risk of locally myopic decisions when Q values are close or sparse global rewards make training unstable.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| RQL improves solution quality over RH and deterministic heuristics | RQL obtains the lowest AOTV on J10, J20, J30 and J60 | Supported by current experiments |
| Rollout assistance is necessary beyond plain Q-learning | QL deteriorates strongly on J30 and J60, while RQL remains competitive | Supported by current experiments |
| RQL is not a low-cost dispatching heuristic | RQL runtime is much higher than RH and deterministic heuristics, especially on J60 | Supported by current experiments |
| RQL convergence is scale-dependent | J10-J20 converge more clearly; J30 improves with variance; J60 improves but does not meet Q-difference threshold | Supported by current convergence results |
| The method is general for all large-scale scheduling settings | Current data do not fully support this claim | Do not claim without more evidence |

## Revision strategy

1. Strengthen the research problem: emphasize the combined difficulty caused by skill heterogeneity, task interruption and multiple milestone penalties.
2. Separate algorithmic contribution from empirical effect: explain what RQL changes in the decision process before reporting performance.
3. State runtime cost explicitly instead of hiding it.
4. Avoid claiming global optimality or universal convergence.
5. Keep citation placeholders until references are verified.

