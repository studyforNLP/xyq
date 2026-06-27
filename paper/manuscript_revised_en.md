# Rollout-Q-learning for Multi-milestone Multi-skill Interruptible Project Scheduling

## Abstract

Multi-skill project scheduling becomes difficult when tasks are interruptible, processing times depend on worker skills, and several intermediate milestones carry delay penalties. In this setting, a dispatching decision must balance immediate resource use with future milestone risk. Fixed heuristic rules are fast, but they cannot adapt their decision preference across repeated scheduling episodes. Plain Q-learning can learn from experience, but sparse terminal penalties and a large action space often make the learned Q table unstable. This paper proposes a rollout-Q-learning (RQL) algorithm for multi-milestone multi-skill interruptible project scheduling. The method combines a six-dimensional discrete state representation, complete task-worker action tuples, Q-learning updates, rollout cost-to-go feedback, checkpoint selection and beam-search decoding. Computational experiments on J10-J60 instances show that RQL obtains the lowest average total weighted delay among the tested methods. Compared with the rollout heuristic baseline, RQL reduces AOTV by 10.7%, 16.3%, 7.6% and 10.6% on J10, J20, J30 and J60, respectively. The comparison with plain Q-learning indicates that rollout cost-to-go is essential for maintaining solution quality as the problem scale increases. Runtime and convergence analyses also show a clear boundary: RQL is a solution-quality-oriented algorithm, not a low-cost dispatching rule, and large-scale convergence remains limited by rollout simulation cost.

**Keywords**: project scheduling; reinforcement learning; rollout algorithm; Q-learning; multi-skill scheduling; milestone delay; interruptible tasks

## 1. Introduction

Project scheduling in knowledge-intensive service environments often involves heterogeneous personnel, precedence-constrained tasks and delivery milestones. A software project, engineering service project or technical consulting project may contain tasks that can only be executed by workers with specific skills. Even when several workers are feasible, their processing times can differ because skill levels are heterogeneous. The scheduler must therefore decide not only which task to execute, but also which worker should execute it.

Milestone delivery adds another layer of difficulty. In many projects, the final completion time is not the only performance measure. Intermediate milestones may correspond to contract deliverables, customer review points or integration events. Delaying one milestone may be costly even if the whole project later recovers. A scheduling algorithm should therefore account for weighted delay penalties distributed across the project network rather than only optimizing makespan.

This study considers an additional feature: tasks are interruptible. When a running task is interrupted, its remaining processing time and original worker must be recorded. The task can later resume only with the same worker. This assumption reflects service settings where knowledge continuity or responsibility constraints prevent arbitrary reassignment after interruption. However, it also couples current resource decisions with future feasibility, because an interruption creates a future resume obligation.

Heuristic dispatching rules offer a natural baseline for this problem. Rules such as shortest processing time, earliest completion time, slack-based priority and maximum-successor priority are easy to implement and computationally cheap. Their limitation is that their priority logic is fixed. A rule that works well for one project state may be inappropriate when milestone urgency, worker utilization and interrupted tasks change. Reinforcement learning provides a mechanism for learning state-dependent preferences, but plain Q-learning faces two obstacles in this setting. First, the main objective is observed through delayed milestone penalties, so the learning signal is sparse. Second, the action space includes task-worker combinations, making many candidate actions difficult to distinguish during training.

To address these obstacles, this paper proposes a rollout-Q-learning (RQL) algorithm. The core idea is to let Q-learning learn reusable action preferences while rollout cost-to-go supplies forward-looking evaluation. During training, rollout evaluation contributes to local feedback. During final decoding, it assists action selection when Q values are not sufficiently distinct. This hybrid design is intended to reduce the weakness of both components: Q-learning alone lacks reliable lookahead, while rollout alone does not accumulate learned preferences across episodes.

The contributions are as follows.

1. A dynamic scheduling framework is implemented for multi-skill personnel, interruptible tasks and multiple weighted milestones.
2. A rollout-Q-learning algorithm is proposed, integrating Q-table learning, rollout cost-to-go evaluation, checkpoint selection and beam-search decoding.
3. A six-dimensional discrete state representation is used to describe project load, skill bottleneck, milestone urgency, resource utilization, critical running task pressure and interruption pressure.
4. Experiments on J10-J60 instances show that RQL improves solution quality relative to heuristic rules, rollout heuristic and plain Q-learning, while also revealing the runtime and convergence limits of the method.

The rest of the paper is organized as follows. Section 2 defines the scheduling problem. Section 3 describes the RQL algorithm. Section 4 presents the experimental design. Section 5 reports comparison, runtime and convergence results. Section 6 discusses implications and limitations.

## 2. Problem setting

Consider a project network with `n` task nodes and precedence relations. The network contains a virtual start node and a virtual terminal node. Thus, a J10 instance has 10 real tasks and 12 total nodes. A task can start only after all predecessors have been completed.

Let the project be executed by a set of workers. Each worker has a skill set and a skill level. For a given task, only workers with the required skill are feasible. Processing time is worker-dependent, so assigning the same task to different feasible workers may produce different durations.

At each decision point, the scheduler selects feasible task-worker actions. A worker can process at most one task at a time, and a task can be processed by at most one worker. A running task can be interrupted when a conflict requires worker release. Once interrupted, the task records its original worker and remaining processing time. It can only be resumed by that worker.

The project contains multiple milestones. Milestone `m` has a due time `D_m`, an actual completion time `C_m`, and a penalty weight `w_m`. The objective is to minimize total weighted milestone delay:

```text
minimize  sum_m w_m * max(0, C_m - D_m).
```

This objective differs from a pure makespan objective because it penalizes intermediate delivery delay. A feasible schedule can therefore be evaluated by the total penalty accumulated across milestone events.

## 3. Rollout-Q-learning algorithm

### 3.1 Overview

RQL consists of three decision layers (Fig. 1). The first layer extracts a compact state from the current project environment. The second layer trains a Q table through repeated scheduling episodes. The third layer constructs the final schedule using the trained Q table, rollout cost-to-go fallback and beam search.

The method is designed around a practical observation: the scheduler often faces several feasible task-worker assignments with similar apparent value. A purely greedy rule may select the wrong action because it ignores downstream milestone risk. A Q table may also be uncertain because the state-action pair has not been sufficiently explored. Rollout cost-to-go helps by simulating future consequences under a heuristic continuation policy.

### 3.2 State and action representation

The state representation uses six discrete features:

1. `NLF`: normalized load feature.
2. `SBI`: skill bottleneck indicator.
3. `MUR`: milestone urgency ratio.
4. `RUR`: resource utilization ratio.
5. `CRT`: critical running task pressure.
6. `ITN`: interrupted task number.

These features are used to keep the Q table finite while retaining information about project load, resource pressure and milestone urgency. The action is represented as:

```text
(action_type, task_id, person_id)
```

The implemented action types are assignment, resume and continue. Interruption is handled as an automatic conflict-resolution action rather than an unconstrained learning action. This design prevents the learning process from treating arbitrary interruption as a free decision.

### 3.3 Training reward

The training signal combines local and global information. The local reward captures immediate execution feedback and rollout lookahead gain. The global reward is the negative total weighted milestone delay penalty. Higher reward therefore corresponds to a lower scheduling penalty.

The current formal configuration uses scale-specific convergence strategies. J10 uses a checkpoint profile. J20, J30 and larger scales use a combined profile that includes lower late-stage exploration, checkpoint selection, milestone-penalty-weight normalization of the global reward and lookahead warm-up. The purpose is to reduce unstable late-stage exploration and prevent the reward scale from changing excessively across problem sizes.

### 3.4 Cost-to-go assisted decoding

After training, the final schedule is decoded from the learned Q table. If the highest Q values are sufficiently distinct, the decoder can rely on the learned preference. If candidate Q values are close, rollout cost-to-go evaluates the downstream penalty of candidate actions. Beam search keeps multiple partial action sets, reducing the risk of committing too early to a locally attractive but globally poor action.

The final decoder is therefore not pure greedy Q-learning. It is a hybrid construction procedure in which learned preference, rollout evaluation and controlled beam expansion interact.

## 4. Experimental design

### 4.1 Instances and protocols

Experiments use the generated case files in the `cases/` directory. The main results focus on J10, J20, J30 and J60. J10-J30 use 100 cases with 3 seeds. J60 uses 20 cases with 3 seeds because runtime is much higher at this scale. Each J-scale file contains `J + 2` nodes, including the virtual start and terminal nodes.

### 4.2 Compared methods

Five methods are reported in the main manuscript:

1. `RQL`: the proposed rollout-Q-learning method.
2. `RH`: a rollout heuristic without Q-table learning.
3. `MXS+MF`: a deterministic rule selecting the task with the maximum number of successors and then assigning the feasible worker with the earliest finish time.
4. `MPV-SLK-EFT`: a milestone-priority, slack and earliest-finish-time heuristic.
5. `QL`: plain Q-learning without rollout cost-to-go assistance.

### 4.3 Metrics

The primary metric is `AOTV`, the average total weighted delay penalty. `Std` reports the standard deviation of instance-level penalties. `ARPD` measures average relative percentage deviation from the best value found on the same instance. `Time` reports total runtime, including training and final solution construction. Lower values indicate better performance for AOTV, Std, ARPD and Time, although Time is interpreted as computational cost rather than solution quality.

## 5. Results

### 5.1 RQL obtains the best average delay penalty

RQL obtains the lowest AOTV on all four main scales (Fig. 2). On J10, RQL achieves an AOTV of 1.301, compared with 1.457 for RH and 1.992 for QL. On J20, RQL obtains 2.277, compared with 2.722 for RH and 5.143 for QL. On J30, RQL obtains 2.473, compared with 2.677 for RH and 13.229 for QL. On J60, RQL obtains 9.260, compared with 10.355 for RH and 55.748 for QL.

Relative to RH, RQL reduces AOTV by 10.7%, 16.3%, 7.6% and 10.6% on J10, J20, J30 and J60, respectively. This consistent improvement indicates that the learned Q preference provides useful information beyond the rollout heuristic alone.

The comparison with QL is more substantial. QL performs acceptably on J10, but its quality deteriorates sharply on J30 and J60. This result supports the design motivation of RQL: Q-learning benefits from rollout cost-to-go because the lookahead signal helps distinguish actions when terminal rewards are sparse and delayed.

Table 1. Main comparison on J10-J60. Lower values are better.

| Scale | Method | AOTV | Std | ARPD (%) | Time (s) |
|---|---|---:|---:|---:|---:|
| J10 | RQL | 1.301 | 1.272 | 8.146 | 6.579 |
| J10 | RH | 1.457 | 1.361 | 23.086 | 0.083 |
| J10 | MXS+MF | 2.284 | 1.973 | 87.599 | 0.001 |
| J10 | MPV-SLK-EFT | 2.202 | 2.072 | 76.001 | 0.001 |
| J10 | QL | 1.992 | 1.733 | 56.634 | 7.534 |
| J20 | RQL | 2.277 | 1.822 | 25.423 | 41.175 |
| J20 | RH | 2.722 | 2.266 | 51.443 | 0.397 |
| J20 | MXS+MF | 3.332 | 2.500 | 90.247 | 0.003 |
| J20 | MPV-SLK-EFT | 3.833 | 3.173 | 120.477 | 0.003 |
| J20 | QL | 5.143 | 4.018 | 202.370 | 17.195 |
| J30 | RQL | 2.473 | 2.265 | 25.316 | 178.812 |
| J30 | RH | 2.677 | 2.416 | 38.520 | 1.058 |
| J30 | MXS+MF | 3.587 | 3.375 | 77.926 | 0.004 |
| J30 | MPV-SLK-EFT | 3.833 | 3.522 | 93.907 | 0.004 |
| J30 | QL | 13.229 | 7.375 | 679.082 | 27.770 |
| J60 | RQL | 9.260 | 4.338 | 22.736 | 724.079 |
| J60 | RH | 10.355 | 4.283 | 40.217 | 13.839 |
| J60 | MXS+MF | 11.785 | 4.429 | 68.332 | 0.016 |
| J60 | MPV-SLK-EFT | 15.025 | 7.068 | 104.891 | 0.015 |
| J60 | QL | 55.748 | 13.598 | 728.444 | 58.682 |

### 5.2 RQL trades runtime for quality

RQL requires much more runtime than deterministic heuristics and RH (Fig. 3). On J60, its average runtime is 724.079 s, compared with 13.839 s for RH and less than 0.02 s for the deterministic heuristics. The source of this overhead is the repeated rollout cost-to-go simulation used during training and final decoding.

This result defines the operating position of the method. RQL is appropriate when solution quality is the priority and offline training time is acceptable. It is not appropriate when near-instant online dispatching is required.

### 5.3 Convergence is scale-dependent

The convergence experiment records training steps and episode reward, where reward equals the negative total weighted delay penalty. A reward closer to zero indicates a better episode. The reward curves show clear improvement on J10-J30 and weaker improvement on J60 (Fig. 4).

On J10 and J20, smoothed rewards stabilize and the Q-table maximum difference approaches the 10^-3 level. On J30, rewards improve but retain visible variance. On J60, rewards improve from -67.044 to -50.664, but the Q-table difference remains above the convergence threshold. The J60 result should therefore be interpreted as partial improvement rather than full convergence.

## 6. Discussion

The results show that RQL improves scheduling quality by combining two complementary sources of decision information. Q-learning accumulates state-action preference over repeated episodes. Rollout cost-to-go evaluates downstream consequence when learned values are ambiguous. The performance gap between RQL and QL suggests that the rollout component is not optional in this problem setting.

The method also has clear limitations. First, cost-to-go simulation makes training expensive, especially on J60 and larger instances. Second, convergence becomes less stable as the project scale increases. Third, the current implementation uses a tabular Q representation, which may not generalize efficiently to much larger state-action spaces.

These limitations suggest several technical directions. Candidate action pruning could reduce unnecessary cost-to-go calls. Adaptive rollout depth could allocate simulation budget to difficult states. Stronger caching and parallel simulation could reduce runtime. Function approximation could replace or supplement the Q table when the state space becomes too large.

## 7. Conclusion

This paper proposed a rollout-Q-learning algorithm for multi-milestone multi-skill interruptible project scheduling. The algorithm combines six-dimensional state abstraction, complete action tuples, Q-learning, rollout cost-to-go, checkpoint selection and beam-search decoding. Experiments on J10-J60 instances show that RQL consistently reduces weighted milestone delay relative to heuristic rules, rollout heuristic and plain Q-learning. The improvement is obtained at a substantial computational cost, and full convergence is not observed on J60 under the current budget. The method is therefore best understood as a solution-quality-oriented scheduling approach with explicit scalability limitations.

## Figure captions

**Figure 1. Rollout-Q-learning framework for interruptible multi-skill project scheduling.** The framework combines six-dimensional state extraction, task-worker action selection, Q-table updating, rollout cost-to-go evaluation and final beam-search decoding.

**Figure 2. Solution quality comparison on J10-J60.** RQL obtains the lowest AOTV and ARPD among the selected methods. The improvement is consistent across the reported scales.

**Figure 3. Runtime and quality-time trade-off.** RQL requires substantially more runtime because rollout cost-to-go simulation is repeatedly invoked during training and decoding.

**Figure 4. Scale-dependent convergence behaviour.** Training reward improves on all reported scales, but Q-table stability is strongest on J10-J20 and weakest on J60.

## Citation placeholders

The following citation groups should be verified before submission:

1. TODO[REF-RCPSP]: resource-constrained project scheduling.
2. TODO[REF-MULTISKILL]: multi-skill project scheduling.
3. TODO[REF-INTERRUPTIBLE]: interruptible scheduling.
4. TODO[REF-ROLLOUT]: rollout algorithms and approximate dynamic programming.
5. TODO[REF-RL-SCHEDULING]: reinforcement learning for scheduling.
6. TODO[REF-QLEARNING]: tabular Q-learning.

