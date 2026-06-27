# A Rollout-Q-learning Algorithm for Multi-skill Interruptible Project Scheduling with Multiple Milestones

## Abstract

Dynamic project scheduling in service-oriented projects is difficult when task execution depends on heterogeneous personnel skills, tasks may be interrupted and resumed, and multiple milestones carry different delay penalties. Rule-based dispatching methods are fast but usually lack adaptive decision preferences, whereas plain Q-learning can struggle to learn stable values from sparse terminal penalties. Here we introduce a rollout-Q-learning (RQL) algorithm for multi-skill interruptible project scheduling with multiple milestones. RQL represents each decision state using six discrete features, learns action preferences over complete action tuples, and integrates rollout cost-to-go evaluation into both training feedback and final schedule decoding. Across J10-J60 benchmark instances, RQL consistently obtains the lowest average total weighted delay among the tested methods. Relative to the rollout heuristic baseline, RQL reduces AOTV by 10.7%, 16.3%, 7.6% and 10.6% on J10, J20, J30 and J60, respectively. Compared with plain Q-learning, RQL is markedly more robust, especially on J30 and J60. The improvement comes at a clear computational cost, with runtime increasing rapidly at larger scales because cost-to-go simulation dominates training. Convergence analysis shows stable training behaviour on J10-J20, partial but useful improvement on J30, and continued instability on J60. These results position RQL as a solution-quality-oriented dynamic scheduling algorithm rather than a low-cost dispatching heuristic.

## Introduction

Project delivery in knowledge-intensive service systems often depends on the coordinated execution of precedence-constrained tasks by personnel with heterogeneous skill sets. Examples include software development, technical consulting, engineering service delivery and maintenance projects, where the same task may require different processing times when assigned to different workers. In such settings, the scheduling objective is not only to finish the whole project early, but also to control delay at important intermediate deliverables. Multiple milestones therefore create a temporally distributed penalty structure: a schedule can be unacceptable even when the final completion time is reasonable, if critical intermediate events are delayed.

This paper focuses on a dynamic scheduling setting with three coupled characteristics. First, each task can be processed only by personnel who possess the required skill, and processing time is person-dependent. Second, tasks are interruptible; once interrupted, a task must resume later with the original worker from the remaining processing time. Third, the project contains multiple milestones with planned completion times and penalty weights. The objective is to minimize the total weighted milestone delay. This setting is more expressive than a single-makespan objective, but it also makes dispatching difficult because local decisions can influence milestone delays through precedence relations and future resource conflicts.

Existing heuristic dispatching rules are attractive because they are simple and fast. Rules such as shortest processing time, earliest completion time, slack-based priority or maximum-successor priority can be executed online with minimal overhead. However, their decision criteria are usually fixed and local. They do not learn from repeated scheduling episodes, and they may fail when the best local rule changes with the state of the project. Plain tabular Q-learning provides a natural way to learn action preferences, but in this problem it faces two technical obstacles. The first is sparse and delayed feedback: the main objective is only fully observed after milestone completion. The second is action ambiguity: many feasible task-person actions can have similar Q values during training, especially under large state-action spaces.

We address this gap with a rollout-Q-learning algorithm. The key idea is to combine learned action preferences with short-horizon rollout evaluation. Q-learning provides a reusable preference table over discrete state-action pairs, while rollout cost-to-go supplies a forward-looking estimate when immediate Q values are insufficiently discriminative. The final decoding stage further combines Q-value ranking, cost-to-go fallback and beam search. This design turns rollout from a standalone heuristic into an evaluator that supports both training and construction.

The main contributions are as follows.

1. We define a dynamic scheduling framework for multi-skill personnel, interruptible tasks and multiple weighted milestones.
2. We propose RQL, a hybrid algorithm that combines tabular Q-learning with rollout cost-to-go evaluation and checkpoint-based decoding.
3. We design a six-dimensional discrete state representation, consisting of NLF, SBI, MUR, RUR, CRT and ITN, and use complete action tuples of the form `(action_type, task_id, person_id)`.
4. We provide comparative experiments on J10-J60 instances against heuristic rules, rollout heuristic and plain Q-learning, showing consistent improvements in solution quality and exposing the quality-time trade-off.

## Problem description

Consider a project network with `n` task nodes. The network includes precedence constraints, a virtual start node and a virtual terminal node. A J10 instance therefore contains 10 real tasks and 12 total nodes. Let each service worker have a skill set and a skill level. A task can be assigned only to a feasible worker, and the processing time of a task depends on the assigned worker.

At each decision time, the scheduler chooses feasible task-worker actions. Each worker can process at most one task at a time, and each task can be processed by at most one worker. A running task may be interrupted when a resource conflict needs to be resolved. The interrupted task records its assigned worker and remaining processing time; it can only resume with the same worker.

The project contains a set of milestone events. Each milestone has a due time and a penalty weight. If the actual finish time of a milestone event is later than its due time, weighted delay is incurred. The objective is

```text
minimize penalty = sum_m w_m * max(0, C_m - D_m),
```

where `C_m` is the actual completion time of milestone `m`, `D_m` is its planned completion time and `w_m` is its penalty weight.

## Rollout-Q-learning algorithm

### Overview

RQL has three modules (Fig. 1). The first module converts the project status into a discrete state. The second module trains a Q table using local action feedback, global milestone reward and rollout cost-to-go improvement. The third module constructs the final schedule using the learned Q table, cost-to-go fallback and beam search.

### State and action representation

The state representation contains six features:

1. `NLF`, the normalized load feature.
2. `SBI`, the skill bottleneck indicator.
3. `MUR`, the milestone urgency ratio.
4. `RUR`, the resource utilization ratio.
5. `CRT`, the critical running task pressure.
6. `ITN`, the interrupted task number.

These features are discretized to keep tabular Q-learning tractable. The action is represented as a complete tuple:

```text
(action_type, task_id, person_id)
```

The implemented action types include assignment, resume and continue. Interruptions are inserted automatically when a worker conflict requires interruption, which avoids treating interruption as an unconstrained independent learning action.

### Reward design

RQL combines local and global feedback. The local reward reflects immediate execution feedback and lookahead improvement from cost-to-go. The global reward is the negative total weighted milestone delay, so higher reward corresponds to lower penalty. In the combined convergence profile, the global reward is normalized by the total milestone penalty weight, reducing scale-induced reward distortion across different problem sizes. A warm-up mechanism gradually introduces lookahead reward, and checkpointing stores the best decoded Q table observed during training.

### Rollout cost-to-go and final decoding

The cost-to-go evaluator simulates candidate action consequences and estimates the remaining milestone delay penalty. During training, it contributes to local reward through lookahead gain. During final decoding, it is used when the Q table cannot clearly distinguish among candidate actions. A beam-search decoder keeps multiple partial action sets and accepts beam improvements only when the cost-to-go gain is sufficient. This design is intended to reduce greedy misselection when Q values remain close.

## Computational experiments

### Experimental setup

Experiments use existing instances in the `cases/` folder. The main manuscript focuses on J10, J20, J30 and J60. J10-J30 use 100 cases with 3 seeds; J60 uses 20 cases with 3 seeds. Larger J90 and J120 results are available but are not emphasized because runtime becomes substantial.

The compared methods are:

1. `RQL`, the proposed rollout-Q-learning method.
2. `RH`, a pure rollout heuristic without Q-learning.
3. `MXS+MF`, a heuristic that first selects the task with the maximum number of successors and then assigns the feasible worker with the earliest finish time.
4. `MPV-SLK-EFT`, a milestone-priority, slack and earliest-finish-time heuristic.
5. `QL`, plain Q-learning without rollout cost-to-go assistance.

The main metrics are:

1. `AOTV`, the average total weighted delay penalty across instances.
2. `Std`, the standard deviation of instance-level average penalty.
3. `ARPD`, the average relative percentage deviation from the best value obtained on the same instance.
4. `Time`, the total runtime including training and decoding.

Lower values are better for all four metrics except where runtime is interpreted as computational cost rather than solution quality.

### Main comparison

RQL achieves the lowest AOTV on all four main scales (Fig. 2 and Table 1). On J10, RQL obtains an AOTV of 1.301, compared with 1.457 for RH and 1.992 for QL. On J20, RQL reduces AOTV to 2.277, compared with 2.722 for RH and 5.143 for QL. On J30, RQL obtains 2.473, lower than RH at 2.677 and much lower than QL at 13.229. On J60, RQL obtains 9.260, compared with 10.355 for RH and 55.748 for QL.

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

Relative to RH, RQL reduces AOTV by 10.7% on J10, 16.3% on J20, 7.6% on J30 and 10.6% on J60. The comparison with QL is more revealing: QL performs reasonably on J10 but deteriorates sharply on J30 and J60. This indicates that learned Q values alone are insufficient in larger state-action spaces, while rollout cost-to-go provides necessary forward evaluation.

### Quality-time trade-off

RQL is not a fast heuristic. Its runtime is substantially higher than deterministic dispatching rules and RH (Fig. 3). For example, on J60 RQL takes 724.079 s on average, whereas RH takes 13.839 s and the two deterministic heuristics take less than 0.02 s. This gap is expected because RQL repeatedly evaluates candidate decisions through rollout cost-to-go simulation during training and decoding.

The runtime result should therefore be interpreted as a quality-time trade-off. RQL is suitable when lower milestone delay is more important than low computational overhead. In contrast, deterministic heuristics remain useful when near-instantaneous schedules are required.

### Convergence behaviour

The convergence experiment records the relationship between training steps and episode reward, where episode reward equals the negative total weighted milestone delay penalty. Thus, a reward closer to zero indicates a lower penalty. The reward curves show clear improvement on J10-J30 and weaker but still positive improvement on J60 (Fig. 4).

On J10 and J20, the smoothed reward approaches stable values and the average log10 maximum Q-table difference approaches the 10^-3 level. This supports a convergence claim for small and medium instances. On J30, rewards improve substantially but retain visible variance across cases and seeds. On J60, rewards improve from -67.044 to -50.664, but the Q-table change remains far above the convergence threshold. The decoded penalty curve also shows that the last Q table can be worse than a previous checkpoint. Therefore, J60 should be described as improved but not fully converged under the current training budget.

## Discussion

The experiments support the central claim that integrating rollout cost-to-go with Q-learning improves solution quality for multi-milestone interruptible project scheduling. The advantage is consistent across J10-J60, and the comparison with QL shows that rollout assistance is not merely an implementation detail; it is necessary for stabilizing decisions when terminal rewards are sparse and action values are close.

The method also exposes a practical limitation. RQL improves weighted delay but pays for it through expensive cost-to-go simulation. The cost becomes substantial at J60 and above. This limitation is not only a runtime issue; it also affects convergence because larger instances enlarge the state-action space and make Q-table updates noisier. Checkpointing, scale-specific training budgets and reward normalization are therefore important components of the current implementation.

Future work should focus on reducing rollout cost without losing lookahead quality. Promising directions include candidate-action pruning, adaptive rollout depth, stronger cost-to-go caching, parallel rollout simulation and function approximation for Q values. A second direction is to develop scale-specific convergence criteria, because the current uniform Q-difference threshold is strict for large instances.

## Methods

### Instance validation

Each benchmark file is parsed before execution. For scale `J`, the total node count must be `J + 2`, including the virtual start and terminal nodes. Invalid files are recorded in the failure sheet and excluded from the experiment.

### Training profile

The formal RQL setting uses a scale-specific convergence profile. J10 uses the checkpoint profile, while J20, J30 and larger scales use the combined profile. The combined profile includes lower late-stage exploration, checkpoint selection, milestone-penalty-weight normalization of global reward and lookahead warm-up. The default training parameters include `alpha = 0.2`, `gamma = 0.99`, initial `epsilon = 0.2`, `epsilon_decay = 0.99`, `epsilon_min = 0.02`, `global_reward_scale = 0.1`, `local_reward_scale = 0.4` and `lookahead_reward_weight = 1.5`.

### Final schedule construction

After training, the final solution is decoded using the learned Q table. When Q values are not sufficiently distinguishable, cost-to-go fallback evaluates candidate action sets. Beam search retains multiple partial action sets to avoid premature greedy decisions. The final objective value is the total weighted milestone delay penalty.

### Figure generation

All manuscript figures were regenerated from the experiment workbooks using Python. Because the current Python environment does not include `matplotlib`, the plotting script writes editable SVG directly with the Python standard library. Source data are exported as CSV files alongside the figures.

## Figure captions

**Figure 1 | Rollout-Q-learning framework for interruptible multi-skill project scheduling.** The method maps the dynamic project state to a six-dimensional discrete representation, selects complete task-worker action tuples, updates a Q table using local and global rewards, and uses rollout cost-to-go evaluation to assist training and final decoding.

**Figure 2 | RQL reduces weighted milestone delay across J10-J60.** AOTV and ARPD comparisons show that RQL achieves the lowest penalty among the tested methods on all main scales. The improvement over RH ranges from 7.6% to 16.3% in AOTV.

**Figure 3 | Solution quality comes with a clear training-time cost.** Runtime comparisons show that RQL occupies the low-penalty but high-cost region because rollout cost-to-go simulation dominates training.

**Figure 4 | RQL training exhibits scale-dependent convergence behaviour.** Training reward improves on all scales, but Q-table stability is strongest on J10-J20, partial on J30 and insufficient on J60 under the current training budget.

## Data and code availability

The experiment code is available in the current repository. The main result workbooks are stored under `results/`, and the regenerated figure source data are stored under `paper/data/`.

## References to complete

This draft intentionally avoids fabricated references. Before submission, add citations for: resource-constrained project scheduling, multi-skill project scheduling, interruptible scheduling, rollout algorithms, reinforcement learning for scheduling and tabular Q-learning.

