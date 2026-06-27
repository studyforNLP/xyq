# Figure contract

## Overall figure strategy

Core conclusion: RQL improves solution quality for multi-milestone multi-skill interruptible project scheduling, but its advantage is obtained through a computationally expensive rollout-assisted training process.

Figure archetype: schematic-led composite plus quantitative grids.

Target journal/output: SCI/Nature-style manuscript draft, editable SVG figures.

Backend: Python. The current environment lacks matplotlib, so figures are generated as editable SVG using Python standard-library code.

## Figure map

### Figure 1

Core conclusion: RQL combines state abstraction, Q-learning and rollout cost-to-go evaluation into a single dynamic scheduling framework.

Evidence role: method overview.

Reviewer risk: the schematic should not overclaim optimality; it only explains algorithm flow.

### Figure 2

Core conclusion: RQL obtains the lowest AOTV and ARPD among the selected methods on J10-J60.

Evidence role: primary performance comparison.

Reviewer risk: J60 uses fewer instances than J10-J30, so the manuscript must state the protocol explicitly.

### Figure 3

Core conclusion: RQL's solution-quality gain requires substantially more runtime.

Evidence role: quality-time trade-off.

Reviewer risk: runtime should include training and decoding, not only online construction time.

### Figure 4

Core conclusion: RQL shows clear convergence trends on J10-J30, while J60 improves but does not fully converge under the current budget.

Evidence role: convergence and limitation evidence.

Reviewer risk: do not claim global or full convergence for all scales.

