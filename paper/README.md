# Paper draft package

This folder contains a manuscript draft and regenerated figures for the Rollout-Q-learning scheduling study.

## Files

- `manuscript.md`: Nature-style manuscript draft.
- `figure_contract.md`: figure logic, evidence role and reviewer-risk notes.
- `scripts/make_paper_figures.py`: Python script that regenerates all figures and source CSV files.
- `figures/fig1_rql_framework.svg`: method framework.
- `figures/fig2_solution_quality.svg`: AOTV and ARPD comparison.
- `figures/fig3_runtime_tradeoff.svg`: runtime and quality-time trade-off.
- `figures/fig4_convergence.svg`: reward, penalty and Q-table convergence curves.
- `data/main_results.csv`: source data for the main comparison figures.
- `data/convergence_reward_curve.csv`: source data for the reward convergence curve.
- `data/convergence_curve_summary.csv`: source data for auxiliary convergence curves.

## Regenerate figures

```powershell
E:\conda\python.exe paper\scripts\make_paper_figures.py
```

The script uses only the Python standard library and reads the current experiment workbooks from `results/`.

