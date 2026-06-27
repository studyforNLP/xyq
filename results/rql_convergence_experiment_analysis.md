# RQL 收敛性实验分析

- 规模: J10,J20,J30
- 每规模案例数: 10
- seed 数: 3
- 方案: baseline,scale_ep,low_epsilon,checkpoint,norm_warmup

## 总体汇总

| variant | mean_penalty | convergence_rate | mean_rollouts | mean_max_q_diff | mean_time |
|---|---:|---:|---:|---:|---:|
| baseline | 4.1978 | 0.2778 | 456.1 | 0.026139 | 45.224 |
| checkpoint | 2.9322 | 0.4333 | 865.8 | 0.011012 | 94.352 |
| low_epsilon | 4.6044 | 0.4333 | 865.8 | 0.011012 | 94.932 |
| norm_warmup | 8.2678 | 0.2889 | 857.8 | 0.003134 | 41.757 |
| scale_ep | 4.6944 | 0.8778 | 714.7 | 0.001682 | 69.997 |

## 关键发现

- overall / Best mean penalty variant: checkpoint mean_penalty=2.9322, convergence_rate=0.4333
- overall / Best convergence variant: scale_ep convergence_rate=0.8778, mean_penalty=4.6944
- baseline_comparison / checkpoint: penalty_change_vs_baseline=-30.15%, convergence_rate_delta=0.1556, mean_max_q_diff=0.011012
- baseline_comparison / low_epsilon: penalty_change_vs_baseline=9.69%, convergence_rate_delta=0.1556, mean_max_q_diff=0.011012
- baseline_comparison / norm_warmup: penalty_change_vs_baseline=96.96%, convergence_rate_delta=0.0111, mean_max_q_diff=0.003134
- baseline_comparison / scale_ep: penalty_change_vs_baseline=11.83%, convergence_rate_delta=0.6000, mean_max_q_diff=0.001682
- scale_best / J10: best_quality=checkpoint mean_penalty=1.7067; best_convergence=checkpoint convergence_rate=0.7667
- scale_best / J20: best_quality=checkpoint mean_penalty=2.6967; best_convergence=scale_ep convergence_rate=0.9333
- scale_best / J30: best_quality=checkpoint mean_penalty=4.3933; best_convergence=scale_ep convergence_rate=1.0000

## 分规模汇总

| scale | variant | mean_penalty | convergence_rate | mean_max_q_diff | mean_time |
|---|---|---:|---:|---:|---:|
| J10 | baseline | 1.7100 | 0.7000 | 0.002958 | 6.593 |
| J10 | checkpoint | 1.7067 | 0.7667 | 0.001256 | 6.100 |
| J10 | low_epsilon | 1.7333 | 0.7667 | 0.001256 | 6.003 |
| J10 | norm_warmup | 3.2167 | 0.3333 | 0.001274 | 6.607 |
| J10 | scale_ep | 1.7100 | 0.7000 | 0.002958 | 6.652 |
| J20 | baseline | 3.4833 | 0.1333 | 0.011718 | 28.891 |
| J20 | checkpoint | 2.6967 | 0.4000 | 0.002629 | 41.202 |
| J20 | low_epsilon | 3.0633 | 0.4000 | 0.002629 | 41.228 |
| J20 | norm_warmup | 7.2967 | 0.2667 | 0.003959 | 29.200 |
| J20 | scale_ep | 3.5900 | 0.9333 | 0.001127 | 36.013 |
| J30 | baseline | 7.4000 | 0.0000 | 0.063741 | 100.189 |
| J30 | checkpoint | 4.3933 | 0.1333 | 0.029152 | 235.754 |
| J30 | low_epsilon | 9.0167 | 0.1333 | 0.029152 | 237.566 |
| J30 | norm_warmup | 14.2900 | 0.2667 | 0.004169 | 89.463 |
| J30 | scale_ep | 8.7833 | 1.0000 | 0.000962 | 167.327 |
