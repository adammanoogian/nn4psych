# Nassar 2021 RBO fits — aggregate summary

- Total fits: **264**
- Status PASS: **227** / FAILED: **37**
- Subjects (unique): 133
- Cohort split: patients=101, controls=32

## Behavioral fit quality (median)

| cohort | condition | r_obs_vs_pred | RMSE | n_divergences | rhat_max |
|---|---|---:|---:|---:|---:|
| control | changepoint | -0.044 | 49.65 | 0 | 1.002 |
| control | oddball | -0.311 | 31.87 | 0 | 1.002 |
| patient | changepoint | -0.023 | 49.11 | 0 | 1.002 |
| patient | oddball | -0.231 | 34.36 | 0 | 1.002 |

## Posterior means (median across subjects)

| cohort | condition | H | LW | log_UU | sigma_motor | sigma_LR |
|---|---|---:|---:|---:|---:|---:|
| control | changepoint | 0.010 | 0.106 | 0.954 | 19.155 | 4.510 |
| control | oddball | 0.071 | 0.174 | 0.087 | 21.910 | 2.775 |
| patient | changepoint | 0.010 | 0.093 | 0.940 | 18.828 | 4.381 |
| patient | oddball | 0.069 | 0.223 | 0.613 | 19.323 | 2.427 |
