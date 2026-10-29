## Label-error detection, average precision (20% noise, 5 seeds)

| Detector | Features | CV | Uniform | Class-pair | Clustered |
|---|---|---|---|---|---|
| Neighbour vote | – | – | 0.813 | 0.613 | 0.234 |
| Self-confidence | spectral | random | 0.908 | 0.535 | 0.513 |
| Self-confidence | spectral | spatial | 0.902 | 0.519 | 0.543 |
| Smoothed SC (ours) | spectral | random | 0.944 | 0.630 | 0.614 |
| Smoothed SC (ours) | spectral | spatial | 0.939 | 0.610 | **0.653** |
| Self-confidence | spectral + x,y | random | 0.931 | 0.579 | 0.427 |
| Self-confidence | spectral + x,y | spatial | 0.910 | 0.523 | 0.519 |
| Smoothed SC (ours) | spectral + x,y | random | **0.957** | **0.672** | 0.478 |
| Smoothed SC (ours) | spectral + x,y | spatial | 0.940 | 0.607 | 0.588 |

## Downstream macro-F1 on clean test labels (20% noise, features: spectral + x,y)

| Training labels | Uniform | Class-pair | Clustered |
|---|---|---|---|
| No cleaning | 0.805 | 0.789 | 0.759 |
| Remove top-n: neighbour vote | 0.815 | 0.807 | 0.740 |
| Remove top-n: SC, random CV | 0.807 | 0.781 | 0.758 |
| Remove top-n: SC, spatial CV | 0.798 | 0.778 | 0.759 |
| Remove top-n: smoothed SC, spatial CV (ours) | 0.804 | 0.779 | 0.763 |
| Oracle: remove all corrupted | 0.810 | 0.810 | 0.810 |
| Oracle: all labels correct | 0.812 | 0.812 | 0.812 |
