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

## Real hyperspectral scenes, average precision (20% injected noise)

| Scene | Detector | Uniform | Class-pair | Clustered |
|---|---|---|---|---|
| Indian Pines | Neighbour vote | **0.998** | **0.970** | 0.279 |
| Indian Pines | SC, random CV | 0.940 | 0.484 | 0.268 |
| Indian Pines | SC, spatial CV | 0.636 | 0.247 | 0.225 |
| Indian Pines | Smoothed SC, random CV | 0.981 | 0.572 | 0.283 |
| Indian Pines | Smoothed SC, spatial CV | 0.748 | 0.255 | 0.239 |
| Indian Pines | SC, random CV, +x,y | 0.971 | 0.623 | 0.288 |
| Indian Pines | SC, spatial CV, +x,y | 0.713 | 0.266 | 0.219 |
| Indian Pines | Smoothed SC, random CV, +x,y | 0.993 | 0.762 | **0.313** |
| Indian Pines | Smoothed SC, spatial CV, +x,y | 0.793 | 0.277 | 0.227 |
| Pavia University | Neighbour vote | 0.944 | 0.780 | 0.222 |
| Pavia University | SC, random CV | 0.963 | 0.685 | 0.370 |
| Pavia University | SC, spatial CV | 0.595 | 0.360 | 0.338 |
| Pavia University | Smoothed SC, random CV | 0.988 | 0.805 | **0.412** |
| Pavia University | Smoothed SC, spatial CV | 0.613 | 0.380 | 0.352 |
| Pavia University | SC, random CV, +x,y | 0.985 | 0.808 | 0.382 |
| Pavia University | SC, spatial CV, +x,y | 0.605 | 0.386 | 0.374 |
| Pavia University | Smoothed SC, random CV, +x,y | **0.997** | **0.898** | 0.400 |
| Pavia University | Smoothed SC, spatial CV, +x,y | 0.606 | 0.406 | 0.384 |
