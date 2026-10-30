# geolabelqa

**Spatially aware quality assurance for geospatial machine-learning training labels.**

[![CI](https://github.com/sidewalkds-oss/geolabelqa/actions/workflows/ci.yml/badge.svg)](https://github.com/sidewalkds-oss/geolabelqa/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Land-cover maps, crop classifiers and building detectors all depend on hand-labelled
training samples, and those labels are wrong more often than anyone would like.
The mistakes are rarely random. One annotator misreads cropland as grassland across a
whole tile, or an outdated basemap shifts every label in a district. Errors like these
come in **spatial clusters**, and the usual checks struggle with them.

`geolabelqa` ranks geolocated samples by how likely their label is wrong, and comes with
a paper that tests what actually works, on a synthetic benchmark **and on two real
hyperspectral scenes** (Indian Pines, Pavia University).

* **Smoothed self-confidence:** out-of-fold class probabilities, averaged over each
  sample's *k* nearest spatial neighbours, then scored as 1 − p(given label). It was the
  one change that helped on every dataset.
* **Transferability check:** measures whether spatial block cross-validation will help
  or hurt your audit, before you run it.
* **Confident-learning flags** (Northcutt et al., 2021) and a neighbour-vote score for context.
* A dependency-light **GeoJSON polygon validator** (self-intersections, slivers, winding
  order, unclosed rings, holes outside shells, duplicates, missing labels).

📄 **Paper:** [`paper/paper.tex`](paper/paper.tex), *Spatially Aware Label-Error Detection
for Geospatial Machine-Learning Training Data*. CI builds the PDF on every push.

---

## Installation

```bash
git clone https://github.com/sidewalkds-oss/geolabelqa.git
cd geolabelqa
pip install -e ".[dev]"
```

Runtime dependencies: `numpy`, `scipy`, `scikit-learn`, `pandas`. No GDAL/GEOS needed.

## Quick start

### Python

```python
from geolabelqa import LabelAuditor

# X: (n, n_features) e.g. Sentinel-2 bands/indices; coords: (n, 2) in a PROJECTED CRS
auditor = LabelAuditor(k=10)

# Optional: should the audit use spatial block folds?
print(auditor.transferability_gap(X, labels, coords))   # {'random': .., 'spatial': .., 'gap': ..}
# Small gap (a few points)  -> LabelAuditor(cv="spatial", block_size=...)
# Large gap                 -> keep the default random folds

report = auditor.audit(X, labels, coords)
report.head(20)          # most suspicious samples first
report[report.cl_flag]   # confident-learning flags
```

| column | meaning |
|---|---|
| `index` | row in your input |
| `label` / `suggested_label` | given label and the model's best alternative |
| `suspicion` | 0–1; 1 − smoothed out-of-fold probability of the given label |
| `cl_flag` | flagged by the confident-learning rule |
| `neighbour_disagreement` | share of *k* nearest neighbours with a different label |

### Command line

```bash
geolabelqa demo -o demo.csv                     # synthetic dataset with 15% clustered errors
geolabelqa audit demo.csv --x x --y y --label label \
    --features blue,green,red,nir,swir1,swir2,ndvi,ndwi,ndbi -o report.csv
geolabelqa validate annotations.geojson --label-property class
```

`validate` exits non-zero if any error-level issue is found, so it can gate a CI pipeline
for an annotation repository.

## Results

Average precision for ranking corrupted labels at a 20% error rate (chance = 0.20),
5 seeds, random-forest classifier. Real scenes use their expert ground truth with
injected noise.

| Detector | Synthetic, clustered | Indian Pines, clustered | Pavia, clustered | Indian Pines, uniform | Pavia, uniform |
|---|---|---|---|---|---|
| Neighbour vote | 0.234 | 0.279 | 0.222 | **0.998** | 0.944 |
| Self-confidence, random CV (standard confident learning) | 0.513 | 0.268 | 0.370 | 0.940 | 0.963 |
| Self-confidence, spatial CV | 0.543 | 0.225 | 0.338 | 0.636 | 0.595 |
| **Smoothed self-confidence, random CV** (default) | 0.614 | 0.283 | **0.412** | 0.981 | **0.988** |
| Smoothed self-confidence, spatial CV | **0.653** | 0.239 | 0.352 | 0.748 | 0.613 |

*Features: spectral only. The paper reports both feature sets, class-pair noise and more.*

What the evidence says:

- **Smoothing helps:** it improved AP in 180/180 synthetic and 107/120 real comparisons
  (Wilcoxon p < 0.05 in every noise regime).
- **Neighbour voting is blind to clustered errors,** near chance on all three datasets, even
  though it is near-perfect for isolated errors on the real scenes.
- **Spatial CV is not a free win.** It beat random CV on synthetic clustered noise (15/15
  runs) but lost on the real scenes. There, held-out blocks are hard to predict even with
  clean labels: accuracy drops by **24 points**, against 1 point on the synthetic landscape.
  `transferability_gap()` measures this for your data.
- **Clustered errors remain hard,** with a best real-scene AP of 0.31–0.41. Removing flagged
  samples *hurt* downstream accuracy under clustered noise, so use the ranking to direct
  human review, not automatic deletion.

## Reproducing the paper

```bash
python experiments/run_benchmark.py     # synthetic benchmark (~30 min)
python experiments/run_real.py          # real scenes (~30 min), needs data/ (below)
python experiments/transferability.py   # clean-label random vs spatial CV accuracy
python experiments/make_figures.py      # all figures and LaTeX tables
```

The real scenes are the Indian Pines (corrected) and Pavia University MATLAB files from the
[UPV/EHU Hyperspectral Remote Sensing Scenes](https://ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
page. Place `Indian_pines_corrected.mat`, `Indian_pines_gt.mat`, `PaviaU.mat` and
`PaviaU_gt.mat` in `data/`. The paper's runs used verified copies whose image sizes and
per-class pixel counts match the published values.

## Repository layout

```
src/geolabelqa/
  auditor.py      LabelAuditor (audit + transferability_gap), the high-level API
  detectors.py    out-of-fold probabilities, self-confidence, CL flags, smoothing, neighbour vote
  spatial_cv.py   SpatialBlockKFold (with optional buffer)
  scenes.py       loaders for the real hyperspectral scenes
  geometry.py     GeoJSON polygon validator
  synthetic.py    synthetic land-cover generator and noise models
  metrics.py      AUROC / AP / precision@N, flag precision/recall/F1
  cli.py          `geolabelqa` command
experiments/      benchmarks, transferability check, figure/table generation
results/          raw CSVs from every experiment
paper/            LaTeX source, figures, tables
tests/            pytest suite
```

## Limitations

The noise is injected, not observed. Even on the real scenes, expert ground truth is
treated as clean. The transferability diagnostic is backed by three datasets and is not
yet a calibrated decision rule. See the paper's limitations section.

## Citation

```bibtex
@misc{omar2026geolabelqa,
  title  = {Spatially Aware Label-Error Detection for Geospatial Machine-Learning Training Data},
  author = {Omar, Kaisi},
  year   = {2026},
  url    = {https://github.com/sidewalkds-oss/geolabelqa}
}
```

## License

MIT
