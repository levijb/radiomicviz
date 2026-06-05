# Phase 3 Spec — Habitat Clustering

Living implementation spec for RadiomicViz Phase 3. Update this file as decisions change.

---

## What Gets Built

### New files

| File | Purpose |
|---|---|
| `src/radiomicviz/habitat.py` | `HabitatResult` dataclass — output contract for clustering |
| `src/radiomicviz/cluster.py` | `cluster_habitats()` and `batch_cluster()` |
| `tests/test_clustering.py` | Full test suite using synthetic fixtures |

### Modified files

| File | Change |
|---|---|
| `src/radiomicviz/__init__.py` | Export `cluster_habitats`, `batch_cluster`, `HabitatResult` |
| `src/radiomicviz/cli.py` | Add `cluster` and `batch-cluster` Click commands |
| `src/radiomicviz/viewer/app.py` | Add habitat label map overlay support to manifest |
| `src/radiomicviz/viewer/templates/viewer.html` | Add HABITATS sidebar panel |
| `CLAUDE.md` | Phase 3 patterns and gotchas |
| `README.md` | Clustering section + citation |

---

## Implementation Steps

### Step A — HabitatResult (habitat.py) ✅

Define the output dataclass and its export methods. No clustering logic yet.

**Verify:** `pytest tests/test_clustering.py::TestHabitatResult`

### Step B — cluster_habitats() (cluster.py)

Implement the full pipeline (parse → matrix → clean → normalize → PCA → cluster → back-project → stats).
Accept `ExtractionResult` or `(4D NIfTI path, mask path)`.

**Verify:** `pytest tests/test_clustering.py::TestClusterHabitats`

### Step C — CLI commands (cli.py)

Add `cluster` and `batch-cluster` Click commands wrapping Step B functions.

**Verify:** `radiomicviz cluster --help`, `radiomicviz batch-cluster --help`

### Step D — batch_cluster() (cluster.py)

Parallel batch wrapper using joblib. Reads a cohort CSV (same format as batch_extract).

**Verify:** `pytest tests/test_clustering.py::TestBatchCluster`

### Step E — Viewer integration

Wire `HabitatResult.view()` (currently a NotImplementedError stub). Add HABITATS panel to
`viewer.html`. Update Flask manifest with `habitat_map` and `habitat_probs` keys.

**Verify:** Manual — launch viewer, confirm label map renders with discrete colormap.

---

## HabitatResult Dataclass

```python
@dataclass
class HabitatResult:
    label_map: np.ndarray          # 3D int array, same shape as image, 0=background
    probabilities: Optional[np.ndarray]  # 4D (x,y,z,k), GMM soft assignments only
    n_clusters: int
    cluster_stats: pd.DataFrame    # per-habitat mean/std/median/p10/p90 per feature
    selection_criteria: dict       # {k: {"bic": ..., "silhouette": ..., "ch": ..., "db": ...}}
    selected_k: int                # which k was auto-selected (or user-forced)
    features_used: list[str]       # after redundancy elimination
    features_dropped: list[str]    # which features were eliminated and why
    pca_variance_explained: Optional[float]  # if PCA was used
    metadata: dict                 # method, params, timestamp, radiomicviz_version
    mask_nii: Optional[nib.Nifti1Image]  # retained for NIfTI export
```

### Methods

| Method | Description |
|---|---|
| `.to_nifti(path)` | Save `label_map` as int16 3D NIfTI, affine from `mask_nii` |
| `.to_prob_nifti(path)` | Save `probabilities` as float32 4D NIfTI; raises if GMM not used |
| `.to_csv(path)` | Save `cluster_stats` CSV + `.metadata.json` sidecar |
| `.summary()` | Formatted string: n_clusters, voxels/habitat, features used/dropped |
| `.view(...)` | Launch Flask viewer with label map as discrete overlay |

---

## cluster_habitats() Signature

```python
def cluster_habitats(
    source: Union[ExtractionResult, str, Path],
    mask: Optional[Union[str, Path]] = None,
    method: str = "gmm",
    n_clusters: Union[int, str] = "auto",
    k_range: tuple[int, int] = (2, 6),
    redundancy_threshold: float = 0.70,
    pca_components: Optional[Union[int, str]] = None,
    gmm_covariance_type: str = "full",
    random_state: int = 42,
    min_cluster_size: int = 10,
) -> HabitatResult
```

---

## Pipeline (Step B Implementation Order)

1. **Parse input** — accept `ExtractionResult` or `(4D NIfTI path + mask path)`
2. **Extract voxel-by-feature matrix** — within-mask voxels only; shape `(n_voxels, n_features)`
3. **Drop bad features** — zero variance; >10% NaN. Log each.
4. **Impute NaNs** — per-feature median.
5. **Spearman redundancy elimination** — for each correlated pair (`|r| > threshold`), drop
   the lower-variance one. Log which features were dropped.
6. **Z-score normalize** — each column.
7. **Optional PCA** — if `pca_components` is not None. `"auto"` = retain 95% variance. Log
   explained variance ratio.
8. **K selection (if `n_clusters == "auto"`):**
   - Fit GMM or K-means for each k in `k_range`
   - Compute BIC (GMM only), Silhouette, Calinski-Harabasz, Davies-Bouldin
   - Select k: BIC gradient method for GMM; Silhouette maximum for K-means
   - Store all criteria in `selection_criteria`
9. **Final fit** — GMM: hard (`predict`) + soft (`predict_proba`); K-means: hard only.
10. **Auto-fallback covariance** — if `n_voxels < n_features_after_pca * 10`, switch to
    `"diag"` and warn.
11. **Back-project** — per-voxel labels → full image volume via mask index. 0 = background.
12. **Cluster stats** — mean/std/median/p10/p90 per original feature per habitat.
13. **Warn if small** — any habitat < `min_cluster_size` voxels.
14. **Return HabitatResult.**

---

## CLI Commands

```bash
# Single subject
radiomicviz cluster \
  --feature-4d features_4d.nii.gz \
  --mask ROI.nii.gz \
  --method gmm \
  --n-clusters auto \
  --k-range 2 6 \
  --redundancy-threshold 0.70 \
  --pca-components auto \
  --gmm-covariance full \
  --random-state 42 \
  --output-dir ./habitats/

# Batch
radiomicviz batch-cluster \
  --subjects cohort.csv \
  --feature-4d-col feature_4d_path \
  --mask-col Mask \
  --output-dir ./habitats/ \
  --n-jobs 4 \
  [same clustering params as above]
```

---

## Viewer Integration (Step E)

`HabitatResult.view()` calls the same `_serve` / `launch_viewer` infrastructure as
`ExtractionResult.view()`. The habitat label map NIfTI is passed as an additional overlay.

NiiVue renders integer NIfTIs with a discrete colormap natively ("actc" palette).

### New manifest keys

```json
{
  "habitat_map": "habitats.nii.gz",
  "habitat_probs": "habitats_probs.nii.gz"
}
```

`habitat_probs` is `null` when the clustering method is K-means.

### HABITATS sidebar panel (viewer.html)

Position: below ACTIVE OVERLAYS.

Contents:
- Habitat label map loaded with `"actc"` colormap
- Opacity slider
- HTML table of `cluster_stats` (mean of top 3 features per habitat)
- "Show probabilities" toggle (GMM only; loads probability 4D NIfTI)

---

## What Is Deferred to Phase 4

See PHASE4_SPEC.md.

---

## Citation

Habitat clustering methodology informed by:
- Prior et al. (2024). Identification of Precise 3D CT Radiomics for Habitat Computation by
  Machine Learning in Cancer. *Radiology: Artificial Intelligence*, 6(2), e230118.
- Bernatowicz et al. (2021). Robust imaging habitat computation using voxel-wise radiomics
  features. *Scientific Reports*, 11, 20133.
- Source code reference: https://github.com/radiomicsgroup/precise-habitats
