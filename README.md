# RadiomicViz

Interactive 3D radiomics extraction, visualization, and analysis for neuroimaging.

RadiomicViz wraps PyRadiomics with strict input validation, built-in presets, structured output, habitat clustering, and cluster submission tooling.

## Installation


### From source with conda (recommended)

This approach handles pyradiomics' C++ compilation cleanly across platforms.

1. Clone the repository:
```bash
   git clone https://github.com/levijb/radiomicviz.git
   cd radiomicviz
```

2. Create the conda environment:
```bash
   conda env create -f environment.yaml
   conda activate radiomicviz
```

3. Install pyradiomics and the package:
```bash
   pip install setuptools numpy wheel
   pip install pyradiomics==3.0.1 --no-build-isolation --no-deps
   pip install SimpleITK==2.2.1
   pip install -e . --no-deps
```

### With pip only

If you prefer not to use conda, make sure you have a C++ compiler available (MSVC on Windows, gcc on Linux/macOS) since pyradiomics builds from source.

```bash
# Core extraction
pip install radiomicviz

# With viewer (Phase 2)
pip install radiomicviz[viewer]

# With analysis tools (clustering, SHAP, etc.)
pip install radiomicviz[analysis]

# Everything
pip install radiomicviz[all]
```

### Development install (pip only)

```bash
git clone https://github.com/YOUR_USERNAME/radiomicviz.git
cd radiomicviz
pip install -e ".[dev]"
```

### Troubleshooting

**pyradiomics fails to install:** The latest release (3.1.0) has a known build issue. Always pin to 3.0.1:

```bash
pip install setuptools numpy wheel
pip install pyradiomics==3.0.1 --no-build-isolation --no-deps
pip install SimpleITK==2.2.1
```

The `--no-build-isolation` flag lets the build step find your already-installed numpy, and pinning to 3.0.1 avoids a missing `versioneer` error in 3.1.0. pyradiomics pulls in a newer SimpleITK as a dependency that breaks `SetGlobalDefaultCoordinateTolerance`; re-pinning to 2.2.1 fixes this.
## Quick Start

### Python API

```python
from radiomicviz import extract, batch_extract, validate_inputs

# 1. Validate first (optional but recommended)
report = validate_inputs("sub01_T1.nii.gz", "sub01_lesions.nii.gz")
print(report)  # shows any issues

# 2. Single-subject extraction
result = extract(
    "sub01_T1.nii.gz",
    "sub01_lesions.nii.gz",
    preset="mri-default",
    mode="roi",
)
print(result.summary())
result.features.head()          # pandas DataFrame
result.to_csv("features.csv")   # export with metadata sidecar
result.to_nifti("./nifti_out/") # paint features back onto brain

# 3. Batch extraction
results = batch_extract(
    "cohort.csv",
    image_col="t1_path",
    mask_col="mask_path",
    preset="mri-texture",
    n_jobs=4,
    output_dir="./radiomics_output/",
)
# Outputs: per-subject CSVs, combined_features.csv, batch_manifest.json
```

### CLI

```bash
# Single subject
radiomicviz extract \
    --image sub01_T1.nii.gz \
    --mask sub01_lesions.nii.gz \
    --preset mri-default \
    --output features.csv

# Batch
radiomicviz batch-extract \
    --subjects cohort.csv \
    --image-col t1_path \
    --mask-col mask_path \
    --preset mri-texture \
    --n-jobs 8 \
    --output-dir ./radiomics_output/

# Validate inputs before extraction
radiomicviz validate --image sub01_T1.nii.gz --mask sub01_lesions.nii.gz

# Browse presets
radiomicviz list-presets
radiomicviz show-preset mri-texture
```

## Cohort CSV Generation

`generate-csv` builds the cohort CSV required by `batch-extract` from a BIDS-like study folder. Use it when your data is organized under a `Subjects/{subject}/{session}/` hierarchy — one call produces a ready-to-use CSV with columns `subject_id`, `session`, `mask_name`, `Image`, `Mask`.

### Expected folder structure

```
study_folder/
└── Subjects/
    └── sub-01/
        └── ses-01/
            ├── T1/                          ← --image-subdir (default: T1)
            │   └── sub-01*{suffix}          ← --image-suffix (required)
            └── derivatives/segmentation/    ← --mask-dir (default)
                └── *.nii.gz                 ← --mask-suffix (default: .nii.gz)
```

### CLI

Use `--dry-run` first to verify the glob finds the right files before writing the CSV:

```bash
# Dry run — verify glob pattern before writing
radiomicviz generate-csv \
    --study-folder /mnt/lustre/lab/Levi/ \
    --output-csv-name cohort \
    --image-suffix _T1_lesion_filled_combined_mask_bet_n4_nu.nii.gz \
    --dry-run

# Write the CSV
radiomicviz generate-csv \
    --study-folder /mnt/lustre/lab/Levi/ \
    --output-csv-name cohort \
    --image-suffix _T1_lesion_filled_combined_mask_bet_n4_nu.nii.gz

# Custom layout (FLAIR images in a different subdir, masks in a custom folder)
radiomicviz generate-csv \
    --study-folder /data/study/ \
    --output-csv-name cohort \
    --image-suffix _FLAIR_brain.nii.gz \
    --image-subdir FLAIR \
    --mask-dir derivatives/masks \
    --mask-suffix _tract.nii.gz \
    --output-dir /data/study/outputs/
```

### Python API

```python
from radiomicviz import generate_cohort_csv

summary = generate_cohort_csv(
    study_folder="/mnt/lustre/lab/Levi/",
    output_csv_name="cohort",
    image_suffix="_T1_lesion_filled_combined_mask_bet_n4_nu.nii.gz",
)
print(f"{summary['n_subjects']} subjects, {summary['n_rows']} rows → {summary['csv_path']}")
```

Then feed the CSV to `batch-extract`:

```bash
radiomicviz batch-extract \
    --subjects cohort.csv \
    --image-col Image \
    --mask-col Mask \
    --preset mri-default \
    --n-jobs 8 \
    --output-dir ./radiomics_output/
```

## Presets

Built-in extraction configurations. Use `show_preset("name")` to inspect the full YAML.

| Preset | Purpose | Image Types | Feature Classes |
|---|---|---|---|
| `mri-default` | Balanced starting point | Original | All (shape, firstorder, GLCM, GLRLM, GLSZM, GLDM) |
| `mri-texture` | Texture features only | Original | GLCM, GLRLM, GLSZM, GLDM, NGTDM |
| `mri-firstorder` | Shape + first-order stats | Original | Shape, firstorder |
| `mri-voxelwise` | Habitat clustering workflows (voxelSetting included) | Original | Curated first-order + texture subset |
| `mri-all-transforms` | Exhaustive (thousands of features) | Original, LoG, Wavelet, Square, SquareRoot, Logarithm, Exponential, Gradient, LBP2D, LBP3D | All |
| `mri-voxelwise-wholebrain` | Whole-brain voxelwise (voxelSetting included) | Original | firstorder (10) + GLCM (7) + GLRLM (4) |
| `minimal` | Fast sanity checks | Original | Shape + 8 first-order stats |

### Standardized Settings

All presets share a common discretization pipeline so that features extracted with
any preset are directly comparable:

- **Normalization:** enabled (`normalize: true`, `normalizeScale: 100`) — required for MRI where intensity units are arbitrary
- **Binning:** `binCount: 32` (adaptive bin count, not fixed bin width)
- **Intensity shift:** `voxelArrayShift: 300` (ensures positive values for texture computation)
- **Resampling:** disabled (`resampledPixelSpacing: null`) — assumes data is already in consistent voxel space
- **Interpolation:** B-spline (`sitkBSpline`)

Presets differ only in which feature classes and image transforms they enable,
not in how the image is preprocessed. This means you can run `mri-firstorder`
on a subject, then later run `mri-texture` on the same subject, and safely
combine the results.

If your data has inconsistent voxel sizes across subjects, add resampling via
config override:

```python
result = extract("t1.nii.gz", "mask.nii.gz",
                 preset="mri-default",
                 overrides={"setting": {"resampledPixelSpacing": [1, 1, 1]}})
```

### Custom configs

```python
# Use your own YAML
result = extract("t1.nii.gz", "mask.nii.gz", config="my_params.yaml")

# Or start from a preset and tweak
result = extract("t1.nii.gz", "mask.nii.gz",
                 preset="mri-default",
                 overrides={"binWidth": 50, "label": 2})
```

### Voxelwise Extraction

In voxelwise mode, PyRadiomics slides a kernel across every voxel inside the mask and
computes features at each position, producing a 3D feature map per feature rather than
a single summary value per ROI. Use this for spatial heterogeneity analysis — mapping
texture variation across a tract or identifying habitat subregions within a tumor.

**Which preset to use:**

| Use case | Preset |
|---|---|
| ROI extraction (one scalar per feature per ROI) | `mri-default`, `mri-texture`, `mri-firstorder`, `mri-all-transforms`, `minimal` |
| Voxelwise on a single ROI (one tract, one lesion) | `mri-voxelwise` |
| Voxelwise over the whole brain with a parcellation mask | `mri-voxelwise-wholebrain` |

**`brain_mode`** (voxelwise only, controls how multi-label masks are handled):

| Value | Behavior |
|---|---|
| `"whole"` | Binarizes the mask (all nonzero → 1), extracts once over the full brain. Fast, loses region identity. |
| `"per-region"` | Extracts each label separately. Accurate but slow with many labels. |
| `"hybrid"` | Binarizes and extracts once, stores the original label map for post-hoc analysis via `result.features_by_region(label)`. |

**Single ROI (e.g., one white-matter tract):**

```bash
radiomicviz extract -i t1.nii.gz -m CST_L.nii.gz --preset mri-voxelwise --mode voxelwise -o features.csv
```

```python
result = extract("t1.nii.gz", "CST_L.nii.gz", preset="mri-voxelwise", mode="voxelwise")
result.to_4d_nifti("feature_maps.nii.gz")
```

**Output options:**

- **4D NIfTI** (default in voxelwise mode): all feature maps stacked into a single
  `(x, y, z, n_features)` NIfTI file, with a sidecar JSON listing the feature names.
  Good for viewing with the RadiomicViz viewer (`--feature-4d`) and for downstream
  analysis that loads the full stack at once.
- **Individual `.nrrd` files** (`--save-maps`): one file per feature per ROI label,
  saved in a subdirectory named by the ROI (`--roi-name`, or `label{N}` as a fallback).
  Good for selective loading of specific features, multi-region viewer mode
  (`--subject-dir`), and as input to habitat clustering.

```bash
# Single subject: get both 4D NIfTI and individual .nrrd files
radiomicviz extract \
  -i t1.nii.gz -m CST_L.nii.gz \
  --preset mri-voxelwise --mode voxelwise \
  --save-maps --roi-name CST_L \
  -o features.csv
```

```bash
# Batch: save .nrrd maps for all subjects
radiomicviz batch-extract \
  -s cohort.csv --image-col Image --mask-col Mask \
  --preset mri-voxelwise --mode voxelwise \
  --save-maps --roi-name-col mask_name \
  -o output/ -n 4
```

**Whole-brain parcellation:**

```python
result = extract("t1.nii.gz", "parcellation.nii.gz",
                 preset="mri-voxelwise-wholebrain",
                 mode="voxelwise", brain_mode="hybrid")
```

### Whole-brain voxelwise extraction

```python
# Strategy 1: One binarized whole-brain mask
result = extract("t1.nii.gz", "samseg.nii.gz",
                 preset="mri-voxelwise-wholebrain",
                 mode="voxelwise",
                 brain_mode="whole")

# Strategy 2: Per-region extraction
result = extract("t1.nii.gz", "samseg.nii.gz",
                 preset="mri-voxelwise-wholebrain",
                 mode="voxelwise",
                 brain_mode="per-region")

# Strategy 3: Hybrid — extract whole-brain, analyze per-region later
result = extract("t1.nii.gz", "samseg.nii.gz",
                 preset="mri-voxelwise-wholebrain",
                 mode="voxelwise",
                 brain_mode="hybrid")

# Post-hoc region analysis (hybrid only)
regions = result.available_regions()        # [2, 3, 4, ...]
hippo_df = result.features_by_region(17)    # label 17 = hippocampus
caudate_df = result.features_by_region(11)  # label 11 = caudate
```

## Input Format

**Image**: Any 3D NIfTI file (.nii or .nii.gz). T1, FLAIR, QSM, or any quantitative map.

**Mask**: 3D NIfTI with integer labels. `0` = background, nonzero integers = ROIs. Can be binary (single ROI) or multi-label.

**Requirements** (enforced by validation):
- Image and mask must have the same shape and affine
- Mask values must be non-negative integers
- At least one nonzero voxel in the mask
- ROIs with < 10 voxels trigger a warning (texture features unreliable)

**Subjects CSV** (for batch mode):

```csv
subject_id,t1_path,mask_path,group
sub01,/data/sub01/t1.nii.gz,/data/sub01/mask.nii.gz,MS
sub02,/data/sub02/t1.nii.gz,/data/sub02/mask.nii.gz,HC
```

Column names are flexible — you specify them via `image_col` and `mask_col`. Subject ID is auto-detected from columns named `subject_id`, `Patient`, `participant_id`, or `ID`.

## Output Format

`extract()` returns an `ExtractionResult` with:

| Attribute | Type | Description |
|---|---|---|
| `.features` | `pd.DataFrame` | Rows = ROI labels, columns = feature names |
| `.metadata` | `ExtractionMetadata` | Image path, mask path, config used, timestamps, versions |
| `.diagnostics` | `list[ROIDiagnostic]` | Per-ROI voxel count, bounding box, warnings |
| `.feature_names` | `list[str]` | All feature names |
| `.n_features` | `int` | Number of features extracted |
| `.n_rois` | `int` | Number of ROIs |

**Export methods:**
- `.to_csv(path)` — features CSV + metadata JSON sidecar
- `.to_nifti(dir)` — each feature as a 3D NIfTI (choropleth)
- `.to_4d_nifti(path)` — all voxelwise features stacked as 4D + sidecar

## SLURM Cluster Submission

Three strategies for HPC:

```bash
# Strategy 1: single — one job, entire cohort (like your current script)
radiomicviz generate-slurm \
    --subjects cohort.csv \
    --image-col t1_path \
    --mask-col mask_path \
    --strategy single \
    --conda-env radiomics_env \
    --conda-sh /path/to/conda.sh \
    --constraint "cpu8mem64a"

# Strategy 2: array — one SLURM array task per subject (max parallelism)
radiomicviz generate-slurm \
    --subjects cohort.csv \
    --image-col t1_path \
    --mask-col mask_path \
    --strategy array

# Strategy 3: chunked — split into N chunks, one job per chunk
radiomicviz generate-slurm \
    --subjects cohort.csv \
    --image-col t1_path \
    --mask-col mask_path \
    --strategy chunked \
    --chunks 10
```

Generated scripts handle conda activation, logging, and error reporting. The `array` strategy also generates a `merge_results.sh` script to combine outputs after all tasks complete.

## Habitat Clustering

After voxelwise extraction, `cluster_habitats()` groups voxels within the ROI into spatially distinct imaging habitats using Gaussian Mixture Models (GMM) or K-means clustering.

### Python API

```python
from radiomicviz import extract, cluster_habitats

# Step 1: voxelwise extraction
result = extract(
    "sub01_T1.nii.gz",
    "sub01_mask.nii.gz",
    preset="mri-voxelwise",
    mode="voxelwise",
)

# Step 2: cluster into habitats
habitats = cluster_habitats(
    result,
    method="gmm",          # "gmm" (default) or "kmeans"
    n_clusters="auto",     # auto-selects k in range 2–6 via BIC
    redundancy_threshold=0.70,   # Spearman |r| cutoff for feature filtering
    pca_components=None,   # None, int, or "auto" (95% variance)
    random_state=42,
)

print(habitats.summary())           # human-readable summary
habitats.to_nifti("habitats.nii.gz")  # 3D integer label map
habitats.to_csv("cluster_stats.csv")  # per-habitat feature statistics
habitats.view()                     # browser viewer with discrete overlay
```

### CLI

```bash
# Single subject
radiomicviz cluster \
  --feature-4d features_4d.nii.gz \
  --mask ROI.nii.gz \
  --method gmm \
  --n-clusters auto \
  --output-dir ./habitats/

# Batch
radiomicviz batch-cluster \
  --subjects cohort.csv \
  --feature-4d-col feature_4d_path \
  --mask-col Mask \
  --output-dir ./habitats/ \
  --n-jobs 4
```

### Outputs

| File | Description |
|---|---|
| `habitats.nii.gz` | 3D integer NIfTI — habitat label per voxel (0 = background) |
| `habitats_probs.nii.gz` | 4D float NIfTI — GMM soft assignment probabilities (optional) |
| `cluster_stats.csv` | Per-habitat mean/std/median/p10/p90 for each feature |
| `cluster_stats.metadata.json` | Clustering provenance (method, k, features used/dropped) |
| `batch_manifest.json` | Batch run summary — successes, failures, timing (batch mode) |

### Clustering pipeline

Each `cluster_habitats()` call runs in order: drop bad features → impute NaNs → Spearman redundancy elimination → z-score normalize → optional PCA → auto-select k (BIC gradient for GMM, Silhouette max for K-means) → fit → back-project to image space → compute per-habitat statistics.

### Methods

Habitat clustering methodology informed by:

- Prior O, et al. (2024). Identification of Precise 3D CT Radiomics for Habitat Computation by Machine Learning in Cancer. *Radiology: Artificial Intelligence*, 6(2), e230118. https://doi.org/10.1148/ryai.230118
- Bernatowicz K, et al. (2021). Robust imaging habitat computation using voxel-wise radiomics features. *Scientific Reports*, 11, 20133. https://doi.org/10.1038/s41598-021-99701-2
- Reference implementation: https://github.com/radiomicsgroup/precise-habitats

## Running the Tests

### Prerequisites

```bash
# Core suite (no real data needed — uses synthetic NIfTI fixtures)
pip install -e ".[dev]"

# sklearn-dependent habitat clustering test (optional)
pip install -e ".[analysis]"
```

Extraction tests also require pyradiomics installed per the pinned-version steps above. If it isn't installed, the suite skips those tests cleanly.

### Run everything

```bash
pytest
```

### Run the extraction test files only

```bash
pytest tests/test_roi_extraction.py tests/test_habitat_extraction.py tests/test_batch_extraction.py -v
```

### What's covered

| File | What it tests |
|---|---|
| `tests/test_validate.py` | Input validation: shape, affine, empty mask, float values |
| `tests/test_config.py` | Preset loading and config resolution |
| `tests/test_extract.py` | Core extraction, voxelwise brain modes, basic batch |
| `tests/test_roi_extraction.py` | ROI extraction with mri-default/texture/firstorder; CSV and NIfTI export |
| `tests/test_habitat_extraction.py` | mri-voxelwise preset; curated feature count; clustering-readiness |
| `tests/test_batch_extraction.py` | batch_extract(): error isolation, parallel runs, combined CSV, manifest |
| `tests/test_clustering.py` | HabitatResult export methods, cluster_habitats() pipeline (GMM + K-means), batch_cluster() |

See [TESTING.md](TESTING.md) for full details on prerequisites, individual test selection, and reading failures.

## Development

```bash
pip install -e ".[dev]"
ruff check src/           # lint
mypy src/radiomicviz/     # type check
```

## Methods & Attribution

Habitat clustering methodology informed by:

- Prior et al. (2024). Identification of Precise 3D CT Radiomics for Habitat Computation by Machine Learning in Cancer. *Radiology: Artificial Intelligence*, 6(2), e230118.
- Bernatowicz et al. (2021). Robust imaging habitat computation using voxel-wise radiomics features. *Scientific Reports*, 11, 20133.
- Source code reference: https://github.com/radiomicsgroup/precise-habitats

## License

MIT
