# RadiomicViz

Interactive 3D radiomics extraction, visualization, and analysis for neuroimaging.

RadiomicViz wraps PyRadiomics with strict input validation, built-in presets, structured output, and cluster submission tooling — replacing the bespoke scripts you rewrite for every project.

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

## Presets

Built-in extraction configurations. Use `show_preset("name")` to inspect the full YAML.

| Preset | Purpose | Image Types | Feature Classes |
|---|---|---|---|
| `mri-default` | Balanced starting point | Original | All (shape, firstorder, GLCM, GLRLM, GLSZM, GLDM) |
| `mri-texture` | Texture features only | Original | GLCM, GLRLM, GLSZM, GLDM, NGTDM |
| `mri-firstorder` | Shape + first-order stats | Original | Shape, firstorder |
| `mri-habitat` | Habitat clustering workflows | Original | Curated first-order + texture subset |
| `mri-all-transforms` | Exhaustive (thousands of features) | Original, LoG, Wavelet, Square, SquareRoot, Logarithm, Exponential, Gradient, LBP2D, LBP3D | All |
| `mri-wholebrain` | Whole-brain voxelwise | Original | firstorder (10) + GLCM (7) + GLRLM (4) |
| `minimal` | Fast sanity checks | Original | Shape + 8 first-order stats |

All presets use `binCount: 32` and normalization except `mri-all-transforms` (which uses `binWidth: 25` with no normalization, matching the standard PyRadiomics example config).

### Custom configs

```python
# Use your own YAML
result = extract("t1.nii.gz", "mask.nii.gz", config="my_params.yaml")

# Or start from a preset and tweak
result = extract("t1.nii.gz", "mask.nii.gz",
                 preset="mri-default",
                 overrides={"binWidth": 50, "label": 2})
```

### Whole-brain voxelwise extraction

```python
# Strategy 1: One binarized whole-brain mask
result = extract("t1.nii.gz", "samseg.nii.gz",
                 preset="mri-wholebrain",
                 mode="voxelwise",
                 brain_mode="whole")

# Strategy 2: Per-region extraction
result = extract("t1.nii.gz", "samseg.nii.gz",
                 preset="mri-wholebrain",
                 mode="voxelwise",
                 brain_mode="per-region")

# Strategy 3: Hybrid — extract whole-brain, analyze per-region later
result = extract("t1.nii.gz", "samseg.nii.gz",
                 preset="mri-wholebrain",
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
| `tests/test_habitat_extraction.py` | mri-habitat preset; curated feature count; clustering-readiness |
| `tests/test_batch_extraction.py` | batch_extract(): error isolation, parallel runs, combined CSV, manifest |

See [TESTING.md](TESTING.md) for full details on prerequisites, individual test selection, and reading failures.

## Development

```bash
pip install -e ".[dev]"
ruff check src/           # lint
mypy src/radiomicviz/     # type check
```

## License

MIT
