# RadiomicViz — CLAUDE.md

Guidelines for AI-assisted development of RadiomicViz. Merge with task-specific instructions as needed.

## What This Project Is

RadiomicViz is a Python package that wraps PyRadiomics with input validation, config presets, structured output, batch processing, SLURM tooling, an interactive 3D browser viewer, and habitat clustering. It replaces the bespoke scripts neuroimaging researchers rewrite for every radiomics project.

**Repository:** `https://github.com/levijb/radiomicviz`
**Layout:** `src/` layout, editable install via `pip install -e ".[dev]"`
**License:** MIT

## Project Phases

- **Phase 1 (complete):** Extraction layer — PyRadiomics wrapper with CLI, presets, batch, SLURM.
- **Phase 2 (complete):** Interactive 3D viewer — Flask + Niivue.js. result.view() and radiomicviz view CLI.
- **Phase 3 (complete):** Habitat clustering — GMM/K-means on voxelwise feature maps. See PHASE3_SPEC.md.
- **Phase 4 (planned):** ICC precision filtering, UMAP viewer, cross-subject alignment, SHAP saliency. See PHASE4_SPEC.md.

## Package Structure

```
src/radiomicviz/
├── __init__.py           # public API: extract, batch_extract, validate_inputs, etc.
├── _version.py           # v0.2.0
├── validate.py           # 9 input checks (shape, affine, empty mask, float mask, etc.)
├── config.py             # preset loading with fallback chain
├── result.py             # ExtractionResult dataclass — the central contract
├── habitat.py            # HabitatResult dataclass — clustering output contract
├── cluster.py            # cluster_habitats() and batch_cluster() functions
├── extract.py            # single-subject PyRadiomics wrapper (ROI + voxelwise)
├── batch.py              # parallel batch extraction with joblib
├── cohort.py             # configurable cohort CSV generator (BIDS-like folder traversal, --image-suffix, --mask-dir, etc.)
├── cli.py                # Click CLI: extract, batch-extract, validate, generate-slurm, etc.
├── _slurm.py             # SLURM script generator (single, array, chunked strategies)
├── presets/              # 7 YAML configs (mri-default, mri-texture, etc.)
└── viewer/               # Phase 2 — Flask + Niivue.js browser viewer
    ├── __init__.py
    ├── app.py
    └── templates/viewer.html
```

## Architecture Principles

1. **ExtractionResult is the contract.** It bridges extraction (Phase 1), viewer (Phase 2), and clustering (Phase 3). `result.view()` launches the browser viewer. Pass result to `cluster_habitats()` to compute habitats. Do not bypass this dataclass.
2. **Presets over raw config.** Users pick a named preset; custom YAML is the escape hatch. Fallback chain: custom config > named preset > mri-default.
3. **joblib for parallelism.** Not Python's multiprocessing module. joblib handles the GIL, serialization, and cleanup better for scientific workloads.
4. **Browser-based viewer only.** No Qt, no napari, no desktop GUI. Flask serves NIfTIs, Niivue.js renders via WebGL. Works over SSH with VS Code port forwarding.
5. **Click for CLI.** All commands under `radiomicviz` entry point.
6. **HabitatResult is the clustering contract.** `cluster_habitats(result)` → `HabitatResult`. `habitat.view()` launches the viewer with a discrete habitat overlay. Do not bypass this dataclass. The chain is: `ExtractionResult` → `cluster_habitats()` → `HabitatResult` → viewer.

---

## Critical: PyRadiomics Gotchas

These are hard-won bug fixes. Violating any of them will break the pipeline silently or loudly.

### Installation Order Matters

PyRadiomics has a fragile build. The correct install sequence is:

```bash
# Step 1: Create conda env (gets numpy, pandas, etc.)
conda env create -f environment.yaml
conda activate radiomicviz

# Step 2: Install pyradiomics AFTER numpy exists
pip install pyradiomics==3.0.1 --no-build-isolation --no-deps

# Step 3: Re-pin SimpleITK (pyradiomics pulls a newer version)
pip install SimpleITK==2.2.1

# Step 4: Editable install of the package itself
pip install -e . --no-deps
```

**Why these constraints:**
- `pyradiomics` is NOT on conda-forge. Must use pip.
- Version 3.1.0 has a broken build (missing `versioneer`). Pin to `3.0.1`.
- `--no-build-isolation` lets the build find the already-installed numpy for C++ compilation.
- pyradiomics drags in a newer SimpleITK as a dependency, which breaks `SetGlobalDefaultCoordinateTolerance`. Always re-pin SimpleITK to 2.2.1 after installing pyradiomics.

### Voxelwise Mode Config Injection

**The bug:** Setting `extractor.settings['voxelBased'] = True` on the extractor object after initialization does NOT work. PyRadiomics' internal logic to skip shape features only fires during `__init__`, so shape extraction crashes with "Shape features are not available in voxel-based mode."

**The fix:** Inject a `voxelSetting` block into the config dictionary BEFORE instantiating `RadiomicsFeatureExtractor`:

```python
# CORRECT — inject before instantiation
config["voxelSetting"] = {"voxelBased": True, "kernelRadius": 3}
extractor = featureextractor.RadiomicsFeatureExtractor(config)

# WRONG — setting after init does not trigger shape-skip logic
extractor = featureextractor.RadiomicsFeatureExtractor(config)
extractor.settings["voxelBased"] = True  # too late!
```

Alternatively, `voxelBased=True` can be passed directly to `extractor.execute()`:
```python
result = extractor.execute(image_path, mask_path, voxelBased=True, label=label)
```
In this case, feature map values come back as `SimpleITK.Image` objects (not floats). Check with `isinstance(val, sitk.Image)`.

### Voxelwise Output Format

- `extractor.execute()` returns an `OrderedDict`
- Scalar values = diagnostics (version info, config hash, etc.)
- `sitk.Image` values = voxelwise feature maps (one per feature)
- Save each feature map as `.nrrd` with `sitk.WriteImage(val, path)`
- For 4D NIfTI export: stack all feature maps along the 4th dimension using nibabel

### Build Backend

`pyproject.toml` must use:
```toml
build-backend = "setuptools.build_meta"
```
NOT `setuptools.backends._legacy:_Backend` (which was incorrectly generated once and will fail).

---

## Dataset & Cluster Context

### Zenodo MRI Dataset
- ~2,100 subjects in BIDS-like structure
- Path on cluster: `/mnt/lustre/lab/general/ctcn_imaging/Levi/`
- Structure: `Subjects/{subject}/{session}/derivatives/segmentation/*.nii.gz`
- T1 images: `{subject}_T1_bet_n4_nu.nii.gz` (suffix passed via `--image-suffix` to `generate-csv`)
- 13 tract segmentation masks per subject: CST_L, CST_R, AF_R, IFOF_L, etc.

### Cohort CSV Format

```csv
subject_id,session,mask_name,Image,Mask
sub001,ses-01,CST_L,/path/to/t1.nii.gz,/path/to/CST_L.nii.gz
```

Columns: `["subject_id", "session", "mask_name", "Image", "Mask"]`
The `Image` and `Mask` columns contain absolute paths. `subject_id` is auto-detected by RadiomicViz.

### Cluster Environment
- HPC with SLURM scheduler
- Conda environment: `radiomicviz` (Python 3.10, PyRadiomics 3.0.1, SimpleITK 2.2.1)
- Typical runtime: ~4 minutes per subject (ROI mode)
- Three SLURM strategies: `single` (one big job), `array` (one task per subject), `chunked` (split into N chunks)

### Voxelwise Output Naming

Files should be named: `{subject_id}_{session}_{modality}_{mask_name}_features4d.nii.gz`
Modality is inferred from the image filename (look for "t1" or "flair", case-insensitive). Default to "unknown".
Each subject gets its own subfolder under `per_subject/`.

---

## Presets

7 built-in YAML configs in `src/radiomicviz/presets/`:

| Preset | Use Case |
|---|---|
| `mri-default` | Balanced starting point — all feature classes, original images |
| `mri-texture` | Texture-only (GLCM, GLRLM, GLSZM, GLDM, NGTDM) |
| `mri-firstorder` | Shape + first-order statistics |
| `mri-voxelwise` | Curated subset for habitat clustering |
| `mri-all-transforms` | Exhaustive — LoG, Wavelet, Square, SquareRoot, Logarithm, Exponential, Gradient, LBP2D, LBP3D |
| `mri-voxelwise-wholebrain` | Whole-brain voxelwise (firstorder + GLCM + GLRLM) |
| `minimal` | Fast sanity checks (shape + 8 first-order stats) |

All presets use `binCount: 32` with `normalize: true` and `normalizeScale: 100` for cross-preset comparability. `mri-voxelwise` and `mri-voxelwise-wholebrain` include `voxelSetting` blocks for voxelwise extraction.

---

## Development Workflow

### Two-tool workflow
- **Web Claude (claude.ai):** Planning, architecture, design discussions, debugging strategies, writing prompts for Claude Code.
- **Terminal Claude Code:** Implementation, code changes, git operations. Uses local git credentials — avoids HTTP 403 push errors that desktop Claude Code hits.

### Running tests
```bash
# Validation + config tests (no real data needed — synthetic NIfTI fixtures)
pytest tests/test_validate.py tests/test_config.py -v

# Extraction tests (need pyradiomics installed)
pytest tests/test_extract.py -v

# Everything
pytest
```

| File | What it tests |
|---|---|
| `tests/test_validate.py` | Input validation: shape, affine, empty mask, float values |
| `tests/test_config.py` | Preset loading and config resolution |
| `tests/test_extract.py` | Core extraction, voxelwise brain modes, basic batch |
| `tests/test_clustering.py` | HabitatResult export methods, cluster_habitats() pipeline (GMM + K-means), batch_cluster() |

### Linting
```bash
ruff check src/
mypy src/radiomicviz/
```

### Quick smoke test with real data
```python
from radiomicviz import validate_inputs, extract

report = validate_inputs("sub01_T1.nii.gz", "sub01_mask.nii.gz")
result = extract("sub01_T1.nii.gz", "sub01_mask.nii.gz", preset="mri-default")
result.to_csv("test_output.csv")
```

```python
# Clustering smoke test
from radiomicviz import extract, cluster_habitats

result = extract("sub01_T1.nii.gz", "sub01_mask.nii.gz",
                 preset="mri-voxelwise", mode="voxelwise")
habitats = cluster_habitats(result, method="gmm", n_clusters="auto")
print(habitats.summary())
habitats.to_nifti("habitats.nii.gz")
habitats.view()  # launches browser with discrete habitat overlay
```

---

## Phase 2: Viewer (Complete)

See `VIEWER_SPEC.md` for the full spec. The viewer is built and functional.

- **Stack:** Flask (Python file server) + Niivue.js (WebGL renderer in browser)
- **API:** `result.view()` or `radiomicviz view --image t1.nii.gz --mask mask.nii.gz --subject-dir <dir>`
- **Routes:** `GET /` (viewer HTML), `GET /data/<file>` (NIfTI/NRRD files), `GET /api/volumes` (JSON manifest)
- **Works over SSH:** VS Code port forwarding — no X11, no desktop GUI
- **Niivue CDN:** `https://unpkg.com/@niivue/niivue/dist/niivue.umd.js`
- **Dependencies:** `flask>=2.3` (add via `pip install radiomicviz[viewer]`)

### Viewer UI (implemented)
- Orthogonal slice views (axial, coronal, sagittal) + 3D toggle
- Mask overlay with adjustable opacity
- ADD OVERLAY panel — always visible; select region + feature + colormap, stack multiple overlays
- Per-overlay Min/Max range sliders with colorbar
- 4D NIfTI mode: volume index slider, colormap reset on feature switch
- Crosshair navigation with voxel coordinate + value readout
- Save Screenshot action
- Background image switcher

### Known bugs fixed
- Critical spatial misalignment in `to_4d_nifti()` — cropped SimpleITK feature maps were saved with full-volume affine (fixed)
- Axis ordering in `_expand_to_full_shape()` — SimpleITK LPS / `(z,y,x)` ordering not correctly handled during padding (fixed)
- NiiVue colormap range not resetting when switching features in 4D mode (fixed)
- Flask was missing from the conda environment (added)
- `batch.py` Unicode encoding error on `errors.log` (fixed)

---

## Phase 3: Habitat Clustering (Complete)

### Data contract chain
ExtractionResult → cluster_habitats() → HabitatResult → .view()

### cluster_habitats() pipeline (in order)
1. Parse input — ExtractionResult or (4D NIfTI path + mask path)
2. Extract within-mask voxel×feature matrix
3. Drop constant and >10% NaN features; impute remaining NaNs with median
4. Spearman redundancy elimination (default threshold |r| > 0.70)
5. Z-score normalize per feature
6. Optional PCA (pca_components=None|int|"auto")
7. Covariance auto-fallback: if n_voxels < n_features×10 → switch to "diag", log warning
8. k selection if n_clusters="auto": BIC gradient for GMM, Silhouette max for K-means
9. Final fit at selected k
10. Back-project labels to full image shape (0 = background)
11. Compute per-habitat cluster_stats (mean/std/median/p10/p90 per feature)
12. Warn if any habitat has fewer than min_cluster_size voxels
13. Return HabitatResult

### Key implementation patterns

**Deferred viewer import in habitat.py** — avoids circular import at runtime:
```python
def view(self, **kwargs):
    from radiomicviz.viewer import launch_viewer_from_habitat
    launch_viewer_from_habitat(self, **kwargs)
```
Never import launch_viewer_from_habitat at module level in habitat.py.

**actc colormap** — NiiVue's built-in discrete/categorical colormap. Always use "actc" (not "categorical" or any custom palette) when loading a habitat label NIfTI in the viewer. The label map must be saved as int16 dtype for NiiVue to render it correctly as discrete colors.

**HabitatResult.probabilities shape** — (x, y, z, k) float32 for GMM; None for K-means. The 4th axis k matches n_clusters. Back-projected to full image shape the same way as label_map (0-probability for background voxels).

**Manifest keys for habitat viewer** — the Flask manifest gains two optional keys:
  "habitat_map":   url_key | null   (3D int16 NIfTI)
  "habitat_probs": url_key | null   (4D float32 NIfTI, GMM only)
Both are always present in the manifest dict (null when not used) so the Jinja2 template never raises KeyError.

---

## Behavioral Guidelines

Adapted from Andrej Karpathy's CLAUDE.md. These bias toward caution over speed.

### 1. Think Before Coding

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

- No features beyond what was asked.
- No abstractions for single-use code.
- No speculative "flexibility" or "configurability."
- If you write 200 lines and it could be 50, rewrite it.
- Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

**The test:** Every changed line should trace directly to the request.

### 4. Goal-Driven Execution

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

### 5. Respect the Existing Codebase

- `src/` layout — all source under `src/radiomicviz/`
- Click for CLI, not argparse
- joblib for parallelism, not multiprocessing
- ExtractionResult is the central data contract — don't bypass it
- Presets are YAML files in `src/radiomicviz/presets/` — don't hardcode configs
- Type hints on all public functions
- Docstrings in NumPy format
- Line length: 100 (ruff config)
- Target Python: 3.9+ (but developed/tested primarily on 3.10)

### 6. Dependencies

Before adding a new dependency:
- Check if it's already in `pyproject.toml`
- If it's only needed for viewer/analysis, put it in the appropriate optional group
- Never add pyradiomics to conda — it must be pip-installed
- Never upgrade SimpleITK past 2.2.1

---

## Common Pitfalls (Don't Repeat These)

1. **Don't set `voxelBased=True` after extractor init.** Inject into config dict first. (See "Voxelwise Mode Config Injection" above.)
2. **Don't use `pyradiomics==3.1.0`.** It's broken. Pin to 3.0.1.
3. **Don't use `multiprocessing.Pool`.** Use joblib.
4. **Don't build a desktop GUI for the viewer.** Browser-only via Flask + Niivue.js.
5. **Don't flatten batch output.** Each subject gets its own subfolder.
6. **Don't name output files by CSV row index.** Use `{subject_id}_{session}_{modality}_{mask_name}`.
7. **Don't assume shape features work in voxelwise mode.** They don't. PyRadiomics will crash.
8. **Don't forget `--no-build-isolation` when installing pyradiomics.** It needs numpy pre-installed for C++ compilation.
9. **Don't use `setuptools.backends._legacy:_Backend` as build backend.** Use `setuptools.build_meta`.
10. **Don't push from desktop Claude Code.** Use terminal Claude Code for git operations (local credentials).
11. **Don't save cropped feature maps with the full-volume affine.** `to_4d_nifti()` must use the cropped bounding-box affine from SimpleITK, then pad back to full volume with the original affine.
12. **Don't forget the SimpleITK axis swap in `_expand_to_full_shape()`.** SimpleITK returns arrays in `(z, y, x)` order — transpose to `(x, y, z)` before padding into the full-volume array.
13. **Don't add `flask` to `environment.yaml` under conda packages.** It's a pip dep — install via `pip install radiomicviz[viewer]` or add to the `pip:` block in environment.yaml.
14. **Don't open `errors.log` in `batch.py` without `encoding="utf-8"`.** Unicode in subject IDs or paths will crash the batch run on Windows or non-UTF-8 systems.
15. **Don't change preset discretization settings independently.** All 7 presets must share `binCount: 32`, `normalize: true`, `normalizeScale: 100`, `voxelArrayShift: 300`, `resampledPixelSpacing: null`. Cross-preset comparisons are only valid when preprocessing is identical.
16. **Don't import `launch_viewer_from_habitat` at module level in `habitat.py`.** It creates a circular import. Use a local import inside `view()` instead.
17. **Don't save habitat `label_map` as float.** Save as int16. NiiVue only renders discrete colors with the `actc` colormap on integer volumes.
18. **Don't pass `n_clusters` > `k_range` max to `cluster_habitats()`.** If forcing `n_clusters` as an int, it must be within `k_range` or clustering will fit only at that k regardless of criteria — this is intentional but can surprise.

---

*These guidelines are working if: diffs are minimal and focused, pyradiomics installs don't break, voxelwise extraction doesn't crash on shape features, and clarifying questions come before implementation rather than after mistakes.*
