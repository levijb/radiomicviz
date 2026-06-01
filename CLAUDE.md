# RadiomicViz — CLAUDE.md

Guidelines for AI-assisted development of RadiomicViz. Merge with task-specific instructions as needed.

## What This Project Is

RadiomicViz is a Python package that wraps PyRadiomics with input validation, config presets, structured output, batch processing, SLURM tooling, and (eventually) an interactive 3D brain viewer. It replaces the bespoke scripts neuroimaging researchers rewrite for every radiomics project.

**Repository:** `https://github.com/levijb/radiomicviz`
**Layout:** `src/` layout, editable install via `pip install -e ".[dev]"`
**License:** MIT

## Project Phases

- **Phase 1 (active):** Extraction layer — working. PyRadiomics wrapper with CLI, presets, batch, SLURM.
- **Phase 2 (planned):** Interactive 3D viewer — browser-based, Flask + Niivue.js. See `VIEWER_SPEC.md`.
- **Phase 3 (future):** Analysis modules — clustering (GMM, PCA, UMAP), model saliency (SHAP), cohort statistics.

## Package Structure

```
src/radiomicviz/
├── __init__.py           # public API: extract, batch_extract, validate_inputs, etc.
├── _version.py           # v0.1.0
├── validate.py           # 9 input checks (shape, affine, empty mask, float mask, etc.)
├── config.py             # preset loading with fallback chain
├── result.py             # ExtractionResult dataclass — the central contract
├── extract.py            # single-subject PyRadiomics wrapper (ROI + voxelwise)
├── batch.py              # parallel batch extraction with joblib
├── cohort.py             # cohort CSV generator for Zenodo dataset
├── cli.py                # Click CLI: extract, batch-extract, validate, generate-slurm, etc.
├── _slurm.py             # SLURM script generator (single, array, chunked strategies)
├── presets/              # 7 YAML configs (mri-default, mri-texture, etc.)
└── viewer/               # Phase 2 — Flask + Niivue.js browser viewer
    ├── __init__.py
    ├── app.py
    └── templates/viewer.html
```

## Architecture Principles

1. **ExtractionResult is the contract.** It bridges Phase 1 (extraction) and Phase 2 (viewer). `result.view()` will launch the browser viewer directly. Do not bypass this dataclass.
2. **Presets over raw config.** Users pick a named preset; custom YAML is the escape hatch. Fallback chain: custom config > named preset > mri-default.
3. **joblib for parallelism.** Not Python's multiprocessing module. joblib handles the GIL, serialization, and cleanup better for scientific workloads.
4. **Browser-based viewer only.** No Qt, no napari, no desktop GUI. Flask serves NIfTIs, Niivue.js renders via WebGL. Works over SSH with VS Code port forwarding.
5. **Click for CLI.** All commands under `radiomicviz` entry point.

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
- T1 images: `{subject}_T1_lesion_filled_combined_mask_bet_n4_nu.nii.gz`
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
| `mri-habitat` | Curated subset for habitat clustering |
| `mri-all-transforms` | Exhaustive — LoG, Wavelet, Square, SquareRoot, Logarithm, Exponential, Gradient, LBP2D, LBP3D |
| `mri-wholebrain` | Whole-brain voxelwise (firstorder + GLCM + GLRLM) |
| `minimal` | Fast sanity checks (shape + 8 first-order stats) |

Most presets use `binCount: 32` with normalization. `mri-all-transforms` uses `binWidth: 25` without normalization (matching the standard PyRadiomics example config).

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

---

## Phase 2: Viewer (Planned)

See `VIEWER_SPEC.md` for the full spec. Key points:

- **Stack:** Flask (Python file server) + Niivue.js (WebGL renderer in browser)
- **API:** `result.view()` or `radiomicviz view --image t1.nii.gz --mask mask.nii.gz`
- **Routes:** `GET /` (viewer HTML), `GET /data/<file>` (NIfTI files), `GET /api/volumes` (JSON manifest)
- **Why not Panel/napari:** Works over SSH, no X11 needed, one small dependency (Flask), all rendering is client-side
- **Niivue CDN:** `https://unpkg.com/@niivue/niivue/dist/niivue.umd.js`
- **Dependencies:** Only `flask>=2.3` (nibabel already a core dep, Niivue loaded from CDN)

### Viewer UI (must-haves)
- Orthogonal slice views (axial, coronal, sagittal)
- 3D volume rendering toggle
- Mask overlay with adjustable opacity
- Feature map dropdown (populated from `/api/volumes`)
- Colormap selector (viridis, hot, cool, inferno, etc.)
- Crosshair navigation with voxel coordinates

---

## Phase 3: Analysis (Future)

- Clustering / dimensionality reduction: GMM, PCA, UMAP on extracted features
- Model saliency: SHAP / permutation importance back-projected to brain space
- Cohort mode: subject browser, group overlays, statistical maps
- Dependencies: `scikit-learn`, `scipy`, `shap`, `umap-learn`

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

---

*These guidelines are working if: diffs are minimal and focused, pyradiomics installs don't break, voxelwise extraction doesn't crash on shape features, and clarifying questions come before implementation rather than after mistakes.*
