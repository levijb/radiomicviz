# Testing RadiomicViz

A practical guide for anyone who just cloned the repo.

---

## Prerequisites

### Core test suite (all tests except the sklearn clustering test)

```bash
# Follow the README install steps first, then:
pip install -e ".[dev]"
```

This installs pytest, pytest-cov, ruff, and mypy. PyRadiomics must be installed
separately per the README's pinned-version instructions (see below).

### sklearn-dependent test (habitat clustering)

```bash
pip install -e ".[analysis]"
```

`scikit-learn` is in the `analysis` extras group, not `dev`. Without it, the
`TestHabitatClusteringReady::test_sklearn_scaler_succeeds` test skips cleanly
via `pytest.importorskip`. All other habitat tests still run.

### PyRadiomics

Extraction tests require PyRadiomics installed with the **exact** pinned version
from the README. The correct install sequence is:

```bash
conda env create -f environment.yaml   # creates the radiomicviz conda env
conda activate radiomicviz
pip install pyradiomics==3.0.1 --no-build-isolation --no-deps
pip install SimpleITK==2.2.1
pip install -e . --no-deps
```

If PyRadiomics is not installed, the entire test suite is skipped cleanly via
`pytest.importorskip("radiomics")` at the top of each extraction test module.

---

## Running the tests

### Everything

```bash
pytest
```

### The three new extraction test files only

```bash
pytest tests/test_roi_extraction.py tests/test_habitat_extraction.py tests/test_batch_extraction.py -v
```

### A single test file

```bash
pytest tests/test_roi_extraction.py -v
```

### A single test class or test

```bash
pytest tests/test_roi_extraction.py::TestSingleLabelDefault -v
pytest tests/test_roi_extraction.py::TestSingleLabelDefault::test_feature_count -v
```

### With full tracebacks (useful for debugging failures)

```bash
pytest --tb=long
```

### Save output to a log file

```bash
pytest -v --tb=short 2>&1 | tee test_output.log
```

---

## What each test file covers

| File | What it tests |
|---|---|
| `tests/test_validate.py` | Input validation: shape, affine, empty mask, float mask, etc. |
| `tests/test_config.py` | Preset loading, config resolution, override merging |
| `tests/test_extract.py` | Basic extraction, multi-label, voxelwise brain modes, batch |
| `tests/test_roi_extraction.py` | ROI extraction with mri-default, mri-texture, mri-firstorder; CSV/NIfTI export; validation guards |
| `tests/test_habitat_extraction.py` | mri-habitat preset; feature count vs default; clustering-readiness |
| `tests/test_batch_extraction.py` | batch_extract() with n_jobs=1 and n_jobs=2; error isolation; combined CSV; manifest |

---

## Reading a failure

**Short tracebacks** (`--tb=short`, the default) show the failing assert and the
five lines around it. Good for understanding *what* failed.

**Long tracebacks** (`--tb=long`) show the full call stack. Use when you need
to see how the failure propagated through the code.

**The test output log** (`test_output.log` from `tee`) persists the full output
so you can inspect it after the run or share it in a bug report.

### Common failure patterns

| Symptom | Likely cause |
|---|---|
| All extraction tests skipped | PyRadiomics not installed — follow README install steps |
| `SetGlobalDefaultCoordinateTolerance` error | Wrong SimpleITK version; pin to 2.2.1 |
| `Shape features not available in voxel-based mode` | Someone set `voxelBased=True` after extractor init — see CLAUDE.md |
| `NaN in features` | Fixture ROI is too small or uniform — increase ROI size |
| `habitat n_features >= default n_features` | `mri-habitat.yaml` missing `featureClass` section |

---

## Notes

- Tests create small synthetic 20×20×20 NIfTI volumes (no real data needed).
- Session-scoped fixtures are created once and reused across the session.
- `tmp_path` and `tmp_path_factory` fixtures are pytest builtins — no cleanup needed.
- Extraction tests are the slowest (~30 s for the full ROI suite, ~2 min for batch). 
  Use `-x` (stop on first failure) during development.
