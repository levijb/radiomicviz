# Phase 4 Spec — Deferred Features

Planning doc for features explicitly deferred from Phase 3. Not a design spec —
just enough context to understand what each item is, why it was deferred, and
what Phase 3 prerequisite it depends on.

---

## 1. ICC-Based Feature Precision Filtering

Filters features by intraclass correlation coefficient (ICC) before clustering, removing
features whose measurements are not reproducible under small perturbations (e.g., slight
mask erosion/dilation). Requires running extraction multiple times per subject with
perturbation applied via MIRP or a similar tool, then computing ICC across runs.
Deferred because it adds a multi-run extraction pipeline that doesn't exist yet; it is
only useful once `cluster_habitats()` (Phase 3 Step B) is working and the feature matrix
shape is known.

**Prerequisite:** Phase 3 Step B (`cluster_habitats()` + feature selection pipeline).

---

## 2. Cross-Subject Habitat Alignment

After clustering independently per subject, habitat labels are arbitrary (Habitat 1 in
subject A may correspond to Habitat 2 in subject B). The Hungarian algorithm matches
habitat labels across subjects before group-level statistics are computed. Deferred
because it requires a complete cohort of `HabitatResult` objects from `batch_cluster()`,
which is built in Phase 3 Step D.

**Prerequisite:** Phase 3 Step D (`batch_cluster()`).

---

## 3. Spatially-Aware Clustering

Adds the voxel's position along the ROI's principal axis as an additional clustering
feature. Relevant for geometrically elongated structures like white matter tracts (e.g.,
CST, arcuate fasciculus), where spatial gradients carry biological meaning beyond voxel
intensity. Requires computing the principal axis of the ROI mask, which is a non-trivial
geometric operation. Deferred until basic clustering is validated and the spatial feature
definition is settled.

**Prerequisite:** Phase 3 Step B; validated cluster quality on real tract data.

---

## 4. UMAP Visualization in the Viewer

Projects the voxel feature space to 2D using UMAP and shows a scatter plot in the viewer
sidebar, with points colored by habitat label. This is a visualization tool, not a
clustering preprocessor (UMAP as a preprocessing step is explicitly out of scope). Deferred
because the HABITATS sidebar panel (Phase 3 Step E) must exist first, and `umap-learn` adds
a non-trivial dependency that is not yet in `pyproject.toml`.

**Prerequisite:** Phase 3 Step E (viewer HABITATS panel); add `umap-learn` to `[analysis]`
extras.

---

## 5. Multi-Kernel-Radius Presets

Adds `mri-voxelwise-r3.yaml` (kernelRadius: 3) alongside the existing `mri-voxelwise`
preset (kernelRadius: 1) to enable reproducibility comparisons across kernel radii.
Deferred because the correct discretization settings for the larger kernel need validation
against real data, and adding a new preset without that validation could mislead users.

**Prerequisite:** Phase 3 extraction runs on real data with `mri-voxelwise`; compare
feature distributions before introducing a second preset.

---

## 6. Cohort-Level Habitat Statistics

Group overlays and statistical maps comparing habitat composition across subject groups
(e.g., patients vs. controls, responders vs. non-responders). Requires cross-subject
habitat alignment (item 2 above) and a group-assignment column in the cohort CSV.
Also requires the viewer to support loading multiple subjects' label maps simultaneously,
which is a substantial UI addition.

**Prerequisite:** Phase 4 item 2 (cross-subject alignment); extended viewer manifest.

---

## 7. Model Saliency (SHAP / Permutation Importance)

Back-projects feature importance scores from a trained classifier (SHAP values or
permutation importance) into brain space, producing a 3D saliency map per habitat.
Requires a downstream classification step (not in scope for Phase 3) and depends on
`shap` being added to `pyproject.toml`. Most useful once cohort-level habitat statistics
(item 6) can define the target classes.

**Prerequisite:** Phase 4 item 6 (cohort statistics); trained classifier; add `shap` to
`[analysis]` extras.
