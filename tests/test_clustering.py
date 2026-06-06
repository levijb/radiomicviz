"""Tests for Phase 3 clustering module.

Step A: TestHabitatResult — dataclass construction and export methods.
Step B: TestClusterHabitats — cluster_habitats() pipeline.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from radiomicviz.habitat import HabitatResult
from radiomicviz.cluster import cluster_habitats
from radiomicviz.result import ExtractionResult, ExtractionMetadata


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_label_map(shape, n_clusters=3) -> np.ndarray:
    label_map = np.zeros(shape, dtype=np.int32)
    # Fill the central region with random habitat labels
    sl = tuple(slice(s // 4, 3 * s // 4) for s in shape)
    region_shape = tuple(s // 4 * 2 for s in shape)
    label_map[sl] = np.random.randint(1, n_clusters + 1, size=region_shape)
    return label_map


def _make_probabilities(shape, n_clusters=3) -> np.ndarray:
    probs = np.zeros((*shape, n_clusters), dtype=np.float32)
    sl = tuple(slice(s // 4, 3 * s // 4) for s in shape)
    region_shape = tuple(s // 4 * 2 for s in shape)
    n_voxels = int(np.prod(region_shape))
    raw = np.random.dirichlet(np.ones(n_clusters), size=n_voxels)
    probs[sl] = raw.reshape(*region_shape, n_clusters)
    return probs


def _make_cluster_stats(n_clusters=3) -> pd.DataFrame:
    features = ["feat_a", "feat_b"]
    stats = ["mean", "std", "median", "p10", "p90"]
    index = pd.Index(range(1, n_clusters + 1), name="habitat")
    columns = pd.MultiIndex.from_product([features, stats], names=["feature", "stat"])
    data = np.random.randn(n_clusters, len(features) * len(stats))
    return pd.DataFrame(data, index=index, columns=columns)


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Reuse the conftest session-scoped mask (already saved to disk + loaded).
# This ensures nibabel/numpy LAPACK is initialized before any in-process
# NiftiImage construction in these tests.
@pytest.fixture(scope="session")
def mask_nii(synthetic_mask_single):
    """Load the conftest mask as a NiftiImage for HabitatResult construction."""
    return nib.load(str(synthetic_mask_single))


@pytest.fixture
def n_clusters():
    return 3


@pytest.fixture
def habitat_result_gmm(n_clusters, mask_nii):
    """A realistic HabitatResult from a GMM run."""
    np.random.seed(0)
    shape = mask_nii.shape[:3]
    return HabitatResult(
        label_map=_make_label_map(shape, n_clusters=n_clusters),
        probabilities=_make_probabilities(shape, n_clusters=n_clusters),
        n_clusters=n_clusters,
        cluster_stats=_make_cluster_stats(n_clusters=n_clusters),
        selection_criteria={
            2: {"bic": -1200.0, "silhouette": 0.45, "ch": 300.0, "db": 0.8},
            3: {"bic": -1350.0, "silhouette": 0.51, "ch": 320.0, "db": 0.75},
            4: {"bic": -1300.0, "silhouette": 0.48, "ch": 310.0, "db": 0.77},
        },
        selected_k=n_clusters,
        features_used=["feat_a", "feat_b"],
        features_dropped=["feat_c [reason: constant]"],
        pca_variance_explained=0.92,
        metadata={
            "method": "gmm",
            "gmm_covariance_type": "full",
            "random_state": 42,
            "timestamp": "2026-06-05T00:00:00",
        },
        mask_nii=mask_nii,
    )


@pytest.fixture
def habitat_result_kmeans(n_clusters, mask_nii):
    """A HabitatResult from a K-means run (no probabilities)."""
    np.random.seed(1)
    shape = mask_nii.shape[:3]
    return HabitatResult(
        label_map=_make_label_map(shape, n_clusters=n_clusters),
        probabilities=None,
        n_clusters=n_clusters,
        cluster_stats=_make_cluster_stats(n_clusters=n_clusters),
        selection_criteria={},
        selected_k=n_clusters,
        features_used=["feat_a", "feat_b"],
        features_dropped=[],
        pca_variance_explained=None,
        metadata={"method": "kmeans", "random_state": 42},
        mask_nii=mask_nii,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHabitatResult:

    # -- to_nifti() ------------------------------------------------------------

    def test_to_nifti_writes_file(self, tmp_path, habitat_result_gmm):
        out = tmp_path / "habitats.nii.gz"
        returned = habitat_result_gmm.to_nifti(out)
        assert returned == out
        assert out.exists()

    def test_to_nifti_integer_dtype(self, tmp_path, habitat_result_gmm):
        out = tmp_path / "habitats.nii.gz"
        habitat_result_gmm.to_nifti(out)
        nii = nib.load(str(out))
        assert np.issubdtype(nii.get_data_dtype(), np.integer)

    def test_to_nifti_correct_shape(self, tmp_path, habitat_result_gmm):
        out = tmp_path / "habitats.nii.gz"
        habitat_result_gmm.to_nifti(out)
        nii = nib.load(str(out))
        assert nii.shape == habitat_result_gmm.label_map.shape

    def test_to_nifti_creates_parent_dirs(self, tmp_path, habitat_result_gmm):
        out = tmp_path / "subdir" / "habitats.nii.gz"
        habitat_result_gmm.to_nifti(out)
        assert out.exists()

    def test_to_nifti_raises_without_mask_nii(self, habitat_result_gmm):
        habitat_result_gmm.mask_nii = None
        with pytest.raises(ValueError, match="mask_nii"):
            habitat_result_gmm.to_nifti("/tmp/out.nii.gz")

    # -- to_prob_nifti() -------------------------------------------------------

    def test_to_prob_nifti_gmm(self, tmp_path, habitat_result_gmm, n_clusters):
        out = tmp_path / "probs.nii.gz"
        returned = habitat_result_gmm.to_prob_nifti(out)
        assert returned == out
        assert out.exists()
        nii = nib.load(str(out))
        assert nii.ndim == 4
        assert nii.shape[3] == n_clusters

    def test_to_prob_nifti_float_dtype(self, tmp_path, habitat_result_gmm):
        out = tmp_path / "probs.nii.gz"
        habitat_result_gmm.to_prob_nifti(out)
        nii = nib.load(str(out))
        assert np.issubdtype(nii.get_data_dtype(), np.floating)

    def test_to_prob_nifti_raises_for_kmeans(self, tmp_path, habitat_result_kmeans):
        out = tmp_path / "probs.nii.gz"
        with pytest.raises(ValueError, match="GMM"):
            habitat_result_kmeans.to_prob_nifti(out)

    def test_to_prob_nifti_raises_without_mask_nii(self, habitat_result_gmm):
        habitat_result_gmm.mask_nii = None
        with pytest.raises(ValueError, match="mask_nii"):
            habitat_result_gmm.to_prob_nifti("/tmp/probs.nii.gz")

    # -- to_csv() --------------------------------------------------------------

    def test_to_csv_writes_csv(self, tmp_path, habitat_result_gmm):
        out = tmp_path / "stats.csv"
        returned = habitat_result_gmm.to_csv(out)
        assert returned == out
        assert out.exists()

    def test_to_csv_content(self, tmp_path, habitat_result_gmm, n_clusters):
        out = tmp_path / "stats.csv"
        habitat_result_gmm.to_csv(out)
        df = pd.read_csv(out, header=[0, 1], index_col=0)
        assert len(df) == n_clusters

    def test_to_csv_writes_metadata_sidecar(self, tmp_path, habitat_result_gmm):
        out = tmp_path / "stats.csv"
        habitat_result_gmm.to_csv(out)
        sidecar = tmp_path / "stats.metadata.json"
        assert sidecar.exists()

    def test_to_csv_sidecar_is_valid_json(self, tmp_path, habitat_result_gmm):
        out = tmp_path / "stats.csv"
        habitat_result_gmm.to_csv(out)
        sidecar = tmp_path / "stats.metadata.json"
        with open(sidecar, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data.get("method") == "gmm"

    # -- summary() -------------------------------------------------------------

    def test_summary_returns_nonempty_string(self, habitat_result_gmm):
        s = habitat_result_gmm.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_n_clusters(self, habitat_result_gmm, n_clusters):
        s = habitat_result_gmm.summary()
        assert str(n_clusters) in s

    def test_summary_lists_features_used(self, habitat_result_gmm):
        s = habitat_result_gmm.summary()
        for feat in habitat_result_gmm.features_used:
            assert feat in s

    def test_summary_lists_dropped_features(self, habitat_result_gmm):
        s = habitat_result_gmm.summary()
        assert "feat_c" in s

    def test_summary_no_dropped_kmeans(self, habitat_result_kmeans):
        s = habitat_result_kmeans.summary()
        assert isinstance(s, str)

    # -- view() ----------------------------------------------------------------

    def test_view_delegates_to_viewer(self, habitat_result_gmm):
        from unittest.mock import patch
        with patch("radiomicviz.viewer.launch_viewer_from_habitat") as mock_fn:
            habitat_result_gmm.view(port=9999, open_browser=False)
            mock_fn.assert_called_once_with(habitat_result_gmm, port=9999, open_browser=False)

    # -- repr ------------------------------------------------------------------

    def test_repr_contains_method_and_n_clusters(self, habitat_result_gmm, n_clusters):
        r = repr(habitat_result_gmm)
        assert "gmm" in r
        assert str(n_clusters) in r


# ── Step B fixtures ───────────────────────────────────────────────────────────

def _make_feature_maps(shape, mask_arr, seed=42):
    """
    Four synthetic feature maps on a full-volume numpy array.

    feat_a, feat_b, feat_c are independent random Gaussian.
    feat_d is feat_c + tiny noise so Spearman |r| > 0.70 within mask voxels.
    """
    rng = np.random.default_rng(seed)
    n_mask = int(mask_arr.sum())
    maps = {}
    for fname in ("feat_a", "feat_b", "feat_c"):
        arr = np.zeros(shape, dtype=np.float64)
        arr[mask_arr] = rng.standard_normal(n_mask)
        maps[fname] = arr

    # correlated partner: feat_d ≈ feat_c (Spearman r > 0.99)
    arr_d = maps["feat_c"].copy()
    arr_d[mask_arr] += rng.standard_normal(n_mask) * 0.01
    maps["feat_d"] = arr_d
    return {"label1": maps}


@pytest.fixture(scope="module")
def voxelwise_result(synthetic_image, habitat_mask):
    """Synthetic voxelwise ExtractionResult for clustering tests."""
    img_nii = nib.load(str(synthetic_image))
    mask_nii = nib.load(str(habitat_mask))
    shape = img_nii.shape[:3]
    mask_arr = np.asarray(mask_nii.dataobj) > 0

    meta = ExtractionMetadata(
        image_path=str(synthetic_image),
        mask_path=str(habitat_mask),
        config_source="test",
        mode="voxelwise",
    )
    return ExtractionResult(
        features=pd.DataFrame(),
        metadata=meta,
        feature_maps=_make_feature_maps(shape, mask_arr),
        mask_nii=mask_nii,
    )


@pytest.fixture(scope="module")
def nifti_4d_and_mask(tmp_path_factory, voxelwise_result, habitat_mask):
    """Write a 4D NIfTI from the voxelwise_result fixture; return (path, mask_path)."""
    tmp_dir = tmp_path_factory.mktemp("cluster_nifti")
    nifti_path = tmp_dir / "features4d.nii.gz"

    mask_nii = voxelwise_result.mask_nii
    fm = voxelwise_result.feature_maps
    arrays, names = [], []
    for lk in sorted(fm):
        for fn in sorted(fm[lk]):
            arrays.append(fm[lk][fn])
            names.append(f"{lk}_{fn}")

    stack = np.stack(arrays, axis=-1).astype(np.float32)
    nib.save(nib.Nifti1Image(stack, mask_nii.affine), str(nifti_path))

    sidecar = nifti_path.with_suffix("").with_suffix(".features.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({"features": names}, f)

    return nifti_path, Path(habitat_mask)


# ── TestClusterHabitats ───────────────────────────────────────────────────────

class TestClusterHabitats:

    # Use a small k_range to keep tests fast.
    _KR = (2, 3)

    def test_gmm_returns_valid_habitat_result(self, voxelwise_result):
        result = cluster_habitats(
            voxelwise_result, method="gmm", n_clusters=2, k_range=self._KR
        )
        assert isinstance(result, HabitatResult)

    def test_kmeans_returns_valid_habitat_result(self, voxelwise_result):
        result = cluster_habitats(
            voxelwise_result, method="kmeans", n_clusters=2, k_range=self._KR
        )
        assert isinstance(result, HabitatResult)

    def test_label_map_shape_matches_image(self, voxelwise_result):
        result = cluster_habitats(
            voxelwise_result, method="gmm", n_clusters=2, k_range=self._KR
        )
        expected_shape = voxelwise_result.mask_nii.shape[:3]
        assert result.label_map.shape == expected_shape

    def test_label_map_values_in_range(self, voxelwise_result):
        result = cluster_habitats(
            voxelwise_result, method="gmm", n_clusters=2, k_range=self._KR
        )
        unique = set(np.unique(result.label_map).tolist())
        allowed = set(range(0, result.n_clusters + 1))
        assert unique <= allowed

    def test_probabilities_none_for_kmeans(self, voxelwise_result):
        result = cluster_habitats(
            voxelwise_result, method="kmeans", n_clusters=2, k_range=self._KR
        )
        assert result.probabilities is None

    def test_probabilities_shape_for_gmm(self, voxelwise_result):
        k = 2
        result = cluster_habitats(
            voxelwise_result, method="gmm", n_clusters=k, k_range=self._KR
        )
        assert result.probabilities is not None
        expected = (*voxelwise_result.mask_nii.shape[:3], k)
        assert result.probabilities.shape == expected

    def test_auto_n_clusters_returns_int_in_range_gmm(self, voxelwise_result):
        result = cluster_habitats(
            voxelwise_result, method="gmm", n_clusters="auto", k_range=self._KR
        )
        assert isinstance(result.n_clusters, int)
        assert self._KR[0] <= result.n_clusters <= self._KR[1]

    def test_auto_n_clusters_returns_int_in_range_kmeans(self, voxelwise_result):
        result = cluster_habitats(
            voxelwise_result, method="kmeans", n_clusters="auto", k_range=self._KR
        )
        assert isinstance(result.n_clusters, int)
        assert self._KR[0] <= result.n_clusters <= self._KR[1]

    def test_forced_n_clusters_respected(self, voxelwise_result):
        for k in (2, 3):
            result = cluster_habitats(
                voxelwise_result, method="gmm", n_clusters=k, k_range=self._KR
            )
            assert result.n_clusters == k
            assert result.selected_k == k

    def test_features_dropped_populated_for_correlated_pair(self, voxelwise_result):
        # feat_d ≈ feat_c in the fixture — should trigger redundancy elimination.
        result = cluster_habitats(
            voxelwise_result,
            method="gmm",
            n_clusters=2,
            k_range=self._KR,
            redundancy_threshold=0.70,
        )
        assert len(result.features_dropped) > 0
        assert any("correlated" in f for f in result.features_dropped)

    def test_cluster_stats_shape(self, voxelwise_result):
        k = 2
        result = cluster_habitats(
            voxelwise_result, method="gmm", n_clusters=k, k_range=self._KR
        )
        assert len(result.cluster_stats) == k
        assert isinstance(result.cluster_stats.columns, pd.MultiIndex)
        assert result.cluster_stats.columns.names == ["feature", "stat"]
        assert set(result.cluster_stats.columns.get_level_values("stat")) == {
            "mean", "std", "median", "p10", "p90"
        }

    def test_nifti_path_input(self, nifti_4d_and_mask):
        nifti_path, mask_path = nifti_4d_and_mask
        result = cluster_habitats(
            nifti_path,
            mask=mask_path,
            method="gmm",
            n_clusters=2,
            k_range=self._KR,
        )
        assert isinstance(result, HabitatResult)
        mask_nii = nib.load(str(mask_path))
        assert result.label_map.shape == mask_nii.shape[:3]


# ── Step D fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def batch_subjects_csv(tmp_path_factory, nifti_4d_and_mask):
    """CSV with two subjects both using the same synthetic 4D NIfTI."""
    nifti_path, mask_path = nifti_4d_and_mask
    tmp_dir = tmp_path_factory.mktemp("batch_cluster_csv")
    df = pd.DataFrame({
        "subject_id": ["sub01", "sub02"],
        "feature_4d": [str(nifti_path), str(nifti_path)],
        "mask": [str(mask_path), str(mask_path)],
    })
    csv_path = tmp_dir / "subjects.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ── TestBatchCluster ──────────────────────────────────────────────────────────

class TestBatchCluster:

    _KR = (2, 3)

    def test_two_subjects_completes(self, tmp_path, batch_subjects_csv):
        from radiomicviz.cluster import batch_cluster
        manifest = batch_cluster(
            subjects=batch_subjects_csv,
            feature_4d_col="feature_4d",
            mask_col="mask",
            output_dir=tmp_path / "out",
            n_clusters=2,
            k_range=self._KR,
        )
        assert isinstance(manifest, dict)
        assert manifest["succeeded_rows"] == 2
        assert manifest["failed_rows"] == 0

    def test_manifest_has_successes_and_failures_keys(self, tmp_path, batch_subjects_csv):
        from radiomicviz.cluster import batch_cluster
        manifest = batch_cluster(
            subjects=batch_subjects_csv,
            feature_4d_col="feature_4d",
            mask_col="mask",
            output_dir=tmp_path / "out",
            n_clusters=2,
            k_range=self._KR,
        )
        assert "successes" in manifest
        assert "failures" in manifest
        assert isinstance(manifest["successes"], list)
        assert isinstance(manifest["failures"], list)

    def test_output_files_exist_for_each_subject(self, tmp_path, batch_subjects_csv):
        from radiomicviz.cluster import batch_cluster
        out = tmp_path / "out"
        batch_cluster(
            subjects=batch_subjects_csv,
            feature_4d_col="feature_4d",
            mask_col="mask",
            output_dir=out,
            n_clusters=2,
            k_range=self._KR,
        )
        for sub_id in ("sub01", "sub02"):
            assert (out / sub_id / "habitats.nii.gz").exists()
            assert (out / sub_id / "cluster_stats.csv").exists()

    def test_manifest_json_written_to_disk(self, tmp_path, batch_subjects_csv):
        from radiomicviz.cluster import batch_cluster
        out = tmp_path / "out"
        manifest = batch_cluster(
            subjects=batch_subjects_csv,
            feature_4d_col="feature_4d",
            mask_col="mask",
            output_dir=out,
            n_clusters=2,
            k_range=self._KR,
        )
        assert Path(manifest["manifest_path"]).exists()

    def test_n_jobs_2_parallel_path(self, tmp_path, batch_subjects_csv):
        from radiomicviz.cluster import batch_cluster
        manifest = batch_cluster(
            subjects=batch_subjects_csv,
            feature_4d_col="feature_4d",
            mask_col="mask",
            output_dir=tmp_path / "out",
            n_jobs=2,
            n_clusters=2,
            k_range=self._KR,
        )
        assert manifest["succeeded_rows"] == 2

    def test_bad_path_isolated_as_failure(self, tmp_path, nifti_4d_and_mask):
        from radiomicviz.cluster import batch_cluster
        nifti_path, mask_path = nifti_4d_and_mask
        df = pd.DataFrame({
            "subject_id": ["good", "bad"],
            "feature_4d": [str(nifti_path), "/nonexistent/path.nii.gz"],
            "mask": [str(mask_path), str(mask_path)],
        })
        manifest = batch_cluster(
            subjects=df,
            feature_4d_col="feature_4d",
            mask_col="mask",
            output_dir=tmp_path / "out",
            n_clusters=2,
            k_range=self._KR,
        )
        assert manifest["succeeded_rows"] == 1
        assert manifest["failed_rows"] == 1
        assert manifest["failures"][0]["subject_id"] == "bad"
