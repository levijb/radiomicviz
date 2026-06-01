"""
mri-habitat preset extraction tests.

mri-habitat is a curated first-order + GLCM + GLRLM subset designed for
habitat clustering workflows (intra-tumor heterogeneity analysis). It uses
binCount=64 and no normalization, unlike mri-default (binCount=32, normalize=true).

Because the bin settings differ, we do not assert habitat feature names are a
subset of mri-default names — the names are the same format (original_*) but
the computation differs, and the task guidance says to skip the subset assertion
when bin settings diverge.
"""

import pandas as pd
import pytest

pytest.importorskip("radiomics", reason="pyradiomics not installed")


class TestHabitatConfig:
    """
    Preset config loading — no file-not-found, no structural errors.

    Would catch: YAML syntax errors, missing required keys, wrong preset name.
    """

    def test_preset_loads(self):
        """load_preset('mri-habitat') returns a dict with featureClass."""
        from radiomicviz.config import load_preset
        cfg = load_preset("mri-habitat")
        assert isinstance(cfg, dict)
        assert "featureClass" in cfg, (
            "mri-habitat YAML is missing 'featureClass' — preset would extract "
            "all feature classes instead of the intended curated subset."
        )

    def test_preset_has_image_type(self):
        """mri-habitat specifies at least one imageType."""
        from radiomicviz.config import load_preset
        cfg = load_preset("mri-habitat")
        assert "imageType" in cfg
        assert len(cfg["imageType"]) > 0

    def test_preset_has_bin_count(self):
        """mri-habitat setting section has binCount for intensity discretization."""
        from radiomicviz.config import load_preset
        cfg = load_preset("mri-habitat")
        assert "setting" in cfg
        assert "binCount" in cfg["setting"]


class TestHabitatEndToEnd:
    """
    End-to-end habitat extraction on the habitat_mask fixture.

    habitat_mask is 10x10x8 = 800 voxels in a high-gradient region.
    This size is large enough that GLCM/GLRLM features compute without NaN.
    """

    @pytest.fixture(scope="class")
    def result(self, synthetic_image, habitat_mask):
        from radiomicviz import extract
        return extract(
            synthetic_image,
            habitat_mask,
            preset="mri-habitat",
            mode="roi",
        )

    def test_extraction_completes(self, result):
        """mri-habitat extraction returns an ExtractionResult without error."""
        from radiomicviz.result import ExtractionResult
        assert isinstance(result, ExtractionResult)

    def test_n_rois_is_one(self, result):
        """habitat_mask is single-label → n_rois == 1."""
        assert result.n_rois == 1

    def test_has_features(self, result):
        """mri-habitat produces exactly 42 features: firstorder(18) + glcm(8) + glrlm(16)."""
        assert result.n_features == 42, (
            f"Expected 42 features (mri-habitat, PyRadiomics 3.0.1), got {result.n_features}"
        )

    def test_no_nan_values(self, result):
        """800-voxel ROI with real variance → no NaN features."""
        df = result.features
        nan_features = df.columns[df.isna().any()].tolist()
        if nan_features:
            nan_vals = df[nan_features].iloc[0].to_dict()
            pytest.fail(
                f"NaN in {len(nan_features)} habitat features: {nan_vals}. "
                "If the fixture is the culprit, increase habitat_mask size."
            )

    def test_mostly_nonzero(self, result):
        """At least 80% of feature values should be non-zero."""
        df = result.features
        total = df.size
        nonzero = (df != 0).sum().sum()
        pct = nonzero / total
        assert pct >= 0.80, f"Only {pct:.1%} of habitat features are non-zero"


class TestHabitatVsDefault:
    """
    Habitat feature count is strictly less than mri-default.

    Would catch: mri-habitat.yaml missing featureClass (extracts everything),
    wrong preset loaded, featureClass restriction not applied.
    """

    @pytest.fixture(scope="class")
    def results(self, synthetic_image, synthetic_mask_single):
        from radiomicviz import extract
        r_def = extract(synthetic_image, synthetic_mask_single, preset="mri-default")
        r_hab = extract(synthetic_image, synthetic_mask_single, preset="mri-habitat")
        return r_def, r_hab

    def test_curated_count(self, results):
        """mri-habitat produces fewer features than mri-default."""
        r_def, r_hab = results
        assert r_hab.n_features < r_def.n_features, (
            f"mri-habitat ({r_hab.n_features}) should have fewer features than "
            f"mri-default ({r_def.n_features}). "
            "Check that mri-habitat.yaml has a featureClass restriction."
        )

    def test_habitat_has_no_shape_features(self, results):
        """mri-habitat omits shape features (not meaningful in clustering / voxelwise)."""
        _, r_hab = results
        shape_cols = [c for c in r_hab.feature_names if "_shape_" in c]
        assert shape_cols == [], f"Shape features in habitat: {shape_cols}"

    def test_habitat_has_firstorder(self, results):
        """mri-habitat includes firstorder features."""
        _, r_hab = results
        fo_cols = [c for c in r_hab.feature_names if "_firstorder_" in c]
        assert len(fo_cols) > 0

    def test_habitat_has_texture(self, results):
        """mri-habitat includes at least GLCM texture features."""
        _, r_hab = results
        glcm_cols = [c for c in r_hab.feature_names if "_glcm_" in c]
        assert len(glcm_cols) > 0


class TestHabitatMultiLabel:
    """
    mri-habitat on multi-label mask → features for all labels.

    Would catch: multi-label loop broken for non-default presets.
    """

    def test_all_labels_extracted(self, synthetic_image, synthetic_mask_multi):
        """mri-habitat on a 3-label mask returns features for labels 1, 2, 3."""
        from radiomicviz import extract
        result = extract(
            synthetic_image,
            synthetic_mask_multi,
            preset="mri-habitat",
            mode="roi",
        )
        assert result.n_rois == 3
        assert list(result.features.index) == [1, 2, 3]


class TestHabitatCsvRoundTrip:
    """
    CSV round-trip fidelity for habitat results.

    Would catch: habitat-specific columns causing CSV corruption.
    """

    def test_roundtrip(self, synthetic_image, habitat_mask, tmp_path):
        """Export habitat features to CSV and reload — column count matches."""
        from radiomicviz import extract
        result = extract(
            synthetic_image,
            habitat_mask,
            preset="mri-habitat",
        )
        csv_path = result.to_csv(tmp_path / "habitat_features.csv")
        reloaded = pd.read_csv(csv_path, index_col=0)
        assert len(reloaded.columns) == result.n_features


class TestHabitatClusteringReady:
    """
    habitat features are finite floats ready for sklearn clustering.

    The sklearn-specific assertion (StandardScaler) runs only when sklearn is
    installed (analysis extras). The finite-float check always runs.
    """

    @pytest.fixture(scope="class")
    def result(self, synthetic_image, habitat_mask):
        from radiomicviz import extract
        return extract(
            synthetic_image,
            habitat_mask,
            preset="mri-habitat",
        )

    def test_no_object_dtype_columns(self, result):
        """All feature columns are numeric, not object dtype."""
        df = result.features
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        assert object_cols == [], f"Object-dtype columns: {object_cols}"

    def test_all_finite_values(self, result):
        """All feature values are finite floats (no NaN, no inf)."""
        import numpy as np
        df = result.features.astype(float)
        assert df.isna().sum().sum() == 0
        assert np.isfinite(df.values).all(), "Non-finite values in habitat features"

    def test_sklearn_scaler_succeeds(self, result):
        """StandardScaler.fit_transform succeeds on habitat features (skips if sklearn absent)."""
        StandardScaler = pytest.importorskip(
            "sklearn.preprocessing", reason="scikit-learn not installed (add analysis extras)"
        ).StandardScaler
        df = result.features.astype(float)
        scaled = StandardScaler().fit_transform(df)
        assert scaled.shape == df.shape
