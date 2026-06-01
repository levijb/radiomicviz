"""
ROI-level radiomic feature extraction tests.

Each test class documents what it verifies and what real bugs it would catch.
All tests use synthetic NIfTI fixtures (no real data required).

Requires pyradiomics 3.0.1 and radiomicviz installed in dev mode.
"""

import json

import pandas as pd
import pytest

# Skip the entire module if pyradiomics is not installed.
# pytest.importorskip skips at collection time rather than at run time.
pytest.importorskip("radiomics", reason="pyradiomics not installed")


class TestSingleLabelDefault:
    """
    Single-label ROI extraction with mri-default preset.

    Tests the happy path: valid inputs → valid ExtractionResult.
    Would catch: import errors, broken config loading, extractor crash,
    empty DataFrame, wrong n_rois, metadata not populated.
    """

    @pytest.fixture(scope="class")
    def result(self, synthetic_image, synthetic_mask_single):
        from radiomicviz import extract
        return extract(
            synthetic_image,
            synthetic_mask_single,
            preset="mri-default",
            mode="roi",
            modality="T1",
            subject_id="test_sub",
        )

    def test_returns_extraction_result(self, result):
        """extract() returns an ExtractionResult with a populated features DataFrame."""
        from radiomicviz.result import ExtractionResult
        assert isinstance(result, ExtractionResult)
        assert isinstance(result.features, pd.DataFrame)

    def test_feature_count(self, result):
        """mri-default extracts shape + firstorder + 5 texture classes (no ngtdm): 100 features.

        Verified against PyRadiomics 3.0.1: shape(14) + firstorder(18) + glcm(22)
        + glrlm(16) + glszm(16) + gldm(14) = 100.
        """
        assert result.n_features == 100, (
            f"Expected 100 features (mri-default, PyRadiomics 3.0.1), got {result.n_features}"
        )

    def test_n_rois_is_one(self, result):
        """Single-label mask → exactly 1 ROI in the result."""
        assert result.n_rois == 1

    def test_metadata_image_path(self, result, synthetic_image):
        """Metadata records the correct image path."""
        assert str(synthetic_image) in result.metadata.image_path

    def test_metadata_mask_path(self, result, synthetic_mask_single):
        """Metadata records the correct mask path."""
        assert str(synthetic_mask_single) in result.metadata.mask_path

    def test_metadata_preset(self, result):
        """Metadata config_source contains the preset name."""
        assert "mri-default" in result.metadata.config_source

    def test_metadata_pyradiomics_version(self, result):
        """Metadata records the PyRadiomics version string."""
        assert result.metadata.pyradiomics_version is not None
        assert len(result.metadata.pyradiomics_version) > 0

    def test_metadata_extraction_time(self, result):
        """Metadata records a positive extraction time."""
        assert result.metadata.extraction_time_seconds is not None
        assert result.metadata.extraction_time_seconds > 0

    def test_no_nan_values(self, result):
        """No NaN values: fixture has sufficient variance for all feature classes."""
        nan_mask = result.features.isna()
        nan_features = result.features.columns[nan_mask.any()].tolist()
        assert nan_features == [], f"NaN in features: {nan_features}"

    def test_summary_contains_key_info(self, result):
        """summary() returns a non-empty string with feature and ROI counts."""
        s = result.summary()
        assert "Features:" in s
        assert "ROIs:" in s
        assert len(s) > 20


class TestMultiLabelExtraction:
    """
    Multi-label ROI extraction: all labels extracted in one call.

    Would catch: wrong label enumeration, label-skip bugs, wrong n_rois,
    missing labels from the DataFrame index, inconsistent feature counts
    across labels.
    """

    @pytest.fixture(scope="class")
    def result(self, synthetic_image, synthetic_mask_multi):
        from radiomicviz import extract
        return extract(
            synthetic_image,
            synthetic_mask_multi,
            preset="mri-default",
            mode="roi",
        )

    def test_n_rois_is_three(self, result):
        """Multi-label mask with labels 1,2,3 → exactly 3 ROIs."""
        assert result.n_rois == 3

    def test_all_labels_in_index(self, result):
        """All three labels appear as rows in the features DataFrame."""
        assert list(result.features.index) == [1, 2, 3]

    def test_consistent_feature_count(self, result):
        """All ROIs have the same number of feature columns (no ragged DataFrame)."""
        n_cols = len(result.features.columns)
        assert n_cols > 0
        for label_val in result.features.index:
            assert len(result.features.loc[label_val]) == n_cols


class TestSingleLabelSelection:
    """
    label= parameter restricts extraction to one specific label.

    Would catch: label filtering broken, extractor ignoring label param,
    off-by-one errors, wrong label returned.
    """

    def test_label_two_only(self, synthetic_image, synthetic_mask_multi):
        """label=2 on a 3-label mask returns exactly ROI 2, n_rois==1."""
        from radiomicviz import extract
        result = extract(
            synthetic_image,
            synthetic_mask_multi,
            preset="mri-default",
            mode="roi",
            label=2,
        )
        assert result.n_rois == 1
        assert result.features.index[0] == 2


class TestTexturePreset:
    """
    mri-texture preset: texture features only, no shape or first-order.

    Would catch: feature class filtering broken in config, wrong preset loaded,
    shape features sneaking into a texture-only result.
    """

    @pytest.fixture(scope="class")
    def result(self, synthetic_image, synthetic_mask_single):
        from radiomicviz import extract
        return extract(
            synthetic_image,
            synthetic_mask_single,
            preset="mri-texture",
            mode="roi",
        )

    def test_no_shape_features(self, result):
        """mri-texture must not contain any shape_ feature columns."""
        shape_cols = [c for c in result.feature_names if "_shape_" in c]
        assert shape_cols == [], f"Shape features present: {shape_cols}"

    def test_no_firstorder_features(self, result):
        """mri-texture must not contain any firstorder_ feature columns."""
        fo_cols = [c for c in result.feature_names if "_firstorder_" in c]
        assert fo_cols == [], f"First-order features present: {fo_cols}"

    def test_has_glcm_features(self, result):
        """mri-texture result contains GLCM texture features."""
        glcm_cols = [c for c in result.feature_names if "_glcm_" in c]
        assert len(glcm_cols) > 0

    def test_has_glrlm_features(self, result):
        """mri-texture result contains GLRLM run-length features."""
        glrlm_cols = [c for c in result.feature_names if "_glrlm_" in c]
        assert len(glrlm_cols) > 0

    def test_has_ngtdm_features(self, result):
        """mri-texture includes ngtdm (unlike mri-default which omits it)."""
        ngtdm_cols = [c for c in result.feature_names if "_ngtdm_" in c]
        assert len(ngtdm_cols) > 0


class TestFirstorderPreset:
    """
    mri-firstorder preset: shape + first-order only, no texture.

    Would catch: texture features leaking in, preset YAML not respected.
    """

    @pytest.fixture(scope="class")
    def result(self, synthetic_image, synthetic_mask_single):
        from radiomicviz import extract
        return extract(
            synthetic_image,
            synthetic_mask_single,
            preset="mri-firstorder",
            mode="roi",
        )

    def test_feature_count(self, result):
        """mri-firstorder: shape(14) + firstorder(18) = 32 features."""
        assert result.n_features == 32, (
            f"Expected 32 features (mri-firstorder, PyRadiomics 3.0.1), got {result.n_features}"
        )

    def test_no_texture_features(self, result):
        """mri-firstorder must have zero texture feature columns."""
        texture_prefixes = ("_glcm_", "_glrlm_", "_glszm_", "_gldm_", "_ngtdm_")
        texture_cols = [
            c for c in result.feature_names
            if any(p in c for p in texture_prefixes)
        ]
        assert texture_cols == [], f"Texture features present: {texture_cols}"

    def test_has_shape_features(self, result):
        """mri-firstorder result contains 14 shape features."""
        shape_cols = [c for c in result.feature_names if "_shape_" in c]
        assert len(shape_cols) == 14

    def test_has_firstorder_features(self, result):
        """mri-firstorder result contains 18 first-order statistics."""
        fo_cols = [c for c in result.feature_names if "_firstorder_" in c]
        assert len(fo_cols) == 18


class TestCsvRoundTrip:
    """
    CSV export and reload.

    Would catch: index corruption, dtype coercion, column count mismatch,
    metadata sidecar not written.
    """

    def test_roundtrip_column_count(self, synthetic_image, synthetic_mask_single, tmp_path):
        """Export to CSV, reload with pandas — column count matches n_features."""
        from radiomicviz import extract
        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="mri-default",
        )
        csv_path = result.to_csv(tmp_path / "features.csv")

        assert csv_path.exists()
        reloaded = pd.read_csv(csv_path, index_col=0)
        assert len(reloaded.columns) == result.n_features

    def test_metadata_sidecar_written(self, synthetic_image, synthetic_mask_single, tmp_path):
        """to_csv() writes a companion .metadata.json file."""
        from radiomicviz import extract
        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="minimal",
        )
        csv_path = result.to_csv(tmp_path / "out.csv")
        meta_path = tmp_path / "out.metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "image_path" in meta

    def test_no_data_corruption(self, synthetic_image, synthetic_mask_single, tmp_path):
        """Reloaded CSV has the same numeric values (within float precision)."""
        from radiomicviz import extract
        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="mri-firstorder",
        )
        csv_path = result.to_csv(tmp_path / "fp_check.csv")
        reloaded = pd.read_csv(csv_path, index_col=0)
        # Compare a first-order feature that should survive CSV round-trip
        col = result.feature_names[0]
        original_val = result.features.iloc[0][col]
        reloaded_val = reloaded.iloc[0][col]
        assert abs(original_val - reloaded_val) < 1e-6


class TestNiftiChoropleth:
    """
    to_nifti() writes one .nii.gz per feature, non-zero where mask is.

    Would catch: mask not retained, NIfTI not written, wrong voxel values,
    affine not preserved.
    """

    def test_files_created(self, synthetic_image, synthetic_mask_single, tmp_path):
        """to_nifti() creates at least one .nii.gz file per feature."""
        import nibabel as nib
        from radiomicviz import extract
        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="mri-firstorder",
            retain_mask=True,
        )
        paths = result.to_nifti(tmp_path / "nifti_out")
        assert len(paths) == result.n_features
        assert all(p.exists() for p in paths)

    def test_nonzero_in_mask(self, synthetic_image, synthetic_mask_single, tmp_path):
        """Feature values are non-zero where the mask is active."""
        import nibabel as nib
        import numpy as np
        from radiomicviz import extract
        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="minimal",
            retain_mask=True,
        )
        paths = result.to_nifti(tmp_path / "nifti_nonzero")
        mask_nii = nib.load(str(synthetic_mask_single))
        mask_data = np.asarray(mask_nii.dataobj).astype(np.int32)
        # Check first feature map: values inside mask should not all be zero
        feat_nii = nib.load(str(paths[0]))
        feat_data = np.asarray(feat_nii.dataobj)
        inside_mask = feat_data[mask_data == 1]
        assert inside_mask.max() != 0.0, "All feature values inside mask are zero"


class TestFeatureNameStability:
    """
    Feature ordering is deterministic across repeated calls.

    Would catch: dict ordering non-determinism, PyRadiomics version changes,
    randomized feature ordering bugs.
    """

    def test_identical_names_on_repeat(self, synthetic_image, synthetic_mask_single):
        """Same extraction twice → identical feature_names list in same order."""
        from radiomicviz import extract
        r1 = extract(synthetic_image, synthetic_mask_single, preset="mri-firstorder")
        r2 = extract(synthetic_image, synthetic_mask_single, preset="mri-firstorder")
        assert r1.feature_names == r2.feature_names


class TestValidationGuards:
    """
    Input validation raises ValueError before hitting PyRadiomics.

    Would catch: validation layer removed, validation not integrated with
    extract(), wrong exception type raised.
    """

    def test_empty_mask_raises(self, synthetic_image, empty_mask):
        """Empty mask (all zeros) → ValueError with 'Validation failed'."""
        from radiomicviz import extract
        with pytest.raises(ValueError, match="Validation failed"):
            extract(synthetic_image, empty_mask, preset="minimal")

    def test_mismatched_shape_raises(self, mismatched_image, synthetic_mask_single):
        """Image shape different from mask → ValueError from validation."""
        from radiomicviz import extract
        with pytest.raises(ValueError, match="Validation failed"):
            extract(mismatched_image, synthetic_mask_single, preset="minimal")
