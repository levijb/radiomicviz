"""Tests for radiomicviz.validate."""

import pytest
from radiomicviz.validate import validate_inputs


class TestFileChecks:
    def test_image_not_found(self, synthetic_mask_single):
        report = validate_inputs("/nonexistent/image.nii.gz", synthetic_mask_single)
        assert not report.ok
        assert any(i.check == "image_not_found" for i in report.errors)

    def test_mask_not_found(self, synthetic_image):
        report = validate_inputs(synthetic_image, "/nonexistent/mask.nii.gz")
        assert not report.ok
        assert any(i.check == "mask_not_found" for i in report.errors)


class TestShapeAndAffine:
    def test_valid_pair_passes(self, synthetic_image, synthetic_mask_single):
        report = validate_inputs(synthetic_image, synthetic_mask_single)
        assert report.ok

    def test_shape_mismatch(self, mismatched_image, synthetic_mask_single):
        report = validate_inputs(mismatched_image, synthetic_mask_single)
        assert not report.ok
        assert any(i.check == "shape_mismatch" for i in report.errors)


class TestMaskChecks:
    def test_empty_mask(self, synthetic_image, empty_mask):
        report = validate_inputs(synthetic_image, empty_mask)
        assert not report.ok
        assert any(i.check == "mask_empty" for i in report.errors)

    def test_float_mask(self, synthetic_image, float_mask):
        report = validate_inputs(synthetic_image, float_mask)
        assert not report.ok
        assert any(i.check == "mask_not_integer" for i in report.errors)

    def test_multi_label_detected(self, synthetic_image, synthetic_mask_multi):
        report = validate_inputs(synthetic_image, synthetic_mask_multi)
        assert report.ok
        info_msgs = [i for i in report.issues if i.check == "mask_labels"]
        assert len(info_msgs) == 1
        assert "3" in info_msgs[0].message  # should mention 3 labels

    def test_tiny_roi_warns(self, synthetic_image, tiny_mask):
        report = validate_inputs(synthetic_image, tiny_mask)
        # Should pass (not error) but warn
        assert report.ok
        assert any(i.check == "label_small_roi" for i in report.warnings)

    def test_missing_label(self, synthetic_image, synthetic_mask_single):
        report = validate_inputs(synthetic_image, synthetic_mask_single, label=99)
        assert not report.ok
        assert any(i.check == "label_missing" for i in report.errors)


class TestImageQuality:
    def test_nan_image_warns(self, nan_image, synthetic_mask_single):
        report = validate_inputs(nan_image, synthetic_mask_single)
        # NaN outside the mask may not trigger, but our fixture has NaN in the volume
        assert any(i.check == "image_has_nan" for i in report.warnings)


class TestReport:
    def test_raise_on_errors(self, synthetic_image, empty_mask):
        report = validate_inputs(synthetic_image, empty_mask)
        with pytest.raises(ValueError, match="Validation failed"):
            report.raise_on_errors()

    def test_str_output(self, synthetic_image, synthetic_mask_single):
        report = validate_inputs(synthetic_image, synthetic_mask_single)
        s = str(report)
        assert "PASSED" in s

    def test_str_output_failed(self, synthetic_image, empty_mask):
        report = validate_inputs(synthetic_image, empty_mask)
        s = str(report)
        assert "FAILED" in s
