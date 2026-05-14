"""Tests for radiomicviz.extract and radiomicviz.batch."""

import pytest
import pandas as pd
from pathlib import Path

# These tests require pyradiomics to be installed
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("radiomics", reason="pyradiomics not installed"),
    reason="pyradiomics not installed",
)


class TestExtract:
    def test_basic_roi_extraction(self, synthetic_image, synthetic_mask_single):
        from radiomicviz import extract

        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="minimal",
            mode="roi",
        )
        assert result.n_features > 0
        assert result.n_rois == 1
        assert isinstance(result.features, pd.DataFrame)
        assert result.metadata.mode == "roi"

    def test_multi_label_extraction(self, synthetic_image, synthetic_mask_multi):
        from radiomicviz import extract

        result = extract(
            synthetic_image,
            synthetic_mask_multi,
            preset="minimal",
            mode="roi",
        )
        assert result.n_rois == 3  # labels 1, 2, 3
        assert list(result.features.index) == [1, 2, 3]

    def test_single_label_extraction(self, synthetic_image, synthetic_mask_multi):
        from radiomicviz import extract

        result = extract(
            synthetic_image,
            synthetic_mask_multi,
            preset="minimal",
            mode="roi",
            label=2,
        )
        assert result.n_rois == 1
        assert result.features.index[0] == 2

    def test_csv_export(self, synthetic_image, synthetic_mask_single, tmp_path):
        from radiomicviz import extract

        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="minimal",
        )
        csv_path = result.to_csv(tmp_path / "test_out.csv")
        assert csv_path.exists()
        # Metadata sidecar
        assert (tmp_path / "test_out.metadata.json").exists()
        # Reload and check
        df = pd.read_csv(csv_path, index_col=0)
        assert len(df.columns) == result.n_features

    def test_nifti_export(self, synthetic_image, synthetic_mask_single, tmp_path):
        from radiomicviz import extract

        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="minimal",
            retain_mask=True,
        )
        nifti_dir = tmp_path / "nifti_out"
        paths = result.to_nifti(nifti_dir)
        assert len(paths) > 0
        assert all(p.exists() for p in paths)

    def test_validation_failure_raises(self, synthetic_image, empty_mask):
        from radiomicviz import extract

        with pytest.raises(ValueError, match="Validation failed"):
            extract(synthetic_image, empty_mask, preset="minimal")

    def test_skip_validation(self, mismatched_image, synthetic_mask_single):
        """With skip_validation, extraction may fail at PyRadiomics level."""
        from radiomicviz import extract

        # Should not raise ValueError from validation
        # (but may raise from PyRadiomics due to shape mismatch)
        with pytest.raises(Exception):
            extract(
                mismatched_image,
                synthetic_mask_single,
                preset="minimal",
                skip_validation=True,
            )

    def test_result_summary(self, synthetic_image, synthetic_mask_single):
        from radiomicviz import extract

        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="minimal",
            modality="T1",
            subject_id="test_sub",
        )
        summary = result.summary()
        assert "T1" in summary or "test_sub" in summary
        assert "Features:" in summary

    def test_metadata_populated(self, synthetic_image, synthetic_mask_single):
        from radiomicviz import extract

        result = extract(
            synthetic_image,
            synthetic_mask_single,
            preset="minimal",
            modality="FLAIR",
            subject_id="sub99",
        )
        assert result.metadata.modality == "FLAIR"
        assert result.metadata.subject_id == "sub99"
        assert result.metadata.pyradiomics_version is not None
        assert result.metadata.extraction_time_seconds > 0


class TestBatchExtract:
    def test_basic_batch(self, subjects_csv, tmp_path):
        from radiomicviz import batch_extract

        results = batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            output_dir=tmp_path / "batch_out",
        )
        assert len(results) == 3
        # Check combined CSV was created
        assert (tmp_path / "batch_out" / "combined_features.csv").exists()
        # Check manifest
        assert (tmp_path / "batch_out" / "batch_manifest.json").exists()

    def test_batch_continues_on_error(self, subjects_csv_with_bad, tmp_path):
        from radiomicviz import batch_extract

        results = batch_extract(
            subjects_csv_with_bad,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            output_dir=tmp_path / "batch_partial",
            continue_on_error=True,
        )
        # Should have 1 success (good_sub) and 1 failure (bad_sub)
        assert "good_sub" in results
        assert "bad_sub" not in results
        # Error log should exist
        assert (tmp_path / "batch_partial" / "errors.log").exists()

    def test_batch_subject_id_detection(self, subjects_csv, tmp_path):
        from radiomicviz import batch_extract

        results = batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            output_dir=tmp_path / "batch_ids",
        )
        # Should auto-detect "subject_id" column
        assert "sub01" in results
        assert "sub02" in results
        assert "sub03" in results
