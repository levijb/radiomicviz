"""
Batch extraction tests.

batch_extract() runs extraction over a CSV cohort, saves per-subject CSVs,
a combined features CSV, and a manifest. These tests verify the full batch
pipeline including error isolation and parallel execution.

Requires pyradiomics. Uses the minimal preset for speed.
"""

import json

import pandas as pd
import pytest

pytest.importorskip("radiomics", reason="pyradiomics not installed")


class TestBasicBatch:
    """
    3-subject batch with n_jobs=1, mri-default preset.

    Would catch: batch loop broken, combined CSV not written, manifest missing,
    wrong number of results returned.
    """

    @pytest.fixture(scope="class")
    def batch_result(self, subjects_csv, tmp_path_factory):
        from radiomicviz import batch_extract
        out_dir = tmp_path_factory.mktemp("batch_basic")
        results = batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="mri-default",
            n_jobs=1,
            output_dir=out_dir,
        )
        return results, out_dir

    def test_result_count(self, batch_result):
        """3-subject CSV → 3 results returned (one per CSV row)."""
        results, _ = batch_result
        assert len(results) == 3

    def test_combined_csv_exists(self, batch_result):
        """combined_features.csv is written to output_dir."""
        _, out_dir = batch_result
        assert (out_dir / "combined_features.csv").exists()

    def test_manifest_exists(self, batch_result):
        """batch_manifest.json is written to output_dir."""
        _, out_dir = batch_result
        assert (out_dir / "batch_manifest.json").exists()

    def test_subject_ids_in_results(self, batch_result):
        """Results dict is keyed by subject_id (auto-detected from 'subject_id' column)."""
        results, _ = batch_result
        assert "sub01" in results
        assert "sub02" in results
        assert "sub03" in results


class TestCombinedCsvStructure:
    """
    combined_features.csv correctness.

    Would catch: subject_id column missing, feature columns missing, file not
    loadable by pandas, wrong number of rows.
    """

    @pytest.fixture(scope="class")
    def combined_df(self, subjects_csv, tmp_path_factory):
        from radiomicviz import batch_extract
        out_dir = tmp_path_factory.mktemp("batch_combined")
        batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            n_jobs=1,
            output_dir=out_dir,
        )
        return pd.read_csv(out_dir / "combined_features.csv")

    def test_has_subject_id_column(self, combined_df):
        """combined_features.csv must have a 'subject_id' column."""
        assert "subject_id" in combined_df.columns

    def test_has_label_column(self, combined_df):
        """combined_features.csv must have a 'label' column (ROI label value)."""
        assert "label" in combined_df.columns

    def test_all_subjects_present(self, combined_df):
        """Each subject appears at least once (one row per ROI label)."""
        subjects = set(combined_df["subject_id"])
        assert "sub01" in subjects
        assert "sub02" in subjects
        assert "sub03" in subjects

    def test_has_feature_columns(self, combined_df):
        """Combined CSV has feature columns beyond subject_id and label."""
        feature_cols = [c for c in combined_df.columns if c not in ("subject_id", "label")]
        assert len(feature_cols) > 0

    def test_loadable_without_errors(self, combined_df):
        """combined_features.csv loads cleanly (no parse errors, correct dtypes)."""
        # Subject column should be string, label should be numeric
        assert combined_df["subject_id"].dtype == object  # str
        assert pd.api.types.is_numeric_dtype(combined_df["label"])


class TestPerSubjectOutputs:
    """
    Per-subject subfolder structure under output_dir/subjects/.

    Would catch: per-subject CSVs not written, wrong directory structure,
    subject IDs mangled in folder names.
    """

    @pytest.fixture(scope="class")
    def out_dir(self, subjects_csv, tmp_path_factory):
        from radiomicviz import batch_extract
        out_dir = tmp_path_factory.mktemp("batch_per_subj")
        batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            n_jobs=1,
            output_dir=out_dir,
        )
        return out_dir

    def test_subjects_dir_exists(self, out_dir):
        """output_dir/subjects/ directory is created."""
        assert (out_dir / "subjects").is_dir()

    def test_per_subject_csv_exists(self, out_dir):
        """Each subject gets its own subfolder with a features CSV."""
        subjects_dir = out_dir / "subjects"
        for sub_id in ["sub01", "sub02", "sub03"]:
            sub_dir = subjects_dir / sub_id
            assert sub_dir.is_dir(), f"Missing subfolder: {sub_dir}"
            csv_files = list(sub_dir.glob("*.csv"))
            assert len(csv_files) > 0, f"No CSV in {sub_dir}"


class TestErrorIsolation:
    """
    One bad subject (empty mask) → the other two succeed, failure is recorded.

    Would catch: exception propagation killing the whole batch, bad subject
    silently returning a result, error not logged.
    """

    @pytest.fixture(scope="class")
    def batch_result(self, subjects_csv_three_with_bad, tmp_path_factory):
        from radiomicviz import batch_extract
        out_dir = tmp_path_factory.mktemp("batch_errors")
        results = batch_extract(
            subjects_csv_three_with_bad,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            n_jobs=1,
            output_dir=out_dir,
            continue_on_error=True,
        )
        return results, out_dir

    def test_good_subjects_succeed(self, batch_result):
        """Subjects with valid masks appear in results dict."""
        results, _ = batch_result
        assert "alpha" in results
        assert "gamma" in results

    def test_bad_subject_not_in_results(self, batch_result):
        """Subject with empty mask is NOT in results (it failed)."""
        results, _ = batch_result
        assert "beta" not in results

    def test_errors_log_written(self, batch_result):
        """errors.log is written to output_dir when any subject fails."""
        _, out_dir = batch_result
        assert (out_dir / "errors.log").exists()

    def test_manifest_records_failure_count(self, batch_result):
        """batch_manifest.json records failed_rows > 0."""
        _, out_dir = batch_result
        manifest = json.loads((out_dir / "batch_manifest.json").read_text())
        assert manifest["failed_rows"] == 1
        assert manifest["succeeded_rows"] == 2


class TestManifestContents:
    """
    batch_manifest.json structure and required fields.

    Would catch: missing timestamp, version, preset, counts; wrong types.
    """

    @pytest.fixture(scope="class")
    def manifest(self, subjects_csv, tmp_path_factory):
        from radiomicviz import batch_extract
        out_dir = tmp_path_factory.mktemp("batch_manifest")
        batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            n_jobs=1,
            output_dir=out_dir,
        )
        return json.loads((out_dir / "batch_manifest.json").read_text())

    def test_has_timestamp(self, manifest):
        """Manifest records an ISO timestamp."""
        assert "timestamp" in manifest
        assert len(manifest["timestamp"]) > 0

    def test_has_preset(self, manifest):
        """Manifest records the preset name used."""
        assert "preset" in manifest
        assert manifest["preset"] == "minimal"

    def test_has_counts(self, manifest):
        """Manifest records total_rows, succeeded_rows, failed_rows."""
        assert "total_rows" in manifest
        assert "succeeded_rows" in manifest
        assert "failed_rows" in manifest
        assert manifest["total_rows"] == manifest["succeeded_rows"] + manifest["failed_rows"]

    def test_has_radiomicviz_version(self, manifest):
        """Manifest records the radiomicviz package version."""
        assert "radiomicviz_version" in manifest
        assert manifest["radiomicviz_version"] is not None


class TestParallelBatch:
    """
    n_jobs=2 parallel run produces identical results to n_jobs=1.

    Would catch: joblib serialization errors, race conditions, feature count
    differences between serial and parallel modes.
    """

    def test_parallel_result_count(self, subjects_csv, tmp_path):
        """n_jobs=2 returns same number of results as n_jobs=1."""
        from radiomicviz import batch_extract
        results_serial = batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            n_jobs=1,
            output_dir=tmp_path / "serial",
        )
        results_parallel = batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            n_jobs=2,
            output_dir=tmp_path / "parallel",
        )
        assert len(results_parallel) == len(results_serial)

    def test_parallel_feature_count_matches(self, subjects_csv, tmp_path):
        """n_jobs=2 and n_jobs=1 produce the same feature count for sub01."""
        from radiomicviz import batch_extract
        r1 = batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            n_jobs=1,
            output_dir=tmp_path / "serial2",
        )
        r2 = batch_extract(
            subjects_csv,
            image_col="t1_path",
            mask_col="mask_path",
            preset="minimal",
            n_jobs=2,
            output_dir=tmp_path / "parallel2",
        )
        assert r1["sub01"].n_features == r2["sub01"].n_features
