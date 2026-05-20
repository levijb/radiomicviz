"""
Batch extraction across multiple subjects.

Handles parallel processing with joblib, per-subject error isolation,
combined output CSV, and a manifest log. Designed to replace hand-written
for-loops and your cluster_batchprocessing_parallel.py script.

Usage:
    >>> from radiomicviz import batch_extract
    >>> results = batch_extract(
    ...     subjects_csv="cohort.csv",
    ...     image_col="t1_path",
    ...     mask_col="mask_path",
    ...     preset="mri-default",
    ...     n_jobs=4,
    ...     output_dir="./radiomics_output/",
    ... )
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from radiomicviz._version import __version__
from radiomicviz.extract import extract
from radiomicviz.result import ExtractionResult

logger = logging.getLogger("radiomicviz.batch")


def batch_extract(
    subjects_csv: Union[str, Path],
    image_col: str,
    mask_col: str,
    *,
    preset: Optional[str] = None,
    config: Optional[Union[str, Path, dict]] = None,
    overrides: Optional[dict[str, Any]] = None,
    mode: str = "roi",
    label: Optional[int] = None,
    roi_name: Optional[str] = None,
    roi_name_col: Optional[str] = None,
    modality: Optional[str] = None,
    subject_id_col: Optional[str] = None,
    label_col: Optional[str] = None,
    n_jobs: int = 1,
    output_dir: Optional[Union[str, Path]] = None,
    skip_validation: bool = False,
    continue_on_error: bool = True,
    save_maps: bool = False,
) -> dict[str, ExtractionResult]:
    """
    Extract radiomics features from a cohort of subjects.

    Parameters
    ----------
    subjects_csv : str or Path
        CSV file with one row per subject. Must contain columns for image
        and mask file paths at minimum.
    image_col : str
        Column name containing image file paths.
    mask_col : str
        Column name containing mask file paths.
    preset : str, optional
        Built-in preset name.
    config : str, Path, or dict, optional
        Custom PyRadiomics config (overrides preset).
    overrides : dict, optional
        Settings merged into config's ``setting`` section.
    mode : str
        ``"roi"`` or ``"voxelwise"``.
    label : int, optional
        Label to extract (overrides per-subject label_col).
    roi_name : str, optional
        Meaningful name for the ROI used as the voxelwise output folder name
        (e.g. ``"Left_whole_thalamus"``). Applied to every subject when set.
        Overridden per-subject by ``roi_name_col`` values.
    roi_name_col : str, optional
        Column in the CSV with per-subject ROI names. Takes priority over
        ``roi_name``. Falls back to ``"label{N}"`` when neither is provided.
    modality : str, optional
        Modality label for metadata.
    subject_id_col : str, optional
        Column containing subject IDs. If None, uses row index or a
        ``"subject_id"`` / ``"Patient"`` column if present.
    label_col : str, optional
        Column containing per-subject label values (e.g. if different
        subjects have different ROI labels to extract).
    n_jobs : int
        Number of parallel workers. Default 1 (sequential).
    output_dir : str or Path, optional
        If provided, saves per-subject CSVs, combined CSV, and manifest.
    skip_validation : bool
        Skip per-subject input validation.
    continue_on_error : bool
        If True (default), log errors and continue. If False, raise on
        first failure.

    Returns
    -------
    dict[str, ExtractionResult]
        Mapping of subject_id → ExtractionResult for successful extractions.

    Examples
    --------
    >>> results = batch_extract(
    ...     "cohort.csv",
    ...     image_col="t1_path",
    ...     mask_col="mask_path",
    ...     preset="mri-texture",
    ...     n_jobs=4,
    ...     output_dir="./output/",
    ... )
    >>> print(f"Extracted {len(results)} subjects")
    """
    subjects_csv = Path(subjects_csv)
    if not subjects_csv.exists():
        raise FileNotFoundError(f"Subjects CSV not found: {subjects_csv}")

    # -- Load subject table ------------------------------------------------
    df = pd.read_csv(subjects_csv)
    logger.info("Loaded %d subjects from %s", len(df), subjects_csv)

    # Validate required columns
    for col_name, col_val in [("image", image_col), ("mask", mask_col)]:
        if col_val not in df.columns:
            raise ValueError(
                f"Column '{col_val}' not found in {subjects_csv}. "
                f"Available columns: {list(df.columns)}"
            )

    # Resolve subject ID column
    id_col = _resolve_id_col(df, subject_id_col)
    logger.info("Using '%s' as subject ID column", id_col)

    # Auto-detect roi_name column from common names if not explicitly provided
    if roi_name_col is None:
        for candidate in ["mask_name", "roi_name"]:
            if candidate in df.columns:
                roi_name_col = candidate
                logger.info("Auto-detected roi_name_col='%s' from CSV columns", roi_name_col)
                break

    # -- Prepare output directory ------------------------------------------
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subjects_dir = output_dir / "subjects"
        subjects_dir.mkdir(exist_ok=True)

    # -- Build job list ----------------------------------------------------
    jobs = []
    for idx, row in df.iterrows():
        sub_id = str(row[id_col])
        sub_label = int(row[label_col]) if label_col and label_col in df.columns else label

        if roi_name_col and roi_name_col in df.columns and pd.notna(row[roi_name_col]):
            sub_roi_name = str(row[roi_name_col])
        else:
            sub_roi_name = roi_name

        # Unique key per CSV row so multiple ROIs per subject don't overwrite each other.
        # Falls back to sub_id alone when there's no roi_name (one row per subject).
        job_key = f"{sub_id}__{sub_roi_name}" if sub_roi_name else sub_id

        jobs.append({
            "job_key": job_key,
            "subject_id": sub_id,
            "image": row[image_col],
            "mask": row[mask_col],
            "label": sub_label,
            "roi_name": sub_roi_name,
            "row_data": row.to_dict(),
        })

    # -- Execute -----------------------------------------------------------
    t0 = time.time()

    if n_jobs == 1:
        raw_results = []
        for job in jobs:
            raw_results.append(
                _extract_one(
                    job, preset=preset, config=config, overrides=overrides,
                    mode=mode, modality=modality,
                    skip_validation=skip_validation,
                    continue_on_error=continue_on_error,
                )
            )
    else:
        from joblib import Parallel, delayed
        raw_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_extract_one)(
                job, preset=preset, config=config, overrides=overrides,
                mode=mode, modality=modality,
                skip_validation=skip_validation,
                continue_on_error=continue_on_error,
            )
            for job in jobs
        )

    total_time = time.time() - t0

    # -- Collect results ---------------------------------------------------
    results: dict[str, ExtractionResult] = {}
    failures: list[dict[str, str]] = []

    for job, (sub_id, result_or_error) in zip(jobs, raw_results):
        job_key = job["job_key"]
        if isinstance(result_or_error, ExtractionResult):
            results[job_key] = result_or_error
        else:
            failures.append({"subject_id": sub_id, "job_key": job_key, "error": result_or_error})

    # -- Save outputs ------------------------------------------------------
    if output_dir:
        _save_batch_outputs(
            results, failures, output_dir, subjects_dir,
            subjects_csv, total_time, save_maps=save_maps
        )

    # -- Summary -----------------------------------------------------------
    n_unique_subjects = len({r.metadata.subject_id for r in results.values()})
    logger.info(
        "Batch complete: %d/%d rows succeeded (%d unique subjects), %d failed (%.1fs total)",
        len(results), len(jobs), n_unique_subjects, len(failures), total_time,
    )
    if failures:
        for f in failures:
            logger.error("  FAILED %s [%s]: %s", f["subject_id"], f["job_key"], f["error"])

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_id_col(df: pd.DataFrame, user_col: Optional[str]) -> str:
    """Figure out which column to use as subject ID."""
    if user_col and user_col in df.columns:
        return user_col

    for candidate in ["subject_id", "Subject", "Patient", "participant_id", "ID", "id"]:
        if candidate in df.columns:
            return candidate

    df["_row_index"] = [f"sub_{i:04d}" for i in range(len(df))]
    return "_row_index"


def _extract_one(
    job: dict,
    *,
    preset: Optional[str],
    config: Optional[Union[str, Path, dict]],
    overrides: Optional[dict],
    mode: str,
    modality: Optional[str],
    skip_validation: bool,
    continue_on_error: bool,
) -> tuple[str, Union[ExtractionResult, str]]:
    """Extract one subject. Returns (subject_id, result_or_error_string)."""
    sub_id = job["subject_id"]

    try:
        result = extract(
            image=job["image"],
            mask=job["mask"],
            preset=preset,
            config=config,
            overrides=overrides,
            mode=mode,
            label=job.get("label"),
            roi_name=job.get("roi_name"),
            modality=modality,
            subject_id=sub_id,
            skip_validation=skip_validation,
            retain_mask=(mode == "voxelwise"),  # needed for 4D NIfTI export
        )
        return (sub_id, result)

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Subject %s failed: %s", sub_id, error_msg)
        if not continue_on_error:
            raise
        return (sub_id, error_msg)


def _save_batch_outputs(
    results: dict[str, ExtractionResult],
    failures: list[dict[str, str]],
    output_dir: Path,
    subjects_dir: Path,
    subjects_csv: Path,
    total_time: float,
    save_maps: bool = False,
) -> None:
    """Save combined CSV, per-subject CSVs, and manifest."""

    # Per-subject outputs
    # results keys are job_keys (e.g. "sub001__AF_L"); use metadata for the directory name.
    for job_key, result in results.items():
        actual_sub_id = result.metadata.subject_id or job_key
        safe_id = actual_sub_id.replace("/", "_").replace(" ", "_")
        sub_dir = subjects_dir / safe_id
        sub_dir.mkdir(exist_ok=True)

        if result.metadata.mode == "voxelwise":
            # Save one .nrrd per feature: sub_dir/<roi_name>/feature_name.nrrd
            if result.feature_maps:
                import SimpleITK as sitk
                n_saved = 0
                for label_key in sorted(result.feature_maps):
                    label_dir = sub_dir / label_key
                    label_dir.mkdir(exist_ok=True)
                    for feat_name, img in sorted(result.feature_maps[label_key].items()):
                        sitk.WriteImage(img, str(label_dir / f"{feat_name}.nrrd"))
                        n_saved += 1
                logger.info("Saved %d .nrrd maps for %s", n_saved, job_key)
        else:
            result.to_csv(sub_dir / f"{safe_id}_roi_features.csv", include_metadata=False)

    # Combined CSV — ROI mode only; voxelwise outputs .nrrd files per subject
    roi_results = {k: r for k, r in results.items() if r.metadata.mode == "roi"}
    if roi_results:
        combined_rows = []
        for job_key, result in roi_results.items():
            actual_sub_id = result.metadata.subject_id or job_key
            for _, row in result.features.iterrows():
                row_dict = row.to_dict()
                row_dict["subject_id"] = actual_sub_id
                row_dict["label"] = row.name
                combined_rows.append(row_dict)

        combined_df = pd.DataFrame(combined_rows)
        cols = ["subject_id", "label"] + [
            c for c in combined_df.columns if c not in ("subject_id", "label")
        ]
        combined_df = combined_df[cols]
        combined_path = output_dir / "combined_features.csv"
        combined_df.to_csv(combined_path, index=False)
        logger.info("Combined features saved to %s", combined_path)

    # Error log
    if failures:
        errors_path = output_dir / "errors.log"
        with open(errors_path, "w") as f:
            for fail in failures:
                f.write(f"{fail['subject_id']}\t{fail['error']}\n")
        logger.info("Error log saved to %s", errors_path)

    # Manifest
    manifest = {
        "radiomicviz_version": __version__,
        "timestamp": datetime.now().isoformat(),
        "subjects_csv": str(subjects_csv),
        "total_rows": len(results) + len(failures),
        "succeeded_rows": len(results),
        "failed_rows": len(failures),
        "total_time_seconds": round(total_time, 2),
        "failed_subjects": failures,
    }
    manifest_path = output_dir / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Manifest saved to %s", manifest_path)