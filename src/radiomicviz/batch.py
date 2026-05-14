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
    modality: Optional[str] = None,
    subject_id_col: Optional[str] = None,
    label_col: Optional[str] = None,
    n_jobs: int = 1,
    output_dir: Optional[Union[str, Path]] = None,
    skip_validation: bool = False,
    continue_on_error: bool = True,
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

    # -- Prepare output directory ------------------------------------------
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        per_subject_dir = output_dir / "per_subject"
        per_subject_dir.mkdir(exist_ok=True)

    # -- Build job list ----------------------------------------------------
    jobs = []
    for idx, row in df.iterrows():
        sub_id = str(row[id_col])
        sub_label = int(row[label_col]) if label_col and label_col in df.columns else label

        jobs.append({
            "subject_id": sub_id,
            "image": row[image_col],
            "mask": row[mask_col],
            "label": sub_label,
            "row_data": row.to_dict(),
        })

    # -- Execute -----------------------------------------------------------
    t0 = time.time()

    if n_jobs == 1:
        # Sequential — simpler debugging, no joblib overhead
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
        if isinstance(result_or_error, ExtractionResult):
            results[sub_id] = result_or_error
        else:
            failures.append({"subject_id": sub_id, "error": result_or_error})

    # -- Save outputs ------------------------------------------------------
    if output_dir:
        _save_batch_outputs(
            results, failures, output_dir, per_subject_dir,
            subjects_csv, total_time,
        )

    # -- Summary -----------------------------------------------------------
    logger.info(
        "Batch complete: %d/%d succeeded, %d failed (%.1fs total)",
        len(results), len(jobs), len(failures), total_time,
    )
    if failures:
        for f in failures:
            logger.error("  FAILED %s: %s", f["subject_id"], f["error"])

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_id_col(df: pd.DataFrame, user_col: Optional[str]) -> str:
    """Figure out which column to use as subject ID."""
    if user_col and user_col in df.columns:
        return user_col

    # Auto-detect common names
    for candidate in ["subject_id", "Subject", "Patient", "participant_id", "ID", "id"]:
        if candidate in df.columns:
            return candidate

    # Fall back to row index
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
            modality=modality,
            subject_id=sub_id,
            skip_validation=skip_validation,
            retain_mask=False,  # don't hold masks in memory for batch
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
    per_subject_dir: Path,
    subjects_csv: Path,
    total_time: float,
) -> None:
    """Save combined CSV, per-subject CSVs, and manifest."""

    # Per-subject CSVs
    for sub_id, result in results.items():
        safe_id = sub_id.replace("/", "_").replace(" ", "_")
        result.to_csv(per_subject_dir / f"{safe_id}.csv", include_metadata=False)

    # Combined CSV
    if results:
        combined_rows = []
        for sub_id, result in results.items():
            for _, row in result.features.iterrows():
                row_dict = row.to_dict()
                row_dict["subject_id"] = sub_id
                row_dict["label"] = row.name  # the index (label)
                combined_rows.append(row_dict)

        combined_df = pd.DataFrame(combined_rows)
        # Move subject_id and label to front
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
        "total_subjects": len(results) + len(failures),
        "succeeded": len(results),
        "failed": len(failures),
        "total_time_seconds": round(total_time, 2),
        "failed_subjects": failures,
    }
    manifest_path = output_dir / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Manifest saved to %s", manifest_path)
