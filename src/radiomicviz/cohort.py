"""Generate a cohort CSV compatible with RadiomicViz batch_extract.

Usage:
    radiomicviz generate-csv \\
        --study-folder /path/to/study \\
        --output-csv-name cohort \\
        --image-suffix _T1_lesion_filled_combined_mask_bet_n4_nu.nii.gz

Then run extraction with:
    radiomicviz batch-extract \\
        --subjects cohort.csv \\
        --image-col Image \\
        --mask-col Mask \\
        --preset mri-default \\
        --n-jobs 8 \\
        --output-dir ./radiomics_output/
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_COLUMNS = ["subject_id", "session", "mask_name", "Image", "Mask"]


def generate_cohort_csv(
    study_folder: Union[str, Path],
    output_csv_name: str,
    image_suffix: str,
    image_subdir: Union[str, list[str]] = "T1",
    mask_dir: str = "derivatives/segmentation",
    mask_suffix: str = ".nii.gz",
    exclude_patterns: Union[list[str], None] = None,
    recursive: bool = False,
    dry_run: bool = False,
    output_dir: Union[str, Path, None] = None,
) -> dict:
    """Generate a cohort CSV from a BIDS-like study folder.

    Traverses ``study_folder/Subjects/{subject}/{session}/`` and builds one
    CSV row per mask file found, paired with the matching image file.

    Parameters
    ----------
    study_folder : str or Path
        Root study folder containing a ``Subjects/`` directory.
    output_csv_name : str
        Name of the output CSV file (without ``.csv`` extension).
    image_suffix : str
        Filename suffix to glob for image files
        (e.g. ``_T1_lesion_filled_combined_mask_bet_n4_nu.nii.gz``).
    image_subdir : str or list[str], optional
        Subdirectory or subdirectories under each session to search for images.
        Accepts a comma-separated string or a list. Searched in order; first
        match wins. Default: ``"T1"``.
    mask_dir : str, optional
        Relative path from session directory to mask folder.
        Default: ``"derivatives/segmentation"``.
    mask_suffix : str, optional
        Suffix filter for mask files. Default: ``".nii.gz"``.
    exclude_patterns : list[str] or None, optional
        Substrings in image filenames to skip. Default: ``["mask", "seg"]``.
    recursive : bool, optional
        If ``True``, search ``image_subdir`` recursively. Default: ``False``.
    dry_run : bool, optional
        Log what would be written without creating the CSV. Default: ``False``.
    output_dir : str, Path, or None, optional
        Directory to write the CSV. Defaults to ``study_folder`` if ``None``.

    Returns
    -------
    dict
        ``{"csv_path": Path, "n_subjects": int, "n_sessions": int,
        "n_rows": int, "warnings": list[str]}``

    Raises
    ------
    ValueError
        If the Subjects directory is not found or no rows are generated.
    """
    study_folder = Path(study_folder)
    output_dir = Path(output_dir) if output_dir is not None else study_folder

    if exclude_patterns is None:
        exclude_patterns = ["mask", "seg"]

    if isinstance(image_subdir, str):
        image_subdir = [s.strip() for s in image_subdir.split(",")]

    subjects_path = study_folder / "Subjects"
    if not subjects_path.is_dir():
        raise ValueError(f"Subjects directory not found at {subjects_path}")

    csv_rows: list[list[str]] = []
    warnings: list[str] = []

    for subject_dir in sorted(subjects_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name

        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name

            mask_path_dir = session_dir / mask_dir
            if not mask_path_dir.is_dir():
                msg = f"Mask directory not found: {subject}/{session}/{mask_dir}"
                logger.warning(msg)
                warnings.append(msg)
                continue

            mask_files = sorted(
                p for p in mask_path_dir.iterdir()
                if p.is_file() and p.name.endswith(mask_suffix)
            )
            if not mask_files:
                msg = f"No mask files found for {subject}/{session} in {mask_path_dir}"
                logger.warning(msg)
                warnings.append(msg)
                continue

            image_path: Union[Path, None] = None
            for subdir in image_subdir:
                search_root = session_dir / subdir
                if not search_root.is_dir():
                    continue
                glob_pattern = f"**/*{image_suffix}" if recursive else f"*{image_suffix}"
                candidates = [
                    p for p in search_root.glob(glob_pattern)
                    if p.is_file()
                    and not any(pat.lower() in p.name.lower() for pat in exclude_patterns)
                ]
                if candidates:
                    image_path = candidates[0]
                    break

            if image_path is None:
                msg = f"No image found for {subject}/{session} (suffix: {image_suffix!r})"
                logger.warning(msg)
                warnings.append(msg)
                continue

            for mask_file in mask_files:
                mask_name = mask_file.name
                if mask_name.endswith(".nii.gz"):
                    mask_name = mask_name[:-7]
                elif mask_name.endswith(".nii"):
                    mask_name = mask_name[:-4]
                logger.info("Found: %s/%s — %s", subject, session, mask_file.name)
                csv_rows.append([subject, session, mask_name, str(image_path), str(mask_file)])

    if not csv_rows:
        raise ValueError(
            "No rows generated. Check --image-suffix and --image-subdir arguments."
        )

    n_subjects = len({row[0] for row in csv_rows})
    n_sessions = len({(row[0], row[1]) for row in csv_rows})
    n_rows = len(csv_rows)
    csv_path = output_dir / (output_csv_name + ".csv")

    if dry_run:
        logger.info("Dry run — no file written.")
        logger.info("Would write: %s", csv_path)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_COLUMNS)
            writer.writerows(csv_rows)
        logger.info("CSV written: %s", csv_path)

    logger.info(
        "Subjects: %d, Sessions: %d, Rows: %d, Warnings: %d",
        n_subjects, n_sessions, n_rows, len(warnings),
    )

    return {
        "csv_path": csv_path,
        "n_subjects": n_subjects,
        "n_sessions": n_sessions,
        "n_rows": n_rows,
        "warnings": warnings,
    }
