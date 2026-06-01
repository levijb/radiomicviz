"""
Input validation for radiomics extraction.

Validates NIfTI images and masks before extraction begins, catching common
neuroimaging pitfalls (shape mismatches, empty masks, corrupt headers, etc.)
early with clear, actionable error messages.

Usage:
    >>> from radiomicviz import validate_inputs
    >>> report = validate_inputs("t1.nii.gz", "mask.nii.gz")
    >>> report.ok  # True if all checks passed
    >>> print(report)  # human-readable summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger("radiomicviz.validate")

# ---------------------------------------------------------------------------
# Thresholds / constants
# ---------------------------------------------------------------------------
MIN_VOXELS_FOR_TEXTURE = 10
MAX_ORIENTATION_DIFF_DEG = 1.0  # degrees
ATOL_AFFINE = 1e-3


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
@dataclass
class ValidationIssue:
    """A single validation finding."""

    level: str  # "error", "warning", "info"
    check: str  # short machine-readable label, e.g. "mask_empty"
    message: str  # human-readable explanation

    def __str__(self) -> str:
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}
        return f"  {icon.get(self.level, '?')} [{self.level.upper()}] {self.check}: {self.message}"


@dataclass
class ValidationReport:
    """Collection of validation findings for one image–mask pair."""

    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    issues: list[ValidationIssue] = field(default_factory=list)

    # Convenience accessors ------------------------------------------------
    @property
    def ok(self) -> bool:
        """True if no errors (warnings are acceptable)."""
        return not any(i.level == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def _add(self, level: str, check: str, message: str) -> None:
        issue = ValidationIssue(level=level, check=check, message=message)
        self.issues.append(issue)
        getattr(logger, level if level != "error" else "error")(
            "%s: %s", check, message
        )

    def error(self, check: str, message: str) -> None:
        self._add("error", check, message)

    def warn(self, check: str, message: str) -> None:
        self._add("warning", check, message)

    def info(self, check: str, message: str) -> None:
        self._add("info", check, message)

    def raise_on_errors(self) -> None:
        """Raise ``ValueError`` if any errors exist."""
        if not self.ok:
            msgs = "\n".join(str(e) for e in self.errors)
            raise ValueError(
                f"Validation failed for image={self.image_path}, "
                f"mask={self.mask_path}:\n{msgs}"
            )

    def __str__(self) -> str:
        status = "PASSED" if self.ok else "FAILED"
        header = (
            f"Validation {status} | image={self.image_path} | mask={self.mask_path}"
        )
        if not self.issues:
            return f"{header}\n  ✓ All checks passed"
        body = "\n".join(str(i) for i in self.issues)
        return f"{header}\n{body}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate_inputs(
    image: Union[str, Path],
    mask: Union[str, Path],
    *,
    label: Optional[int] = None,
    min_voxels: int = MIN_VOXELS_FOR_TEXTURE,
) -> ValidationReport:
    """
    Validate an image–mask pair before radiomics extraction.

    Parameters
    ----------
    image : str or Path
        Path to the NIfTI image file (T1, FLAIR, QSM, etc.).
    mask : str or Path
        Path to the NIfTI mask file (binary or multi-label integer).
    label : int, optional
        If provided, only validate this specific label within the mask.
        If None, all nonzero labels are checked.
    min_voxels : int, optional
        Minimum number of voxels per ROI label required for texture features
        to be reliable. Default is 10.

    Returns
    -------
    ValidationReport
        Object with ``.ok`` property (bool), ``.errors``, ``.warnings``,
        and a human-readable ``__str__``. Call ``.raise_on_errors()`` to
        raise ValueError on failure.

    Examples
    --------
    >>> report = validate_inputs("sub01_T1.nii.gz", "sub01_lesions.nii.gz")
    >>> if not report.ok:
    ...     print(report)
    ...     raise SystemExit(1)

    >>> # Or just crash immediately on errors:
    >>> validate_inputs("sub01_T1.nii.gz", "sub01_mask.nii.gz").raise_on_errors()
    """
    image = Path(image)
    mask = Path(mask)
    report = ValidationReport(image_path=str(image), mask_path=str(mask))

    # -- 1. File existence --------------------------------------------------
    if not image.exists():
        report.error("image_not_found", f"Image file not found: {image}")
        return report  # can't continue without the file
    if not mask.exists():
        report.error("mask_not_found", f"Mask file not found: {mask}")
        return report

    # -- 2. Loadability -----------------------------------------------------
    try:
        img_nii = nib.load(str(image))
    except Exception as exc:
        report.error("image_load_failed", f"Cannot load image: {exc}")
        return report

    try:
        mask_nii = nib.load(str(mask))
    except Exception as exc:
        report.error("mask_load_failed", f"Cannot load mask: {exc}")
        return report

    # -- 3. Dimensionality --------------------------------------------------
    img_shape = img_nii.shape
    mask_shape = mask_nii.shape

    if len(img_shape) < 3:
        report.error("image_ndim", f"Image has {len(img_shape)}D shape {img_shape}, expected ≥3D")
    if len(mask_shape) < 3:
        report.error("mask_ndim", f"Mask has {len(mask_shape)}D shape {mask_shape}, expected 3D")
    if len(mask_shape) > 3:
        report.warn("mask_ndim", f"Mask is {len(mask_shape)}D {mask_shape}; only first 3D will be used")

    # -- 4. Shape match -----------------------------------------------------
    if img_shape[:3] != mask_shape[:3]:
        report.error(
            "shape_mismatch",
            f"Image shape {img_shape[:3]} ≠ mask shape {mask_shape[:3]}. "
            f"Resample one to match the other before extraction.",
        )
        return report  # per-voxel checks below require matching shapes

    # -- 5. Affine match ----------------------------------------------------
    img_affine = img_nii.affine
    mask_affine = mask_nii.affine
    if not np.allclose(img_affine, mask_affine, atol=ATOL_AFFINE):
        # Check if it's just a rounding issue vs a real mismatch
        max_diff = np.max(np.abs(img_affine - mask_affine))
        if max_diff < 0.1:
            report.warn(
                "affine_minor_diff",
                f"Affine matrices differ slightly (max diff={max_diff:.6f}). "
                f"Probably fine, but verify alignment visually.",
            )
        else:
            report.error(
                "affine_mismatch",
                f"Affine matrices differ significantly (max diff={max_diff:.4f}). "
                f"Image and mask are likely in different spaces. "
                f"Register/resample the mask to the image space.",
            )

    # -- 6. Load mask data & check dtype ------------------------------------
    try:
        mask_data = np.asarray(mask_nii.dataobj, dtype=np.float64)
    except Exception as exc:
        report.error("mask_read_failed", f"Cannot read mask data: {exc}")
        return report

    # Check for non-integer values
    if not np.allclose(mask_data, np.round(mask_data)):
        report.error(
            "mask_not_integer",
            "Mask contains non-integer values. Expected integer labels "
            "(0=background, 1,2,...=ROIs). If this is a probability map, "
            "threshold it first.",
        )
    mask_data = np.round(mask_data).astype(np.int32)

    # Check for negative values
    if np.any(mask_data < 0):
        report.error(
            "mask_negative_values",
            "Mask contains negative values. All ROI labels must be ≥ 0 "
            "(0 = background).",
        )

    # -- 7. Mask not empty --------------------------------------------------
    unique_labels = np.unique(mask_data)
    nonzero_labels = unique_labels[unique_labels != 0]

    if len(nonzero_labels) == 0:
        report.error("mask_empty", "Mask contains only zeros (no ROI voxels).")
        return report

    report.info(
        "mask_labels",
        f"Found {len(nonzero_labels)} label(s): {nonzero_labels.tolist()}",
    )

    # -- 8. Per-label voxel counts ------------------------------------------
    labels_to_check = [label] if label is not None else nonzero_labels.tolist()

    for lbl in labels_to_check:
        if lbl not in nonzero_labels:
            report.error(
                "label_missing",
                f"Requested label {lbl} not found in mask. "
                f"Available labels: {nonzero_labels.tolist()}",
            )
            continue

        n_voxels = int(np.sum(mask_data == lbl))
        if n_voxels == 0:
            report.error("label_empty", f"Label {lbl} has 0 voxels.")
        elif n_voxels < min_voxels:
            report.warn(
                "label_small_roi",
                f"Label {lbl} has only {n_voxels} voxels (< {min_voxels}). "
                f"Texture features (GLCM, GLRLM, etc.) may be unreliable.",
            )
        else:
            report.info("label_voxels", f"Label {lbl}: {n_voxels} voxels")

    # -- 9. Image data quality ----------------------------------------------
    try:
        # Only load a subset for large images to keep validation fast
        img_data = np.asarray(img_nii.dataobj, dtype=np.float64)
    except Exception as exc:
        report.error("image_read_failed", f"Cannot read image data: {exc}")
        return report

    n_nan = int(np.sum(np.isnan(img_data)))
    n_inf = int(np.sum(np.isinf(img_data)))
    if n_nan > 0:
        report.warn(
            "image_has_nan",
            f"Image contains {n_nan} NaN voxels "
            f"({100 * n_nan / img_data.size:.2f}% of volume).",
        )
    if n_inf > 0:
        report.warn(
            "image_has_inf",
            f"Image contains {n_inf} Inf voxels "
            f"({100 * n_inf / img_data.size:.2f}% of volume).",
        )

    # Check if image has zero variance within the mask
    for lbl in labels_to_check:
        if lbl not in nonzero_labels:
            continue
        roi_vals = img_data[mask_data == lbl]
        roi_vals = roi_vals[np.isfinite(roi_vals)]
        if len(roi_vals) > 0 and np.std(roi_vals) == 0:
            report.warn(
                "zero_variance",
                f"Image has zero variance within label {lbl} "
                f"(all voxels = {roi_vals[0]:.4f}). "
                f"Most radiomic features will be undefined.",
            )

    return report
