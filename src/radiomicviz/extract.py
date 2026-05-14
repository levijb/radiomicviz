"""
Single-subject radiomics extraction.

Wraps PyRadiomics with validation, config resolution, structured output,
and diagnostic capture. Supports both ROI-level and voxelwise extraction.

Usage:
    >>> from radiomicviz import extract
    >>> result = extract("t1.nii.gz", "mask.nii.gz", preset="mri-default")
    >>> result.features.head()
    >>> result.to_csv("features.csv")
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd

from radiomicviz._version import __version__
from radiomicviz.config import config_to_yaml, resolve_config, save_config
from radiomicviz.result import (
    ExtractionMetadata,
    ExtractionResult,
    ROIDiagnostic,
)
from radiomicviz.validate import validate_inputs

logger = logging.getLogger("radiomicviz.extract")

# PyRadiomics diagnostic keys we strip from the feature columns
_DIAG_PREFIX = "diagnostics_"


def extract(
    image: Union[str, Path],
    mask: Union[str, Path],
    *,
    preset: Optional[str] = None,
    config: Optional[Union[str, Path, dict]] = None,
    overrides: Optional[dict[str, Any]] = None,
    mode: str = "roi",
    label: Optional[int] = None,
    modality: Optional[str] = None,
    subject_id: Optional[str] = None,
    skip_validation: bool = False,
    retain_mask: bool = True,
    voxelwise_kernel: int = 1,
) -> ExtractionResult:
    """
    Extract radiomic features from an image–mask pair.

    Parameters
    ----------
    image : str or Path
        Path to the NIfTI image (T1, FLAIR, QSM, etc.).
    mask : str or Path
        Path to the NIfTI ROI mask (binary or multi-label).
    preset : str, optional
        Name of a built-in preset (e.g. ``"mri-default"``).
        Ignored if ``config`` is provided.
    config : str, Path, or dict, optional
        Custom PyRadiomics config (YAML path or parsed dict).
        Takes priority over ``preset``.
    overrides : dict, optional
        Settings merged into the config's ``setting`` section.
        Useful for quick tweaks (e.g. ``{"label": 2, "binWidth": 32}``).
    mode : str
        ``"roi"`` for one feature vector per ROI label, or
        ``"voxelwise"`` for per-voxel feature maps.
    label : int, optional
        Extract features only for this mask label. If None, extracts
        for all nonzero labels.
    modality : str, optional
        Label for the image modality (e.g. ``"T1"``, ``"FLAIR"``).
        Stored in metadata, not used for extraction.
    subject_id : str, optional
        Subject identifier. Stored in metadata.
    skip_validation : bool
        If True, skip input validation (faster, but you're on your own).
    retain_mask : bool
        If True, keep the mask NIfTI in the result for later NIfTI export.
    voxelwise_kernel : int
        Kernel radius for voxelwise extraction (only used if mode="voxelwise").

    Returns
    -------
    ExtractionResult
        Structured result with ``.features``, ``.metadata``,
        ``.diagnostics``, and export methods.

    Raises
    ------
    ValueError
        If validation fails (and ``skip_validation=False``).
    FileNotFoundError
        If image, mask, or config file doesn't exist.

    Examples
    --------
    >>> result = extract("sub01_T1.nii.gz", "sub01_lesions.nii.gz",
    ...                  preset="mri-texture", label=1)
    >>> print(result.summary())
    >>> result.to_csv("sub01_features.csv")

    >>> # With custom config + overrides
    >>> result = extract("sub01_T1.nii.gz", "sub01_mask.nii.gz",
    ...                  config="my_params.yaml",
    ...                  overrides={"binWidth": 32})
    """
    image = Path(image)
    mask = Path(mask)

    # -- Validation --------------------------------------------------------
    if not skip_validation:
        report = validate_inputs(image, mask, label=label)
        report.raise_on_errors()
        if report.warnings:
            for w in report.warnings:
                logger.warning(str(w))

    # -- Config resolution -------------------------------------------------
    resolved_config, config_source = resolve_config(
        preset=preset, config=config, overrides=overrides
    )

    # If a label is specified, inject into config settings
    if label is not None:
        if "setting" not in resolved_config:
            resolved_config["setting"] = {}
        resolved_config["setting"]["label"] = label

    # -- PyRadiomics setup -------------------------------------------------
    import radiomics
    from radiomics.featureextractor import RadiomicsFeatureExtractor

    # Write resolved config to a temp YAML (PyRadiomics wants a file or dict)
    extractor = RadiomicsFeatureExtractor(resolved_config)

    if mode == "voxelwise":
        extractor.settings["voxelBased"] = True
        extractor.settings["kernelRadius"] = voxelwise_kernel

    # -- Determine labels to extract ---------------------------------------
    mask_nii = nib.load(str(mask))
    mask_data = np.asarray(mask_nii.dataobj).astype(np.int32)

    if label is not None:
        labels_to_extract = [label]
    else:
        labels_to_extract = sorted(
            int(v) for v in np.unique(mask_data) if v != 0
        )

    # -- Extract -----------------------------------------------------------
    t0 = time.time()
    all_features = {}
    diagnostics = []

    for lbl in labels_to_extract:
        n_voxels = int(np.sum(mask_data == lbl))
        diag = ROIDiagnostic(label=lbl, n_voxels=n_voxels)

        # Compute bounding box
        coords = np.argwhere(mask_data == lbl)
        if len(coords) > 0:
            bb_min = coords.min(axis=0)
            bb_max = coords.max(axis=0)
            diag.bounding_box = tuple(
                (int(lo), int(hi)) for lo, hi in zip(bb_min, bb_max)
            )

        try:
            # Set label in extractor for this ROI
            extractor.settings["label"] = int(lbl)

            logger.info(
                "Extracting label %d (%d voxels) from %s",
                lbl, n_voxels, image.name,
            )

            result = extractor.execute(str(image), str(mask), label=int(lbl))

            # Separate diagnostic keys from feature keys
            feat_dict = {}
            for key, val in result.items():
                if key.startswith(_DIAG_PREFIX):
                    continue  # skip diagnostics from pyradiomics
                # PyRadiomics returns SimpleITK images for voxelwise
                if mode == "voxelwise":
                    import SimpleITK as sitk
                    if isinstance(val, sitk.Image):
                        feat_dict[key] = sitk.GetArrayFromImage(val)
                    else:
                        feat_dict[key] = val
                else:
                    feat_dict[key] = val

            all_features[lbl] = feat_dict
            diag.extraction_ok = True

        except Exception as exc:
            logger.error("Extraction failed for label %d: %s", lbl, exc)
            diag.extraction_ok = False
            diag.error_message = str(exc)

        diagnostics.append(diag)

    extraction_time = time.time() - t0

    # -- Assemble result ---------------------------------------------------
    if mode == "roi":
        features_df = _build_roi_dataframe(all_features)
        feature_maps = None
    else:
        features_df, feature_maps = _build_voxelwise_result(all_features)

    metadata = ExtractionMetadata(
        image_path=str(image),
        mask_path=str(mask),
        config_source=config_source,
        config=resolved_config,
        mode=mode,
        label=label,
        modality=modality,
        subject_id=subject_id,
        pyradiomics_version=radiomics.__version__,
        radiomicviz_version=__version__,
        extraction_time_seconds=round(extraction_time, 2),
    )

    return ExtractionResult(
        features=features_df,
        metadata=metadata,
        diagnostics=diagnostics,
        feature_maps=feature_maps,
        mask_nii=mask_nii if retain_mask else None,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_roi_dataframe(
    all_features: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    """Build a DataFrame from per-label feature dicts."""
    if not all_features:
        return pd.DataFrame()

    rows = []
    for lbl, feats in all_features.items():
        row = {"label": lbl}
        for key, val in feats.items():
            # Convert numpy scalars to Python types
            if hasattr(val, "item"):
                val = val.item()
            row[key] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("label")

    # Sort columns: shape features first, then first-order, then texture
    df = _sort_feature_columns(df)

    return df


def _build_voxelwise_result(
    all_features: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build summary DataFrame + feature map dict from voxelwise results."""
    if not all_features:
        return pd.DataFrame(), {}

    # For voxelwise, we typically have one label; stack feature maps
    # and compute summary stats
    feature_maps = {}
    summary_rows = []

    for lbl, feats in all_features.items():
        row = {"label": lbl}
        for key, val in feats.items():
            if isinstance(val, np.ndarray):
                feature_maps[f"label{lbl}_{key}"] = val
                finite = val[np.isfinite(val)]
                if len(finite) > 0:
                    row[f"{key}_mean"] = float(np.mean(finite))
                    row[f"{key}_std"] = float(np.std(finite))
                    row[f"{key}_median"] = float(np.median(finite))
            else:
                if hasattr(val, "item"):
                    val = val.item()
                row[key] = val
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows).set_index("label")
    return df, feature_maps


def _sort_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort feature columns by class for readability."""
    order_map = {
        "shape": 0,
        "firstorder": 1,
        "glcm": 2,
        "glrlm": 3,
        "glszm": 4,
        "gldm": 5,
        "ngtdm": 6,
    }

    def _sort_key(col: str) -> tuple[int, str]:
        col_lower = col.lower()
        for class_name, priority in order_map.items():
            if class_name in col_lower:
                return (priority, col)
        return (99, col)

    sorted_cols = sorted(df.columns, key=_sort_key)
    return df[sorted_cols]
