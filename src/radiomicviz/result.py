"""
ExtractionResult: standardized output from radiomics extraction.

Provides a consistent interface regardless of whether extraction was
ROI-level or voxelwise, with export methods for CSV, NIfTI, and
direct handoff to the viewer.

Usage:
    >>> result = extract("t1.nii.gz", "mask.nii.gz")
    >>> result.features          # pd.DataFrame
    >>> result.diagnostics       # per-ROI diagnostic info
    >>> result.to_csv("out.csv")
    >>> result.to_nifti("out/")  # ROI-level features back to brain space
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger("radiomicviz.result")


@dataclass
class ExtractionMetadata:
    """Provenance information for an extraction run."""

    image_path: str
    mask_path: str
    config_source: str  # e.g. "preset: mri-default"
    config: dict[str, Any] = field(default_factory=dict)
    mode: str = "roi"  # "roi" or "voxelwise"
    label: Optional[int] = None
    modality: Optional[str] = None
    subject_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pyradiomics_version: Optional[str] = None
    radiomicviz_version: Optional[str] = None
    extraction_time_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ROIDiagnostic:
    """Per-ROI diagnostic information."""

    label: int
    n_voxels: int
    bounding_box: Optional[tuple] = None  # ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    warnings: list[str] = field(default_factory=list)
    extraction_ok: bool = True
    error_message: Optional[str] = None


@dataclass
class ExtractionResult:
    """
    Standardized output from a single-subject radiomics extraction.

    Attributes
    ----------
    features : pd.DataFrame
        Extracted features. For ROI mode: rows = ROI labels, columns = features.
        For voxelwise mode: this is a summary (mean across voxels per feature).
    metadata : ExtractionMetadata
        Full provenance information.
    diagnostics : list of ROIDiagnostic
        Per-ROI extraction diagnostics (voxel counts, warnings).
    feature_maps : dict, optional
        For voxelwise mode only: ``{feature_name: np.ndarray}`` of 3D arrays.
    mask_nii : nibabel.Nifti1Image, optional
        The mask used, retained for NIfTI export.
    """

    features: pd.DataFrame
    metadata: ExtractionMetadata
    diagnostics: list[ROIDiagnostic] = field(default_factory=list)
    feature_maps: Optional[dict[str, np.ndarray]] = None
    mask_nii: Optional[nib.Nifti1Image] = None

    # -- Properties --------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """List of all extracted feature names."""
        return list(self.features.columns)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_rois(self) -> int:
        return len(self.features)

    @property
    def is_voxelwise(self) -> bool:
        return self.metadata.mode == "voxelwise"

    @property
    def diagnostics_df(self) -> pd.DataFrame:
        """Diagnostics as a DataFrame."""
        if not self.diagnostics:
            return pd.DataFrame()
        records = []
        for d in self.diagnostics:
            records.append({
                "label": d.label,
                "n_voxels": d.n_voxels,
                "extraction_ok": d.extraction_ok,
                "warnings": "; ".join(d.warnings) if d.warnings else "",
                "error": d.error_message or "",
            })
        return pd.DataFrame(records)

    # -- Export methods ----------------------------------------------------

    def to_csv(self, path: Union[str, Path], include_metadata: bool = True) -> Path:
        """
        Export features to CSV.

        Parameters
        ----------
        path : str or Path
            Output CSV path.
        include_metadata : bool
            If True, writes a companion ``_metadata.json`` alongside the CSV.

        Returns
        -------
        Path
            The CSV path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.features.to_csv(path, index=True)
        logger.info("Features saved to %s (%d features × %d ROIs)",
                     path, self.n_features, self.n_rois)

        if include_metadata:
            meta_path = path.with_suffix(".metadata.json")
            with open(meta_path, "w") as f:
                json.dump(self.metadata.to_dict(), f, indent=2, default=str)
            logger.info("Metadata saved to %s", meta_path)

        return path

    def to_nifti(
        self,
        output_dir: Union[str, Path],
        features: Optional[list[str]] = None,
    ) -> list[Path]:
        """
        Write ROI-level features back as NIfTI volumes.

        Each feature becomes a 3D volume where each ROI is filled with that
        feature's value — like a choropleth on the brain. Useful for
        visualizing features in standard neuroimaging viewers.

        Parameters
        ----------
        output_dir : str or Path
            Directory for output NIfTI files.
        features : list of str, optional
            Subset of features to export. Default: all.

        Returns
        -------
        list of Path
            Paths to the created NIfTI files.

        Raises
        ------
        ValueError
            If mask_nii was not retained during extraction.
        """
        if self.mask_nii is None:
            raise ValueError(
                "Cannot export to NIfTI: mask was not retained. "
                "Re-run extraction with retain_mask=True."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_data = np.asarray(self.mask_nii.dataobj).astype(np.int32)
        affine = self.mask_nii.affine
        header = self.mask_nii.header

        features_to_export = features or self.feature_names
        paths = []

        for feat_name in features_to_export:
            if feat_name not in self.features.columns:
                logger.warning("Feature '%s' not in results, skipping", feat_name)
                continue

            vol = np.zeros_like(mask_data, dtype=np.float64)
            for label_val in self.features.index:
                val = self.features.loc[label_val, feat_name]
                if pd.notna(val):
                    vol[mask_data == label_val] = float(val)

            # Sanitize filename
            safe_name = feat_name.replace("/", "_").replace(" ", "_")
            out_path = output_dir / f"{safe_name}.nii.gz"
            nib.save(
                nib.Nifti1Image(vol, affine, header),
                str(out_path),
            )
            paths.append(out_path)

        logger.info("Exported %d feature maps to %s", len(paths), output_dir)
        return paths

    def to_4d_nifti(self, path: Union[str, Path]) -> Path:
        """
        Stack all feature maps into a single 4D NIfTI + sidecar JSON.

        Only available for voxelwise results.

        Parameters
        ----------
        path : str or Path
            Output path for the 4D NIfTI.

        Returns
        -------
        Path
            Path to the 4D NIfTI file.
        """
        if not self.is_voxelwise or self.feature_maps is None:
            raise ValueError("to_4d_nifti() is only available for voxelwise results.")

        if self.mask_nii is None:
            raise ValueError("mask_nii required for NIfTI export.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        feat_names = sorted(self.feature_maps.keys())
        stack = np.stack([self.feature_maps[f] for f in feat_names], axis=-1)

        nib.save(
            nib.Nifti1Image(stack, self.mask_nii.affine, self.mask_nii.header),
            str(path),
        )

        # Sidecar with feature ordering
        sidecar_path = path.with_suffix("").with_suffix(".features.json")
        with open(sidecar_path, "w") as f:
            json.dump({"features": feat_names}, f, indent=2)

        logger.info("Saved 4D NIfTI (%d features) to %s", len(feat_names), path)
        return path

    # -- Display -----------------------------------------------------------

    def __repr__(self) -> str:
        mode = "voxelwise" if self.is_voxelwise else "ROI"
        return (
            f"ExtractionResult(mode={mode}, n_features={self.n_features}, "
            f"n_rois={self.n_rois}, image={self.metadata.image_path})"
        )

    def view(
        self,
        port: int = 0,
        features: Optional[list[str]] = None,
        open_browser: bool = True,
    ) -> None:
        """Launch interactive browser viewer for this result."""
        from radiomicviz.viewer import launch_viewer_from_result
        launch_viewer_from_result(self, port=port, features=features,
                                  open_browser=open_browser)

    def summary(self) -> str:
        """Human-readable summary of extraction results."""
        lines = [
            f"RadiomicViz Extraction Result",
            f"  Mode: {self.metadata.mode}",
            f"  Image: {self.metadata.image_path}",
            f"  Mask: {self.metadata.mask_path}",
            f"  Config: {self.metadata.config_source}",
            f"  Features: {self.n_features}",
            f"  ROIs: {self.n_rois}",
        ]
        if self.metadata.extraction_time_seconds:
            lines.append(f"  Time: {self.metadata.extraction_time_seconds:.1f}s")
        if self.diagnostics:
            n_ok = sum(1 for d in self.diagnostics if d.extraction_ok)
            n_warn = sum(1 for d in self.diagnostics if d.warnings)
            lines.append(f"  Diagnostics: {n_ok}/{len(self.diagnostics)} OK, {n_warn} with warnings")
        return "\n".join(lines)
