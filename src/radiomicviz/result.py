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


def _expand_to_full_shape(
    feat_img: Any,
    arr: np.ndarray,
    full_shape: tuple,
    nib_affine: np.ndarray,
) -> np.ndarray:
    """Place a spatially cropped SimpleITK feature map into a full-size numpy array.

    PyRadiomics voxelwise extraction always crops the output to the mask bounding
    box. This reconstructs the full-size (i, j, k) array (NaN outside the cropped
    region) using the feature map's *own* geometry — origin, spacing, and
    direction — so the data is placed both at the correct bounding-box corner and
    with the correct axis orientation.

    ``arr`` is the feature map already transposed to (i, j, k) SimpleITK index
    order (i.e. ``sitk.GetArrayFromImage(feat_img).T``). The feature map can carry
    a direction matrix whose axes are flipped or permuted relative to the mask's
    nibabel grid; using the origin alone (as a previous version did) places the
    bounding box correctly but leaves the interior axis-swapped. We derive the
    signed permutation between the two grids and reorient ``arr`` accordingly
    before placement. When the axes already align this is a no-op.

    Assumes standard neuroimaging LPS↔RAS convention (SimpleITK LPS, nibabel RAS).
    """
    lps2ras = np.array([-1.0, -1.0, 1.0])

    # Feature map index→physical mapping (SimpleITK LPS), converted to RAS so it
    # is comparable with the nibabel affine. Columns of ``feat_rot_ras`` give the
    # RAS displacement of a one-voxel step along each feature index axis.
    spacing = np.array(feat_img.GetSpacing(), dtype=float)
    direction = np.array(feat_img.GetDirection(), dtype=float).reshape(3, 3)
    origin_lps = np.array(feat_img.GetOrigin(), dtype=float)
    feat_rot_ras = (direction * spacing) * lps2ras[:, None]
    feat_origin_ras = origin_lps * lps2ras

    mask_rot = nib_affine[:3, :3]
    mask_origin = nib_affine[:3, 3]
    inv_mask_rot = np.linalg.inv(mask_rot)

    # ``M[:, f]`` expresses feature index-axis ``f`` in mask index coordinates;
    # for two axis-aligned grids of equal spacing this is a signed permutation.
    # ``t`` is the mask-index location of feature voxel (0, 0, 0).
    M = inv_mask_rot @ feat_rot_ras
    t = inv_mask_rot @ (feat_origin_ras - mask_origin)

    # Reorient arr's axes from feature order into mask (i, j, k) order.
    perm = [0, 0, 0]      # perm[mask_axis] = feature_axis
    signs = [1, 1, 1]
    for f in range(3):
        m = int(np.argmax(np.abs(M[:, f])))
        perm[m] = f
        signs[m] = 1 if M[m, f] >= 0 else -1
    arr_m = np.transpose(arr, perm)
    for m in range(3):
        if signs[m] < 0:
            arr_m = np.flip(arr_m, axis=m)

    # The reoriented array's [0, 0, 0] sits at the minimum mask-index corner of
    # the mapped bounding box. Compute that corner over all 8 box corners.
    dims = np.array(arr.shape, dtype=float) - 1.0
    corners = [
        M @ (dims * np.array(c, dtype=float)) + t
        for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
    ]
    offset = np.round(np.min(corners, axis=0)).astype(int)

    full_arr = np.full(full_shape, np.nan, dtype=np.float64)
    dst_start = np.maximum(0, offset)
    src_start = np.maximum(0, -offset)
    end = np.minimum(np.array(full_shape), offset + np.array(arr_m.shape))

    if np.any(end <= dst_start):
        return full_arr  # feature map doesn't overlap with full image

    dst = tuple(slice(int(dst_start[i]), int(end[i])) for i in range(3))
    src = tuple(
        slice(int(src_start[i]), int(src_start[i] + end[i] - dst_start[i]))
        for i in range(3)
    )
    full_arr[dst] = arr_m[src]
    return full_arr


@dataclass
class ExtractionMetadata:
    """Provenance information for an extraction run."""

    image_path: str
    mask_path: str
    config_source: str  # e.g. "preset: mri-default"
    config: dict[str, Any] = field(default_factory=dict)
    mode: str = "roi"  # "roi" or "voxelwise"
    label: Optional[int] = None
    roi_name: Optional[str] = None
    modality: Optional[str] = None
    subject_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pyradiomics_version: Optional[str] = None
    radiomicviz_version: Optional[str] = None
    extraction_time_seconds: Optional[float] = None
    brain_mode: Optional[str] = None  # "whole", "per-region", "hybrid", or None

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
    # Voxelwise: {label_key: {feature_name: sitk.Image}}, e.g. {"label1": {"original_firstorder_Mean": ...}}
    # ROI: None
    feature_maps: Optional[dict[str, Any]] = None
    mask_nii: Optional[nib.Nifti1Image] = None
    original_label_map: Optional[np.ndarray] = None  # only set for brain_mode="hybrid"

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

        Only available for voxelwise results. feature_maps must be the nested
        {label_key: {feat_name: sitk.Image}} structure produced by extract().

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

        import SimpleITK as sitk

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten {label_key: {feat: sitk.Image}} into a stable ordered list.
        # sitk.GetArrayFromImage returns (z, y, x); transpose to (x, y, z) for nibabel.
        #
        # PyRadiomics voxelwise extraction (preCrop: true) crops each feature map
        # to the mask bounding box, so the cropped arrays are smaller than the full
        # volume. Pad each map back to the mask's full shape using its LPS origin so
        # the saved NIfTI's affine (the full-volume mask affine) is spatially correct.
        full_shape = self.mask_nii.shape[:3]
        nib_affine = self.mask_nii.affine

        ordered_names: list[str] = []
        ordered_arrays: list[np.ndarray] = []
        for label_key in sorted(self.feature_maps):
            for feat_name in sorted(self.feature_maps[label_key]):
                val = self.feature_maps[label_key][feat_name]
                arr = sitk.GetArrayFromImage(val).T if isinstance(val, sitk.Image) else np.asarray(val)
                if arr.shape != full_shape:
                    if isinstance(val, sitk.Image):
                        arr = _expand_to_full_shape(val, arr, full_shape, nib_affine)
                    else:
                        logger.warning(
                            "Cannot align feature '%s' to full volume "
                            "(shapes %s vs %s); skipping.",
                            feat_name, arr.shape, full_shape,
                        )
                        continue
                ordered_names.append(f"{label_key}_{feat_name}")
                ordered_arrays.append(arr)

        stack = np.stack(ordered_arrays, axis=-1).astype(np.float32)

        # Copy spatial info from mask; reset scaling so float values display correctly.
        header = self.mask_nii.header.copy()
        header.set_data_shape(stack.shape)
        header.set_data_dtype(np.float32)
        header["scl_slope"] = 1.0
        header["scl_inter"] = 0.0

        nib.save(
            nib.Nifti1Image(stack, self.mask_nii.affine, header),
            str(path),
        )

        sidecar_path = path.with_suffix("").with_suffix(".features.json")
        with open(sidecar_path, "w") as f:
            json.dump({"features": ordered_names}, f, indent=2)

        logger.info("Saved 4D NIfTI (%d features) to %s", len(ordered_names), path)
        return path

    def to_nrrd(self, output_dir: Union[str, Path]) -> list[Path]:
        """
        Save each voxelwise feature map as an individual ``.nrrd`` file.

        Only available for voxelwise results. feature_maps must be the nested
        ``{label_key: {feat_name: sitk.Image}}`` structure produced by
        extract(). One subdirectory is created per ROI label (named by the
        label key, i.e. the ROI name or ``label{N}``), with one ``.nrrd`` file
        per feature inside it.

        Parameters
        ----------
        output_dir : str or Path
            Directory under which the per-ROI subdirectories are written.

        Returns
        -------
        list of Path
            Paths to the created ``.nrrd`` files.
        """
        if not self.is_voxelwise or self.feature_maps is None:
            raise ValueError("to_nrrd() is only available for voxelwise results.")

        import SimpleITK as sitk

        output_dir = Path(output_dir)
        paths: list[Path] = []
        for label_key in sorted(self.feature_maps):
            label_dir = output_dir / label_key
            label_dir.mkdir(parents=True, exist_ok=True)
            for feat_name, img in sorted(self.feature_maps[label_key].items()):
                out_path = label_dir / f"{feat_name}.nrrd"
                sitk.WriteImage(img, str(out_path))
                paths.append(out_path)

        logger.info("Saved %d .nrrd feature maps to %s", len(paths), output_dir)
        return paths

    # -- Hybrid / brain-mode helpers ---------------------------------------

    def features_by_region(self, region_label: int) -> pd.DataFrame:
        """
        Filter voxelwise feature maps to voxels belonging to a specific
        anatomical region. Only available when brain_mode='hybrid'.

        Parameters
        ----------
        region_label : int
            The label value from the original parcellation mask.

        Returns
        -------
        pd.DataFrame
            Feature values for voxels in the specified region.
            Columns = feature names, rows = voxels.

        Raises
        ------
        ValueError
            If no original_label_map is available (not hybrid mode).
        """
        if self.original_label_map is None:
            raise ValueError(
                "features_by_region() requires brain_mode='hybrid'. "
                "Re-run extraction with brain_mode='hybrid'."
            )
        if self.feature_maps is None:
            raise ValueError("No voxelwise feature maps available.")

        import SimpleITK as sitk

        region_mask = (self.original_label_map == region_label)
        full_shape = self.original_label_map.shape  # (x, y, z) nibabel convention
        nib_affine = self.mask_nii.affine if self.mask_nii is not None else None

        data: dict[str, np.ndarray] = {}
        for _label_key, label_maps in self.feature_maps.items():
            for feat_name, feat_img in label_maps.items():
                if isinstance(feat_img, sitk.Image):
                    # sitk returns (z,y,x); transpose to (x,y,z) to match nibabel
                    arr = sitk.GetArrayFromImage(feat_img).T
                else:
                    arr = np.asarray(feat_img)

                if arr.shape != full_shape:
                    # PyRadiomics voxelwise always crops to the mask bounding box.
                    # Reconstruct the full-size array by computing the crop offset
                    # from the feature map's spatial metadata.
                    if isinstance(feat_img, sitk.Image) and nib_affine is not None:
                        arr = _expand_to_full_shape(feat_img, arr, full_shape, nib_affine)
                    else:
                        logger.warning(
                            "Cannot align feature '%s' to original space "
                            "(shapes %s vs %s); skipping.",
                            feat_name, arr.shape, full_shape,
                        )
                        continue

                data[feat_name] = arr[region_mask]

        return pd.DataFrame(data)

    def available_regions(self) -> Optional[list[int]]:
        """
        List available region labels from the original parcellation.
        Only available in hybrid mode.

        Returns
        -------
        list of int or None
        """
        if self.original_label_map is None:
            return None
        labels = np.unique(self.original_label_map)
        return sorted(int(v) for v in labels if v != 0)

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
        if self.metadata.subject_id:
            lines.append(f"  Subject: {self.metadata.subject_id}")
        if self.metadata.modality:
            lines.append(f"  Modality: {self.metadata.modality}")
        if self.metadata.extraction_time_seconds:
            lines.append(f"  Time: {self.metadata.extraction_time_seconds:.1f}s")
        if self.metadata.brain_mode:
            lines.append(f"  Brain mode: {self.metadata.brain_mode}")
        if self.original_label_map is not None:
            n_regions = len(self.available_regions())
            lines.append(f"  Parcellation regions: {n_regions}")
        if self.diagnostics:
            n_ok = sum(1 for d in self.diagnostics if d.extraction_ok)
            n_warn = sum(1 for d in self.diagnostics if d.warnings)
            lines.append(f"  Diagnostics: {n_ok}/{len(self.diagnostics)} OK, {n_warn} with warnings")
        return "\n".join(lines)
