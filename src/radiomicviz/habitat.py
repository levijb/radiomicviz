"""
HabitatResult: standardized output from habitat clustering.

Provides a consistent interface for accessing cluster label maps, probability
maps, per-habitat statistics, and provenance, with export methods for NIfTI,
CSV, and viewer launch.

Usage:
    >>> result = cluster_habitats(extraction_result, method="gmm")
    >>> result.summary()
    >>> result.to_nifti("habitats.nii.gz")
    >>> result.to_csv("cluster_stats.csv")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger("radiomicviz.habitat")


@dataclass
class HabitatResult:
    """
    Standardized output from habitat clustering on voxelwise radiomics features.

    Attributes
    ----------
    label_map : np.ndarray
        3D integer array with the same spatial shape as the source image.
        0 = background (outside mask), 1..k = habitat labels.
    probabilities : np.ndarray, optional
        4D float array of shape (x, y, z, k) holding GMM soft-assignment
        probabilities per voxel. None for K-means results.
    n_clusters : int
        Number of habitats in the final fit.
    cluster_stats : pd.DataFrame
        Per-habitat descriptive statistics. Index = habitat label (int).
        Columns are MultiIndex (feature_name, stat) where stat is one of
        mean, std, median, p10, p90.
    selection_criteria : dict
        Mapping ``{k: {"bic": float|None, "silhouette": float, "ch": float,
        "db": float}}`` for every k evaluated during auto-selection.
        Empty dict when n_clusters was specified explicitly.
    selected_k : int
        The k value used for the final fit (equals n_clusters).
    features_used : list of str
        Feature names retained after redundancy elimination and (if applicable)
        PCA transformation.
    features_dropped : list of str
        Feature names removed before clustering, with reason appended as a
        suffix, e.g. ``"feature_name [reason: constant]"``.
    pca_variance_explained : float, optional
        Cumulative variance explained by the retained PCA components.
        None when PCA was not applied.
    metadata : dict
        Provenance: method, params, timestamp, radiomicviz_version.
    mask_nii : nibabel.Nifti1Image, optional
        The mask NIfTI retained for spatial export.
    """

    label_map: np.ndarray
    probabilities: Optional[np.ndarray]
    n_clusters: int
    cluster_stats: pd.DataFrame
    selection_criteria: dict
    selected_k: int
    features_used: list
    features_dropped: list
    pca_variance_explained: Optional[float]
    metadata: dict
    mask_nii: Optional[nib.Nifti1Image] = field(default=None, repr=False)

    # -- Export methods --------------------------------------------------------

    def to_nifti(self, path: Union[str, Path]) -> Path:
        """
        Save the habitat label map as an int16 3D NIfTI.

        Parameters
        ----------
        path : str or Path
            Output path for the NIfTI file.

        Returns
        -------
        Path
            Path to the written file.

        Raises
        ------
        ValueError
            If mask_nii was not retained (affine unavailable).
        """
        if self.mask_nii is None:
            raise ValueError(
                "Cannot export to NIfTI: mask_nii was not retained. "
                "Re-run cluster_habitats() with a result that has mask_nii set."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        affine = self.mask_nii.affine
        header = self.mask_nii.header.copy()
        header.set_data_dtype(np.int16)
        header["scl_slope"] = 1.0
        header["scl_inter"] = 0.0

        nib.save(
            nib.Nifti1Image(self.label_map.astype(np.int16), affine, header),
            str(path),
        )
        logger.info("Habitat label map saved to %s", path)
        return path

    def to_prob_nifti(self, path: Union[str, Path]) -> Path:
        """
        Save GMM soft-assignment probabilities as a float32 4D NIfTI.

        The 4th dimension corresponds to habitat index (0-indexed). Voxels
        outside the mask have probability 0 across all k channels.

        Parameters
        ----------
        path : str or Path
            Output path for the 4D NIfTI file.

        Returns
        -------
        Path
            Path to the written file.

        Raises
        ------
        ValueError
            If probabilities is None (K-means result) or mask_nii is missing.
        """
        if self.probabilities is None:
            raise ValueError(
                "Probability maps are only available for GMM results. "
                "Re-run cluster_habitats() with method='gmm'."
            )
        if self.mask_nii is None:
            raise ValueError(
                "Cannot export to NIfTI: mask_nii was not retained."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        affine = self.mask_nii.affine
        header = self.mask_nii.header.copy()
        header.set_data_dtype(np.float32)
        header["scl_slope"] = 1.0
        header["scl_inter"] = 0.0

        nib.save(
            nib.Nifti1Image(self.probabilities.astype(np.float32), affine, header),
            str(path),
        )
        logger.info("Habitat probability maps saved to %s", path)
        return path

    def to_csv(self, path: Union[str, Path]) -> Path:
        """
        Save cluster statistics to CSV with a companion .metadata.json sidecar.

        Parameters
        ----------
        path : str or Path
            Output CSV path. The sidecar is written alongside with the suffix
            ``.metadata.json``.

        Returns
        -------
        Path
            Path to the CSV file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cluster_stats.to_csv(path)
        logger.info("Cluster stats saved to %s", path)

        meta_path = path.with_suffix("").with_suffix(".metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info("Metadata saved to %s", meta_path)

        return path

    # -- Human-readable summary ------------------------------------------------

    def summary(self) -> str:
        """
        Return a formatted human-readable summary of the clustering result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "RadiomicViz Habitat Clustering Result",
            f"  Method:       {self.metadata.get('method', 'unknown')}",
            f"  n_clusters:   {self.n_clusters}",
            f"  selected_k:   {self.selected_k}",
        ]

        mask_voxels = int((self.label_map > 0).sum())
        lines.append(f"  Mask voxels:  {mask_voxels}")

        for hab_label in range(1, self.n_clusters + 1):
            n = int((self.label_map == hab_label).sum())
            pct = 100.0 * n / mask_voxels if mask_voxels > 0 else 0.0
            lines.append(f"  Habitat {hab_label}:    {n} voxels ({pct:.1f}%)")

        lines.append(f"  Features used ({len(self.features_used)}):")
        for f in self.features_used:
            lines.append(f"    - {f}")

        if self.features_dropped:
            lines.append(f"  Features dropped ({len(self.features_dropped)}):")
            for f in self.features_dropped:
                lines.append(f"    - {f}")

        if self.pca_variance_explained is not None:
            lines.append(
                f"  PCA variance explained: {self.pca_variance_explained:.3f}"
            )

        if self.selection_criteria:
            lines.append("  K-selection criteria:")
            for k, crit in sorted(self.selection_criteria.items()):
                parts = [f"k={k}"]
                if crit.get("bic") is not None:
                    parts.append(f"BIC={crit['bic']:.1f}")
                parts.append(f"sil={crit.get('silhouette', float('nan')):.3f}")
                lines.append("    " + "  ".join(parts))

        return "\n".join(lines)

    # -- Viewer stub -----------------------------------------------------------

    def view(self, port: int = 0, open_browser: bool = True) -> None:
        """
        Launch the interactive browser viewer with the habitat label map.

        Raises
        ------
        NotImplementedError
            Viewer integration is implemented in cluster.py Step E.
        """
        raise NotImplementedError(
            "Viewer integration coming in cluster.py Step E"
        )

    # -- Display ---------------------------------------------------------------

    def __repr__(self) -> str:
        method = self.metadata.get("method", "unknown")
        return (
            f"HabitatResult(method={method}, n_clusters={self.n_clusters}, "
            f"features_used={len(self.features_used)})"
        )
