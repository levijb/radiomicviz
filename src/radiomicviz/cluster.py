"""Habitat clustering pipeline for voxelwise radiomics features."""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from radiomicviz._version import __version__
from radiomicviz.habitat import HabitatResult
from radiomicviz.result import ExtractionResult, _expand_to_full_shape

logger = logging.getLogger("radiomicviz.cluster")


def _spearman_matrix(X: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix via rank transformation (ties averaged)."""
    X_ranked = np.apply_along_axis(rankdata, 0, X).astype(np.float64)
    return np.corrcoef(X_ranked.T)


def _matrix_from_extraction_result(
    source: ExtractionResult,
) -> tuple[np.ndarray, np.ndarray, list[str], nib.Nifti1Image]:
    """
    Returns
    -------
    X : ndarray (n_voxels, n_features)  float64, within-mask voxels only
    mask_idx : ndarray of int, flat C-order indices of mask voxels
    feature_names : list of str
    mask_nii : nib.Nifti1Image
    """
    if not source.is_voxelwise or source.feature_maps is None:
        raise ValueError(
            "cluster_habitats() requires a voxelwise ExtractionResult "
            "(metadata.mode='voxelwise' with feature_maps present)."
        )
    if source.mask_nii is None:
        raise ValueError("ExtractionResult.mask_nii is required for clustering.")

    try:
        import SimpleITK as sitk  # noqa: F401 — presence check only
        _sitk = sitk
    except ImportError:
        _sitk = None

    mask_arr = np.asarray(source.mask_nii.dataobj) > 0
    full_shape = mask_arr.shape[:3]
    nib_affine = source.mask_nii.affine
    mask_flat = mask_arr.ravel()
    mask_idx = np.where(mask_flat)[0]

    feature_names: list[str] = []
    cols: list[np.ndarray] = []

    for label_key in sorted(source.feature_maps):
        for feat_name, val in sorted(source.feature_maps[label_key].items()):
            if _sitk is not None and isinstance(val, _sitk.Image):
                arr = _sitk.GetArrayFromImage(val).T  # (z,y,x) → (x,y,z)
                if arr.shape != full_shape:
                    arr = _expand_to_full_shape(val, arr, full_shape, nib_affine)
            else:
                arr = np.asarray(val, dtype=np.float64)
                if arr.shape != full_shape:
                    logger.warning(
                        "Feature '%s' shape %s != full shape %s; skipping.",
                        feat_name, arr.shape, full_shape,
                    )
                    continue
            cols.append(arr.ravel()[mask_flat].astype(np.float64))
            feature_names.append(f"{label_key}_{feat_name}")

    if not cols:
        raise ValueError("No feature maps could be extracted from ExtractionResult.")

    return np.column_stack(cols), mask_idx, feature_names, source.mask_nii


def _matrix_from_nifti(
    nifti_path: Union[str, Path],
    mask_path: Union[str, Path],
) -> tuple[np.ndarray, np.ndarray, list[str], nib.Nifti1Image]:
    """
    Returns
    -------
    X : ndarray (n_voxels, n_features)  float64
    mask_idx : ndarray of int
    feature_names : list of str
    mask_nii : nib.Nifti1Image
    """
    nifti_path = Path(nifti_path)
    nii_4d = nib.load(str(nifti_path))
    mask_nii = nib.load(str(mask_path))

    mask_arr = np.asarray(mask_nii.dataobj) > 0
    mask_idx = np.where(mask_arr.ravel())[0]

    data = np.asarray(nii_4d.dataobj, dtype=np.float64)
    if data.ndim == 3:
        data = data[..., np.newaxis]

    X = data[mask_arr]  # (n_voxels, n_features)

    sidecar = nifti_path.with_suffix("").with_suffix(".features.json")
    if sidecar.exists():
        with open(sidecar, encoding="utf-8") as f:
            feature_names = json.load(f)["features"]
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    return X, mask_idx, feature_names, mask_nii


def cluster_habitats(
    source: Union[ExtractionResult, str, Path],
    mask: Optional[Union[str, Path]] = None,
    method: str = "gmm",
    n_clusters: Union[int, str] = "auto",
    k_range: tuple = (2, 6),
    redundancy_threshold: float = 0.70,
    pca_components: Optional[Union[int, str]] = None,
    gmm_covariance_type: str = "full",
    random_state: int = 42,
    min_cluster_size: int = 10,
) -> HabitatResult:
    """
    Cluster voxelwise radiomics features into habitat regions.

    Parameters
    ----------
    source : ExtractionResult or str or Path
        Voxelwise ExtractionResult, or path to a 4D NIfTI file
        (sidecar ``<name>.features.json`` used for feature names if present).
    mask : str or Path, optional
        Path to a binary mask NIfTI. Required when source is a path.
    method : {"gmm", "kmeans"}
        Clustering algorithm.
    n_clusters : int or "auto"
        Fixed cluster count, or "auto" to select via BIC gradient (GMM)
        or silhouette maximum (K-means).
    k_range : (int, int)
        Min/max k evaluated during auto-selection.
    redundancy_threshold : float
        Spearman |r| above which the lower-variance feature is dropped.
    pca_components : int, "auto", or None
        PCA dimensionality. "auto" retains 95 %% variance. None skips PCA.
    gmm_covariance_type : {"full", "tied", "diag", "spherical"}
        GMM covariance structure; may be overridden to "diag" automatically.
    random_state : int
        Random seed for reproducibility.
    min_cluster_size : int
        Minimum voxels per habitat; raises UserWarning if violated.

    Returns
    -------
    HabitatResult
    """
    # ── Step 1: Parse input ────────────────────────────────────────────────────
    if isinstance(source, ExtractionResult):
        X_raw, mask_idx, raw_names, mask_nii = _matrix_from_extraction_result(source)
    elif isinstance(source, (str, Path)):
        if mask is None:
            raise ValueError("mask parameter required when source is a NIfTI path.")
        X_raw, mask_idx, raw_names, mask_nii = _matrix_from_nifti(source, mask)
    else:
        raise TypeError(f"source must be ExtractionResult or path, got {type(source)}")

    full_shape = tuple(int(x) for x in mask_nii.header.get_data_shape()[:3])
    n_voxels, n_raw = X_raw.shape
    logger.info("Feature matrix: %d voxels × %d features", n_voxels, n_raw)

    # ── Step 3: Drop bad features ─────────────────────────────────────────────
    features_dropped: list[str] = []
    keep3 = np.ones(n_raw, dtype=bool)

    for i, fname in enumerate(raw_names):
        col = X_raw[:, i]
        nan_frac = float(np.isnan(col).mean())
        if nan_frac > 0.10:
            features_dropped.append(
                f"{fname} [reason: high NaN fraction ({nan_frac:.1%})]"
            )
            keep3[i] = False
            logger.info("Dropping '%s': high NaN fraction (%.0f%%)", fname, nan_frac * 100)
        elif np.nanvar(col) == 0.0:
            features_dropped.append(f"{fname} [reason: constant]")
            keep3[i] = False
            logger.info("Dropping '%s': constant", fname)

    X1 = X_raw[:, keep3].copy()
    names1 = [raw_names[i] for i in range(n_raw) if keep3[i]]

    # ── Step 4: Impute NaNs ───────────────────────────────────────────────────
    for i in range(X1.shape[1]):
        nan_mask = np.isnan(X1[:, i])
        if nan_mask.any():
            X1[nan_mask, i] = float(np.nanmedian(X1[:, i]))

    # ── Step 5: Spearman redundancy elimination ────────────────────────────────
    n1 = X1.shape[1]
    if n1 > 1:
        rho = _spearman_matrix(X1)
        variances = np.var(X1, axis=0)
        drop5 = np.zeros(n1, dtype=bool)

        for i in range(n1):
            if drop5[i]:
                continue
            for j in range(i + 1, n1):
                if drop5[j]:
                    continue
                if abs(rho[i, j]) > redundancy_threshold:
                    d = j if variances[i] >= variances[j] else i
                    partner = names1[i if d == j else j]
                    reason = (
                        f"correlated with {partner} (|r|={abs(rho[i, j]):.3f})"
                    )
                    features_dropped.append(f"{names1[d]} [reason: {reason}]")
                    drop5[d] = True
                    logger.info("Dropping '%s': %s", names1[d], reason)

        X2 = X1[:, ~drop5]
        names2 = [names1[i] for i in range(n1) if not drop5[i]]
    else:
        X2, names2 = X1, names1

    # ── Step 6: Z-score normalize ─────────────────────────────────────────────
    scaler = StandardScaler()
    X3 = scaler.fit_transform(X2)

    # ── Step 7: Optional PCA ──────────────────────────────────────────────────
    pca_variance_explained: Optional[float] = None

    if pca_components is not None:
        n_comp: Union[int, float]
        if pca_components == "auto":
            n_comp = 0.95
        else:
            n_comp = min(int(pca_components), X3.shape[1], n_voxels)
        pca_model = PCA(n_components=n_comp)
        X_fit = pca_model.fit_transform(X3)
        pca_variance_explained = float(np.sum(pca_model.explained_variance_ratio_))
        features_used: list[str] = [f"PC{i + 1}" for i in range(X_fit.shape[1])]
        logger.info(
            "PCA: %d components, %.1f%% variance",
            X_fit.shape[1], pca_variance_explained * 100,
        )
    else:
        X_fit = X3
        features_used = list(names2)

    n_fit = X_fit.shape[1]

    # ── Step 10: Auto-fallback covariance (before any GMM fitting) ─────────────
    eff_cov = gmm_covariance_type
    if method == "gmm" and n_voxels < n_fit * 10 and gmm_covariance_type == "full":
        eff_cov = "diag"
        warnings.warn(
            f"n_voxels ({n_voxels}) < n_features*10 ({n_fit * 10}): "
            f"GMM covariance switched from 'full' to 'diag'.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "GMM covariance auto-fallback: 'full' → 'diag' "
            "(%d voxels, %d features after PCA/cleaning)",
            n_voxels, n_fit,
        )

    # ── Step 8: K selection ───────────────────────────────────────────────────
    selection_criteria: dict = {}

    if n_clusters == "auto":
        k_vals = list(range(k_range[0], k_range[1] + 1))
        bic_per_k: dict[int, float] = {}

        for k in k_vals:
            if method == "gmm":
                m = GaussianMixture(
                    n_components=k,
                    covariance_type=eff_cov,
                    random_state=random_state,
                    n_init=3,
                )
                m.fit(X_fit)
                labs = m.predict(X_fit)
                bic_per_k[k] = float(m.bic(X_fit))
            else:
                m = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labs = m.fit_predict(X_fit)

            n_unique = len(np.unique(labs))
            sil = float(silhouette_score(X_fit, labs)) if n_unique > 1 else float("nan")
            ch  = float(calinski_harabasz_score(X_fit, labs)) if n_unique > 1 else float("nan")
            db  = float(davies_bouldin_score(X_fit, labs)) if n_unique > 1 else float("nan")

            selection_criteria[k] = {
                "bic": bic_per_k.get(k),
                "silhouette": sil,
                "ch": ch,
                "db": db,
            }

        if method == "gmm":
            if len(k_vals) > 1:
                drops = [
                    bic_per_k[k_vals[i]] - bic_per_k[k_vals[i + 1]]
                    for i in range(len(k_vals) - 1)
                ]
                selected_k = k_vals[int(np.argmax(drops)) + 1]
            else:
                selected_k = k_vals[0]
        else:
            valid = {
                k: selection_criteria[k]["silhouette"]
                for k in k_vals
                if not np.isnan(selection_criteria[k]["silhouette"])
            }
            selected_k = max(valid, key=valid.__getitem__) if valid else k_vals[0]

        logger.info("k auto-selected: %d (method=%s)", selected_k, method)
    else:
        selected_k = int(n_clusters)

    # ── Step 9: Final fit ─────────────────────────────────────────────────────
    if method == "gmm":
        final = GaussianMixture(
            n_components=selected_k,
            covariance_type=eff_cov,
            random_state=random_state,
            n_init=5,
        )
        final.fit(X_fit)
        raw_labels = final.predict(X_fit)
        raw_probs: Optional[np.ndarray] = final.predict_proba(X_fit)
    else:
        final = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        raw_labels = final.fit_predict(X_fit)
        raw_probs = None

    labels = raw_labels + 1  # 0-indexed → 1-indexed

    # ── Step 11: Back-project to full image volume ─────────────────────────────
    label_map = np.zeros(full_shape, dtype=np.int32)
    label_map.reshape(-1)[mask_idx] = labels

    if raw_probs is not None:
        prob_map: Optional[np.ndarray] = np.zeros(
            (*full_shape, selected_k), dtype=np.float32
        )
        prob_map.reshape(-1, selected_k)[mask_idx] = raw_probs.astype(np.float32)
    else:
        prob_map = None

    # ── Step 12: Cluster stats (pre-normalization feature values) ──────────────
    rows = []
    for hab in range(1, selected_k + 1):
        m = labels == hab
        row: dict = {}
        for fi, fn in enumerate(names2):
            v = X2[m, fi]
            row[(fn, "mean")]   = float(np.mean(v))
            row[(fn, "std")]    = float(np.std(v))
            row[(fn, "median")] = float(np.median(v))
            row[(fn, "p10")]    = float(np.percentile(v, 10))
            row[(fn, "p90")]    = float(np.percentile(v, 90))
        rows.append(row)

    cluster_stats = pd.DataFrame(
        rows,
        index=pd.Index(range(1, selected_k + 1), name="habitat"),
    )
    cluster_stats.columns = pd.MultiIndex.from_tuples(
        list(cluster_stats.columns), names=["feature", "stat"]
    )

    # ── Step 13: Warn if small habitats ────────────────────────────────────────
    for hab in range(1, selected_k + 1):
        n_hab = int(np.sum(labels == hab))
        if n_hab < min_cluster_size:
            warnings.warn(
                f"Habitat {hab} has only {n_hab} voxels "
                f"(min_cluster_size={min_cluster_size}).",
                UserWarning,
                stacklevel=2,
            )
            logger.warning("Habitat %d: only %d voxels", hab, n_hab)

    # ── Step 14: Return HabitatResult ─────────────────────────────────────────
    return HabitatResult(
        label_map=label_map,
        probabilities=prob_map,
        n_clusters=selected_k,
        cluster_stats=cluster_stats,
        selection_criteria=selection_criteria,
        selected_k=selected_k,
        features_used=features_used,
        features_dropped=features_dropped,
        pca_variance_explained=pca_variance_explained,
        metadata={
            "method": method,
            "gmm_covariance_type": eff_cov if method == "gmm" else None,
            "n_clusters_requested": n_clusters,
            "k_range": list(k_range),
            "redundancy_threshold": redundancy_threshold,
            "pca_components": pca_components,
            "random_state": random_state,
            "min_cluster_size": min_cluster_size,
            "n_voxels": n_voxels,
            "n_features_input": n_raw,
            "n_features_after_cleaning": len(names2),
            "timestamp": datetime.now().isoformat(),
            "radiomicviz_version": __version__,
        },
        mask_nii=mask_nii,
    )


# ---------------------------------------------------------------------------
# Batch helpers  (module-level so joblib workers can pickle them)
# ---------------------------------------------------------------------------

def _resolve_id_col(df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    """Return a subject-ID column name, trying user_col then common names."""
    if user_col and user_col in df.columns:
        return user_col
    for candidate in ("subject_id", "Subject", "Patient", "participant_id", "ID", "id"):
        if candidate in df.columns:
            return candidate
    df["_row_index"] = [f"sub_{i:04d}" for i in range(len(df))]
    return "_row_index"


def _cluster_one(
    job: dict,
    method: str,
    n_clusters,
    k_range: tuple,
    redundancy_threshold: float,
    pca_components,
    gmm_covariance_type: str,
    random_state: int,
    min_cluster_size: int,
) -> tuple:
    """Per-subject worker for joblib; returns (subject_id, HabitatResult | error_string)."""
    sub_id = job["subject_id"]
    try:
        result = cluster_habitats(
            source=job["feature_4d"],
            mask=job["mask"],
            method=method,
            n_clusters=n_clusters,
            k_range=k_range,
            redundancy_threshold=redundancy_threshold,
            pca_components=pca_components,
            gmm_covariance_type=gmm_covariance_type,
            random_state=random_state,
            min_cluster_size=min_cluster_size,
        )
        return (sub_id, result)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Subject %s failed: %s", sub_id, error_msg)
        return (sub_id, error_msg)


def batch_cluster(
    subjects: Union[str, Path, pd.DataFrame],
    feature_4d_col: str,
    mask_col: str,
    output_dir: Union[str, Path],
    subject_id_col: Optional[str] = None,
    n_jobs: int = 1,
    save_probs: bool = False,
    method: str = "gmm",
    n_clusters: Union[int, str] = "auto",
    k_range: tuple = (2, 6),
    redundancy_threshold: float = 0.70,
    pca_components: Optional[Union[int, str]] = None,
    gmm_covariance_type: str = "full",
    random_state: int = 42,
    min_cluster_size: int = 10,
) -> dict:
    """
    Cluster voxelwise radiomics features for a cohort of subjects.

    Parameters
    ----------
    subjects : str, Path, or pd.DataFrame
        CSV path or already-loaded DataFrame with per-subject rows.
    feature_4d_col : str
        Column containing paths to 4D NIfTI feature maps.
    mask_col : str
        Column containing paths to binary mask NIfTIs.
    output_dir : str or Path
        Root output directory. Per-subject results go to
        ``{output_dir}/{subject_id}/``.
    subject_id_col : str, optional
        Column with subject IDs. Auto-detected when None.
    n_jobs : int
        Parallel workers (joblib). 1 = sequential.
    save_probs : bool
        Save GMM probability map when available.
    method, n_clusters, k_range, redundancy_threshold, pca_components,
    gmm_covariance_type, random_state, min_cluster_size :
        Forwarded to ``cluster_habitats()``.

    Returns
    -------
    dict
        Manifest with keys: ``radiomicviz_version``, ``timestamp``,
        ``subjects_csv``, ``total_rows``, ``succeeded_rows``,
        ``failed_rows``, ``total_time_seconds``, ``cluster_params``,
        ``successes``, ``failures``, ``manifest_path``.
    """
    if isinstance(subjects, pd.DataFrame):
        df = subjects.copy()
        subjects_label = "<DataFrame>"
    else:
        subjects_path = Path(subjects)
        if not subjects_path.exists():
            raise FileNotFoundError(f"Subjects CSV not found: {subjects_path}")
        df = pd.read_csv(subjects_path)
        subjects_label = str(subjects_path)
        logger.info("Loaded %d subjects from %s", len(df), subjects_path)

    for col in (feature_4d_col, mask_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {list(df.columns)}"
            )

    id_col = _resolve_id_col(df, subject_id_col)
    logger.info("Subject ID column: '%s'", id_col)

    jobs = [
        {
            "subject_id": str(row[id_col]),
            "feature_4d": row[feature_4d_col],
            "mask": row[mask_col],
        }
        for _, row in df.iterrows()
    ]

    cluster_kwargs: dict = dict(
        method=method,
        n_clusters=n_clusters,
        k_range=k_range,
        redundancy_threshold=redundancy_threshold,
        pca_components=pca_components,
        gmm_covariance_type=gmm_covariance_type,
        random_state=random_state,
        min_cluster_size=min_cluster_size,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if n_jobs == 1:
        raw = [_cluster_one(job, **cluster_kwargs) for job in jobs]
    else:
        from joblib import Parallel, delayed
        raw = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_cluster_one)(job, **cluster_kwargs) for job in jobs
        )

    total_time = time.time() - t0

    successes: list[dict] = []
    failures: list[dict] = []

    for job, (sub_id, result_or_error) in zip(jobs, raw):
        safe_id = sub_id.replace("/", "_").replace(" ", "_")
        if isinstance(result_or_error, HabitatResult):
            sub_dir = out / safe_id
            sub_dir.mkdir(parents=True, exist_ok=True)
            result_or_error.to_nifti(sub_dir / "habitats.nii.gz")
            result_or_error.to_csv(sub_dir / "cluster_stats.csv")
            if save_probs and result_or_error.probabilities is not None:
                result_or_error.to_prob_nifti(sub_dir / "habitats_probs.nii.gz")
            successes.append({"subject_id": sub_id, "output_dir": str(sub_dir)})
            logger.info("Subject %s: OK → %s", sub_id, sub_dir)
        else:
            failures.append({"subject_id": sub_id, "error": result_or_error})
            logger.error("Subject %s: FAILED — %s", sub_id, result_or_error)

    manifest: dict = {
        "radiomicviz_version": __version__,
        "timestamp": datetime.now().isoformat(),
        "subjects_csv": subjects_label,
        "total_rows": len(jobs),
        "succeeded_rows": len(successes),
        "failed_rows": len(failures),
        "total_time_seconds": round(total_time, 2),
        "cluster_params": {k: str(v) for k, v in cluster_kwargs.items()},
        "successes": successes,
        "failures": failures,
    }

    manifest_path = out / "batch_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Manifest written to %s", manifest_path)
    manifest["manifest_path"] = str(manifest_path)

    logger.info(
        "Batch complete: %d/%d succeeded, %d failed (%.1fs)",
        len(successes), len(jobs), len(failures), total_time,
    )
    return manifest
