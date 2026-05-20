"""
Flask server for the RadiomicViz browser viewer.

Routes:
  GET /              → viewer.html
  GET /data/<file>   → serve a NIfTI (or .nrrd converted on-the-fly) from the
                       registered files dict
  GET /api/volumes   → JSON manifest of available volumes

.nrrd conversion
----------------
NiiVue.js identifies formats by URL extension, so .nrrd files are served under
a .nii.gz URL.  When the real path on disk ends with .nrrd, the route converts
it with SimpleITK, writes a temp .nii.gz, caches the result, and serves that.
The cache is per-process (module-level dict); conversion happens at most once
per unique source path.
"""
from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_file

logger = logging.getLogger("radiomicviz.viewer")

# Module-level NRRD conversion cache shared across all Flask app instances in
# the same process.  Key: absolute nrrd path string → converted nii.gz path string.
_nrrd_cache: dict[str, str] = {}
_nrrd_cache_lock = threading.Lock()
_nrrd_tmpdir: Optional[str] = None
_nrrd_tmpdir_lock = threading.Lock()

# Threshold cache: (absolute path, threshold float) → processed nii.gz path
_thresh_cache: dict[tuple, str] = {}
_thresh_cache_lock = threading.Lock()


def _ensure_nrrd_tmpdir() -> str:
    global _nrrd_tmpdir
    with _nrrd_tmpdir_lock:
        if _nrrd_tmpdir is None:
            _nrrd_tmpdir = tempfile.mkdtemp(prefix="radiomicviz_nrrd_")
        return _nrrd_tmpdir


def _nrrd_as_nifti(nrrd_path_str: str) -> str:
    """Return path to a .nii.gz converted from *nrrd_path_str*, cached after first call."""
    with _nrrd_cache_lock:
        cached = _nrrd_cache.get(nrrd_path_str)
    if cached:
        return cached

    import SimpleITK as sitk  # deferred — not always installed

    nrrd_path = Path(nrrd_path_str)
    tmpdir = _ensure_nrrd_tmpdir()

    # Prefix with parent dir name to avoid collisions between ROI directories
    # that share feature names (e.g. AF_L/feat.nrrd and AF_R/feat.nrrd).
    out_name = f"{nrrd_path.parent.name}__{nrrd_path.stem}.nii.gz"
    out_path = Path(tmpdir) / out_name

    logger.debug("Converting %s → %s", nrrd_path_str, out_path)
    img = sitk.ReadImage(nrrd_path_str)
    sitk.WriteImage(img, str(out_path))

    with _nrrd_cache_lock:
        _nrrd_cache[nrrd_path_str] = str(out_path)

    return str(out_path)


def _apply_threshold(nifti_path: str, threshold: float) -> str:
    """Return path to a NIfTI with voxels where |value| <= threshold zeroed out, cached."""
    cache_key = (nifti_path, threshold)
    with _thresh_cache_lock:
        cached = _thresh_cache.get(cache_key)
    if cached and Path(cached).exists():
        return cached

    import nibabel as nib  # core dep — always available
    import numpy as np

    img = nib.load(nifti_path)
    arr = np.asarray(img.dataobj).copy().astype(np.float32)
    arr[np.abs(arr) <= threshold] = 0

    tmpdir = _ensure_nrrd_tmpdir()
    stem = Path(nifti_path).name.replace(".nii.gz", "").replace(".nii", "")
    out_name = f"thresh{threshold:.4f}_{stem}.nii.gz"
    out_path = Path(tmpdir) / out_name

    logger.debug("Thresholding %s at %.4f → %s", nifti_path, threshold, out_path)
    new_img = nib.Nifti1Image(arr, img.affine, img.header)
    nib.save(new_img, str(out_path))

    with _thresh_cache_lock:
        _thresh_cache[cache_key] = str(out_path)

    return str(out_path)


def create_app(files: dict[str, str], manifest: dict) -> Flask:
    """
    Build the Flask app.

    Parameters
    ----------
    files : dict[str, str]
        Mapping of URL filename → absolute path on disk.  For .nrrd source
        files the URL key should already end with .nii.gz (see _register in
        __init__.py); the route converts transparently.
    manifest : dict
        {
          "image": "image.nii.gz",
          "mask": "mask.nii.gz" | null,
          "overlays": ["feat.nii.gz", ...],
          "feature_4d": "maps.nii.gz" | null,
          "feature_4d_features": [...],
          "feature_4d_n_frames": int,
        }
    """
    app = Flask(__name__, template_folder="templates")
    app.config["FILES"] = files
    app.config["MANIFEST"] = manifest

    @app.route("/")
    def index():
        return render_template("viewer.html", manifest=manifest)

    @app.route("/data/<path:filename>")
    def serve_data(filename):
        real_path = app.config["FILES"].get(filename)
        if not real_path or not Path(real_path).exists():
            abort(404)

        if real_path.endswith(".nrrd"):
            try:
                real_path = _nrrd_as_nifti(real_path)
            except Exception as exc:
                logger.error("NRRD conversion failed for %s: %s", real_path, exc)
                abort(500)

        threshold_str = request.args.get("threshold")
        if threshold_str is not None:
            try:
                threshold = float(threshold_str)
                real_path = _apply_threshold(real_path, threshold)
            except Exception as exc:
                logger.error("Threshold failed for %s: %s", real_path, exc)
                abort(500)

        return send_file(str(real_path), mimetype="application/octet-stream")

    @app.route("/api/volumes")
    def volumes():
        return jsonify(app.config["MANIFEST"])

    @app.route("/api/backgrounds")
    def backgrounds():
        return jsonify({"files": app.config["MANIFEST"].get("backgrounds", [])})

    return app
