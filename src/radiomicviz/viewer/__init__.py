"""
RadiomicViz browser viewer.

Public API
----------
launch_viewer(image, mask, overlays, feature_4d, port, open_browser)
    Launch the viewer from NIfTI files on disk.

launch_viewer_from_result(result, port, features, open_browser)
    Launch the viewer from an ExtractionResult object, writing NIfTIs
    to a temp directory as needed.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import tempfile
import threading
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from radiomicviz.result import ExtractionResult

logger = logging.getLogger("radiomicviz.viewer")


# ── Public API ────────────────────────────────────────────────────────────────

def launch_viewer(
    image: str | Path,
    mask: Optional[str | Path] = None,
    overlays: Optional[list[str | Path]] = None,
    feature_4d: Optional[str | Path] = None,
    overlay_dir: Optional[str | Path] = None,
    port: int = 0,
    open_browser: bool = True,
) -> None:
    """
    Launch the browser viewer from NIfTI or NRRD files on disk.

    Parameters
    ----------
    image : str or Path
        Background image NIfTI.
    mask : str or Path, optional
        Mask NIfTI (shown as semi-transparent red overlay).
    overlays : list of str or Path, optional
        Feature map NIfTIs or NRRD files selectable via dropdown.
        NRRD files are converted to NIfTI on-the-fly by the server.
    feature_4d : str or Path, optional
        4D NIfTI with stacked feature maps.  A sidecar
        ``<stem>.features.json`` is read if present for feature names.
    overlay_dir : str or Path, optional
        Directory to scan for overlay files.  All ``*.nrrd`` and
        ``*.nii.gz`` files found are added to the overlay list.
        Combined with any explicit ``overlays`` entries.
    port : int
        Port to bind (0 = pick a free port automatically).
    open_browser : bool
        Open the system browser automatically.
    """
    _check_flask()

    files: dict[str, str] = {}

    image_name = _register(files, Path(image).resolve())

    mask_name = None
    if mask:
        mask_name = _register(files, Path(mask).resolve())

    # Combine explicit overlays with any files found in overlay_dir
    all_overlays: list[Path] = [Path(ov).resolve() for ov in (overlays or [])]
    if overlay_dir:
        all_overlays.extend(_collect_overlay_dir(overlay_dir))

    overlay_names: list[str] = []
    for ov in all_overlays:
        overlay_names.append(_register(files, ov))

    feat4d_name = None
    feat4d_features: list[str] = []
    feat4d_n_frames = 1
    if feature_4d:
        p = Path(feature_4d).resolve()
        feat4d_name = _register(files, p)
        sidecar = p.with_suffix("").with_suffix(".features.json")
        if sidecar.exists():
            with open(sidecar) as f:
                feat4d_features = json.load(f).get("features", [])
        # Read frame count from NIfTI header
        try:
            nii = nib.load(str(p))
            feat4d_n_frames = int(nii.shape[3]) if len(nii.shape) > 3 else 1
        except Exception:
            feat4d_n_frames = len(feat4d_features) or 1

    manifest = {
        "image": image_name,
        "mask": mask_name,
        "overlays": overlay_names,
        "feature_4d": feat4d_name,
        "feature_4d_features": feat4d_features,
        "feature_4d_n_frames": feat4d_n_frames,
    }

    _serve(files=files, manifest=manifest, port=port, open_browser=open_browser)


def launch_viewer_from_result(
    result: "ExtractionResult",
    port: int = 0,
    features: Optional[list[str]] = None,
    open_browser: bool = True,
) -> None:
    """
    Launch the browser viewer from an ExtractionResult.

    Writes NIfTIs to a temporary directory, serves them, then cleans up
    after the viewer is closed (Ctrl+C).

    Parameters
    ----------
    result : ExtractionResult
        Extraction result from ``extract()`` or ``batch_extract()``.
    port : int
        Port to bind (0 = auto).
    features : list of str, optional
        Subset of feature names to expose as overlays.  Defaults to all.
    open_browser : bool
        Open the system browser automatically.
    """
    _check_flask()

    tmp_dir = tempfile.mkdtemp(prefix="radiomicviz_viewer_")
    logger.debug("Viewer temp dir: %s", tmp_dir)

    try:
        _prepare_result_files(result, tmp_dir, features)
        files = _scan_dir(tmp_dir)
        manifest = _build_result_manifest(result, tmp_dir, features)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    def _cleanup():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Viewer temp dir cleaned up")

    _serve(files=files, manifest=manifest, port=port, open_browser=open_browser,
           on_stop=_cleanup)


# ── File helpers ──────────────────────────────────────────────────────────────

def _register(files: dict[str, str], path: Path, name: Optional[str] = None) -> str:
    """Add *path* to the files registry, return the URL key used.

    .nrrd files are registered under a .nii.gz URL key so NiiVue (which
    identifies format by URL extension) receives a recognisable extension.
    The server converts the file on-the-fly when that URL is requested.
    """
    if name is None:
        raw = path.name
        name = raw[:-5] + ".nii.gz" if raw.endswith(".nrrd") else raw
    key = name
    if key in files and files[key] != str(path):
        # Disambiguate with parent directory name
        key = f"{path.parent.name}__{name}"
    files[key] = str(path)
    return key


def _collect_overlay_dir(directory: str | Path) -> list[Path]:
    """Return sorted list of .nrrd and .nii.gz files found in *directory*."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"overlay_dir is not a directory: {d}")
    files = sorted(
        p for p in d.iterdir()
        if p.is_file() and (p.suffix == ".nrrd" or p.name.endswith(".nii.gz"))
    )
    if not files:
        logger.warning("overlay_dir %s contains no .nrrd or .nii.gz files", d)
    return files


def _scan_dir(directory: str) -> dict[str, str]:
    d = Path(directory)
    return {f.name: str(f) for f in d.iterdir() if f.is_file()}


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(str(src), str(dst))
    except (OSError, NotImplementedError):
        shutil.copy2(str(src), str(dst))


# ── Result → temp dir ─────────────────────────────────────────────────────────

def _prepare_result_files(
    result: "ExtractionResult",
    tmp_dir: str,
    features: Optional[list[str]],
) -> None:
    """Save all required NIfTIs into tmp_dir."""
    import pandas as pd  # already a core dep

    tmp = Path(tmp_dir)

    # Background image
    image_src = Path(result.metadata.image_path)
    _link_or_copy(image_src, tmp / image_src.name)

    # Mask
    if result.mask_nii is not None:
        nib.save(result.mask_nii, str(tmp / "mask.nii.gz"))
    elif result.metadata.mask_path:
        mask_src = Path(result.metadata.mask_path)
        if mask_src.exists():
            _link_or_copy(mask_src, tmp / mask_src.name)

    # Feature maps
    if result.is_voxelwise and result.feature_maps:
        affine = result.mask_nii.affine if result.mask_nii else np.eye(4)
        header = result.mask_nii.header if result.mask_nii else None
        feat_names = features or list(result.feature_maps.keys())
        for name in feat_names:
            if name not in result.feature_maps:
                continue
            arr = result.feature_maps[name].astype(np.float32)
            safe = _safe_name(name)
            nib.save(nib.Nifti1Image(arr, affine, header), str(tmp / f"{safe}.nii.gz"))

    elif not result.is_voxelwise and result.mask_nii is not None:
        # ROI mode: choropleth — one NIfTI per feature
        mask_data = np.asarray(result.mask_nii.dataobj).astype(np.int32)
        affine = result.mask_nii.affine
        header = result.mask_nii.header
        feat_names = features or list(result.features.columns)
        for feat_name in feat_names:
            if feat_name not in result.features.columns:
                continue
            vol = np.zeros_like(mask_data, dtype=np.float32)
            for label_val in result.features.index:
                val = result.features.loc[label_val, feat_name]
                if pd.notna(val):
                    vol[mask_data == label_val] = float(val)
            safe = _safe_name(feat_name)
            nib.save(nib.Nifti1Image(vol, affine, header), str(tmp / f"{safe}.nii.gz"))

    elif not result.is_voxelwise and result.mask_nii is None:
        logger.warning(
            "ROI-mode result has no mask_nii — feature overlays unavailable. "
            "Re-run extraction with retain_mask=True to enable overlay visualization."
        )


def _build_result_manifest(
    result: "ExtractionResult",
    tmp_dir: str,
    features: Optional[list[str]],
) -> dict:
    tmp = Path(tmp_dir)
    image_name = Path(result.metadata.image_path).name

    # Mask name
    if result.mask_nii is not None:
        mask_name: Optional[str] = "mask.nii.gz"
    elif result.metadata.mask_path:
        mask_src = Path(result.metadata.mask_path)
        mask_name = mask_src.name if (tmp / mask_src.name).exists() else None
    else:
        mask_name = None

    # Overlay names
    overlays: list[str] = []
    if result.is_voxelwise and result.feature_maps:
        feat_names = features or list(result.feature_maps.keys())
        for name in feat_names:
            if name in result.feature_maps:
                overlays.append(f"{_safe_name(name)}.nii.gz")
    elif not result.is_voxelwise and result.mask_nii is not None:
        feat_names = features or list(result.features.columns)
        for feat_name in feat_names:
            if feat_name in result.features.columns:
                overlays.append(f"{_safe_name(feat_name)}.nii.gz")

    return {
        "image": image_name,
        "mask": mask_name,
        "overlays": overlays,
        "feature_4d": None,
        "feature_4d_features": [],
        "feature_4d_n_frames": 1,
    }


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


# ── Flask server ──────────────────────────────────────────────────────────────

def _check_flask() -> None:
    try:
        import flask  # noqa: F401
    except ImportError:
        raise ImportError(
            "Flask is required for the viewer. Install it with:\n"
            "  pip install flask\n"
            "or:\n"
            "  pip install 'radiomicviz[viewer]'"
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _serve(
    files: dict[str, str],
    manifest: dict,
    port: int,
    open_browser: bool,
    on_stop=None,
) -> None:
    from radiomicviz.viewer.app import create_app

    if port == 0:
        port = _free_port()

    app = create_app(files=files, manifest=manifest)

    # Suppress Flask's default request logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    server_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0", port=port, use_reloader=False, threaded=True
        ),
        daemon=True,
    )
    server_thread.start()

    url = f"http://localhost:{port}"
    print(f"\nRadiomicViz viewer -> {url}")
    print("Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        # Join with timeout so KeyboardInterrupt is catchable on all platforms
        while server_thread.is_alive():
            server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        if on_stop:
            on_stop()
