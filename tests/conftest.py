"""
Shared test fixtures for RadiomicViz.

Creates small synthetic NIfTI images and masks so tests run without
real neuroimaging data. All fixtures are session-scoped (created once).
"""

import numpy as np
import nibabel as nib
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory):
    """Session-level temp directory for test NIfTIs."""
    return tmp_path_factory.mktemp("radiomicviz_test_data")


@pytest.fixture(scope="session")
def synthetic_image(tmp_data_dir):
    """
    A small 20x20x20 NIfTI image with realistic-ish signal.

    Contains a gradient + noise so features are non-trivial.
    """
    np.random.seed(42)
    shape = (20, 20, 20)
    # Gradient along x-axis + Gaussian noise
    x = np.linspace(0, 100, shape[0])
    data = np.broadcast_to(x[:, None, None], shape).copy().astype(np.float64)
    data += np.random.normal(0, 10, shape)
    # Ensure no negative values (mimics real MRI intensity)
    data = np.clip(data, 0, None)

    affine = np.eye(4)
    affine[0, 0] = 1.0  # 1mm isotropic
    affine[1, 1] = 1.0
    affine[2, 2] = 1.0

    img = nib.Nifti1Image(data, affine)
    path = tmp_data_dir / "test_image.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture(scope="session")
def synthetic_mask_single(tmp_data_dir, synthetic_image):
    """Binary mask (label=1) — a cube in the center of the volume."""
    img = nib.load(str(synthetic_image))
    shape = img.shape[:3]
    mask = np.zeros(shape, dtype=np.int16)
    # 8x8x8 cube centered in the volume
    c = [s // 2 for s in shape]
    mask[c[0]-4:c[0]+4, c[1]-4:c[1]+4, c[2]-4:c[2]+4] = 1

    nii = nib.Nifti1Image(mask, img.affine)
    path = tmp_data_dir / "test_mask_single.nii.gz"
    nib.save(nii, str(path))
    return path


@pytest.fixture(scope="session")
def synthetic_mask_multi(tmp_data_dir, synthetic_image):
    """Multi-label mask (labels 1,2,3) — three non-overlapping cubes."""
    img = nib.load(str(synthetic_image))
    shape = img.shape[:3]
    mask = np.zeros(shape, dtype=np.int16)

    # Label 1: left block
    mask[2:8, 6:14, 6:14] = 1
    # Label 2: center block
    mask[8:14, 6:14, 6:14] = 2
    # Label 3: right block
    mask[14:18, 6:14, 6:14] = 3

    nii = nib.Nifti1Image(mask, img.affine)
    path = tmp_data_dir / "test_mask_multi.nii.gz"
    nib.save(nii, str(path))
    return path


@pytest.fixture(scope="session")
def empty_mask(tmp_data_dir, synthetic_image):
    """Mask with all zeros (should fail validation)."""
    img = nib.load(str(synthetic_image))
    mask = np.zeros(img.shape[:3], dtype=np.int16)
    nii = nib.Nifti1Image(mask, img.affine)
    path = tmp_data_dir / "test_mask_empty.nii.gz"
    nib.save(nii, str(path))
    return path


@pytest.fixture(scope="session")
def float_mask(tmp_data_dir, synthetic_image):
    """Mask with float values (should fail validation)."""
    img = nib.load(str(synthetic_image))
    mask = np.zeros(img.shape[:3], dtype=np.float64)
    c = [s // 2 for s in img.shape[:3]]
    mask[c[0]-4:c[0]+4, c[1]-4:c[1]+4, c[2]-4:c[2]+4] = 0.7
    nii = nib.Nifti1Image(mask, img.affine)
    path = tmp_data_dir / "test_mask_float.nii.gz"
    nib.save(nii, str(path))
    return path


@pytest.fixture(scope="session")
def mismatched_image(tmp_data_dir):
    """Image with different shape than the masks (30x30x30)."""
    np.random.seed(99)
    data = np.random.normal(50, 15, (30, 30, 30)).astype(np.float64)
    data = np.clip(data, 0, None)
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_data_dir / "test_image_mismatch.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture(scope="session")
def tiny_mask(tmp_data_dir, synthetic_image):
    """Mask with only 3 voxels (should warn about small ROI)."""
    img = nib.load(str(synthetic_image))
    mask = np.zeros(img.shape[:3], dtype=np.int16)
    mask[10, 10, 10] = 1
    mask[10, 10, 11] = 1
    mask[10, 11, 10] = 1
    nii = nib.Nifti1Image(mask, img.affine)
    path = tmp_data_dir / "test_mask_tiny.nii.gz"
    nib.save(nii, str(path))
    return path


@pytest.fixture(scope="session")
def nan_image(tmp_data_dir, synthetic_image):
    """Image with some NaN voxels."""
    img = nib.load(str(synthetic_image))
    data = np.asarray(img.dataobj).astype(np.float64).copy()
    data[5:8, 5:8, 5:8] = np.nan
    nii = nib.Nifti1Image(data, img.affine)
    path = tmp_data_dir / "test_image_nan.nii.gz"
    nib.save(nii, str(path))
    return path


@pytest.fixture(scope="session")
def subjects_csv(tmp_data_dir, synthetic_image, synthetic_mask_single, synthetic_mask_multi):
    """A small CSV for batch testing with 3 subjects."""
    df = pd.DataFrame({
        "subject_id": ["sub01", "sub02", "sub03"],
        "t1_path": [str(synthetic_image)] * 3,
        "mask_path": [
            str(synthetic_mask_single),
            str(synthetic_mask_multi),
            str(synthetic_mask_single),
        ],
        "group": ["MS", "HC", "MS"],
    })
    path = tmp_data_dir / "test_cohort.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def subjects_csv_with_bad(tmp_data_dir, synthetic_image, synthetic_mask_single, empty_mask):
    """CSV with one good subject and one that should fail (empty mask)."""
    df = pd.DataFrame({
        "subject_id": ["good_sub", "bad_sub"],
        "t1_path": [str(synthetic_image)] * 2,
        "mask_path": [str(synthetic_mask_single), str(empty_mask)],
    })
    path = tmp_data_dir / "test_cohort_with_bad.csv"
    df.to_csv(path, index=False)
    return path
