"""
RadiomicViz: Interactive 3D radiomics extraction, visualization, and analysis.

Quick start:
    >>> from radiomicviz import extract, batch_extract, validate, show_preset
    >>> result = extract("t1.nii.gz", "mask.nii.gz", preset="mri-default")
    >>> result.features.head()
"""

from radiomicviz._version import __version__
from radiomicviz.config import list_presets, show_preset
from radiomicviz.extract import extract
from radiomicviz.batch import batch_extract
from radiomicviz.validate import validate_inputs
from radiomicviz.result import ExtractionResult

__all__ = [
    "__version__",
    "extract",
    "batch_extract",
    "validate_inputs",
    "show_preset",
    "list_presets",
    "ExtractionResult",
]
