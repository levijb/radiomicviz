"""Tests for radiomicviz.config."""

import pytest
from radiomicviz.config import (
    list_presets,
    load_preset,
    resolve_config,
    show_preset,
    save_config,
)


class TestPresets:
    def test_list_presets_returns_all(self):
        presets = list_presets()
        assert "mri-default" in presets
        assert "mri-texture" in presets
        assert "mri-firstorder" in presets
        assert "mri-habitat" in presets
        assert "mri-all-transforms" in presets
        assert "minimal" in presets

    def test_load_preset(self):
        config = load_preset("mri-default")
        assert "setting" in config
        assert "imageType" in config
        assert "featureClass" in config

    def test_load_nonexistent_preset(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_preset("nonexistent-preset")

    def test_show_preset(self, capsys):
        content = show_preset("minimal")
        assert "firstorder" in content
        captured = capsys.readouterr()
        assert "Preset: minimal" in captured.out

    def test_all_transforms_preset_has_all_filters(self):
        config = load_preset("mri-all-transforms")
        image_types = config["imageType"]
        expected = ["Original", "LoG", "Wavelet", "Square", "SquareRoot",
                    "Logarithm", "Exponential", "Gradient", "LBP2D", "LBP3D"]
        for t in expected:
            assert t in image_types, f"Missing image type: {t}"


class TestResolveConfig:
    def test_default_fallback(self):
        config, source = resolve_config()
        assert "mri-default" in source
        assert "featureClass" in config

    def test_preset_override(self):
        config, source = resolve_config(preset="minimal")
        assert "minimal" in source

    def test_config_dict(self):
        custom = {
            "imageType": {"Original": {}},
            "featureClass": {"firstorder": None},
        }
        config, source = resolve_config(config=custom)
        assert "user-provided dict" in source

    def test_overrides_merge(self):
        config, source = resolve_config(
            preset="mri-default",
            overrides={"binWidth": 50}
        )
        assert config["setting"]["binWidth"] == 50
        assert "overrides" in source

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="imageType"):
            resolve_config(config={"nothing": "here"})

    def test_config_file(self, tmp_path):
        # Save a preset and reload it as a file
        config = load_preset("minimal")
        path = save_config(config, tmp_path / "test_config.yaml")
        loaded, source = resolve_config(config=path)
        assert "custom file" in source
        assert "featureClass" in loaded
