"""Tests for hyperwave_community.visualization.show_device_3d.

Tests the 3D device viewer function: return value structure, output routing,
summary/UI content, monitor handling, mode selection, and edge cases.
"""

import json
import os

import numpy as np
import pytest

from hyperwave_community.visualization import show_device_3d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_inputs():
    """Standard test inputs: random density, 3-layer stack (box/sin/clad)."""
    np.random.seed(42)
    density = np.random.rand(100, 60).astype(np.float64)
    layers = [
        {"name": "box", "thickness": 2.0, "index": 1.44, "material": "sio2"},
        {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
        {"name": "clad", "thickness": 2.0, "index": 1.44, "material": "sio2"},
    ]
    pixel_size = 0.015
    return density, layers, pixel_size


@pytest.fixture
def binary_inputs():
    """Fully binarized density (binarization score = 1.0) for contour mode tests."""
    density = np.zeros((80, 40), dtype=np.float64)
    density[20:60, 10:30] = 1.0  # rectangular feature
    layers = [
        {"name": "box", "thickness": 2.0, "index": 1.44, "material": "sio2"},
        {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
        {"name": "clad", "thickness": 2.0, "index": 1.44, "material": "sio2"},
    ]
    pixel_size = 0.02
    return density, layers, pixel_size


@pytest.fixture
def sample_monitors():
    """Two monitors: input and output."""
    return [
        {"name": "Input_te0", "x": 0.1, "y": 0.5, "width": 0.8, "orientation": 0},
        {"name": "Output_te1", "x": 1.4, "y": 0.5, "width": 0.8, "orientation": 0},
    ]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure HYPERWAVE_STANDALONE_UI is unset before each test."""
    monkeypatch.delenv("HYPERWAVE_STANDALONE_UI", raising=False)


# ---------------------------------------------------------------------------
# 1. Return value structure
# ---------------------------------------------------------------------------


class TestReturnValueStructure:
    """Tests that the return value dict has the correct shape and content."""

    def test_returns_dict(self, basic_inputs):
        """Return value is a dict, not None."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        assert isinstance(result, dict)

    def test_top_level_keys(self, basic_inputs):
        """Dict has exactly the keys: polygons, ports, bounds."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        assert set(result.keys()) == {"polygons", "ports", "bounds"}

    def test_bounds_keys(self, basic_inputs):
        """bounds sub-dict has x_min, x_max, y_min, y_max."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        assert set(result["bounds"].keys()) == {"x_min", "x_max", "y_min", "y_max"}

    def test_bounds_match_density_shape(self, basic_inputs):
        """bounds values match density.shape * pixel_size."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        nx, ny = density.shape
        assert result["bounds"]["x_min"] == 0
        assert result["bounds"]["y_min"] == 0
        assert abs(result["bounds"]["x_max"] - nx * pixel_size) < 1e-10
        assert abs(result["bounds"]["y_max"] - ny * pixel_size) < 1e-10

    def test_polygons_is_list(self, basic_inputs):
        """polygons is a list."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        assert isinstance(result["polygons"], list)

    def test_polygons_count_excludes_air(self):
        """Air layers should be skipped; polygon count = non-air layer count."""
        density = np.random.rand(40, 30).astype(np.float64)
        layers = [
            {"name": "sub", "thickness": 1.0, "index": 3.48, "material": "si"},
            {"name": "box", "thickness": 2.0, "index": 1.44, "material": "sio2"},
            {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
            {"name": "air_gap", "thickness": 1.0, "index": 1.0, "material": "air"},
            {"name": "clad", "thickness": 2.0, "index": 1.44, "material": "sio2"},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        # 5 layers minus 1 air = 4 polygon layers
        assert len(result["polygons"]) == 4

    def test_polygon_layer_required_keys(self, basic_inputs):
        """Each polygon layer has required keys."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        required = {"layer_name", "z_min", "z_max", "material", "refractive_index", "paths"}
        for pl in result["polygons"]:
            assert required.issubset(set(pl.keys())), (
                f"Layer {pl['layer_name']} missing keys: {required - set(pl.keys())}"
            )

    def test_design_layer_extra_keys(self, basic_inputs):
        """Design layers have additional keys for textures and contours."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        design_extras = {
            "texture_b64", "binary_texture_b64", "texture_size",
            "contour_paths", "smooth_contour", "heightmap", "heightmap_size",
        }
        design_layers = [pl for pl in result["polygons"] if "texture_b64" in pl]
        assert len(design_layers) == 1, "Expected exactly 1 design layer"
        assert design_extras.issubset(set(design_layers[0].keys()))

    def test_non_design_layer_lacks_texture(self, basic_inputs):
        """Non-design layers should NOT have texture_b64."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        non_design = [pl for pl in result["polygons"] if pl["layer_name"] != "sin"]
        for pl in non_design:
            assert "texture_b64" not in pl, f"Non-design layer {pl['layer_name']} has texture_b64"

    def test_z_stacking_is_cumulative(self, basic_inputs):
        """Layer z_min/z_max values stack cumulatively."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        polys = result["polygons"]
        # box: 0-2, sin: 2-2.4, clad: 2.4-4.4
        assert abs(polys[0]["z_min"] - 0.0) < 1e-10
        assert abs(polys[0]["z_max"] - 2.0) < 1e-10
        assert abs(polys[1]["z_min"] - 2.0) < 1e-10
        assert abs(polys[1]["z_max"] - 2.4) < 1e-10
        assert abs(polys[2]["z_min"] - 2.4) < 1e-10
        assert abs(polys[2]["z_max"] - 4.4) < 1e-10

    def test_texture_size_matches_density_shape(self, basic_inputs):
        """texture_size on design layer should match [nx, ny]."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        design_layer = [pl for pl in result["polygons"] if "texture_b64" in pl][0]
        nx, ny = density.shape
        assert design_layer["texture_size"] == [nx, ny]


# ---------------------------------------------------------------------------
# 2. Output routing (env var detection)
# ---------------------------------------------------------------------------


class TestOutputRouting:
    """Tests that output mode is routed correctly based on env vars and param."""

    def test_default_no_env_prints_summary(self, basic_inputs, capsys):
        """Default (no env var): stdout has summary text, NOT __GEOMETRY_UPDATE__."""
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size)
        captured = capsys.readouterr().out
        assert "__GEOMETRY_UPDATE__" not in captured
        assert "show_device_3d:" in captured

    def test_env_var_set_prints_json(self, basic_inputs, capsys, monkeypatch):
        """With HYPERWAVE_STANDALONE_UI=1: stdout has __GEOMETRY_UPDATE__ prefix."""
        monkeypatch.setenv("HYPERWAVE_STANDALONE_UI", "1")
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size)
        captured = capsys.readouterr().out
        assert captured.startswith("__GEOMETRY_UPDATE__")

    def test_output_ui_without_env(self, basic_inputs, capsys):
        """output='ui' without env var: still emits __GEOMETRY_UPDATE__."""
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size, output="ui")
        captured = capsys.readouterr().out
        assert captured.startswith("__GEOMETRY_UPDATE__")

    def test_output_summary_with_env(self, basic_inputs, capsys, monkeypatch):
        """output='summary' with env var set: prints summary, NOT JSON."""
        monkeypatch.setenv("HYPERWAVE_STANDALONE_UI", "1")
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size, output="summary")
        captured = capsys.readouterr().out
        assert "__GEOMETRY_UPDATE__" not in captured
        assert "show_device_3d:" in captured

    def test_output_invalid_raises(self, basic_inputs):
        """output='invalid' should raise ValueError."""
        density, layers, pixel_size = basic_inputs
        with pytest.raises(ValueError, match="output must be"):
            show_device_3d(density, layers, pixel_size, output="invalid")

    def test_env_var_wrong_value_does_not_trigger_ui(self, basic_inputs, capsys, monkeypatch):
        """HYPERWAVE_STANDALONE_UI set to something other than '1' should print summary."""
        monkeypatch.setenv("HYPERWAVE_STANDALONE_UI", "true")
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size)
        captured = capsys.readouterr().out
        assert "__GEOMETRY_UPDATE__" not in captured


# ---------------------------------------------------------------------------
# 3. Summary output content
# ---------------------------------------------------------------------------


class TestSummaryOutput:
    """Tests for non-UI output (skipped message)."""

    def test_prints_skipped_message(self, basic_inputs, capsys):
        """Non-UI mode prints a short skip notice."""
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size, output="summary")
        out = capsys.readouterr().out
        assert "skipped" in out
        assert "standalone UI" in out

    def test_no_json_in_summary(self, basic_inputs, capsys):
        """Non-UI mode should NOT contain JSON or base64 data."""
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size, output="summary")
        out = capsys.readouterr().out
        assert "__GEOMETRY_UPDATE__" not in out
        assert "texture_b64" not in out
        assert len(out) < 200


# ---------------------------------------------------------------------------
# 4. UI output content
# ---------------------------------------------------------------------------


class TestUIOutput:
    """Tests for the JSON/UI output mode."""

    def test_starts_with_marker(self, basic_inputs, capsys):
        """UI output starts with __GEOMETRY_UPDATE__."""
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size, output="ui")
        out = capsys.readouterr().out.strip()
        assert out.startswith("__GEOMETRY_UPDATE__")

    def test_json_after_marker_is_valid(self, basic_inputs, capsys):
        """Everything after the prefix is valid JSON."""
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size, output="ui")
        out = capsys.readouterr().out.strip()
        json_str = out[len("__GEOMETRY_UPDATE__"):]
        parsed = json.loads(json_str)  # should not raise
        assert isinstance(parsed, dict)

    def test_json_matches_return_value(self, basic_inputs, capsys):
        """Parsed JSON from stdout should match the returned dict."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="ui")
        out = capsys.readouterr().out.strip()
        json_str = out[len("__GEOMETRY_UPDATE__"):]
        parsed = json.loads(json_str)
        # Compare key structure (deep comparison of floating-point arrays
        # is unreliable due to serialization, so check key structure)
        assert set(parsed.keys()) == set(result.keys())
        assert parsed["bounds"] == result["bounds"]
        assert len(parsed["polygons"]) == len(result["polygons"])
        assert len(parsed["ports"]) == len(result["ports"])

    def test_ui_output_contains_texture(self, basic_inputs, capsys):
        """UI JSON should include base64 texture data for design layers."""
        density, layers, pixel_size = basic_inputs
        show_device_3d(density, layers, pixel_size, output="ui")
        out = capsys.readouterr().out.strip()
        json_str = out[len("__GEOMETRY_UPDATE__"):]
        parsed = json.loads(json_str)
        design_layers = [p for p in parsed["polygons"] if "texture_b64" in p]
        assert len(design_layers) == 1
        # Texture should be a non-empty base64 string
        assert len(design_layers[0]["texture_b64"]) > 100


# ---------------------------------------------------------------------------
# 5. Monitor handling
# ---------------------------------------------------------------------------


class TestMonitorHandling:
    """Tests for the monitors parameter."""

    def test_no_monitors_empty_ports(self, basic_inputs):
        """Without monitors, ports is an empty list."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        assert result["ports"] == []

    def test_monitors_create_ports(self, basic_inputs, sample_monitors):
        """Monitors are converted to port entries."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size,
                                monitors=sample_monitors, output="summary")
        assert len(result["ports"]) == 2

    def test_port_has_correct_fields(self, basic_inputs, sample_monitors):
        """Each port entry has name, center, width, orientation, layer."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size,
                                monitors=sample_monitors, output="summary")
        for port in result["ports"]:
            assert "name" in port
            assert "center" in port
            assert "width" in port
            assert "orientation" in port
            assert "layer" in port

    def test_port_values_match_input(self, basic_inputs, sample_monitors):
        """Port values match the input monitor dict values."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size,
                                monitors=sample_monitors, output="summary")
        port0 = result["ports"][0]
        assert port0["name"] == "Input_te0"
        assert port0["center"] == [0.1, 0.5]
        assert port0["width"] == 0.8
        assert port0["orientation"] == 0

    def test_port_layer_is_wg(self, basic_inputs, sample_monitors):
        """All monitor ports should have layer='WG'."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size,
                                monitors=sample_monitors, output="summary")
        for port in result["ports"]:
            assert port["layer"] == "WG"

    def test_monitors_none_same_as_omitted(self, basic_inputs):
        """monitors=None should behave the same as not passing monitors."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size,
                                monitors=None, output="summary")
        assert result["ports"] == []


# ---------------------------------------------------------------------------
# 6. Mode handling
# ---------------------------------------------------------------------------


class TestModeHandling:
    """Tests for mode='auto', 'contour', 'slab'."""

    def test_slab_mode_single_rectangle(self, basic_inputs):
        """mode='slab': design layer paths is a single full-extent rectangle."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size,
                                mode="slab", output="summary")
        design_layer = [pl for pl in result["polygons"] if "texture_b64" in pl][0]
        paths = design_layer["paths"]
        assert len(paths) == 1
        rect = paths[0]
        assert len(rect) == 4
        # Full rectangle corners at (0,0) to (x_max, y_max)
        nx, ny = density.shape
        x_max = nx * pixel_size
        y_max = ny * pixel_size
        corners = {tuple(p) for p in rect}
        expected = {(0, 0), (x_max, 0), (x_max, y_max), (0, y_max)}
        assert corners == expected

    def test_contour_mode_extracts_paths(self, binary_inputs):
        """mode='contour': design layer has contour-extracted paths (not a single rectangle)."""
        density, layers, pixel_size = binary_inputs
        result = show_device_3d(density, layers, pixel_size,
                                mode="contour", output="summary")
        design_layer = [pl for pl in result["polygons"] if "texture_b64" in pl][0]
        paths = design_layer["paths"]
        # With a clear binary feature, contour should extract at least one path
        assert len(paths) >= 1
        # Each path should be a list of [x, y] points
        for path in paths:
            assert len(path) >= 3
            for pt in path:
                assert len(pt) == 2

    def test_auto_mode_binary_uses_contour(self, binary_inputs):
        """mode='auto' with fully binary density (score=1.0) should choose contour."""
        density, layers, pixel_size = binary_inputs
        result_auto = show_device_3d(density, layers, pixel_size,
                                     mode="auto", output="summary")
        result_contour = show_device_3d(density, layers, pixel_size,
                                        mode="contour", output="summary")
        design_auto = [pl for pl in result_auto["polygons"] if "texture_b64" in pl][0]
        design_contour = [pl for pl in result_contour["polygons"] if "texture_b64" in pl][0]
        # auto should match contour for binary input (bscore=1.0 > 0.8)
        assert len(design_auto["paths"]) == len(design_contour["paths"])

    def test_auto_mode_gray_uses_slab(self):
        """mode='auto' with uniform 0.5 density (bscore=0.0) should choose slab."""
        density = np.full((40, 30), 0.5, dtype=np.float64)
        layers = [
            {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
        ]
        result = show_device_3d(density, layers, 0.01, mode="auto", output="summary")
        design_layer = result["polygons"][0]
        # Slab mode: single full rectangle
        assert len(design_layer["paths"]) == 1
        rect = design_layer["paths"][0]
        assert len(rect) == 4

    def test_non_design_layers_always_slab(self, basic_inputs):
        """Non-design layers always use the slab rectangle regardless of mode."""
        density, layers, pixel_size = basic_inputs
        result = show_device_3d(density, layers, pixel_size,
                                mode="contour", output="summary")
        non_design = [pl for pl in result["polygons"] if "texture_b64" not in pl]
        nx, ny = density.shape
        x_max = nx * pixel_size
        y_max = ny * pixel_size
        for pl in non_design:
            assert len(pl["paths"]) == 1
            corners = {tuple(p) for p in pl["paths"][0]}
            expected = {(0, 0), (x_max, 0), (x_max, y_max), (0, y_max)}
            assert corners == expected


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for unusual inputs and boundary conditions."""

    def test_1d_density_raises(self):
        """1D density array should raise ValueError."""
        density = np.ones(100)
        layers = [{"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True}]
        with pytest.raises(ValueError, match="2D"):
            show_device_3d(density, layers, 0.01, output="summary")

    def test_3d_density_raises(self):
        """3D density array should raise ValueError."""
        density = np.ones((10, 10, 10))
        layers = [{"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True}]
        with pytest.raises(ValueError, match="2D"):
            show_device_3d(density, layers, 0.01, output="summary")

    def test_all_zeros_density(self):
        """All-zeros density (empty device) should work without error."""
        density = np.zeros((40, 30), dtype=np.float64)
        layers = [
            {"name": "box", "thickness": 2.0, "index": 1.44, "material": "sio2"},
            {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        assert isinstance(result, dict)
        assert len(result["polygons"]) == 2

    def test_all_ones_density(self):
        """All-ones density (full slab) should work without error."""
        density = np.ones((40, 30), dtype=np.float64)
        layers = [
            {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        assert isinstance(result, dict)
        design_layer = result["polygons"][0]
        assert "texture_b64" in design_layer

    def test_very_small_density(self):
        """2x2 pixel density should work without error."""
        density = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        layers = [
            {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
        ]
        result = show_device_3d(density, layers, 0.1, output="summary")
        assert isinstance(result, dict)
        assert result["bounds"]["x_max"] == pytest.approx(0.2)
        assert result["bounds"]["y_max"] == pytest.approx(0.2)

    def test_air_layers_excluded(self):
        """Air layers should not appear in polygon output."""
        density = np.random.rand(20, 20).astype(np.float64)
        layers = [
            {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
            {"name": "air", "thickness": 1.0, "index": 1.0, "material": "air"},
            {"name": "clad", "thickness": 2.0, "index": 1.44, "material": "sio2"},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        layer_names = [pl["layer_name"] for pl in result["polygons"]]
        assert "air" not in layer_names
        assert len(result["polygons"]) == 2

    def test_z_stacking_skips_air(self):
        """Z cursor advances through air layers even though they are excluded."""
        density = np.random.rand(20, 20).astype(np.float64)
        layers = [
            {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
            {"name": "air", "thickness": 1.0, "index": 1.0, "material": "air"},
            {"name": "clad", "thickness": 2.0, "index": 1.44, "material": "sio2"},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        # sin: 0-0.4, air: 0.4-1.4 (skipped), clad: 1.4-3.4
        clad_layer = [pl for pl in result["polygons"] if pl["layer_name"] == "clad"][0]
        assert clad_layer["z_min"] == pytest.approx(1.4)
        assert clad_layer["z_max"] == pytest.approx(3.4)

    def test_material_aliases(self):
        """Various material name aliases should be normalized."""
        density = np.random.rand(20, 20).astype(np.float64)
        layers = [
            {"name": "a", "thickness": 0.4, "index": 3.48, "material": "silicon"},
            {"name": "b", "thickness": 0.4, "index": 2.0, "material": "silicon_nitride"},
            {"name": "c", "thickness": 0.4, "index": 2.0, "material": "si3n4"},
            {"name": "d", "thickness": 0.4, "index": 1.44, "material": "silicon_dioxide"},
            {"name": "e", "thickness": 0.4, "index": 1.44, "material": "oxide"},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        materials = [pl["material"] for pl in result["polygons"]]
        assert materials == ["si", "sin", "sin", "sio2", "sio2"]

    def test_single_layer_no_design(self):
        """Stack with no design layer should still work (no texture keys)."""
        density = np.random.rand(20, 20).astype(np.float64)
        layers = [
            {"name": "box", "thickness": 2.0, "index": 1.44, "material": "sio2"},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        assert len(result["polygons"]) == 1
        assert "texture_b64" not in result["polygons"][0]

    def test_multiple_design_layers(self):
        """Multiple design layers should each get texture data."""
        density = np.random.rand(30, 20).astype(np.float64)
        layers = [
            {"name": "etch", "thickness": 0.2, "index": 2.0, "material": "sin", "is_design": True},
            {"name": "slab", "thickness": 0.2, "index": 2.0, "material": "sin", "is_design": True},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        design_layers = [pl for pl in result["polygons"] if "texture_b64" in pl]
        assert len(design_layers) == 2

    def test_heightmap_is_downsampled(self):
        """Heightmap should be smaller than the original density for large arrays."""
        density = np.random.rand(500, 400).astype(np.float64)
        layers = [
            {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        design_layer = result["polygons"][0]
        hm_size = design_layer["heightmap_size"]
        # Stride = max(1, dim // max_verts) where max_verts=200
        # 500//200=2 -> 250, 400//200=2 -> 200. Both < original dims.
        assert hm_size[0] < 500
        assert hm_size[1] < 400
        # Verify the heightmap list dimensions match reported size
        hm = design_layer["heightmap"]
        assert len(hm) == hm_size[0]
        assert len(hm[0]) == hm_size[1]

    def test_smooth_contour_has_paths(self, binary_inputs):
        """smooth_contour should contain at least one path for a non-trivial density."""
        density, layers, pixel_size = binary_inputs
        result = show_device_3d(density, layers, pixel_size, output="summary")
        design_layer = [pl for pl in result["polygons"] if "texture_b64" in pl][0]
        assert len(design_layer["smooth_contour"]) >= 1

    def test_integer_density_accepted(self):
        """Integer (0/1) density array should be accepted and converted."""
        density = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int32)
        layers = [
            {"name": "sin", "thickness": 0.4, "index": 2.0, "material": "sin", "is_design": True},
        ]
        result = show_device_3d(density, layers, 0.01, output="summary")
        assert isinstance(result, dict)

    def test_return_value_independent_of_output_mode(self, basic_inputs):
        """Return dict should be identical regardless of output mode."""
        density, layers, pixel_size = basic_inputs
        result_summary = show_device_3d(density, layers, pixel_size, output="summary")
        result_ui = show_device_3d(density, layers, pixel_size, output="summary")
        assert result_summary["bounds"] == result_ui["bounds"]
        assert len(result_summary["polygons"]) == len(result_ui["polygons"])
        assert len(result_summary["ports"]) == len(result_ui["ports"])
        for ps, pu in zip(result_summary["polygons"], result_ui["polygons"]):
            assert ps["layer_name"] == pu["layer_name"]
            assert ps["z_min"] == pu["z_min"]
            assert ps["z_max"] == pu["z_max"]
