"""Tests for the new inverse design API (objectives, types, pipeline, etc.)."""

import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Objectives (tree construction + serialization, NO JAX)
# ---------------------------------------------------------------------------

class TestObjectives:
    def test_mode_coupling_tree(self):
        from hyperwave_community.objectives import mode_coupling
        mf = np.random.randn(1, 6, 8, 8).astype(np.complex64)
        obj = mode_coupling(mf, 1.0, 0.5, "wg")
        assert obj is not None

    def test_operator_overloading(self):
        from hyperwave_community import objectives as obj
        mf = np.random.randn(1, 6, 4, 4).astype(np.complex64)
        e = obj.mode_coupling(mf, 1.0, 0.5, "wg")
        p = obj.power("wg")
        loss = -(0.7 * e + 0.3 * p)
        assert isinstance(loss, obj.Objective)

    def test_serialize_produces_dict(self):
        from hyperwave_community.objectives import mode_coupling
        mf = np.random.randn(1, 6, 4, 4).astype(np.complex64)
        loss = -mode_coupling(mf, 1.0, 0.5, "wg")
        spec, arrays = loss.serialize()
        assert isinstance(spec, dict)
        assert spec["type"] == "neg"
        assert len(arrays) == 1

    def test_array_dedup(self):
        from hyperwave_community.objectives import mode_coupling
        mf = np.random.randn(1, 6, 4, 4).astype(np.complex64)
        e1 = mode_coupling(mf, 1.0, 0.5, "p1")
        e2 = mode_coupling(mf, 1.0, 0.5, "p2")
        _, arrays = (e1 + e2).serialize()
        assert len(arrays) == 1

    def test_min_of(self):
        from hyperwave_community import objectives as obj
        mf = np.random.randn(1, 6, 4, 4).astype(np.complex64)
        terms = [obj.mode_coupling(mf, 1.0, 0.5, "wg", freq_idx=i) for i in range(3)]
        loss = obj.min_of(*terms)
        spec, _ = loss.serialize()
        assert spec["type"] == "min_of"
        assert len(spec["terms"]) == 3

    def test_custom_field_math(self):
        from hyperwave_community import objectives as obj
        ey = obj.field("Ey", "focus")
        hz = obj.field("Hz", "focus")
        loss = obj.sum_spatial(obj.real(ey * obj.conj(hz)))
        spec, _ = loss.serialize()
        assert spec["type"] == "sum_spatial"

    def test_repr(self):
        from hyperwave_community.objectives import mode_coupling
        mf = np.random.randn(1, 6, 4, 4).astype(np.complex64)
        loss = -mode_coupling(mf, 1.0, 0.5, "wg")
        r = repr(loss)
        assert "mode_coupling" in r

    def test_field_validation(self):
        from hyperwave_community.objectives import field
        field("Ey", "wg")
        with pytest.raises(ValueError):
            field("Bx", "wg")

    def test_intensity_validation(self):
        from hyperwave_community.objectives import intensity
        intensity("Ez", "wg")
        with pytest.raises(ValueError):
            intensity("Hx", "wg")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TestTypes:
    def test_design_properties(self):
        from hyperwave_community.types import Design
        d = Design(thetas={"etch": np.ones((50, 50), dtype=np.float32)},
                   density_radii={"etch": 6}, efficiency=0.75)
        assert d.shape == (50, 50)
        assert d.layer_names == ["etch"]
        assert d.design_mask().all()

    def test_design_multi_layer(self):
        from hyperwave_community.types import Design
        d = Design(thetas={"etch": np.ones((40, 40)), "slab": np.zeros((40, 40))},
                   density_radii={"etch": 4, "slab": 4})
        assert len(d.layer_names) == 2

    def test_drc_report_pass(self):
        from hyperwave_community.types import DrcReport
        r = DrcReport(cd_violations=5, cd_pct=0.3, gap_violations=3, gap_pct=0.2,
                      binarization_score=0.95, disk_radius=3, min_feature_nm=122.5,
                      min_gap_nm=122.5, design_pixels=3600)
        assert r.passed is True
        assert "PASS" in r.status

    def test_drc_report_fail(self):
        from hyperwave_community.types import DrcReport
        r = DrcReport(cd_violations=100, cd_pct=5.0, gap_violations=50, gap_pct=2.5,
                      binarization_score=0.8, disk_radius=3, min_feature_nm=122.5,
                      min_gap_nm=122.5, design_pixels=2000)
        assert r.passed is False


# ---------------------------------------------------------------------------
# Pipeline (local functions)
# ---------------------------------------------------------------------------

class TestSurgery:
    def test_returns_design(self):
        from hyperwave_community.types import Design
        from hyperwave_community.pipeline import surgery
        np.random.seed(42)
        d = Design(thetas={"etch": np.random.rand(80, 80).astype(np.float32)},
                   density_radii={"etch": 4})
        result = surgery(d)
        assert isinstance(result, Design)
        assert result.phase == "surgery"
        assert result.removed_islands >= 0

    def test_uniform_no_surgery(self):
        from hyperwave_community.types import Design
        from hyperwave_community.pipeline import surgery
        d = Design(thetas={"etch": np.ones((60, 60), dtype=np.float32)},
                   density_radii={"etch": 4})
        result = surgery(d)
        assert result.removed_islands == 0
        assert result.filled_holes == 0


class TestCheckDrc:
    def test_returns_report(self):
        from hyperwave_community.types import Design
        from hyperwave_community.pipeline import check_drc
        np.random.seed(42)
        d = Design(thetas={"etch": np.random.rand(80, 80).astype(np.float32)},
                   density_radii={"etch": 4})
        r = check_drc(d, disk_radius=2)
        assert hasattr(r, "cd_pct")
        assert hasattr(r, "passed")
        assert hasattr(r, "status")

    def test_uniform_passes(self):
        from hyperwave_community.types import Design
        from hyperwave_community.pipeline import check_drc
        d = Design(thetas={"etch": np.ones((60, 60), dtype=np.float32)},
                   density_radii={"etch": 4})
        r = check_drc(d)
        assert r.passed is True


class TestExportGds:
    def test_produces_file(self):
        from hyperwave_community.types import Design
        from hyperwave_community.pipeline import export_gds
        np.random.seed(42)
        d = Design(thetas={"etch": np.random.rand(60, 60).astype(np.float32)},
                   density_radii={"etch": 4})
        with tempfile.TemporaryDirectory() as td:
            path = export_gds(d, filename=os.path.join(td, "test.gds"))
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# LayerStack
# ---------------------------------------------------------------------------

class TestBuildDevice:
    def test_build(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "box", "thickness": 2.0, "index": 1.44},
                {"name": "etch", "thickness": 0.11, "index": 3.48,
                 "design": True, "density_radius": 6},
                {"name": "clad", "thickness": 2.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        assert len(device.shape) == 3
        assert len(device.design_layers_info) == 1
        assert device.design_layers_info[0]["name"] == "etch"
        assert device.design_layers_info[0]["density_radius"] == 6

    def test_recipe_params(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "clad", "thickness": 1.0, "index": 1.44},
                {"name": "core", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 8},
                {"name": "clad2", "thickness": 1.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=80,
        )
        assert "grid_shape" in device.recipe_params
        assert "layers_template" in device.recipe_params

    def test_density_radius_required(self):
        from hyperwave_community.device import build_device
        with pytest.raises(ValueError, match="density_radius is required"):
            build_device(
                layers=[{"name": "etch", "thickness": 0.11, "index": 3.48,
                         "design": True}],
                grid=0.035, wavelength=1.55, nx=100,
            )

    def test_density_eta_validated(self):
        from hyperwave_community.device import build_device
        with pytest.raises(ValueError, match="density_eta"):
            build_device(
                layers=[{"name": "etch", "thickness": 0.11, "index": 3.48,
                         "design": True, "density_radius": 6, "density_eta": 0.9}],
                grid=0.035, wavelength=1.55, nx=100,
            )

    def test_layers_template_format(self):
        """layers_template must match Modal optimizer's expected format."""
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "box", "thickness": 2.0, "index": 1.44},
                {"name": "etch", "thickness": 0.11, "index": 3.48,
                 "design": True, "density_radius": 6},
                {"name": "slab", "thickness": 0.11, "index": 3.48,
                 "design": True, "density_radius": 6},
                {"name": "clad", "thickness": 2.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        lt = device.recipe_params["layers_template"]
        assert len(lt) == 4

        # Fixed layers
        assert lt[0]["layer_type"] == "slab"
        assert "permittivity" in lt[0]["params"]
        assert "thickness" in lt[0]["params"]
        assert lt[3]["layer_type"] == "slab"

        # Design layers indexed sequentially
        assert lt[1]["layer_type"] == "design_0"
        assert isinstance(lt[1]["params"]["permittivity"], tuple)
        assert lt[2]["layer_type"] == "design_1"

    def test_waveguide_mask_custom(self):
        from hyperwave_community.device import build_device
        wg = np.zeros((100, 100), dtype=bool)
        wg[:, 40:60] = True
        device = build_device(
            layers=[
                {"name": "clad", "thickness": 1.0, "index": 1.44},
                {"name": "etch", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 6, "waveguide_mask": wg},
                {"name": "clad2", "thickness": 1.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        mask = device.design_layers_info[0]["waveguide_mask"]
        assert mask.sum() > 0  # not all zeros


# ---------------------------------------------------------------------------
# Waveguide mode
# ---------------------------------------------------------------------------

class TestWaveguideMode:
    def test_solve(self):
        from hyperwave_community.waveguide_mode import solve_waveguide_mode
        mode_field, n_eff = solve_waveguide_mode(
            grid=0.050, waveguide_width=0.5, waveguide_height=0.22,
            n_core=3.48, n_clad=1.44, wavelength=1.55,
            cross_section_size=40,
        )
        assert mode_field.ndim == 5
        assert 2.0 < n_eff < 3.5

    def test_higher_order_te1(self):
        from hyperwave_community.waveguide_mode import solve_waveguide_mode
        mf0, neff0 = solve_waveguide_mode(
            grid=0.030, waveguide_width=2.0, waveguide_height=0.4,
            n_core=2.0, n_clad=1.44, wavelength=1.55,
            mode_number=0, cross_section_size=80,
        )
        mf2, neff2 = solve_waveguide_mode(
            grid=0.030, waveguide_width=2.0, waveguide_height=0.4,
            n_core=2.0, n_clad=1.44, wavelength=1.55,
            mode_number=2, cross_section_size=80,
        )
        assert neff0 > neff2, "TE0 should have higher n_eff than TE1"
        assert neff2 > 1.44, "TE1 should be guided (n_eff > n_clad)"


# ---------------------------------------------------------------------------
# SDK gap fixes
# ---------------------------------------------------------------------------

class TestVerticalRadius:
    def test_build_device_vertical_radius(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "clad", "thickness": 1.0, "index": 1.44},
                {"name": "core", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 4},
                {"name": "clad2", "thickness": 1.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
            vertical_radius=2,
        )
        assert device.vertical_radius == 2

    def test_default_vertical_radius_is_2(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "clad", "thickness": 1.0, "index": 1.44},
                {"name": "core", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 4},
                {"name": "clad2", "thickness": 1.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        assert device.vertical_radius == 2


class TestGetSourceConfig:
    def test_returns_bounds(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "clad", "thickness": 2.5, "index": 1.44},
                {"name": "sin", "thickness": 0.4, "index": 2.0,
                 "design": True, "density_radius": 8,
                 "waveguide_width": 2.0},
                {"name": "clad2", "thickness": 2.5, "index": 1.44},
            ],
            grid=0.030, wavelength=1.55, nx=200, ny=200,
        )
        cfg = device.get_source_config("sin", x_position=50)
        assert "source_position" in cfg
        assert "perpendicular_bounds" in cfg
        assert "z_bounds" in cfg
        yl, yh = cfg["perpendicular_bounds"]
        assert yh > yl
        zl, zh = cfg["z_bounds"]
        assert zh > zl

    def test_missing_waveguide_width_raises(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "clad", "thickness": 1.0, "index": 1.44},
                {"name": "core", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 4},
                {"name": "clad2", "thickness": 1.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        with pytest.raises(ValueError, match="waveguide_y_center"):
            device.get_source_config("core")


class TestCladdingInference:
    """Tests for eps_clad inference from adjacent layers."""

    def test_standard_3layer_stack(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "box", "thickness": 2.0, "index": 1.44},
                {"name": "etch", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 6},
                {"name": "clad", "thickness": 2.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        dl = device.design_layers_info[0]
        assert dl["eps_range"] == pytest.approx((1.44**2, 3.48**2), rel=1e-4)

    def test_explicit_cladding_index_overrides(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "box", "thickness": 2.0, "index": 1.44},
                {"name": "etch", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 6, "cladding_index": 1.0},
                {"name": "clad", "thickness": 2.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        dl = device.design_layers_info[0]
        assert dl["eps_range"] == pytest.approx((1.0, 3.48**2), rel=1e-4)

    def test_two_design_layers_different_cladding(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "clad_top", "thickness": 1.0, "index": 1.44},
                {"name": "sin", "thickness": 0.4, "index": 2.0,
                 "design": True, "density_radius": 4},
                {"name": "gap", "thickness": 0.2, "index": 1.44},
                {"name": "si", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 4},
                {"name": "box", "thickness": 2.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        sin_dl = [d for d in device.design_layers_info if d["name"] == "sin"][0]
        si_dl = [d for d in device.design_layers_info if d["name"] == "si"][0]
        assert sin_dl["eps_range"][0] == pytest.approx(1.44**2, rel=1e-4)
        assert si_dl["eps_range"][0] == pytest.approx(1.44**2, rel=1e-4)

    def test_adjacent_design_layers_infer_from_further_neighbor(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "box", "thickness": 2.0, "index": 1.44},
                {"name": "layer1", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 4},
                {"name": "layer2", "thickness": 0.22, "index": 2.0,
                 "design": True, "density_radius": 4},
                {"name": "clad", "thickness": 2.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        l1 = [d for d in device.design_layers_info if d["name"] == "layer1"][0]
        l2 = [d for d in device.design_layers_info if d["name"] == "layer2"][0]
        assert l1["eps_range"][0] == pytest.approx(1.44**2, rel=1e-4)
        assert l2["eps_range"][0] == pytest.approx(1.44**2, rel=1e-4)

    def test_all_design_layers_warns_and_falls_back(self):
        import warnings
        from hyperwave_community.device import build_device
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            device = build_device(
                layers=[
                    {"name": "a", "thickness": 0.22, "index": 3.48,
                     "design": True, "density_radius": 4},
                    {"name": "b", "thickness": 0.22, "index": 2.0,
                     "design": True, "density_radius": 4},
                ],
                grid=0.035, wavelength=1.55, nx=100,
            )
        clad_warnings = [x for x in w if "no adjacent non-design layer" in str(x.message)]
        assert len(clad_warnings) >= 2
        for dl in device.design_layers_info:
            assert dl["eps_range"][0] == pytest.approx(1.0, rel=1e-4)

    def test_layers_template_matches_design_info(self):
        from hyperwave_community.device import build_device
        device = build_device(
            layers=[
                {"name": "box", "thickness": 2.0, "index": 1.44},
                {"name": "etch", "thickness": 0.22, "index": 3.48,
                 "design": True, "density_radius": 6},
                {"name": "clad", "thickness": 2.0, "index": 1.44},
            ],
            grid=0.035, wavelength=1.55, nx=100,
        )
        lt = device.recipe_params["layers_template"]
        design_lt = [l for l in lt if l["layer_type"].startswith("design_")]
        dl = device.design_layers_info[0]
        assert design_lt[0]["params"]["permittivity"] == pytest.approx(
            tuple(dl["eps_range"]), rel=1e-4)


class TestAbsorberParams:
    def test_flat_device(self):
        from hyperwave_community.absorption import absorber_params
        ap = absorber_params(1.55, 0.030, (932, 366, 179), device_type="flat")
        assert "absorption_widths" in ap
        assert "abs_coeff" in ap
        assert len(ap["absorption_widths"]) == 3
        assert ap["abs_coeff"] > 0

    def test_grating_device(self):
        from hyperwave_community.absorption import absorber_params
        ap = absorber_params(1.55, 0.030, (932, 366, 179), device_type="grating")
        assert "absorption_widths" in ap
        assert "abs_coeff" in ap
        assert ap["abs_coeff"] > 0

    def test_flat_vs_grating_differ(self):
        from hyperwave_community.absorption import absorber_params
        flat = absorber_params(1.55, 0.030, (932, 366, 179), device_type="flat")
        grating = absorber_params(1.55, 0.030, (932, 366, 179), device_type="grating")
        assert flat["absorption_widths"] != grating["absorption_widths"]

    def test_invalid_device_type(self):
        from hyperwave_community.absorption import absorber_params
        with pytest.raises(ValueError):
            absorber_params(1.55, 0.030, (100, 100, 100), device_type="invalid")


class TestComputeModeCrossPower:
    def test_six_component_mode(self):
        from hyperwave_community.pipeline import compute_mode_cross_power
        mf = np.random.randn(1, 6, 1, 20, 20).astype(np.complex64)
        p = compute_mode_cross_power(mf)
        assert p > 0

    def test_three_component_fallback(self):
        from hyperwave_community.pipeline import compute_mode_cross_power
        mf = np.random.randn(1, 3, 1, 20, 20).astype(np.complex64)
        p = compute_mode_cross_power(mf)
        assert p == 1.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_round_trip(self):
        from hyperwave_community.types import Design, OptimizationResult
        from hyperwave_community.checkpoint import load_checkpoint

        d = Design(thetas={"etch": np.random.rand(30, 30).astype(np.float32)},
                   density_radii={"etch": 6}, efficiency=0.5, step=25)
        r = OptimizationResult(design=d, history=[], phase="freeform", n_steps=25,
                               schedule_config={"beta_init": 1.0}, n_steps_planned=100)
        with tempfile.TemporaryDirectory() as td:
            path = r.save(os.path.join(td, "run"))
            loaded = load_checkpoint(path)
            np.testing.assert_array_equal(loaded.thetas["etch"], d.thetas["etch"])
            assert loaded.schedule_config["beta_init"] == 1.0


# ---------------------------------------------------------------------------
# run_optimization() is retired (legacy single-layer cloud API)
# ---------------------------------------------------------------------------

class TestRunOptimizationRetired:
    """The legacy run_optimization() cloud path is dead: its GPU backend
    (the structure_spec streaming function behind /inverse_design_start) was
    never deployed. It must now fail loudly and point users to optimize()."""

    def _minimal_kwargs(self):
        return dict(
            theta=np.zeros((4, 4), dtype=np.float32),
            source_field=np.zeros((1, 6, 1, 4, 1), dtype=np.complex64),
            source_offset=(0, 0, 0),
            freq_band=(0.1, 0.1, 1),
            structure_spec={},
            loss_monitor_shape=(1, 4, 1), loss_monitor_offset=(0, 0, 0),
            design_monitor_shape=(4, 4, 1), design_monitor_offset=(0, 0, 0),
            mode_field=np.zeros((1, 6, 1, 4, 1), dtype=np.complex64),
            input_power=1.0, mode_cross_power=1.0,
        )

    def test_run_optimization_raises_on_use(self):
        """Iterating the generator raises RuntimeError mentioning optimize()."""
        from hyperwave_community.api_client import run_optimization
        gen = run_optimization(**self._minimal_kwargs())
        with pytest.raises(RuntimeError, match="optimize"):
            next(gen)

    def test_message_is_actionable(self):
        """Error says it is retired and names the replacement optimize() call."""
        from hyperwave_community.api_client import run_optimization
        with pytest.raises(RuntimeError) as exc:
            list(run_optimization(**self._minimal_kwargs()))
        msg = str(exc.value).lower()
        assert ("retired" in msg) or ("no longer" in msg)
        assert "optimize(" in msg

    def test_retired_before_any_network_or_validation(self):
        """Must raise before touching API config / network -- works with no key
        configured and even with an otherwise-invalid loss spec."""
        from hyperwave_community.api_client import run_optimization
        kwargs = self._minimal_kwargs()
        kwargs.pop("mode_field"); kwargs.pop("input_power"); kwargs.pop("mode_cross_power")
        # No loss spec at all would normally ValueError; retirement preempts it.
        with pytest.raises(RuntimeError, match="optimize"):
            next(run_optimization(**kwargs))
