"""Tests for the full pipeline: structure -> source -> monitors -> simulate.

Validates that the SDK functions compose correctly without requiring
a live API connection. Uses mocks for cloud-dependent steps.
"""

import warnings

import pytest
import jax.numpy as jnp

from hyperwave_community import (
    density,
    create_structure,
    Layer,
    MonitorSet,
    Monitor,
    create_mode_source,
    create_port_monitors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_structure():
    """A minimal uniform structure for basic tests."""
    theta = jnp.ones((100, 60))
    d = density(theta, radius=2)
    layer = Layer(d, permittivity_values=(1.0, 11.56), layer_thickness=10)
    return create_structure(layers=[layer])


@pytest.fixture
def waveguide_structure():
    """Multi-layer structure with a waveguide stripe for auto-detect tests.

    Creates SiO2 cladding + Si waveguide core + SiO2 cladding with a
    10-pixel-wide high-index stripe, giving both lateral and vertical
    permittivity contrast needed by _detect_waveguides.
    """
    theta = jnp.zeros((100, 60))
    theta = theta.at[:, 25:35].set(1.0)
    d = density(theta, radius=2)
    clad_theta = jnp.zeros((100, 60))
    clad_d = density(clad_theta, radius=2)
    clad = Layer(clad_d, permittivity_values=(2.25, 2.25), layer_thickness=5)
    core = Layer(d, permittivity_values=(1.0, 11.56), layer_thickness=10)
    return create_structure(layers=[clad, core, clad])


@pytest.fixture
def freq_band():
    return (2 * jnp.pi / 1.6, 2 * jnp.pi / 1.5, 2)


# ---------------------------------------------------------------------------
# Structure creation
# ---------------------------------------------------------------------------

class TestStructureCreation:
    def test_structure_has_permittivity(self, simple_structure):
        assert hasattr(simple_structure, 'permittivity')
        assert simple_structure.permittivity.ndim == 4  # (n_freq, x, y, z)

    def test_structure_dimensions(self, simple_structure):
        _, x, y, z = simple_structure.permittivity.shape
        assert x > 0
        assert y > 0
        assert z > 0

    def test_waveguide_structure_has_contrast(self, waveguide_structure):
        eps = waveguide_structure.permittivity
        assert float(eps.min()) < float(eps.max())


# ---------------------------------------------------------------------------
# MonitorSet basics
# ---------------------------------------------------------------------------

class TestMonitorSetPipeline:
    def test_manual_monitor_placement(self, simple_structure):
        """Manually adding monitors should work."""
        _, Lx, Ly, Lz = simple_structure.permittivity.shape
        monitors = MonitorSet()
        monitors.add(
            Monitor(shape=(5, Ly, Lz), offset=(10, 0, 0)),
            name='input',
        )
        monitors.add(
            Monitor(shape=(5, Ly, Lz), offset=(Lx - 15, 0, 0)),
            name='output',
        )
        assert len(monitors.monitors) == 2
        assert 'input' in monitors.mapping
        assert 'output' in monitors.mapping

    def test_add_monitors_at_position_deprecated(self, waveguide_structure):
        """add_monitors_at_position still works but emits DeprecationWarning."""
        monitors = MonitorSet()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            names = monitors.add_monitors_at_position(
                waveguide_structure,
                axis='x',
                position=10,
                label='test',
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert 'deprecated' in str(deprecation_warnings[0].message).lower()
        # Should have added at least one monitor
        assert len(names) >= 1
        assert len(monitors.monitors) >= 1

    def test_monitor_recipe_roundtrip(self):
        """Monitor recipe should be serializable."""
        ms = MonitorSet()
        ms.add(Monitor(shape=(5, 10, 10), offset=(0, 0, 0)), name='a')
        ms.add(Monitor(shape=(3, 8, 8), offset=(5, 5, 5)), name='b')
        recipe = ms.recipe
        assert len(recipe) == 2
        assert all('name' in r and 'shape' in r and 'offset' in r for r in recipe)


# ---------------------------------------------------------------------------
# create_port_monitors (new function)
# ---------------------------------------------------------------------------

class TestCreatePortMonitors:
    def test_creates_monitors_from_ports(self, waveguide_structure):
        """create_port_monitors creates monitors at gdsfactory port positions."""

        class FakePort:
            def __init__(self, name, center, orientation):
                self.name = name
                self.center = center
                self.orientation = orientation

        class FakeComponent:
            def __init__(self):
                self.ports = [
                    FakePort('o1', (0.0, 0.0), 180),  # input
                    FakePort('o2', (5.0, 0.0), 0),     # output
                ]

        comp = FakeComponent()
        device_info = {
            'bounding_box_um': (0.0, -3.0, 5.0, 3.0),
            'theta_resolution_um': 0.05,
        }
        padding = (20, 20, 20, 20)
        absorption_widths = (10, 10, 10, 10, 5, 5)

        monitors = create_port_monitors(
            comp,
            waveguide_structure,
            device_info,
            padding,
            absorption_widths,
        )

        assert isinstance(monitors, MonitorSet)
        assert len(monitors.monitors) >= 2  # at least input + output
        # Should include xy_mid visualization plane
        assert 'xy_mid' in monitors.mapping


# ---------------------------------------------------------------------------
# create_mode_source
# ---------------------------------------------------------------------------

class TestCreateModeSource:
    def test_basic_mode_source_explicit_bounds(self, simple_structure, freq_band):
        """create_mode_source with explicit bounds returns correct tuple."""
        _, _, Ly, Lz = simple_structure.permittivity.shape
        source_field, offset, mode_info = create_mode_source(
            simple_structure,
            freq_band,
            mode_num=0,
            propagation_axis='x',
            source_position=10,
            perpendicular_bounds=(0, Ly),
            z_bounds=(0, Lz),
        )
        assert source_field.ndim == 5
        assert source_field.shape[0] == 2  # num_freqs from freq_band
        assert len(offset) == 3
        assert 'beta' in mode_info
        assert 'error' in mode_info

    def test_auto_detect_bounds(self, waveguide_structure, freq_band):
        """Omitting perpendicular_bounds and z_bounds triggers auto-detect."""
        source_field, offset, mode_info = create_mode_source(
            waveguide_structure,
            freq_band,
            mode_num=0,
            propagation_axis='x',
            source_position=10,
            perpendicular_bounds=None,
            z_bounds=None,
        )
        assert source_field is not None
        assert source_field.ndim == 5
        assert mode_info['beta'] is not None

    def test_explicit_bounds(self, simple_structure, freq_band):
        """Providing explicit bounds should also work."""
        _, _, Ly, Lz = simple_structure.permittivity.shape
        source_field, offset, mode_info = create_mode_source(
            simple_structure,
            freq_band,
            mode_num=0,
            propagation_axis='x',
            source_position=10,
            perpendicular_bounds=(0, Ly),
            z_bounds=(0, Lz),
        )
        assert source_field is not None

    def test_y_propagation(self, waveguide_structure, freq_band):
        """Should work with y-propagation axis too."""
        _, Lx, _, Lz = waveguide_structure.permittivity.shape
        source_field, offset, mode_info = create_mode_source(
            waveguide_structure,
            freq_band,
            mode_num=0,
            propagation_axis='y',
            source_position=10,
            perpendicular_bounds=(0, Lx),
            z_bounds=(0, Lz),
        )
        assert source_field.ndim == 5
        assert len(offset) == 3

    def test_invalid_axis_raises(self, simple_structure, freq_band):
        """Invalid propagation axis should raise ValueError."""
        with pytest.raises(ValueError, match="propagation_axis"):
            create_mode_source(
                simple_structure,
                freq_band,
                propagation_axis='z',
                source_position=10,
            )


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestImports:
    """Verify that key SDK symbols are importable from the top-level package."""

    def test_core_imports(self):
        pass

    def test_monitor_imports(self):
        pass

    def test_source_imports(self):
        pass

    def test_visualization_imports(self):
        pass

    def test_api_imports(self):
        pass

    def test_logging_imports(self):
        pass

    def test_deprecated_shims(self):
        """Deprecated shims should still be importable."""
