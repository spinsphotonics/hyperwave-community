"""Tests for hyperwave_community.monitors module.

Tests Monitor, MonitorSet, S_from_slice, and field analysis functions.
"""

import pytest
import jax.numpy as jnp

from hyperwave_community.monitors import (
    Monitor,
    MonitorSet,
    S_from_slice,
    get_field_slice,
    get_field_intensity,
    get_electric_field_intensity,
    get_magnetic_field_intensity,
)


class TestMonitor:
    """Tests for the Monitor dataclass."""

    def test_basic_creation(self):
        """Create a valid monitor."""
        mon = Monitor(shape=(10, 20, 30), offset=(5, 5, 5))
        assert mon.shape == (10, 20, 30)
        assert mon.offset == (5, 5, 5)

    def test_negative_offset(self):
        """Negative offsets should be allowed."""
        mon = Monitor(shape=(10, 10, 10), offset=(-5, 0, 0))
        assert mon.offset == (-5, 0, 0)

    def test_zero_shape_raises(self):
        """Zero-size shape should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Monitor(shape=(0, 10, 10), offset=(0, 0, 0))

    def test_negative_shape_raises(self):
        """Negative shape should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Monitor(shape=(-1, 10, 10), offset=(0, 0, 0))

    def test_wrong_shape_length(self):
        """Shape with wrong number of dims should raise."""
        with pytest.raises(ValueError, match="3 dimensions"):
            Monitor(shape=(10, 20), offset=(0, 0, 0))

    def test_wrong_offset_length(self):
        """Offset with wrong number of dims should raise."""
        with pytest.raises(ValueError, match="3 dimensions"):
            Monitor(shape=(10, 10, 10), offset=(0, 0))

    def test_float_shape_raises(self):
        """Float in shape should raise ValueError."""
        with pytest.raises(ValueError, match="integers"):
            Monitor(shape=(10.5, 10, 10), offset=(0, 0, 0))

    def test_recipe(self):
        """Monitor recipe should contain shape and offset."""
        mon = Monitor(shape=(10, 20, 30), offset=(1, 2, 3))
        recipe = mon.recipe
        assert recipe['shape'] == (10, 20, 30)
        assert recipe['offset'] == (1, 2, 3)


class TestMonitorSet:
    """Tests for the MonitorSet class."""

    def test_empty_creation(self):
        """Empty MonitorSet should work."""
        ms = MonitorSet()
        assert len(ms.monitors) == 0
        assert len(ms.mapping) == 0

    def test_add_monitor(self):
        """Adding a monitor should increase count."""
        ms = MonitorSet()
        idx = ms.add(Monitor(shape=(10, 10, 10), offset=(0, 0, 0)), name='test')
        assert idx == 0
        assert len(ms.monitors) == 1
        assert 'test' in ms.mapping

    def test_add_multiple(self):
        """Adding multiple monitors should assign sequential indices."""
        ms = MonitorSet()
        i0 = ms.add(Monitor(shape=(10, 10, 10), offset=(0, 0, 0)), name='a')
        i1 = ms.add(Monitor(shape=(5, 5, 5), offset=(1, 1, 1)), name='b')
        assert i0 == 0
        assert i1 == 1
        assert len(ms.monitors) == 2

    def test_auto_name(self):
        """Monitor without name should get automatic name."""
        ms = MonitorSet()
        ms.add(Monitor(shape=(10, 10, 10), offset=(0, 0, 0)))
        assert 'monitor_0' in ms.mapping

    def test_duplicate_name_raises(self):
        """Adding monitor with duplicate name should raise."""
        ms = MonitorSet()
        ms.add(Monitor(shape=(10, 10, 10), offset=(0, 0, 0)), name='dup')
        with pytest.raises(ValueError):
            ms.add(Monitor(shape=(5, 5, 5), offset=(1, 1, 1)), name='dup')

    def test_non_monitor_raises(self):
        """Adding non-Monitor object should raise TypeError."""
        ms = MonitorSet()
        with pytest.raises(TypeError, match="Monitor"):
            ms.add("not a monitor", name='bad')


class TestSFromSlice:
    """Tests for S_from_slice Poynting vector calculation."""

    def test_output_shape(self, field_slice_4d):
        """Output shape should be (N_freq, 3, Ny, Nx)."""
        S = S_from_slice(field_slice_4d)
        assert S.shape == (1, 3, 8, 8)

    def test_positive_sx(self, field_slice_4d):
        """Ey and Hz should produce positive Sx (power in +x)."""
        S = S_from_slice(field_slice_4d)
        Sx = S[0, 0, :, :]
        # S_x = 0.5 * Re(Ey * Hz*) should be positive
        assert jnp.all(Sx >= 0)

    def test_zero_field_zero_power(self):
        """Zero field should give zero Poynting vector."""
        field = jnp.zeros((1, 6, 8, 8), dtype=jnp.complex64)
        S = S_from_slice(field)
        assert jnp.allclose(S, 0.0)

    def test_multi_freq(self):
        """Should work with multiple frequencies."""
        field = jnp.zeros((3, 6, 8, 8), dtype=jnp.complex64)
        field = field.at[:, 1, :, :].set(1.0)  # Ey
        field = field.at[:, 5, :, :].set(1.0)  # Hz
        S = S_from_slice(field)
        assert S.shape == (3, 3, 8, 8)

    def test_poynting_magnitude(self):
        """Check Poynting vector magnitude for known input."""
        field = jnp.zeros((1, 6, 4, 4), dtype=jnp.complex64)
        # Ey = 2, Hz = 3 -> Sx = 0.5 * Re(Ey * Hz*) = 0.5 * 2 * 3 = 3.0
        field = field.at[0, 1, :, :].set(2.0)
        field = field.at[0, 5, :, :].set(3.0)
        S = S_from_slice(field)
        assert jnp.allclose(S[0, 0, :, :], 3.0)


class TestFieldAnalysis:
    """Tests for field intensity functions."""

    def test_field_intensity_shape(self, simple_field_data):
        """get_field_intensity should return correct shape."""
        intensity = get_field_intensity(simple_field_data)
        assert intensity.shape == (1, 10, 10, 10)

    def test_electric_intensity_shape(self, simple_field_data):
        """get_electric_field_intensity should return correct shape."""
        e_intensity = get_electric_field_intensity(simple_field_data)
        assert e_intensity.shape == (1, 10, 10, 10)

    def test_magnetic_intensity_shape(self, simple_field_data):
        """get_magnetic_field_intensity should return correct shape."""
        h_intensity = get_magnetic_field_intensity(simple_field_data)
        assert h_intensity.shape == (1, 10, 10, 10)

    def test_total_equals_sum(self, simple_field_data):
        """Total intensity should equal electric + magnetic."""
        total = get_field_intensity(simple_field_data)
        e_int = get_electric_field_intensity(simple_field_data)
        h_int = get_magnetic_field_intensity(simple_field_data)
        assert jnp.allclose(total, e_int + h_int, atol=1e-6)

    def test_zero_field_zero_intensity(self):
        """Zero field should give zero intensity."""
        field = jnp.zeros((1, 6, 5, 5, 5))
        intensity = get_field_intensity(field)
        assert jnp.allclose(intensity, 0.0)

    def test_nonnegative(self, simple_field_data):
        """Intensities should always be non-negative."""
        assert jnp.all(get_field_intensity(simple_field_data) >= 0)
        assert jnp.all(get_electric_field_intensity(simple_field_data) >= 0)
        assert jnp.all(get_magnetic_field_intensity(simple_field_data) >= 0)


class TestGetFieldSlice:
    """Tests for get_field_slice."""

    def test_x_slice(self, simple_field_data):
        """Slicing along x should reduce x dimension to 1."""
        sliced = get_field_slice(simple_field_data, axis='x', position=5)
        assert sliced.shape[2] == 1 or sliced.ndim == 4

    def test_y_slice(self, simple_field_data):
        """Slicing along y should work."""
        sliced = get_field_slice(simple_field_data, axis='y', position=5)
        assert sliced is not None

    def test_z_slice(self, simple_field_data):
        """Slicing along z should work."""
        sliced = get_field_slice(simple_field_data, axis='z', position=5)
        assert sliced is not None
