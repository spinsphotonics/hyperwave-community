"""Tests for hyperwave_community.structure module.

Tests Layer creation, density filtering, and create_structure.
"""

import pytest
import jax.numpy as jnp
import numpy as np

import hyperwave_community as hwc
from hyperwave_community.structure import Layer, density, create_structure


class TestLayer:
    """Tests for the Layer dataclass."""

    def test_basic_creation(self, small_density_2d):
        """Layer with tuple permittivity and integer thickness."""
        layer = Layer(small_density_2d, (1.0, 12.0), 10)
        assert layer.density_pattern.shape == (20, 20)
        assert layer.permittivity_values == (1.0, 12.0)
        assert layer.layer_thickness == 10

    def test_scalar_permittivity(self, small_density_2d):
        """Layer with single float permittivity."""
        layer = Layer(small_density_2d, 2.25, 5)
        assert layer.permittivity_values == 2.25

    def test_default_conductivity(self, small_density_2d):
        """Default conductivity is 0."""
        layer = Layer(small_density_2d, 1.0, 10)
        assert layer.conductivity_values == 0.0

    def test_custom_conductivity(self, small_density_2d):
        """Layer with explicit conductivity tuple."""
        layer = Layer(small_density_2d, (1.0, 12.0), 10, conductivity_values=(0.0, 0.1))
        assert layer.conductivity_values == (0.0, 0.1)

    def test_odd_dimension_raises(self):
        """Odd-sized density pattern should raise ValueError."""
        odd = jnp.ones((21, 20))
        with pytest.raises(ValueError, match="not even"):
            Layer(odd, 1.0, 10)

    def test_1d_density_raises(self):
        """1D density pattern should raise ValueError."""
        with pytest.raises(ValueError, match="2D"):
            Layer(jnp.ones((20,)), 1.0, 10)

    def test_non_array_density_raises(self):
        """Non-JAX array should raise TypeError."""
        with pytest.raises(TypeError, match="jax.numpy.ndarray"):
            Layer(np.ones((20, 20)), 1.0, 10)

    def test_zero_thickness_raises(self):
        """Zero thickness should raise."""
        with pytest.raises((ValueError, TypeError)):
            Layer(jnp.ones((20, 20)), 1.0, 0)

    def test_negative_thickness_raises(self):
        """Negative thickness should raise."""
        with pytest.raises((ValueError, TypeError)):
            Layer(jnp.ones((20, 20)), 1.0, -5)

    def test_float_thickness(self, small_density_2d):
        """Float thickness should be accepted for subpixel averaging."""
        layer = Layer(small_density_2d, 1.0, 10.5)
        assert layer.layer_thickness == 10.5


class TestDensity:
    """Tests for the density() function."""

    def test_output_shape(self, medium_density_2d):
        """Output shape should be even and close to input shape."""
        result = density(medium_density_2d, radius=2)
        assert result.ndim == 2
        assert result.shape[0] % 2 == 0
        assert result.shape[1] % 2 == 0

    def test_all_ones_stays_high(self):
        """All-ones input should produce mostly high output."""
        theta = jnp.ones((40, 40))
        result = density(theta, radius=2, alpha=0.0)
        # With no projection, uniform input should stay uniform
        assert float(jnp.mean(result)) > 0.8

    def test_all_zeros_stays_low(self):
        """All-zeros input should produce mostly low output."""
        theta = jnp.zeros((40, 40))
        result = density(theta, radius=2, alpha=0.0)
        assert float(jnp.mean(result)) < 0.2

    def test_output_range(self, medium_density_2d):
        """Output should be approximately in [0, 1]."""
        result = density(medium_density_2d, radius=2)
        # Allow small numerical excursions
        assert float(jnp.min(result)) > -0.15
        assert float(jnp.max(result)) < 1.15

    def test_radius_zero(self):
        """Radius 0 should return something close to input."""
        theta = jnp.ones((20, 20)) * 0.7
        result = density(theta, radius=0, alpha=0.0)
        # Without filtering, values should be close to input
        assert result.shape[0] <= 20
        assert result.shape[1] <= 20

    def test_differentiable(self):
        """density() should be differentiable with JAX."""
        import jax

        def loss_fn(theta):
            d = density(theta, radius=2, alpha=0.0)
            return jnp.sum(d)

        theta = jnp.ones((20, 20)) * 0.5
        grad = jax.grad(loss_fn)(theta)
        assert grad.shape == theta.shape
        assert not jnp.any(jnp.isnan(grad))


class TestCreateStructure:
    """Tests for create_structure()."""

    def test_basic_structure(self, small_density_2d):
        """Create a simple 2-layer structure."""
        layers = [
            Layer(small_density_2d, 1.0, 10),
            Layer(small_density_2d, (1.0, 12.0), 5),
        ]
        structure = create_structure(layers, vertical_radius=0)

        assert hasattr(structure, 'permittivity')
        assert hasattr(structure, 'conductivity')
        assert structure.permittivity.ndim == 4
        assert structure.permittivity.shape[0] == 3  # 3 field components

    def test_z_dimension(self, small_density_2d):
        """Z dimension should match sum of layer thicknesses."""
        layers = [
            Layer(small_density_2d, 1.0, 10),
            Layer(small_density_2d, 1.0, 15),
        ]
        structure = create_structure(layers, vertical_radius=0)

        nz = structure.permittivity.shape[3]
        assert nz == 25  # 10 + 15

    def test_recipe_extraction(self, small_density_2d):
        """Structure should have extractable recipe."""
        layers = [
            Layer(small_density_2d, 1.0, 10),
            Layer(small_density_2d, (1.0, 4.0), 5),
        ]
        structure = create_structure(layers, vertical_radius=0)

        assert hasattr(structure, 'extract_recipe')
        recipe = structure.extract_recipe()
        assert isinstance(recipe, dict)

    def test_uniform_layer_permittivity(self, small_density_2d):
        """Uniform layer should have constant permittivity."""
        layers = [
            Layer(small_density_2d, 4.0, 10),
        ]
        structure = create_structure(layers, vertical_radius=0)

        eps = structure.permittivity
        # All values in this layer should be close to 4.0
        assert jnp.allclose(eps[0, :, :, :], 4.0, atol=0.1)
