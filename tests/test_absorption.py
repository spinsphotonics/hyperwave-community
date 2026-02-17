"""Tests for hyperwave_community.absorption module.

Tests absorber parameter computation and absorption mask creation.
"""

import jax.numpy as jnp

from hyperwave_community.absorption import (
    get_optimized_absorber_params,
    create_absorption_mask,
)


class TestGetOptimizedAbsorberParams:
    """Tests for get_optimized_absorber_params."""

    def test_default_params(self):
        """Default (20nm) should return baseline values."""
        params = get_optimized_absorber_params()
        assert params['absorber_width'] == 82
        assert abs(params['absorber_coeff'] - 0.000617) < 0.001

    def test_higher_resolution(self):
        """Higher resolution (10nm) should give more cells."""
        params = get_optimized_absorber_params(resolution_nm=10.0)
        assert params['absorber_width'] > 82  # More cells at finer resolution

    def test_lower_resolution(self):
        """Lower resolution (40nm) should give fewer cells."""
        params = get_optimized_absorber_params(resolution_nm=40.0)
        assert params['absorber_width'] < 82

    def test_coeff_scaling(self):
        """Coefficient should scale with 1/scale^2."""
        params_20 = get_optimized_absorber_params(resolution_nm=20.0)
        params_40 = get_optimized_absorber_params(resolution_nm=40.0)
        # At 40nm, scale = 2, so coeff should be ~1/4 of 20nm value
        ratio = params_20['absorber_coeff'] / params_40['absorber_coeff']
        assert abs(ratio - 4.0) < 0.1

    def test_with_structure_dimensions(self):
        """Should return absorption_widths when dimensions provided."""
        params = get_optimized_absorber_params(
            resolution_nm=20.0,
            structure_dimensions=(800, 200, 100),
        )
        assert 'absorption_widths' in params
        assert len(params['absorption_widths']) == 3

    def test_absorption_widths_capped(self):
        """Absorption widths should be capped at 25% of dimension."""
        params = get_optimized_absorber_params(
            resolution_nm=20.0,
            structure_dimensions=(100, 100, 100),
        )
        widths = params['absorption_widths']
        assert widths[0] <= 25  # 25% of 100
        assert widths[1] <= 25
        assert widths[2] <= 25

    def test_minimum_widths(self):
        """Absorption widths should have minimum of 20."""
        params = get_optimized_absorber_params(
            resolution_nm=20.0,
            structure_dimensions=(1000, 1000, 1000),
        )
        widths = params['absorption_widths']
        assert widths[0] >= 20
        assert widths[1] >= 20
        assert widths[2] >= 20

    def test_baseline_info(self):
        """Should include baseline optimization info."""
        params = get_optimized_absorber_params()
        assert 'baseline_info' in params
        assert params['baseline_info']['resolution_nm'] == 20.0


class TestCreateAbsorptionMask:
    """Tests for create_absorption_mask."""

    def test_output_shape(self):
        """Output should be (3, xx, yy, zz)."""
        mask = create_absorption_mask(
            grid_shape=(50, 50, 50),
            absorption_widths=(10, 10, 10),
            absorption_coeff=0.001,
            show_plots=False,
        )
        assert mask.shape == (3, 50, 50, 50)

    def test_center_is_zero(self):
        """Center of the domain should have zero absorption."""
        mask = create_absorption_mask(
            grid_shape=(100, 100, 100),
            absorption_widths=(20, 20, 20),
            absorption_coeff=0.001,
            show_plots=False,
        )
        # Center region should be zero
        center = mask[:, 40:60, 40:60, 40:60]
        assert jnp.allclose(center, 0.0)

    def test_boundary_nonzero(self):
        """Boundary regions should have nonzero absorption."""
        mask = create_absorption_mask(
            grid_shape=(100, 100, 100),
            absorption_widths=(20, 20, 20),
            absorption_coeff=0.001,
            show_plots=False,
        )
        # Edge should have absorption
        edge_values = mask[:, 0, 50, 50]
        assert jnp.any(edge_values > 0)

    def test_nonnegative(self):
        """Absorption values should be non-negative."""
        mask = create_absorption_mask(
            grid_shape=(50, 50, 50),
            absorption_widths=(10, 10, 10),
            absorption_coeff=0.001,
            show_plots=False,
        )
        assert jnp.all(mask >= 0)

    def test_symmetric(self):
        """Absorption should be symmetric about the center."""
        mask = create_absorption_mask(
            grid_shape=(60, 60, 60),
            absorption_widths=(15, 15, 15),
            absorption_coeff=0.001,
            show_plots=False,
        )
        # Check x-symmetry for first component
        left = mask[0, :, 30, 30]
        right = jnp.flip(mask[0, :, 30, 30])
        assert jnp.allclose(left, right, atol=1e-6)
