"""Tests for hyperwave_community.data_io module.

Tests GDS generation and theta round-tripping.
Requires gdstk and skimage.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import os
import tempfile

from hyperwave_community.data_io import (
    generate_gds_from_density,
    gds_to_theta,
    _winding_number,
    _is_clockwise,
)


class TestWindingNumber:
    """Tests for the _winding_number helper."""

    def test_point_inside_square(self):
        """Point inside a square polygon should have nonzero winding number."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        wn = _winding_number(square, (0.5, 0.5))
        assert wn != 0

    def test_point_outside_square(self):
        """Point outside a square polygon should have zero winding number."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        wn = _winding_number(square, (2.0, 2.0))
        assert wn == 0


class TestIsClockwise:
    """Tests for the _is_clockwise helper."""

    def test_clockwise_square(self):
        """Clockwise-ordered square should return True."""
        cw = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        assert _is_clockwise(cw) is True or _is_clockwise(cw) is np.True_

    def test_counterclockwise_square(self):
        """Counter-clockwise-ordered square should return False."""
        ccw = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert _is_clockwise(ccw) is False or _is_clockwise(ccw) is np.False_


class TestGenerateGds:
    """Tests for generate_gds_from_density."""

    def test_basic_generation(self):
        """Should produce a GDS file from a simple density."""
        density_arr = np.ones((50, 50))
        with tempfile.NamedTemporaryFile(suffix='.gds', delete=False) as f:
            path = f.name
        try:
            generate_gds_from_density(
                density_array=density_arr,
                level=0.5,
                output_filename=path,
                resolution=0.020,
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_empty_density(self):
        """All-zero density should produce a GDS (possibly with no polygons)."""
        density_arr = np.zeros((50, 50))
        with tempfile.NamedTemporaryFile(suffix='.gds', delete=False) as f:
            path = f.name
        try:
            generate_gds_from_density(
                density_array=density_arr,
                level=0.5,
                output_filename=path,
                resolution=0.020,
            )
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestGdsRoundTrip:
    """Tests for GDS export -> import round-tripping."""

    def test_round_trip_simple(self):
        """Exporting and re-importing should preserve structure roughly."""
        # Create a simple block pattern
        density_arr = np.zeros((100, 100))
        density_arr[25:75, 25:75] = 1.0

        with tempfile.NamedTemporaryFile(suffix='.gds', delete=False) as f:
            path = f.name
        try:
            generate_gds_from_density(
                density_array=density_arr,
                level=0.5,
                output_filename=path,
                resolution=0.020,
            )

            # gds_to_theta returns (theta, info_dict)
            theta_back, info = gds_to_theta(path, resolution=0.020)
            assert theta_back.ndim == 2
            assert theta_back.shape[0] > 0
            assert theta_back.shape[1] > 0
        finally:
            os.unlink(path)
