"""Shared fixtures for hyperwave-community tests."""

import pytest
import jax.numpy as jnp
import numpy as np


@pytest.fixture
def small_density_2d():
    """Small 2D density pattern for structure tests."""
    return jnp.ones((20, 20))


@pytest.fixture
def medium_density_2d():
    """Medium 2D density pattern for density filter tests."""
    return jnp.ones((40, 40)) * 0.5


@pytest.fixture
def simple_field_data():
    """Simple synthetic field data (N_freq=1, 6 components, 10x10x10)."""
    np.random.seed(42)
    # Create field with known structure for reproducible tests
    field = np.zeros((1, 6, 10, 10, 10), dtype=np.complex64)
    # Set Ey and Hz to create power flow in +x direction
    field[0, 1, :, :, :] = 1.0 + 0j   # Ey
    field[0, 5, :, :, :] = 1.0 + 0j   # Hz
    return jnp.array(field)


@pytest.fixture
def field_slice_4d():
    """4D field slice for Poynting vector tests (N_freq=1, 6, Ny=8, Nx=8)."""
    field = jnp.zeros((1, 6, 8, 8), dtype=jnp.complex64)
    # Ey * Hz* gives Sx > 0
    field = field.at[0, 1, :, :].set(1.0 + 0j)   # Ey
    field = field.at[0, 5, :, :].set(1.0 + 0j)   # Hz
    return field
