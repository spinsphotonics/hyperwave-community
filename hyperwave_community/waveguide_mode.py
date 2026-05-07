"""Waveguide mode solver wrapper.

Provides solve_waveguide_mode() matching the tutorial API.
Wraps the existing mode_solver.mode() function.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp

from hyperwave_community.mode_solver import mode as _mode_solve


def solve_waveguide_mode(
    grid: float,
    waveguide_width: float,
    waveguide_height: float,
    n_core: float,
    n_clad: float,
    wavelength: float,
    mode_number: int = 0,
    propagation_axis: int = 0,
    cross_section_size: int = 80,
) -> Tuple[np.ndarray, float]:
    """Solve for a waveguide eigenmode.

    Builds a waveguide cross-section permittivity, solves the eigenvalue
    problem, and returns the mode field and effective index.

    Args:
        grid: FDTD grid spacing in um.
        waveguide_width: Waveguide width in um.
        waveguide_height: Waveguide height in um (total device layer).
        n_core: Core refractive index.
        n_clad: Cladding refractive index.
        wavelength: Operating wavelength in um.
        mode_number: Which mode to solve for. For rectangular waveguides:
            0 = TE0 (fundamental), 1 = TM0, 2 = TE1, 3 = TM1, etc.
            The exact ordering depends on waveguide geometry.
        propagation_axis: Propagation direction (0=x, 1=y, 2=z).
        cross_section_size: Size of the cross-section grid in pixels.

    Returns:
        (mode_field, n_eff) where mode_field has shape (1, 6, 1, Ny, Nz)
        for x-propagation and n_eff is the effective refractive index.
    """
    eps_core = n_core ** 2
    eps_clad = n_clad ** 2

    wg_w_px = int(round(waveguide_width / grid))
    wg_h_px = int(round(waveguide_height / grid))

    # Make sure cross section is even
    cs = cross_section_size + (cross_section_size % 2)

    # Build YZ permittivity cross-section
    eps_yz = np.full((cs, cs), eps_clad, dtype=np.float32)
    y_center = cs // 2
    z_center = cs // 2
    y0 = y_center - wg_w_px // 2
    y1 = y_center + wg_w_px // 2
    z0 = z_center - wg_h_px // 2
    z1 = z_center + wg_h_px // 2
    eps_yz[y0:y1, z0:z1] = eps_core

    # Mode solver expects (3, xx, yy, zz) permittivity
    # For x-propagation: xx=1 (thin slab), yy=cs, zz=cs
    eps_4d = jnp.stack([jnp.array(eps_yz)] * 3, axis=0)[:, jnp.newaxis, :, :]

    # Frequency
    wl_px = wavelength / grid
    freq = 2 * np.pi / wl_px
    freq_band = (float(freq), float(freq), 1)

    # Solve
    mode_field, beta_arr, errs = _mode_solve(
        freq_band=freq_band,
        permittivity=eps_4d,
        axis=propagation_axis,
        mode_num=mode_number,
    )

    n_eff = float(beta_arr[0]) / (2 * np.pi / wl_px)

    return np.array(mode_field), float(n_eff)
