"""Waveguide mode solver wrapper.

Provides solve_waveguide_mode() matching the tutorial API.
Wraps the existing mode_solver.mode() function and derives the
full 6-component (E+H) field from Faraday/Ampere's law.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp

from hyperwave_community.mode_solver import mode as _mode_solve


def _spatial_diff(field, axis, is_forward):
    if is_forward:
        return jnp.roll(field, shift=-1, axis=axis) - field
    return field - jnp.roll(field, shift=+1, axis=axis)


def _broadcast_beta(betas, field_shape):
    return betas[:, None, None, None] * jnp.ones((1,) + field_shape)


def _curl_with_beta(field, betas, axis, field_shape, is_forward, beta_sign=+1):
    fx, fy, fz = field[:, 0], field[:, 1], field[:, 2]
    beta_bc = _broadcast_beta(betas, field_shape)

    def d(component, ax):
        if ax == axis:
            return beta_sign * 1j * beta_bc * component
        return _spatial_diff(component, axis=ax - 3, is_forward=is_forward)

    curl_x = d(fz, 1) - d(fy, 2)
    curl_y = d(fx, 2) - d(fz, 0)
    curl_z = d(fy, 0) - d(fx, 1)
    return jnp.stack([curl_x, curl_y, curl_z], axis=1)


def _mode_fields(fields, betas, freq_band, permittivity, axis, normalize=True):
    """Derive all 6 field components (E+H) from the 3-component E-field."""
    omegas = jnp.linspace(freq_band[0], freq_band[1], int(freq_band[2]))
    field_shape = fields.shape[2:]

    ax_i = (axis + 1) % 3
    ax_j = (axis + 2) % 3
    ax_k = axis

    e_full = jnp.zeros((fields.shape[0], 3) + field_shape, dtype=complex)
    e_full = e_full.at[:, ax_i].set(fields[:, ax_i])
    e_full = e_full.at[:, ax_j].set(fields[:, ax_j])

    h_full = _curl_with_beta(
        e_full, betas, axis, field_shape, is_forward=True, beta_sign=+1
    )
    omega_bc = omegas[:, None, None, None]
    h_full = h_full / (1j * omega_bc)[:, None]

    curl_h = _curl_with_beta(
        h_full, betas, axis, field_shape, is_forward=False, beta_sign=+1
    )
    eps_k = permittivity[ax_k]
    e_k = curl_h[:, ax_k] / (1j * omega_bc * eps_k)
    e_full = e_full.at[:, ax_k].set(e_k)

    if normalize:
        s_k = jnp.real(
            jnp.sum(
                e_full[:, ax_i] * jnp.conj(h_full[:, ax_j])
                - e_full[:, ax_j] * jnp.conj(h_full[:, ax_i]),
                axis=(-3, -2, -1),
            )
        )
        norm = jnp.sqrt(jnp.abs(s_k))[:, None, None, None]
        e_full = e_full / norm[:, None]
        h_full = h_full / norm[:, None]

    return jnp.concatenate([e_full, h_full], axis=1)


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
    """Solve for a waveguide eigenmode with full E+H fields.

    Builds a waveguide cross-section permittivity, solves the eigenvalue
    problem, then derives the complete 6-component electromagnetic field
    (Ex, Ey, Ez, Hx, Hy, Hz) using Faraday and Ampere's laws.

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
        The 6 components are [Ex, Ey, Ez, Hx, Hy, Hz], power-normalized.
    """
    eps_core = n_core ** 2
    eps_clad = n_clad ** 2

    wg_w_px = int(round(waveguide_width / grid))
    wg_h_px = int(round(waveguide_height / grid))

    cs = cross_section_size + (cross_section_size % 2)

    eps_yz = np.full((cs, cs), eps_clad, dtype=np.float32)
    y_center = cs // 2
    z_center = cs // 2
    y0 = y_center - wg_w_px // 2
    y1 = y_center + wg_w_px // 2
    z0 = z_center - wg_h_px // 2
    z1 = z_center + wg_h_px // 2
    eps_yz[y0:y1, z0:z1] = eps_core

    eps_4d = jnp.stack([jnp.array(eps_yz)] * 3, axis=0)[:, jnp.newaxis, :, :]

    wl_px = wavelength / grid
    freq = 2 * np.pi / wl_px
    freq_band = (float(freq), float(freq), 1)

    e_fields, beta_arr, errs = _mode_solve(
        freq_band=freq_band,
        permittivity=eps_4d,
        axis=propagation_axis,
        mode_num=mode_number,
    )

    full_field = _mode_fields(
        e_fields, beta_arr, freq_band, eps_4d, axis=propagation_axis, normalize=True
    )

    n_eff = float(beta_arr[0]) / (2 * np.pi / wl_px)

    return np.array(full_field), float(n_eff)