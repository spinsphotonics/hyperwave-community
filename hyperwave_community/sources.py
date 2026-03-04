import jax.numpy as jnp
from typing import Optional, Tuple
from functools import partial
import numpy as np

from ._logging import logger

# Import existing functions to avoid duplication
# Use relative imports for hyperwave_community modules


# ---------------------------------------------------------------------------
# Wave equation error helpers (finite-difference curl for free-space post-processing)
# ---------------------------------------------------------------------------

def _spatial_diff(field, axis, is_forward):
    """Forward or backward finite difference along *axis*."""
    if is_forward:
        return np.roll(field, -1, axis=axis) - field
    return field - np.roll(field, 1, axis=axis)


def _curl_3d(field, is_forward):
    """Curl of a (..., 3, x, y, z) vector field via finite differences."""
    fx = field[..., 0, :, :, :]
    fy = field[..., 1, :, :, :]
    fz = field[..., 2, :, :, :]
    _d = partial(_spatial_diff, is_forward=is_forward)

    def dx(f):
        return _d(f, axis=-3)

    def dy(f):
        return _d(f, axis=-2)

    def dz(f):
        return _d(f, axis=-1)
    return np.stack([dy(fz) - dz(fy), dz(fx) - dx(fz), dx(fy) - dy(fx)],
                    axis=-4)


def _wave_equation_error_free_space(field, frequencies):
    """Wave equation residual in free space (eps=1, sigma=0, no source).

    Args:
        field: (N_freq, 6, x, y, z) complex field values.
        frequencies: Array of angular frequencies.

    Returns:
        (N_freq, 6, x, y, z) error tensor.
    """
    w = np.asarray(frequencies, dtype=np.float64).reshape(-1, 1, 1, 1, 1)
    e = field[:, 0:3]
    h = field[:, 3:6]
    error_e = _curl_3d(e, is_forward=True) + 1j * w * h
    error_h = _curl_3d(h, is_forward=False) - 1j * w * e
    return np.concatenate([error_e, error_h], axis=1)


# ---------------------------------------------------------------------------
# Cloud-based Gaussian source generation
# ---------------------------------------------------------------------------

def generate_gaussian_source(
    sim_shape: Tuple[int, int, int],
    frequencies,
    source_pos: Tuple[int, int, int],
    waist_radius: float,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = 'y',
    max_steps: int = 5000,
    check_every_n: int = 1000,
    absorption_widths: Tuple[int, int, int] = None,
    absorption_coeff: float = None,
    wavelength_um: float = None,
    dx_um: float = None,
    gpu_type: str = "B200",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Gaussian beam source via cloud GPU FDTD simulation.

    Runs FDTD in free space on a cloud GPU, then computes the wave equation
    error locally to produce a clean unidirectional source field.  This is
    the cloud-compatible replacement for ``create_gaussian_source`` which
    requires the full hyperwave solver package.

    Args:
        sim_shape: Simulation domain (Lx, Ly, Lz) in structure grid pixels.
        frequencies: Array of angular frequencies (omega = 2*pi/lambda).
        source_pos: Source center (x, y, z) in structure grid pixels.
        waist_radius: Beam waist radius in structure grid pixels.
        theta: Tilt angle in degrees (negative tilts toward -x).
        phi: Azimuthal angle in degrees (0=XZ plane).
        polarization: ``'x'`` or ``'y'``.
        max_steps: Maximum FDTD time steps.
        check_every_n: Steps between convergence checks.
        absorption_widths: PML widths ``(x, y, z)`` in pixels.  If ``None``
            (default), computed from ``wavelength_um``/``dx_um`` via
            ``absorber_params()`` when available, otherwise from a legacy
            heuristic.
        absorption_coeff: PML absorption coefficient.  If ``None`` (default),
            computed from ``wavelength_um``/``dx_um`` via ``absorber_params()``
            when available, otherwise from a legacy heuristic.
        wavelength_um: Wavelength in micrometers.  When provided together with
            ``dx_um``, enables ``absorber_params()`` for auto-computed PML
            defaults (recommended).
        dx_um: Grid spacing in micrometers.  See ``wavelength_um``.
        gpu_type: Cloud GPU type (``"B200"``, ``"H100"``, etc.).

    Returns:
        ``(source_field, input_power)`` where *source_field* has shape
        ``(N_freq, 6, Lx, Ly, 1)`` and *input_power* has shape ``(N_freq,)``.
    """
    from .structure import recipe_from_params
    from .api_client import simulate as api_simulate
    from .monitors import get_power_through_plane

    Lx, Ly, Lz = sim_shape
    frequencies = np.asarray(frequencies)
    num_freqs = len(frequencies)

    # Auto-compute absorption params if not explicitly provided.
    if absorption_widths is None or absorption_coeff is None:
        if wavelength_um is not None and dx_um is not None:
            # Use absorber_params() for consistent, validated defaults
            from .absorption import absorber_params
            ap = absorber_params(wavelength_um, dx_um, structure_dimensions=sim_shape)
            if absorption_widths is None:
                absorption_widths = ap["absorption_widths"]
            if absorption_coeff is None:
                absorption_coeff = ap["abs_coeff"]
        else:
            # Legacy fallback: derive from frequency when wl/dx not provided
            wl_px = 2 * np.pi / float(np.mean(frequencies))
            if absorption_widths is None:
                w_xy = max(20, int(round(1.06 * wl_px)))
                w_z = max(15, int(round(0.7 * wl_px)))
                absorption_widths = (
                    min(w_xy, Lx // 4),
                    min(w_xy, Ly // 4),
                    min(w_z, Lz // 4),
                )
            if absorption_coeff is None:
                absorption_coeff = 1.03e-7 * wl_px ** 2

    logger.debug(f"Absorber widths (x,y,z): {absorption_widths}")
    logger.debug(f"Absorber coefficient: {absorption_coeff:.6f}")

    # Source offset: full XY span, at source z position
    source_offset = (0, 0, int(source_pos[2]))

    # ----- Build Gaussian initial field -----
    x = np.arange(Lx, dtype=np.float64)[:, None] - source_pos[0]
    y = np.arange(Ly, dtype=np.float64)[None, :] - source_pos[1]
    gaussian_xy = np.exp(-(x**2 + y**2) / (2 * waist_radius**2))

    theta_rad = theta * np.pi / 180
    phi_rad = phi * np.pi / 180

    initial_source = np.zeros((num_freqs, 6, Lx, Ly, 1), dtype=np.complex64)
    pol_idx = 0 if polarization == 'x' else 1

    for fi in range(num_freqs):
        k0 = float(frequencies[fi])
        kx = k0 * np.sin(theta_rad) * np.cos(phi_rad)
        ky = k0 * np.sin(theta_rad) * np.sin(phi_rad)
        phase = np.exp(-1j * (kx * x + ky * y))
        initial_source[fi, pol_idx, :, :, 0] = gaussian_xy * phase

    # ----- Air-only structure recipe (lightweight, no large arrays) -----
    air_recipe = recipe_from_params(
        grid_shape=(2 * Lx, 2 * Ly),
        layers=[{
            'density': 'uniform',
            'value': 0.0,
            'permittivity': 1.0,
            'thickness': float(Lz),
        }],
        vertical_radius=0.0,
    )

    # ----- Monitor at source plane (z-thickness = 4 for wave equation error) -----
    z_offset = max(int(source_pos[2]) - 1, 0)
    monitors_recipe = [{
        'name': 'Output_source_plane',
        'shape': (Lx, Ly, 4),
        'offset': (0, 0, z_offset),
    }]
    # NOTE: Monitor must be named 'Output_*' for the /early_stopping endpoint.
    # We use convergence="full" below to bypass early stopping for source gen.

    # ----- Frequency band -----
    freq_min = float(np.min(frequencies))
    freq_max = float(np.max(frequencies))
    freq_band = (freq_min, freq_max, num_freqs)

    # ----- Run FDTD on cloud GPU -----
    response = api_simulate(
        structure_recipe=air_recipe,
        source_field=initial_source,
        source_offset=source_offset,
        freq_band=freq_band,
        monitors_recipe=monitors_recipe,
        simulation_steps=max_steps,
        check_every_n=check_every_n,
        source_ramp_periods=10,
        add_absorption=True,
        absorption_widths=absorption_widths,
        absorption_coeff=absorption_coeff,
        gpu_type=gpu_type,
        convergence="full",
    )

    # ----- Post-process: wave equation error -----
    field = np.array(response['monitor_data']['Output_source_plane'])  # (N, 6, Lx, Ly, 4)

    # Zero out field components that aren't part of the source plane
    # (same masking as the reference implementation)
    field[:, (0, 1, 5), :, :, :2] = 0
    field[:, (2, 3, 4), :, :, :1] = 0

    error = _wave_equation_error_free_space(field, frequencies)

    # Extract error at source z (index 1 within the 4-pixel slab)
    err_plane = error[:, :, :, :, 1:2]  # (N, 6, Lx, Ly, 1)

    # Swap E and H components to create source field
    swap_idx = np.array([3, 4, 5, 0, 1, 2])
    err_src_field = err_plane[:, swap_idx, :, :, :]

    # Zero out z-components (Ez, Hz)
    err_src_field[:, 2, :, :, :] = 0.0
    err_src_field[:, 5, :, :, :] = 0.0

    # ----- Compute input power -----
    input_power = np.abs(np.array(get_power_through_plane(
        field=jnp.array(err_src_field), axis='z', position=0
    )))

    return err_src_field.astype(np.complex64), input_power

