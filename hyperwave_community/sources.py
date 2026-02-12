import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import NamedTuple, Optional, Tuple, List, Dict, Union
from functools import partial
from dataclasses import dataclass
import matplotlib.patches as patches
import numpy as np

# Import existing functions to avoid duplication
# Use relative imports for hyperwave_community modules
from . import monitors as hwm



def create_gaussian_source(
    sim_shape: Tuple[int, int, int],
    frequencies: jnp.ndarray,
    source_pos: Tuple[int, int, int],
    waist_radius: float,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = 'x',
    x_span: float = 1.0,
    y_span: float = 1.0,
    conductivity=None,
    max_steps: int = 5_000,
    check_every_n: int = 200,
    show_plots: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create a Gaussian beam source using wave equation error method.

    This function creates a Gaussian beam source by running an FDTD simulation
    in free space and computing the wave equation error to generate a clean
    source field. This follows the exact methodology of the legacy gaussian_source_z.

    Args:
        sim_shape: Simulation domain shape (x, y, z) in pixels.
        frequencies: Array of angular frequencies (omega = 2*pi/lambda).
        source_pos: Source position (x, y, z) in pixels.
        waist_radius: Waist radius of the Gaussian beam in pixels.
        theta: Tilt angle in degrees for beam steering.
        phi: Azimuthal angle in degrees for beam steering (0=XZ plane, 90=YZ plane).
        polarization: Polarization direction ('x' or 'y'). Use 'y' for TE mode
            with phi=0 (XZ plane), 'x' for TE mode with phi=90 (YZ plane).
        x_span: X span of the source in pixels.
        y_span: Y span of the source in pixels.
        conductivity: Conductivity array (3, xx, yy, zz) if needed.
        max_steps: Maximum FDTD steps for error calculation.
        check_every_n: Steps between error checks.
        show_plots: Whether to display field plots.

    Returns:
        Tuple of (source_field, input_power) where:
        - source_field: Array with shape (num_freqs, 6, x_span, y_span, 1)
        - input_power: Power array for each frequency

    Note:
        This is computationally expensive as it runs an FDTD simulation
        to compute the clean source field via wave equation error.
    """
    num_freqs = len(frequencies)

    # Convert to integer spans
    x_span = int(x_span)
    y_span = int(y_span)
    z_span = sim_shape[2]

    shape = (x_span, y_span, z_span)
    c = tuple(s // 2 for s in shape)

    # Calculate source offset
    pos_x_offset = source_pos[0] - c[0]
    pos_y_offset = source_pos[1] - c[1]
    pos_z_offset = source_pos[2]
    source_offset = (pos_x_offset, pos_y_offset, pos_z_offset)

    # Create Gaussian profile
    x = jnp.linspace(-c[0], c[0], shape[0])[:, None]
    y = jnp.linspace(-c[1], c[1], shape[1])[None, :]
    r = jnp.sqrt(x**2 + y**2)
    gaussian_xy = jnp.exp(-(r**2) / (2 * waist_radius**2))

    # Convert angles to radians
    theta_rad = theta * jnp.pi / 180
    phi_rad = phi * jnp.pi / 180

    # Create initial source field with specified polarization
    initial_source_field = jnp.zeros((num_freqs, 6, shape[0], shape[1], 1), dtype=jnp.complex64)
    pol_idx = 0 if polarization == 'x' else 1  # 0=Ex, 1=Ey
    for fi, freq in enumerate(frequencies):
        k0 = freq  # k = omega/c with c=1
        kx = k0 * jnp.sin(theta_rad) * jnp.cos(phi_rad)
        ky = k0 * jnp.sin(theta_rad) * jnp.sin(phi_rad)
        phase = jnp.exp(-1j * (kx * x + ky * y))
        field_xy = gaussian_xy * phase
        initial_source_field = initial_source_field.at[fi, pol_idx, :, :, 0].set(field_xy)

    # Create air permittivity for simulation
    p_air = jnp.ones((3, sim_shape[0], sim_shape[1], sim_shape[2]))

    # Set up monitors
    monitor_xy = hw.solve.Monitor(
        shape=(sim_shape[0], sim_shape[1], 4),
        offset=(0, 0, source_offset[2] - 1)
    )
    monitor_xz = hw.solve.Monitor(
        shape=(sim_shape[0], 1, sim_shape[2]),
        offset=(0, sim_shape[1]//2, 0)
    )

    # Get frequency band
    freq_min = jnp.min(frequencies)
    freq_max = jnp.max(frequencies)
    freq_band = (freq_min, freq_max, num_freqs)

    # Run simulation to get field slices
    field_slices, _, _ = hw.solve.mem_efficient_multi_freq(
        freq_band=freq_band,
        permittivity=p_air,
        conductivity=conductivity,
        source_field=initial_source_field,
        source_offset=source_offset,
        monitors=[monitor_xy, monitor_xz],
        source_ramp_periods=10,
        max_steps=max_steps,
        check_every_n=check_every_n,
    )

    # Plot if requested
    if show_plots:
        xy_field = field_slices[0][0, 0, :, :, 1]
        xy_real = jnp.real(xy_field)
        xy_abs = jnp.abs(xy_field)
        v1 = jnp.max(jnp.abs(xy_real))/10

        y_mid = sim_shape[1] // 2
        xz_field = field_slices[1][0, 0, :, y_mid, :]
        xz_real = jnp.real(xz_field)
        xz_abs = jnp.abs(xz_field)
        v2 = jnp.max(jnp.abs(xz_real))/10

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        im00 = axes[0, 0].imshow(xy_real.T, cmap='viridis', vmax=+v1, vmin=-v1, origin='upper')
        axes[0, 0].set_title("Real(Ex) — XY Slice")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        fig.colorbar(im00, ax=axes[0, 0], label="Re(Ex)")

        im01 = axes[0, 1].imshow(xy_abs.T, cmap='viridis', vmax=+v1, vmin=-v1, origin='upper')
        axes[0, 1].set_title("|Ex| — XY Slice")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("y")
        fig.colorbar(im01, ax=axes[0, 1], label="|Ex|")

        im10 = axes[1, 0].imshow(xz_real.T, cmap='viridis', vmax=+v2, vmin=-v2, origin='upper')
        axes[1, 0].set_title("Real(Ex) — XZ Slice")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("z")
        fig.colorbar(im10, ax=axes[1, 0], label="Re(Ex)")

        im11 = axes[1, 1].imshow(xz_abs.T, cmap='viridis', vmax=+v2, vmin=-v2, origin='upper')
        axes[1, 1].set_title("|Ex| — XZ Slice")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("z")
        fig.colorbar(im11, ax=axes[1, 1], label="|Ex|")

        plt.tight_layout()
        plt.show()

    # Process field for wave equation error
    field = field_slices[0]
    source_field = jnp.zeros_like(field, dtype=jnp.complex64)
    field = field.at[:, (0, 1, 5), :, :, :2].set(0)
    field = field.at[:, (2, 3, 4), :, :, :1].set(0)

    # Compute wave equation error
    error = hw.solve.wave_equation_error_full(
        field=field,
        freq_band=hw.solve.FreqBand(*freq_band),
        permittivity=p_air[:, :, :, source_pos[2]-1:source_pos[2]+3],
        conductivity=conductivity[:, :, :, source_pos[2]-1:source_pos[2]+3] if conductivity is not None else None,
        source_field=source_field,
        source_offset=(0, 0, 0),
    )

    # Extract error plane at source height
    err_plane = error[:, :, :, :, 1]

    # Swap E and H components for source input
    err_src_field = jnp.take_along_axis(
        err_plane[..., None],
        indices=jnp.reshape(jnp.array([3, 4, 5, 0, 1, 2]), (1, 6, 1, 1, 1)),
        axis=1
    )
    err_src_field = err_src_field.at[:, (2, 5), ...].set(0.0)

    # Calculate input power
    input_power = jnp.abs(hw.monitors.get_power_through_plane(
        field=err_src_field, axis='z', position=0
    ))

    return err_src_field, input_power


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
    dx = lambda f: _d(f, axis=-3)
    dy = lambda f: _d(f, axis=-2)
    dz = lambda f: _d(f, axis=-1)
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
            (default), automatically computed from the wavelength to ensure
            at least ~1 wavelength of absorber on each side.
        absorption_coeff: PML absorption coefficient.  If ``None`` (default),
            automatically scaled from Bayesian-optimized baseline values.
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

    # Auto-compute absorption params from wavelength if not specified.
    # Baseline: width=82 cells at wl_px=77.5 (20nm res, 1550nm),
    # coeff=6.17e-4.  Both scale with (wl_px / 77.5).
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

    print(f"Absorber widths (x,y,z): {absorption_widths}")
    print(f"Absorber coefficient: {absorption_coeff:.6f}")

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


def create_mode_source(
    position: int,
    axis: str,
    frequencies: jnp.ndarray,
    structure,
    x_range: Optional[Tuple[int, int]] = None,
    y_range: Optional[Tuple[int, int]] = None,
    z_range: Optional[Tuple[int, int]] = None,
    mode_number: int = 0,
    amplitude: complex = 1.0,
    mode_solver=None
) -> Tuple[jnp.ndarray, Tuple[int, int, int]]:
    """Create a mode source field for waveguide mode injection.

    Computes electromagnetic modes in a waveguide by solving in the transverse
    plane perpendicular to the propagation direction, based on the waveguide's
    cross-sectional geometry and material distribution.

    Args:
        position: Position along the propagation axis (in pixels).
        axis: Propagation axis as a string ('x', 'y', or 'z').
            - 'x': Mode propagates along X, solved in YZ plane
            - 'y': Mode propagates along Y, solved in XZ plane
            - 'z': Mode propagates along Z, solved in XY plane
        frequencies: Array of angular frequencies (omega = 2*pi/lambda).
        structure: Structure object with permittivity distribution for mode solving.
        x_range: Tuple (start, stop) defining the X extent of the source (in pixels).
                 If None, uses full X dimension from structure. Only used when axis != 'x'.
        y_range: Tuple (start, stop) defining the Y extent of the source (in pixels).
                 If None, uses full Y dimension from structure. Only used when axis != 'y'.
        z_range: Tuple (start, stop) defining the Z extent of the source (in pixels).
                 If None, uses full Z dimension from structure. Only used when axis != 'z'.
        mode_number: Mode number to inject (0 for fundamental mode).
        amplitude: Complex amplitude of the injected mode.
        mode_solver: Optional custom mode solver function. If None, returns placeholder.

    Returns:
        Tuple of (source_field, offset) where:
        - source_field: Array with shape (num_freqs, 6, xx, yy, zz)
        - offset: Tuple (x, y, z) indicating where to place the source

    Note:
        This is an expensive operation as it solves for the waveguide modes.
        Store the result and reuse it rather than recomputing.
    """
    # Validate axis
    if axis not in ['x', 'y', 'z']:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
    if mode_number < 0:
        raise ValueError(f"mode_number must be non-negative, got {mode_number}")

    # Get structure dimensions
    if len(structure.permittivity.shape) == 4:
        _, x_dim, y_dim, z_dim = structure.permittivity.shape
    else:
        x_dim, y_dim, z_dim = structure.permittivity.shape

    # Set default ranges to full domain if not specified
    if x_range is None:
        x_range = (0, x_dim)
    if y_range is None:
        y_range = (0, y_dim)
    if z_range is None:
        z_range = (0, z_dim)

    # Determine shape and offset based on axis
    if axis == 'x':
        if x_range != (0, x_dim):
            raise ValueError("x_range should not be specified for axis='x' (use position instead)")
        shape = (1, y_range[1] - y_range[0], z_range[1] - z_range[0])
        offset = (position, y_range[0], z_range[0])

    elif axis == 'y':
        if y_range != (0, y_dim):
            raise ValueError("y_range should not be specified for axis='y' (use position instead)")
        shape = (x_range[1] - x_range[0], 1, z_range[1] - z_range[0])
        offset = (x_range[0], position, z_range[0])

    elif axis == 'z':
        if z_range != (0, z_dim):
            raise ValueError("z_range should not be specified for axis='z' (use position instead)")
        shape = (x_range[1] - x_range[0], y_range[1] - y_range[0], 1)
        offset = (x_range[0], y_range[0], position)

    # Validate ranges
    for range_val in [x_range, y_range, z_range]:
        if range_val is not None:
            if len(range_val) != 2:
                raise ValueError(f"Range must be a tuple of (start, stop), got {range_val}")
            if range_val[1] <= range_val[0]:
                raise ValueError(f"Range stop must be greater than start, got {range_val}")

    # Generate the mode field
    num_freqs = len(frequencies)
    source_field = jnp.zeros((num_freqs, 6, *shape), dtype=jnp.complex64)

    if mode_solver is not None:
        # Use provided mode solver to compute actual modes
        for fi, freq in enumerate(frequencies):
            mode_field = mode_solver(
                structure=structure,
                position=position,
                axis=axis,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                frequency=freq,
                mode_number=mode_number
            )
            source_field = source_field.at[fi].set(mode_field)
    else:
        # Placeholder: Create a simple field pattern
        if axis == 'x':
            source_field = source_field.at[:, 0, 0, :, :].set(amplitude)
        elif axis == 'y':
            source_field = source_field.at[:, 1, :, 0, :].set(amplitude)
        elif axis == 'z':
            source_field = source_field.at[:, 2, :, :, 0].set(amplitude)

    return source_field, offset


def mode_converter(*args, **kwargs):
    raise ImportError(
        "mode_converter() requires the full hyperwave solver package. "
        "Use hwc.mode_convert() instead for cloud-based E to E+H conversion."
    )


