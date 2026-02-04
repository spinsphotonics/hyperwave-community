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

# Solver functions (only available in full hyperwave package)
try:
    from . import solve as hws
    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False
    hws = None


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


# @title new gaussian
def create_gaussian_source_full_span_on_modal(
    air_recipe: dict,
    frequencies: jnp.ndarray,
    source_pos: Tuple[int, int, int],
    waist_radius: float,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = 'x',
    max_steps: int = 5_000,
    check_every_n: int = 200,
    show_plots: bool = False,
    gpu_type: str = "B200",
    add_absorption: bool = True,
    absorption_widths: Tuple[int, int, int] = (30, 30, 20),
    absorption_coeff: float = 1e-4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create a Gaussian beam source spanning full simulation domain using Modal GPU.

    This function creates a Gaussian beam source by running an FDTD simulation
    in free space on a Modal GPU and computing the wave equation error to generate
    a clean source field.

    Args:
        air_recipe: Free-space structure recipe (uniform air, permittivity=1.0).
                    User must construct this with matching dimensions to their
                    device structure. Use hwst.recipe_from_params() to create.
        frequencies: Array of angular frequencies (omega = 2*pi/lambda).
        source_pos: Source position (x, y, z) in absolute pixels (0,0,0 is origin).
        waist_radius: Waist radius of the Gaussian beam in pixels.
        theta: Tilt angle in degrees for beam steering.
        phi: Azimuthal angle in degrees for beam steering (0=XZ plane, 90=YZ plane).
        polarization: Polarization direction ('x' or 'y').
        max_steps: Maximum FDTD steps for error calculation.
        check_every_n: Steps between error checks.
        show_plots: Whether to display field plots.
        gpu_type: Modal GPU type (B200, H100, A100, A10G, T4, L40S, etc.).
        add_absorption: Whether to add PML boundaries to free-space simulation.
                        Must be True. User must provide absorption_widths and absorption_coeff.
        absorption_widths: PML widths (x, y, z). Required parameter.
                           Default: (30, 30, 20).
        absorption_coeff: PML absorption coefficient. Required parameter.
                          Default: 1e-4.

    Returns:
        Tuple of (source_field, input_power) where:
        - source_field: Array with shape (num_freqs, 6, sim_x, sim_y, 1)
        - input_power: Power array for each frequency

    Example:
        # Create air recipe matching your device dimensions
        air_recipe = hwst.recipe_from_params(
            grid_shape=(Lx, Ly)
            layers=[{
                'density': 'uniform',
                'value': 0.0,
                'permittivity': 1.0,
                'thickness': Lz,
                'conductivity': 0.0
            }],
            vertical_radius=0.0,
            use_simple_averaging=True
        )
        
        # Generate source
        source_field, power = create_gaussian_source_full_span_on_modal(
            air_recipe=air_recipe,
            frequencies=jnp.array([0.196, 0.209]),
            source_pos=(250, 125, 50),
            waist_radius=20.0,
            polarization='x'
        )

    Note:
        This is computationally expensive as it runs an FDTD simulation
        to compute the clean source field via wave equation error.
        Requires Modal deployment: modal deploy hyperwave/simulate_modal.py
    """
    # This function requires Modal deployment from the full hyperwave package
    raise ImportError(
        "generate_gaussian_source requires Modal deployment from the full hyperwave package. "
        "Use the API workflow with hwc.solve_mode_source() for mode sources instead."
    )

    # Extract simulation shape from air recipe
    # air_recipe['metadata']['final_shape'] is (freq, x, y, z) - skip freq dimension
    sim_shape = tuple(air_recipe['metadata']['final_shape'][1:])  # (x, y, z)

    print(f"\nCreating Gaussian source for simulation shape: {sim_shape}")
    print(f"Source position: {source_pos}")
    print(f"Waist radius: {waist_radius} pixels")
    print(f"Polarization: {polarization}")

    num_freqs = len(frequencies)

    # Source spans full simulation domain
    shape = sim_shape  # (x, y, z)

    # Source offset is now simple: (0, 0, z_pos)
    # XY offsets are 0 because source spans full domain
    source_offset = (0, 0, source_pos[2])

    # Create Gaussian profile centered at source_pos in absolute coordinates
    # X and Y coordinates relative to origin (0, 0)
    x = jnp.arange(shape[0])[:, None] - source_pos[0]
    y = jnp.arange(shape[1])[None, :] - source_pos[1]
    r = jnp.sqrt(x**2 + y**2)
    gaussian_xy = jnp.exp(-(r**2) / (2 * waist_radius**2))

    # Convert angles to radians
    theta_rad = theta * jnp.pi / 180
    phi_rad = phi * jnp.pi / 180

    # Create initial source field with specified polarization
    initial_source_field = jnp.zeros((num_freqs, 6, shape[0], shape[1], 1), dtype=jnp.complex64)
    pol_idx = 0 if polarization == 'x' else 1  # Get polarization index
    
    for fi, freq in enumerate(frequencies):
        k0 = freq  # k = omega/c with c=1
        kx = k0 * jnp.sin(theta_rad) * jnp.cos(phi_rad)
        ky = k0 * jnp.sin(theta_rad) * jnp.sin(phi_rad)
        phase = jnp.exp(-1j * (kx * x + ky * y))
        field_xy = gaussian_xy * phase
        initial_source_field = initial_source_field.at[fi, pol_idx, :, :, 0].set(field_xy)

    # Set up monitors spanning full domain
    monitors = hwm.MonitorSet()

    # XY monitor at source plane with thickness 4
    monitors.add(
        hwm.Monitor(
            shape=(sim_shape[0], sim_shape[1], 4),
            offset=(0, 0, source_offset[2] - 1)
        ),
        name="xy"
    )

    # Choose between XZ and YZ monitor based on phi angle
    # If phi == 0: beam propagates in XZ plane → use XZ monitor
    # If phi == 90: beam propagates in YZ plane → use YZ monitor
    if abs(phi) < 45:  # Closer to XZ plane
        monitors.add(
            hwm.Monitor(
                shape=(sim_shape[0], 1, sim_shape[2]),
                offset=(0, sim_shape[1]//2, 0)
            ),
            name="xz"
        )
        cross_plane = "xz"
    else:  # Closer to YZ plane
        monitors.add(
            hwm.Monitor(
                shape=(1, sim_shape[1], sim_shape[2]),
                offset=(sim_shape[0]//2, 0, 0)
            ),
            name="yz"
        )
        cross_plane = "yz"

    print(f"\nMonitors configured:")
    print(f"  XY monitor: shape={monitors.monitors[0].shape}, offset={monitors.monitors[0].offset}")
    print(f"  {cross_plane.upper()} monitor: shape={monitors.monitors[1].shape}, offset={monitors.monitors[1].offset}")

    # Get frequency band
    freq_min = jnp.min(frequencies)
    freq_max = jnp.max(frequencies)
    freq_band = (freq_min, freq_max, num_freqs)

    # Run simulation on Modal GPU
    print(f"\nRunning Gaussian source generation on Modal {gpu_type}...")
    response = simulate_on_modal(
        structure_recipe=air_recipe,
        source_field=initial_source_field,
        source_offset=source_offset,
        freq_band=freq_band,
        monitors=monitors,
        mode_info=None,
        max_steps=max_steps,
        check_every_n=check_every_n,
        source_ramp_periods=10,
        add_absorption=add_absorption,
        absorption_widths=absorption_widths,
        absorption_coeff=absorption_coeff,
        gpu_type=gpu_type
    )

    print(f"✓ Modal simulation completed in {response.get('sim_time', 0):.2f}s")

    # Extract field slices from response (list of monitor fields)
    field_slices = response['monitor_data']

    print(f"\nField shapes from monitors:")
    print(f"  XY monitor: {field_slices[0].shape}")
    print(f"  {cross_plane.upper()} monitor: {field_slices[1].shape}")

    # Plot if requested
    if show_plots:
        # Determine field component label based on polarization
        field_label = f"E{polarization}"  # Ex or Ey
        
        # XY plane - use polarization index
        xy_field = field_slices[0][0, pol_idx, :, :, 1]
        xy_real  = jnp.real(xy_field)
        xy_abs   = jnp.abs(xy_field)
        v1 = jnp.max(jnp.abs(xy_real))/10

        # Cross plane (XZ or YZ) - use polarization index
        if cross_plane == "xz":
            # XZ monitor has Y dimension of 1, so access index 0
            cross_field = field_slices[1][0, pol_idx, :, 0, :]
        else:  # yz
            # YZ monitor has X dimension of 1, so access index 0
            cross_field = field_slices[1][0, pol_idx, 0, :, :]
            
        cross_real = jnp.real(cross_field)
        cross_abs  = jnp.abs(cross_field)
        v2 = jnp.max(jnp.abs(cross_real))/10

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Top-left: real XY
        im00 = axes[0, 0].imshow(xy_real.T, cmap='viridis', vmax=+v1, vmin=-v1, origin='upper')
        axes[0, 0].set_title(f"Real({field_label}) - XY Slice")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        fig.colorbar(im00, ax=axes[0, 0], label=f"Re({field_label})")

        # Top-right: abs XY
        im01 = axes[0, 1].imshow(xy_abs.T, cmap='viridis', vmax=+v1, vmin=-v1, origin='upper')
        axes[0, 1].set_title(f"|{field_label}| - XY Slice")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("y")
        fig.colorbar(im01, ax=axes[0, 1], label=f"|{field_label}|")

        # Bottom-left: real cross-plane
        im10 = axes[1, 0].imshow(cross_real.T, cmap='viridis', vmax=+v2, vmin=-v2, origin='upper')
        axes[1, 0].set_title(f"Real({field_label}) - {cross_plane.upper()} Slice")
        axes[1, 0].set_xlabel(cross_plane[0])
        axes[1, 0].set_ylabel(cross_plane[1])
        fig.colorbar(im10, ax=axes[1, 0], label=f"Re({field_label})")

        # Bottom-right: abs cross-plane
        im11 = axes[1, 1].imshow(cross_abs.T, cmap='viridis', vmax=+v2, vmin=-v2, origin='upper')
        axes[1, 1].set_title(f"|{field_label}| - {cross_plane.upper()} Slice")
        axes[1, 1].set_xlabel(cross_plane[0])
        axes[1, 1].set_ylabel(cross_plane[1])
        fig.colorbar(im11, ax=axes[1, 1], label=f"|{field_label}|")

        plt.tight_layout()
        plt.show()

    # Process field for wave equation error (runs locally after Modal returns)
    print("\nComputing wave equation error locally...")
    field = field_slices[0]
    source_field_zero = jnp.zeros_like(field, dtype=jnp.complex64)
    field = field.at[:, (0, 1, 5), :, :, :2].set(0)
    field = field.at[:, (2, 3, 4), :, :, :1].set(0)

    # Compute wave equation error in free space (zero conductivity)
    # The field from monitor has z-dimension of 4, so permittivity must match
    field_z_size = field.shape[-1]
    error = hw.solve.wave_equation_error_full(
        field=field,
        freq_band=hw.solve.FreqBand(*freq_band),
        permittivity=jnp.ones((3, sim_shape[0], sim_shape[1], field_z_size)),
        conductivity=jnp.zeros((3, sim_shape[0], sim_shape[1], field_z_size)),  # Zero conductivity for free space
        source_field=source_field_zero,
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

    print(f"✓ Source field generated with shape: {err_src_field.shape}")
    print(f"✓ Input power: {input_power}")

    return err_src_field, input_power


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


def create_gaussian_source_on_modal(
    sim_shape: Tuple[int, int, int],
    frequencies: jnp.ndarray,
    source_pos: Tuple[int, int, int],
    waist_radius: float,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = 'x',
    x_span: float = 1.0,
    y_span: float = 1.0,
    max_steps: int = 5_000,
    check_every_n: int = 200,
    show_plots: bool = False,
    gpu_type: str = "H100",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create a Gaussian beam source using wave equation error method on Modal GPU.

    This function creates a Gaussian beam source by running an FDTD simulation
    in free space on a Modal GPU and computing the wave equation error to generate
    a clean source field. This follows the exact methodology of create_gaussian_source
    but runs on Modal infrastructure for GPU acceleration.

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
        max_steps: Maximum FDTD steps for error calculation.
        check_every_n: Steps between error checks.
        show_plots: Whether to display field plots.
        gpu_type: Modal GPU type (H100, A100, A10G, T4, L40S, etc.).

    Returns:
        Tuple of (source_field, input_power) where:
        - source_field: Array with shape (num_freqs, 6, x_span, y_span, 1)
        - input_power: Power array for each frequency

    Note:
        This is computationally expensive as it runs an FDTD simulation
        to compute the clean source field via wave equation error.
        Requires Modal deployment: modal deploy hyperwave/simulate_modal.py

        Source generation is performed in free space (no absorption boundaries).
        Add absorption boundaries in your main simulation structure, not here.
    """
    # This function requires Modal deployment from the full hyperwave package
    raise ImportError(
        "create_gaussian_source_on_modal requires Modal deployment from the full hyperwave package. "
        "Use the API workflow with hwc.solve_mode_source() for mode sources instead."
    )

    from . import structure as hwst

    num_freqs = len(frequencies)

    # Convert to integer spans
    x_span = int(x_span)
    y_span = int(y_span)
    z_span = sim_shape[2]

    shape = (x_span, y_span, z_span)
    c = tuple(s // 2 for s in shape)

    # Calculate source offset - use (0, 0, z_pos) to avoid broadcasting issues
    # This is the same fix used in regular simulate_on_modal calls
    source_offset = (0, 0, source_pos[2])

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

    # For Gaussian source generation, we need free space (no absorption)
    # Create raw arrays and a custom recipe that includes them directly
    print(f"\nCreating air structure with sim_shape: {sim_shape}")

    # Create uniform air permittivity and zero conductivity directly
    p_air = jnp.ones((3, sim_shape[0], sim_shape[1], sim_shape[2]))
    c_air = jnp.zeros((3, sim_shape[0], sim_shape[1], sim_shape[2]))

    # Create a custom recipe structure that includes raw arrays
    # This bypasses layer reconstruction
    import numpy as np
    custom_recipe = {
        'raw_arrays': True,  # Flag to indicate this uses raw arrays
        'permittivity': np.array(p_air).tolist(),  # Convert to list for JSON serialization
        'conductivity': np.array(c_air).tolist(),
        'metadata': {
            'final_shape': list(p_air.shape),
            'direct_arrays': True
        }
    }

    print(f"Created custom recipe with shape: {p_air.shape}")

    # Set up monitors (same as local version)
    monitors = hwm.MonitorSet()
    monitors.add(
        hwm.Monitor(
            shape=(sim_shape[0], sim_shape[1], 4),
            offset=(0, 0, source_offset[2] - 1)
        ),
        name="xy"
    )
    monitors.add(
        hwm.Monitor(
            shape=(sim_shape[0], 1, sim_shape[2]),
            offset=(0, sim_shape[1]//2, 0)
        ),
        name="xz"
    )

    # Get frequency band
    freq_min = jnp.min(frequencies)
    freq_max = jnp.max(frequencies)
    freq_band = (freq_min, freq_max, num_freqs)

    # Run simulation on Modal GPU
    print(f"Running Gaussian source generation on Modal {gpu_type}...")
    response = simulate_on_modal(
        structure_recipe=custom_recipe,
        source_field=initial_source_field,
        source_offset=source_offset,
        freq_band=freq_band,
        monitors=monitors,
        mode_info=None,
        max_steps=max_steps,
        check_every_n=check_every_n,
        source_ramp_periods=10,
        add_absorption=False,  # No absorption for free space
        gpu_type=gpu_type
    )

    # Extract field slices from response (same order as local version)
    field_slices = response['monitor_data']

    # Debug: Print field shapes
    print(f"\nField shapes from monitors:")
    print(f"  XY monitor: {field_slices[0].shape}")
    print(f"  XZ monitor: {field_slices[1].shape}")

    # Explicitly free memory
    del structure

    # Plot if requested
    if show_plots:
        xy_field = field_slices[0][0, 0, :, :, 1]
        xy_real = jnp.real(xy_field)
        xy_abs = jnp.abs(xy_field)
        v1 = jnp.max(jnp.abs(xy_real))/10

        # XZ monitor has Y dimension of 1, so access index 0
        xz_field = field_slices[1][0, 0, :, 0, :]
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

    # Compute wave equation error (free space, no conductivity)
    # The field from monitor has z-dimension of 4, so permittivity must match
    field_z_size = field.shape[-1]
    error = hw.solve.wave_equation_error_full(
        field=field,
        freq_band=hw.solve.FreqBand(*freq_band),
        permittivity=jnp.ones((3, sim_shape[0], sim_shape[1], field_z_size)),
        conductivity=None,  # Free space for source generation
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


# NOTE: mode_converter uses local simulation (mem_efficient_multi_freq) instead of Modal GPU.
# Reasons:
# 1. Small simulation size (~2.5M voxels) - runs in 10-20s locally
# 2. Modal overhead (cold start + serialization) would make it slower
# 3. Requires exact permittivity slice from structure - Modal's Layer/recipe system
#    doesn't easily support extruding arbitrary 2D permittivity arrays
#
# The permittivity_slice MUST match the structure where the mode was solved.
# If propagated through a different structure, the H-field won't be the correct
# eigenmode H-field, causing errors in mode overlap calculations.


def mode_converter(
    mode_E_field: jnp.ndarray,
    freq_band: Tuple[float, float, int],
    permittivity_slice: jnp.ndarray,
    propagation_axis: str = 'x',
    x_propagation_length: int = 500,
    absorption_width: int = 20,
    absorption_coeff: float = 4.89e-3,
    max_steps: int = 5000,
    check_every_n: int = 1000,
    source_ramp_periods: float = 5.0,
    visualize: bool = True,
) -> jnp.ndarray:
    """Convert E-only mode field to full E+H field via short waveguide simulation.

    This function takes the E-field output from hws.mode() and creates the
    corresponding H-field by propagating through a short straight waveguide
    that matches the structure where the mode was solved.

    Args:
        mode_E_field: E-field mode pattern from hws.mode() with shape
            (num_freqs, 3, 1, y, z) for x-propagation or (num_freqs, 3, x, 1, z)
            for y-propagation.
        freq_band: Frequency band (min, max, num_points).
        permittivity_slice: 2D permittivity slice from the structure where mode
            was solved. For x-propagation, this should be structure.permittivity[0, x_pos, :, :]
            which gives the YZ cross-section. Shape should match mode field dimensions.
        propagation_axis: Direction of mode propagation ('x' or 'y').
        x_propagation_length: Length along propagation axis in grid units (default: 60).
        absorption_width: Width of absorbing boundaries (default: 20).
        absorption_coeff: Absorption coefficient (default: 4.89e-3).
        max_steps: Maximum simulation steps (default: 10000).
        check_every_n: Convergence check interval (default: 1000).
        source_ramp_periods: Source ramp-up periods (default: 5.0).
        visualize: Show XZ propagation debug plot (default: True).

    Returns:
        Full mode field with shape (num_freqs, 6, 1, y, z) containing
        both E and H field components. Same spatial dimensions as input mode.

    Note:
        The returned field is extracted from a monitor after propagation,
        providing physically consistent E and H fields suitable for mode
        overlap calculations. The permittivity_slice ensures the propagation
        structure exactly matches where the mode was originally solved.
    """
    from . import absorption as hwa
    import matplotlib.pyplot as plt

    # Validate inputs
    if propagation_axis not in ['x', 'y']:
        raise ValueError(f"propagation_axis must be 'x' or 'y', got '{propagation_axis}'")

    # Extract mode dimensions based on propagation axis
    if propagation_axis == 'x':
        # mode_E_field shape: (num_freqs, 3, 1, y, z)
        _, _, _, mode_y, mode_z = mode_E_field.shape
        mode_perp = mode_y
        mode_vert = mode_z
    else:
        # mode_E_field shape: (num_freqs, 3, x, 1, z)
        _, _, mode_x, _, mode_z = mode_E_field.shape
        mode_perp = mode_x
        mode_vert = mode_z

    # Validate permittivity_slice dimensions match mode field
    if permittivity_slice.shape != (mode_perp, mode_vert):
        raise ValueError(
            f"permittivity_slice shape {permittivity_slice.shape} doesn't match "
            f"mode field dimensions ({mode_perp}, {mode_vert})"
        )

    # Total X dimension includes absorbers on both ends plus propagation region
    total_x = 2 * absorption_width + x_propagation_length
    total_x = total_x + (total_x % 2)  # Make even

    # Build permittivity array by extruding the slice along propagation axis
    if propagation_axis == 'x':
        # Extrude YZ slice along X: (3, x, y, z)
        # Tile the slice for all X positions
        eps_2d = permittivity_slice[jnp.newaxis, :, :]  # (1, y, z)
        eps_2d = jnp.tile(eps_2d, (total_x, 1, 1))  # (x, y, z)
        eps = jnp.stack([eps_2d, eps_2d, eps_2d], axis=0)  # (3, x, y, z)
    else:
        # Extrude XZ slice along Y: (3, x, y, z)
        eps_2d = permittivity_slice[:, jnp.newaxis, :]  # (x, 1, z)
        eps_2d = jnp.tile(eps_2d, (1, total_x, 1))  # (x, y, z)
        eps = jnp.stack([eps_2d, eps_2d, eps_2d], axis=0)  # (3, x, y, z)

    # Create conductivity (start with zeros)
    cond = jnp.zeros_like(eps)

    # Add absorption boundaries
    if propagation_axis == 'x':
        grid_shape = (total_x, mode_perp, mode_vert)
        abs_widths = (absorption_width, absorption_width // 2, absorption_width // 2)
    else:
        grid_shape = (mode_perp, total_x, mode_vert)
        abs_widths = (absorption_width // 2, absorption_width, absorption_width // 2)

    absorption_mask = hwa.create_absorption_mask(
        grid_shape=grid_shape,
        absorption_widths=abs_widths,
        absorption_coeff=absorption_coeff,
        show_plots=False
    )
    cond = cond + absorption_mask

    # Create source field - use mode E-field directly (no padding needed)
    if propagation_axis == 'x':
        # Create source with E-field only (H zeros)
        source_field = jnp.concatenate([mode_E_field, jnp.zeros_like(mode_E_field)], axis=1)
        # Source position: just after the left absorber
        source_offset = (absorption_width + 5, 0, 0)
        # Monitor position: before the right absorber
        monitor_x = total_x - absorption_width - 10
        monitor_shape = (1, mode_perp, mode_vert)
        monitor_offset = (monitor_x, 0, 0)
    else:
        source_field = jnp.concatenate([mode_E_field, jnp.zeros_like(mode_E_field)], axis=1)
        source_offset = (0, absorption_width + 5, 0)
        monitor_y = total_x - absorption_width - 10
        monitor_shape = (mode_perp, 1, mode_vert)
        monitor_offset = (0, monitor_y, 0)

    # Set up monitors
    output_monitor = hwm.Monitor(shape=monitor_shape, offset=monitor_offset)

    if visualize and propagation_axis == 'x':
        # Add XZ slice monitor for visualization
        y_center = mode_perp // 2
        xz_monitor = hwm.Monitor(shape=(total_x, 1, mode_vert), offset=(0, y_center, 0))
        monitors = [xz_monitor, output_monitor]
    else:
        monitors = [output_monitor]

    # Run simulation

    out_list, steps, errs = hws.mem_efficient_multi_freq(
        freq_band=freq_band,
        permittivity=eps,
        conductivity=cond,
        source_field=source_field,
        source_offset=source_offset,
        monitors=monitors,
        source_ramp_periods=source_ramp_periods,
        max_steps=max_steps,
        check_every_n=check_every_n,
    )

    # Extract output field
    if visualize and propagation_axis == 'x':
        xz_field = out_list[0]
        full_mode_field = out_list[1]

        # Visualization
        xz_E = xz_field[0, 0:3, :, 0, :]
        xz_H = xz_field[0, 3:6, :, 0, :]
        output_E = full_mode_field[0, 0:3, 0, :, :]
        output_H = full_mode_field[0, 3:6, 0, :, :]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # XZ Permittivity
        y_center = mode_perp // 2
        ax = axes[0, 0]
        eps_xz = eps[0, :, y_center, :]
        im = ax.imshow(eps_xz.T, aspect='auto', origin='upper', cmap='PuOr')
        ax.axvline(x=absorption_width, color='yellow', linestyle='--', linewidth=2, label='Absorber')
        ax.axvline(x=total_x - absorption_width, color='yellow', linestyle='--', linewidth=2)
        ax.axvline(x=source_offset[0], color='green', linewidth=2, label='Source')
        ax.axvline(x=monitor_x, color='red', linewidth=2, label='Monitor')
        ax.set_title('XZ Permittivity')
        ax.set_xlabel('X'); ax.set_ylabel('Z')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax)

        # XZ |E|^2
        ax = axes[0, 1]
        E_intensity = jnp.sum(jnp.abs(xz_E)**2, axis=0)
        im = ax.imshow(E_intensity.T, aspect='auto', origin='upper', cmap='hot')
        ax.axvline(x=source_offset[0], color='green', linewidth=2)
        ax.axvline(x=monitor_x, color='red', linewidth=2)
        ax.set_title(f'XZ |E|^2 (max={float(jnp.max(E_intensity)):.6f})')
        ax.set_xlabel('X'); ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax)

        # XZ |H|^2
        ax = axes[0, 2]
        H_intensity = jnp.sum(jnp.abs(xz_H)**2, axis=0)
        im = ax.imshow(H_intensity.T, aspect='auto', origin='upper', cmap='viridis')
        ax.axvline(x=source_offset[0], color='green', linewidth=2)
        ax.axvline(x=monitor_x, color='red', linewidth=2)
        ax.set_title(f'XZ |H|^2 (max={float(jnp.max(H_intensity)):.6f})')
        ax.set_xlabel('X'); ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax)

        # Field along X
        ax = axes[1, 0]
        z_center = mode_vert // 2
        E_line = jnp.sqrt(jnp.sum(jnp.abs(xz_E[:, :, z_center])**2, axis=0))
        H_line = jnp.sqrt(jnp.sum(jnp.abs(xz_H[:, :, z_center])**2, axis=0))
        ax.plot(range(total_x), E_line, 'r-', label='|E|', linewidth=2)
        ax.plot(range(total_x), H_line, 'b-', label='|H|', linewidth=2)
        ax.axvline(x=absorption_width, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=total_x - absorption_width, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=source_offset[0], color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=monitor_x, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('X'); ax.set_ylabel('Field magnitude')
        ax.set_title('Field along X (z=center)')
        ax.legend(); ax.grid(True, alpha=0.3)

        # Input |E|^2
        ax = axes[1, 1]
        E_in_int = jnp.sum(jnp.abs(mode_E_field[0, :, 0, :, :])**2, axis=0)
        im = ax.imshow(E_in_int.T, aspect='auto', origin='upper', cmap='hot')
        ax.set_title(f'Input |E|^2 (max={float(jnp.max(E_in_int)):.6f})')
        ax.set_xlabel('Y'); ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax)

        # Output |E|^2
        ax = axes[1, 2]
        E_out_int = jnp.sum(jnp.abs(output_E)**2, axis=0)
        im = ax.imshow(E_out_int.T, aspect='auto', origin='upper', cmap='hot')
        ax.set_title(f'Output |E|^2 (max={float(jnp.max(E_out_int)):.6f})')
        ax.set_xlabel('Y'); ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax)

        plt.suptitle('Mode Converter Propagation', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Print stats
        print(f"\nMode converter results:")
        print(f"  Simulation: {steps[-1]} steps, final error: {jnp.max(errs[-1]):.6f}")
        print(f"  Input max |E|:  {float(jnp.max(jnp.abs(mode_E_field))):.6f}")
        print(f"  Output max |E|: {float(jnp.max(jnp.abs(output_E))):.6f}")
        print(f"  Output max |H|: {float(jnp.max(jnp.abs(output_H))):.6f}")
        print(f"  H/E ratio: {float(jnp.max(jnp.abs(output_H))/jnp.max(jnp.abs(output_E))):.4f}")
    else:
        full_mode_field = out_list[0]

    return full_mode_field


def create_mode_and_absorber_params(
    structure,
    freq_band: Tuple[float, float, int],
    mode_num: int = 0,
    source_position: int = 65,
    propagation_axis: str = "x",
    n_bayesian_calls: int = 30,
    visualize: bool = True,
) -> Dict:
    """Create unidirectional mode source and Bayesian-optimize absorber parameters.

    This function combines mode solving with Bayesian optimization to find optimal
    absorber parameters that minimize standing waves. It uses FFT-based periodicity
    detection to quantify standing wave content.

    Args:
        structure: Hyperwave Structure object with permittivity
        freq_band: (omega_min, omega_max, num_freqs) frequency band tuple
        mode_num: Mode number to solve for (0 = fundamental)
        source_position: Position along propagation axis for source
        propagation_axis: Propagation direction ('x' or 'y')
        n_bayesian_calls: Number of Bayesian optimization iterations
        visualize: Whether to show visualization plots

    Returns:
        Dict with keys:
            'source_field': Unidirectional source field array (num_freqs, 6, 1, y, z)
            'source_offset': (x, y, z) offset tuple for source placement
            'mode_info': Mode solver info dict with 'field', 'beta', 'error'
            'absorber_width': Optimized absorber width in pixels
            'absorber_coeff': Optimized absorber coefficient
            'waveguide_info': Dict with y_bounds, z_bounds, monitor info
    """
    # This function requires Modal deployment from the full hyperwave package
    raise ImportError(
        "create_mode_and_absorber_params requires the full hyperwave package with Modal deployment. "
        "Use hwc.get_default_absorber_params() to get recommended absorber parameters instead."
    )

    _, Lx, Ly, Lz = structure.permittivity.shape

    # Step 1: Detect waveguide using MonitorSet
    temp_monitors = hwm.MonitorSet()
    temp_monitors.add_monitors_at_position(
        structure=structure,
        axis=propagation_axis,
        position=source_position,
        label="source"
    )
    source_monitor = temp_monitors.monitors[0]

    # Step 2: Expand bounds to 2x size for mode solving
    y_min_orig = source_monitor.offset[1]
    y_max_orig = y_min_orig + source_monitor.shape[1]
    z_min_orig = source_monitor.offset[2]
    z_max_orig = z_min_orig + source_monitor.shape[2]

    y_center = (y_min_orig + y_max_orig) // 2
    z_center = (z_min_orig + z_max_orig) // 2
    y_half = source_monitor.shape[1]
    z_half = source_monitor.shape[2]

    y_min = max(0, y_center - y_half)
    y_max = min(Ly, y_center + y_half)
    z_min = max(0, z_center - z_half)
    z_max = min(Lz, z_center + z_half)

    # Step 3: Solve for E-only mode
    source_field_E_only, source_offset, mode_info = hwsim.create_mode_source(
        structure=structure,
        freq_band=freq_band,
        mode_num=mode_num,
        propagation_axis=propagation_axis,
        source_position=source_position,
        perpendicular_bounds=(y_min, y_max),
        z_bounds=(z_min, z_max),
        visualize=visualize,
    )

    # Extract permittivity slice and mode field
    if propagation_axis == 'x':
        eps_slice = structure.permittivity[0, source_position, y_min:y_max, z_min:z_max]
    else:
        eps_slice = structure.permittivity[0, y_min:y_max, source_position, z_min:z_max]

    mode_E_field = mode_info['field'][:, :, :, y_min:y_max, z_min:z_max]

    # Step 4: Batch Bayesian optimization on Modal GPUs
    print("Running batch Bayesian optimization (6 parallel x 5 rounds = 30 evals)...")

    # Serialize data for Modal
    mode_E_field_np = np.array(mode_E_field)
    eps_slice_np = np.array(eps_slice)

    opt = Optimizer(
        dimensions=[
            Integer(20, 100, name='absorption_width'),
            Real(1e-5, 1e-2, prior='log-uniform', name='absorption_coeff'),
        ],
        base_estimator='GP',
        acq_func='EI',
        n_initial_points=6,
        random_state=42,
    )

    batch_size = 6
    n_batches = 5

    for batch_num in range(n_batches):
        print(f"\nBatch {batch_num + 1}/{n_batches}:")
        points = opt.ask(n_points=batch_size)

        # Evaluate batch in parallel on Modal
        metrics = evaluate_absorber_params_on_modal(
            points=points,
            mode_E_field=mode_E_field_np,
            eps_slice=eps_slice_np,
            freq_band=freq_band,
        )

        for i, (p, m) in enumerate(zip(points, metrics)):
            print(f"  width={int(p[0])}, coeff={p[1]:.6f} -> metric={m:.6f}")

        opt.tell(points, metrics)

    best_idx = np.argmin(opt.yi)
    best_width = int(opt.Xi[best_idx][0])
    best_coeff = float(opt.Xi[best_idx][1])
    print(f"\nOptimal absorber params: width={best_width}, coeff={best_coeff:.6f}")

    # Step 5: Generate final unidirectional source with optimal params
    source_field_unidirectional = mode_converter(
        mode_E_field=mode_E_field,
        freq_band=freq_band,
        permittivity_slice=eps_slice,
        propagation_axis=propagation_axis,
        x_propagation_length=500,
        absorption_width=best_width,
        absorption_coeff=best_coeff,
        max_steps=5000,
        check_every_n=1000,
        visualize=visualize,
    )

    return {
        'source_field': source_field_unidirectional,
        'source_offset': source_offset,
        'mode_info': mode_info,
        'absorber_width': best_width,
        'absorber_coeff': best_coeff,
        'waveguide_info': {
            'y_bounds': (y_min, y_max),
            'z_bounds': (z_min, z_max),
            'monitor_shape': source_monitor.shape,
            'monitor_offset': source_monitor.offset,
        }
    }