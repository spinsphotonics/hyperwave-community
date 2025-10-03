"""Source generation for photonic simulations.

This module provides functions for creating electromagnetic sources for FDTD simulations,
including modal sources (waveguide modes) and Gaussian beam sources.

Main functions:
    create_mode_source: Generate waveguide mode source (local eigenvalue solver)
    create_gaussian_source: Generate unidirectional Gaussian source (API call)
    mode: Low-level mode solver for advanced users
"""

from functools import partial
from math import prod
from typing import NamedTuple, Tuple, Dict, Optional

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class FreqBand(NamedTuple):
    """Frequency band specification.

    Describes `num` regularly spaced frequency values within `[start, stop]`.
    For `num=1`, represents a single frequency at `(start + stop) / 2`.

    Attributes:
        start: Lower frequency bound (angular frequency in rad/s).
        stop: Upper frequency bound (angular frequency in rad/s).
        num: Number of equally-spaced frequencies.
    """
    start: float
    stop: float
    num: int

    @property
    def values(self) -> jax.Array:
        """Get array of frequency values.

        Returns:
            Array of shape (num,) containing frequency values.
        """
        if self.num == 1:
            return jnp.array([(self.start + self.stop) / 2])
        return jnp.linspace(self.start, self.stop, self.num)


def mode(
    freq_band: Tuple[float, float, int],
    permittivity: jax.Array,
    axis: int,
    mode_num: int,
    random_seed: int = 0,
    min_modes_in_solve: int = 10,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Solve for waveguide propagating modes using eigenvalue decomposition.

    Computes electromagnetic modes in a waveguide by solving the eigenvalue problem
    in the transverse plane perpendicular to the propagation direction.

    Args:
        freq_band: Frequency specification as (start, stop, num_points).
            Values are angular frequencies in rad/s.
        permittivity: Permittivity distribution with shape (3, xx, yy, zz).
            First dimension corresponds to x, y, z components.
        axis: Propagation axis where 0=x, 1=y, 2=z.
        mode_num: Mode number to extract (0 for fundamental mode).
        random_seed: Random seed for eigenvalue solver initialization.
        min_modes_in_solve: Minimum number of modes to include in eigenvalue solve.
            Solver computes max(min_modes_in_solve, 2*(mode_num+1)) modes for accuracy.

    Returns:
        Tuple of (field, beta, error) where:
            - field: E-field mode pattern, shape (num_freqs, 3, xx, yy, zz)
            - beta: Propagation constants, shape (num_freqs,)
            - error: Eigenvalue solver errors, shape (num_freqs,)

    Note:
        Performance scales with cross-sectional area. Typical times:
        - Small waveguide (50×100 pixels): ~100ms
        - Large device (200×500 pixels): ~3-5 seconds

    Example:
        >>> # Solve for fundamental mode in silicon waveguide
        >>> freq_band = (2*jnp.pi/1.6, 2*jnp.pi/1.5, 3)  # 1.5-1.6 μm
        >>> eps = create_waveguide_permittivity()  # Shape (3, x, y, z)
        >>> field, beta, err = mode(freq_band, eps, axis=0, mode_num=0)
        >>> print(f"Propagation constant: {beta[0]:.4f}")
    """
    freq_band = FreqBand(*freq_band)
    field_shape = permittivity.shape[1:]
    shape = (2 * prod(field_shape), min_modes_in_solve)

    # Solve for either `min_modes_in_solve` modes, or else twice as many modes
    # as needed, whichever is greater. Solving for more modes improves accuracy.
    min_modes_in_solve = max(min_modes_in_solve, (mode_num + 1) * 2)

    # Initial value for eigenvalue problem
    x = jax.random.normal(jax.random.PRNGKey(random_seed), shape)

    # Solve eigenvalue problem at each frequency
    errs, modes, betas = [], [], []
    for omega in freq_band.values:
        op = _wg_operator(omega, permittivity, axis)
        betas_squared, u, _ = lobpcg_standard(op, x)
        err = jnp.linalg.norm(op(u) - betas_squared * u, axis=0)
        mode = jnp.reshape(u.T, (-1, 2) + field_shape)

        errs.append(err[mode_num])
        modes.append(mode[mode_num])
        betas.append(jnp.sqrt(betas_squared[mode_num]))

    errs = jnp.stack(errs)
    betas = jnp.stack(betas)
    modes = jnp.stack(modes)

    # Assign mode patterns to correct field components based on propagation axis
    fields = jnp.zeros((modes.shape[0], 3) + modes.shape[-3:])
    if axis == 0:  # X-propagation
        fields = fields.at[:, 1].set(modes[:, 1])
        fields = fields.at[:, 2].set(modes[:, 0])
    elif axis == 1:  # Y-propagation
        fields = fields.at[:, 0].set(modes[:, 1])
        fields = fields.at[:, 2].set(modes[:, 0])
    elif axis == 2:  # Z-propagation
        fields = fields.at[:, 0].set(modes[:, 1])
        fields = fields.at[:, 1].set(modes[:, 0])

    return fields, betas, errs


def create_mode_source(
    structure,
    freq_band: Tuple[float, float, int],
    mode_num: int = 0,
    propagation_axis: str = 'x',
    source_position: int = 10,
    perpendicular_bounds: Optional[Tuple[int, int]] = None,
    visualize: bool = False,
    visualize_permittivity: bool = False,
    debug: bool = False,
) -> Tuple[jnp.ndarray, Tuple[int, int, int], Dict]:
    """Create modal source for waveguide simulation.

    Generates electromagnetic source field by computing the propagating mode of a
    waveguide structure. The mode is solved at a specified position along the
    propagation axis.

    Args:
        structure: Structure object with permittivity attribute of shape (3, x, y, z).
        freq_band: Frequency specification as (min, max, num_points).
            Values are angular frequencies in rad/s.
        mode_num: Mode number to solve for (0=fundamental, 1=first higher order, etc.).
        propagation_axis: Direction of mode propagation, either 'x' or 'y'.
        source_position: Position along propagation axis where mode is solved (in pixels).
        perpendicular_bounds: Optional bounds in perpendicular direction as (min, max).
            For x-propagation: specifies Y bounds (y_min, y_max).
            For y-propagation: specifies X bounds (x_min, x_max).
            If None, uses full extent of structure.
        visualize: If True, display mode field profile.
        visualize_permittivity: If True, display permittivity at source plane.
        debug: If True, print detailed solver information.

    Returns:
        Tuple of (source_field, source_offset, mode_info) where:
            - source_field: Complex field array, shape (num_freqs, 6, x, y, z).
              First 3 components are E-field, last 3 are H-field (zeros).
            - source_offset: Corner position (x, y, z) for source placement in simulation.
            - mode_info: Dictionary containing:
                * 'field': E-field mode pattern
                * 'beta': Propagation constant
                * 'error': Eigenvalue solver error

    Raises:
        ValueError: If propagation_axis is not 'x' or 'y'.

    Note:
        This function runs locally using CPU eigenvalue solver. Computation time
        scales with waveguide cross-section:
        - Small (<100×200 pixels): ~100-500ms
        - Large (>200×500 pixels): ~3-5 seconds

    Example:
        >>> import hyperwave_community as hwc
        >>> # Create waveguide structure
        >>> structure = hwc.create_structure(layers=[...])
        >>>
        >>> # Generate mode source
        >>> freq_band = (2*jnp.pi/1.6, 2*jnp.pi/1.5, 2)
        >>> source_field, offset, mode_info = hwc.create_mode_source(
        ...     structure=structure,
        ...     freq_band=freq_band,
        ...     mode_num=0,
        ...     propagation_axis='x',
        ...     source_position=80
        ... )
        >>> print(f"Mode beta: {mode_info['beta']}")
    """
    # Validate propagation axis
    if propagation_axis not in ['x', 'y']:
        raise ValueError(f"propagation_axis must be 'x' or 'y', got '{propagation_axis}'")

    # Get full structure dimensions
    _, full_x_size, full_y_size, full_z_size = structure.permittivity.shape

    # Process bounds and extract source plane based on propagation direction
    if propagation_axis == 'x':
        # X-propagating mode (horizontal)
        y_min, y_max = perpendicular_bounds if perpendicular_bounds else (0, full_y_size)

        # Extract YZ plane at source_position in X (always full Z)
        if perpendicular_bounds:
            source_plane = structure.permittivity[:, source_position:source_position + 1, y_min:y_max, :]
            if debug:
                print(f"X-propagating mode in Y region: [{y_min}:{y_max}]")
        else:
            source_plane = structure.permittivity[:, source_position:source_position + 1, :, :]

        solve_axis = 0  # Solve along X axis

    else:  # propagation_axis == 'y'
        # Y-propagating mode (vertical)
        x_min, x_max = perpendicular_bounds if perpendicular_bounds else (0, full_x_size)

        # Extract XZ plane at source_position in Y (always full Z)
        if perpendicular_bounds:
            source_plane = structure.permittivity[:, x_min:x_max, source_position:source_position + 1, :]
            if debug:
                print(f"Y-propagating mode in X region: [{x_min}:{x_max}]")
        else:
            source_plane = structure.permittivity[:, :, source_position:source_position + 1, :]

        solve_axis = 1  # Solve along Y axis

    if debug:
        print(f"\n=== Mode {mode_num} Solver ===")
        print(f"Propagation axis: {propagation_axis}")
        print(f"Source position: {propagation_axis}={source_position}")
        print(f"Source plane shape: {source_plane.shape}")

    # Solve for mode
    mode_E_field, beta, err = mode(
        freq_band=freq_band,
        permittivity=source_plane,
        axis=solve_axis,
        mode_num=mode_num
    )

    if debug:
        print(f"Mode beta: {beta}")
        print(f"Mode error: {err}")
        mode_slice = jnp.squeeze(mode_E_field[0, :, 0, :, :])
        print(f"Mode field shape: {mode_slice.shape}")

    # Enhanced visualization if requested
    if visualize and MATPLOTLIB_AVAILABLE:
        # Extract the correct slice based on propagation axis
        if propagation_axis == 'x':
            mode_slice = jnp.squeeze(mode_E_field[0, :, 0, :, :])
            xlabel, ylabel = 'Y (cells)', 'Z (cells)'
        else:  # y propagation
            mode_slice = jnp.squeeze(mode_E_field[0, :, :, 0, :])
            xlabel, ylabel = 'X (cells)', 'Z (cells)'

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Mode {mode_num} E-field Profile (β = {float(beta[0]):.4f})", fontsize=14)

        component_names = ['Ex', 'Ey', 'Ez']
        for i, (ax, comp_name) in enumerate(zip(axes, component_names)):
            mode_comp = mode_slice[i, :, :]
            mode_mag = jnp.abs(mode_comp)
            vmax = float(jnp.max(mode_mag)) or 1.0

            im = ax.imshow(mode_mag.T, cmap='viridis', origin='upper',
                          vmin=0, vmax=vmax, aspect='auto')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{comp_name} Magnitude')
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

    # Visualize permittivity at source plane if requested
    if visualize_permittivity and MATPLOTLIB_AVAILABLE:
        if propagation_axis == 'x':
            source_eps = jnp.real(source_plane[0, 0, :, :])
            xlabel, ylabel = 'Y (cells)', 'Z (cells)'
        else:  # y propagation
            source_eps = jnp.real(source_plane[0, :, 0, :])
            xlabel, ylabel = 'X (cells)', 'Z (cells)'

        plt.figure(figsize=(8, 6))
        plt.imshow(source_eps.T, cmap='PuOr', origin='upper', aspect='auto')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Permittivity at Source Plane ({propagation_axis}={source_position})')
        plt.colorbar(label='Real(ε)')
        plt.show()

    # If we solved in a restricted region, pad the mode field back to full size
    if perpendicular_bounds is not None:
        if propagation_axis == 'x':
            # Create zero array of full YZ size for X-propagating mode
            full_mode_E_field = jnp.zeros((mode_E_field.shape[0], mode_E_field.shape[1],
                                          1, full_y_size, full_z_size), dtype=mode_E_field.dtype)

            # Place the solved mode in the correct Y region (Z is already full)
            full_mode_E_field = full_mode_E_field.at[:, :, 0, y_min:y_max, :].set(
                mode_E_field[:, :, 0, :, :]
            )
            mode_E_field = full_mode_E_field

            if debug:
                print(f"Padded mode field from Y size {y_max - y_min} to full Y size {full_y_size}")

        else:  # propagation_axis == 'y'
            # Create zero array of full XZ size for Y-propagating mode
            full_mode_E_field = jnp.zeros((mode_E_field.shape[0], mode_E_field.shape[1],
                                          full_x_size, 1, full_z_size), dtype=mode_E_field.dtype)

            # Place the solved mode in the correct X region (Z is already full)
            full_mode_E_field = full_mode_E_field.at[:, :, x_min:x_max, 0, :].set(
                mode_E_field[:, :, :, 0, :]
            )
            mode_E_field = full_mode_E_field

            if debug:
                print(f"Padded mode field from X size {x_max - x_min} to full X size {full_x_size}")

    # Create full source field (E and H components)
    source_field = jnp.concatenate([mode_E_field, jnp.zeros_like(mode_E_field)], axis=1)

    # Set source offset based on propagation axis
    if propagation_axis == 'x':
        # Source is 1 pixel thick in X, full Y and Z
        source_offset = (source_position, 0, 0)
    else:  # propagation_axis == 'y'
        # Source is 1 pixel thick in Y, full X and Z
        source_offset = (0, source_position, 0)

    mode_info = {'field': mode_E_field, 'beta': beta, 'error': err}

    if debug:
        print(f"Source field shape: {source_field.shape}")
        print(f"Source offset: {source_offset}")

    return source_field, source_offset, mode_info


def create_gaussian_source(
    structure_shape: Tuple[int, int, int, int],
    conductivity_boundary: jax.Array,
    freq_band: Tuple[float, float, int],
    source_z_pos: int,
    polarization: str = 'x',
    max_steps: int = 5000,
    check_every_n: int = 1000,
    gpu_type: str = "H100",
    api_key: Optional[str] = None,
) -> Tuple[jax.Array, Tuple[int, int, int], Dict]:
    """Create unidirectional Gaussian beam source via API.

    Generates a truly unidirectional Gaussian beam using the wave equation error
    method. This prevents backward propagation artifacts that occur with standard
    analytical Gaussian sources.

    **Note**: This function requires network access and API credentials. It submits
    a job to GPU servers for computation (~20-30 seconds).

    Args:
        structure_shape: Shape of simulation domain as (3, Lx, Ly, Lz).
        conductivity_boundary: Absorption boundary mask, shape (Lx, Ly, Lz).
        freq_band: Frequency specification as (min, max, num_points).
            Values are angular frequencies in rad/s.
        source_z_pos: Z-position for Gaussian source injection (in pixels).
        polarization: Polarization direction, either 'x' or 'y'.
        max_steps: Maximum FDTD steps for source generation.
        check_every_n: Convergence check interval.
        gpu_type: GPU type to use (H100, A100, A10G, L4).
        api_key: API authentication key. If None, reads from HYPERWAVE_API_KEY
            environment variable.

    Returns:
        Tuple of (source_field, source_offset, source_info) where:
            - source_field: Complex field array, shape (num_freqs, 6, x, y, z)
            - source_offset: Corner position (x, y, z) for source placement
            - source_info: Dict with 'power', 'total_time', 'fdtd_time', 'gpu_type'

    Raises:
        RuntimeError: If API call fails or authentication is invalid.
        ValueError: If polarization is not 'x' or 'y'.

    Note:
        Unlike analytical Gaussian beams, this source is truly unidirectional.
        Reflections from structures will not propagate backward through the
        source plane, eliminating common artifacts in metasurface simulations.

    Example:
        >>> import hyperwave_community as hwc
        >>> import os
        >>>
        >>> # Set API key
        >>> api_key = 'your-key-here'
        >>>
        >>> # Create absorption boundaries
        >>> abs_mask = hwc.create_absorption_mask(
        ...     shape=(500, 500, 200),
        ...     absorption_widths=(90, 90, 90)
        ... )
        >>>
        >>> # Generate Gaussian source
        >>> source, offset, info = hwc.create_gaussian_source(
        ...     structure_shape=(3, 500, 500, 200),
        ...     conductivity_boundary=abs_mask,
        ...     freq_band=(2*jnp.pi/0.55, 2*jnp.pi/0.55, 1),
        ...     source_z_pos=60,
        ...     polarization='x',
        ...     api_key=api_key
        ... )
    """
    from . import api_client

    result = api_client.generate_gaussian_source(
        structure_shape=structure_shape,
        conductivity_boundary=conductivity_boundary,
        freq_band=freq_band,
        source_z_pos=source_z_pos,
        polarization=polarization,
        max_steps=max_steps,
        check_every_n=check_every_n,
        gpu_type=gpu_type,
        api_key=api_key
    )

    source_info = {
        'power': result['source_power'],
        'total_time': result['total_time'],
        'fdtd_time': result['fdtd_time'],
        'gpu_type': result['gpu_type']
    }

    return result['source_field'], result['source_position'], source_info


# =============================================================================
# Private helper functions
# =============================================================================

def _wg_operator(omega: float, permittivity: jax.Array, axis: int):
    """Create waveguide operator for mode eigenvalue problem.

    Constructs the differential operator for solving waveguide modes. The operator
    acts on H-fields in the transverse plane.

    Args:
        omega: Angular frequency in rad/s.
        permittivity: Permittivity distribution, shape (3, xx, yy, zz).
        axis: Propagation axis where 0=x, 1=y, 2=z.

    Returns:
        Operator function that can be used with iterative eigenvalue solvers.
    """
    shape = permittivity.shape[1:]

    dfi, dbi, dfj, dbj = [
        partial(
            _spatial_diff,
            axis=((axis + axis_shift) % 3) - 3,
            is_forward=is_forward,
        )
        for (axis_shift, is_forward) in ((1, True), (1, False), (2, True), (2, False))
    ]

    def _split(u):
        return jnp.split(u, indices_or_sections=2, axis=1)

    def _concat(u):
        return jnp.concatenate(u, axis=1)

    def curl_to_k(u):
        ui, uj = _split(u)
        return dbi(uj) - dbj(ui)

    def curl_to_ij(u):
        return _concat([-dfj(u), dfi(u)])

    def div(u):
        ui, uj = _split(u)
        return dfi(ui) + dfj(uj)

    def grad(u):
        return _concat([dbi(u), dbj(u)])

    ei, ej, ek = tuple(permittivity[(i + 1) % 3] for i in range(3))
    eji = jnp.stack([ej, ei], axis=0)

    def op(x):
        u = jnp.reshape(x.T, (-1, 2) + shape)
        return jnp.reshape(
            omega**2 * eji * u + eji * curl_to_ij(curl_to_k(u) / ek) + grad(div(u)),
            x.shape[::-1],
        ).T

    return op


def _spatial_diff(field: jax.Array, axis: int, is_forward: bool) -> jax.Array:
    """Compute spatial difference along specified axis.

    Args:
        field: Field array to differentiate.
        axis: Axis along which to compute difference.
        is_forward: If True, forward difference; if False, backward difference.

    Returns:
        Differentiated field array with same shape as input.
    """
    if is_forward:
        return jnp.roll(field, shift=-1, axis=axis) - field
    return field - jnp.roll(field, shift=+1, axis=axis)


