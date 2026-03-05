"""High-level simulation wrappers for hyperwave.

This module provides convenient wrapper functions for running FDTD simulations
with the hyperwave solver. It includes support for various source types,
automatic monitor placement, and visualization.
"""

import jax.numpy as jnp
from typing import Tuple, Optional, Dict

from ._logging import logger

# Mode solver (available in hyperwave-community)
try:
    from .mode_solver import mode
    MODE_SOLVER_AVAILABLE = True
except ImportError:
    MODE_SOLVER_AVAILABLE = False
    mode = None


def create_mode_source(
    structure,
    freq_band: Tuple[float, float, int],
    mode_num: int = 0,
    propagation_axis: str = 'x',
    source_position: int = 10,
    perpendicular_bounds: Optional[Tuple[int, int]] = None,
    z_bounds: Optional[Tuple[int, int]] = None,
    visualize: bool = False,
    visualize_permittivity: bool = False,
    debug: bool = False,
) -> Tuple[jnp.ndarray, Tuple[int, int, int], Dict]:
    """Create a modal source for simulation.

    Args:
        structure: Structure object with permittivity.
        freq_band: Frequency band (min, max, num_points).
        mode_num: Mode number to solve for.
        propagation_axis: Direction of mode propagation ('x' or 'y').
        source_position: Position along propagation axis for mode solving.
        perpendicular_bounds: Optional (min, max) bounds for the axis perpendicular to propagation.
                              For x-propagation: Y bounds (y_min, y_max).
                              For y-propagation: X bounds (x_min, x_max).
                              None means use full extent.
        z_bounds: Optional (min, max) bounds for the Z axis.
                  Useful for limiting mode solver to specific layers (e.g., silicon layer in SOI).
                  None means use full Z extent.
        visualize: Whether to visualize the mode field.
        visualize_permittivity: Whether to visualize the permittivity at source plane.
        debug: Whether to print debug information.

    Returns:
        Tuple of (source_field, source_offset, mode_info) where:
            - source_field: Array of shape (num_freqs, 6, spatial_dims...)
            - source_offset: (x, y, z) CORNER position for source placement
            - mode_info: Dict with 'field', 'beta', and 'error'
    """
    # Check if mode solver is available
    if not MODE_SOLVER_AVAILABLE:
        raise ImportError(
            "Mode solver not available. This usually means JAX is not properly installed. "
            "Try: pip install jax jaxlib"
        )

    # Validate propagation axis
    if propagation_axis not in ['x', 'y']:
        raise ValueError(f"propagation_axis must be 'x' or 'y', got '{propagation_axis}'")

    # Get full structure dimensions
    _, full_x_size, full_y_size, full_z_size = structure.permittivity.shape

    # Auto-detect waveguide bounds if not provided
    if perpendicular_bounds is None or z_bounds is None:
        from .monitors import _detect_waveguides

        if propagation_axis == 'x':
            waveguides = _detect_waveguides(
                structure, x_position=source_position, z_position=None, axis='y'
            )
            if waveguides:
                wg = waveguides[0]
                if perpendicular_bounds is None:
                    y_center = wg['center']
                    y_expand = 2 * wg['width']
                    perpendicular_bounds = (
                        max(0, y_center - y_expand),
                        min(full_y_size, y_center + y_expand),
                    )
                    logger.info(
                        "Auto-detected waveguide at y=%d (width=%d), "
                        "using perpendicular_bounds=(%d, %d)",
                        wg['center'], wg['width'],
                        perpendicular_bounds[0], perpendicular_bounds[1],
                    )
                if z_bounds is None and 'z_core' in wg:
                    z_center = wg['z_core']
                    z_expand = 2 * wg['width']
                    z_bounds = (
                        max(0, z_center - z_expand),
                        min(full_z_size, z_center + z_expand),
                    )
                    logger.info(
                        "Auto-detected z_core=%d, using z_bounds=(%d, %d)",
                        z_center, z_bounds[0], z_bounds[1],
                    )
        else:  # propagation_axis == 'y'
            waveguides = _detect_waveguides(
                structure, y_position=source_position, z_position=None, axis='x'
            )
            if waveguides:
                wg = waveguides[0]
                if perpendicular_bounds is None:
                    x_center = wg['center']
                    x_expand = 2 * wg['width']
                    perpendicular_bounds = (
                        max(0, x_center - x_expand),
                        min(full_x_size, x_center + x_expand),
                    )
                    logger.info(
                        "Auto-detected waveguide at x=%d (width=%d), "
                        "using perpendicular_bounds=(%d, %d)",
                        wg['center'], wg['width'],
                        perpendicular_bounds[0], perpendicular_bounds[1],
                    )
                if z_bounds is None and 'z_core' in wg:
                    z_center = wg['z_core']
                    z_expand = 2 * wg['width']
                    z_bounds = (
                        max(0, z_center - z_expand),
                        min(full_z_size, z_center + z_expand),
                    )
                    logger.info(
                        "Auto-detected z_core=%d, using z_bounds=(%d, %d)",
                        z_center, z_bounds[0], z_bounds[1],
                    )

    # Process Z bounds (applies to both propagation directions)
    z_min, z_max = z_bounds if z_bounds else (0, full_z_size)

    # Process bounds and extract source plane based on propagation direction
    if propagation_axis == 'x':
        # X-propagating mode (horizontal)
        # perpendicular_bounds specifies Y bounds for x-propagation
        y_min, y_max = perpendicular_bounds if perpendicular_bounds else (0, full_y_size)

        # Extract YZ plane at source_position in X with both Y and Z bounds
        source_plane = structure.permittivity[:, source_position:source_position + 1, y_min:y_max, z_min:z_max]

        if debug:
            logger.debug(f"X-propagating mode in Y region: [{y_min}:{y_max}], Z region: [{z_min}:{z_max}]")

        solve_axis = 0  # Solve along X axis
        source_offset = (source_position, y_min, z_min)

    else:  # propagation_axis == 'y'
        # Y-propagating mode (vertical)
        # perpendicular_bounds specifies X bounds for y-propagation
        x_min, x_max = perpendicular_bounds if perpendicular_bounds else (0, full_x_size)

        # Extract XZ plane at source_position in Y with both X and Z bounds
        source_plane = structure.permittivity[:, x_min:x_max, source_position:source_position + 1, z_min:z_max]

        if debug:
            logger.debug(f"Y-propagating mode in X region: [{x_min}:{x_max}], Z region: [{z_min}:{z_max}]")

        solve_axis = 1  # Solve along Y axis
        source_offset = (x_min, source_position, z_min)

    if debug:
        logger.debug(f"=== Mode {mode_num} Solver ===")
        logger.debug(f"Propagation axis: {propagation_axis}")
        logger.debug(f"Source position: {propagation_axis}={source_position}")
        logger.debug(f"Source plane shape: {source_plane.shape}")

    # Solve for mode
    mode_E_field, beta, err = mode(
        freq_band=freq_band,
        permittivity=source_plane,
        axis=solve_axis,
        mode_num=mode_num
    )

    if debug:
        logger.debug(f"Mode beta: {beta}")
        logger.debug(f"Mode error: {err}")
        mode_slice = jnp.squeeze(mode_E_field[0, :, 0, :, :])
        logger.debug(f"Mode field shape: {mode_slice.shape}")

    # Enhanced visualization if requested
    if visualize:
        from .visualization import plot_mode
        plot_mode(mode_E_field, beta, mode_num, propagation_axis=propagation_axis)

    # Visualize permittivity at source plane if requested
    if visualize_permittivity:
        from .visualization import plot_structure
        plot_structure(source_plane, axis=propagation_axis, position=source_position)

    # If we solved in a restricted region, pad the mode field back to full size
    if perpendicular_bounds is not None or z_bounds is not None:
        if propagation_axis == 'x':
            # Create zero array of full YZ size for X-propagating mode
            full_mode_E_field = jnp.zeros((mode_E_field.shape[0], mode_E_field.shape[1],
                                          1, full_y_size, full_z_size), dtype=mode_E_field.dtype)

            # Place the solved mode in the correct Y and Z region
            full_mode_E_field = full_mode_E_field.at[:, :, 0, y_min:y_max, z_min:z_max].set(
                mode_E_field[:, :, 0, :, :]
            )
            mode_E_field = full_mode_E_field

            # Reset offset since mode field is now positioned within full array
            source_offset = (source_position, 0, 0)

            if debug:
                logger.debug(f"Padded mode field from Y[{y_max - y_min}] Z[{z_max - z_min}] to full Y[{full_y_size}] Z[{full_z_size}]")

        else:  # propagation_axis == 'y'
            # Create zero array of full XZ size for Y-propagating mode
            full_mode_E_field = jnp.zeros((mode_E_field.shape[0], mode_E_field.shape[1],
                                          full_x_size, 1, full_z_size), dtype=mode_E_field.dtype)

            # Place the solved mode in the correct X and Z region
            full_mode_E_field = full_mode_E_field.at[:, :, x_min:x_max, 0, z_min:z_max].set(
                mode_E_field[:, :, :, 0, :]
            )
            mode_E_field = full_mode_E_field

            # Reset offset since mode field is now positioned within full array
            source_offset = (0, source_position, 0)

            if debug:
                logger.debug(f"Padded mode field from X[{x_max - x_min}] Z[{z_max - z_min}] to full X[{full_x_size}] Z[{full_z_size}]")

    # Create full source field (E and H components)
    source_field = jnp.concatenate([mode_E_field, jnp.zeros_like(mode_E_field)], axis=1)

    # Auto-trim: if bounds were used, trim source field to just the bounds region
    # This reduces data sent to cloud
    if perpendicular_bounds is not None or z_bounds is not None:
        if propagation_axis == 'x':
            y_min_used = perpendicular_bounds[0] if perpendicular_bounds else 0
            y_max_used = perpendicular_bounds[1] if perpendicular_bounds else full_y_size
            z_min_used = z_bounds[0] if z_bounds else 0
            z_max_used = z_bounds[1] if z_bounds else full_z_size
            source_field = source_field[:, :, :, y_min_used:y_max_used, z_min_used:z_max_used]
            source_offset = (source_position, y_min_used, z_min_used)
        else:
            x_min_used = perpendicular_bounds[0] if perpendicular_bounds else 0
            x_max_used = perpendicular_bounds[1] if perpendicular_bounds else full_x_size
            z_min_used = z_bounds[0] if z_bounds else 0
            z_max_used = z_bounds[1] if z_bounds else full_z_size
            source_field = source_field[:, :, x_min_used:x_max_used, :, z_min_used:z_max_used]
            source_offset = (x_min_used, source_position, z_min_used)

    mode_info = {'field': mode_E_field, 'beta': beta, 'error': err}

    logger.info("Source: shape=%s, offset=%s, beta=%s",
                source_field.shape, source_offset, beta)

    return source_field, source_offset, mode_info
    