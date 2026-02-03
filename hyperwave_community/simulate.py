"""High-level simulation wrappers for hyperwave.

This module provides convenient wrapper functions for running FDTD simulations
with the hyperwave solver. It includes support for various source types,
automatic monitor placement, and visualization.
"""

import time
import jax.numpy as jnp
from typing import Tuple, Optional, Dict

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Local solver functions (only available in full hyperwave package)
try:
    from .solve import mem_efficient_multi_freq, mode, gaussian_source
    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False
    mem_efficient_multi_freq = None
    mode = None
    gaussian_source = None

# Use relative imports for hyperwave_community modules
from . import monitors as hwm
from .monitors import MonitorSet
from . import structure as hwst
from . import absorption as hwa


def simulate(
    structure,
    source_field: jnp.ndarray,
    source_offset: Tuple[int, int, int],
    freq_band: Tuple[float, float, int],
    max_steps: int = 10000,
    monitor_positions: Optional[MonitorSet] = None,
    visualize: bool = False,
    field_to_plot: str = 'all',
    check_every_n: int = 1000,
    source_ramp_periods: float = 5.0,
    add_viz_monitors: bool = True,
    debug: bool = False,
) -> Dict:
    """Run FDTD simulation with provided source field.

    Args:
        structure: Structure object with permittivity and conductivity.
        source_field: Source field array of shape (num_freqs, 6, x, y, z).
        source_offset: (x, y, z) CORNER position for source placement.
        freq_band: Frequency band (min, max, num_points).
        max_steps: Maximum simulation steps.
        monitor_positions: MonitorSet object or None for auto-generated default monitors.
        visualize: Whether to show plots during simulation.
        field_to_plot: Field component to visualize. Options:
            - 'all': Show total field intensity (default)
            - 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz': Individual components
            - 'E': Total electric field intensity
            - 'H': Total magnetic field intensity
        check_every_n: Check convergence every n steps.
        source_ramp_periods: Source ramp-up periods.
        add_viz_monitors: Whether to automatically add visualization monitors (xy_mid, xz_mid).
        debug: Whether to print detailed debug information.

    Returns:
        Dictionary with simulation results including:
            - 'monitor_data': Output from all monitors
            - 'monitor_names': Mapping of names to indices
            - 'convergence': (steps, errors) arrays
            - 'performance': Grid points × steps per second
            - 'monitors': MonitorSet object

    Note:
        This function requires the full hyperwave solver which is not included
        in hyperwave-community. For cloud GPU simulation, use the API workflow
        with hwc.run_simulation() or the local workflow with hwc.simulate_local().
    """
    if not SOLVER_AVAILABLE:
        raise ImportError(
            "Local FDTD solver not available in hyperwave-community. "
            "Use the API workflow with hwc.run_simulation() instead, "
            "or install the full hyperwave package for local simulation."
        )

    structure_shape = structure.permittivity.shape
    _, x_dim, y_dim, z_dim = structure_shape

    # Set up monitors
    if monitor_positions is not None:
        # Use the provided MonitorSet directly
        monitors = monitor_positions
    else:
        # Create default monitors
        monitors = hwm.MonitorSet()

        # Add default input/output monitors
        monitors.add_monitors_at_position(
            structure=structure,
            axis="x",
            position=50,
            label="input"
        )
        monitors.add_monitors_at_position(
            structure=structure,
            axis="x",
            position=x_dim - 50,
            label="output"
        )

    # Add visualization monitors at mid-planes if requested and not already present
    if add_viz_monitors:
        y_mid = y_dim // 2
        z_mid = z_dim // 2

        if 'xy_mid' not in monitors.mapping:
            xy_monitor = hwm.Monitor(
                shape=(x_dim, y_dim, 1),
                offset=(0, 0, z_mid)
            )
            monitors.add(xy_monitor, name='xy_mid')

        if 'xz_mid' not in monitors.mapping:
            xz_monitor = hwm.Monitor(
                shape=(x_dim, 1, z_dim),
                offset=(0, y_mid, 0)
            )
            monitors.add(xz_monitor, name='xz_mid')

    if debug:
        print(f"Monitor setup: {monitors.list_monitors()}")
        print(f"Source offset: {source_offset}")
        print(f"Source field shape: {source_field.shape}")
        print(f"Structure shape: {structure_shape}")
    else:
        print(f"Running simulation with {len(monitors)} monitors...")

    # Run simulation
    start_time = time.time()
    out_list, steps, errs = mem_efficient_multi_freq(
        freq_band=freq_band,
        permittivity=structure.permittivity,
        conductivity=structure.conductivity,
        source_field=source_field,
        source_offset=source_offset,
        monitors=monitors.monitors,
        source_ramp_periods=source_ramp_periods,
        max_steps=max_steps,
        check_every_n=check_every_n,
    )
    sim_time = time.time() - start_time

    # Visualize if requested
    if visualize and steps and errs:
        visualize_convergence(steps, errs)

    if visualize:
        visualize_fields(out_list, monitors.mapping, field_component=field_to_plot)

    # Calculate performance
    last_step = steps[-1] if steps else 0
    performance = (x_dim * y_dim * z_dim * last_step) / max(sim_time, 1e-9)

    print(f"Simulation complete in {sim_time:.2f}s")
    if debug:
        print(f"Final step: {last_step}")
        print(f"Final error: {jnp.max(errs[-1]) if errs else 'N/A'}")
        print(f"Performance: {performance:,.0f} grid-points × steps/s")

        # Debug monitor output
        print("\n=== Monitor Debug Info ===")
        print(f"Total monitors in output: {len(out_list)}")
        for name, idx in monitors.mapping.items():
            if idx < len(out_list):
                data = out_list[idx]
                has_data = jnp.any(data != 0) if data.size > 0 else False
                max_val = float(jnp.max(jnp.abs(data))) if data.size > 0 else 0.0
                print(f"  {name}: shape={data.shape}, has_data={has_data}, max={max_val:.2e}")

    return {
        'monitor_data': out_list,
        'monitor_names': monitors.mapping,
        'convergence': (steps, errs),
        'performance': performance,
        'monitors': monitors
    }


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
    # Validate propagation axis
    if propagation_axis not in ['x', 'y']:
        raise ValueError(f"propagation_axis must be 'x' or 'y', got '{propagation_axis}'")

    # Get full structure dimensions
    _, full_x_size, full_y_size, full_z_size = structure.permittivity.shape

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
            print(f"X-propagating mode in Y region: [{y_min}:{y_max}], Z region: [{z_min}:{z_max}]")

        solve_axis = 0  # Solve along X axis
        source_offset = (source_position, y_min, z_min)

    else:  # propagation_axis == 'y'
        # Y-propagating mode (vertical)
        # perpendicular_bounds specifies X bounds for y-propagation
        x_min, x_max = perpendicular_bounds if perpendicular_bounds else (0, full_x_size)

        # Extract XZ plane at source_position in Y with both X and Z bounds
        source_plane = structure.permittivity[:, x_min:x_max, source_position:source_position + 1, z_min:z_max]

        if debug:
            print(f"Y-propagating mode in X region: [{x_min}:{x_max}], Z region: [{z_min}:{z_max}]")

        solve_axis = 1  # Solve along Y axis
        source_offset = (x_min, source_position, z_min)

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

            if debug:
                print(f"Padded mode field from Y[{y_max - y_min}] Z[{z_max - z_min}] to full Y[{full_y_size}] Z[{full_z_size}]")

        else:  # propagation_axis == 'y'
            # Create zero array of full XZ size for Y-propagating mode
            full_mode_E_field = jnp.zeros((mode_E_field.shape[0], mode_E_field.shape[1],
                                          full_x_size, 1, full_z_size), dtype=mode_E_field.dtype)

            # Place the solved mode in the correct X and Z region
            full_mode_E_field = full_mode_E_field.at[:, :, x_min:x_max, 0, z_min:z_max].set(
                mode_E_field[:, :, :, 0, :]
            )
            mode_E_field = full_mode_E_field

            if debug:
                print(f"Padded mode field from X[{x_max - x_min}] Z[{z_max - z_min}] to full X[{full_x_size}] Z[{full_z_size}]")

    # Create full source field (E and H components)
    source_field = jnp.concatenate([mode_E_field, jnp.zeros_like(mode_E_field)], axis=1)

    # source_offset was already set above based on the bounds used
    # It points to the corner of the solved region, not always (0,0,0)

    mode_info = {'field': mode_E_field, 'beta': beta, 'error': err}

    if debug:
        print(f"Source field shape: {source_field.shape}")
        print(f"Source offset: {source_offset}")

    return source_field, source_offset, mode_info


def create_gaussian_source_wrapper(
    structure,
    freq_band: Tuple[float, float, int],
    source_pos: Tuple[float, float, float] = (0, 0, 0),
    r_waist: float = 5.2,
    theta: float = 0.0,
    phi: float = 0.0,
    dz: float = 0.08,
    max_steps: int = 5000,
    check_every_n: int = 1000,
) -> Tuple[jnp.ndarray, Tuple[int, int, int], Dict]:
    """Create a Gaussian source for simulation.

    Args:
        structure: Structure object with permittivity and conductivity.
        freq_band: Frequency band (min, max, num_points).
        source_pos: (x, y, z) position of source center in um.
        r_waist: Waist radius of Gaussian beam in um.
        theta: Tilt angle in degrees.
        phi: Propagation angle in degrees.
        dz: Grid spacing in um.
        max_steps: Maximum FDTD steps for field calculation.
        check_every_n: Check convergence every n steps.

    Returns:
        Tuple of (source_field, source_offset, source_info) where:
            - source_field: Array of shape (num_freqs, 6, x, y, z)
            - source_offset: (x, y, z) CORNER position for source placement
            - source_info: Dict with 'input_power'
    """
    structure_shape = structure.permittivity.shape

    # Create Gaussian source
    source_field, input_power = gaussian_source(
        sim_shape=structure_shape[1:],
        freq_band=freq_band,
        source_pos=source_pos,
        r_waist=r_waist,
        theta=theta,
        phi=phi,
        dz=dz,
        permittivity=structure.permittivity,
        conductivity=structure.conductivity,
        max_steps=max_steps,
        check_every_n=check_every_n,
    )

    # The gaussian_source function already returns center-based offsets
    # so we just convert from physical units to grid units
    source_offset = (
        int(source_pos[0] / dz),
        int(source_pos[1] / dz),
        int(source_pos[2] / dz)
    )

    source_info = {'input_power': input_power}

    return source_field, source_offset, source_info


# Visualization functions specific to simulate module
def visualize_convergence(steps, errs, figsize=(12, 5)):
    """Enhanced convergence visualization with multiple metrics.

    Args:
        steps: Array of step numbers.
        errs: Array of error values.
        figsize: Figure size tuple.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    e = jnp.array([jnp.max(er) for er in errs])

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Main convergence plot
    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(steps, e, ".-", color='blue', linewidth=2, markersize=8)
    ax1.set_title("FDTD Convergence History", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Simulation Steps", fontsize=12)
    ax1.set_ylabel("Maximum Error", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # Add convergence rate annotation if enough data points
    if len(steps) > 10:
        # Calculate average convergence rate
        mid_idx = len(steps) // 2
        if mid_idx > 0 and e[mid_idx] > 0:
            conv_rate = (jnp.log10(e[-1]) - jnp.log10(e[mid_idx])) / (steps[-1] - steps[mid_idx])
            ax1.text(0.95, 0.95, f'Conv. Rate: {conv_rate:.2e}/step',
                    transform=ax1.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Statistics panel
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    stats_text = f"""Convergence Statistics

    Final Error: {float(e[-1]):.2e}
    Initial Error: {float(e[0]):.2e}
    Reduction: {float(e[0]/e[-1]):.1f}x
    Total Steps: {steps[-1]}

    Min Error: {float(jnp.min(e)):.2e}
    Max Error: {float(jnp.max(e)):.2e}
    """

    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    plt.show()


def visualize_fields(out_list, monitor_mapping, field_component='all', freq_idx=0):
    """Enhanced field visualization with multiple component options.

    Args:
        out_list: List of monitor outputs.
        monitor_mapping: Dictionary mapping monitor names to indices.
        field_component: Which field component to visualize:
            - 'all': Total field intensity (|E|² + |H|²)
            - 'E': Electric field intensity (|E|²)
            - 'H': Magnetic field intensity (|H|²)
            - 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz': Individual components
        freq_idx: Frequency index to visualize.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    # Process each mid-plane monitor
    for plane_name in ['xy_mid', 'xz_mid']:
        if plane_name not in monitor_mapping:
            continue

        plane_data = out_list[monitor_mapping[plane_name]]

        if field_component == 'all':
            # Total field intensity
            E_intensity = jnp.sum(jnp.abs(plane_data[freq_idx, 0:3, :, :, :])**2, axis=0)
            H_intensity = jnp.sum(jnp.abs(plane_data[freq_idx, 3:6, :, :, :])**2, axis=0)
            field_2d = jnp.squeeze(jnp.sqrt(E_intensity + H_intensity))
            title_suffix = "Total Field Intensity"
            cmap = 'hot'
        elif field_component == 'E':
            # Electric field intensity
            E_intensity = jnp.sum(jnp.abs(plane_data[freq_idx, 0:3, :, :, :])**2, axis=0)
            field_2d = jnp.squeeze(jnp.sqrt(E_intensity))
            title_suffix = "Electric Field Intensity"
            cmap = 'viridis'
        elif field_component == 'H':
            # Magnetic field intensity
            H_intensity = jnp.sum(jnp.abs(plane_data[freq_idx, 3:6, :, :, :])**2, axis=0)
            field_2d = jnp.squeeze(jnp.sqrt(H_intensity))
            title_suffix = "Magnetic Field Intensity"
            cmap = 'plasma'
        else:
            # Individual component
            field_map = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
            if field_component not in field_map:
                print(f"Unknown field component: {field_component}")
                return
            fi = field_map[field_component]
            field_2d = jnp.squeeze(plane_data[freq_idx, fi, :, :, :])
            title_suffix = field_component
            cmap = 'RdBu' if jnp.min(field_2d) < 0 else 'viridis'

        # Create visualization
        if plane_name == 'xy_mid':
            plane_label = "XY"
            xlabel, ylabel = "X (cells)", "Y (cells)"
        else:
            plane_label = "XZ"
            xlabel, ylabel = "X (cells)", "Z (cells)"

        if field_component in ['all', 'E', 'H']:
            # Single intensity plot
            fig, ax = plt.subplots(figsize=(10, 8))

            im = ax.imshow(field_2d.T, cmap=cmap, origin='upper', aspect='auto')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{plane_label} Mid-plane - {title_suffix}", fontsize=14, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, shrink=0.9)
            cbar.set_label('Field Amplitude', fontsize=11)

        else:
            # Real, Imaginary, and Magnitude for individual components
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"{plane_label} Mid-plane - {title_suffix}", fontsize=14, fontweight='bold')

            for ax, (fn, title, cmap_choice) in zip(axes,
                                                     [(jnp.real, "Real", "RdBu"),
                                                      (jnp.imag, "Imaginary", "RdBu"),
                                                      (jnp.abs, "Magnitude", "viridis")]):
                arr = fn(field_2d)
                vmax = float(jnp.max(jnp.abs(arr))) or 1.0
                vmin = -vmax if title != "Magnitude" else 0.0

                im = ax.imshow(arr.T, cmap=cmap_choice, origin='upper',
                             vmin=vmin, vmax=vmax, aspect='auto')
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(title, fontsize=12)

                cbar = plt.colorbar(im, ax=ax, shrink=0.9)
                cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()
        plt.show()


def visualize_mode(mode_field, beta, mode_num):
    """Enhanced mode field visualization.

    Args:
        mode_field: Mode field array.
        beta: Propagation constant.
        mode_num: Mode number.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    mode_slice = jnp.squeeze(mode_field[0, :, 0, :, :])

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    # E-field components
    for i, comp_name in enumerate(['Ex', 'Ey', 'Ez']):
        ax = fig.add_subplot(gs[0, i])
        mode_mag = jnp.abs(mode_slice[i, :, :])
        vmax = float(jnp.max(mode_mag)) or 1.0

        im = ax.imshow(mode_mag.T, cmap='viridis', origin='upper',
                      vmin=0, vmax=vmax, aspect='auto')
        ax.set_xlabel('Y (cells)', fontsize=10)
        ax.set_ylabel('Z (cells)', fontsize=10)
        ax.set_title(f'{comp_name} Magnitude', fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Total E-field intensity
    ax = fig.add_subplot(gs[1, :])
    total_intensity = jnp.sqrt(jnp.sum(jnp.abs(mode_slice)**2, axis=0))

    im = ax.imshow(total_intensity.T, cmap='hot', origin='upper', aspect='auto')
    ax.set_xlabel('Y (cells)', fontsize=11)
    ax.set_ylabel('Z (cells)', fontsize=11)
    ax.set_title('Total E-field Intensity', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.9)

    fig.suptitle(f"Mode {mode_num} | β = {float(beta[0]):.4f}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def quick_view_monitors(results, component='Hz', cmap='inferno'):
    """Quick visualization of each monitor's first frequency slice.

    Args:
        results: Dictionary from simulate() containing monitor data.
        component: Which field to show:
            - 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz': Individual components
            - '|E|': Electric field magnitude
            - '|H|': Magnetic field magnitude
            - 'all': Total field intensity |E|²+|H|²
        cmap: Colormap to use for visualization (default 'inferno').

    Example:
        >>> results = simulate(...)
        >>> quick_view_monitors(results, 'Hz')  # View Hz component
        >>> quick_view_monitors(results, '|E|')  # View electric field magnitude
        >>> quick_view_monitors(results, 'all')  # View total field intensity
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return

    for name, idx in results['monitor_names'].items():
        monitor_data = results['monitor_data'][idx]

        # Extract first frequency
        if component in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            # Single component
            field_map = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
            comp_idx = field_map[component]
            field_3d = monitor_data[0, comp_idx, :, :, :]
        elif component == '|E|':
            # Electric field magnitude
            E_fields = monitor_data[0, 0:3, :, :, :]
            field_3d = jnp.sqrt(jnp.sum(jnp.abs(E_fields)**2, axis=0))
        elif component == '|H|':
            # Magnetic field magnitude
            H_fields = monitor_data[0, 3:6, :, :, :]
            field_3d = jnp.sqrt(jnp.sum(jnp.abs(H_fields)**2, axis=0))
        elif component == 'all':
            # Total field intensity
            E_fields = monitor_data[0, 0:3, :, :, :]
            H_fields = monitor_data[0, 3:6, :, :, :]
            field_3d = jnp.sqrt(jnp.sum(jnp.abs(E_fields)**2, axis=0) +
                                jnp.sum(jnp.abs(H_fields)**2, axis=0))
        else:
            raise ValueError(f"Unknown component: {component}")

        # Find which dimension is singleton (size 1) or small and average/squeeze
        if field_3d.shape[0] == 1:
            field_2d = field_3d[0, :, :]  # YZ plane
            xlabel, ylabel = 'Y', 'Z'
        elif field_3d.shape[1] == 1:
            field_2d = field_3d[:, 0, :]  # XZ plane
            xlabel, ylabel = 'X', 'Z'
        elif field_3d.shape[2] == 1:
            field_2d = field_3d[:, :, 0]  # XY plane
            xlabel, ylabel = 'X', 'Y'
        else:
            # If no singleton dimension, average across smallest dimension
            min_dim = jnp.argmin(jnp.array(field_3d.shape))
            if min_dim == 0:
                field_2d = jnp.mean(field_3d, axis=0)  # Average across X
                xlabel, ylabel = 'Y', 'Z'
            elif min_dim == 1:
                field_2d = jnp.mean(field_3d, axis=1)  # Average across Y
                xlabel, ylabel = 'X', 'Z'
            else:
                field_2d = jnp.mean(field_3d, axis=2)  # Average across Z
                xlabel, ylabel = 'X', 'Y'

        plt.figure(figsize=(6, 4))

        # For complex fields, take magnitude; for real fields (|E|, |H|, all), use directly
        if jnp.iscomplexobj(field_2d):
            display_field = jnp.abs(field_2d)
            title_prefix = f"|{component}|"
        else:
            display_field = field_2d
            title_prefix = component

        plt.imshow(display_field.T, cmap=cmap, origin='upper', aspect='auto')
        plt.colorbar()
        plt.title(f"{name} - {title_prefix} (freq 0)")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    