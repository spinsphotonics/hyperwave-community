"""Field monitoring utilities for electromagnetic FDTD simulations.

This module provides functions for extracting and analyzing field data from
full FDTD simulation results, including power flow calculations, field slicing,
monitor visualization, and automatic waveguide detection for optimal monitor placement.

Field data format: (N_freq, 6, Nx, Ny, Nz) where the 6 components are:
[Ex, Ey, Ez, Hx, Hy, Hz]
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass


@dataclass
class Monitor:
    """Monitor configuration for field extraction during FDTD simulation.

    Args:
        shape: Monitor volume dimensions (xx, yy, zz). All values must be positive.
        offset: Monitor volume position offset (x, y, z). Can be negative.
    """
    shape: Tuple[int, int, int]
    offset: Tuple[int, int, int]
    
    def __post_init__(self):
        """Validate monitor parameters."""
        if len(self.shape) != 3:
            raise ValueError(f"shape must have 3 dimensions, got {len(self.shape)}")
        if len(self.offset) != 3:
            raise ValueError(f"offset must have 3 dimensions, got {len(self.offset)}")
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"All shape dimensions must be positive, got {self.shape}")
        if not all(isinstance(s, int) for s in self.shape):
            raise ValueError(f"All shape dimensions must be integers, got {self.shape}")
        if not all(isinstance(o, int) for o in self.offset):
            raise ValueError(f"All offset dimensions must be integers, got {self.offset}")

    @property
    def recipe(self) -> Dict:
        """Return serializable recipe for reconstructing this monitor.

        Returns:
            Dictionary containing shape and offset that can be used to reconstruct
            the monitor configuration.
        """
        return {
            'shape': self.shape,
            'offset': self.offset
        }




def S_from_slice(field_slice: jnp.ndarray) -> jnp.ndarray:
    """Calculate Poynting vector from a field slice.
    
    Args:
        field_slice: Field data with shape (N_freq, 6, Ny, Nx) where 6 components 
                    are [Ex, Ey, Ez, Hx, Hy, Hz].
    
    Returns:
        Poynting vector S with shape (N_freq, 3, Ny, Nx) representing the 
        time-averaged power flow density in x, y, z directions.
        
    Note:
        Poynting vector: S = 0.5 * Re(E × H*)
    """
    N_freq = field_slice.shape[0]
    Ny, Nx = field_slice.shape[2], field_slice.shape[3]
    S = jnp.zeros((N_freq, 3, Ny, Nx))

    # Extract field components
    Ex, Ey, Ez = field_slice[:, 0, :, :], field_slice[:, 1, :, :], field_slice[:, 2, :, :]
    Hx_c, Hy_c, Hz_c = jnp.conj(field_slice[:, 3, :, :]), jnp.conj(field_slice[:, 4, :, :]), jnp.conj(field_slice[:, 5, :, :])

    # Calculate Poynting vector components: S = 0.5 * Re(E × H*)
    S = S.at[:, 0, :, :].set(0.5 * jnp.real((Ey * Hz_c) - (Ez * Hy_c)))  # Sx
    S = S.at[:, 1, :, :].set(0.5 * jnp.real((Ez * Hx_c) - (Ex * Hz_c)))  # Sy
    S = S.at[:, 2, :, :].set(0.5 * jnp.real((Ex * Hy_c) - (Ey * Hx_c)))  # Sz

    return S


def power_from_a_box(
    field: jnp.ndarray, 
    Lx: int, 
    Ly: int, 
    Lz: int, 
    Lx_total: int, 
    Ly_total: int, 
    Lz_total: int
) -> jnp.ndarray:
    """Calculate net power flowing out of a box monitor.
    
    Args:
        field: Full field data with shape (N_freq, 6, Lx_total, Ly_total, Lz_total).
        Lx, Ly, Lz: Dimensions of the box monitor.
        Lx_total, Ly_total, Lz_total: Total simulation domain dimensions.
    
    Returns:
        Net power flowing out of the box for each frequency (N_freq,).
        
    Note:
        Box is centered in the simulation domain. Power is calculated by 
        integrating Poynting vector over all six faces of the box.
    """
    # Calculate center of simulation domain
    cx, cy, cz = Lx_total // 2, Ly_total // 2, Lz_total // 2

    # Extract field slices for each face of the box
    S_xneg = S_from_slice(field[:, :, cx - Lx//2, :, :])  # Left face
    S_xpos = S_from_slice(field[:, :, cx + Lx//2, :, :])  # Right face
    S_yneg = S_from_slice(field[:, :, :, cy - Ly//2, :])  # Front face  
    S_ypos = S_from_slice(field[:, :, :, cy + Ly//2, :])  # Back face
    S_zneg = S_from_slice(field[:, :, :, :, cz - Lz//2])  # Bottom face
    S_zpos = S_from_slice(field[:, :, :, :, cz + Lz//2])  # Top face

    # Calculate net power flow through each pair of faces
    # Positive direction is outward from the box
    P_net_x = jnp.sum(S_xpos[:, 0, :, :] - S_xneg[:, 0, :, :], axis=(1, 2))
    P_net_y = jnp.sum(S_ypos[:, 1, :, :] - S_yneg[:, 1, :, :], axis=(1, 2))
    P_net_z = jnp.sum(S_zpos[:, 2, :, :] - S_zneg[:, 2, :, :], axis=(1, 2))

    # Total power is sum of all three directions
    P_total = P_net_x + P_net_y + P_net_z

    return P_total


def get_field_slice(
    field: jnp.ndarray, 
    axis: str, 
    position: int
) -> jnp.ndarray:
    """Extract a 2D slice from the full 3D field data.
    
    Args:
        field: Full field data with shape (N_freq, 6, Nx, Ny, Nz).
        axis: Axis to slice along ('x', 'y', or 'z').
        position: Index position along the specified axis.
    
    Returns:
        2D field slice with shape (N_freq, 6, dim1, dim2).
        
    Example:
        >>> # Get xy-plane slice at z=10
        >>> xy_slice = get_field_slice(field, 'z', 10)
    """
    if axis == 'x':
        return field[:, :, position, :, :]
    elif axis == 'y':
        return field[:, :, :, position, :]
    elif axis == 'z':
        return field[:, :, :, :, position]
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")


def get_power_through_plane(
    field: jnp.ndarray, 
    axis: str, 
    position: int
) -> jnp.ndarray:
    """Calculate power flow through a plane.
    
    Args:
        field: Full field data with shape (N_freq, 6, Nx, Ny, Nz).
        axis: Normal axis of the plane ('x', 'y', or 'z').
        position: Index position of the plane along the specified axis.
    
    Returns:
        Power flow through the plane for each frequency (N_freq,).
        
    Example:
        >>> # Calculate power flow through a plane at x=50
        >>> power = get_power_through_plane(field, 'x', 50)
    """
    field_slice = get_field_slice(field, axis, position)
    S = S_from_slice(field_slice)
    
    # Sum power flow normal to the plane
    if axis == 'x':
        return jnp.sum(S[:, 0, :, :], axis=(1, 2))  # Sx component
    elif axis == 'y':
        return jnp.sum(S[:, 1, :, :], axis=(1, 2))  # Sy component
    elif axis == 'z':
        return jnp.sum(S[:, 2, :, :], axis=(1, 2))  # Sz component


def get_field_intensity(field: jnp.ndarray) -> jnp.ndarray:
    """Calculate electromagnetic field intensity |E|^2 + |H|^2.
    
    Args:
        field: Field data with shape (N_freq, 6, ...).
    
    Returns:
        Field intensity with shape (N_freq, ...).
    """
    E_intensity = jnp.sum(jnp.abs(field[:, 0:3, ...])**2, axis=1)  # |E|^2
    H_intensity = jnp.sum(jnp.abs(field[:, 3:6, ...])**2, axis=1)  # |H|^2
    return E_intensity + H_intensity


def get_electric_field_intensity(field: jnp.ndarray) -> jnp.ndarray:
    """Calculate electric field intensity |E|^2.
    
    Args:
        field: Field data with shape (N_freq, 6, ...).
    
    Returns:
        Electric field intensity with shape (N_freq, ...).
    """
    return jnp.sum(jnp.abs(field[:, 0:3, ...])**2, axis=1)


def get_magnetic_field_intensity(field: jnp.ndarray) -> jnp.ndarray:
    """Calculate magnetic field intensity |H|^2.
    
    Args:
        field: Field data with shape (N_freq, 6, ...).
    
    Returns:
        Magnetic field intensity with shape (N_freq, ...).
    """
    return jnp.sum(jnp.abs(field[:, 3:6, ...])**2, axis=1)


def view_monitors(structure, monitors: Union[List, 'MonitorSet'], monitor_mapping: Optional[Dict[str, int]] = None,
                 axis: str = "z", position: Optional[int] = None,
                 figsize: Tuple[int, int] = (12, 8), show_structure: bool = True,
                 alpha_structure: float = 0.3, alpha_monitors: float = 0.7,
                 source_position: Optional[int] = None, absorber_boundary = None) -> None:
    """Visualize monitor positions overlaid on structure cross-sections.

    This function creates a visualization showing where monitors are positioned
    relative to the photonic structure. It displays monitor outlines only where
    they are actually intersected by the viewing plane, avoiding visual clutter
    from monitors that lie entirely within the plane.

    Args:
        structure: Structure object containing permittivity distribution.
            Must have a `permittivity` attribute with shape (components, nx, ny, nz).
        monitors: Either a MonitorSet object or a List of Monitor objects.
            If MonitorSet, the mapping is extracted automatically.
            If List, uses monitor_mapping parameter for names.
        monitor_mapping: Optional dictionary mapping monitor names to indices.
            Only used when monitors is a List. Ignored when monitors is a MonitorSet.
        axis: Viewing axis for the cross-section slice ('x', 'y', or 'z').
            - 'x': Shows YZ plane (Y horizontal, Z vertical)
            - 'y': Shows XZ plane (X horizontal, Z vertical)  
            - 'z': Shows XY plane (X horizontal, Y vertical)
        position: Position along the specified axis for the slice. If None,
            uses the middle position along that axis.
        figsize: Figure size as (width, height) tuple.
        show_structure: Whether to display the structure permittivity as background.
        alpha_structure: Transparency level for structure display (0.0-1.0).
        alpha_monitors: Transparency level for monitor outlines (0.0-1.0).
        source_position: Optional X-position of the source plane to visualize.
        absorber_boundary: Optional absorption mask from hyperwave.absorption.create_absorption_mask 
            with shape (3, xx, yy, zz) representing Ex, Ey, Ez absorption coefficients.
    
    Returns:
        None. Displays the plot using matplotlib.
        
    Note:
        - Only monitors that extend in the viewing direction are shown
        - Monitors with thickness=1 in the viewing direction are not displayed
        - Z axis is always oriented vertically with 0 at top
        - Monitor coordinates assume hyperwave coordinate system conventions
        
    Example:
        >>> from hyperwave.structure import create_structure
        >>> from hyperwave.monitors import Monitor, create_waveguide_monitors
        >>>
        >>> # Create structure
        >>> structure = create_structure(layers)
        >>>
        >>> # Method 1: Using MonitorSet (recommended)
        >>> monitor_set = create_waveguide_monitors(structure, monitor_type='both')
        >>> view_monitors(structure, monitor_set, axis="z", position=100)
        >>>
        >>> # Method 2: Using separate list and mapping (legacy)
        >>> monitors = [
        ...     Monitor(shape=(20, 20, 5), offset=(10, 10, 50)),
        ...     Monitor(shape=(20, 20, 5), offset=(10, 10, 150))
        ... ]
        >>> monitor_mapping = {'reflection': 0, 'transmission': 1}
        >>> view_monitors(structure, monitors, monitor_mapping, axis="z", position=100)
    """
    # Handle MonitorSet input
    if isinstance(monitors, MonitorSet):
        monitor_list = monitors.monitors
        monitor_mapping = monitors.mapping
    else:
        monitor_list = monitors
        # Use provided mapping or default to None

    structure_shape = structure.permittivity.shape[1:]  # Skip components dimension
    Lx, Ly, Lz = structure_shape
    
    if position is None:
        if axis == 'x':
            position = Lx // 2
        elif axis == 'y':
            position = Ly // 2
        elif axis == 'z':
            position = Lz // 2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_structure:
        eps_array = structure.permittivity
        eps_real = jnp.real(eps_array[0])
        
        if axis == 'x':
            structure_slice = eps_real[position, :, :]
            extent = [0, Ly, Lz, 0]
            xlabel, ylabel = 'Y (cells)', 'Z (cells)'
        elif axis == 'y':
            structure_slice = eps_real[:, position, :]
            extent = [0, Lx, Lz, 0]
            xlabel, ylabel = 'X (cells)', 'Z (cells)'
        elif axis == 'z':
            structure_slice = eps_real[:, :, position]
            extent = [0, Lx, Ly, 0]
            xlabel, ylabel = 'X (cells)', 'Y (cells)'
        
        im = ax.imshow(structure_slice.T, extent=extent,
                      cmap='PuOr', alpha=alpha_structure, aspect='auto', origin='upper')
    
    monitor_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, monitor in enumerate(monitor_list):
        monitor_shape = monitor.shape
        monitor_offset = monitor.offset

        monitor_name = None
        if monitor_mapping:
            for name, idx in monitor_mapping.items():
                if idx == i:
                    monitor_name = name
                    break
        
        if monitor_name is None:
            monitor_name = f"Monitor {i}"
        
        # Convert monitor coordinates to absolute positions
        # Offsets are corner-based (direct array indices) to match solve._get
        mx_start = monitor_offset[0]
        my_start = monitor_offset[1]
        mz_start = monitor_offset[2]

        mx_end = mx_start + monitor_shape[0]
        my_end = my_start + monitor_shape[1]
        mz_end = mz_start + monitor_shape[2]
        
        # Only show monitors that are actually cut by the viewing plane
        # Skip monitors that have thickness=1 in the viewing direction
        intersects = False
        rect_params = None
        
        if axis == 'x':
            if monitor_shape[0] > 1 and mx_start <= position <= mx_end:
                intersects = True
                rect_params = {
                    'xy': (my_start, mz_start),
                    'width': my_end - my_start,
                    'height': mz_end - mz_start
                }
        elif axis == 'y':
            if monitor_shape[1] > 1 and my_start <= position <= my_end:
                intersects = True
                rect_params = {
                    'xy': (mx_start, mz_start),
                    'width': mx_end - mx_start,
                    'height': mz_end - mz_start
                }
        elif axis == 'z':
            if mz_start <= position < mz_start + monitor_shape[2]:
                intersects = True
                rect_params = {
                    'xy': (mx_start, my_start),
                    'width': mx_end - mx_start,
                    'height': my_end - my_start
                }
        
        if intersects and rect_params:
            color = monitor_colors[i % len(monitor_colors)]
            
            # Create label with shape and offset info
            label_text = f"{monitor_name} | Shape: {monitor_shape} | Offset: {monitor_offset}"
            
            rect = patches.Rectangle(
                rect_params['xy'], 
                rect_params['width'], 
                rect_params['height'],
                linewidth=3, 
                edgecolor=color, 
                facecolor='none',
                label=label_text
            )
            ax.add_patch(rect)
            
            # Add label on plot
            text_x = rect_params['xy'][0] + rect_params['width'] + 5
            text_y = rect_params['xy'][1] + rect_params['height'] / 2
                
            ax.text(text_x, text_y, monitor_name, 
                   ha='left', va='center', fontsize=10, 
                   fontweight='bold', color=color)
    
    # Add source plane visualization if provided
    if source_position is not None:
        source_color = 'yellow'
        source_linewidth = 4
        
        if axis == 'x' and source_position == position:
            # Source plane is at the current X slice - show as full YZ plane outline
            ax.axhline(y=0, color=source_color, linewidth=source_linewidth, 
                      linestyle='--', alpha=0.8, label='Source Plane')
            ax.axhline(y=Lz, color=source_color, linewidth=source_linewidth, 
                      linestyle='--', alpha=0.8)
            ax.axvline(x=0, color=source_color, linewidth=source_linewidth, 
                      linestyle='--', alpha=0.8)
            ax.axvline(x=Ly, color=source_color, linewidth=source_linewidth, 
                      linestyle='--', alpha=0.8)
        elif axis == 'y':
            # Show source plane as vertical line at X position
            ax.axvline(x=source_position, color=source_color, linewidth=source_linewidth,
                      linestyle='--', alpha=0.8, label=f'Source Plane (X={source_position})')
        elif axis == 'z':
            # Show source plane as vertical line at X position
            ax.axvline(x=source_position, color=source_color, linewidth=source_linewidth,
                      linestyle='--', alpha=0.8, label=f'Source Plane (X={source_position})')
    
    # Add absorber boundary visualization if provided
    if absorber_boundary is not None:
        # Expect absorption mask shape (3, xx, yy, zz) from create_absorption_mask
        if absorber_boundary.ndim != 4 or absorber_boundary.shape[0] != 3:
            raise ValueError(f"absorber_boundary must have shape (3, xx, yy, zz), got {absorber_boundary.shape}")
        
        # Use first component (Ex) for visualization
        absorber_3d = absorber_boundary[0]  # Shape: (xx, yy, zz)
        absorber_slice = None
        
        if axis == 'x':
            if position < absorber_3d.shape[0]:
                absorber_slice = absorber_3d[position, :, :]
        elif axis == 'y':
            if position < absorber_3d.shape[1]:
                absorber_slice = absorber_3d[:, position, :]
        elif axis == 'z':
            if position < absorber_3d.shape[2]:
                absorber_slice = absorber_3d[:, :, position]
        
        if absorber_slice is not None:
            # Create absorber overlay - show regions where absorption > threshold
            absorber_threshold = 1e-6
            absorber_mask = absorber_slice > absorber_threshold
            
            if jnp.sum(absorber_mask) > 0:  # Only show if there are absorbing regions
                # Use square root scaling for better visualization (like the absorption module)
                absorber_overlay = jnp.sqrt(absorber_slice)
                absorber_overlay = jnp.where(absorber_mask, absorber_overlay, 0.0)
                
                # Overlay absorber regions in red with transparency
                ax.imshow(absorber_overlay.T, extent=extent, cmap='Reds',
                         alpha=0.4, aspect='auto', vmin=0, vmax=jnp.max(absorber_overlay), origin='upper')
                
                # Add legend entry for absorbers
                ax.plot([], [], 's', color='red', alpha=0.4, markersize=10, 
                       label='Absorber Regions')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = f'Monitor Positions ({axis.upper()} slice at {position}) | Structure: {Lx}×{Ly}×{Lz}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def reconstruct_monitorset_from_recipe(recipe: List[Dict]) -> 'MonitorSet':
    """Reconstruct a MonitorSet from a recipe.

    Args:
        recipe: List of monitor configurations from MonitorSet.recipe

    Returns:
        Reconstructed MonitorSet object with all monitors and names.
    """
    monitor_set = MonitorSet()
    for config in recipe:
        monitor = Monitor(
            shape=tuple(config['shape']),
            offset=tuple(config['offset'])
        )
        monitor_set.add(monitor, name=config['name'])
    return monitor_set


class MonitorSet:
    """Container for managing multiple monitors with add/remove/replace operations.

    This class provides a clean interface for managing monitors in FDTD simulations,
    with support for named monitors and automatic indexing.

    Attributes:
        monitors: List of Monitor objects
        mapping: Dictionary mapping names to monitor indices
    """

    def __init__(self, monitors: Optional[List[Monitor]] = None,
                 mapping: Optional[Dict[str, int]] = None):
        """Initialize MonitorSet with optional existing monitors.

        Args:
            monitors: Initial list of Monitor objects.
            mapping: Initial dictionary mapping names to indices.
        """
        self.monitors = monitors if monitors is not None else []
        self.mapping = mapping if mapping is not None else {}
        self._validate_consistency()

    def _validate_consistency(self):
        """Ensure mapping indices match actual monitor list indices."""
        for name, idx in self.mapping.items():
            if idx >= len(self.monitors):
                raise ValueError(f"Invalid mapping: '{name}' points to index {idx} "
                               f"but only {len(self.monitors)} monitors exist")

    def add(self, monitor: Monitor, name: Optional[str] = None) -> int:
        """Add a new monitor to the set.

        Args:
            monitor: Monitor object to add.
            name: Optional name for the monitor. If None, uses 'monitor_N'.

        Returns:
            Index of the added monitor.

        Raises:
            ValueError: If the name already exists in the mapping.
            TypeError: If monitor is not a Monitor instance.
        """
        if not isinstance(monitor, Monitor):
            raise TypeError(f"Expected Monitor instance, got {type(monitor).__name__}")

        if name is None:
            # Generate automatic name
            name = f"monitor_{len(self.monitors)}"
            # Ensure uniqueness
            while name in self.mapping:
                name = f"monitor_{len(self.monitors)}_{np.random.randint(1000)}"

        if name in self.mapping:
            raise ValueError(f"Monitor with name '{name}' already exists.")

        idx = len(self.monitors)
        self.monitors.append(monitor)
        self.mapping[name] = idx

        return idx

    def remove(self, identifier: Union[str, int, Monitor]) -> Monitor:
        """Remove a monitor from the set.

        Args:
            identifier: Can be:
                - str: Name of the monitor to remove
                - int: Index of the monitor to remove
                - Monitor: The monitor object to remove

        Returns:
            The removed Monitor object.

        Raises:
            ValueError: If monitor not found or invalid identifier.
            TypeError: If identifier type is not supported.

        Note:
            Removing a monitor will update all indices in the mapping
            to maintain consistency.
        """
        # Determine index to remove
        if isinstance(identifier, str):
            if identifier not in self.mapping:
                raise ValueError(f"Monitor with name '{identifier}' not found")
            idx = self.mapping[identifier]
            name_to_remove = identifier
        elif isinstance(identifier, int):
            if identifier < 0 or identifier >= len(self.monitors):
                raise ValueError(f"Monitor index {identifier} out of range "
                               f"[0, {len(self.monitors)-1}]")
            idx = identifier
            # Find the name(s) that map to this index
            name_to_remove = None
            for name, mapped_idx in self.mapping.items():
                if mapped_idx == idx:
                    name_to_remove = name
                    break
        elif isinstance(identifier, Monitor):
            try:
                idx = self.monitors.index(identifier)
            except ValueError:
                raise ValueError("Monitor object not found in the set")
            # Find the name(s) that map to this index
            name_to_remove = None
            for name, mapped_idx in self.mapping.items():
                if mapped_idx == idx:
                    name_to_remove = name
                    break
        else:
            raise TypeError(f"Invalid identifier type: {type(identifier).__name__}. "
                          f"Expected str, int, or Monitor")

        # Remove the monitor
        removed = self.monitors.pop(idx)

        # Remove from mapping if it has a name
        if name_to_remove:
            del self.mapping[name_to_remove]

        # Update all indices in mapping that were after the removed one
        for name, mapped_idx in self.mapping.items():
            if mapped_idx > idx:
                self.mapping[name] = mapped_idx - 1

        return removed

    def clear(self) -> None:
        """Remove all monitors from the set."""
        self.monitors = []
        self.mapping = {}

    def get(self, identifier: Union[str, int]) -> Monitor:
        """Get a monitor by name or index.

        Args:
            identifier: Name (str) or index (int) of the monitor.

        Returns:
            The requested Monitor object.

        Raises:
            ValueError: If monitor not found.
            TypeError: If identifier type is not supported.
        """
        if isinstance(identifier, str):
            if identifier not in self.mapping:
                raise ValueError(f"Monitor with name '{identifier}' not found")
            return self.monitors[self.mapping[identifier]]
        elif isinstance(identifier, int):
            if identifier < 0 or identifier >= len(self.monitors):
                raise ValueError(f"Monitor index {identifier} out of range "
                               f"[0, {len(self.monitors)-1}]")
            return self.monitors[identifier]
        else:
            raise TypeError(f"Invalid identifier type: {type(identifier).__name__}. "
                          f"Expected str or int")

    def list_monitors(self) -> List[str]:
        """Get list of all monitor names.

        Returns:
            List of monitor names (keys only).
        """
        return list(self.mapping.keys())

    def __len__(self) -> int:
        """Return number of monitors."""
        return len(self.monitors)

    def __getitem__(self, key: Union[str, int]) -> Monitor:
        """Allow dict-like and list-like access to monitors."""
        return self.get(key)

    def __repr__(self) -> str:
        """String representation of MonitorSet."""
        return f"MonitorSet({len(self.monitors)} monitors, {len(self.mapping)} named)"

    def to_tuple(self) -> Tuple[List[Monitor], Dict[str, int]]:
        """Convert to tuple format for backward compatibility.

        Returns:
            Tuple of (monitors_list, monitor_mapping) for legacy code.
        """
        return self.monitors, self.mapping

    def view(self, structure, axis: str = "z", position: Optional[int] = None,
             figsize: Tuple[int, int] = (12, 8), show_structure: bool = True,
             alpha_structure: float = 0.3, alpha_monitors: float = 0.7,
             source_position: Optional[int] = None, absorber_boundary = None) -> None:
        """Visualize monitors in this set overlaid on structure cross-sections.

        This is a convenience method that calls view_monitors() with this MonitorSet.
        See view_monitors() for full documentation of visualization options.

        Args:
            structure: Structure object containing permittivity distribution.
            axis: Viewing axis for the cross-section slice ('x', 'y', or 'z').
            position: Position along the specified axis for the slice.
            figsize: Figure size as (width, height) tuple.
            show_structure: Whether to display the structure permittivity.
            alpha_structure: Transparency level for structure display (0.0-1.0).
            alpha_monitors: Transparency level for monitor outlines (0.0-1.0).
            source_position: Optional X-position of the source plane.
            absorber_boundary: Optional absorption mask from hyperwave.absorption.

        Note:
            This method provides a convenient object-oriented interface:
            >>> monitor_set = MonitorSet()
            >>> monitor_set.view(structure, axis='z', position=100)
        """
        view_monitors(structure, self, None, axis, position, figsize,
                     show_structure, alpha_structure, alpha_monitors,
                     source_position, absorber_boundary)

    def add_monitors_at_position(
        self,
        structure,
        axis: str,
        position: int,
        monitor_thickness: int = 5,
        width_factor: float = 2.5,
        height_factor: Optional[float] = None,
        min_width_factor: float = 1.5,
        label: Optional[str] = None,
        verbose: bool = False
    ) -> List[str]:
        """Add monitors automatically along a specific axis at a specific position.

        Detects waveguides or other high-permittivity features along the specified
        axis at the given position and automatically creates appropriately sized
        monitors for each detected feature. Monitors are added to this MonitorSet
        with automatic naming based on feature count.

        WARNING: This function is designed for structures with waveguides or other
        high-permittivity features. It will not work correctly for freespace regions
        or uniform permittivity regions as it relies on permittivity contrast to detect
        features. For such cases, use the add() method directly to place monitors manually.

        Args:
            structure: Structure object with permittivity attribute.
            axis: Axis perpendicular to the monitors ('x', 'y', or 'z').
                - 'x': Creates YZ plane monitors at specified X position
                - 'y': Creates XZ plane monitors at specified Y position
                - 'z': Creates XY plane monitors at specified Z position
            position: Position along the axis where monitors should be placed (in pixels).
            monitor_thickness: Thickness of monitors in the direction of the axis (in pixels).
                Default is 5 pixels. This is absolute, not relative.
            width_factor: Multiplier for waveguide width to determine monitor lateral extent.
                Default is 2.5 (monitor will be 2.5x the waveguide width).
            height_factor: Multiplier for monitor height (Z dimension). If None, uses the
                same value as width_factor. For Z axis monitors, this parameter is ignored.
                Default is None (same as width_factor).
            min_width_factor: Minimum multiplier for waveguide width.
                Default is 1.5. Ensures monitors are at least 1.5x waveguide width.
            label: Label for monitor names. If None, uses 'mon_axisXXX' format
                where XXX is the position value. If provided, smart suffixes are added:
                - Single monitor: uses label as-is (e.g., 'input')
                - Two monitors: adds '_top'/'_bottom' or '_left'/'_right' (e.g., 'input_top')
                - Multiple monitors: adds index (e.g., 'input_0', 'input_1')
            verbose: Whether to print detection information. Default is False.

        Returns:
            List of names of the added monitors.

        Raises:
            ValueError: If axis is not 'x', 'y', or 'z'.

        Note:
            - Monitor naming follows these conventions:
              * Single waveguide: Uses prefix directly (e.g., 'input')
              * Two waveguides: Adds '_top'/'_bottom' for X axis, '_left'/'_right' for Y
              * Multiple waveguides: Adds index suffix (e.g., 'coupling_0', 'coupling_1')
            - This function is not intended for freespace or uniform permittivity regions.
              Use the add() method instead for those use cases.

        Example:
            >>> # Add monitors using method syntax
            >>> monitor_set = MonitorSet()
            >>> names = monitor_set.add_monitors_at_position(
            ...     structure, axis='x', position=100, height_factor=2.0
            ... )
            >>> print(f"Added {len(names)} monitors: {names}")
        """
        # Call the module-level function with this MonitorSet
        return add_monitors_at_position(
            self, structure, axis, position,
            monitor_thickness, width_factor, height_factor, min_width_factor,
            label, verbose
        )

    @property
    def recipe(self) -> List[Dict]:
        """Return serializable recipe for reconstructing this MonitorSet.

        Returns:
            List of dictionaries, each containing:
                - name: Monitor name/label
                - shape: Monitor shape tuple
                - offset: Monitor offset tuple
            Can be used to reconstruct the MonitorSet configuration.
        """
        recipe_list = []
        for name, idx in self.mapping.items():
            monitor = self.monitors[idx]
            recipe_list.append({
                'name': name,
                'shape': monitor.shape,
                'offset': monitor.offset
            })
        return recipe_list


# =============================================================================
# Automatic Waveguide Detection and Monitor Placement
# =============================================================================

def _detect_waveguides(
    structure,
    x_position: Optional[int] = None,
    y_position: Optional[int] = None,
    z_position: Optional[int] = None,
    axis: str = 'y',
    threshold_method: str = 'auto'
) -> List[Dict]:
    """Detect waveguide positions and dimensions by analyzing permittivity.

    Analyzes a cross-section of the structure to identify high-permittivity
    regions (waveguides) along the specified axis. Uses thresholding to
    distinguish waveguide material from background.

    Args:
        structure: Structure object with permittivity attribute.
        x_position: X position for YZ slice analysis for detecting Y waveguides
            (in pixels). If None, uses middle of X dimension.
        y_position: Y position for XZ slice analysis for detecting X waveguides
            (in pixels). If None, uses middle of Y dimension.
        z_position: Z position for XY slice analysis (in pixels). If None, uses
            middle of Z dimension.
        axis: Which axis to scan along ('x' or 'y'). Default is 'y'.
        threshold_method: Method for determining waveguide boundaries. Options:
            - 'auto': Midpoint between min and max permittivity (default)
            - 'otsu': Otsu's automatic thresholding algorithm
            - float: Manual threshold value for permittivity

    Returns:
        List of dictionaries containing waveguide information. Each dictionary
        contains:
            - 'center': Center position along the specified axis (int)
            - 'start': Start position of waveguide (int)
            - 'end': End position of waveguide (int)
            - 'width': Width of the waveguide in pixels (int)
            - 'axis': The axis along which waveguide was detected (str)

    Raises:
        ValueError: If axis is not 'x' or 'y'.

    Note:
        Only detects waveguides with width >= 3 pixels to filter out noise.
        Results are sorted by center position along the detection axis.
    """
    # Get permittivity array (remove frequency dimension if present)
    if len(structure.permittivity.shape) == 4:
        eps_array = structure.permittivity[0]
    else:
        eps_array = structure.permittivity

    x_dim, y_dim, z_dim = eps_array.shape

    # Set default positions to middle if not specified
    if x_position is None:
        x_position = x_dim // 2
    if y_position is None:
        y_position = y_dim // 2
    if z_position is None:
        z_position = z_dim // 2

    # Get the appropriate slice based on axis
    if axis == 'y':
        # Detect waveguides along Y axis using XZ slice at x_position
        slice_1d = eps_array[x_position, :, z_position]  # Y variation
    elif axis == 'x':
        # Detect waveguides along X axis using YZ slice at y_position
        slice_1d = eps_array[:, y_position, z_position]  # X variation
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")

    # Convert to numpy for processing
    slice_1d = np.array(slice_1d).real

    # Determine threshold
    if threshold_method == 'auto':
        # Simple midpoint between min and max
        threshold = (np.max(slice_1d) + np.min(slice_1d)) / 2
    elif threshold_method == 'otsu':
        # Otsu's method for automatic thresholding
        hist, bin_edges = np.histogram(slice_1d, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate probabilities
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        # Calculate means
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / weight2

        # Calculate variance
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        # Find maximum variance
        idx = np.argmax(variance)
        threshold = bin_centers[idx]
    else:
        # Use provided threshold value
        threshold = float(threshold_method)

    # Find high permittivity regions
    high_eps_mask = slice_1d > threshold

    # Find connected regions (waveguides)
    waveguide_info = []
    in_waveguide = False
    wg_start = 0

    for pos in range(len(high_eps_mask)):
        if high_eps_mask[pos] and not in_waveguide:
            # Start of waveguide
            wg_start = pos
            in_waveguide = True
        elif not high_eps_mask[pos] and in_waveguide:
            # End of waveguide
            wg_end = pos - 1
            wg_center = (wg_start + wg_end) // 2
            wg_width = wg_end - wg_start + 1

            # Only include if width is reasonable (filter out noise)
            if wg_width >= 3:  # At least 3 pixels wide
                waveguide_info.append({
                    'center': int(wg_center),
                    'start': int(wg_start),
                    'end': int(wg_end),
                    'width': int(wg_width),
                    'axis': axis
                })
            in_waveguide = False

    # Handle case where waveguide extends to edge
    if in_waveguide:
        wg_end = len(high_eps_mask) - 1
        wg_center = (wg_start + wg_end) // 2
        wg_width = wg_end - wg_start + 1

        if wg_width >= 3:
            waveguide_info.append({
                'center': int(wg_center),
                'start': int(wg_start),
                'end': int(wg_end),
                'width': int(wg_width),
                'axis': axis
            })

    return sorted(waveguide_info, key=lambda x: x['center'])


def add_monitors_at_position(
    monitor_set: MonitorSet,
    structure,
    axis: str,
    position: int,
    monitor_thickness: int = 5,
    width_factor: float = 2.5,
    height_factor: Optional[float] = None,
    min_width_factor: float = 1.5,
    label: Optional[str] = None,
    verbose: bool = True
) -> List[str]:
    """Add monitors automatically along a specific axis at a specific position.

    Detects waveguides or other high-permittivity features along the specified
    axis at the given position and automatically creates appropriately sized
    monitors for each detected feature. Monitors are added to the provided
    MonitorSet with automatic naming based on feature count.

    WARNING: This function is designed for structures with waveguides or other
    high-permittivity features. It will not work correctly for freespace regions
    or uniform permittivity regions as it relies on permittivity contrast to detect
    features. For such cases, use the MonitorSet.add() method directly to place
    monitors manually.

    Args:
        monitor_set: MonitorSet to add monitors to.
        structure: Structure object with permittivity attribute and get_shape() method.
        axis: Axis perpendicular to the monitors ('x', 'y', or 'z').
            - 'x': Creates YZ plane monitors at specified X position
            - 'y': Creates XZ plane monitors at specified Y position
            - 'z': Creates XY plane monitors at specified Z position
        position: Position along the axis where monitors should be placed (in pixels).
        monitor_thickness: Thickness of monitors in the direction of the axis (in pixels).
            Default is 5 pixels. This is absolute, not relative.
        width_factor: Multiplier for waveguide width to determine monitor lateral extent.
            Default is 2.5 (monitor will be 2.5x the waveguide width).
        height_factor: Multiplier for monitor height (Z dimension). If None, uses the
            same value as width_factor. For Z axis monitors, this parameter is ignored.
            Default is None (same as width_factor).
        min_width_factor: Minimum multiplier for waveguide width.
            Default is 1.5. Ensures monitors are at least 1.5x waveguide width.
        label: Prefix for monitor names. If None, uses 'mon_axisXXX' format
            where XXX is the position value. If provided, smart suffixes are added:
                - Single monitor: uses label as-is (e.g., 'input')
                - Two monitors: adds '_top'/'_bottom' or '_left'/'_right' (e.g., 'input_top')
                - Multiple monitors: adds index (e.g., 'input_0', 'input_1')
        verbose: Whether to print detection information. Default is False.

    Returns:
        List of names of the added monitors.

    Raises:
        ValueError: If axis is not 'x', 'y', or 'z'.

    Note:
        - Monitor naming follows these conventions:
          * Single waveguide: Uses prefix directly (e.g., 'input')
          * Two waveguides: Adds '_top'/'_bottom' for X axis, '_left'/'_right' for Y
          * Multiple waveguides: Adds index suffix (e.g., 'coupling_0', 'coupling_1')
        - This function is not intended for freespace or uniform permittivity regions.
          Use the MonitorSet.add() method instead for those use cases.

    Example:
        >>> # Add monitors across waveguides at X=100
        >>> monitor_set = MonitorSet()
        >>> names = add_monitors_at_position(
        ...     monitor_set, structure, axis='x', position=100, height_factor=2.0
        ... )
        >>> print(f"Added {len(names)} monitors: {names}")
    """
    # Get structure dimensions
    if len(structure.permittivity.shape) == 4:
        _, x_dim, y_dim, z_dim = structure.permittivity.shape
    else:
        x_dim, y_dim, z_dim = structure.permittivity.shape

    added_names = []

    # Default name prefix based on axis and position
    if label is None:
        label = f"mon_{axis}{position}"

    # Detect features based on axis
    if axis == 'x':
        # YZ plane monitors at X position - detect along Y
        waveguides = _detect_waveguides(
            structure,
            x_position=position,
            z_position=z_dim // 2,
            axis='y'
        )

        if verbose:
            print(f"Detected {len(waveguides)} features along Y at X={position}")

        for i, wg in enumerate(waveguides):
            # Calculate monitor extent based on waveguide width
            # Use the larger of width_factor or min_width_factor
            effective_factor = max(width_factor, min_width_factor)
            desired_half_extent = int(wg['width'] * effective_factor) // 2

            # Calculate monitor position centered on waveguide
            # Clamp each edge independently to domain boundaries
            y_center = wg['center']
            y_start = max(0, y_center - desired_half_extent)
            y_end = min(y_dim, y_center + desired_half_extent)
            y_height = y_end - y_start

            # Calculate Z dimension based on height_factor
            # If not specified, use the same factor as width
            effective_height_factor = height_factor if height_factor is not None else width_factor
            desired_z_half_extent = int(wg['width'] * effective_height_factor) // 2

            # Center in Z, clamp each edge independently
            z_center = z_dim // 2
            z_start = max(0, z_center - desired_z_half_extent)
            z_end = min(z_dim, z_center + desired_z_half_extent)
            z_height = z_end - z_start

            monitor = Monitor(
                shape=(monitor_thickness, y_height, z_height),
                offset=(position, y_start, z_start)
            )

            # Generate name
            if len(waveguides) == 1:
                name = label
            elif len(waveguides) == 2:
                name = f"{label}_{'top' if i == 0 else 'bottom'}"
            else:
                name = f"{label}_{i}"

            monitor_set.add(monitor, name=name)
            added_names.append(name)

            if verbose:
                print(f"  Added '{name}': Y=[{y_start}, {y_start+y_height}], width={wg['width']}px")

    elif axis == 'y':
        # XZ plane monitors at Y position - detect along X
        waveguides = _detect_waveguides(
            structure,
            y_position=position,
            z_position=z_dim // 2,
            axis='x'
        )

        if verbose:
            print(f"Detected {len(waveguides)} features along X at Y={position}")

        for i, wg in enumerate(waveguides):
            # Calculate monitor extent based on waveguide width
            # Use the larger of width_factor or min_width_factor
            effective_factor = max(width_factor, min_width_factor)
            desired_half_extent = int(wg['width'] * effective_factor) // 2

            # Calculate monitor position centered on waveguide
            # Clamp each edge independently to domain boundaries
            x_center = wg['center']
            x_start = max(0, x_center - desired_half_extent)
            x_end = min(x_dim, x_center + desired_half_extent)
            x_width = x_end - x_start

            # Calculate Z dimension based on height_factor
            # If not specified, use the same factor as width
            effective_height_factor = height_factor if height_factor is not None else width_factor
            desired_z_half_extent = int(wg['width'] * effective_height_factor) // 2

            # Center in Z, clamp each edge independently
            z_center = z_dim // 2
            z_start = max(0, z_center - desired_z_half_extent)
            z_end = min(z_dim, z_center + desired_z_half_extent)
            z_height = z_end - z_start

            monitor = Monitor(
                shape=(x_width, monitor_thickness, z_height),
                offset=(x_start, position, z_start)
            )

            # Generate name
            if len(waveguides) == 1:
                name = label
            elif len(waveguides) == 2:
                name = f"{label}_{'left' if i == 0 else 'right'}"
            else:
                name = f"{label}_{i}"

            monitor_set.add(monitor, name=name)
            added_names.append(name)

            if verbose:
                print(f"  Added '{name}': X=[{x_start}, {x_start+x_width}], width={wg['width']}px")

    elif axis == 'z':
        # XY plane monitors at Z position - detect along Y (most common)
        # Note: height_factor is ignored for Z-axis monitors since they are XY planes
        waveguides = _detect_waveguides(
            structure,
            x_position=x_dim // 2,
            z_position=position,
            axis='y'
        )

        if verbose:
            print(f"Detected {len(waveguides)} features along Y at Z={position}")

        for i, wg in enumerate(waveguides):
            # For Z-plane monitors, typically want full X extent
            monitor = Monitor(
                shape=(x_dim, wg['width'], monitor_thickness),
                offset=(0, wg['start'], position)
            )

            # Generate name
            if len(waveguides) == 1:
                name = label
            else:
                name = f"{label}_{i}"

            monitor_set.add(monitor, name=name)
            added_names.append(name)

            if verbose:
                print(f"  Added '{name}': Y=[{wg['start']}, {wg['end']}]")

    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")

    # Handle case where no waveguides detected
    if not waveguides:
        if verbose:
            print(f"  Warning: No features detected, adding fallback monitor")

        # Calculate Z dimension based on height_factor for X and Y axis monitors
        if axis != 'z':
            # If not specified, use the same factor as width
            effective_height_factor = height_factor if height_factor is not None else width_factor
            # Use a reasonable default size for fallback (quarter of dimension)
            fallback_width = max(y_dim // 4 if axis == 'x' else x_dim // 4, 10)
            desired_z_half_extent = int(fallback_width * effective_height_factor) // 2

            # Center in Z, clamp each edge independently
            z_center = z_dim // 2
            z_start = max(0, z_center - desired_z_half_extent)
            z_end = min(z_dim, z_center + desired_z_half_extent)
            z_height = z_end - z_start
        else:
            z_start = 0
            z_height = z_dim

        # Add a single fallback monitor
        if axis == 'x':
            monitor = Monitor(
                shape=(monitor_thickness, y_dim // 4, z_height),
                offset=(position, y_dim // 2 - y_dim // 8, z_start)
            )
        elif axis == 'y':
            monitor = Monitor(
                shape=(x_dim // 4, monitor_thickness, z_height),
                offset=(x_dim // 2 - x_dim // 8, position, z_start)
            )
        else:  # z
            monitor = Monitor(
                shape=(x_dim, y_dim, monitor_thickness),
                offset=(0, 0, position)
            )

        name = f"{label}_fallback"
        monitor_set.add(monitor, name=name)
        added_names.append(name)

    return added_names
