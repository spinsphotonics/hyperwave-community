"""Adiabatic absorber utilities for electromagnetic FDTD simulations.

This module provides functions for creating adiabatic absorbers that gradually
absorb electromagnetic waves at simulation boundaries while preserving device
physics in the center region. The absorbers use quadratic profiles to ensure
smooth, reflection-free transitions in 3D.

Key features:
- Zero absorption in center region (preserves device physics)
- Quadratic absorption profiles at boundaries (prevents reflections)
- Proper handling of FDTD Yee grid offsets for Ex, Ey, Ez field components
- 3D absorption that works in all three spatial dimensions
- Inward padding: absorption occurs at boundaries of the original grid

The new inward padding approach:
- Input permittivity: (3, xx, yy, zz)
- Output absorption mask: (3, xx, yy, zz) - same shape as input
- Center region: absorption-free zone where device physics is preserved
- Boundary regions: quadratic absorption profiles at grid edges
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp


def _absorption_profiles(numcells: int, width: float, smoothness: float) -> jax.Array:
    """Create 1D quadratic absorption profiles for adiabatic boundary conditions.
    
    This function creates the fundamental building block of adiabatic absorbers:
    a 1D profile that's zero in the center and increases quadratically toward edges.
    
    Args:
        numcells: Number of grid cells along this dimension.
        width: Distance from center where absorption starts (absorption-free zone).
        smoothness: Absorption coefficient (controls absorption strength).
        
    Returns:
        (2, numcells) array: Two offset profiles for staggered grid (FDTD Yee cell).
        
    Physics:
        - Center region (|pos| < width): Zero absorption to preserve device physics
        - Edge regions (|pos| > width): Quadratic increase σ ∝ (distance)²
        - Smooth transition prevents electromagnetic reflections
    """
    if numcells <= 0:
        raise ValueError(f"numcells must be positive, got {numcells}")
    if width < 0:
        raise ValueError(f"width must be non-negative, got {width}")
    if smoothness < 0:
        raise ValueError(f"smoothness must be non-negative, got {smoothness}")
    
    # Find the center of the grid
    center = (numcells - 1) / 2
    
    # Create two offset arrays for staggered FDTD grid (Yee cell)
    # offset[0] = [0] for cell centers, offset[1] = [0.5] for cell edges
    offset = jnp.array([[0], [0.5]])
    
    # Generate position arrays: pos[0] = [0,1,2,...], pos[1] = [0.5,1.5,2.5,...]
    pos = jnp.arange(numcells) + offset
    
    # Calculate distance from center, then subtract the absorption-free width
    # This creates a "dead zone" of width 2*width where absorption is zero
    pos = jnp.abs(pos - center) - center + width
    
    # Clip negative values to zero (no absorption in the center region)
    pos = jnp.clip(pos, a_min=0, a_max=None)
    
    # Apply quadratic profile: σ(r) = smoothness × r²
    # Quadratic ensures smooth, reflection-free absorption
    return smoothness * jnp.power(pos, 2)


def _cross_profiles_3d(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """Combine three 1D absorption profiles into a 3D profile using maximum operation.
    
    Takes the element-wise maximum of three 1D profiles to create a 3D absorption
    pattern. This ensures absorption occurs near ANY boundary (x OR y OR z direction).
    
    Args:
        x: (numcells_x,) array of x-direction absorption values.
        y: (numcells_y,) array of y-direction absorption values.
        z: (numcells_z,) array of z-direction absorption values.
        
    Returns:
        (1, numcells_x, numcells_y, numcells_z) array: 3D absorption pattern.
        
    Physics:
        Using maximum ensures that points near ANY edge get absorbed.
        This creates a 3D "box frame" absorption pattern around all edges.
    """
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        raise ValueError("x, y, and z must be 1D arrays")
    
    # Create 3D meshgrid and take element-wise maximum
    # meshgrid creates: X[i,j,k] = x[i], Y[i,j,k] = y[j], Z[i,j,k] = z[k]
    # maximum takes max at each (i,j,k) point -> absorb if near any edge
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    return jnp.maximum(jnp.maximum(X, Y), Z)[None, ...]


def _absorption_mask_3d_custom(xx: int, yy: int, zz: int, width_x: int, width_y: int, width_z: int, smoothness: float) -> jax.Array:
    """Create 3D absorption pattern with custom absorption width for each direction.
    
    FDTD uses a staggered "Yee grid" where different field components (Ex, Ey, Ez)
    are located at different positions within each cell. This function creates
    absorption patterns for all three field components with proper 3D offsets
    and custom absorption width for each direction.
    
    Args:
        xx: Number of grid cells in x direction.
        yy: Number of grid cells in y direction.
        zz: Number of grid cells in z direction.
        width_x: Absorption width in x direction (cells from each boundary).
        width_y: Absorption width in y direction (cells from each boundary).
        width_z: Absorption width in z direction (cells from each boundary).
        smoothness: Absorption coefficient strength.
        
    Returns:
        (3, xx, yy, zz) array: Absorption for [Ex, Ey, Ez] field components.
        
    Grid Layout (Yee cell):
        Ex: Located at (i+0.5, j+0, k+0) - x-edge centered
        Ey: Located at (i+0, j+0.5, k+0) - y-edge centered  
        Ez: Located at (i+0, j+0, k+0.5) - z-edge centered
    """
    if xx <= 0 or yy <= 0 or zz <= 0:
        raise ValueError(f"Grid dimensions must be positive, got xx={xx}, yy={yy}, zz={zz}")
    if width_x < 0 or width_y < 0 or width_z < 0:
        raise ValueError(f"Absorption widths must be non-negative, got width_x={width_x}, width_y={width_y}, width_z={width_z}")
    if smoothness < 0:
        raise ValueError(f"smoothness must be non-negative, got {smoothness}")
    
    # Generate 1D profiles for x, y, and z directions with proper offsets
    x = _absorption_profiles(xx, width_x, smoothness)  # Shape: (2, xx)
    y = _absorption_profiles(yy, width_y, smoothness)  # Shape: (2, yy)
    z = _absorption_profiles(zz, width_z, smoothness)  # Shape: (2, zz)
    
    # Combine profiles for each field component with appropriate offsets:
    # Ex component: x-offset=0.5, y-offset=0, z-offset=0 -> x[0], y[1], z[1]
    # Ey component: x-offset=0, y-offset=0.5, z-offset=0 -> x[1], y[0], z[1]  
    # Ez component: x-offset=0, y-offset=0, z-offset=0.5 -> x[1], y[1], z[0]
    return jnp.concatenate([
        _cross_profiles_3d(x[0], y[1], z[1]),  # Ex absorption
        _cross_profiles_3d(x[1], y[0], z[1]),  # Ey absorption
        _cross_profiles_3d(x[1], y[1], z[0])   # Ez absorption
    ])


def _absorption_mask_3d(xx: int, yy: int, zz: int, width: float, smoothness: float) -> jax.Array:
    """Create 3D absorption pattern with proper FDTD grid offsets for all field components.
    
    FDTD uses a staggered "Yee grid" where different field components (Ex, Ey, Ez)
    are located at different positions within each cell. This function creates
    absorption patterns for all three field components with proper 3D offsets.
    
    Args:
        xx: Number of grid cells in x direction.
        yy: Number of grid cells in y direction.
        zz: Number of grid cells in z direction.
        width: Absorption width from each boundary.
        smoothness: Absorption coefficient strength.
        
    Returns:
        (3, xx, yy, zz) array: Absorption for [Ex, Ey, Ez] field components.
        
    Grid Layout (Yee cell):
        Ex: Located at (i+0.5, j+0, k+0) - x-edge centered
        Ey: Located at (i+0, j+0.5, k+0) - y-edge centered  
        Ez: Located at (i+0, j+0, k+0.5) - z-edge centered
    """
    if xx <= 0 or yy <= 0 or zz <= 0:
        raise ValueError(f"Grid dimensions must be positive, got xx={xx}, yy={yy}, zz={zz}")
    if width < 0:
        raise ValueError(f"width must be non-negative, got {width}")
    if smoothness < 0:
        raise ValueError(f"smoothness must be non-negative, got {smoothness}")
    
    # Generate 1D profiles for x, y, and z directions with proper offsets
    x = _absorption_profiles(xx, width, smoothness)  # Shape: (2, xx)
    y = _absorption_profiles(yy, width, smoothness)  # Shape: (2, yy)
    z = _absorption_profiles(zz, width, smoothness)  # Shape: (2, zz)
    
    # Combine profiles for each field component with appropriate offsets:
    # Ex component: x-offset=0.5, y-offset=0, z-offset=0 -> x[0], y[1], z[1]
    # Ey component: x-offset=0, y-offset=0.5, z-offset=0 -> x[1], y[0], z[1]  
    # Ez component: x-offset=0, y-offset=0, z-offset=0.5 -> x[1], y[1], z[0]
    return jnp.concatenate([
        _cross_profiles_3d(x[0], y[1], z[1]),  # Ex absorption
        _cross_profiles_3d(x[1], y[0], z[1]),  # Ey absorption
        _cross_profiles_3d(x[1], y[1], z[0])   # Ez absorption
    ])


def create_absorption_mask(
    grid_shape: Tuple[int, int, int],
    absorption_widths: Tuple[int, int, int],
    absorption_coeff: float,
    show_plots: bool = True
) -> jax.Array:
    """Create 3D adiabatic absorption mask with inward padding logic.
    
    This function creates a 3D absorption mask that:
    1. Uses inward padding: absorption occurs at grid boundaries
    2. Creates zero absorption in the center region (preserves device physics)
    3. Increases quadratically toward all boundaries (prevents reflections)
    4. Handles all three field components (Ex, Ey, Ez) with proper Yee grid offsets
    5. Optionally visualizes the absorption mask when show_plots=True
    
    Args:
        grid_shape: Tuple of (xx, yy, zz) grid dimensions.
        absorption_widths: Tuple of (x_pad, y_pad, z_pad) absorption widths from each boundary.
        absorption_coeff: Maximum absorption strength at boundaries.
        show_plots: Whether to display absorption mask visualization (default: True).
        
    Returns:
        absorption_mask: (3, xx, yy, zz) absorption array
        
    Physics:
        - Center region stays at zero absorption (perfect device simulation)
        - Boundary regions gradually absorb waves (no artificial reflections)
        - Quadratic profile ensures adiabatic (slowly-varying) transitions
        - Works in full 3D with proper field component offsets
        
    Example:
        >>> import jax.numpy as jnp
        >>> from hyperwave.absorption import create_absorption_mask
        >>> 
        >>> # Create absorption mask with inward padding and visualization
        >>> grid_shape = (100, 80, 60)  # (xx, yy, zz)
        >>> absorption_widths = (10, 8, 6)  # (x_pad, y_pad, z_pad)
        >>> absorption_coeff = 1e-1
        >>> 
        >>> mask = create_absorption_mask(grid_shape, absorption_widths, absorption_coeff)
        >>> print(f"Grid shape: {grid_shape}")
        >>> print(f"Absorption mask shape: {mask.shape}")
        >>> print(f"Center region size: {grid_shape[0] - 2*absorption_widths[0]} x {grid_shape[1] - 2*absorption_widths[1]} x {grid_shape[2] - 2*absorption_widths[2]}")
        >>> 
        >>> # Create without visualization
        >>> mask = create_absorption_mask(grid_shape, absorption_widths, absorption_coeff, show_plots=False)
    """
    # Input validation
    if not isinstance(grid_shape, tuple) or len(grid_shape) != 3:
        raise ValueError("grid_shape must be a tuple of length 3")
    
    xx, yy, zz = grid_shape
    if not all(isinstance(d, int) and d > 0 for d in grid_shape):
        raise ValueError("grid_shape must contain positive integers")
    
    if not isinstance(absorption_widths, tuple) or len(absorption_widths) != 3:
        raise ValueError("absorption_widths must be a tuple of length 3")
    
    x_pad, y_pad, z_pad = absorption_widths
    if not all(isinstance(w, int) and w >= 0 for w in absorption_widths):
        raise ValueError("absorption_widths must contain non-negative integers")
    
    if not isinstance(absorption_coeff, (int, float)) or absorption_coeff < 0:
        raise ValueError("absorption_coeff must be a non-negative number")
    
    # Validate that absorption widths don't exceed half the grid dimensions
    max_x_width = xx // 2
    max_y_width = yy // 2
    max_z_width = zz // 2
    
    if x_pad > max_x_width:
        raise ValueError(
            f"x_pad ({x_pad}) cannot exceed half the x dimension ({max_x_width}). "
            f"Grid dimensions: {xx} x {yy} x {zz}"
        )
    if y_pad > max_y_width:
        raise ValueError(
            f"y_pad ({y_pad}) cannot exceed half the y dimension ({max_y_width}). "
            f"Grid dimensions: {xx} x {yy} x {zz}"
        )
    if z_pad > max_z_width:
        raise ValueError(
            f"z_pad ({z_pad}) cannot exceed half the z dimension ({max_z_width}). "
            f"Grid dimensions: {xx} x {yy} x {zz}"
        )
    
    # Create 3D absorption mask with inward padding
    absorption_mask = _absorption_mask_3d_custom(xx, yy, zz, x_pad, y_pad, z_pad, absorption_coeff)
    
    # Final validation
    if jnp.any(jnp.isnan(absorption_mask)):
        raise ValueError("Generated absorption mask contains NaN values")
    if jnp.any(jnp.isinf(absorption_mask)):
        raise ValueError("Generated absorption mask contains infinite values")
    
    # Visualize if requested
    if show_plots:
        _view_absorption_mask_internal(absorption_mask)
    
    return absorption_mask


def _view_absorption_mask_internal(
    absorption_mask: jax.Array,
    slice_positions: Optional[Tuple[float, float, float]] = None,
    figsize: Tuple[int, int] = (15, 4),
    cmap: str = 'grey_r',
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Visualize the 3D absorption mask for the 'absorbers' component (Ex).

    Plots three slices (XY, XZ, YZ) of the first field component,
    labeled as 'absorbers'.

    Args:
        absorption_mask: (3, xx, yy, zz) absorption array. Only index 0 is used.
        slice_positions: (x_frac, y_frac, z_frac) fractional positions (0–1).
                         Defaults to center slices.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap name.
        save_path: Optional path to save the figure.
        show_plot: Whether to display the plot.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # Validate input shape
    if absorption_mask.ndim != 4 or absorption_mask.shape[0] < 1:
        raise ValueError(
            f"absorption_mask must be at least 4D with first dimension ≥1, got {absorption_mask.shape}"
        )

    # Default to center slices
    if slice_positions is None:
        slice_positions = (0.5, 0.5, 0.5)
    x_frac, y_frac, z_frac = slice_positions

    _, xx, yy, zz = absorption_mask.shape
    x_idx = int(x_frac * (xx - 1))
    y_idx = int(y_frac * (yy - 1))
    z_idx = int(z_frac * (zz - 1))

    # Use only the first component and apply square-root scaling
    data = np.sqrt(np.array(absorption_mask[0]))
    vmin, vmax = data.min(), data.max()

    # Set up a 1×3 grid
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    # Descriptions for each slice
    slice_info = [
        ('XY', data[:, :, z_idx], (0, xx-1, 0, yy-1), 'X index', 'Y index'),
        ('XZ', data[:, y_idx, :], (0, xx-1, 0, zz-1), 'X index', 'Z index'),
        ('YZ', data[x_idx, :, :], (0, yy-1, 0, zz-1), 'Y index', 'Z index'),
    ]

    for i, (name, slice_data, extent, xlabel, ylabel) in enumerate(slice_info):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(
            slice_data.T,
            origin='upper',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect='equal'
        )
        ax.set_title(f'absorbers: {name} slice')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Colorbar with α label in a larger font
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(r'$\sqrt{\alpha}$', fontsize=12)
        cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f'3D Absorbers Mask slices at {slice_positions}',
        fontsize=14, fontweight='bold'
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)