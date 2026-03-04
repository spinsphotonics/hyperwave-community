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

from typing import Tuple, Optional, Dict
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
) -> jax.Array:
    """Create 3D adiabatic absorption mask with inward padding logic.

    This function creates a 3D absorption mask that:
    1. Uses inward padding: absorption occurs at grid boundaries
    2. Creates zero absorption in the center region (preserves device physics)
    3. Increases quadratically toward all boundaries (prevents reflections)
    4. Handles all three field components (Ex, Ey, Ez) with proper Yee grid offsets

    Args:
        grid_shape: Tuple of (xx, yy, zz) grid dimensions.
        absorption_widths: Tuple of (x_pad, y_pad, z_pad) absorption widths from each boundary.
        absorption_coeff: Maximum absorption strength at boundaries.

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
        >>> # Create absorption mask with inward padding
        >>> grid_shape = (100, 80, 60)  # (xx, yy, zz)
        >>> absorption_widths = (10, 8, 6)  # (x_pad, y_pad, z_pad)
        >>> absorption_coeff = 1e-1
        >>>
        >>> mask = create_absorption_mask(grid_shape, absorption_widths, absorption_coeff)
        >>> print(f"Grid shape: {grid_shape}")
        >>> print(f"Absorption mask shape: {mask.shape}")
        >>> print(f"Center region size: {grid_shape[0] - 2*absorption_widths[0]} x {grid_shape[1] - 2*absorption_widths[1]} x {grid_shape[2] - 2*absorption_widths[2]}")
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

    return absorption_mask


def rescale_absorption_mask(
    original_grid_shape: Tuple[int, int, int],
    original_absorption_widths: Tuple[int, int, int],
    original_absorption_coeff: float,
    new_grid_shape: Tuple[int, int, int],
) -> jax.Array:
    """Rescale absorption parameters to a new resolution while preserving the profile.

    This function takes the parameters used to create an absorption mask at one
    resolution and automatically scales them to work at a different resolution,
    preserving the physical absorption profile.

    Args:
        original_grid_shape: Tuple of (xx_old, yy_old, zz_old) original grid dimensions.
        original_absorption_widths: Tuple of (x_width, y_width, z_width) original absorption widths.
        original_absorption_coeff: Original absorption coefficient (smoothness parameter).
        new_grid_shape: Tuple of (xx_new, yy_new, zz_new) target grid dimensions.

    Returns:
        new_absorption_mask: (3, xx_new, yy_new, zz_new) rescaled absorption array.

    Algorithm:
        1. Calculate resolution scaling factors for each axis
        2. Scale absorption widths proportionally
        3. Scale coefficient to preserve physical absorption strength

    Example:
        >>> # Create mask at low resolution
        >>> mask_low = create_absorption_mask((100, 80, 60), (20, 15, 10), 0.05)
        >>>
        >>> # Rescale to high resolution using original parameters
        >>> mask_high = rescale_absorption_mask(
        ...     original_grid_shape=(100, 80, 60),
        ...     original_absorption_widths=(20, 15, 10),
        ...     original_absorption_coeff=0.05,
        ...     new_grid_shape=(200, 160, 120)
        ... )
        >>>
        >>> print(f"Rescaled shape: {mask_high.shape}")
    """
    xx_old, yy_old, zz_old = original_grid_shape
    width_x_old, width_y_old, width_z_old = original_absorption_widths
    xx_new, yy_new, zz_new = new_grid_shape

    # Calculate resolution scaling factors
    scale_x = xx_new / xx_old
    scale_y = yy_new / yy_old
    scale_z = zz_new / zz_old

    # Scale widths proportionally to new resolution
    new_width_x = int(width_x_old * scale_x)
    new_width_y = int(width_y_old * scale_y)
    new_width_z = int(width_z_old * scale_z)

    # Scale smoothness coefficient to preserve physical absorption
    # At physical distance d from boundary:
    #   - Old grid: d pixels away, absorption = coeff × d²
    #   - New grid: (scale × d) pixels away, absorption = new_coeff × (scale × d)²
    # To keep same absorption: new_coeff × (scale × d)² = coeff × d²
    # Therefore: new_coeff = coeff / scale²
    avg_scale = (scale_x + scale_y + scale_z) / 3
    new_coeff = original_absorption_coeff / (avg_scale ** 2)

    # Create new absorption mask with scaled parameters
    new_absorption_mask = create_absorption_mask(
        grid_shape=new_grid_shape,
        absorption_widths=(new_width_x, new_width_y, new_width_z),
        absorption_coeff=new_coeff,
    )

    return new_absorption_mask


def absorber_params(
    wavelength_um: float,
    dx_um: float,
    structure_dimensions: Tuple[int, int, int] = None,
) -> Dict[str, any]:
    """Compute absorber parameters from wavelength and grid spacing.

    Uses power-law fits to Bayesian-optimized results with minimum floors
    derived from validated simulation defaults. The BO was trained on
    {1310, 1550}nm x {25, 35, 50}nm configs (100 trials each).

    The Z-direction fit (R2=0.676) and coefficient fit (R2=0.887) are
    reliable. The XY fit (R2=0.133) is floored at 2.1 um because the
    BO training setup (straight-down source) did not exercise lateral
    absorption sufficiently.

    Args:
        wavelength_um: Wavelength in micrometers (e.g. 1.31 or 1.55).
        dx_um: Grid spacing in micrometers (e.g. 0.025 for 25nm).
        structure_dimensions: Optional (Lx, Ly, Lz) in grid cells.
            If provided, returns integer absorption_widths capped at 25%
            of each dimension.

    Returns:
        Dictionary with:
            - abs_xy_um: XY absorber width in micrometers
            - abs_z_um: Z absorber width in micrometers
            - abs_coeff: Absorption coefficient
            - absorption_widths: (x, y, z) int tuple (if structure_dimensions given)
    """
    wl = wavelength_um
    dx = dx_um

    # Power-law fits: param = a * wl^b * dx^c
    # From v2 BO (straight-down source, 100 trials, 6 configs)
    # Floors match the simulate() defaults at 35nm: (60, 40, 40) / 1e-4
    abs_xy_um = max(2.1, 0.062 * wl ** 1.389 * dx ** (-0.619))
    abs_z_um = max(1.4, 1.244 * wl ** 1.758 * dx ** 0.159)
    abs_coeff = max(1e-4, 2.876 * wl ** (-1.607) * dx ** 2.579)

    result = {
        "abs_xy_um": abs_xy_um,
        "abs_z_um": abs_z_um,
        "abs_coeff": abs_coeff,
    }

    if structure_dimensions is not None:
        Lx, Ly, Lz = structure_dimensions
        abs_xy = min(int(round(abs_xy_um / dx)), Lx // 4)
        abs_z = min(int(round(abs_z_um / dx)), Lz // 4)
        result["absorption_widths"] = (abs_xy, abs_xy, abs_z)

    return result


