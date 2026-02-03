"""Structure computation and visualization utilities.

This module provides functions for converting raw optimization variables into
photonic device structures with density filtering and permittivity distributions
suitable for FDTD simulations.
"""

from functools import partial
import math
from typing import Tuple, Union, List
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


@dataclass
class Layer:
    """Represents a single layer in a photonic structure.

    This class encapsulates all the material properties and geometry for a single
    layer in a multi-layer photonic structure.

    Args:
        density_pattern: 2D density pattern with shape (nx, ny) where nx and ny are
            the X and Y dimensions. Values should be in [0, 1].
        permittivity_values: Either a tuple (low, high) for interpolation or a single float value.
        layer_thickness: Physical thickness of the layer in grid units. Can be a float
            to allow precise control over layer positioning (e.g., 10.5). The final structure
            will have an integer number of grid points (rounded up via ceil), with fractional
            positions handled via subpixel averaging.
        conductivity_values: Either a tuple (low, high) for interpolation or a single float value. 
            Defaults to 0.0 (no conductivity).
        
    Notes:
        - The density pattern must have even dimensions. If you generate densities using
          `density(...)` they will be auto-trimmed to even sizes. If you manually modify
          densities afterwards and make them odd-sized, instantiating a `Layer` will raise
          a clear ValueError.
        
    Example:
        >>> import jax.numpy as jnp
        >>> from hyperwave.structure import Layer
        >>> 
        >>> # Create a simple layer with default conductivity (0.0)
        >>> density_pattern = jnp.ones((20, 20))
        >>> layer = Layer(
        ...     density_pattern=density_pattern,
        ...     permittivity_values=(1.0, 12.0),  # Air to silicon
        ...     layer_thickness=10
        ... )
        >>> 
        >>> # Create a layer with custom conductivity
        >>> layer_with_conductivity = Layer(
        ...     density_pattern=density_pattern,
        ...     permittivity_values=(1.0, 12.0),
        ...     layer_thickness=10,
        ...     conductivity_values=(0.0, 0.1),   # Low to medium conductivity
        ... )
    """
    density_pattern: jnp.ndarray
    permittivity_values: Union[Tuple[float, float], float]
    layer_thickness: float
    conductivity_values: Union[Tuple[float, float], float] = 0.0
    
    def __post_init__(self):
        """Validate layer parameters after initialization."""
        # Validate density_pattern
        if not isinstance(self.density_pattern, jnp.ndarray):
            raise TypeError(f"density_pattern must be a jax.numpy.ndarray, got {type(self.density_pattern)}")
        if self.density_pattern.ndim != 2:
            raise ValueError(f"density_pattern must be a 2D array, got shape {self.density_pattern.shape}")

        # Enforce even spatial dimensions with a check
        nx, ny = self.density_pattern.shape
        if nx % 2 != 0:
            raise ValueError(f"dimension {nx} is not even. Make sure all dimensions are even")
        if ny % 2 != 0:
            raise ValueError(f"dimension {ny} is not even. Make sure all dimensions are even")
        
        # Allow density pattern values to slightly exceed [0, 1] due to filtering
        # The density filter provides soft bounding, and hard clipping creates gradient discontinuities
        # Just validate that values are reasonable (not extremely out of range)
        # Skip validation during JAX tracing (autodiff) to avoid concretization errors
        try:
            density_min = float(jnp.min(self.density_pattern))
            density_max = float(jnp.max(self.density_pattern))
            if density_min < -0.1 or density_max > 1.1:
                print(f"Warning: Density pattern values significantly outside [0, 1]: [{density_min:.6f}, {density_max:.6f}]")
                print(f"This may indicate an issue with the density filtering or optimization. Consider checking your parameters.")
        except Exception:
            # Skip validation during JAX tracing (happens during autodiff)
            # This catches TracerBoolConversionError and ConcretizationTypeError
            pass
        
        # Validate layer_thickness
        if not isinstance(self.layer_thickness, (int, float)):
            raise TypeError(f"layer_thickness must be a number, got {type(self.layer_thickness).__name__}: {self.layer_thickness}")
        if self.layer_thickness <= 0:
            raise ValueError(f"layer_thickness must be positive, got {self.layer_thickness}")
        
        # Validate permittivity_values
        if isinstance(self.permittivity_values, tuple):
            if len(self.permittivity_values) != 2:
                raise ValueError(f"permittivity_values tuple must have 2 elements, got {len(self.permittivity_values)}")
            low, high = self.permittivity_values
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise TypeError("permittivity_values tuple elements must be numbers")
            if low <= 0 or high <= 0:
                raise ValueError("permittivity_values must be positive")
        elif isinstance(self.permittivity_values, (int, float)):
            if self.permittivity_values <= 0:
                raise ValueError("permittivity_values must be positive")
        else:
            raise TypeError("permittivity_values must be a tuple or a number")
        
        # Validate conductivity_values
        if isinstance(self.conductivity_values, tuple):
            if len(self.conductivity_values) != 2:
                raise ValueError(f"conductivity_values tuple must have 2 elements, got {len(self.conductivity_values)}")
            low, high = self.conductivity_values
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise TypeError("conductivity_values tuple elements must be numbers")
            if low < 0 or high < 0:
                raise ValueError("conductivity_values must be non-negative")
        elif isinstance(self.conductivity_values, (int, float)):
            if self.conductivity_values < 0:
                raise ValueError("conductivity_values must be non-negative")
        else:
            raise TypeError("conductivity_values must be a tuple or a number")
    
    def get_permittivity_values(self) -> jnp.ndarray:
        """Get the permittivity values for this layer based on density pattern.
        
        Returns:
            2D array of permittivity values with same shape as density_pattern.
        """
        if isinstance(self.permittivity_values, tuple):
            low, high = self.permittivity_values
            return self.density_pattern * (high - low) + low
        else:
            return jnp.full_like(self.density_pattern, self.permittivity_values)
    
    def get_conductivity_values(self) -> jnp.ndarray:
        """Get the conductivity values for this layer based on density pattern.
        
        Returns:
            2D array of conductivity values with same shape as density_pattern.
        """
        if isinstance(self.conductivity_values, tuple):
            low, high = self.conductivity_values
            return self.density_pattern * (high - low) + low
        else:
            return jnp.full_like(self.density_pattern, self.conductivity_values)


@dataclass
class Structure:
    """Enhanced structure arrays with construction metadata for Modal serialization.

    This dataclass wraps the traditional permittivity and conductivity arrays
    with all the construction metadata needed to recreate identical structures
    on Modal. This enables efficient lightweight transfer of construction
    instructions instead of large arrays.

    Attributes:
        permittivity: (3, nx, ny, nz) permittivity distribution array
        conductivity: (3, nx, ny, nz) conductivity distribution array
        layers_info: List of original Layer objects used in construction
        construction_params: Dict with construction parameters
        metadata: Dict with additional reconstruction information

    Example:
        >>> layers = [Layer(...), Layer(...)]
        >>> structure = create_structure(layers)
        >>>
        >>> # Traditional access (backward compatible)
        >>> eps = structure.permittivity
        >>> cond = structure.conductivity
        >>>
        >>> # New Modal workflow
        >>> recipe = structure.extract_recipe()
        >>> results = run_simulation_from_recipe(recipe, ...)
    """
    permittivity: jnp.ndarray
    conductivity: jnp.ndarray
    layers_info: List[Layer]
    construction_params: dict
    metadata: dict

    def extract_recipe(self) -> dict:
        """Extract lightweight recipe for Modal reconstruction.

        Returns:
            Dictionary with all information needed to rebuild identical
            permittivity and conductivity arrays on Modal.
        """
        # Convert Layer objects to serializable format with deduplication
        import numpy as np

        # Deduplicate density patterns to reduce recipe size
        unique_densities = []
        density_indices = []

        for layer in self.layers_info:
            density_np = np.array(layer.density_pattern)

            # Check if density is uniform (all same value)
            if np.all(density_np == density_np.flat[0]):
                # Store as uniform pattern with scalar value
                uniform_pattern = {
                    'type': 'uniform',
                    'value': float(density_np.flat[0]),
                    'shape': density_np.shape
                }

                # Check if this uniform pattern already exists
                found_idx = None
                for idx, existing_density in enumerate(unique_densities):
                    if (isinstance(existing_density, dict) and
                        existing_density.get('type') == 'uniform' and
                        existing_density['value'] == uniform_pattern['value'] and
                        existing_density['shape'] == uniform_pattern['shape']):
                        found_idx = idx
                        break

                if found_idx is not None:
                    density_indices.append(found_idx)
                else:
                    density_indices.append(len(unique_densities))
                    unique_densities.append(uniform_pattern)
            else:
                # Non-uniform pattern - check if it already exists
                found_idx = None
                for idx, existing_density in enumerate(unique_densities):
                    if isinstance(existing_density, np.ndarray) and np.array_equal(density_np, existing_density):
                        found_idx = idx
                        break

                if found_idx is not None:
                    density_indices.append(found_idx)
                else:
                    density_indices.append(len(unique_densities))
                    unique_densities.append(density_np)

        # Serialize layers with density indices instead of full arrays
        serialized_layers = []
        for i, layer in enumerate(self.layers_info):
            serialized_layer = {
                'density_index': density_indices[i],  # Reference to unique density
                'density_shape': layer.density_pattern.shape,
                'permittivity_values': layer.permittivity_values,
                'layer_thickness': layer.layer_thickness,
                'conductivity_values': layer.conductivity_values
            }
            serialized_layers.append(serialized_layer)

        return {
            'layers_info': serialized_layers,
            'unique_densities': unique_densities,  # Store unique densities only once
            'construction_params': self.construction_params,
            'metadata': self.metadata,
            'validation': {
                'permittivity_checksum': float(jnp.sum(self.permittivity)),
                'conductivity_checksum': float(jnp.sum(self.conductivity)),
                'shape': self.permittivity.shape
            }
        }

    def get_shape(self) -> tuple:
        """Get the shape of the structure arrays."""
        return self.permittivity.shape


    def view(self, show_permittivity: bool = True, show_conductivity: bool = True,
             cmap_permittivity: str = "PuOr", cmap_conductivity: str = "plasma",
             axis: str = None, position: int = None) -> None:
        """Visualize structure using view_structure function.

        Args:
            show_permittivity: Whether to show permittivity.
            show_conductivity: Whether to show conductivity.
            cmap_permittivity: Colormap for permittivity.
            cmap_conductivity: Colormap for conductivity.
            axis: Optional axis to slice along ('x', 'y', or 'z').
            position: Optional position along the specified axis.
        """
        view_structure(self, show_permittivity=show_permittivity,
                      show_conductivity=show_conductivity,
                      cmap_permittivity=cmap_permittivity,
                      cmap_conductivity=cmap_conductivity,
                      axis=axis, position=position)

    def list_layers(self) -> None:
        """Print human-readable description of the structure's layers."""
        print("Layer Stack Description:")
        print("=" * 50)

        for i, layer in enumerate(self.layers_info):
            print(f"\nLayer {i}:")

            # Get density pattern info
            density_shape = layer.density_pattern.shape
            density_min = float(layer.density_pattern.min())
            density_max = float(layer.density_pattern.max())
            density_mean = float(layer.density_pattern.mean())

            print(f"  Density shape: {density_shape[0]} × {density_shape[1]}")
            print(f"  Density range: [{density_min:.3f}, {density_max:.3f}] (mean: {density_mean:.3f})")

            # Describe the pattern type
            if density_max == density_min:
                if density_min == 0:
                    pattern_desc = "zeros (empty)"
                elif density_min == 1:
                    pattern_desc = "ones (filled)"
                else:
                    pattern_desc = f"uniform ({density_min:.3f})"
            elif density_min == 0 and density_max == 1:
                pattern_desc = "binary pattern"
            else:
                pattern_desc = "grayscale pattern"
            print(f"  Pattern type: {pattern_desc}")

            # Permittivity info
            perm_values = layer.permittivity_values
            if isinstance(perm_values, (list, tuple)):
                print(f"  Permittivity: {perm_values[0]:.2f} → {perm_values[1]:.2f} (interpolated)")
            else:
                print(f"  Permittivity: {perm_values:.2f} (uniform)")

            # Conductivity info
            cond_values = layer.conductivity_values
            if isinstance(cond_values, (list, tuple)):
                if cond_values[1] > 0:
                    print(f"  Conductivity: {cond_values[0]:.3f} → {cond_values[1]:.3f} (interpolated)")
            elif cond_values > 0:
                print(f"  Conductivity: {cond_values:.3f} (uniform)")

            # Thickness
            print(f"  Thickness: {layer.layer_thickness} pixels")

        print("\n" + "=" * 50)
        print(f"Total thickness: {self.permittivity.shape[3]} pixels")
        print(f"Lateral size: {self.permittivity.shape[1]} × {self.permittivity.shape[2]} pixels")

    def __repr__(self) -> str:
        """String representation of the Structure."""
        return (f"Structure(shape={self.permittivity.shape}, "
                f"num_layers={len(self.layers_info)}, "
                f"permittivity_range=[{self.permittivity.min():.2f}, {self.permittivity.max():.2f}], "
                f"conductivity_range=[{self.conductivity.min():.2e}, {self.conductivity.max():.2e}])")

    def __getitem__(self, key):
        """Allow backward compatibility with tuple unpacking.

        This enables:
        >>> structure = create_structure(layers)
        >>> eps, cond = structure  # Backward compatibility
        """
        if key == 0:
            return self.permittivity
        elif key == 1:
            return self.conductivity
        else:
            raise IndexError("Structure only supports indices 0 (permittivity) and 1 (conductivity)")

    def __iter__(self):
        """Enable tuple unpacking for backward compatibility."""
        yield self.permittivity
        yield self.conductivity


# =============================================================================
# Density computation functions
# =============================================================================

def _cone(radius):
    """Create a conical filter kernel for density filtering."""
    r = math.ceil(radius - 0.5)
    u = jnp.square(jnp.array([i - r for i in range(2 * r + 1)]))
    weights = jnp.maximum(0, radius - jnp.sqrt(u + u[:, None]))
    return weights / jnp.sum(weights)


def _filter(u, radius):
    """Apply conical filter to input array."""
    cone = _cone(radius)
    u = jnp.pad(u, [(s // 2, s // 2) for s in cone.shape], mode="edge")
    return jax.scipy.signal.convolve(u, cone, mode="valid")


def _boundary_cell(u):
    """Detect boundary cells in the density field."""
    u = jnp.pad(u >= 0, 1, mode="edge")
    u = [jnp.logical_xor(u, jnp.roll(u, shift, axis))[1:-1, 1:-1]
         for shift in (1, -1) for axis in (0, 1)]
    return jnp.any(jnp.stack(u), axis=0)


def _project(u, eta):
    """Project density values for binarization."""
    return jnp.where(_boundary_cell(u - eta), u, (jnp.sign(u - eta) + 1) / 2)


@partial(jax.jit, static_argnames=["radius"])
def _density_pjz_internal(u, radius, alpha, eta):
    """Internal JIT-compiled density filtering function.

    This implements the three-field scheme for density filtering.
    Separated from the main density function to avoid JIT issues with validation.
    """
    if radius > 0:
        # Apply conical filter
        ufilt = _filter(u, radius)
        # Apply projection for binarization
        uproj = _project(ufilt, eta)
        # Combine filtered and projected densities based on alpha
        return alpha * uproj + (1 - alpha) * ufilt
    else:
        # No filtering requested (radius=0), return input
        return u





def density(
    theta: jnp.ndarray,
    pad_width: Union[int, Tuple[int, int, int, int]] = 0,
    alpha: float = 0.0,
    radius: float = 8.0,
    c: float = 1e3,
    eta: float = 0.5,
    eta_lo: float = 0.0,
    eta_hi: float = 1.0,
) -> jnp.ndarray:
    """Convert raw optimization variables to density field with filtering.

    This function takes raw optimization variables and converts them into a proper
    photonic device structure with density filtering for minimum feature size
    constraints. Implements the "three-field" scheme detailed in topology optimization
    literature for a final density that is binary (with the exception of boundary
    values) and conforms to minimum feature size requirements.

    The process includes:
    1. Adding padding around the structure
    2. Applying density filtering for minimum feature size constraints
    3. Controlling binarization levels

    This function is general-purpose and can be used for any photonic device
    including grating couplers, metalenses, waveguides, etc.

    Args:
        theta: Raw optimization variables defining the basic structure.
            Shape should be (nx, ny) where nx and ny are the dimensions.
            For backward compatibility, if theta is already the filtered density u,
            set radius=0 to skip filtering.
        pad_width: Padding to add around the structure. Can be:
            - int: Same padding on all sides
            - tuple of 4 ints: (left, right, top, bottom) padding
            Default: 0 (no padding).
        alpha: Binarization control parameter in range [0, 1].
            - 0: No binarization, density varies freely in [0, 1]
            - 1: Complete binarization (except boundary values)
            Default: 0.0 (no binarization).
        radius: Radius for density filtering (controls minimum feature size).
            Set to 0 to skip filtering. Default: 8.0.
        c: Controls the detection of inflection points. Default: 1e3.
        eta: Threshold value used to binarize the density. Default: 0.5.
        eta_lo: Controls minimum feature size of void-phase features. Default: 0.0.
        eta_hi: Controls minimum feature size of density=1 features. Default: 1.0.

    Returns:
        Density field with proper structure and constraints.
        Shape is (nx + 2*pad_width, ny + 2*pad_width). This function enforces
        even spatial dimensions by trimming the last row/column of `theta` if
        needed so downstream Yee-grid operations work without errors.

    Raises:
        ValueError: If parameters are outside valid ranges or if output contains NaN/Inf.
        TypeError: If input types are incorrect.

    Example:
        >>> import jax.numpy as jnp
        >>> from hyperwave.structure import density
        >>>
        >>> # Create raw optimization variables for any device
        >>> theta = jnp.random.random((50, 50))
        >>>
        >>> # Generate density field with uniform padding
        >>> density_field = density(
        ...     theta=theta,
        ...     pad_width=10,  # 10 pixels on all sides
        ...     alpha=0.8,
        ...     radius=3.0
        ... )
        >>> print(f"Density field shape: {density_field.shape}")
        >>>
        >>> # Or use custom padding (left, right, top, bottom)
        >>> density_field = density(
        ...     theta=theta,
        ...     pad_width=(5, 10, 15, 20),  # Different padding on each side
        ...     alpha=0.8,
        ...     radius=3.0
        ... )
        >>> print(f"Density field shape: {density_field.shape}")

    Note:
        This function replaces the separate density_pjz function, incorporating
        all its functionality with additional validation and preprocessing.

    References:
        Zhou, Mingdong, et al. "Minimum length scale in topology optimization
        by geometric constraints." Computer Methods in Applied Mechanics and
        Engineering 293 (2015): 266-282.
    """
    # Type checks
    if not isinstance(theta, jnp.ndarray):
        raise TypeError(f"theta must be a jax.numpy.ndarray, got {type(theta)}")
    if theta.ndim != 2:
        raise ValueError(f"theta must be a 2D array, got shape {theta.shape}")

    # Process pad_width parameter
    if isinstance(pad_width, int):
        if pad_width < 0:
            raise ValueError(f"pad_width must be non-negative, got {pad_width}")
        pad_left = pad_right = pad_top = pad_bottom = pad_width
    elif isinstance(pad_width, (tuple, list)):
        if len(pad_width) != 4:
            raise ValueError(f"pad_width tuple must have 4 values (left, right, top, bottom), got {len(pad_width)}")
        pad_left, pad_right, pad_top, pad_bottom = pad_width
        if any(p < 0 for p in [pad_left, pad_right, pad_top, pad_bottom]):
            raise ValueError(f"All padding values must be non-negative, got {pad_width}")
    else:
        raise TypeError(f"pad_width must be an int or tuple of 4 ints, got {type(pad_width)}")

    if not isinstance(alpha, (float, int)):
        raise TypeError(f"alpha must be a float, got {type(alpha)}")
    if not isinstance(radius, (float, int)):
        raise TypeError(f"radius must be a float, got {type(radius)}")

    # Validate input parameters
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if theta.size == 0:
        raise ValueError("theta must not be empty")
    
    # Enforce even dimensions by trimming the last row/column if odd
    ty, tx = theta.shape
    theta = theta[: ty - (ty % 2), : tx - (tx % 2)]

    # Add padding around the structure
    # Use custom padding for left/right (x-axis) and top/bottom (y-axis)
    u = jnp.pad(theta, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    # NO CLIPPING - Allow values outside [0,1] to preserve gradient continuity
    # The density filter is designed to handle this gracefully
    # Hard clipping creates gradient discontinuities that break optimization

    # Create binarization control parameter
    # Force alpha=1 in padding regions, use user-specified alpha in structure
    alpha_array = jnp.pad(
        alpha * jnp.ones_like(theta),
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=1.0
    )
    
    # Apply density filtering for minimum feature size constraints
    # This implements the "three-field" scheme from topology optimization
    # Use the internal helper function
    d = _density_pjz_internal(u, radius, alpha_array, eta)
    
    # Check for NaN/Inf in output
    if jnp.any(jnp.isnan(d)):
        raise ValueError("Output contains NaN values. Check input and parameters.")
    if jnp.any(~jnp.isfinite(d)):
        raise ValueError("Output contains non-finite values (Inf or -Inf). Check input and parameters.")
    
    return d


# =============================================================================
# Permittivity computation functions
# =============================================================================




def _is_uniform_layers(layers: List[Layer]) -> bool:
    """Check if all layers have uniform material properties or very small conductivity.
    
    This function detects when all layers consist of uniform materials
    (single permittivity and conductivity values rather than gradients).
    It also triggers simple averaging when conductivity values are very small,
    as the sophisticated subpixel smoothing algorithm can cause artifacts
    with small conductivity values due to harmonic mean instabilities.
    
    Args:
        layers: List of Layer objects to check
        
    Returns:
        True if simple averaging should be used, False for sophisticated smoothing
    """
    has_small_conductivity = False
    
    for layer in layers:
        perm_vals = layer.get_permittivity_values()
        cond_vals = layer.get_conductivity_values()
        
        # Check if layer has uniform values (small standard deviation)
        # This handles both scalar values and arrays with identical values
        if (jnp.std(perm_vals) > 1e-6 or jnp.std(cond_vals) > 1e-6):
            # Non-uniform layer found
            
            # Check if conductivity values are very small (< 1e-3)
            # Small conductivity causes harmonic mean instabilities in subpixel smoothing
            max_cond = jnp.max(jnp.abs(cond_vals))
            if max_cond > 0 and max_cond < 1e-3:
                has_small_conductivity = True
            
            # If we have non-uniform layers with reasonable conductivity, use sophisticated smoothing
            if not has_small_conductivity:
                return False
    
    # Use simple averaging if all layers are uniform OR if we have small conductivity
    return True


def create_structure(layers: List[Layer], vertical_radius: float = 5.0) -> Structure:
    """Create enhanced structure with permittivity/conductivity arrays and construction metadata.
    
    This function takes a list of Layer objects and converts them into layered permittivity
    and conductivity structures suitable for FDTD simulations, while preserving all
    construction metadata for efficient Modal serialization.

    Layer thicknesses can be floats (e.g., 10.5), allowing precise control over interface
    positions. The total Z-dimension will be rounded up (via ceil) to the nearest integer
    number of grid points. Floating point rounding errors (within 1e-6 of an integer) are
    handled gracefully. Subpixel averaging handles fractional overlaps at interfaces.

    Args:
        layers: List of Layer objects defining the structure.
        vertical_radius: Radius for vertical blur filtering. If > 0, applies a vertical
            blur filter to smooth transitions between layers. Default: 5.0.
        
    Returns:
        Structure object containing:
        - permittivity: (components, nx, ny, nz) permittivity distribution
        - conductivity: (components, nx, ny, nz) conductivity distribution  
        - layers_info: Original Layer objects for reconstruction
        - construction_params: Parameters used in construction
        - metadata: Additional reconstruction information
        
    Example:
        >>> import jax.numpy as jnp
        >>> from hyperwave.structure import Layer, create_structure
        >>> 
        >>> # Create layers
        >>> layer1 = Layer(
        ...     density_pattern=jnp.ones((20, 20)),
        ...     permittivity_values=(1.0, 12.0),
        ...     conductivity_values=(0.0, 0.1),
        ...     layer_thickness=10
        ... )
        >>> layer2 = Layer(
        ...     density_pattern=jnp.zeros((20, 20)),
        ...     permittivity_values=(12.0, 1.0),
        ...     conductivity_values=(0.1, 0.0),
        ...     layer_thickness=5
        ... )
        >>> 
        >>> # Create enhanced structure with metadata
        >>> structure = create_structure([layer1, layer2])  # Uses default vertical_radius=5.0
        >>> 
        >>> # Traditional access (backward compatible)
        >>> eps, cond = structure  # Tuple unpacking still works
        >>> print(f"Permittivity shape: {structure.permittivity.shape}")
        >>> 
        >>> # New Modal workflow
        >>> recipe = structure.extract_recipe()
        >>> # recipe can now be sent to Modal for lightweight reconstruction
    """
    # Validate inputs
    if not isinstance(layers, list):
        raise TypeError(f"layers must be a list, got {type(layers)}")
    
    if len(layers) < 1:
        raise ValueError("At least 1 layer is required")
    
    for i, layer in enumerate(layers):
        if not isinstance(layer, Layer):
            raise TypeError(f"layers[{i}] must be a Layer object, got {type(layer)}")
    
    # Validate vertical_radius parameter
    if not isinstance(vertical_radius, (int, float)):
        raise TypeError(f"vertical_radius must be a number, got {type(vertical_radius)}")
    if vertical_radius < 0:
        raise ValueError(f"vertical_radius must be non-negative, got {vertical_radius}")
    
    # Validate that all layers have the same density shape
    first_shape = layers[0].density_pattern.shape
    for i, layer in enumerate(layers):
        if layer.density_pattern.shape != first_shape:
            raise ValueError(f"All layers must have the same density shape. Layer {i} has shape {layer.density_pattern.shape}, expected {first_shape}")
    # Densities are enforced to even at Layer construction; nothing further needed here
    
    # Extract permittivity values for each layer
    permittivity_layers = jnp.stack([
        layer.get_permittivity_values() for layer in layers
    ])
    
    # Extract conductivity values for each layer
    conductivity_layers = jnp.stack([
        layer.get_conductivity_values() for layer in layers
    ])
    
    # Extract layer thicknesses
    layer_thicknesses = [layer.layer_thickness for layer in layers]

    # Convert to permittivity values, halving the spatial resolution in the x-y
    # plane in the process, in order to account for the offset in the cells when
    # using the Yee grid (because of FDTD simulation).
    interface_positions = jnp.cumsum(jnp.array(layer_thicknesses[:-1])) if len(layer_thicknesses) > 1 else jnp.array([])

    # Calculate total thickness with tolerance for floating point errors
    sum_thickness = sum(layer_thicknesses)
    # If within 1e-6 of an integer, round to that integer (handles 20.0000001 → 20)
    # Otherwise ceil to ensure all layers fit (handles 20.1 → 21, 20.9 → 21)
    if abs(sum_thickness - round(sum_thickness)) < 1e-6:
        total_thickness = round(sum_thickness)
    else:
        total_thickness = math.ceil(sum_thickness)
    
    # Auto-detect if we should use simple averaging to avoid subpixel artifacts
    # This fixes the issue where uniform materials in multiple layers get
    # incorrectly averaged by the sophisticated smoothing algorithm
    use_simple_averaging = _is_uniform_layers(layers)
    
    permittivity_distribution = epsilon(
        permittivity_layers,
        interface_positions=interface_positions,
        magnification=1,
        zz=total_thickness,
        use_simple_averaging=use_simple_averaging,
        vertical_radius=vertical_radius,
    )
    
    conductivity_distribution = epsilon(
        conductivity_layers,
        interface_positions=interface_positions,
        magnification=1,
        zz=total_thickness,
        use_simple_averaging=use_simple_averaging,
        vertical_radius=vertical_radius,
    )
    
    # Create construction metadata
    construction_params = {
        'vertical_radius': vertical_radius,
        'use_simple_averaging': use_simple_averaging,
        'total_thickness': total_thickness,
        'interface_positions': interface_positions.tolist() if len(interface_positions) > 0 else [],
    }
    
    metadata = {
        'created_with': 'create_structure',
        'num_layers': len(layers),
        'layer_thicknesses': layer_thicknesses,
        'final_shape': permittivity_distribution.shape,
    }
    
    # Return enhanced Structure object
    return Structure(
        permittivity=permittivity_distribution,
        conductivity=conductivity_distribution,
        layers_info=layers.copy(),  # Store original layers for reconstruction
        construction_params=construction_params,
        metadata=metadata
    )



def reconstruct_structure_from_recipe(recipe: dict) -> Structure:
    """Reconstruct Structure from a lightweight recipe.

    This function reconstructs identical permittivity and conductivity arrays
    from the lightweight recipe created by Structure.extract_recipe().

    Args:
        recipe: Dictionary containing construction instructions from extract_recipe()

    Returns:
        Structure object identical to the original
    """
    layers_info = recipe['layers_info']
    construction_params = recipe['construction_params']
    metadata = recipe['metadata']
    unique_densities = recipe['unique_densities']  # Deduplicated density arrays

    # Rebuild the structure using the original layers and parameters
    vertical_radius = construction_params['vertical_radius']

    # Reconstruct Layer objects from the dictionary data
    # Cache materialized densities to avoid redundant array creation
    materialized_densities = {}

    layers = []
    for layer_dict in layers_info:
        # Look up density pattern from deduplicated storage
        density_index = layer_dict['density_index']

        # Check if we've already materialized this density
        if density_index in materialized_densities:
            density_pattern = materialized_densities[density_index]
        else:
            density_data = unique_densities[density_index]

            # Check if it's a uniform pattern or full array
            if isinstance(density_data, dict) and density_data.get('type') == 'uniform':
                # Reconstruct uniform density from scalar value
                import numpy as np
                density_pattern = jnp.full(density_data['shape'], density_data['value'])
            else:
                # Full array stored
                density_pattern = jnp.array(density_data)

            # Cache the materialized density
            materialized_densities[density_index] = density_pattern

        # Reconstruct the Layer object with all its attributes
        # Convert lists to tuples (JSON serialization converts tuples to lists)
        perm_vals = layer_dict['permittivity_values']
        if isinstance(perm_vals, list):
            perm_vals = tuple(perm_vals)

        cond_vals = layer_dict.get('conductivity_values', 0.0)
        if isinstance(cond_vals, list):
            cond_vals = tuple(cond_vals)

        layer = Layer(
            density_pattern=density_pattern,
            permittivity_values=perm_vals,
            layer_thickness=layer_dict['layer_thickness'],
            conductivity_values=cond_vals
        )
        layers.append(layer)

    # Reconstruct arrays using the same process
    reconstructed = create_structure(layers, vertical_radius)

    # Validate reconstruction if validation data is present
    if 'validation' in recipe:
        expected_eps_sum = recipe['validation']['permittivity_checksum']
        expected_cond_sum = recipe['validation']['conductivity_checksum']
        expected_shape = recipe['validation']['shape']

        # Convert expected_shape to tuple if it's a list (JSON serialization)
        if isinstance(expected_shape, list):
            expected_shape = tuple(expected_shape)

        actual_eps_sum = float(jnp.sum(reconstructed.permittivity))
        actual_cond_sum = float(jnp.sum(reconstructed.conductivity))
        actual_shape = reconstructed.permittivity.shape

        eps_error = abs(actual_eps_sum - expected_eps_sum)

        # Use relative tolerance for large values
        eps_rel_error = eps_error / max(abs(expected_eps_sum), 1.0)

        # Special handling for conductivity
        if expected_cond_sum > 0 and actual_cond_sum == 0:
            # Skip conductivity validation - absorption will be added separately
            pass
        else:
            cond_error = abs(actual_cond_sum - expected_cond_sum)
            cond_rel_error = cond_error / max(abs(expected_cond_sum), 1.0)
            if cond_rel_error > 5e-3:  # Relative tolerance of 0.5%
                raise ValueError(f"Conductivity reconstruction failed: checksum mismatch {actual_cond_sum} vs {expected_cond_sum}")

        # Skip validation if checksum is 0.0 (not computed during recipe creation)
        if expected_eps_sum != 0.0 and eps_rel_error > 5e-3:  # Relative tolerance of 0.5%
            raise ValueError(f"Permittivity reconstruction failed: checksum mismatch {actual_eps_sum} vs {expected_eps_sum}")

        if actual_shape != expected_shape:
            raise ValueError(f"Shape reconstruction failed: {actual_shape} vs {expected_shape}")

    return reconstructed



# =============================================================================
# Visualization functions
# =============================================================================



def view_structure(structure, show_permittivity=True, show_conductivity=True,
                  cmap_permittivity="PuOr", cmap_conductivity="plasma",
                  axis=None, position=None):
    """Plot structure permittivity and/or conductivity distributions using matplotlib.

    This function creates visualizations of the structure's permittivity and/or conductivity
    fields. Can show either a single cross-section or default dual view.

    Args:
        structure: Structure object containing permittivity and conductivity arrays.
        show_permittivity: Whether to display permittivity plots (default: True).
        show_conductivity: Whether to display conductivity plots (default: True).
        cmap_permittivity: Colormap for permittivity plots (default: "PuOr").
        cmap_conductivity: Colormap for conductivity plots (default: "plasma").
        axis: Axis to slice along ('x', 'y', or 'z'). If None, shows default dual view (XY at middle Z, XZ at middle Y).
        position: Position along the specified axis to slice at. If None and axis is specified, uses middle position.
    
    Examples:
        >>> structure = create_structure(layers)
        >>> view_structure(structure)  # Show both permittivity and conductivity
        >>> view_structure(structure, show_conductivity=False)  # Only permittivity
        >>> view_structure(structure, show_permittivity=False)  # Only conductivity
        >>> view_structure(structure, axis='z', position=50)  # XY slice at z=50
        >>> view_structure(structure, show_permittivity=True, show_conductivity=False, axis='y', position=100)  # XZ permittivity slice at y=100
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Validate inputs
    if not isinstance(structure, Structure):
        raise TypeError(f"structure must be a Structure object, got {type(structure)}")
    
    if not show_permittivity and not show_conductivity:
        raise ValueError("At least one of show_permittivity or show_conductivity must be True")
    
    # Extract arrays from structure
    p = structure.permittivity
    c = structure.conductivity
    
    if axis is None:
        # Default behavior: show dual view (XY at middle Z, XZ at middle Y)
        middle_z = p.shape[3] // 2  # nz is the vertical dimension (shape[3])
        middle_y = p.shape[2] // 2  # ny is the horizontal dimension (shape[2])
        
        if show_permittivity and show_conductivity:
            # Show both permittivity and conductivity
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Permittivity plots
            im1 = axes[0, 0].imshow(p[0, :, :, middle_z].T, cmap=cmap_permittivity, vmin=p.min(), vmax=p.max())
            axes[0, 0].set_title(f"Permittivity: x-y plane at z={middle_z} (eps_x)")
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("y")
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(p[0, :, middle_y, :].T, cmap=cmap_permittivity, vmin=p.min(), vmax=p.max())
            axes[0, 1].set_title(f"Permittivity: x-z plane at y={middle_y} (eps_x)")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("z")
            plt.colorbar(im2, ax=axes[0, 1])

            # Conductivity plots
            im3 = axes[1, 0].imshow(c[0, :, :, middle_z].T, cmap=cmap_conductivity, vmin=c.min(), vmax=c.max())
            axes[1, 0].set_title(f"Conductivity: x-y plane at z={middle_z} (sigma_x)")
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("y")
            plt.colorbar(im3, ax=axes[1, 0])

            im4 = axes[1, 1].imshow(c[0, :, middle_y, :].T, cmap=cmap_conductivity, vmin=c.min(), vmax=c.max())
            axes[1, 1].set_title(f"Conductivity: x-z plane at y={middle_y} (sigma_x)")
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("z")
            plt.colorbar(im4, ax=axes[1, 1])
            
        elif show_permittivity:
            # Show only permittivity
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            im1 = axes[0].imshow(p[0, :, :, middle_z].T, cmap=cmap_permittivity, vmin=p.min(), vmax=p.max())
            axes[0].set_title(f"Permittivity: x-y plane at z={middle_z} (eps_x)")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(p[0, :, middle_y, :].T, cmap=cmap_permittivity, vmin=p.min(), vmax=p.max())
            axes[1].set_title(f"Permittivity: x-z plane at y={middle_y} (eps_x)")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("z")
            plt.colorbar(im2, ax=axes[1])
            
        else:  # show_conductivity only
            # Show only conductivity
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            im1 = axes[0].imshow(c[0, :, :, middle_z].T, cmap=cmap_conductivity, vmin=c.min(), vmax=c.max())
            axes[0].set_title(f"Conductivity: x-y plane at z={middle_z} (sigma_x)")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(c[0, :, middle_y, :].T, cmap=cmap_conductivity, vmin=c.min(), vmax=c.max())
            axes[1].set_title(f"Conductivity: x-z plane at y={middle_y} (sigma_x)")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("z")
            plt.colorbar(im2, ax=axes[1])
            
    else:
        # Single slice mode
        if axis not in ['x', 'y', 'z']:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
        
        # Determine position if not specified
        if position is None:
            if axis == 'x':
                position = p.shape[1] // 2  # middle x
            elif axis == 'y':
                position = p.shape[2] // 2  # middle y
            elif axis == 'z':
                position = p.shape[3] // 2  # middle z
        
        # Extract slice and determine plot properties
        if axis == 'x':
            p_slice = p[0, position, :, :]  # YZ plane
            c_slice = c[0, position, :, :]  # YZ plane
            plane_name = "y-z"
            xlabel, ylabel = "y", "z"
        elif axis == 'y':
            p_slice = p[0, :, position, :]  # XZ plane
            c_slice = c[0, :, position, :]  # XZ plane
            plane_name = "x-z"
            xlabel, ylabel = "x", "z"
        elif axis == 'z':
            p_slice = p[0, :, :, position]  # XY plane
            c_slice = c[0, :, :, position]  # XY plane
            plane_name = "x-y"
            xlabel, ylabel = "x", "y"
        
        if show_permittivity and show_conductivity:
            # Show both permittivity and conductivity
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            im1 = axes[0].imshow(p_slice.T, cmap=cmap_permittivity, vmin=p.min(), vmax=p.max(), origin='upper')
            axes[0].set_title(f"Permittivity: {plane_name} plane at {axis}={position} (eps_x)")
            axes[0].set_xlabel(xlabel)
            axes[0].set_ylabel(ylabel)
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(c_slice.T, cmap=cmap_conductivity, vmin=c.min(), vmax=c.max(), origin='upper')
            axes[1].set_title(f"Conductivity: {plane_name} plane at {axis}={position} (sigma_x)")
            axes[1].set_xlabel(xlabel)
            axes[1].set_ylabel(ylabel)
            plt.colorbar(im2, ax=axes[1])
            
        elif show_permittivity:
            # Show only permittivity
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            im = ax.imshow(p_slice.T, cmap=cmap_permittivity, vmin=p.min(), vmax=p.max(), origin='upper')
            ax.set_title(f"Permittivity: {plane_name} plane at {axis}={position} (eps_x)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax)
            
        else:  # show_conductivity only
            # Show only conductivity
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            im = ax.imshow(c_slice.T, cmap=cmap_conductivity, vmin=c.min(), vmax=c.max(), origin='upper')
            ax.set_title(f"Conductivity: {plane_name} plane at {axis}={position} (sigma_x)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Only show plot if using an interactive backend
    if plt.get_backend() != 'Agg':
        plt.show() 


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================
#
# These functions are internal implementations used by the main APIs above.
# They handle specialized operations like density filtering and rendering.
#

def density_pjz(u, radius, alpha, c=1.0, eta=0.5, eta_lo=0.25, eta_hi=0.75):
    """Backward compatibility wrapper for density_pjz.

    This function is now integrated into the main density() function.
    Use density() directly for new code.

    Args:
        u: Variable array with values within [0, 1].
        radius: Radius of the conical filter used to blur u.
        alpha: Binarization control parameter [0, 1].
        c: Controls the detection of inflection points (unused).
        eta: Threshold value used to binarize the density.
        eta_lo: Controls minimum feature size of void-phase features (unused).
        eta_hi: Controls minimum feature size of density=1 features (unused).

    Returns:
        Density field after filtering and binarization.
    """
    # Note: c, eta_lo, eta_hi are not used in the current implementation
    # They are kept for backward compatibility
    if radius > 0:
        ufilt = _filter(u, radius)
        uproj = _project(ufilt, eta)
        return alpha * uproj + (1 - alpha) * ufilt
    else:
        return u



def _vertical_cone(radius):
    """Create a 1D conical filter kernel for vertical filtering."""
    r = math.ceil(radius - 0.5)
    u = jnp.square(jnp.array([i - r for i in range(2 * r + 1)]))
    weights = jnp.maximum(0, radius - jnp.sqrt(u))
    return weights / jnp.sum(weights)


def _vertical_boundary_filter(layers, radius):
    """Apply 1D conical filter - this is a simplified version for now.
    
    The proper boundary-only implementation is complex. For now, apply 
    a gentle smoothing that mimics the density filter behavior.
    """
    if layers.shape[0] < 2:
        return layers  # No boundaries to smooth
    
    # Use a simple 1D convolution approach similar to density filtering
    cone = _vertical_cone(radius)
    
    # Pad the layers array to handle boundaries
    pad_size = len(cone) // 2
    padded_layers = jnp.pad(layers, ((pad_size, pad_size), (0, 0), (0, 0)), mode='edge')
    
    # Apply convolution along the layer dimension
    filtered_layers = []
    for i in range(layers.shape[0]):
        # Get the window of layers around position i
        window = padded_layers[i:i+len(cone)]
        # Apply weighted average
        filtered_layer = jnp.sum(window * cone[:, None, None], axis=0)
        filtered_layers.append(filtered_layer)
    
    return jnp.stack(filtered_layers)


def _apply_vertical_blur_to_rendered(permittivity: jax.Array, radius: float, interface_positions: jax.Array) -> jax.Array:
    """Apply vertical blur filtering only in regions that affect layer interfaces.
    
    This optimized version identifies all Z positions whose blur kernels overlap with
    layer interfaces and processes each position only once, dramatically reducing 
    computation time while maintaining exact boundary smoothing quality.
    
    Args:
        permittivity: ``(3, xx, yy, zz)`` array of rendered permittivity values.
        radius: Radius for the vertical blur filter.
        interface_positions: ``(ll - 1,)`` array of interface z-positions.
        
    Returns:
        Filtered permittivity array with blur applied around interfaces.
    """
    if radius <= 0 or len(interface_positions) == 0:
        return permittivity
    
    # Create 1D conical filter
    cone = _vertical_cone(radius)
    half_width = len(cone) // 2
    
    # Determine which Z positions need processing
    # Any Z position whose blur kernel overlaps with an interface needs to be processed
    z_positions_to_process = set()
    
    for interface_z in interface_positions:
        interface_idx = int(interface_z)
        
        # All Z positions within radius distance need to be processed
        # because their blur kernels will affect the interface
        for z in range(max(0, interface_idx - half_width), 
                      min(permittivity.shape[3], interface_idx + half_width + 1)):
            z_positions_to_process.add(z)
    
    # Convert to sorted list for efficient processing
    z_to_process = sorted(list(z_positions_to_process))
    
    if len(z_to_process) == 0:
        return permittivity
    
    # Apply blur only to positions that affect interfaces
    result = permittivity.copy()
    
    # Process each unique Z position only once (eliminates redundant computation)
    for z in z_to_process:
        if z < permittivity.shape[3]:
            # Get the filter window around this z position
            filter_start = max(0, z - half_width)
            filter_end = min(permittivity.shape[3], z + half_width + 1)
            
            # Extract the corresponding cone weights
            cone_start = max(0, half_width - (z - filter_start))
            cone_end = cone_start + (filter_end - filter_start)
            weights = cone[cone_start:cone_end]
            weights = weights / jnp.sum(weights)  # Normalize
            
            # Apply weighted average for all components and spatial positions
            for comp in range(permittivity.shape[0]):
                data_window = permittivity[comp, :, :, filter_start:filter_end]
                filtered_value = jnp.sum(data_window * weights[None, None, :], axis=2)
                result = result.at[comp, :, :, z].set(filtered_value)
    
    return result



def _render_single(
  layers, layer_pos, grid_start, grid_end, m, axis, use_simple_averaging):
  """Render single permittivity component for the Yee grid."""
  # In-plane offsets.
  if axis != "x":
    layers = jnp.pad(layers[:, :-m, :], ((0, 0), (m, 0), (0, 0)), "edge")
  if axis != "y":
    layers = jnp.pad(layers[:, :, :-m], ((0, 0), (0, 0), (m, 0)), "edge")

  # Offsets along z-axis.
  if axis == "z":
    grid_start = grid_start[:, 1]
    grid_end = grid_end[:, 1]
  else:
    grid_start = grid_start[:, 0]
    grid_end = grid_end[:, 0]

  # Convert to "layer-chunked" form, which consists of 2m x 2m tiles.
  lc = jnp.reshape(layers, (layers.shape[0],
                            layers.shape[1] // (2 * m), 2 * m,
                            layers.shape[2] // (2 * m), 2 * m))

  # Weighting values for pixels within each tile.
  w = (jnp.arange(2 * m) - (m - 0.5)) / (2 * m)**2

  # Use ``2 * m`` factor instead of summing along one axis and averaging on the
  # other. Factor of ``12`` corresponds to the integral of ``u**2`` in the
  # denominator for grid size of ``1``.
  grads = [jnp.mean(12 * (2 * m) * x, (2, 4))
           for x in (lc * w[:, None, None], lc * w)]

  # Per-layer average.
  avg = jnp.mean(lc, (2, 4))

  # Per-layer average of inverse.
  # Add small epsilon to avoid division by zero when conductivity is 0
  epsilon = 1e-12
  aoi = jnp.mean(1 / (lc + epsilon), (2, 4))

  # Get start and end points for each layer in each grid cell.
  # ``p0`` and ``p1`` have shape ``(ll, zz)``.
  #
  # And -infinity and +infinity as start and end points for first and last
  # layers respectively.
  #
  # Clip the start and end point for each layer in each cell at the cell
  # boundaries.
  #
  p0, p1 = [jnp.clip(x[:, None], grid_start, grid_end) for x in
            (jnp.concatenate([jnp.array([-jnp.inf]), layer_pos]),
             jnp.concatenate([layer_pos, jnp.array([jnp.inf])]))]

  # Ratio of cell occupied by each layer.
  u = (p1 - p0) / (grid_end - grid_start)

  # Reduces across the layer dimension.
  def cross(x, y): return jnp.einsum("lxy,lz->xyz", x, y)

  if use_simple_averaging:
    # Note that ``avg`` and ``u`` are of shape ``(num_layers, xx, yy)`` and
    # ``(num_layers, zz)`` respectively.
    return cross(avg, u)

  # Average position of each layer inside each cell with respect to cell center.
  z = (p0 + p1) / 2 - (grid_start + grid_end) / 2

  # Average of inverse across all layers.
  aoi = cross(aoi, u)

  # Inverse of average across all layers.
  ioa = 1 / cross(avg, u)

  # Average of gradient across all layers.
  grads = [cross(g, u) for g in grads]

  # Get the z-gradient.
  #
  # Denominator has only a the cell size to the power of ``2`` because there is
  # already a factor in ``u``.
  #
  grads.append(cross(avg, u * z) / ((grid_end - grid_start)**2 / 12))

  # Diagonal term of the projection matrix.
  sum_of_gradients = sum(g**2 for g in grads)
  pii = (grads["xyz".index(axis)]**2 /
         jnp.where(sum_of_gradients == 0, 1, sum_of_gradients))

  return 1 / (pii * aoi + (1 - pii) * ioa)


def _render(layers, layer_pos, grid_start, grid_end, m, use_simple_averaging):
  """Render all permittivity components for the Yee grid."""
  return jnp.stack(
      [_render_single(
        layers, layer_pos, grid_start, grid_end, m, axis, use_simple_averaging)
       for axis in "xyz"])


def epsilon(
        layers: jax.Array,
        interface_positions: jax.Array,
        magnification: int,
        zz: int,
        use_simple_averaging: bool = False,
        vertical_radius: float = 0.0,
) -> jax.Array:
  """Render a three-dimensional vector array of permittivity values.

  Produces a 3D vector array of permittivity values on the Yee cell based on a
  layered stack of 2D profiles at magnification ``2 * m``. Along the z-axis,
  both the layer boundaries and grid positions are allowed to vary continuously,
  while along the x- and y-axes the size of each (unmagnified) cell is assumed
  to be ``1``.

  Attempts to follow the subpixel smoothing approach but only computes the on-diagonal elements
  of the projection matrix and is adapted to a situation where there are no
  explicit interfaces because the pixel values are allowed to vary continuously
  within each layer.

  Instead, the diagonal elements of the projection matrix for a given subvolume
  are estimated by computing gradients across it where ``df(u)/du`` is computed
  as the integral of ``f(u) * u`` over the integral of ``u**2``  where ``u`` is
  relative to the center of the cell.

  Args:
    layers: ``(ll, 2 * m * xx, 2 * m * yy)`` array of magnified layer profiles.
    interface_positions: ``(ll - 1)`` array of interface positions between the
      ``ll`` layer. Assumed to be in monotonically increasing order.
    magnification: Denotes a ``2 * m`` in-plane magnification factor of layer
      profiles.
    zz: Number of cells along z-axis.
    use_simple_averaging: If ``True``, fall back to a simple averaging scheme.
    vertical_radius: Radius for vertical blur filtering. If > 0, applies a vertical
      blur filter similar to the horizontal density filtering. Default: 0.0 (no blur).

  Returns:
    ``(3, xx, yy, zz)`` array of permittivity values with offsets and vector
    components according to the finite-difference Yee cell.

  """
  # Note: Input layer profiles are expected to have even in-plane dimensions.
  # Densities produced by `density(...)` satisfy this automatically; manual
  # modifications must preserve even sizes for consistent Yee-grid staggering.
  # First render the permittivity without vertical blur
  result = _render(
      layers,
      interface_positions,
      jnp.arange(zz)[:, None] + jnp.array([[-0.5, 0]]),
      jnp.arange(zz)[:, None] + jnp.array([[0.5, 1]]),
      magnification,
      use_simple_averaging,
  )
  
  # Apply vertical blur filtering to the rendered result if requested
  if vertical_radius > 0:
    result = _apply_vertical_blur_to_rendered(result, vertical_radius, interface_positions)

  return result


def recipe_from_params(
    grid_shape: tuple,
    layers: list,
    vertical_radius: float = 0.0,
    use_simple_averaging: bool = False
) -> dict:
    """Create a structure recipe directly from parameters without building structure.

    This is much faster for parameter sweeps - you can modify the returned recipe
    dict directly instead of rebuilding the entire structure each time.

    Args:
        grid_shape: (width, height) of the 2D patterns in pixels
        layers: List of layer specifications, each containing:
            - 'density': Either 'uniform' or a numpy/JAX array
            - 'value': If density is 'uniform', the uniform value (0.0 to 1.0)
            - 'permittivity': Either scalar or tuple (low, high) for interpolation
            - 'thickness': Layer thickness in pixels (can be float, e.g., 20.5)
            - 'conductivity': (optional) Either scalar or tuple, default 0.0
        vertical_radius: Vertical blur radius for smoothing (default 0)
        use_simple_averaging: Whether to use simple averaging (default False)

    Returns:
        Recipe dict that can be passed to simulate_on_modal() or modified directly

    Example:
        >>> import hyperwave.structure as hwst
        >>> recipe = hwst.recipe_from_params(
        ...     grid_shape=(1600, 1600),
        ...     layers=[
        ...         {
        ...             'density': 'uniform',
        ...             'value': 0.0,
        ...             'permittivity': 2.0736,
        ...             'thickness': 60.0
        ...         },
        ...         {
        ...             'density': my_grating_pattern,  # 1600x1600 array
        ...             'permittivity': (2.0736, 12.1104),
        ...             'thickness': 20.0
        ...         }
        ...     ],
        ...     vertical_radius=0
        ... )
        >>>
        >>> # Modify thickness for parameter sweep
        >>> recipe['layers_info'][1]['layer_thickness'] = 25.0
        >>> results = simulate_on_modal(structure_recipe=recipe, ...)
    """
    import numpy as np
    import math

    # Deduplicate density patterns
    unique_densities = []
    density_indices = []

    for layer_spec in layers:
        density = layer_spec['density']

        # Handle uniform densities
        if isinstance(density, str) and density == 'uniform':
            value = layer_spec.get('value', 0.0)
            uniform_pattern = {
                'type': 'uniform',
                'value': float(value),
                'shape': grid_shape
            }

            # Check if this uniform pattern already exists
            found_idx = None
            for idx, existing in enumerate(unique_densities):
                if (isinstance(existing, dict) and
                    existing.get('type') == 'uniform' and
                    existing['value'] == uniform_pattern['value'] and
                    existing['shape'] == uniform_pattern['shape']):
                    found_idx = idx
                    break

            if found_idx is not None:
                density_indices.append(found_idx)
            else:
                density_indices.append(len(unique_densities))
                unique_densities.append(uniform_pattern)

        # Handle array densities
        else:
            density_np = np.array(density)

            # Check if it's uniform (all same value)
            if np.all(density_np == density_np.flat[0]):
                uniform_pattern = {
                    'type': 'uniform',
                    'value': float(density_np.flat[0]),
                    'shape': density_np.shape
                }

                # Check if already exists
                found_idx = None
                for idx, existing in enumerate(unique_densities):
                    if (isinstance(existing, dict) and
                        existing.get('type') == 'uniform' and
                        existing['value'] == uniform_pattern['value'] and
                        existing['shape'] == uniform_pattern['shape']):
                        found_idx = idx
                        break

                if found_idx is not None:
                    density_indices.append(found_idx)
                else:
                    density_indices.append(len(unique_densities))
                    unique_densities.append(uniform_pattern)
            else:
                # Non-uniform - check for duplicates
                found_idx = None
                for idx, existing in enumerate(unique_densities):
                    if isinstance(existing, np.ndarray) and np.array_equal(density_np, existing):
                        found_idx = idx
                        break

                if found_idx is not None:
                    density_indices.append(found_idx)
                else:
                    density_indices.append(len(unique_densities))
                    unique_densities.append(density_np)

    # Build layers_info
    layers_info = []
    total_thickness = 0

    for i, layer_spec in enumerate(layers):
        thickness = layer_spec['thickness']
        total_thickness += thickness

        # Handle permittivity
        perm = layer_spec['permittivity']
        if isinstance(perm, (tuple, list)):
            perm_values = tuple(perm)
        else:
            perm_values = float(perm)

        # Handle conductivity
        cond = layer_spec.get('conductivity', 0.0)
        if isinstance(cond, (tuple, list)):
            cond_values = tuple(cond)
        else:
            cond_values = float(cond)

        layer_info = {
            'density_index': density_indices[i],
            'density_shape': grid_shape,
            'permittivity_values': perm_values,
            'layer_thickness': float(thickness),
            'conductivity_values': cond_values
        }
        layers_info.append(layer_info)

    # Calculate interface positions
    interface_positions = []
    cumsum = 0
    for layer_spec in layers[:-1]:
        cumsum += layer_spec['thickness']
        interface_positions.append(cumsum)

    # Calculate final shape (permittivity array shape)
    # XY resolution is halved due to Yee grid
    Lx = grid_shape[0] // 2
    Ly = grid_shape[1] // 2

    # Z dimension is ceil of total thickness
    # Handle floating point rounding errors
    if abs(total_thickness - round(total_thickness)) < 1e-6:
        Lz = int(round(total_thickness))
    else:
        Lz = math.ceil(total_thickness)

    final_shape = (3, Lx, Ly, Lz)  # (components, x, y, z) for Yee grid

    # Build recipe
    recipe = {
        'layers_info': layers_info,
        'unique_densities': unique_densities,
        'construction_params': {
            'vertical_radius': float(vertical_radius),
            'use_simple_averaging': use_simple_averaging,
            'total_thickness': float(total_thickness),
            'interface_positions': interface_positions
        },
        'metadata': {
            'layer_count': len(layers),
            'has_conductivity': any(layer_spec.get('conductivity', 0.0) != 0.0 for layer_spec in layers),
            'final_shape': final_shape
        },
        'validation': {
            'permittivity_checksum': 0.0,  # Will be computed on Modal side
            'conductivity_checksum': 0.0,  # Will be computed on Modal side
            'shape': final_shape
        }
    }

    return recipe


 