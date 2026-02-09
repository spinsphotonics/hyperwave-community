"""Hyperwave API Client for GPU-accelerated FDTD photonics simulations.

WORKFLOW:

CPU Steps (free, require valid API key):
    - build_recipe() - Create structure from GDSFactory component
    - build_monitors() - Create monitors from port information
    - compute_freq_band() - Convert wavelengths to frequencies
    - solve_mode_source() - Solve for waveguide mode

GPU Step (uses credits):
    - run_simulation() - Run FDTD simulation on GPU

Analysis (free, runs locally):
    - analyze_transmission() - Analyze transmission from results
    - get_field_intensity_2d() - Extract 2D field intensity for visualization

Utility Functions:
    - configure_api() - Set API credentials and endpoint
    - get_account_info() - Get account info and credit balance
    - estimate_cost() - Estimate simulation cost before running

Environment Variables:
    HYPERWAVE_API_KEY: API authentication key
    HYPERWAVE_API_URL: API endpoint URL (optional, defaults to production)
"""

import os
import base64
import io
import json
import time
from typing import Dict, Any, Tuple, Optional, List, Callable

import numpy as np
import requests
from dataclasses import dataclass, field


def _to_json_serializable(obj):
    """Recursively convert numpy/JAX arrays to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, 'tolist'):  # numpy/JAX arrays
        return obj.tolist()
    if hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    # Fallback: try to convert to float, then string
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


# =============================================================================
# CONVERGENCE CONFIGURATION
# =============================================================================

@dataclass
class ConvergenceConfig:
    """Configuration for early stopping convergence behavior.

    Use this for fine-grained control over when simulations stop.
    For most users, the string presets ("quick", "default", "thorough", "full")
    are recommended instead.

    Attributes:
        check_every_n: Steps between convergence checks (default: 1000).
        relative_threshold: Relative power change threshold (default: 0.01 = 1%).
        min_stable_checks: Consecutive stable checks required (default: 3).
        min_steps: Minimum steps before checking convergence (default: 0).
        power_threshold: Ignore ports with power below this (default: 1e-6).
        monitors: List of specific monitor names to check. If None, checks all output ports.

    Example:
        >>> config = hwc.ConvergenceConfig(
        ...     check_every_n=500,
        ...     min_stable_checks=5,
        ...     min_steps=3000,
        ... )
        >>> results = hwc.run_simulation(..., convergence=config)
    """
    check_every_n: int = 1000
    relative_threshold: float = 0.01
    min_stable_checks: int = 3
    min_steps: int = 0
    power_threshold: float = 1e-6
    monitors: Optional[List[str]] = None


# Preset convergence configurations
# All presets use 1% relative threshold - differences are in check frequency and stability requirements
CONVERGENCE_PRESETS = {
    "quick": ConvergenceConfig(
        check_every_n=2000,
        relative_threshold=0.01,
        min_stable_checks=2,
        min_steps=0,
        power_threshold=1e-5,
    ),
    "default": ConvergenceConfig(
        check_every_n=1000,
        relative_threshold=0.01,
        min_stable_checks=3,
        min_steps=0,
        power_threshold=1e-6,
    ),
    "thorough": ConvergenceConfig(
        check_every_n=1000,
        relative_threshold=0.01,
        min_stable_checks=5,
        min_steps=5000,
        power_threshold=1e-7,
    ),
    "full": None,  # No early stopping - runs all steps
}


def _resolve_convergence(convergence) -> Optional[ConvergenceConfig]:
    """Resolve convergence parameter to a ConvergenceConfig or None.

    Args:
        convergence: Can be:
            - str: Preset name ("quick", "default", "thorough", "full")
            - ConvergenceConfig: Custom configuration
            - bool: True -> "default", False -> "full" (legacy support)
            - None: Uses "default"

    Returns:
        ConvergenceConfig or None (for "full"/no early stopping)
    """
    if convergence is None:
        return CONVERGENCE_PRESETS["default"]

    if isinstance(convergence, bool):
        # Legacy support: True -> default early stopping, False -> no early stopping
        return CONVERGENCE_PRESETS["default"] if convergence else None

    if isinstance(convergence, str):
        if convergence not in CONVERGENCE_PRESETS:
            valid = list(CONVERGENCE_PRESETS.keys())
            raise ValueError(f"Unknown convergence preset '{convergence}'. Valid options: {valid}")
        return CONVERGENCE_PRESETS[convergence]

    if isinstance(convergence, ConvergenceConfig):
        return convergence

    raise TypeError(f"convergence must be str, bool, ConvergenceConfig, or None. Got {type(convergence)}")


# =============================================================================
# COMPONENT PREVIEW (LOCAL - requires gdsfactory)
# =============================================================================

# List of supported GDSFactory components with their default parameters
SUPPORTED_COMPONENTS = {
    # Basic waveguides
    "straight": {"length": 20.0, "width": 0.5},
    "bend_euler": {"radius": 10.0, "angle": 90, "p": 0.5},
    "bend_s": {"size": (10.0, 5.0), "npoints": 99},
    "taper": {"length": 15.0, "width1": 0.5, "width2": 3.0},

    # Directional couplers
    "coupler": {"gap": 0.25, "length": 15.0, "dx": 8.0, "dy": 4.0},
    "coupler_symmetric": {"gap": 0.234, "dy": 4.0, "dx": 10.0},
    "coupler_ring": {"gap": 0.2, "radius": 10.0, "length_x": 4.0, "length_extension": 3.0},

    # MMI splitters
    "mmi1x2": {"width_mmi": 6.0, "length_mmi": 5.5, "gap_mmi": 0.25, "width_taper": 1.5, "length_taper": 10.0},
    "mmi2x2": {"width_mmi": 6.0, "length_mmi": 5.5, "gap_mmi": 0.25, "width_taper": 1.5, "length_taper": 10.0},
    "mmi2x2_with_sbend": {},  # No parameters

    # Ring resonators
    "ring_single": {"gap": 0.2, "radius": 10.0, "length_x": 4.0, "length_y": 0.6},
    "ring_double": {"gap": 0.2, "radius": 10.0, "length_x": 4.0, "length_y": 0.6},

    # Interferometers
    "mzi": {"delta_length": 10.0, "length_y": 2.0, "length_x": 0.1},

    # Crossings
    "crossing": {},
    "crossing45": {},

    # Grating couplers
    "grating_coupler_elliptical": {"polarization": "te", "taper_length": 16.0, "taper_angle": 40.0,
                                    "wavelength": 1.55, "fiber_angle": 15.0, "grating_line_width": 0.343},
    "grating_coupler_rectangular": {"n_periods": 20, "period": 0.63, "fill_factor": 0.5,
                                     "width_grating": 10.0, "length_taper": 150.0},

    # Spirals
    "spiral_inner_io": {"N": 6, "x_straight_inner_right": 150.0, "x_straight_inner_left": 50.0},

    # Bragg gratings
    "dbr": {"w1": 0.5, "w2": 0.6, "l1": 0.2, "l2": 0.2, "n": 10},
    "dbr_tapered": {"length": 10.0, "period": 0.3, "dc": 0.5, "w1": 0.4, "w2": 1.0, "taper_length": 20.0},
}


def list_components() -> List[str]:
    """List all supported GDSFactory component names.

    Returns:
        List of component name strings that can be used with preview_component()
        and build_recipe().

    Example:
        >>> hwc.list_components()
        ['straight', 'bend_euler', 'mmi2x2', ...]
    """
    return list(SUPPORTED_COMPONENTS.keys())


def get_component_params(component_name: str) -> Dict[str, Any]:
    """Get default parameters for a GDSFactory component.

    Args:
        component_name: Name of the component (e.g., "mmi2x2")

    Returns:
        Dictionary of parameter names and their default values.
        Empty dict means the component takes no parameters.

    Example:
        >>> hwc.get_component_params("mmi2x2")
        {'width_mmi': 6.0, 'length_mmi': 5.5, 'gap_mmi': 0.25, ...}

        >>> hwc.get_component_params("mmi2x2_with_sbend")
        {}  # No parameters
    """
    if component_name not in SUPPORTED_COMPONENTS:
        available = list(SUPPORTED_COMPONENTS.keys())
        raise ValueError(f"Unknown component '{component_name}'. Available: {available}")
    return SUPPORTED_COMPONENTS[component_name].copy()


def preview_component(
    component_name: str,
    component_kwargs: Optional[Dict[str, Any]] = None,
    extension_length: float = 2.0,
    show_plot: bool = True,
) -> Dict[str, Any]:
    """Preview a GDSFactory component locally before building.

    This function loads a GDSFactory component locally to visualize it
    and inspect its ports before calling build_recipe(). Requires gdsfactory
    to be installed locally.

    Args:
        component_name: Name of the gdsfactory component (e.g., "mmi2x2").
        component_kwargs: Optional dict of parameters to customize the component.
            Use get_component_params() to see available parameters.
        extension_length: Length to extend ports in um (default: 2.0).
        show_plot: If True, display the component plot (default: True).

    Returns:
        Dictionary containing:
            - component: The gdsfactory Component object
            - name: Component name
            - ports: List of port info dicts with name, center, orientation, width
            - bounds: Component bounding box (xmin, ymin, xmax, ymax)
            - size_um: Component size in um (width, height)
            - params: Parameters used (defaults merged with provided kwargs)

    Example:
        >>> # Preview with default parameters
        >>> info = hwc.preview_component("mmi2x2")

        >>> # Preview with custom parameters
        >>> info = hwc.preview_component("mmi2x2", {"width_mmi": 8.0, "length_mmi": 7.0})

        >>> # Check available parameters first
        >>> params = hwc.get_component_params("mmi2x2")
        >>> params["width_mmi"] = 8.0  # Modify
        >>> info = hwc.preview_component("mmi2x2", params)
    """
    try:
        import gdsfactory as gf
    except ImportError:
        raise ImportError(
            "gdsfactory is required for preview_component(). "
            "Install with: pip install gdsfactory"
        )

    # Activate PDK
    try:
        gf.CONF.pdk = "generic"
    except Exception:
        pass  # PDK might already be active

    # Get default params and merge with provided kwargs
    if component_name in SUPPORTED_COMPONENTS:
        params = SUPPORTED_COMPONENTS[component_name].copy()
    else:
        params = {}

    if component_kwargs:
        params.update(component_kwargs)

    # Get the component function
    if not hasattr(gf.components, component_name):
        available = [c for c in dir(gf.components) if not c.startswith('_')]
        raise ValueError(f"Unknown component '{component_name}'. Check gf.components for available options.")

    component_func = getattr(gf.components, component_name)

    # Create the component
    try:
        if params:
            component = component_func(**params)
        else:
            component = component_func()
    except TypeError as e:
        # If params don't match, show helpful error
        import inspect
        sig = inspect.signature(component_func)
        valid_params = list(sig.parameters.keys())
        raise TypeError(f"Invalid parameters for {component_name}. Valid params: {valid_params}. Error: {e}")

    # Extend ports
    component = gf.c.extend_ports(component, length=extension_length)

    # Extract port info
    ports_info = []
    for port in component.ports:
        ports_info.append({
            'name': port.name,
            'center': tuple(port.center),
            'orientation': port.orientation,
            'width': port.width,
        })

    # Get bounds (bbox() returns DBox with left/bottom/right/top attributes)
    bounds = component.bbox()
    xmin, ymin = bounds.left, bounds.bottom
    xmax, ymax = bounds.right, bounds.top

    # Plot if requested
    if show_plot:
        component.plot()
        print(f"\nComponent: {component.name}")
        print(f"Size: {xmax - xmin:.2f} x {ymax - ymin:.2f} um")
        print(f"Ports ({len(ports_info)}):")
        for p in ports_info:
            print(f"  {p['name']}: center=({p['center'][0]:.2f}, {p['center'][1]:.2f}), "
                  f"orientation={p['orientation']}°, width={p['width']:.3f}um")
        if params:
            print(f"\nParameters used:")
            for k, v in params.items():
                print(f"  {k}: {v}")

    return {
        'component': component,
        'name': component.name,
        'ports': ports_info,
        'bounds': (xmin, ymin, xmax, ymax),
        'size_um': (xmax - xmin, ymax - ymin),
        'params': params,
    }


def load_component(
    component_name: str,
    component_kwargs: Optional[Dict[str, Any]] = None,
    extension_length: float = 2.0,
    resolution_nm: float = 20.0,
    show_plot: bool = True,
) -> Dict[str, Any]:
    """Load a GDSFactory component and convert to theta (design pattern).

    This is the first step of a two-step workflow:
    1. load_component() → theta (this function)
    2. build_recipe_from_theta() → recipe

    Args:
        component_name: Name of the gdsfactory component (e.g., "mmi2x2", "coupler").
        component_kwargs: Optional dict of parameters to customize the component.
            Use get_component_params() to see available parameters.
        extension_length: Length to extend ports in um (default: 2.0).
        resolution_nm: Grid resolution in nanometers (default: 20.0).
        show_plot: If True, display the theta pattern (default: True).

    Returns:
        Dictionary containing:
            - theta: 2D JAX array of the design pattern
            - device_info: Metadata about the component
            - ports: Port information (name, center, orientation, width)
            - component: The gdsfactory Component object
            - resolution_nm: Resolution used
            - resolution_um: Resolution in micrometers

    Example:
        >>> # Load component with default parameters
        >>> theta_result = hwc.load_component("mmi2x2", resolution_nm=20)
        >>> plt.imshow(theta_result['theta'])

        >>> # Load with custom parameters
        >>> theta_result = hwc.load_component(
        ...     "mmi2x2",
        ...     component_kwargs={"width_mmi": 8.0, "length_mmi": 10.0},
        ...     resolution_nm=20,
        ... )

        >>> # Then build recipe from theta
        >>> recipe_result = hwc.build_recipe_from_theta(
        ...     theta_result=theta_result,
        ...     n_core=3.48,
        ...     n_clad=1.45,
        ... )
    """
    try:
        import gdsfactory as gf
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "gdsfactory is required for load_component(). "
            "Install with: pip install gdsfactory"
        )

    # Import component_to_theta from data_io
    from .data_io import component_to_theta

    # Activate PDK
    try:
        gf.CONF.pdk = "generic"
    except Exception:
        pass
    try:
        gf.gpdk.PDK.activate()
    except Exception:
        pass

    # Get the component function
    if not hasattr(gf.components, component_name):
        available = [name for name in dir(gf.components) if not name.startswith('_')]
        raise ValueError(
            f"Component '{component_name}' not found. "
            f"Available components include: {available[:20]}..."
        )

    component_func = getattr(gf.components, component_name)

    # Build kwargs
    kwargs = component_kwargs or {}

    # Create component with port extensions
    # Use gf.c.extend_ports() to match Modal API behavior
    # This adds actual waveguide segments to the ports (not just empty padding)
    try:
        base_component = component_func(**kwargs)
        if extension_length > 0:
            component = gf.c.extend_ports(base_component, length=extension_length)
        else:
            component = base_component
    except Exception as e:
        # Fallback: just use base component without extension
        print(f"Warning: Could not extend ports: {e}")
        base_component = component_func(**kwargs)
        component = base_component

    # Convert to theta (use extended component for geometry)
    resolution_um = resolution_nm / 1000.0
    theta, device_info = component_to_theta(
        component=component,
        resolution=resolution_um,
    )

    # Get port info (use ORIGINAL component's ports for positions, like Modal API)
    # The Modal uses gf_device.ports (original), not gf_extended.ports
    ports_info = []
    for port in base_component.ports:
        ports_info.append({
            'name': port.name,
            'center': tuple(port.center),
            'orientation': port.orientation,
            'width': port.width,
        })

    # Convert ports to the format expected by build_monitors
    # Port positions need to be offset by bounding box min and converted to STRUCTURE cells
    # Use ORIGINAL component's ports but EXTENDED component's bounding box
    bbox = device_info.get('bounding_box_um', (0, 0, 0, 0))
    xmin_um, ymin_um = bbox[0], bbox[1]

    # Get theta resolution for cell conversion (theta is 2x finer than structure)
    theta_res = device_info.get('theta_resolution_um', resolution_um / 2)

    port_info_dict = {}
    # Use base_component.ports (original) for positions, matching Modal behavior
    # Modal formula: x_struct = int((px_um - x_min) / theta_res / 2)
    # Note: NO padding offset here - that's added in build_recipe_from_theta
    for port in base_component.ports:
        # Get port position in um
        port_x_um, port_y_um = port.center

        # Convert to structure cells using Modal's formula:
        # x_struct = int((px_um - x_min) / theta_res / 2)
        # where theta_res is 0.01 um (10nm), so theta_res / 2 = 0.005 um
        # But we want structure cells, so we divide by resolution_um (0.02 um)
        x_struct = int((port_x_um - xmin_um) / theta_res / 2)
        y_struct = int((port_y_um - ymin_um) / theta_res / 2)

        # is_input: True if port faces inward (orientation ~180 degrees)
        is_input = abs(port.orientation % 360 - 180) < 1

        port_info_dict[port.name] = {
            'x_struct': x_struct,
            'y_struct': y_struct,
            'orientation': port.orientation,
            'is_input': is_input,
            'center_um': (port_x_um, port_y_um),
            'width_um': port.width,
        }

    # Plot if requested
    # Note: theta is sampled at 2x finer resolution than structure
    theta_resolution_nm = resolution_nm / 2
    theta_resolution_um = resolution_um / 2

    if show_plot:
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(theta.T, cmap='gray', origin='lower')
        ax.set_title(f"Theta: {component_name}")
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
        plt.colorbar(im, ax=ax, label="theta")
        plt.tight_layout()
        plt.show()

        print(f"\nComponent: {component_name}")
        print(f"Theta shape: {theta.shape}")
        print(f"Theta resolution: {theta_resolution_nm}nm (structure will be {resolution_nm}nm)")
        print(f"Physical size: {theta.shape[0] * theta_resolution_um:.2f} x {theta.shape[1] * theta_resolution_um:.2f} um")
        print(f"Ports ({len(ports_info)}):")
        for p in ports_info:
            print(f"  {p['name']}: center=({p['center'][0]:.2f}, {p['center'][1]:.2f}), "
                  f"orientation={p['orientation']}°, width={p['width']:.3f}um")

    return {
        'theta': theta,
        'device_info': device_info,
        'ports': ports_info,
        'port_info': port_info_dict,
        'component': component,
        'component_name': component_name,
        'resolution_nm': resolution_nm,
        'resolution_um': resolution_um,
        'extension_length': extension_length,
        'component_kwargs': kwargs,
    }


def build_recipe_from_theta(
    theta_result: Dict[str, Any],
    n_core: float = 3.48,
    n_clad: float = 1.45,
    wg_height_um: float = 0.22,
    total_height_um: float = 4.0,
    padding: Tuple[int, int, int, int] = (100, 100, 0, 0),
    density_radius: int = 3,
    vertical_radius: float = 2.0,
    show_structure: bool = True,
) -> Dict[str, Any]:
    """Build structure recipe from theta (design pattern).

    This is the second step of a two-step workflow:
    1. load_component() → theta
    2. build_recipe_from_theta() → recipe (this function)

    This function runs entirely locally (no API call) and does NOT consume credits.

    Args:
        theta_result: Output from load_component() containing theta and metadata.
        n_core: Core refractive index (default: 3.48 for Silicon).
        n_clad: Cladding refractive index (default: 1.45 for SiO2).
        wg_height_um: Waveguide height in um (default: 0.22).
        total_height_um: Total structure height in um (default: 4.0).
        padding: (left, right, top, bottom) padding in cells (default: (100, 100, 0, 0)).
        density_radius: Radius for density filtering (default: 3).
        vertical_radius: Vertical blur radius (default: 2.0).
        show_structure: If True, show structure visualization (default: True).

    Returns:
        Dictionary containing:
            - recipe: Structure recipe dict for simulation
            - density_core: Core layer density pattern
            - density_clad: Cladding layer density pattern
            - dimensions: (Lx, Ly, Lz) structure dimensions
            - port_info: Port information for monitors
            - layer_config: Layer configuration
            - eps_values: (eps_clad, eps_core) tuple
            - resolution_um: Resolution in micrometers
            - structure: The Structure object (for visualization)

    Example:
        >>> # First load the component
        >>> theta_result = hwc.load_component("mmi2x2", resolution_nm=20)

        >>> # Then build the recipe
        >>> recipe_result = hwc.build_recipe_from_theta(
        ...     theta_result=theta_result,
        ...     n_core=3.48,
        ...     n_clad=1.45,
        ...     wg_height_um=0.22,
        ... )
        >>> print(f"Structure dimensions: {recipe_result['dimensions']}")
    """
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    # Import local structure functions
    from .structure import density, Layer, create_structure

    # Extract from theta_result
    theta = theta_result['theta']
    resolution_um = theta_result['resolution_um']
    port_info = theta_result.get('port_info', {})
    component_name = theta_result.get('component_name', 'unknown')

    print(f"Building recipe from theta...")

    # Calculate permittivities
    eps_core = n_core ** 2
    eps_clad = n_clad ** 2

    # Create density patterns with padding
    density_core = density(
        theta=theta,
        pad_width=padding,
        radius=density_radius,
    )

    density_clad = density(
        theta=jnp.zeros_like(theta),
        pad_width=padding,
        radius=density_radius,
    )

    # Calculate layer thicknesses in cells
    wg_thickness_cells = max(1, int(round(wg_height_um / resolution_um)))
    slab_height_um = (total_height_um - wg_height_um) / 2
    slab_thickness_cells = max(1, int(round(slab_height_um / resolution_um)))

    # Create layers
    core_layer = Layer(
        density_pattern=density_core,
        permittivity_values=(eps_clad, eps_core),
        layer_thickness=wg_thickness_cells,
    )

    clad_layer = Layer(
        density_pattern=density_clad,
        permittivity_values=eps_clad,
        layer_thickness=slab_thickness_cells,
    )

    # Build structure: clad / core / clad
    structure = create_structure(
        layers=[clad_layer, core_layer, clad_layer],
        vertical_radius=vertical_radius,
    )

    # Get dimensions
    _, Lx, Ly, Lz = structure.permittivity.shape
    dimensions = (Lx, Ly, Lz)

    # Adjust port positions for padding
    # Note: In density(), padding = (left, right, top, bottom) where:
    #   - left/right (padding[0]/[1]) pad the Y axis (axis 1)
    #   - top/bottom (padding[2]/[3]) pad the X axis (axis 0)
    adjusted_port_info = {}
    for port_name, port_data in port_info.items():
        # Padding is in theta cells (2x finer), divide by 2 for structure cells
        x_pad_struct = padding[2] // 2
        y_pad_struct = padding[0] // 2
        adjusted_port_info[port_name] = {
            'x_struct': port_data['x_struct'] + x_pad_struct,
            'y_struct': port_data['y_struct'] + y_pad_struct,
            'orientation': port_data['orientation'],
            'is_input': port_data['is_input'],
            'center_um': port_data['center_um'],
            'width_um': port_data['width_um'],
        }

    # Create layer config (match Modal API key names exactly)
    layer_config = {
        'clad_bot_cells': slab_thickness_cells,
        'wg_height_cells': wg_thickness_cells,
        'clad_top_cells': slab_thickness_cells,  # Symmetric cladding
        'vertical_radius': vertical_radius,
    }

    # Extract recipe
    recipe = structure.extract_recipe()

    # Show structure if requested
    if show_structure:
        # View XY slice at waveguide center
        z_center = slab_thickness_cells + wg_thickness_cells // 2
        structure.view(
            show_permittivity=True,
            show_conductivity=False,
            axis="z",
            position=z_center,
            cmap_permittivity="viridis",
        )

    print(f"Recipe built: {Lx}x{Ly}x{Lz} cells")
    print(f"Ports: {list(adjusted_port_info.keys())}")

    return {
        'recipe': recipe,
        'density_core': density_core,
        'density_clad': density_clad,
        'dimensions': dimensions,
        'port_info': adjusted_port_info,
        'layer_config': layer_config,
        'eps_values': (eps_clad, eps_core),
        'resolution_um': resolution_um,
        'padding': padding,
        'structure': structure,
        'device_info': theta_result.get('device_info'),
    }


def build_monitors_local(
    recipe_result: Dict[str, Any],
    source_port: str = "o1",
    monitor_margin_um: float = 1.5,
    source_offset_cells: int = 5,
    show_monitors: bool = True,
) -> Dict[str, Any]:
    """Build monitors locally from recipe_result (no API call).

    This function works with the output of build_recipe_from_theta() to create
    monitors locally using the Structure object.

    Args:
        recipe_result: Output from build_recipe_from_theta() containing structure.
        source_port: Name of port to use as source (default: "o1").
        monitor_margin_um: Margin around waveguide for monitor size (default: 1.5).
        source_offset_cells: Offset of source from port in cells (default: 5).
        show_monitors: If True, show structure with monitor positions (default: True).

    Returns:
        Dictionary containing:
            - monitors: MonitorSet object
            - monitors_recipe: Serialized monitor recipe for simulation
            - source_position: X position of source in cells
            - source_port_name: Name of source port
            - monitor_names: Dict mapping monitor names to indices
            - mode_bounds: Bounds for mode solving

    Example:
        >>> theta_result = hwc.load_component("mmi2x2", resolution_nm=20)
        >>> recipe_result = hwc.build_recipe_from_theta(theta_result)
        >>> monitor_result = hwc.build_monitors_local(recipe_result, source_port="o1")
    """
    from .monitors import MonitorSet, view_monitors

    structure = recipe_result.get('structure')
    if structure is None:
        raise ValueError("recipe_result must contain 'structure' from build_recipe_from_theta()")

    port_info = recipe_result['port_info']
    dimensions = recipe_result['dimensions']
    resolution_um = recipe_result['resolution_um']
    layer_config = recipe_result['layer_config']

    Lx, Ly, Lz = dimensions

    # Get source port info
    if source_port not in port_info:
        available = list(port_info.keys())
        raise ValueError(f"Source port '{source_port}' not found. Available: {available}")

    source_port_data = port_info[source_port]
    source_x = source_port_data['x_struct']

    # Determine if source is on left or right side
    is_left_port = source_x < Lx // 2
    source_offset = source_offset_cells if is_left_port else -source_offset_cells
    source_position = source_x + source_offset

    # Create monitors
    monitors = MonitorSet()

    # Calculate monitor size in cells
    monitor_margin_cells = int(monitor_margin_um / resolution_um)
    clad_bot_cells = layer_config['clad_bot_cells']
    wg_height_cells = layer_config['wg_height_cells']
    z_center = clad_bot_cells + wg_height_cells // 2

    # Add input monitor at source port
    input_label = f"Input_{source_port}"
    monitors.add_monitors_at_position(
        structure=structure,
        axis='x',
        position=source_position,
        label=input_label,
    )

    # Add output monitors at other ports
    output_ports = [p for p in port_info.keys() if p != source_port]
    for port_name in output_ports:
        port_data = port_info[port_name]
        port_x = port_data['x_struct']

        # Offset away from port edge
        is_port_left = port_x < Lx // 2
        port_offset = source_offset_cells if is_port_left else -source_offset_cells
        monitor_x = port_x + port_offset

        output_label = f"Output_{port_name}"
        monitors.add_monitors_at_position(
            structure=structure,
            axis='x',
            position=monitor_x,
            label=output_label,
        )

    # Show monitors if requested
    if show_monitors:
        view_monitors(structure, monitors)

    # Get monitor names from the MonitorSet mapping
    monitor_names = monitors.mapping.copy()

    # Get mode bounds (y bounds around source port for mode solving)
    source_y = source_port_data['y_struct']
    mode_half_width = int(1.5 / resolution_um)  # 1.5 um half-width
    mode_bounds = (
        max(0, source_y - mode_half_width),
        min(Ly, source_y + mode_half_width),
    )

    print(f"Monitors created: {list(monitor_names.keys())}")
    print(f"Source port: {source_port} at x={source_position}")

    return {
        'monitors': monitors,
        'monitors_recipe': monitors.recipe,
        'source_position': source_position,
        'source_port_name': source_port,
        'monitor_names': monitor_names,
        'mode_bounds': mode_bounds,
        'layer_config': layer_config,
    }


# Global API configuration
_API_CONFIG = {
    'api_key': None,
    'api_url': 'https://spinsphotonics--hyperwave-api-fastapi-app.modal.run'
}


def configure_api(api_key: Optional[str] = None, api_url: Optional[str] = None, validate: bool = True) -> Optional[Dict[str, Any]]:
    """Configure API credentials and endpoint, with optional validation.

    Args:
        api_key: API authentication key. If None, uses HYPERWAVE_API_KEY environment variable.
        api_url: API endpoint URL. If None, uses HYPERWAVE_API_URL environment variable
            or defaults to production endpoint.
        validate: If True (default), validates the API key by calling the server.

    Returns:
        Account info dict with 'name', 'email', 'credits_balance' if validation enabled,
        None otherwise.

    Raises:
        ValueError: If no API key is provided and HYPERWAVE_API_KEY is not set.
        RuntimeError: If API key validation fails.

    Example:
        >>> import hyperwave_community as hwc
        >>> account = hwc.configure_api(api_key='your-key-here')
        >>> print(f"Welcome {account['name']}! Credits: {account['credits_balance']}")
    """
    global _API_CONFIG

    if api_key is not None:
        _API_CONFIG['api_key'] = api_key
    elif 'HYPERWAVE_API_KEY' in os.environ:
        _API_CONFIG['api_key'] = os.environ['HYPERWAVE_API_KEY']

    if api_url is not None:
        _API_CONFIG['api_url'] = api_url
    elif 'HYPERWAVE_API_URL' in os.environ:
        _API_CONFIG['api_url'] = os.environ['HYPERWAVE_API_URL']

    if _API_CONFIG['api_key'] is None:
        raise ValueError(
            "API key not provided. Set HYPERWAVE_API_KEY environment variable "
            "or call configure_api(api_key='your-key') first."
        )

    # Validate API key if requested
    if validate:
        try:
            response = requests.post(
                f"{_API_CONFIG['api_url']}/account_info",
                params={"api_key": _API_CONFIG['api_key']},
                timeout=60  # Modal cold start can take time
            )
            if response.status_code == 403:
                raise RuntimeError("Invalid API key. Please check your API key and try again.")
            response.raise_for_status()
            account_info = response.json()
            print(f"✓ API key validated for: {account_info.get('name', 'Unknown')}")
            print(f"  Email: {account_info.get('email', 'N/A')}")
            print(f"  Credits: {account_info.get('credits_balance', 0):.2f}")
            return account_info
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to validate API key: {e}")

    return None


def _get_api_config() -> Dict[str, str]:
    """Get current API configuration."""
    if _API_CONFIG['api_key'] is None:
        if 'HYPERWAVE_API_KEY' in os.environ:
            _API_CONFIG['api_key'] = os.environ['HYPERWAVE_API_KEY']
        else:
            raise RuntimeError(
                "API not configured. Call configure_api() or set HYPERWAVE_API_KEY "
                "environment variable first."
            )
    return _API_CONFIG


def encode_array(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string for API transmission."""
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_array(b64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    buffer = io.BytesIO(base64.b64decode(b64_str))
    return np.load(buffer)


def _handle_api_error(e: requests.exceptions.HTTPError, operation: str) -> None:
    """Handle common API errors with user-friendly messages."""
    if e.response is not None:
        status_code = e.response.status_code
        if status_code == 401:
            print("No API key detected in request.")
            print("Sign up for free at spinsphotonics.com to get your API key.")
        elif status_code == 403:
            print("Provided API key is invalid.")
            print("Please verify your API key in your dashboard at spinsphotonics.com/dashboard")
        elif status_code == 402:
            print(f"Insufficient credits for {operation}.")
            print("Add credits to your account at spinsphotonics.com/billing")
        elif status_code == 429:
            print("Too many concurrent simulations.")
        elif status_code == 502:
            print("Service temporarily unavailable. Please retry.")
        elif status_code == 422:
            print(f"Validation error from API:")
            try:
                error_detail = e.response.json()
                print(f"  {error_detail}")
            except Exception:
                print(f"  {e.response.text}")
        else:
            print(f"Unexpected error (Code: {status_code})")
            try:
                print(f"  Response: {e.response.text[:500]}")
            except Exception:
                pass
    else:
        print("Communication error.")


# =============================================================================
# ACCOUNT INFO
# =============================================================================

def get_account_info(quiet: bool = False) -> Optional[Dict[str, Any]]:
    """Verify API key and get account information including credit balance.

    Args:
        quiet: If True, don't print the greeting message.

    Returns:
        Dictionary with valid, name, email, api_key_prefix, credits_balance, credits_balance_usd.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    try:
        response = requests.post(
            f"{API_URL}/account_info",
            params={"api_key": API_KEY},
            timeout=30
        )
        response.raise_for_status()
        info = response.json()

        if not quiet:
            name = info.get('name', 'User')
            email = info.get('email', '')
            prefix = info.get('api_key_prefix', '')
            balance = info.get('credits_balance', 0)

            print()
            print(f"  Welcome back, {name}!")
            print(f"  {email}")
            print()
            print(f"  API Key:  {prefix}...")
            print(f"  Credits:  {balance:.2f}")
            print()

        return info

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            print("Invalid API key.")
            print("Please verify your API key in your dashboard at spinsphotonics.com/dashboard")
        else:
            print(f"Error getting account info: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting account info: {e}")
        return None


# =============================================================================
# COST ESTIMATION
# =============================================================================

def estimate_cost(
    grid_points: Optional[int] = None,
    structure_shape: Optional[Tuple[int, int, int, int]] = None,
    max_steps: int = 10000,
    gpu_type: str = "H100",
    simulation_type: str = "fdtd_simulation",
) -> Optional[Dict[str, Any]]:
    """Estimate simulation cost before running (no auth required)."""
    API_URL = _API_CONFIG['api_url']

    request_data = {
        "max_steps": max_steps,
        "gpu_type": gpu_type,
        "simulation_type": simulation_type
    }

    if grid_points is not None:
        request_data["grid_points"] = grid_points
    elif structure_shape is not None:
        request_data["structure_shape"] = list(structure_shape)
    else:
        print("Either grid_points or structure_shape must be provided.")
        return None

    try:
        response = requests.post(f"{API_URL}/estimate_cost", json=request_data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error estimating cost: {e}")
        return None


# =============================================================================
# TWO-STAGE WORKFLOW (Recommended)
# =============================================================================

def prepare_simulation(
    device_type: str,
    pdk_config: Dict[str, Any],
    source_port: str = "o1",
    wavelength_um: float = 1.55,
    cells_per_wavelength: int = 25,
    mode_num: int = 0,
    device_params: Optional[Dict[str, Any]] = None,
    # build_recipe parameters
    extension_length: float = 2.0,
    total_height_um: float = 4.0,
    padding: Tuple[int, int, int, int] = (100, 100, 0, 0),
    density_radius: int = 3,
    vertical_radius: float = 2.0,
    # build_monitors parameters
    monitor_x_um: float = 0.1,
    monitor_y_um: float = 1.5,
    monitor_z_um: float = 1.5,
    source_offset_cells: int = 5,
    show_structure: bool = False,
    include_field_monitor: bool = True,
    # compute_freq_band parameters
    n_freqs: int = 1,
    # solve_mode_source parameters
    slice_half_width: int = 5,
    propagation_axis: str = "x",
    show_mode: bool = False,
    # get_default_absorber_params parameters
    absorber_fraction: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Stage 1: Prepare simulation inputs on Modal CPU.

    This performs all setup work needed before simulation:
    - Build structure recipe from GDSFactory component
    - Create monitors at port locations
    - Solve for waveguide mode at source port
    - Return all data needed for simulation

    Args:
        device_type: Device type name (e.g., "mmi2x2", "mmi1x2").
        pdk_config: PDK configuration dict with material parameters:
            - n_core: Core refractive index (e.g., 3.48 for silicon)
            - n_clad: Cladding refractive index (e.g., 1.45 for SiO2)
            - wg_height_um: Waveguide height in micrometers (e.g., 0.22)
            - clad_top_um: Top cladding thickness in micrometers (e.g., 1.89)
            - clad_bot_um: Bottom cladding (BOX) thickness in micrometers (e.g., 2.0)
        source_port: Input port name (e.g., "o1", "o2").
        wavelength_um: Wavelength in micrometers (default: 1.55).
        cells_per_wavelength: FDTD resolution (default: 25). Higher = more accurate but slower.
        mode_num: Mode number to solve (0 = fundamental, default: 0).
        device_params: Optional dict of device-specific parameters.

        # Structure recipe parameters (from build_recipe):
        extension_length: Length to extend ports in um (default: 2.0).
        total_height_um: Total structure height in um (default: 4.0).
        padding: (left, right, top, bottom) padding in theta pixels (default: (100,100,0,0)).
        density_radius: Radius for density filtering (default: 3).
        vertical_radius: Vertical blur radius (default: 2.0).

        # Monitor parameters (from build_monitors):
        monitor_x_um: Monitor thickness in x direction (default: 0.1).
        monitor_y_um: Monitor size in y direction (default: 1.5).
        monitor_z_um: Monitor size in z direction (default: 1.5).
        source_offset_cells: Offset of source from monitor in cells (default: 5).
        show_structure: If True, display structure visualization (default: False).
        include_field_monitor: If True, include xy_mid monitor for field vis (default: True).

        # Frequency band parameters (from compute_freq_band):
        n_freqs: Number of frequency points (default: 1).

        # Mode solver parameters (from solve_mode_source):
        slice_half_width: Half-width of mode slice in cells (default: 5).
        propagation_axis: Propagation axis 'x', 'y', or 'z' (default: "x").
        show_mode: If True, display mode profile visualization (default: False).

        # Absorber parameters (from get_default_absorber_params):
        absorber_fraction: Fraction of structure for absorber (default: 0.1).

    Returns:
        Dict with:
        - mode_preview: Mode visualization data (n_eff, Ex, Ey, Ez)
        - setup_data: Pre-computed setup for run_simulation()
        - wavelength_um: Wavelength used
        - resolution_nm: Resolution in nanometers
        - source_port: Actual source port name

    Example:
        >>> pdk_config = {
        ...     "n_core": 3.48,
        ...     "n_clad": 1.45,
        ...     "wg_height_um": 0.22,
        ...     "clad_top_um": 1.89,
        ...     "clad_bot_um": 2.0,
        ... }
        >>> setup = hwc.prepare_simulation(
        ...     device_type="mmi2x2",
        ...     pdk_config=pdk_config,
        ...     source_port="o1",
        ...     wavelength_um=1.55,
        ...     cells_per_wavelength=25,
        ... )
        >>> print(f"Mode n_eff: {setup['mode_preview']['n_eff']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Validate pdk_config
    required_fields = ["n_core", "n_clad", "wg_height_um", "clad_top_um", "clad_bot_um"]
    missing = [f for f in required_fields if f not in pdk_config]
    if missing:
        raise ValueError(f"pdk_config missing required fields: {missing}")

    print(f"Preparing simulation for {device_type}...")
    print(f"  Source: {source_port}, Wavelength: {wavelength_um} um, Resolution: {cells_per_wavelength} cells/wavelength")

    body = {
        "device_type": device_type,
        "device_params": device_params or {},
        "source_port": source_port,
        "wavelength_um": wavelength_um,
        "cells_per_wavelength": cells_per_wavelength,
        "pdk_config": pdk_config,
        "mode_num": mode_num,
        # build_recipe parameters
        "extension_length": extension_length,
        "total_height_um": total_height_um,
        "padding": list(padding),
        "density_radius": density_radius,
        "vertical_radius": vertical_radius,
        # build_monitors parameters
        "monitor_x_um": monitor_x_um,
        "monitor_y_um": monitor_y_um,
        "monitor_z_um": monitor_z_um,
        "source_offset_cells": source_offset_cells,
        "show_structure": show_structure,
        "include_field_monitor": include_field_monitor,
        # compute_freq_band parameters
        "n_freqs": n_freqs,
        # solve_mode_source parameters
        "slice_half_width": slice_half_width,
        "propagation_axis": propagation_axis,
        "show_mode": show_mode,
        # get_default_absorber_params parameters
        "absorber_fraction": absorber_fraction,
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/prepare_simulation",
            json=body,
            headers=headers,
            timeout=600
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") != "success":
            raise Exception(f"Failed to prepare simulation: {result}")

        mode_preview = result.get("mode_preview", {})
        n_eff = mode_preview.get("n_eff", "N/A")
        print(f"Mode solved: n_eff={n_eff}")
        print(f"Setup complete. Ready for run_simulation()")

        return {
            "mode_preview": mode_preview,
            "setup_data": result.get("setup_data"),
            "wavelength_um": result.get("wavelength_um"),
            "resolution_nm": result.get("resolution_nm"),
            "source_port": result.get("source_port"),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "prepare_simulation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error preparing simulation: {e}")
        return None


def run_simulation(
    device_type: str,
    # Option 1: Pass individual granular results (recommended)
    recipe_result: Optional[Dict[str, Any]] = None,
    monitor_result: Optional[Dict[str, Any]] = None,
    freq_result: Optional[Dict[str, Any]] = None,
    source_result: Optional[Dict[str, Any]] = None,
    # Option 2: Pass pre-packaged setup_data (legacy/advanced)
    setup_data: Optional[Dict[str, Any]] = None,
    # Simulation parameters
    num_steps: int = 20000,
    gpu_type: str = "H100",
    convergence: Optional[str] = "default",
    absorption_widths: List[int] = None,
    absorption_coeff: float = 0.0006173770394704579,
    source_ramp_periods: float = 10.0,
    # Analysis parameters (for analyze_transmission)
    input_monitor: Optional[str] = None,
    output_monitors: Optional[List[str]] = None,
    # Field slice parameters (for get_field_intensity)
    field_monitor_name: str = "xy_mid",
) -> Optional[Dict[str, Any]]:
    """Run FDTD simulation on Modal GPU.

    Two ways to call this function:

    **Option 1: Pass granular results directly (recommended)**
    ```python
    results = hwc.run_simulation(
        device_type="mmi2x2",
        recipe_result=recipe_result,
        monitor_result=monitor_result,
        freq_result=freq_result,
        source_result=source_result,
        gpu_type="H100",
    )
    ```

    **Option 2: Pass pre-packaged setup_data (legacy)**
    ```python
    results = hwc.run_simulation(
        device_type="mmi2x2",
        setup_data=setup_data,
        gpu_type="H100",
    )
    ```

    Args:
        device_type: Device type name (for tracking).
        recipe_result: Result from build_recipe().
        monitor_result: Result from build_monitors().
        freq_result: Result from compute_freq_band().
        source_result: Result from solve_mode_source().
        setup_data: Pre-packaged setup (alternative to individual results).
        num_steps: Maximum FDTD steps (default: 20000).
        gpu_type: GPU type - "B200", "H200", "H100", "A100-80GB", etc.
        convergence: Early stopping behavior. Options:
            - "quick": Stop early, check less frequently (fastest)
            - "default": Balanced approach (recommended)
            - "thorough": Check carefully before stopping (most conservative)
            - "full": No early stopping, run all num_steps
            - ConvergenceConfig: Custom configuration object
        absorption_widths: Absorber widths [x, y, z] in cells (default: [82, 40, 40]).
        absorption_coeff: Absorber coefficient.
        source_ramp_periods: Source ramp-up periods (default: 10.0).

        # Analysis parameters (for analyze_transmission):
        input_monitor: Name of input monitor (default: auto-detect from source_port).
        output_monitors: List of output monitor names (default: auto-detect all non-input ports).

        # Field slice parameters (for get_field_intensity):
        field_monitor_name: Name of 2D field monitor (default: "xy_mid").

    Returns:
        Dict with simulation results:
        - sim_time: GPU simulation time in seconds
        - total_time: Total execution time including overhead
        - monitor_data: Decoded monitor field data
        - powers: Power at each monitor
        - converged: Whether simulation converged (False if convergence="full")
        - convergence_step: Step at which convergence was detected
        - performance: Simulation performance (pts*steps/s)
        - s_parameters: S-parameter results (if analyze_transmission succeeds)
        - field_intensity: 2D field intensity data (if get_field_intensity succeeds)
    """
    import time
    start_time = time.time()

    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Build setup_data from individual results if provided
    if recipe_result is not None and monitor_result is not None and source_result is not None:
        # Package granular results into setup_data format
        # Use _to_json_serializable to convert numpy/JAX arrays to JSON-serializable types
        setup_data = {
            'structure_recipe': _to_json_serializable(recipe_result['recipe']),
            'source_field_b64': encode_array(source_result['source_field']),
            'source_field_shape': list(source_result['source_field'].shape),
            'source_offset': list(source_result['source_offset']),
            'freq_band': list(freq_result['freq_band']) if freq_result else [0.081, 0.081, 1],
            'monitors': _to_json_serializable(monitor_result['monitors']),
            'monitor_names': _to_json_serializable(monitor_result['monitor_names']),
            'dimensions': list(recipe_result['dimensions']),
        }
    elif setup_data is None:
        raise ValueError(
            "Either provide individual results (recipe_result, monitor_result, freq_result, source_result) "
            "or a pre-packaged setup_data dict."
        )
    else:
        # Handle legacy formats: full prepare_simulation result or just setup_data
        if "setup_data" in setup_data and "source_field_base64" not in setup_data:
            setup_data = setup_data["setup_data"]

    # Resolve convergence configuration
    conv_config = _resolve_convergence(convergence)
    use_early_stopping = conv_config is not None

    endpoint = "/early_stopping" if use_early_stopping else "/simulate"
    convergence_name = convergence if isinstance(convergence, str) else ("custom" if conv_config else "full")

    print(f"Starting simulation for {device_type}...")
    print(f"  GPU: {gpu_type}, Max steps: {num_steps}")
    print(f"  Convergence: {convergence_name}")

    # Get absorber params from setup_data if available (from prepare_simulation)
    absorber_params = setup_data.get("absorber_params", {})

    # Get structure dimensions for validation
    dimensions = setup_data.get("dimensions", [1000, 500, 200])
    # Calculate maximum safe absorber widths based on dimensions
    # Absorber width must be less than half the dimension
    max_safe_x = max(10, dimensions[0] // 2 - 10) if len(dimensions) > 0 else 82
    max_safe_y = max(10, dimensions[1] // 2 - 10) if len(dimensions) > 1 else 40
    max_safe_z = max(10, dimensions[2] // 2 - 10) if len(dimensions) > 2 else 40

    # Use absorber params from setup_data if user didn't explicitly override
    if absorption_widths is None:
        # Get from setup_data or use defaults
        if absorber_params and "absorption_widths" in absorber_params:
            absorption_widths = list(absorber_params["absorption_widths"])
        else:
            absorption_widths = [82, 40, 40]  # Default values

    # ALWAYS validate absorption_widths against structure dimensions
    # This prevents "z_pad cannot exceed half the z dimension" errors
    absorption_widths = [
        min(absorption_widths[0], max_safe_x),
        min(absorption_widths[1], max_safe_y),
        min(absorption_widths[2], max_safe_z),
    ]

    # Get absorption_coeff from setup_data if available and not explicitly overridden
    if absorber_params and "absorption_coeff" in absorber_params:
        effective_absorption_coeff = absorber_params["absorption_coeff"]
    else:
        effective_absorption_coeff = absorption_coeff

    # Build request body
    body = {
        "structure_recipe": setup_data.get("structure_recipe"),
        "source_field_b64": setup_data.get("source_field_b64"),
        "source_field_shape": setup_data.get("source_field_shape"),
        "source_offset": setup_data.get("source_offset"),
        "freq_band": setup_data.get("freq_band"),
        "monitors": setup_data.get("monitors"),
        "max_steps": num_steps,
        "check_every_n": conv_config.check_every_n if conv_config else 1000,
        "source_ramp_periods": source_ramp_periods,
        "gpu_type": gpu_type,
        "add_absorption": True,
        "absorption_widths": list(absorption_widths),
        "absorption_coeff": float(effective_absorption_coeff),
    }

    # Add early stopping specific parameters
    if use_early_stopping:
        body["relative_threshold"] = conv_config.relative_threshold
        body["absolute_threshold"] = 1e-10  # Fixed value
        body["significant_power_threshold"] = conv_config.power_threshold
        body["min_stable_checks"] = conv_config.min_stable_checks
        # Note: min_steps is handled by the Modal function internally

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        # Call appropriate endpoint
        print(f"Calling {endpoint} API...")
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=body,
            headers=headers,
            timeout=600  # 10 minute timeout for long simulations
        )
        response.raise_for_status()
        result = response.json()

        sim_time = result.get("sim_time", 0)
        total_time = time.time() - start_time
        converged = result.get("converged", False)

        if use_early_stopping:
            convergence_step = result.get("convergence_step", 0)
            print(f"Simulation completed in {sim_time:.1f}s (total: {total_time:.1f}s)")
            if converged:
                print(f"  Converged at step {convergence_step}")
        else:
            print(f"Simulation completed in {sim_time:.1f}s (total: {total_time:.1f}s)")

        # Check if data is stored externally (large responses)
        data_url = result.get("data_url")
        if data_url:
            print(f"Fetching large response from cloud storage...")
            try:
                import gzip
                url_response = requests.get(data_url, timeout=120)
                url_response.raise_for_status()
                # Decompress and parse JSON
                decompressed = gzip.decompress(url_response.content)
                result = json.loads(decompressed.decode('utf-8'))
                print(f"  Downloaded {len(url_response.content) / 1024 / 1024:.2f} MB")
            except Exception as e:
                print(f"  ERROR: Failed to fetch large response: {e}")
                raise RuntimeError(f"Failed to fetch simulation results from cloud storage: {e}")

        # Check if data is gzip compressed
        data_compressed = result.get("data_compressed", False)

        # Decode base64-encoded arrays from API response
        def decode_b64_dict(d, shapes=None, dtypes=None, is_compressed=False):
            """Decode base64 strings in a dict back to numpy arrays."""
            import numpy as np
            import gzip as gzip_module
            decoded = {}
            for k, v in d.items():
                if isinstance(v, str) and v:
                    try:
                        # Decode base64 to bytes
                        arr_bytes = base64.b64decode(v)

                        # Decompress if gzip compressed
                        if is_compressed:
                            try:
                                arr_bytes = gzip_module.decompress(arr_bytes)
                            except Exception:
                                pass  # Might not be compressed, continue

                        # Get dtype from metadata or infer
                        dtype = np.complex64  # Default for monitor data
                        if dtypes and k in dtypes:
                            dtype = np.dtype(dtypes[k])

                        # Reshape if shape is known
                        if shapes and k in shapes:
                            shape = shapes[k]
                            try:
                                arr = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
                            except ValueError:
                                # Fallback to float32 if complex64 doesn't work
                                try:
                                    arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(shape)
                                except ValueError:
                                    arr = np.frombuffer(arr_bytes, dtype=np.float32)
                        else:
                            # For powers/transmissions, assume float32
                            arr = np.frombuffer(arr_bytes, dtype=np.float32)
                        decoded[k] = arr
                    except Exception:
                        decoded[k] = v  # Keep original if decoding fails
                else:
                    decoded[k] = v
            return decoded

        # Get shapes and dtypes for decoding monitor data
        monitor_shapes = result.get("monitor_data_shapes", {})
        monitor_dtypes = result.get("monitor_data_dtypes", {})

        # Decode the base64-encoded data (with compression support)
        monitor_data = decode_b64_dict(
            result.get("monitor_data_b64", {}),
            shapes=monitor_shapes,
            dtypes=monitor_dtypes,
            is_compressed=data_compressed
        )
        powers = decode_b64_dict(result.get("powers", {}), is_compressed=data_compressed)

        # Build simulation results dict for analysis
        sim_results = {
            "sim_time": sim_time,
            "total_time": total_time,
            "converged": converged,
            "convergence_step": result.get("convergence_step", 0) if use_early_stopping else 0,
            "monitor_data": monitor_data,
            "monitor_data_shapes": monitor_shapes,
            "monitor_names": result.get("monitor_names", {}),
            "powers": powers,
            "performance": result.get("performance", 0),
            # Keep raw result for analyze_transmission
            "_raw_result": result,
        }

        # Run analyze_transmission locally (avoids 413 payload size errors)
        print("Analyzing transmission...")
        try:
            # Use local analyze_transmission function instead of API call
            # Pass sim_results (with decoded monitor_data), not raw result
            trans_result = analyze_transmission(
                sim_results,
                input_monitor=input_monitor or "Input_o1",
                output_monitors=output_monitors,
                print_results=False
            )
            sim_results["s_parameters"] = trans_result
            print(f"  Transmission analyzed: {len(trans_result.get('transmissions', {}))} ports")
        except Exception as e:
            print(f"  Warning: Could not analyze transmission: {e}")
            sim_results["s_parameters"] = None

        # Get field intensity for visualization (if specified monitor exists)
        if field_monitor_name in sim_results.get("monitor_names", {}):
            print(f"Extracting field intensity from '{field_monitor_name}'...")
            try:
                # Use local extraction instead of API call (avoids 413 errors)
                # Pass sim_results (with decoded monitor_data), not raw result
                field_result = get_field_intensity_2d(
                    sim_results,
                    monitor_name=field_monitor_name,
                    dimensions=result.get("dimensions"),
                    resolution_um=result.get("resolution_um", 0.02),
                    freq_band=result.get("freq_band"),
                )
                sim_results["field_intensity"] = field_result
                print(f"  Field intensity extracted: {field_result['intensity'].shape}")
            except Exception as e:
                print(f"  Warning: Could not extract field intensity: {e}")
                sim_results["field_intensity"] = None
        else:
            print(f"  Warning: Could not extract field intensity: Monitor '{field_monitor_name}' not found in results")
            sim_results["field_intensity"] = None

        # Remove raw result from returned dict
        del sim_results["_raw_result"]

        return sim_results

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "run_simulation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error running simulation: {e}")
        return None


# =============================================================================
# LOCAL WORKFLOW - Takes local objects (Structure, MonitorSet, etc.)
# =============================================================================

def simulate(
    structure_recipe: Dict[str, Any],
    source_field,
    source_offset: Tuple[int, int, int],
    freq_band: Tuple[float, float, int],
    monitors_recipe: List[Dict],
    mode_info: Optional[Dict] = None,
    simulation_steps: int = 20000,
    check_every_n: int = 1000,
    source_ramp_periods: float = 5.0,
    add_absorption: bool = True,
    absorption_widths: Tuple[int, int, int] = (60, 40, 40),
    absorption_coeff: float = 1e-4,
    api_key: Optional[str] = None,
    gpu_type: str = "H100",
    convergence: Optional[str] = "default",
) -> Optional[Dict[str, Any]]:
    """Run FDTD simulation on cloud GPU using structure recipe and monitors recipe.

    This function is for the LOCAL WORKFLOW where you create the structure,
    monitors, and source field locally using JAX, then send them to the cloud
    for GPU-accelerated FDTD simulation.

    Args:
        structure_recipe: Recipe dict from structure.extract_recipe().
        source_field: Source field array (JAX or numpy) of shape (num_freqs, 6, x, y, z).
        source_offset: (x, y, z) position offset for source placement.
        freq_band: Frequency band tuple (omega_min, omega_max, num_freqs).
        monitors_recipe: Recipe list from monitors.recipe property.
        mode_info: Optional mode information dictionary from create_mode_source.
        simulation_steps: Maximum FDTD time steps (default: 20000).
        check_every_n: Steps between convergence checks (default: 1000).
        source_ramp_periods: Source ramp-up periods (default: 5.0).
        add_absorption: Whether to add absorbing boundaries (default: True).
        absorption_widths: Absorber widths (x, y, z) in pixels (default: (60, 40, 40)).
        absorption_coeff: Absorption coefficient (default: 1e-4).
        api_key: API key for authentication. If None, uses configured API key.
        gpu_type: GPU type - "B200", "H200", "H100", "A100-80GB", etc.
        convergence: Early stopping preset ("quick", "default", "thorough", "full").

    Returns:
        Dictionary with simulation results including:
            - monitor_data: Dict mapping monitor names to field arrays
            - sim_time: GPU simulation time in seconds
            - performance: Grid points × steps per second
            - converged: Whether simulation converged early
            - convergence_step: Step at which convergence was reached (if applicable)

    Example:
        >>> # Create structure locally
        >>> structure = hwc.create_structure(layers=[...])
        >>> structure_recipe = structure.extract_recipe()
        >>>
        >>> # Create source locally
        >>> source_field, source_offset, mode_info = hwc.create_mode_source(...)
        >>>
        >>> # Create monitors locally
        >>> monitors = hwc.MonitorSet()
        >>> monitors.add_monitors_at_position(...)
        >>> monitors_recipe = monitors.recipe
        >>>
        >>> # Run on cloud GPU
        >>> results = hwc.simulate(
        ...     structure_recipe=structure_recipe,
        ...     source_field=source_field,
        ...     source_offset=source_offset,
        ...     freq_band=freq_band,
        ...     monitors_recipe=monitors_recipe,
        ...     gpu_type="H100"
        ... )
    """
    # Use provided api_key or fall back to configured one
    effective_api_key = api_key or _API_CONFIG.get('api_key')
    if not effective_api_key:
        raise ValueError("No API key provided. Either pass api_key parameter or call configure_api() first.")

    # Configure API if api_key was provided and different from current
    if api_key and api_key != _API_CONFIG.get('api_key'):
        configure_api(api_key=api_key, validate=False)

    API_URL = _API_CONFIG['api_url']

    print(f"\n{'='*60}")
    print("LOCAL WORKFLOW: Preparing simulation data for cloud GPU")
    print(f"{'='*60}")

    start_time = time.time()

    # Get structure dimensions from recipe metadata
    dimensions = structure_recipe.get('metadata', {}).get('final_shape', [0, 0, 0, 0])[1:]
    if not dimensions or dimensions == [0, 0, 0]:
        # Fallback: try to infer from source field shape
        source_array = np.asarray(source_field)
        if len(source_array.shape) >= 5:
            dimensions = list(source_array.shape[2:])  # (num_freqs, 6, x, y, z)
    print(f"  Structure dimensions: {dimensions}")

    # Encode source field to base64
    print("Encoding source field...")
    source_array = np.asarray(source_field)
    source_field_b64 = encode_array(source_array)
    source_field_shape = list(source_array.shape)
    print(f"  Source field shape: {source_field_shape}")

    # Use monitors recipe directly
    print(f"  Monitors: {[m['name'] for m in monitors_recipe]}")

    # Build convergence config
    conv_config = None
    use_early_stopping = convergence != "full"
    if use_early_stopping:
        if isinstance(convergence, str):
            if convergence not in CONVERGENCE_PRESETS:
                raise ValueError(f"Unknown convergence preset: {convergence}. "
                               f"Valid options: {list(CONVERGENCE_PRESETS.keys())}")
            conv_config = CONVERGENCE_PRESETS[convergence]
        elif isinstance(convergence, ConvergenceConfig):
            conv_config = convergence
        else:
            conv_config = CONVERGENCE_PRESETS["default"]

    endpoint = "/early_stopping" if use_early_stopping else "/simulate"

    # Build request body
    body = {
        "structure_recipe": _to_json_serializable(structure_recipe),
        "source_field_b64": source_field_b64,
        "source_field_shape": source_field_shape,
        "source_offset": list(source_offset),
        "freq_band": list(freq_band),
        "monitors": _to_json_serializable(monitors_recipe),
        "max_steps": simulation_steps,
        "check_every_n": conv_config.check_every_n if conv_config else check_every_n,
        "source_ramp_periods": source_ramp_periods,
        "gpu_type": gpu_type,
        "add_absorption": add_absorption,
        "absorption_widths": list(absorption_widths),
        "absorption_coeff": float(absorption_coeff),
    }

    # Add early stopping parameters
    if use_early_stopping and conv_config:
        body["relative_threshold"] = conv_config.relative_threshold
        body["absolute_threshold"] = 1e-10
        body["significant_power_threshold"] = conv_config.power_threshold
        body["min_stable_checks"] = conv_config.min_stable_checks

    print(f"\nStarting GPU simulation...")
    print(f"  GPU: {gpu_type}, Max steps: {simulation_steps}")
    if use_early_stopping:
        print(f"  Convergence: {convergence}")

    headers = {
        "X-API-Key": effective_api_key,
        "Content-Type": "application/json"
    }

    try:
        print(f"Calling {endpoint} API...")
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=body,
            headers=headers,
            timeout=600
        )
        response.raise_for_status()
        result = response.json()

        sim_time = result.get("sim_time", 0)
        total_time = time.time() - start_time
        converged = result.get("converged", False)

        print(f"Simulation completed in {sim_time:.1f}s (total: {total_time:.1f}s)")
        if converged:
            print(f"  Converged at step {result.get('convergence_step', 0)}")

        # Decode monitor data
        monitor_data_raw = result.get("monitor_data_b64", {})
        monitor_shapes = result.get("monitor_data_shapes", {})
        monitor_data = {}

        for name, data_b64 in monitor_data_raw.items():
            if isinstance(data_b64, str) and data_b64:
                shape = monitor_shapes.get(name)
                if shape:
                    try:
                        arr_bytes = base64.b64decode(data_b64)
                        arr = np.frombuffer(arr_bytes, dtype=np.complex64).reshape(shape)
                        monitor_data[name] = arr
                    except Exception as e:
                        print(f"Warning: Failed to decode monitor {name}: {e}")

        # Build monitor_names dict (name -> index for compatibility with quick_view_monitors)
        monitor_names = {name: i for i, name in enumerate(monitor_data.keys())}

        return {
            "monitor_data": monitor_data,
            "monitor_names": monitor_names,
            "sim_time": sim_time,
            "performance": result.get("performance", 0),
            "converged": converged,
            "convergence_step": result.get("convergence_step"),
            "dimensions": dimensions,
            "freq_band": freq_band,
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "simulation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error running simulation: {e}")
        return None


# =============================================================================
# ONE-SHOT WORKFLOW (DEPRECATED - use SDK workflow instead)
# =============================================================================

def simulate_one_shot(
    device_type: str,
    pdk_config: Dict[str, Any],
    source_port: str = "o1",
    wavelength_um: float = 1.55,
    wavelength_span_nm: float = 100,
    num_wavelengths: int = 5,
    cells_per_wavelength: int = 25,
    num_steps: int = 30000,
    check_every_n: int = 1000,
    source_ramp_periods: float = 5.0,
    gpu_type: str = "H100",
    device_params: Optional[Dict[str, Any]] = None,
    min_steps: int = 0,
    min_stable_checks: int = 3,
    absorber_width: Optional[int] = None,
    absorber_coeff: Optional[float] = None,
    significant_power_threshold: float = 1e-6,
    required_ports: Optional[List[str]] = None,
    poll_interval: float = 2.0,
    # build_recipe parameters
    extension_length: float = 2.0,
    total_height_um: float = 4.0,
    padding: Tuple[int, int, int, int] = (100, 100, 0, 0),
    density_radius: int = 3,
    vertical_radius: float = 2.0,
    # build_monitors parameters
    monitor_x_um: float = 0.1,
    monitor_y_um: float = 1.5,
    monitor_z_um: float = 1.5,
    source_offset_cells: int = 5,
    include_field_monitor: bool = True,
    # solve_mode_source parameters
    mode_num: int = 0,
    slice_half_width: int = 5,
    propagation_axis: str = "x",
    # get_default_absorber_params parameters
    absorber_fraction: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """One-shot FDTD simulation on Modal GPUs (combines setup + simulation).

    NOTE: For better control, use the two-stage workflow instead:
        1. prepare_simulation() - Get mode preview and setup data
        2. run_simulation() - Run FDTD with pre-computed setup

    Args:
        device_type: Device type name (e.g., "mmi2x2", "mmi1x2").
        pdk_config: PDK configuration dict with material parameters:
            - n_core: Core refractive index (e.g., 3.48 for silicon)
            - n_clad: Cladding refractive index (e.g., 1.45 for SiO2)
            - wg_height_um: Waveguide height in micrometers (e.g., 0.22)
            - clad_top_um: Top cladding thickness in micrometers (e.g., 1.89)
            - clad_bot_um: Bottom cladding (BOX) thickness in micrometers (e.g., 2.0)
        source_port: Input port name (e.g., "o1", "o2").
        wavelength_um: Center wavelength in micrometers (default: 1.55).
        wavelength_span_nm: Wavelength span in nanometers (default: 100).
        num_wavelengths: Number of wavelength points (default: 5).
        cells_per_wavelength: FDTD resolution (default: 25).
        num_steps: Maximum FDTD steps (default: 30000).
        check_every_n: Convergence check interval (default: 1000).
        source_ramp_periods: Source ramp-up periods (default: 5.0).
        gpu_type: GPU type - "B200", "H200", "H100", "A100-80GB", etc.
        device_params: Optional dict of device-specific parameters.
        min_steps: Minimum FDTD steps before convergence check.
        min_stable_checks: Required consecutive stable checks.
        absorber_width: PML absorber width in pixels.
        absorber_coeff: PML absorption coefficient.
        significant_power_threshold: Min power for port detection.
        required_ports: List of port names that must have power.
        poll_interval: Seconds between status polls.

        # Structure recipe parameters (from build_recipe):
        extension_length: Length to extend ports in um (default: 2.0).
        total_height_um: Total structure height in um (default: 4.0).
        padding: (left, right, top, bottom) padding in theta pixels (default: (100,100,0,0)).
        density_radius: Radius for density filtering (default: 3).
        vertical_radius: Vertical blur radius (default: 2.0).

        # Monitor parameters (from build_monitors):
        monitor_x_um: Monitor thickness in x direction (default: 0.1).
        monitor_y_um: Monitor size in y direction (default: 1.5).
        monitor_z_um: Monitor size in z direction (default: 1.5).
        source_offset_cells: Offset of source from monitor in cells (default: 5).
        include_field_monitor: If True, include xy_mid monitor for field vis (default: True).

        # Mode solver parameters (from solve_mode_source):
        mode_num: Mode number to solve (0 = fundamental, default: 0).
        slice_half_width: Half-width of mode slice in cells (default: 5).
        propagation_axis: Propagation axis 'x', 'y', or 'z' (default: "x").

        # Absorber parameters (from get_default_absorber_params):
        absorber_fraction: Fraction of structure for absorber (default: 0.1).

    Returns:
        Dict with simulation results:
        - s_parameters: Analyzed transmission data
        - field_intensity: 2D field for visualization
        - sim_time: GPU simulation time in seconds
        - converged: Whether simulation converged

    Example:
        >>> pdk_config = {
        ...     "n_core": 3.48,
        ...     "n_clad": 1.45,
        ...     "wg_height_um": 0.22,
        ...     "clad_top_um": 1.89,
        ...     "clad_bot_um": 2.0,
        ... }
        >>> results = hwc.simulate(
        ...     device_type="mmi2x2",
        ...     pdk_config=pdk_config,
        ...     source_port="o1",
        ...     wavelength_um=1.55,
        ... )
        >>> print(f"T_total: {results['s_parameters']['T_total']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Validate pdk_config
    required_fields = ["n_core", "n_clad", "wg_height_um", "clad_top_um", "clad_bot_um"]
    missing = [f for f in required_fields if f not in pdk_config]
    if missing:
        raise ValueError(f"pdk_config missing required fields: {missing}")

    print(f"Starting simulation for {device_type}...")
    print(f"  Source: {source_port}, Wavelength: {wavelength_um} um (+/- {wavelength_span_nm/2} nm)")
    print(f"  GPU: {gpu_type}")

    body = {
        "device_type": device_type,
        "source_port": source_port,
        "wavelength_um": wavelength_um,
        "wavelength_span_nm": wavelength_span_nm,
        "num_wavelengths": num_wavelengths,
        "cells_per_wavelength": cells_per_wavelength,
        "num_steps": num_steps,
        "check_every_n": check_every_n,
        "source_ramp_periods": source_ramp_periods,
        "gpu_type": gpu_type,
        "pdk_config": pdk_config,
        "device_params": device_params or {},
        "min_steps": min_steps,
        "min_stable_checks": min_stable_checks,
        "absorber_width": absorber_width,
        "absorber_coeff": absorber_coeff,
        "significant_power_threshold": significant_power_threshold,
        "required_ports": required_ports,
        # build_recipe parameters
        "extension_length": extension_length,
        "total_height_um": total_height_um,
        "padding": list(padding),
        "density_radius": density_radius,
        "vertical_radius": vertical_radius,
        # build_monitors parameters
        "monitor_x_um": monitor_x_um,
        "monitor_y_um": monitor_y_um,
        "monitor_z_um": monitor_z_um,
        "source_offset_cells": source_offset_cells,
        "include_field_monitor": include_field_monitor,
        # solve_mode_source parameters
        "mode_num": mode_num,
        "slice_half_width": slice_half_width,
        "propagation_axis": propagation_axis,
        # get_default_absorber_params parameters
        "absorber_fraction": absorber_fraction,
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        # Start the job
        response = requests.post(
            f"{API_URL}/simulate/start",
            json=body,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") != "success":
            raise Exception(f"Failed to start simulation: {result}")

        job_id = result.get("job_id")
        print(f"Job started: {job_id[:8]}...")

        # Poll for completion
        last_progress = ""
        while True:
            time.sleep(poll_interval)

            status_response = requests.get(
                f"{API_URL}/simulate/status/{job_id}",
                headers=headers,
                timeout=30
            )
            status_result = status_response.json()

            status = status_result.get("status")
            progress = status_result.get("progress", "")

            if progress and progress != last_progress:
                print(f"  {progress}")
                last_progress = progress

            if status == "completed":
                # Extract results from top-level response
                sim_time = status_result.get("sim_time", 0)
                converged = status_result.get("converged", False)
                print(f"Simulation completed in {sim_time:.1f}s (converged: {converged})")

                # Build result dict
                sim_result = {
                    "s_parameters": status_result.get("s_parameters", {}),
                    "sim_time": sim_time,
                    "converged": converged,
                }

                # Decode field_intensity if present
                field_intensity = status_result.get("field_intensity")
                if field_intensity:
                    if "intensity_2d_b64" in field_intensity:
                        field_intensity["intensity_2d"] = decode_array(field_intensity["intensity_2d_b64"])
                        del field_intensity["intensity_2d_b64"]
                    sim_result["field_intensity"] = field_intensity

                return sim_result

            elif status in ("failed", "error"):
                error = status_result.get("error", "Unknown error")
                print(f"Simulation failed: {error}")
                raise Exception(f"Simulation failed: {error}")

            elif status not in ("starting", "running", "pending"):
                raise Exception(f"Unexpected status: {status}")

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "simulate")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error running simulation: {e}")
        return None


# =============================================================================
# GRANULAR WORKFLOW FUNCTIONS
# =============================================================================

def build_recipe(
    component_name: str,
    component_kwargs: Optional[Dict[str, Any]] = None,
    extension_length: float = 2.0,
    resolution_nm: float = 30.0,
    n_core: float = 3.48,
    n_clad: float = 1.45,
    wg_height_um: float = 0.22,
    total_height_um: float = 4.0,
    padding: Tuple[int, int, int, int] = (100, 100, 0, 0),
    density_radius: int = 3,
    vertical_radius: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Build structure recipe from GDSFactory component on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        component_name: Name of gdsfactory component (e.g., "mmi2x2").
        component_kwargs: Kwargs to pass to component constructor.
        extension_length: Length to extend ports in um.
        resolution_nm: Grid resolution in nanometers.
        n_core: Core refractive index.
        n_clad: Cladding refractive index.
        wg_height_um: Waveguide height in um.
        total_height_um: Total structure height in um.
        padding: (left, right, top, bottom) padding in theta pixels.
        density_radius: Radius for density filtering.
        vertical_radius: Vertical blur radius.

    Returns:
        Dictionary containing recipe, density_core, density_clad, dimensions,
        port_info, layer_config, eps_values, resolution_um.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Building recipe for {component_name}...")

    request_data = {
        "component_name": component_name,
        "component_kwargs": component_kwargs,
        "extension_length": extension_length,
        "resolution_nm": resolution_nm,
        "n_core": n_core,
        "n_clad": n_clad,
        "wg_height_um": wg_height_um,
        "total_height_um": total_height_um,
        "padding": list(padding),
        "density_radius": density_radius,
        "vertical_radius": vertical_radius,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/build_recipe",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        results = response.json()

        dims = results.get('dimensions', [])
        ports = list(results.get('port_info', {}).keys())
        print(f"Recipe built: {dims[0]}x{dims[1]}x{dims[2]} cells")
        print(f"Ports: {ports}")

        return {
            'recipe': results['recipe'],
            'density_core': decode_array(results['density_core_b64']),
            'density_clad': decode_array(results['density_clad_b64']),
            'dimensions': tuple(results['dimensions']),
            'port_info': results['port_info'],
            'layer_config': results['layer_config'],
            'eps_values': tuple(results['eps_values']),
            'resolution_um': results['resolution_um'],
            'padding': tuple(results['padding']),
            'device_info': results.get('device_info'),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "recipe building")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error building recipe: {e}")
        return None


def build_monitors(
    port_info: Dict[str, Any],
    dimensions: Tuple[int, int, int],
    source_port: str = "o1",
    monitor_x_um: float = 0.1,
    monitor_y_um: float = 1.5,
    monitor_z_um: float = 1.5,
    resolution_um: float = 0.03,
    source_offset_cells: int = 5,
    structure_recipe: Dict[str, Any] = None,
    show_structure: bool = True,
    include_field_monitor: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build monitors from port information on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        port_info: Port information from build_recipe()
        dimensions: Structure dimensions (Lx, Ly, Lz)
        source_port: Name of port to use as source
        monitor_x_um: Monitor thickness in x (propagation direction)
        monitor_y_um: Monitor size in y
        monitor_z_um: Monitor size in z
        resolution_um: Grid resolution in micrometers
        source_offset_cells: Offset of source from monitor
        structure_recipe: Recipe from build_recipe() - required for visualization
        show_structure: If True and structure_recipe provided, show structure plot
        include_field_monitor: If True, create xy_mid monitor for field visualization
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Building monitors (source: {source_port})...")

    request_data = {
        "port_info": port_info,
        "dimensions": list(dimensions),
        "source_port": source_port,
        "monitor_x_um": monitor_x_um,
        "monitor_y_um": monitor_y_um,
        "monitor_z_um": monitor_z_um,
        "resolution_um": resolution_um,
        "source_offset_cells": source_offset_cells,
        "include_field_monitor": include_field_monitor,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/build_monitors",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        # Add xy_mid field monitor if requested and not already present
        if include_field_monitor and 'xy_mid' not in result.get('monitor_names', {}):
            Lx, Ly, Lz = dimensions
            z_mid = Lz // 2
            # Create xy_mid monitor covering full xy plane at z=z_mid
            # Must use 'shape' and 'offset' format (same as port monitors)
            # shape: (Lx, Ly, 1) - full xy plane, 1 z-slice
            # offset: (0, 0, z_mid) - starting position
            xy_mid_monitor = {
                'name': 'xy_mid',
                'shape': [Lx, Ly, 1],
                'offset': [0, 0, z_mid],
            }
            # Add to monitors list and names
            monitors = result.get('monitors', [])
            monitor_idx = len(monitors)
            monitors.append(xy_mid_monitor)
            result['monitors'] = monitors
            result['monitor_names']['xy_mid'] = monitor_idx

        print(f"Monitors built: {list(result.get('monitor_names', {}).keys())}")

        # Visualize structure if requested and recipe provided
        if show_structure and structure_recipe is not None:
            visualize_structure(
                structure_recipe=structure_recipe,
                monitors=result['monitors'],
                monitor_names=result['monitor_names'],
                dimensions=dimensions,
                source_position=result['source_position'],
                axis='z',
                show=True,
                figsize=(14, 3),
            )

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "monitor building")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error building monitors: {e}")
        return None


def compute_freq_band(
    wl_min_um: float = 1.55,
    wl_max_um: float = 1.55,
    n_freqs: int = 1,
    resolution_um: float = 0.03,
) -> Optional[Dict[str, Any]]:
    """Compute frequency band from wavelength range on Modal CPU.

    This function runs on CPU and does NOT consume credits.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Computing freq band: {wl_min_um}-{wl_max_um} um ({n_freqs} points)...")

    request_data = {
        "wl_min_um": wl_min_um,
        "wl_max_um": wl_max_um,
        "n_freqs": n_freqs,
        "resolution_um": resolution_um,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/compute_freq_band",
            json=request_data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        results = response.json()
        print(f"Freq band computed: {results['freq_band']}")

        return {
            'freq_band': tuple(results['freq_band']),
            'wavelengths_um': results['wavelengths_um'],
            'frequencies_omega': results['frequencies_omega'],
            'resolution_um': results['resolution_um'],
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "frequency band computation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error computing freq band: {e}")
        return None


def solve_mode_source(
    density_core: np.ndarray,
    density_clad: np.ndarray,
    source_x_position: int,
    mode_bounds: Dict[str, int],
    layer_config: Dict[str, Any],
    eps_values: Tuple[float, float],
    freq_band: Tuple[float, float, int],
    slice_half_width: int = 5,
    mode_num: int = 0,
    propagation_axis: str = "x",
    show_mode: bool = True,
) -> Optional[Dict[str, Any]]:
    """Solve for waveguide mode source field on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        density_core: Core density array from build_recipe()
        density_clad: Cladding density array from build_recipe()
        source_x_position: X position for source plane
        mode_bounds: Mode bounds from build_monitors()
        layer_config: Layer configuration from build_recipe()
        eps_values: Permittivity values (eps_core, eps_clad)
        freq_band: Frequency band (f_min, f_max, n_freqs)
        slice_half_width: Half-width of mode slice in cells
        mode_num: Mode number (0 = fundamental)
        propagation_axis: Propagation axis ('x', 'y', or 'z')
        show_mode: If True, display mode profile visualization
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Solving mode at x={source_x_position}...")

    request_data = {
        "density_core_b64": encode_array(np.array(density_core)),
        "density_core_shape": list(density_core.shape),
        "density_clad_b64": encode_array(np.array(density_clad)),
        "density_clad_shape": list(density_clad.shape),
        "source_x_position": source_x_position,
        "mode_bounds": mode_bounds,
        "layer_config": layer_config,
        "eps_values": list(eps_values),
        "freq_band": list(freq_band),
        "slice_half_width": slice_half_width,
        "mode_num": mode_num,
        "propagation_axis": propagation_axis,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/solve_mode_source",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        results = response.json()

        mode_info = results.get('mode_info', {})
        n_eff = mode_info.get('n_eff', 'N/A')
        print(f"Mode solved: n_eff={n_eff}")

        source_field = decode_array(results['source_field_b64'])
        source_offset = tuple(results['source_offset'])

        result = {
            'source_field': source_field,
            'source_offset': source_offset,
            'mode_info': mode_info,
            'freq_band': tuple(results['freq_band']),
            'solve_time_seconds': results.get('solve_time_seconds', 0.0),
        }

        # Visualize mode if requested
        if show_mode:
            visualize_mode_source(
                source_field=source_field,
                source_offset=source_offset,
                show=True,
            )

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "mode source solving")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error solving mode source: {e}")
        return None


def get_default_absorber_params(
    structure_dimensions: Tuple[int, int, int],
    absorber_fraction: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Get default absorber parameters based on structure size on Modal CPU.

    This function runs on CPU and does NOT consume credits.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Computing absorber params...")

    request_data = {
        "structure_dimensions": list(structure_dimensions),
        "absorber_fraction": absorber_fraction,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/get_default_absorber_params",
            json=request_data,
            headers=headers,
            timeout=120  # Modal cold start can take time
        )
        response.raise_for_status()
        results = response.json()

        print(f"Absorber params computed")

        return {
            'absorption_widths': tuple(results['absorption_widths']),
            'absorption_coeff': results['absorption_coeff'],
            'add_absorption': results['add_absorption'],
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "absorber parameter computation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting absorber params: {e}")
        return None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_structure(
    structure_recipe: Dict[str, Any],
    monitors: list,
    monitor_names: Dict[str, int] = None,
    dimensions: tuple = None,
    source_position: int = None,
    axis: str = "y",
    position: int = None,
    show: bool = True,
    save_path: str = None,
    figsize: tuple = (14, 8),
) -> Dict[str, Any]:
    """Generate structure + monitors visualization.

    This function calls Modal to build the 3D structure and generate
    a 2D cross-section visualization with monitor positions overlaid.
    Useful for debugging simulation setup without running locally.

    Args:
        structure_recipe: Recipe dict from build_recipe()
        monitors: List of monitor dicts (from build_monitors)
        monitor_names: Dict mapping names to indices
        dimensions: (Lx, Ly, Lz) structure dimensions
        source_position: X position of source plane (optional)
        axis: Viewing axis ('x', 'y', or 'z'), default 'y'
        position: Slice position along axis (None = middle)
        show: If True, display the image using matplotlib
        save_path: If provided, save PNG to this path
        figsize: Figure size as (width, height)

    Returns:
        Dictionary containing:
            - image_b64: Base64-encoded PNG image
            - dimensions: Structure dimensions
            - slice_info: Dict with axis and position of slice

    Example:
        >>> recipe_result = hwc.build_recipe(component_name="mmi2x2")
        >>> monitor_result = hwc.build_monitors(...)
        >>> result = hwc.visualize_structure(
        ...     structure_recipe=recipe_result['recipe'],
        ...     monitors=monitor_result['monitors'],
        ...     source_position=monitor_result['source_position'],
        ...     axis='y'
        ... )
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Generating structure visualization (axis={axis})...")

    request_data = {
        "structure_recipe": _to_json_serializable(structure_recipe),
        "monitors": _to_json_serializable(monitors),
        "monitor_names": _to_json_serializable(monitor_names),
        "dimensions": list(dimensions) if dimensions else None,
        "source_position": source_position,
        "axis": axis,
        "position": position,
        "figsize": list(figsize) if figsize else None,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/visualize/structure",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        print(f"Visualization generated: {result['dimensions']}")

        # Display image if requested
        if show:
            try:
                import matplotlib.pyplot as plt
                from PIL import Image
                import io

                image_bytes = base64.b64decode(result['image_b64'])
                image = Image.open(io.BytesIO(image_bytes))

                plt.figure(figsize=(14, 8))
                plt.imshow(image)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("Install matplotlib and Pillow to display images: pip install matplotlib Pillow")

        # Save image if requested
        if save_path:
            image_bytes = base64.b64decode(result['image_b64'])
            with open(save_path, 'wb') as f:
                f.write(image_bytes)
            print(f"Image saved to: {save_path}")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "structure visualization")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error generating visualization: {e}")
        return None


def visualize_mode_source(
    source_field: np.ndarray = None,
    source_field_b64: str = None,
    source_field_shape: list = None,
    source_offset: tuple = None,
    show: bool = True,
    save_path: str = None,
) -> Dict[str, Any]:
    """Generate mode source visualization.

    This function calls Modal to visualize the waveguide mode profile,
    showing field intensity and individual field components (Ex, Ey, Ez).

    Args:
        source_field: Source field array (n_freqs, 6, 1, ny, nz). Either this or source_field_b64 required.
        source_field_b64: Base64-encoded source field. Either this or source_field required.
        source_field_shape: Shape of source field (required if using source_field_b64)
        source_offset: Source offset (x, y, z) for display
        show: If True, display the image using matplotlib
        save_path: If provided, save PNG to this path

    Returns:
        Dictionary containing:
            - image_b64: Base64-encoded PNG image
            - field_shape: Shape of the source field
            - max_intensity: Maximum field intensity

    Example:
        >>> source_result = hwc.solve_mode_source(...)
        >>> result = hwc.visualize_mode_source(
        ...     source_field=source_result['source_field'],
        ...     source_offset=source_result['source_offset'],
        ...     show=True,
        ... )
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Encode source field if provided as array
    if source_field is not None:
        source_field_b64 = encode_array(source_field)
        source_field_shape = list(source_field.shape)
    elif source_field_b64 is None:
        raise ValueError("Either source_field or source_field_b64 must be provided")

    print(f"Generating mode source visualization...")

    request_data = {
        "source_field_b64": source_field_b64,
        "source_field_shape": source_field_shape,
        "source_offset": list(source_offset) if source_offset else None,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/visualize/mode_source",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        print(f"Mode visualization generated: shape={result['field_shape']}, max_intensity={result['max_intensity']:.2e}")

        # Display image if requested
        if show:
            try:
                import matplotlib.pyplot as plt
                from PIL import Image
                import io

                image_bytes = base64.b64decode(result['image_b64'])
                image = Image.open(io.BytesIO(image_bytes))

                plt.figure(figsize=(14, 6))
                plt.imshow(image)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("Install matplotlib and Pillow to display images: pip install matplotlib Pillow")

        # Save image if requested
        if save_path:
            image_bytes = base64.b64decode(result['image_b64'])
            with open(save_path, 'wb') as f:
                f.write(image_bytes)
            print(f"Image saved to: {save_path}")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "mode source visualization")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error generating visualization: {e}")
        return None


# =============================================================================
# Analysis Functions (local, no API calls)
# =============================================================================

def compute_poynting_vector(fields: np.ndarray) -> np.ndarray:
    """Compute Poynting vector from electromagnetic field components.

    Calculates S = 0.5 * Re(E × H*), the time-averaged power flow.

    Args:
        fields: Field array of shape (6, ny, nz) or (6, nx, ny, nz) where
                the first axis contains [Ex, Ey, Ez, Hx, Hy, Hz]

    Returns:
        Poynting vector array of shape (3, ny, nz) or (3, nx, ny, nz)
        containing [Sx, Sy, Sz] components
    """
    Ex, Ey, Ez = fields[0], fields[1], fields[2]
    Hx, Hy, Hz = fields[3], fields[4], fields[5]

    # S = 0.5 * Re(E × H*)
    Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
    Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
    Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))

    return np.stack([Sx, Sy, Sz], axis=0)


def compute_monitor_power(
    monitor_data: np.ndarray,
    direction: str = 'x'
) -> float:
    """Compute total power through a monitor from field data.

    Args:
        monitor_data: Field data from a monitor, shape (n_freqs, 6, ny, nz)
                     or (6, n_samples, ny, nz) depending on monitor type
        direction: Direction of power flow ('x', 'y', or 'z')

    Returns:
        Total power (absolute value) through the monitor
    """
    # Handle different data shapes
    if len(monitor_data.shape) == 4:
        # Shape: (6, n_samples, ny, nz) - average over samples
        fields = monitor_data
        fields_mean = np.mean(fields, axis=1)
    else:
        fields_mean = monitor_data

    S = compute_poynting_vector(fields_mean)

    # Select component based on direction
    direction_map = {'x': 0, 'y': 1, 'z': 2}
    component = direction_map.get(direction.lower(), 0)

    return np.abs(np.sum(S[component, :, :]))


def analyze_transmission(
    results: Dict[str, Any],
    input_monitor: str = "Input_o1",
    output_monitors: List[str] = None,
    direction: str = 'x',
    print_results: bool = True,
) -> Dict[str, Any]:
    """Analyze transmission from simulation results.

    Computes transmission coefficients for each output monitor relative
    to the input monitor using Poynting vector power flow.

    Args:
        results: Simulation results from run_simulation()
        input_monitor: Name of input monitor
        output_monitors: List of output monitor names. If None, auto-detects
                        monitors starting with "Output_"
        direction: Direction of power flow for Poynting vector ('x', 'y', 'z')
        print_results: If True, print formatted results table

    Returns:
        Dictionary with:
            - 'power_in': Input power
            - 'transmissions': Dict mapping monitor name to transmission value
            - 'total_transmission': Sum of all output transmissions
            - 'excess_loss_dB': Excess loss in dB (10*log10(total))
    """
    monitor_data = results.get('monitor_data', {})
    monitor_shapes = results.get('monitor_data_shapes', {})

    # Auto-detect output monitors if not specified
    if output_monitors is None:
        output_monitors = [name for name in monitor_data.keys()
                         if name.startswith("Output_")]

    # Helper to extract fields from monitor data (handles both formats)
    def _get_fields(monitor_name, freq_idx=0):
        """Extract field data, handling various formats and reshaping if needed."""
        data = monitor_data[monitor_name]
        arr = np.array(data)

        # If 1D array, try to reshape using stored shape
        if arr.ndim == 1 and monitor_name in monitor_shapes:
            target_shape = tuple(monitor_shapes[monitor_name])
            n_elements = np.prod(target_shape)
            expected_float32_len = n_elements * 2  # complex64 stored as float32 pairs

            # Handle cases where array might have padding/extra elements
            if len(arr) >= expected_float32_len:
                # Truncate to expected size and view as complex64
                arr = arr[:expected_float32_len].view(np.complex64).reshape(target_shape)
            elif len(arr) == n_elements:
                # Already the right number of elements (complex stored directly)
                arr = arr.reshape(target_shape)
            elif len(arr) * 2 == n_elements:
                # float64 or needs casting to complex
                arr = arr.astype(np.complex64).reshape(target_shape)

        if arr.ndim == 5:
            # Shape: (1, 6, 3, ny, nz) - squeeze and take first freq
            arr = arr[freq_idx]  # Now (6, 3, ny, nz)
            # Average over the 3rd dimension if present
            if arr.ndim == 4 and arr.shape[1] == 3:
                arr = np.mean(arr, axis=1)  # Now (6, ny, nz)
            return arr
        elif arr.ndim == 4:
            # Shape: (n_freqs, 6, ny, nz) - index into frequency axis
            return arr[freq_idx]
        elif arr.ndim == 3:
            # Shape: (6, ny, nz) - use directly
            return arr
        else:
            raise ValueError(f"Unexpected monitor data shape for {monitor_name}: {arr.shape}. Expected 3D, 4D or 5D array.")

    # Compute input power
    if input_monitor not in monitor_data:
        raise ValueError(f"Input monitor '{input_monitor}' not found in results")

    input_fields = _get_fields(input_monitor)
    power_in = compute_monitor_power(input_fields, direction)

    # Compute transmission for each output
    transmissions = {}
    for monitor_name in output_monitors:
        if monitor_name not in monitor_data:
            print(f"Warning: Monitor '{monitor_name}' not found, skipping")
            continue
        output_fields = _get_fields(monitor_name)
        power_out = compute_monitor_power(output_fields, direction)
        transmissions[monitor_name] = power_out / power_in

    total_T = sum(transmissions.values())
    excess_loss_dB = 10 * np.log10(total_T) if total_T > 0 else float('-inf')

    # Print results if requested
    if print_results:
        print(f"{'='*60}")
        print(f"Transmission Analysis (Input: {input_monitor})")
        print(f"{'Monitor':<20} {'Transmission':>12} {'dB':>10}")
        print("-" * 60)
        for name, T in transmissions.items():
            T_dB = 10 * np.log10(T) if T > 0 else float('-inf')
            print(f"{name:<20} {float(T):>12.4f} {float(T_dB):>10.2f}")
        print("-" * 60)
        print(f"{'Total':<20} {float(total_T):>12.4f} {float(excess_loss_dB):>10.2f}")
        print(f"{'='*60}")

    return {
        'power_in': power_in,
        'transmissions': transmissions,
        'total_transmission': total_T,
        'excess_loss_dB': excess_loss_dB,
    }


def get_field_intensity_2d(
    results: Dict[str, Any],
    monitor_name: str = 'xy_mid',
    dimensions: Tuple[int, int, int] = None,
    resolution_um: float = None,
    freq_band: Tuple[float, float, float] = None,
) -> Dict[str, Any]:
    """Extract 2D field intensity from simulation results for plotting.

    Computes |E|² from the specified monitor and provides extent/wavelength
    info for matplotlib plotting.

    Args:
        results: Simulation results from run_simulation()
        monitor_name: Name of the 2D monitor (default: 'xy_mid')
        dimensions: Structure dimensions (Lx, Ly, Lz) for computing extent
        resolution_um: Grid resolution in micrometers
        freq_band: Frequency band tuple (freq_min, freq_max, n_freqs) for wavelength

    Returns:
        Dictionary with:
            - 'intensity': 2D numpy array of |E|² ready for imshow
            - 'extent': [x_min, x_max, y_min, y_max] in μm for imshow extent
            - 'wavelength_nm': Wavelength in nm (if freq_band provided)
    """
    monitor_data = results.get('monitor_data', {})
    monitor_shapes = results.get('monitor_data_shapes', {})

    if monitor_name not in monitor_data:
        raise ValueError(f"Monitor '{monitor_name}' not found in results")

    data = np.array(monitor_data[monitor_name])

    # If 1D array, try to reshape using stored shape
    if data.ndim == 1 and monitor_name in monitor_shapes:
        target_shape = tuple(monitor_shapes[monitor_name])
        n_elements = np.prod(target_shape)
        expected_float32_len = n_elements * 2  # complex64 stored as float32 pairs

        # Handle cases where array might have padding/extra elements
        if len(data) >= expected_float32_len:
            # Truncate to expected size and view as complex64
            data = data[:expected_float32_len].view(np.complex64).reshape(target_shape)
        elif len(data) == n_elements:
            # Already the right number of elements (complex stored directly)
            data = data.reshape(target_shape)

    # Handle different data shapes from various endpoints
    if data.ndim == 5:
        # Two possible shapes:
        # 1. Port monitors: (1, 6, 3, ny, nz) - small 3rd dim, average over it
        # 2. xy_mid monitor: (1, 6, Lx, Ly, 1) - large xy dims, squeeze last dim
        data = data[0]  # Now (6, dim2, dim3, dim4)

        if data.shape[-1] == 1:
            # xy_mid case: (6, Lx, Ly, 1) - squeeze z dimension
            data = data.squeeze(-1)  # Now (6, Lx, Ly)
            E_fields = data[0:3, :, :]  # (3, Lx, Ly)
            field_intensity = np.sum(np.abs(E_fields)**2, axis=0)
            field_2d = field_intensity.T  # (Ly, Lx) for imshow
        else:
            # Port monitor case: (6, 3, ny, nz) - average over small dimension
            E_fields = data[0:3, :, :, :]  # (3, 3, ny, nz)
            E_fields = np.mean(E_fields, axis=1)  # (3, ny, nz)
            field_intensity = np.sum(np.abs(E_fields)**2, axis=0)
            field_2d = field_intensity.T
    elif data.ndim == 4:
        # Shape: (n_freqs, 6, ny, nz) - frequency-indexed data
        # Use first frequency
        E_fields = data[0, 0:3, :, :]
        field_intensity = np.sum(np.abs(E_fields)**2, axis=0)
        field_2d = field_intensity.T
    elif data.ndim == 3:
        # Shape: (6, ny, nz) - direct field data
        E_fields = data[0:3, :, :]
        field_intensity = np.sum(np.abs(E_fields)**2, axis=0)
        field_2d = field_intensity.T
    else:
        raise ValueError(f"Unexpected monitor data shape: {data.shape}. Expected 3D, 4D or 5D array.")

    result = {'intensity': field_2d}

    # Compute extent if dimensions and resolution provided
    if dimensions is not None and resolution_um is not None:
        Lx, Ly, Lz = dimensions
        x_um = Lx * resolution_um
        y_um = Ly * resolution_um
        result['extent'] = [-x_um/2, x_um/2, -y_um/2, y_um/2]

    # Compute wavelength if freq_band provided
    if freq_band is not None and resolution_um is not None:
        freq_at_idx = freq_band[0]
        wl_nm = 2 * np.pi / freq_at_idx * resolution_um * 1000
        result['wavelength_nm'] = wl_nm

    return result


def compute_adjoint_gradient(
    theta: np.ndarray,
    source_field: np.ndarray,
    source_offset: Tuple[int, int, int],
    freq_band: Tuple[float, float, int],
    loss_monitor_shape: Tuple[int, int, int],
    loss_monitor_offset: Tuple[int, int, int],
    design_monitor_shape: Tuple[int, int, int],
    design_monitor_offset: Tuple[int, int, int],
    structure_spec: Dict[str, Any],
    loss_fn: Optional[Callable] = None,
    mode_field: Optional[np.ndarray] = None,
    input_power: Optional[float] = None,
    mode_cross_power: Optional[float] = None,
    mode_axis: int = 0,
    power_axis: Optional[int] = None,
    power_maximize: bool = True,
    intensity_component: Optional[str] = None,
    intensity_maximize: bool = True,
    absorption_widths: Tuple[int, int, int] = (70, 35, 17),
    absorption_coeff: float = 0.00489,
    gpu_type: str = "H100",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute adjoint gradient for inverse design on GPU via API.

    Computes the gradient of a loss function with respect to design variables
    (theta) using the memory-efficient adjoint method. This enables gradient-based
    optimization of photonic devices.

    The 3-part autodiff chain:
        theta -> permittivity (via structure_spec) -> fields (FDTD) -> loss

    Loss function options (in priority order):
        1. loss_fn: Custom function (most flexible)
        2. mode_field: Mode coupling efficiency
        3. power_axis: Poynting vector power (S_from_slice)
        4. intensity_component: Simple |E|^2

    Args:
        theta: Design variables (2D numpy array). Values typically in [0, 1].
        source_field: Source field array, shape (n_freq, 6, sx, sy, sz).
        source_offset: Source injection position (x, y, z) in pixels.
        freq_band: Frequency specification as (omega_min, omega_max, num_freqs).
        loss_monitor_shape: Shape of loss monitor. For power loss, use the
            full cross-section. For point intensity, use (1, 1, 1).
        loss_monitor_offset: Position where loss is computed (x, y, z).
        design_monitor_shape: Shape of design region for gradient computation.
        design_monitor_offset: Offset of design region.
        structure_spec: Structure specification dictionary with:
            - layers_info: list of layer dicts with density_radius, density_alpha,
              permittivity_values, layer_thickness, conductivity_values
            - construction_params: dict with vertical_radius
        loss_fn: Custom loss function (optional). Signature: loss_fn(loss_field) -> scalar.
            loss_field shape: (n_freq, 6, mx, my, mz). Serialized via cloudpickle.
            WARNING: Avoid closures over large arrays (causes memory leaks).
        mode_field: Target mode field for mode coupling loss (optional).
            If provided, also requires input_power and mode_cross_power.
        input_power: Input power for mode coupling loss.
        mode_cross_power: Mode cross power for mode coupling loss.
        mode_axis: Axis for mode overlap calculation (default: 0 for x-propagation).
        power_axis: Axis for Poynting vector power loss (optional).
            0=x, 1=y, 2=z. Uses S_from_slice to compute power through monitor.
        power_maximize: Whether to maximize power (default: True).
        intensity_component: Field component for intensity loss ('Ex', 'Ey', 'Ez').
        intensity_maximize: Whether to maximize intensity (default: True).
        absorption_widths: PML absorption widths (x, y, z) in pixels.
        absorption_coeff: PML absorption coefficient.
        gpu_type: GPU type to use. Options: B200, H200, H100, A100-80GB, A100-40GB, L40S, A10G, T4.
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing:
            - loss: Computed loss value
            - grad_theta: Gradient array (same shape as theta)
            - grad_min: Minimum gradient value
            - grad_max: Maximum gradient value
            - grad_time: Gradient computation time (seconds)
            - total_time: Total time including overhead
            - gpu_type: GPU type used
            - memory_reduction_pct: Memory reduction from efficient adjoint
            - simulation_id: Unique ID for this computation

    Raises:
        ValueError: If no loss specification is provided.

    Note:
        Computation takes approximately 30-120 seconds depending on structure size
        and GPU type. Uses memory-efficient adjoint method (90%+ memory reduction).

    Example 1: Custom loss function
        >>> def my_loss(loss_field):
        ...     import jax.numpy as jnp
        ...     Ez = loss_field[0, 2, :, :, :]
        ...     return -jnp.sum(jnp.abs(Ez)**2)  # Maximize total Ez intensity
        >>>
        >>> result = hwc.compute_adjoint_gradient(
        ...     theta=theta, source_field=source, ...,
        ...     loss_fn=my_loss,
        ...     api_key='your-key'
        ... )

    Example 2: Power through monitor (using S_from_slice)
        >>> result = hwc.compute_adjoint_gradient(
        ...     theta=theta, source_field=source, ...,
        ...     loss_monitor_shape=(1, 50, 50),  # Full YZ cross-section
        ...     loss_monitor_offset=(output_x, 0, 0),
        ...     power_axis=0,  # Maximize Sx (power in x-direction)
        ...     power_maximize=True,
        ...     api_key='your-key'
        ... )

    Example 3: Simple intensity
        >>> result = hwc.compute_adjoint_gradient(
        ...     theta=theta, source_field=source, ...,
        ...     loss_monitor_shape=(1, 1, 1),  # Point monitor
        ...     intensity_component='Ez',
        ...     intensity_maximize=True,
        ...     api_key='your-key'
        ... )
    """
    # Use provided api_key or fall back to configured one
    effective_api_key = api_key or _API_CONFIG.get('api_key')
    if not effective_api_key:
        print("API key required to proceed.")
        print("Sign up for free at spinsphotonics.com to get your API key.")
        return None

    # Configure API if api_key was provided and different from current
    if api_key and api_key != _API_CONFIG.get('api_key'):
        configure_api(api_key=api_key, validate=False)

    API_URL = _API_CONFIG['api_url']

    # Validate loss parameters - at least one must be provided
    if (loss_fn is None and mode_field is None and
        power_axis is None and intensity_component is None):
        raise ValueError(
            "Must provide at least one loss specification:\n"
            "  - loss_fn: Custom loss function\n"
            "  - mode_field: Mode coupling (with input_power, mode_cross_power)\n"
            "  - power_axis: Poynting vector power (0=x, 1=y, 2=z)\n"
            "  - intensity_component: Simple |E|^2 ('Ex', 'Ey', 'Ez')"
        )

    # Encode theta and source_field to base64
    theta_b64 = encode_array(np.array(theta, dtype=np.float32))
    source_field_b64 = encode_array(np.array(source_field))

    # Serialize custom loss function if provided
    loss_fn_pickle_b64 = None
    if loss_fn is not None:
        import cloudpickle
        loss_fn_bytes = cloudpickle.dumps(loss_fn)
        loss_fn_pickle_b64 = base64.b64encode(loss_fn_bytes).decode('utf-8')

    # Prepare mode coupling params if provided
    mode_coupling_params = None
    if mode_field is not None:
        if input_power is None or mode_cross_power is None:
            raise ValueError(
                "mode_field requires input_power and mode_cross_power"
            )
        mode_coupling_params = {
            'mode_field_b64': encode_array(np.array(mode_field)),
            'mode_field_shape': list(np.array(mode_field).shape),
            'input_power': float(input_power),
            'mode_cross_power': float(mode_cross_power),
            'axis': int(mode_axis)
        }

    # Prepare power params if provided
    power_params = None
    if power_axis is not None:
        power_params = {
            'axis': int(power_axis),
            'maximize': power_maximize
        }

    # Prepare intensity params if provided
    intensity_params = None
    if intensity_component is not None:
        intensity_params = {
            'component': intensity_component,
            'maximize': intensity_maximize
        }

    # Build request
    request_data = {
        "theta_b64": theta_b64,
        "theta_shape": list(np.array(theta).shape),
        "source_field_b64": source_field_b64,
        "source_field_shape": list(np.array(source_field).shape),
        "source_offset": list(source_offset),
        "freq_band": [float(x) for x in freq_band],
        "loss_monitor_shape": list(loss_monitor_shape),
        "loss_monitor_offset": list(loss_monitor_offset),
        "design_monitor_shape": list(design_monitor_shape),
        "design_monitor_offset": list(design_monitor_offset),
        "structure_spec": structure_spec,
        "mode_coupling_params": mode_coupling_params,
        "intensity_params": intensity_params,
        "power_params": power_params,
        "loss_fn_pickle_b64": loss_fn_pickle_b64,
        "add_absorption": True,
        "absorption_widths": list(absorption_widths),
        "absorption_coeff": float(absorption_coeff),
        "gpu_type": gpu_type
    }

    headers = {
        "X-API-Key": effective_api_key,
        "Content-Type": "application/json"
    }

    # Debug info
    print(f"\n=== Inverse Design API Request ===")
    print(f"Endpoint: {API_URL}/inverse_design")
    print(f"Theta shape: {request_data['theta_shape']}")
    print(f"Design monitor: shape={design_monitor_shape}, offset={design_monitor_offset}")
    print(f"Loss monitor: shape={loss_monitor_shape}, offset={loss_monitor_offset}")
    print(f"GPU type: {gpu_type}")
    if loss_fn_pickle_b64:
        print(f"Loss type: Custom function (cloudpickle)")
    elif mode_coupling_params:
        print(f"Loss type: Mode coupling")
    elif power_params:
        axis_names = {0: 'x', 1: 'y', 2: 'z'}
        print(f"Loss type: Power S_{axis_names.get(power_axis, power_axis)} via S_from_slice (maximize={power_maximize})")
    elif intensity_params:
        print(f"Loss type: Intensity |{intensity_component}|^2 (maximize={intensity_maximize})")
    print(f"==================================\n")

    # Send request
    try:
        response = requests.post(
            f"{API_URL}/inverse_design",
            json=request_data,
            headers=headers,
            timeout=1800  # 30 minute timeout for inverse design
        )

        response.raise_for_status()

        results = response.json()

        # Decode gradient
        grad_theta = decode_array(results['grad_theta_b64'])

        return {
            'loss': results['loss'],
            'grad_theta': grad_theta,
            'grad_min': results['grad_min'],
            'grad_max': results['grad_max'],
            'grad_time': results['grad_time'],
            'total_time': results['total_time'],
            'gpu_type': results['gpu_type'],
            'memory_reduction_pct': results['memory_reduction_pct'],
            'simulation_id': results.get('simulation_id'),
            'execution_time_seconds': results.get('execution_time_seconds'),
            'computation_time_seconds': results.get('computation_time_seconds')
        }

    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            status_code = e.response.status_code
            response_text = e.response.text

            print(f"\n=== API Error ===")
            print(f"Status Code: {status_code}")
            print(f"Response: {response_text}")
            print(f"=================\n")

            if status_code == 401:
                print("No API key detected in request.")
                print("Sign up for free at spinsphotonics.com to get your API key.")
            elif status_code == 403:
                print("Provided API key is invalid.")
                print("Please verify your API key at spinsphotonics.com/dashboard")
            elif status_code == 402:
                print("Insufficient credits for inverse design computation.")
                print("Minimum required: 1.0 credits")
                print("Add credits at spinsphotonics.com/billing")
            elif status_code == 502:
                print("Service temporarily unavailable. Please retry later.")
            else:
                print(f"Unexpected error (Code: {status_code})")
        return None

    except requests.exceptions.Timeout:
        print("Request timeout.")
        print("Inverse design computation is taking longer than expected. Please try again.")
        return None

    except requests.exceptions.ConnectionError:
        print("Connection failed.")
        print("Unable to reach API servers. Please check your connection.")
        return None

    except requests.exceptions.RequestException:
        print("Communication error.")
        print("Unable to process your request. Please try again later.")
        return None

    except ValueError:
        print("Invalid server response.")
        print("Received malformed data from server.")
        return None


def run_optimization(
    theta: np.ndarray,
    source_field: np.ndarray,
    source_offset: Tuple[int, int, int],
    freq_band: Tuple[float, float, int],
    structure_spec: Dict[str, Any],
    loss_monitor_shape: Tuple[int, int, int],
    loss_monitor_offset: Tuple[int, int, int],
    design_monitor_shape: Tuple[int, int, int],
    design_monitor_offset: Tuple[int, int, int],
    loss_fn: Optional[Callable] = None,
    mode_field: Optional[np.ndarray] = None,
    input_power: Optional[float] = None,
    mode_cross_power: Optional[float] = None,
    mode_axis: int = 0,
    power_axis: Optional[int] = None,
    power_maximize: bool = True,
    intensity_component: Optional[str] = None,
    intensity_maximize: bool = True,
    num_steps: int = 50,
    learning_rate: float = 0.01,
    grad_clip_norm: float = 1.0,
    cosine_decay_alpha: float = 0.1,
    enforce_symmetry: bool = False,
    waveguide_mask: Optional[np.ndarray] = None,
    absorption_widths: Tuple[int, int, int] = (70, 35, 17),
    absorption_coeff: float = 0.00489,
    max_steps: int = 10000,
    check_every_n: int = 1000,
    gpu_type: str = "B200",
    api_key: Optional[str] = None,
) -> 'Generator[Dict[str, Any], None, None]':
    """Run optimization loop on cloud GPU.

    Sends all optimization parameters in a single request. The cloud GPU
    runs the full loop (forward + adjoint FDTD per step, Adam updates) and
    returns results after each completed step.

    **Cancellation:** Stopping iteration (KeyboardInterrupt, ``break``, or
    abandoning the generator) closes the connection and cancels the GPU
    task. You are only charged for steps that actually completed.

    Loss function options (in priority order):
        1. loss_fn: Custom function (most flexible)
        2. mode_field: Mode coupling efficiency
        3. power_axis: Poynting vector power (S_from_slice)
        4. intensity_component: Simple |E|^2

    Args:
        theta: Initial design variables (2D numpy array, values in [0, 1]).
        source_field: Source field array, shape (n_freq, 6, sx, sy, sz).
        source_offset: Source injection position (x, y, z) in pixels.
        freq_band: Frequency specification as (omega_min, omega_max, num_freqs).
        structure_spec: Structure specification dictionary with ``layers_info``
            and ``construction_params``.
        loss_monitor_shape: Shape of loss monitor volume.
        loss_monitor_offset: Position of loss monitor (x, y, z).
        design_monitor_shape: Shape of design region for gradient computation.
        design_monitor_offset: Offset of design region.
        loss_fn: Custom loss function (optional). Serialized via cloudpickle.
        mode_field: Target mode field for mode coupling loss (optional).
        input_power: Input power for mode coupling normalization.
        mode_cross_power: Mode cross power for mode coupling normalization.
        mode_axis: Axis for mode overlap (default: 0 for x-propagation).
        power_axis: Axis for Poynting vector power loss (0=x, 1=y, 2=z).
        power_maximize: Whether to maximize power (default: True).
        intensity_component: Field component for intensity loss ('Ex', 'Ey', 'Ez').
        intensity_maximize: Whether to maximize intensity (default: True).
        num_steps: Number of optimization steps (default: 50).
        learning_rate: Peak learning rate for Adam with cosine decay (default: 0.01).
        grad_clip_norm: Global gradient clipping norm (default: 1.0).
        cosine_decay_alpha: Final LR as fraction of initial (default: 0.1).
        enforce_symmetry: Symmetrize gradient via ``(g + g.T) / 2`` (default: False).
        waveguide_mask: Binary mask where theta is forced to 1.0 (optional).
        absorption_widths: PML absorption widths (x, y, z) in pixels.
        absorption_coeff: PML absorption coefficient.
        max_steps: Maximum FDTD timesteps per simulation (default: 10000).
        check_every_n: FDTD convergence check interval (default: 1000).
        gpu_type: GPU type (default: "B200").
        api_key: API key (overrides configured key).

    Yields:
        dict with keys:
            - step (int): Step number (1-indexed).
            - loss (float): Loss value at this step.
            - efficiency (float): Coupling efficiency (0 to 1) if mode loss is used.
            - theta_b64 (str): Base64-encoded updated theta array.
            - theta (np.ndarray): Updated design variables (decoded from theta_b64).
            - grad_max (float): Maximum absolute gradient value.
            - step_time (float): GPU time for this step in seconds.
            - is_final (bool): True on the last step.

    Raises:
        ValueError: If no loss specification is provided.

    Example:
        >>> results = []
        >>> try:
        ...     for step_result in hwc.run_optimization(
        ...         theta=theta_init,
        ...         source_field=source,
        ...         source_offset=source_offset,
        ...         freq_band=freq_band,
        ...         structure_spec=structure_spec,
        ...         loss_monitor_shape=(1, Ly, Lz),
        ...         loss_monitor_offset=(output_x, 0, 0),
        ...         design_monitor_shape=(Lx, Ly, h_etch),
        ...         design_monitor_offset=(0, 0, z_etch),
        ...         mode_field=mode_full,
        ...         input_power=input_power,
        ...         mode_cross_power=P_mode_cross,
        ...         num_steps=50,
        ...         learning_rate=0.01,
        ...         gpu_type="B200",
        ...     ):
        ...         eff = step_result['efficiency'] * 100
        ...         print(f"Step {step_result['step']}: {eff:.2f}%")
        ...         results.append(step_result)
        ... except KeyboardInterrupt:
        ...     print(f"Stopped after {len(results)} steps.")
    """
    import json

    effective_api_key = api_key or _API_CONFIG.get('api_key')
    if not effective_api_key:
        print("API key required to proceed.")
        print("Sign up for free at spinsphotonics.com to get your API key.")
        return

    if api_key and api_key != _API_CONFIG.get('api_key'):
        configure_api(api_key=api_key, validate=False)

    API_URL = _API_CONFIG['api_url']

    # Validate loss parameters
    if (loss_fn is None and mode_field is None and
            power_axis is None and intensity_component is None):
        raise ValueError(
            "Must provide at least one loss specification:\n"
            "  - loss_fn: Custom loss function\n"
            "  - mode_field: Mode coupling (with input_power, mode_cross_power)\n"
            "  - power_axis: Poynting vector power (0=x, 1=y, 2=z)\n"
            "  - intensity_component: Simple |E|^2 ('Ex', 'Ey', 'Ez')"
        )

    # Encode arrays
    theta_b64 = encode_array(np.array(theta, dtype=np.float32))
    source_field_b64 = encode_array(np.array(source_field))

    # Serialize custom loss function
    loss_fn_pickle_b64 = None
    if loss_fn is not None:
        import cloudpickle
        loss_fn_bytes = cloudpickle.dumps(loss_fn)
        loss_fn_pickle_b64 = base64.b64encode(loss_fn_bytes).decode('utf-8')

    # Mode coupling params
    mode_coupling_params = None
    if mode_field is not None:
        if input_power is None or mode_cross_power is None:
            raise ValueError(
                "mode_field requires input_power and mode_cross_power"
            )
        mode_coupling_params = {
            'mode_field_b64': encode_array(np.array(mode_field)),
            'mode_field_shape': list(np.array(mode_field).shape),
            'input_power': float(input_power),
            'mode_cross_power': float(mode_cross_power),
            'axis': int(mode_axis)
        }

    # Power params
    power_params = None
    if power_axis is not None:
        power_params = {
            'axis': int(power_axis),
            'maximize': power_maximize
        }

    # Intensity params
    intensity_params = None
    if intensity_component is not None:
        intensity_params = {
            'component': intensity_component,
            'maximize': intensity_maximize
        }

    # Waveguide mask
    waveguide_mask_b64 = None
    if waveguide_mask is not None:
        waveguide_mask_b64 = encode_array(np.array(waveguide_mask, dtype=np.float32))

    # Build request
    request_data = {
        "theta_b64": theta_b64,
        "theta_shape": list(np.array(theta).shape),
        "source_field_b64": source_field_b64,
        "source_field_shape": list(np.array(source_field).shape),
        "source_offset": list(source_offset),
        "freq_band": [float(x) for x in freq_band],
        "loss_monitor_shape": list(loss_monitor_shape),
        "loss_monitor_offset": list(loss_monitor_offset),
        "design_monitor_shape": list(design_monitor_shape),
        "design_monitor_offset": list(design_monitor_offset),
        "structure_spec": structure_spec,
        "mode_coupling_params": mode_coupling_params,
        "intensity_params": intensity_params,
        "power_params": power_params,
        "loss_fn_pickle_b64": loss_fn_pickle_b64,
        "waveguide_mask_b64": waveguide_mask_b64,
        "waveguide_mask_shape": list(np.array(waveguide_mask).shape) if waveguide_mask is not None else None,
        "optimizer_config": {
            "num_steps": num_steps,
            "learning_rate": learning_rate,
            "grad_clip_norm": grad_clip_norm,
            "cosine_decay_alpha": cosine_decay_alpha,
            "enforce_symmetry": enforce_symmetry,
        },
        "add_absorption": True,
        "absorption_widths": list(absorption_widths),
        "absorption_coeff": float(absorption_coeff),
        "max_steps": max_steps,
        "check_every_n": check_every_n,
        "gpu_type": gpu_type,
    }

    headers = {
        "X-API-Key": effective_api_key,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    print(f"\n=== Inverse Design Optimization ===")
    print(f"Endpoint: {API_URL}/inverse_design_stream")  # internal endpoint name
    print(f"Theta shape: {request_data['theta_shape']}")
    print(f"Steps: {num_steps}, LR: {learning_rate}, GPU: {gpu_type}")
    if mode_coupling_params:
        print(f"Loss: Mode coupling")
    elif power_params:
        axis_names = {0: 'x', 1: 'y', 2: 'z'}
        print(f"Loss: Power S_{axis_names.get(power_axis, power_axis)} (maximize={power_maximize})")
    elif intensity_params:
        print(f"Loss: Intensity |{intensity_component}|^2 (maximize={intensity_maximize})")
    elif loss_fn_pickle_b64:
        print(f"Loss: Custom function (cloudpickle)")
    print(f"====================================\n")

    response = None
    try:
        # SSE request with long read timeout (GPU steps take ~60-120s each)
        response = requests.post(
            f"{API_URL}/inverse_design_stream",
            json=request_data,
            headers=headers,
            stream=True,
            timeout=(30, None),  # 30s connect, unlimited read
        )
        response.raise_for_status()

        # Parse Server-Sent Events
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith('data: '):
                continue

            event_json = line[6:]  # strip 'data: ' prefix

            # End-of-stream sentinel
            if event_json == '[DONE]':
                break

            try:
                event = json.loads(event_json)
            except json.JSONDecodeError:
                continue

            # Handle error events from server
            if event.get('type') == 'error':
                print(f"\nServer error: {event.get('message', 'Unknown error')}")
                return

            # Decode theta array
            if 'theta_b64' in event:
                event['theta'] = decode_array(event['theta_b64'])

            yield event

    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            status_code = e.response.status_code
            response_text = e.response.text
            print(f"\n=== API Error ===")
            print(f"Status Code: {status_code}")
            print(f"Response: {response_text}")
            print(f"=================\n")
            if status_code == 401:
                print("No API key detected. Sign up at spinsphotonics.com")
            elif status_code == 403:
                print("Invalid API key. Verify at spinsphotonics.com/dashboard")
            elif status_code == 402:
                print("Insufficient credits. Add credits at spinsphotonics.com/billing")
        return

    except requests.exceptions.Timeout:
        print("Connection timeout. Please try again.")
        return

    except requests.exceptions.ConnectionError:
        print("Connection failed. Please check your network.")
        return

    except GeneratorExit:
        # Generator was closed (user interrupted or broke out of loop).
        # Closing the response drops the connection, signaling the server
        # to cancel the GPU task.
        pass

    finally:
        if response is not None:
            response.close()


# =============================================================================
# MODE CONVERTER - Cloud GPU version
# =============================================================================

def mode_convert(
    mode_E_field,
    freq_band,
    permittivity_slice,
    propagation_axis: str = 'x',
    propagation_length: int = 60,
    absorption_width: int = 20,
    absorption_coeff: float = 4.89e-3,
    simulation_steps: int = 5000,
    gpu_type: str = "B200",
    api_key: Optional[str] = None,
):
    """Convert E-only mode field to full E+H field via cloud GPU simulation.

    Cloud-accelerated version of mode_converter(). Runs FDTD on cloud GPU
    instead of local CPU. Uses the raw_arrays path in the early_stopping
    Modal function.

    Args:
        mode_E_field: E-field mode pattern from mode solver with shape
            (num_freqs, 3, 1, y, z) for x-propagation.
        freq_band: Frequency band (min, max, num_points).
        permittivity_slice: 2D permittivity slice (y, z) matching mode field.
        propagation_axis: Direction of mode propagation ('x' or 'y').
        propagation_length: Propagation distance in grid units (default: 60).
        absorption_width: Width of absorbing boundaries (default: 20).
        absorption_coeff: Absorption coefficient (default: 4.89e-3).
        simulation_steps: Maximum FDTD time steps (default: 5000).
        gpu_type: GPU type for cloud simulation (default: "B200").
        api_key: Optional API key override.

    Returns:
        Full mode field with shape (num_freqs, 6, 1, y, z) containing
        both E and H field components.
    """
    from . import absorption as hwa

    if propagation_axis not in ['x', 'y']:
        raise ValueError(f"propagation_axis must be 'x' or 'y', got '{propagation_axis}'")

    # Extract mode dimensions
    if propagation_axis == 'x':
        _, _, _, mode_y, mode_z = mode_E_field.shape
        mode_perp = mode_y
        mode_vert = mode_z
    else:
        _, _, mode_x, _, mode_z = mode_E_field.shape
        mode_perp = mode_x
        mode_vert = mode_z

    # Validate permittivity_slice dimensions
    perm_slice = np.asarray(permittivity_slice)
    if perm_slice.shape != (mode_perp, mode_vert):
        raise ValueError(
            f"permittivity_slice shape {perm_slice.shape} doesn't match "
            f"mode field dimensions ({mode_perp}, {mode_vert})"
        )

    # Build 3D permittivity (same logic as mode_converter)
    total_x = 2 * absorption_width + propagation_length
    total_x = total_x + (total_x % 2)  # Make even

    if propagation_axis == 'x':
        eps_2d = perm_slice[np.newaxis, :, :]  # (1, y, z)
        eps_2d = np.tile(eps_2d, (total_x, 1, 1))  # (x, y, z)
        eps = np.stack([eps_2d, eps_2d, eps_2d], axis=0)  # (3, x, y, z)
    else:
        eps_2d = perm_slice[:, np.newaxis, :]  # (x, 1, z)
        eps_2d = np.tile(eps_2d, (1, total_x, 1))  # (x, y, z)
        eps = np.stack([eps_2d, eps_2d, eps_2d], axis=0)  # (3, x, y, z)

    # Build conductivity with absorption
    cond = np.zeros_like(eps)
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
    cond = cond + np.asarray(absorption_mask)

    # Create source field (E + zeros for H)
    mode_E = np.asarray(mode_E_field)
    source_field = np.concatenate([mode_E, np.zeros_like(mode_E)], axis=1)

    # Source and monitor positions (same as mode_converter)
    if propagation_axis == 'x':
        source_offset = (absorption_width + 5, 0, 0)
        monitor_x = total_x - absorption_width - 10
        monitor_shape = [1, mode_perp, mode_vert]
        monitor_offset = [monitor_x, 0, 0]
    else:
        source_offset = (0, absorption_width + 5, 0)
        monitor_y = total_x - absorption_width - 10
        monitor_shape = [mode_perp, 1, mode_vert]
        monitor_offset = [0, monitor_y, 0]

    # Pack into raw_arrays recipe (uses existing early_stopping raw_arrays path)
    raw_recipe = {
        'raw_arrays': True,
        'permittivity': eps.astype(np.float32).tolist(),
        'conductivity': cond.astype(np.float32).tolist(),
        'metadata': {'final_shape': list(eps.shape)},
    }

    monitors_recipe = [
        {'name': 'Output_mode', 'shape': monitor_shape, 'offset': monitor_offset}
    ]

    print(f"Mode convert: grid {grid_shape}, prop_length={propagation_length}")

    # Call cloud simulation (add_absorption=False since we built it locally)
    result = simulate(
        structure_recipe=raw_recipe,
        source_field=source_field,
        source_offset=source_offset,
        freq_band=freq_band,
        monitors_recipe=monitors_recipe,
        simulation_steps=simulation_steps,
        add_absorption=False,
        absorption_widths=(0, 0, 0),
        absorption_coeff=0.0,
        api_key=api_key,
        gpu_type=gpu_type,
        convergence="default",
    )

    if result is None:
        raise RuntimeError("Cloud simulation failed for mode_convert")

    # Extract mode field from monitor
    mode_field = result['monitor_data'].get('Output_mode')
    if mode_field is None:
        raise RuntimeError("Output_mode monitor not found in simulation results")

    print(f"Mode convert complete: {mode_field.shape}")
    return mode_field
