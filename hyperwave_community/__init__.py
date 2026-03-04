"""Hyperwave Community: Open-source photonics simulation toolkit.

This package provides tools for designing and simulating photonic structures with
GPU-accelerated FDTD via API.

Modules:
    structure: Structure creation and density filtering
    absorption: Adiabatic absorbing boundaries
    monitors: Field monitoring and power analysis
    sources: Mode and Gaussian source generation (local + API)
    simulate: FDTD simulation via API and result visualization
    metasurface: Metasurface pattern utilities
    data_io: GDS file import/export and visualization
    api_client: API configuration and authentication
    visualization: Plotting functions for structures, fields, and monitors
    _logging: Logging configuration

Quick Start:
    >>> import hyperwave_community as hwc
    >>> import jax.numpy as jnp
    >>>
    >>> # Configure API
    >>> hwc.configure_api(api_key='your-key-here')
    >>>
    >>> # Create structure
    >>> theta = jnp.ones((500, 1000))
    >>> density = hwc.density(theta, radius=8)
    >>> layer = hwc.Layer(density, permittivity_values=(1.0, 11.56), layer_thickness=20)
    >>> structure = hwc.create_structure(layers=[layer])
    >>>
    >>> # Create source
    >>> freq_band = (2*jnp.pi/1.6, 2*jnp.pi/1.5, 2)
    >>> source, offset, mode_info = hwc.create_mode_source(
    ...     structure, freq_band, mode_num=0, propagation_axis='x', source_position=80
    ... )
    >>>
    >>> # Setup monitors
    >>> monitors = hwc.MonitorSet()
    >>> monitors.add_monitors_at_position(structure, axis='x', position=100, label='Input')
    >>>
    >>> # Run simulation
    >>> results = hwc.simulate(
    ...     structure=structure,
    ...     source_field=source,
    ...     source_offset=offset,
    ...     freq_band=freq_band,
    ...     monitors=monitors
    ... )
"""

__version__ = "0.1.0"

# Import core structure functions
from .structure import (
    density,
    create_structure,
    Layer,
    Structure,
    recipe_from_params,
)

# Import absorption functions
from .absorption import (
    absorber_params,
    create_absorption_mask,
    get_optimized_absorber_params,
)

# Import monitor functions
from .monitors import (
    Monitor,
    MonitorSet,
    S_from_slice,
    power_from_a_box,
    get_field_slice,
    get_power_through_plane,
    get_field_intensity,
    get_electric_field_intensity,
    get_magnetic_field_intensity,
)

# Import source functions
from .sources import (
    generate_gaussian_source,
)
# create_mode_source is in simulate.py (uses mode solver)
from .simulate import create_mode_source

# Import visualization functions
from .visualization import (
    plot_convergence,
    plot_fields,
    plot_mode,
    plot_monitors,
    plot_monitor_layout,
    plot_absorption_mask,
    plot_structure,
    plot_simulation_overview,
    plot_structure_3d,
    plot_gds,
)

# Import logging configuration
from ._logging import set_verbose, set_debug

# Import API client functions (SDK-style interface)
from .api_client import (
    # Configuration
    configure_api,
    get_account_info,
    estimate_cost,
    # CPU Steps (free)
    build_recipe,
    build_monitors,
    solve_mode_source,
    compute_freq_band,
    get_default_absorber_params,
    # GPU Step (uses credits)
    run_simulation,
    # Analysis functions (local, free)
    analyze_transmission,
    get_field_intensity_2d,
    compute_poynting_vector,
    compute_monitor_power,
    # Visualization functions
    visualize_structure,
    visualize_mode_source,
    # Utility functions
    encode_array,
    decode_array,
    # Convergence configuration
    ConvergenceConfig,
    CONVERGENCE_PRESETS,
    # Inverse design
    compute_adjoint_gradient,
    run_optimization,
    # Component preview functions
    list_components,
    get_component_params,
    preview_component,
)

# Import metasurface utilities
from .metasurface import (
    create_circle_array,
    create_circle_grid,
)

# Import data I/O functions
from .data_io import (
    generate_gds_from_density,
    gds_to_theta,
    component_to_theta,
)

# Import simulate from api_client (the cloud GPU version)
from .api_client import simulate, mode_convert  # noqa: F401


# Deprecation shims for renamed functions
def view_structure(*args, **kwargs):
    import warnings
    warnings.warn("view_structure is deprecated, use plot_structure", DeprecationWarning, stacklevel=2)
    from .visualization import plot_structure
    return plot_structure(*args, **kwargs)


def view_monitors(*args, **kwargs):
    import warnings
    warnings.warn("view_monitors is deprecated, use plot_monitor_layout", DeprecationWarning, stacklevel=2)
    from .visualization import plot_monitor_layout
    return plot_monitor_layout(*args, **kwargs)


def view_gds(*args, **kwargs):
    import warnings
    warnings.warn("view_gds is deprecated, use plot_gds", DeprecationWarning, stacklevel=2)
    from .visualization import plot_gds
    return plot_gds(*args, **kwargs)


# Define public API
__all__ = [
    # Version
    "__version__",

    # Structure
    "density",
    "create_structure",
    "Layer",
    "Structure",
    "recipe_from_params",

    # Absorption
    "absorber_params",
    "create_absorption_mask",
    "get_optimized_absorber_params",

    # Monitors
    "Monitor",
    "MonitorSet",
    "S_from_slice",
    "power_from_a_box",
    "get_field_slice",
    "get_power_through_plane",
    "get_field_intensity",
    "get_electric_field_intensity",
    "get_magnetic_field_intensity",

    # Sources
    "create_mode_source",
    "generate_gaussian_source",

    # Visualization
    "plot_convergence",
    "plot_fields",
    "plot_mode",
    "plot_monitors",
    "plot_monitor_layout",
    "plot_absorption_mask",
    "plot_structure",
    "plot_simulation_overview",
    "plot_structure_3d",
    "plot_gds",

    # Logging
    "set_verbose",
    "set_debug",

    # Metasurface
    "create_circle_array",
    "create_circle_grid",

    # Data I/O
    "generate_gds_from_density",
    "gds_to_theta",
    "component_to_theta",

    # API - Configuration
    "configure_api",
    "get_account_info",
    "estimate_cost",
    "compute_adjoint_gradient",

    # API - CPU Steps (free)
    "build_recipe",
    "build_monitors",
    "solve_mode_source",
    "compute_freq_band",
    "get_default_absorber_params",

    # API - GPU Step (uses credits)
    "run_simulation",
    "run_optimization",

    # API - Analysis (local, free)
    "analyze_transmission",

    # Analysis functions (local)
    "compute_poynting_vector",
    "compute_monitor_power",
    "get_field_intensity_2d",

    # Visualization functions (API)
    "visualize_structure",
    "visualize_mode_source",

    # Utility functions
    "encode_array",
    "decode_array",

    # Convergence configuration
    "ConvergenceConfig",
    "CONVERGENCE_PRESETS",

    # Component preview functions
    "list_components",
    "get_component_params",
    "preview_component",

    # Mode conversion (cloud GPU)
    "mode_convert",

    # Deprecated (use plot_* equivalents)
    "view_structure",
    "view_monitors",
    "view_gds",
]
