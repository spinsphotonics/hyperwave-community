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
    view_structure,
    Layer,
    Structure,
)

# Import absorption functions
from .absorption import (
    create_absorption_mask,
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
    view_monitors,
)

# Import source functions
from .sources import (
    mode,
    create_mode_source,
    create_gaussian_source,
    visualize_gaussian_source,
)

# Import API client functions
from .api_client import (
    configure_api,
    generate_gaussian_source,
)

# Import metasurface utilities
from .metasurface import (
    create_circle_array,
    create_circle_grid,
)

# Import data I/O functions
from .data_io import (
    generate_gds_from_density,
    view_gds,
    gds_to_theta,
    component_to_theta,
)


# Import simulation utilities
from .simulate import (
    simulate,
    quick_view_monitors,
)

# Define public API
__all__ = [
    # Version
    "__version__",

    # Structure
    "density",
    "create_structure",
    "view_structure",
    "Layer",
    "Structure",

    # Absorption
    "create_absorption_mask",

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
    "view_monitors",

    # Sources
    "mode",
    "create_mode_source",
    "create_gaussian_source",
    "visualize_gaussian_source",
    "generate_gaussian_source",

    # Metasurface
    "create_circle_array",
    "create_circle_grid",

    # Data I/O
    "generate_gds_from_density",
    "view_gds",
    "gds_to_theta",
    "component_to_theta",

    # API
    "configure_api",

    # Simulation
    "simulate",
    "quick_view_monitors",
]
