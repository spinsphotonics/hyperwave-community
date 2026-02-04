# Hyperwave Community Package Summary

## Package Structure

```
hyperwave-community/
├── hyperwave_community/          # Main package
│   ├── __init__.py               # Package exports and API
│   ├── sources.py                # Mode solver + Gaussian source API wrapper
│   ├── structure.py              # Structure creation and density filtering
│   ├── absorption.py             # PML absorption boundaries
│   ├── monitors.py               # Field monitoring and power analysis
│   ├── metasurface.py            # Metasurface pattern utilities
│   ├── data_io.py                # GDS file import/export and visualization
│   └── api_client.py             # API communication layer
│
├── examples/                     # Example scripts
│   └── mode_source_waveguide.py  # Complete waveguide simulation example
│
├── setup.py                      # Package installation config
├── README.md                     # User documentation
├── LICENSE                       # MIT License
└── .gitignore                    # Git ignore patterns
```

## Module Breakdown

### sources.py (19 KB)
**Public Functions:**
- `mode(freq_band, permittivity, axis, mode_num, ...)` - Eigenvalue-based mode solver
- `create_mode_source(structure, freq_band, mode_num, ...)` - Generate modal sources for waveguides
- `create_gaussian_source(structure_shape, conductivity_boundary, ...)` - Generate Gaussian sources (API call)

**Classes:**
- `FreqBand` - NamedTuple for frequency band specification

**Execution:** Mode solver runs locally on CPU (~100ms-5s), Gaussian sources via API

### structure.py (53 KB)
**Public Functions:**
- `density(theta, radius, alpha, ...)` - Apply density filtering to design patterns
- `create_structure(layers, vertical_radius)` - Build 3D structures from 2D layers
- `reconstruct_structure_from_recipe(recipe)` - Deserialize structures
- `view_structure(structure, ...)` - Visualize permittivity and conductivity
- `view_density(density, cmap)` - Visualize 2D density patterns

**Classes:**
- `Layer` - Layer specification (density, permittivity, thickness)
- `Structure` - 3D photonic structure container

### absorption.py (16 KB)
**Public Functions:**
- `create_absorption_mask(grid_shape, absorption_widths, absorption_coeff, ...)` - Generate PML boundaries

**Execution:** Pure geometry operations, runs locally

### monitors.py (49 KB)
**Public Functions:**
- `S_from_slice(field_slice)` - Calculate Poynting vector from field data
- `power_from_a_box(field, Lx, Ly, Lz, ...)` - Calculate net power flux
- `get_field_slice(field, axis, position)` - Extract 2D field slices
- `get_power_through_plane(field, axis, position)` - Power through monitor plane
- `get_field_intensity(field)` - Calculate |E|² + |H|²
- `get_electric_field_intensity(field)` - Calculate |E|²
- `get_magnetic_field_intensity(field)` - Calculate |H|²
- `view_monitors(structure, monitors, ...)` - Visualize monitor positions
- `add_monitors_at_position(structure, axis, position, ...)` - Auto-place monitors with waveguide detection

**Classes:**
- `Monitor` - Monitor configuration dataclass
- `MonitorSet` - Container for managing multiple monitors

### metasurface.py (2.8 KB)
**Public Functions:**
- `create_circle_array(size, radius)` - Generate single circular pattern
- `create_circle_grid(radius, edge_separation, nx_circles, ny_circles, ...)` - Generate grid of circles

**Execution:** Pure geometry operations, runs locally

### data_io.py (30 KB)
**Public Functions:**
- `generate_gds_from_density(density_array, level, output_filename, ...)` - Export density to GDS II format
- `view_gds(gds_filepath, density_array, ...)` - Visualize GDS file with optional comparison
- `gds_to_theta(gds_filepath, resolution, layer, ...)` - Import GDS file to theta array
- `component_to_theta(component, resolution, layer, ...)` - Convert gdsfactory component to theta

**Dependencies:**
- gdstk (GDS file I/O)
- scikit-image (contour extraction)
- gdsfactory (optional, for component_to_theta)

**Execution:** File I/O and visualization, runs locally

### api_client.py (14 KB)
**Public Functions:**
- `configure_api(api_key, api_url)` - Set API credentials and endpoint
- `simulate(structure, source_field, monitors, ...)` - Run FDTD simulation on GPU
- `generate_gaussian_source(structure_shape, ...)` - Generate Gaussian source on GPU
- `encode_array(arr)` - Encode numpy array to base64
- `decode_array(b64_str)` - Decode base64 to numpy array

**Execution:** Network requests to GPU API endpoints

## What's NOT Included (Stays Private)

From `solve.py`:
- ❌ `multi_freq()` - Full FDTD solver
- ❌ `mem_efficient_multi_freq()` - Memory-efficient FDTD
- ❌ `gaussian_source()` - Gaussian with FDTD
- ❌ `time_domain()` - Core FDTD time-stepping
- ❌ `wave_equation_error()` - Wave equation error calculation

From `simulate.py`:
- ❌ `simulate()` - Main simulation wrapper (uses API instead)

From `simulate_modal.py` and `gaussian_source_modal.py`:
- ❌ All Modal GPU integration code

## API Endpoints Required

The community package expects these endpoints to exist:

### POST /simulate
**Request:**
- structure_recipe (JSON)
- source_field_b64 (base64 encoded numpy)
- source_offset, freq_band, monitors
- Simulation parameters

**Response:**
- monitor_data_b64 (dict of base64 arrays)
- convergence data
- powers, transmissions
- performance metrics

### POST /generate_gaussian_source
**Request:**
- structure_shape
- conductivity_boundary_b64
- freq_band, source_z_pos, polarization

**Response:**
- source_field_b64
- source_power, source_position
- timing metrics

## Docstring Format

All functions follow NumPy/Google style:

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """Brief one-line summary.

    Extended description providing more detail about what the
    function does and its purpose.

    Args:
        param1: Description of first parameter.
        param2: Description with default value noted.

    Returns:
        Description of return value.

    Raises:
        ValueError: When parameters are invalid.

    Note:
        Optional section for important notes.

    Example:
        >>> import hyperwave_community as hwc
        >>> result = function_name(...)
    """
```

## Installation

```bash
# From source
cd hyperwave-community
pip install -e .

# Or from PyPI (once published)
pip install hyperwave-community
```

## Usage Pattern

```python
import hyperwave_community as hwc
import os

# 1. Configure API
os.environ['HYPERWAVE_API_KEY'] = 'your-key'

# 2. Build structure locally
structure = hwc.create_structure(layers=[...])

# 3. Generate mode source locally (~100ms-5s)
source, offset, info = hwc.create_mode_source(...)

# 4. Setup monitors locally
monitors = hwc.MonitorSet()
monitors.add_monitors_at_position(...)

# 5. Run simulation via API (~25-70s)
results = hwc.simulate(
    structure=structure,
    source_field=source,
    source_offset=offset,
    monitors=monitors,
    ...
)
```

## Key Design Decisions

1. **Mode solver LOCAL**: Eigenvalue solve is fast enough (~100ms-5s) to run on CPU
2. **Gaussian source API**: Requires FDTD (~20-30s), must use GPU
3. **Structure building LOCAL**: Pure geometry operations, no compute needed
4. **GDS I/O LOCAL**: File conversion and visualization, no heavy compute
5. **Simulation API**: Heavy FDTD compute, requires H100 GPU

## Next Steps

1. Update API endpoint URLs in `api_client.py` (currently uses localhost)
2. Add Firebase authentication to backend endpoints
3. Test package installation: `pip install -e .`
4. Run example: `python examples/mode_source_waveguide.py`
5. Publish to PyPI (optional)
6. Update GitHub repository URL in setup.py and README
7. Add more examples (metasurface, optimization, etc.)

## Dependencies

**Required:**
- jax >= 0.4.0 (CPU-only sufficient)
- jaxlib >= 0.4.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- requests >= 2.26.0
- gdstk >= 0.9.0 (GDS file I/O)
- scikit-image >= 0.19.0 (contour extraction)

**Optional:**
- gdsfactory >= 7.0.0 (for component_to_theta)
  - Install with: `pip install hyperwave-community[gdsfactory]`

**Optional (dev):**
- pytest, pytest-cov
- black, flake8, mypy

## Testing

Create tests in `tests/` directory:
```bash
pytest tests/
```

Consider testing:
- Mode solver accuracy
- Structure creation
- API client serialization
- Monitor placement
- Error handling
