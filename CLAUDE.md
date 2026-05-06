# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Git Commit Standards
- **NEVER use Co-Authored-By tags in commit messages**
- **NEVER include emoji or "Generated with Claude Code" footer in commits**
- Keep commit messages clean and professional
- Follow conventional commit format: Brief title, optional detailed body

## CRITICAL: Never Drop Functions During Refactors
- **NEVER remove existing functions during refactoring**
- **NEVER replace sophisticated implementations with simplified versions**
- **ALWAYS preserve all helper functions, even if their purpose isn't immediately clear**
- **NEVER assume a function is "unnecessary" - it may handle critical edge cases**
- When refactoring: ADD new features, but KEEP all existing functionality
- If unsure about a function's purpose, ASK before removing it

## IMPORTANT: Avoid Overengineering
- Keep solutions simple and direct
- Don't create unnecessary abstractions or helper functions
- Provide minimal code that directly solves the problem
- Avoid excessive organization for simple tasks

## CRITICAL: Read and Update JOURNAL.md
- **ALWAYS read `JOURNAL.md` completely before starting work**
- It tracks design decisions, current state, and lessons learned
- **ALWAYS update the journal after completing ANY significant work**
- Document: what changed, which files, key decisions, test results
- Update BEFORE the session ends - context windows forget everything not written down
- Keep it concise (~100 lines max) - archive stale entries if needed

## CRITICAL: 1:1 Replica of Reference - No Unauthorized Deviations
- The notebook `examples/gc_inverse_design.ipynb` MUST be a 1:1 replica of the
  reference GC inverse design from the sibling hyperwave repo.
- **Template:** `hyperwave/devices/gc/template_57pct/gc_template_uniform_20260129_1343.py`
- **Full script:** `hyperwave/devices/gc/inverse_design/gc_inverse_design.py`
- **DO NOT randomly choose design decisions** that change the optimization workflow.
  Every aspect (loss function, gradient computation, optimizer, layer stack, source
  setup) must match the reference unless explicitly discussed and documented.
- The ONLY allowed differences are those required by the cloud-first architecture:
  1. `generate_gaussian_source()` runs on cloud GPU (not local solver)
  2. Structure recipe sent to cloud for reconstruction (not built locally)
  3. Cloud API handles FDTD execution
- Everything else (loss function, mode overlap, optimizer params, layer stack,
  density filtering, etc.) MUST match the reference exactly.

## CRITICAL: No Image Files
- **NEVER save images to the file system (no .png, .jpg, .pdf files)**
- Do not use plt.savefig() or any image export functions
- Only use plt.show() for displaying plots when needed
- All visualization should be display-only, not saved to disk

## Documentation Standards

### Docstring Format
Follow the NumPy/Google style docstrings with some modifications for consistency:

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """Brief one-line summary of the function.

    Optional extended description providing more detail about what the
    function does, its purpose, or important implementation notes.

    Args:
        param1: Description of first parameter. Include units if applicable.
        param2: Description with default value noted. Can span multiple
            lines if needed with proper indentation.

    Returns:
        Description of return value. Include shape for arrays.

    Raises:
        ValueError: When parameters are invalid.
        TypeError: When incorrect types are provided.

    Note:
        Optional section for important notes or warnings.
    """
```

### Module Docstrings
```python
"""Brief module description.

Extended description of module purpose and functionality.
List main components and their relationships.

Main functions:
    function1: Brief description
    function2: Brief description
"""
```

### Key Guidelines:
- Start with brief one-line summary
- Args section lists parameters with types in signature, not docstring
- Use descriptive parameter names
- Include units in descriptions (e.g., "in pixels", "in nm")
- Mention array shapes explicitly
- Private functions (starting with _) should have simpler docstrings
- Keep descriptions concise but complete

## Project Overview
Hyperwave is a photonics simulation and optimization library built on JAX for GPU-accelerated electromagnetics simulations. It implements finite-difference time-domain (FDTD) methods for solving Maxwell's equations.

## Key Modules
- `hyperwave.solve`: Core FDTD solver with multi-frequency support
- `hyperwave.simulate`: High-level simulation interfaces
- `hyperwave.monitors`: Field monitoring and power flow calculations
- `hyperwave.structure`: Material structure definitions (permittivity, conductivity)
- `hyperwave.sources`: Field source configurations
- `hyperwave.metasurface`: Metasurface design utilities
- `hyperwave.absorption`: Absorbing boundary conditions

## Development Commands
```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_solve.py

# Run with verbose output
pytest -v
```

## Working with Notebooks
- Do NOT execute notebook files directly
- When creating code solutions, provide them as Python files in `notebooks/temp/` rather than just text output
- Provide code for the user to copy into their notebook cells
- When creating helper scripts, place them in `notebooks/temp/`
- Use the mcp__ide__executeCode tool for running Python code in Jupyter kernels

## Test Organization
When creating tests:
- **NEVER dump test files in the root directory**
- Create test files in appropriate subdirectories:
  - `tests/` for permanent test files
  - `notebooks/temp/` for temporary test scripts
  - Create a topic-specific subdirectory for multiple related test files (e.g., `tests/monitor_tests/`)
- Keep all related test scripts together in their directory
- Name directories and files clearly to indicate what they test
- Clean up previous test iterations - don't accumulate multiple versions
- **Use GDS factory components for test structures**: When testing with photonic devices, extract components from gdsfactory (couplers, rings, MZIs, etc.) rather than creating random patterns. Use the `component_to_theta()` function in `notebooks/temp/gdsfactory_to_theta.py` or `gds_to_theta.py` for extraction

## Power Calculation Note
For accurate power calculations through monitors, use Poynting flux:
- Use `hyperwave.monitors.S_from_slice()` to compute Poynting vectors
- Integrate flux over monitor surfaces for total power
- Example implementation in `notebooks/temp/multifreq_power_clean.py`

## Monitor Positioning
For automatic monitor positioning on waveguides, use algorithmic detection:
- Analyze permittivity distribution to find waveguide centers
- Use threshold-based detection to identify high-permittivity regions (silicon waveguides)
- Example implementation in `notebooks/temp/directional_coupler_monitors.py`

### Waveguide Detection Algorithm
```python
def find_waveguide_positions(structure, x_position=None):
    """Algorithmically find waveguide positions and widths by analyzing permittivity."""
    eps_array = structure.permittivity[0]  # Remove frequency dimension
    
    # Get Y-slice at specified X position and middle Z
    y_slice = eps_array[x_position, :, z_dim // 2]  # Shape: (y_dim,)
    
    # Find high permittivity regions (waveguides)
    threshold = (jnp.max(y_slice) + jnp.min(y_slice)) / 2
    high_eps_mask = y_slice > threshold
    
    # Find connected regions and calculate centers + widths
    waveguide_info = []
    in_waveguide = False
    wg_start = 0
    
    for y in range(len(high_eps_mask)):
        if high_eps_mask[y] and not in_waveguide:
            wg_start = y
            in_waveguide = True
        elif not high_eps_mask[y] and in_waveguide:
            wg_end = y - 1
            wg_center = (wg_start + wg_end) // 2
            wg_width = wg_end - wg_start + 1
            waveguide_info.append({'center': wg_center, 'width': wg_width})
            in_waveguide = False
    
    return sorted(waveguide_info, key=lambda x: x['center'])

# Calculate adaptive monitor width based on waveguide width
avg_wg_width = (top_wg['width'] + bottom_wg['width']) // 2
monitor_width = int(avg_wg_width * 2.5)  # 2.5x waveguide width for good coverage
```

**Key Features:**
- Returns both waveguide center positions and widths
- Automatically calculates optimal monitor width (2.5× average waveguide width)
- Adapts to any waveguide geometry and structure size
- Monitor offsets use corner-based positioning (all coordinates are absolute from origin)

## Optimization Utilities
For inverse design and optimization tasks:
- `notebooks/temp/optimization_with_region_fixed.py`: Beam splitter optimization with design regions
- Uses JAX autodiff for gradient computation
- Implements live plotting during optimization

## Core Development Philosophy

### KISS (Keep It Simple, Stupid)

Simplicity should be a key goal in design. Choose straightforward solutions over complex ones whenever possible. Simple solutions are easier to understand, maintain, and debug.

### YAGNI (You Aren't Gonna Need It)

Avoid building functionality on speculation. Implement features only when they are needed, not when you anticipate they might be useful in the future.

## Reference Implementation

The canonical reference implementation for the Hyperwave workflow is available in:
- **`notebooks/active/claude_md_reference.py`** - Complete working example following the standard workflow
- **`notebooks/active/simulate_on_modal_fixed.py`** - Generalized function for running simulations on Modal GPUs
- **`notebooks/active/test_modal_fixed.py`** - Verified Modal simulation pipeline with all monitors

This file demonstrates the complete 7-step workflow for using the new Hyperwave functions:

1. **Creating Theta (Design Pattern)** - Binary pattern creation for waveguide structures
2. **Apply Density Filtering** - Smoothing and binarization of design patterns
3. **Build Layer Structure** - Creating 3D structures from 2D layers
4. **Add Absorbing Boundaries** - PML boundary conditions
5. **Configure Monitors** - Field monitoring setup
6. **Create Mode Source and Simulate** - Mode source creation and FDTD simulation
7. **Power Calculation and Transmission** - Poynting flux analysis

See the file for the exact implementation details and parameters.

### Verified Modal Simulation Pipeline

**Files Used:**
- **`notebooks/active/test_modal_fixed.py`** - Main test script that runs the complete simulation pipeline
- **`notebooks/active/simulate_on_modal_fixed.py`** - Modal integration module (imported by test script)
- **`hyperwave/structure.py`** - Contains critical `reconstruct_structure_from_recipe()` function for Modal serialization

**How to Run:**
```bash
cd /Users/jq4386/Github/SPINS/hyperwave/notebooks/active
python test_modal_fixed.py
```

**Output Files Generated:**
- `modal_test_output/monitor_positions.png` - Visualization of monitor placement in structure
- `modal_test_output/simulation_results/monitor_fields_all.png` - 2×2 grid showing all 4 monitors
- `modal_test_output/simulation_results/convergence.png` - Convergence history
- `modal_test_output/simulation_results/transmission.png` - Transmission spectrum

The file **`notebooks/active/test_modal_fixed.py`** contains the complete verified pipeline for running reference structure simulations on Modal H100 GPUs with full monitor visualization.

**Key Implementation Details:**

```python
# 1. Create reference structure (500x1000 theta, 40-pixel waveguide)
theta = jnp.zeros((500, 1000))
center_y = theta.shape[0] // 2
waveguide_width = 40
strip_start = center_y - waveguide_width // 2
strip_end = center_y + waveguide_width // 2
theta = theta.at[strip_start:strip_end, :].set(1.0)

# 2. Density filtering
jax_density = hwst.density(theta=theta, pad_width=0, alpha=0, radius=8)

# 3. Layer structure (SiO2/Si/SiO2 stack)
n_Si, n_SiO = 3.4, 1.45
p_Si, p_SiO = n_Si ** 2, n_SiO ** 2

jax_waveguide = hwst.Layer(
    density_pattern=jax_density,
    permittivity_values=(p_SiO, p_Si),
    layer_thickness=20
)

jax_silica = hwst.Layer(
    density_pattern=jax_density,
    permittivity_values=p_SiO,
    layer_thickness=40
)

jax_structure = hwst.create_structure(
    layers=[jax_silica, jax_waveguide, jax_silica],
    vertical_radius=2
)

# 4. Add absorbing boundaries (locally before Modal)
_, Lx, Ly, Lz = jax_structure.permittivity.shape
abs_width = 70
abs_coeff = 4.89e-3
abs_shape = (abs_width, abs_width//2, abs_width//4)

jax_boundary = hwa.create_absorption_mask(
    grid_shape=(Lx, Ly, Lz),
    absorption_widths=abs_shape,
    absorption_coeff=abs_coeff
)

jax_structure.conductivity = jax_structure.conductivity + jax_boundary

# 5. Configure monitors (Input, Output, xy_mid, xz_mid)
monitors = hwm.MonitorSet()
monitors.add_monitors_at_position(structure=jax_structure, axis="x", position=100, label="Input")
monitors.add_monitors_at_position(structure=jax_structure, axis="x", position=400, label="Output")

# 6. Create mode source (after absorber region)
freq_band = (2 * jnp.pi / 32, 2 * jnp.pi / 30, 2)
source_pos_x = abs_shape[0] + 10  # 80 pixels (70 + 10)

source_field, source_offset, mode_info = hwsim.create_mode_source(
    structure=jax_structure,
    freq_band=freq_band,
    mode_num=0,
    propagation_axis="x",
    source_position=source_pos_x,
    perpendicular_bounds=(0, Ly),
    visualize=False
)

# 7. Run on Modal with absorption
results = simulate_on_modal(
    structure=jax_structure,
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors=monitors,
    mode_info=mode_info,
    max_steps=10000,
    check_every_n=1000,
    source_ramp_periods=5.0,
    add_absorption=True,  # Modal adds additional absorbers
    absorption_widths=(70, 35, 17),
    absorption_coeff=4.89e-3
)
```

**Expected Results:**
- GPU Time: ~24s on Modal H100
- Performance: ~5.3 billion grid-points×steps/s
- Transmission: 0.9988 (average over 2 frequencies)
- Monitor shapes:
  - Input/Output: (2, 6, 5, 52, 52)
  - xy_mid: (2, 6, 500, 250, 1)
  - xz_mid: (2, 6, 500, 1, 100)

**Visualization:**
The script generates a 2×2 grid visualization showing all 4 monitors with total field intensity:

```python
# Visualize all 4 monitors in single plot
for i, name in enumerate(monitor_names):
    idx = results['monitor_names'][name]
    monitor_data = results['monitor_data'][idx]

    # Calculate total field intensity (|E|²+|H|²)
    E_fields = monitor_data[0, 0:3, :, :, :]
    H_fields = monitor_data[0, 3:6, :, :, :]
    field_3d = jnp.sqrt(jnp.sum(jnp.abs(E_fields)**2, axis=0) +
                       jnp.sum(jnp.abs(H_fields)**2, axis=0))

    # Handle different monitor orientations (X, Y, or Z perpendicular)
    # Average or squeeze singleton/small dimensions
```

**Critical Notes:**
1. Source position must be after absorber: `source_pos_x = abs_width + 10`
2. Absorbers applied both locally and on Modal for proper boundary conditions
3. Monitor positions: Input at 100, Output at 400 (within simulation domain)
4. All monitor field dimensions follow format: (N_freq, 6, x, y, z)

### Key Patterns from Reference Implementation

1. **Structure Building**: Always start with theta → density → layers → 3D structure
2. **Monitor Setup**: Use MonitorSet() for organized monitor management
3. **Source Creation**: Use create_mode_source with perpendicular_bounds for waveguide modes
4. **Power Analysis**: Average monitor fields across thickness, then use S_from_slice()
5. **Visualization**: Use built-in visualization flags (visualize=True) during development
6. **Absorption**: Add PML boundaries after structure creation by modifying conductivity

### Important Parameters

- **Density Filtering**: `radius=8` for smoothing, `alpha=0` for no binarization
- **Vertical Blur**: `vertical_radius=2` for z-direction smoothing in structures
- **Absorption**: Width typically 70 pixels, coefficient ~4.89e-3
- **Source Ramp**: `source_ramp_periods=5.0` for gradual turn-on
- **Convergence Check**: `check_every_n=1000` for periodic error monitoring

## FastAPI Simulation Service

**Production API for serving Modal GPU simulations to multiple users.**

### Files
- **Server**: `hyperwave/api/main_binary.py` - FastAPI endpoint with binary serialization
- **Client**: `notebooks/active/run_api_simulation_binary.py` - Example client for `/simulate`
- **Client**: `notebooks/active/test_gaussian_api.py` - Example client for `/generate_gaussian_source`
- **Modal Backend**: `hyperwave/simulate_modal.py` - Modal GPU execution for simulations
- **Modal Backend**: `hyperwave/gaussian_source_modal.py` - Modal GPU execution for Gaussian sources

### Starting the API Server
```bash
cd /Users/jq4386/Github/SPINS/hyperwave
uvicorn hyperwave.api.main_binary:app --host 0.0.0.0 --port 8000
```

### API Endpoints

The API provides two main endpoints:
- **`POST /simulate`** - Run FDTD simulation on Modal GPU
- **`POST /generate_gaussian_source`** - Generate unidirectional Gaussian source on Modal GPU

### API Pipeline: `/simulate` Endpoint

**Client Side (Local):**
1. Create structure using standard Hyperwave workflow
2. Extract structure recipe: `structure.extract_recipe()` (~8MB JSON)
3. Create source field and monitors
4. Encode arrays to base64 for transmission
5. POST request to `/simulate` endpoint

**Server Side (FastAPI):**
1. Receive JSON request with base64-encoded data
2. Decode arrays and fix JSON serialization issues (tuples→lists)
3. Call `simulate_on_modal()` from `hyperwave.simulate_modal` module
4. Submit to Modal GPU cluster

**Modal Side (H100 GPU):**
1. **Container startup**: ~1.5s
2. **Structure reconstruction**: ~13.5s (JAX JIT compilation)
3. **Add absorption boundaries**: ~4.0s
4. **Setup monitors**: ~0.8s
5. **FDTD simulation**: ~24.0s
6. **Power analysis**: ~3.9s
7. Return results to FastAPI server

**Client Side (Local):**
1. Receive results from API
2. Decode base64-encoded monitor data
3. Process and visualize results

### API Pipeline: `/generate_gaussian_source` Endpoint

**Client Side (Local):**
1. Create structure shape and conductivity boundaries (absorption mask)
2. Define frequency band and source parameters
3. Encode conductivity boundary to base64 for transmission
4. POST request to `/generate_gaussian_source` endpoint

**Server Side (FastAPI):**
1. Receive JSON request with base64-encoded conductivity boundary
2. Decode arrays and validate parameters
3. Call `generate_gaussian_source_on_modal()` from `hyperwave.gaussian_source_modal` module
4. Submit to Modal GPU cluster

**Modal Side (H100 GPU):**
1. **Container startup**: ~1.5s
2. **Create initial Gaussian source**: ~2.0s
3. **Run FDTD in free space**: ~18.0s (wave equation error method)
4. **Calculate wave equation error**: ~5.0s
5. **Process source field** (swap E↔H, conjugate): ~1.0s
6. **Calculate source power**: ~0.5s
7. Return source field to FastAPI server

**Client Side (Local):**
1. Receive results from API
2. Decode base64-encoded source field
3. Use source field in simulations or visualize

**Key Difference:**
- `/simulate`: Runs full FDTD simulation with arbitrary source → returns monitor data
- `/generate_gaussian_source`: Creates unidirectional Gaussian source → returns source field for later use

**GPU Selection:**
Both endpoints support dynamic GPU selection via the `gpu_type` parameter:
- `"H100"` (default) - Fastest performance (~5.3 billion grid-points×steps/s)
- `"A100"` - Good balance (~4.1 billion grid-points×steps/s)
- `"T4"` - Budget option (~1.3 billion grid-points×steps/s)
- Other Modal GPU types: `"A10G"`, `"L4"`, etc.

All GPU types produce identical physical results (transmission ~0.9988 for reference structure).

### Performance

**Cold Start (First Request or After Idle):**
- Total time: ~68s
- Modal GPU time: ~24s
- Overhead: ~44s (container boot + JIT compilation + network)

**Warm Container (Subsequent Requests):**
- Not implemented (would require `keep_warm` parameter)
- Cold start overhead unavoidable in serverless architecture
- Each new concurrent user triggers new container = new cold start

**Breakdown:**
- Network transfer (8MB recipe): ~5s
- Modal cold start: ~5s
- JAX JIT compilation: ~13.5s (unavoidable first time per container)
- Structure reconstruction: included in JIT time
- GPU simulation: ~24s
- Result transfer: ~5s

### Key Design Decisions

1. **Recipe vs Raw Arrays**: Send 8MB recipe instead of 300MB raw permittivity/conductivity arrays
2. **Binary Serialization**: Base64-encoded numpy arrays for efficient transport
3. **JSON-based**: Portable, debuggable, standard REST API
4. **Serverless Architecture**: Modal handles GPU provisioning, no infrastructure management
5. **Accept Variable Performance**: 24s (warm) to 68s (cold) is acceptable for production use

### API Request Format

```python
{
    "structure_recipe": {...},  # 8MB JSON with construction instructions
    "source_field_b64": "...",  # Base64-encoded numpy array
    "source_field_shape": [2, 6, 1, 250, 100],
    "source_offset": [80, 0, 0],
    "freq_band": [0.196, 0.209, 2],
    "monitors": {...},
    "mode_info": {...},
    "max_steps": 10000,
    "check_every_n": 1000,
    "source_ramp_periods": 5.0,
    "add_absorption": true,
    "absorption_widths": [70, 35, 17],
    "absorption_coeff": 0.00489,
    "gpu_type": "H100"  # Optional: "H100" (default), "A100", "A10G", "L4", etc.
}
```

### API Response Format

```python
{
    "monitor_data_b64": {"Input": "...", "Output": "...", ...},
    "monitor_data_shapes": {"Input": [2, 6, 5, 52, 52], ...},
    "monitor_names": {"Input": 0, "Output": 1, ...},
    "convergence_steps": "...",
    "convergence_errors": {...},
    "performance": 5.26e9,
    "powers": {"Input": "...", "Output": "..."},
    "transmissions": {"transmission": "..."},
    "sim_time": 24.36,
    "gpu_type": "H100"
}
```

### Usage Example

```python
# Client code
import requests
import base64
import numpy as np

# Create structure, source, monitors (standard workflow)
structure = create_structure(...)
source_field, source_offset, mode_info = create_mode_source(...)
monitors = MonitorSet()

# Prepare request
request = {
    "structure_recipe": structure.extract_recipe(),
    "source_field_b64": base64.b64encode(pickle.dumps(source_field)).decode(),
    "source_offset": list(source_offset),
    "freq_band": list(freq_band),
    "monitors": serialize_monitors(monitors),
    ...
}

# Send to API
response = requests.post("http://localhost:8000/simulate", json=request)
results = response.json()

# Process results
transmission = pickle.loads(base64.b64decode(results['transmissions']['transmission']))
print(f"Average transmission: {transmission.mean():.4f}")
```

## Hyperwave Module Functions Reference

This section provides a comprehensive list of all public functions and classes in the hyperwave modules to avoid redundant function definitions.

### hyperwave.solve

#### Core Simulation Functions

- `multi_freq(freq_band, permittivity, conductivity, source_field, source_offset, source_ramp_periods=10.0, max_steps=5000, check_every_n=200, max_courant_factor=0.99)` - Solves Maxwell's equations for multiple frequencies (full domain storage - memory intensive)
- `mem_efficient_multi_freq(freq_band, permittivity, conductivity, source_field, source_offset, monitors, source_ramp_periods=10.0, max_steps=5000, check_every_n=200, max_courant_factor=0.99, convergence_threshold=1e-6)` - Memory-efficient multi-frequency solver storing only monitor volumes
- `mem_efficient_multi_freq_single_monitor(...)` - Backward compatibility wrapper for single monitor usage
- `mode(freq_band, permittivity, axis, mode_num, random_seed=0, min_modes_in_solve=10)` - Solve for propagating modes
- `gaussian_source(sim_shape, freq_band, source_pos=(0,0,0), r_waist=5.2, theta=0.0, phi=0.0, x_span=1.0, y_span=1.0, dz=0.08, permittivity=None, conductivity=None, max_steps=5000, check_every_n=200)` - Create Gaussian source field
- `time_domain(dt, permittivity, conductivity, source_field, source_waveform, source_offset, output_shapes, output_offsets, output_steps, field)` - Execute FDTD simulation (core accelerated function)
- `wave_equation_error(field, freq_band, permittivity, conductivity, source_field, source_offset)` - Compute wave equation error
- `wave_equation_error_full(...)` - Returns full error tensor instead of scalar
- `even_slice(theta)` - Return theta with last two axes trimmed to even lengths

#### Data Classes
- `Monitor` - Monitor configuration for field extraction (shape, offset)
- `FreqBand` - Describes frequency band (start, stop, num)

### hyperwave.simulate

#### Main Simulation Interface
- `simulate(structure, source_field, source_offset, freq_band, max_steps=10000, monitor_positions=None, visualize=False, field_to_plot='all', check_every_n=1000, source_ramp_periods=5.0)` - Run FDTD simulation
- `create_mode_source(structure, freq_band, mode_num=0, source_x_position=10, visualize=False)` - Create modal source
- `create_gaussian_source_wrapper(structure, freq_band, source_pos=(0,0,0), r_waist=5.2, theta=0.0, phi=0.0, dz=0.08, max_steps=5000, check_every_n=1000)` - Create Gaussian source

#### Visualization Functions
- `visualize_convergence(steps, errs, figsize=(12, 5))` - Enhanced convergence visualization
- `visualize_fields(out_list, monitor_mapping, field_component='all', freq_idx=0)` - Enhanced field visualization
- `visualize_mode(mode_field, beta, mode_num)` - Mode field visualization
- `visualize_monitor_outputs(results, monitor_names=None, field_component='all', freq_idx=0)` - Visualize monitor fields

### hyperwave.monitors

#### Core Monitor Functions
- `S_from_slice(field_slice)` - Calculate Poynting vector from field slice
- `power_from_a_box(field, Lx, Ly, Lz, Lx_total, Ly_total, Lz_total)` - Calculate net power out of box
- `get_field_slice(field, axis, position)` - Extract 2D slice from 3D field
- `get_power_through_plane(field, axis, position)` - Calculate power through plane
- `get_field_intensity(field)` - Calculate |E|²+|H|²
- `get_electric_field_intensity(field)` - Calculate |E|²
- `get_magnetic_field_intensity(field)` - Calculate |H|²
- `view_monitors(structure, monitors, monitor_mapping=None, ...)` - Visualize monitor positions
- `add_monitors_at_position(structure, axis, position, label="", ...)` - Add monitors with auto waveguide detection

#### Classes
- `Monitor` - Monitor configuration dataclass (shape, offset)
- `MonitorSet` - Container for managing multiple monitors with methods: `add()`, `list_monitors()`, `add_monitors_at_position()`

### hyperwave.structure

#### Core Structure Functions
- `density(theta, radius=2, boundary=0)` - Apply density filtering to optimization variables
- `create_structure(layers, vertical_radius=5.0)` - Create 3D structure from layers
- `reconstruct_structure_from_recipe(recipe)` - Reconstruct from saved recipe
- `view_density(d, cmap="PuOr")` - Visualize 2D density
- `view_structure(structure, show_permittivity=True, show_conductivity=True, ...)` - Visualize 3D structure

#### Classes
- `Layer` - Single layer (density_pattern, permittivity_values, layer_thickness, conductivity_values)
- `Structure` - 3D photonic structure (permittivity, conductivity, recipe)

### hyperwave.sources
- `create_gaussian_source(sim_shape, freq_band, source_pos=(10,0,0), r_waist=5.2, theta=0.0, phi=0.0, ...)` - Create Gaussian beam
- `create_mode_source(structure, freq_band, mode_num=0, source_x_position=10)` - Create modal source

### hyperwave.metasurface
- `create_circle_array(size, radius)` - Create single circle pattern
- `create_circle_grid(grid_size, num_cells, radii, pitch=None)` - Create grid of circles

### hyperwave.absorption
- `create_absorption_mask(shape, width=None, smoothness=10.0, width_x=None, width_y=None, width_z=None)` - Create PML absorption mask

### hyperwave.simulate_modal
- `simulate_on_modal(structure, source_field, source_offset, freq_band, monitors, mode_info=None, max_steps=10000, check_every_n=1000, source_ramp_periods=5.0, add_absorption=True, absorption_widths=(70, 35, 20), absorption_coeff=4.89e-3, gpu_type="H100")` - Run FDTD simulation on Modal GPU with dynamic GPU type selection (H100, A100, A10G, L4, etc.)

### hyperwave.gaussian_source_modal
- `generate_gaussian_source_on_modal(structure_shape, conductivity_boundary, freq_band, source_z_pos, polarization='x', max_steps=5000, check_every_n=1000, gpu_type="H100")` - Generate unidirectional Gaussian source on Modal GPU using wave equation error method

### Important Function Notes
1. **Offset Convention**: All functions use CORNER-based offsets for source/monitor positioning
2. **Memory Efficiency**: Use `mem_efficient_multi_freq` instead of `multi_freq` for practical simulations
3. **Field Format**: Arrays use format (N_freq, 6, x, y, z) where 6=[Ex, Ey, Ez, Hx, Hy, Hz]
4. **Frequency**: Uses angular frequency (omega) not regular frequency
5. **JAX Arrays**: All numerical operations use JAX arrays for GPU acceleration
6. **Modal GPU Execution**: Use `simulate_on_modal()` or API endpoints for GPU-accelerated simulations on Modal infrastructure
7. **Dynamic GPU Selection**: All Modal functions support runtime GPU selection (H100, A100, T4, A10G, L4, etc.) via `gpu_type` parameter - verified to produce identical physical results across all GPU types