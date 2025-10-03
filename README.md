# Hyperwave Community

Open-source photonics simulation toolkit with GPU-accelerated FDTD via cloud API.

## Features

- **Local Structure Design**: Create photonic structures with density filtering and layer stacking
- **Modal Source Generation**: Fast eigenvalue-based waveguide mode solver (runs locally)
- **GPU-Accelerated Simulation**: Run FDTD simulations on cloud-based GPUs via API
- **Unidirectional Gaussian Sources**: Generate reflection-free Gaussian beams via API
- **Power Analysis**: Poynting flux calculations and transmission spectra
- **Visualization**: Built-in plotting for structures, fields, and convergence

## Installation

```bash
pip install hyperwave-community
```

Or install from source:

```bash
git clone https://github.com/yourusername/hyperwave-community.git
cd hyperwave-community
pip install -e .
```

## Quick Start

### 1. Get Your API Key

Sign up at [spinsphotonics.com](https://spinsphotonics.com) to get your API key.

```python
import hyperwave_community as hwc

# Get your API key from your dashboard at: https://spinsphotonics.com/dashboard
api_key = "your-api-key-here"
```

### 2. Create Theta (Design Pattern)

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Create design pattern (theta) - binary pattern for waveguide
theta = jnp.zeros((500, 1000))
center_y = theta.shape[0] // 2
waveguide_width = 40
strip_start = center_y - waveguide_width // 2
strip_end = center_y + waveguide_width // 2
theta = theta.at[strip_start:strip_end, :].set(1.0)

# Visualize theta pattern
plt.imshow(theta)

# Apply density filtering for smooth edges
waveguide_density = hwc.density(theta=theta, radius=8, alpha=0)

# Create blank density pattern for cladding layers (all SiO2)
cladding_density = hwc.density(theta=jnp.zeros_like(theta), radius=0, alpha=0)

# Visualize density pattern
plt.imshow(waveguide_density)
```

### 3. Build Structure and Visualize

```python
# Define materials (Silicon on SiO2)
n_Si, n_SiO2 = 3.4, 1.45
eps_Si, eps_SiO2 = n_Si**2, n_SiO2**2

# Define layer stack: SiO2 / Si / SiO2
waveguide_layer = hwc.Layer(
    density_pattern=waveguide_density,  # Use filtered waveguide pattern
    permittivity_values=(eps_SiO2, eps_Si),
    layer_thickness=20
)

cladding_layer = hwc.Layer(
    density_pattern=cladding_density,  # Use blank pattern (all SiO2)
    permittivity_values=eps_SiO2,
    layer_thickness=40
)

# Build 3D structure with vertical blurring
structure = hwc.create_structure(
    layers=[cladding_layer, waveguide_layer, cladding_layer],
    vertical_radius=2
)

# Add adiabatic absorbing boundaries
_, Lx, Ly, Lz = structure.permittivity.shape
abs_width = 70
abs_coeff = 4.89e-3
abs_shape = (abs_width, abs_width//2, abs_width//4)

absorption_boundary = hwc.create_absorption_mask(
    grid_shape=(Lx, Ly, Lz),
    absorption_widths=abs_shape,
    absorption_coeff=abs_coeff
)

structure.conductivity = structure.conductivity + absorption_boundary

# Visualize structure
hwc.view_structure(structure, show_permittivity=True, show_conductivity=False)
```

### 4. Create Mode Source and Visualize

```python
# Define frequency band (telecom wavelengths)
freq_band = (2*jnp.pi/32, 2*jnp.pi/30, 2)  # λ=30-32 pixels, 2 frequencies

# Generate mode source (after absorber region)
source_position = abs_shape[0] + 10  # 80 pixels (70 + 10)
source_field, source_offset, mode_info = hwc.create_mode_source(
    structure=structure,
    freq_band=freq_band,
    mode_num=0,  # Fundamental mode
    propagation_axis='x',
    source_position=source_position,
    perpendicular_bounds=(0, structure.permittivity.shape[2]),
    visualize=True  # Shows mode profile
)

print(f"Mode propagation constant β: {mode_info['beta']}")
print(f"Mode solver error: {mode_info['error']}")
```

### 5. Setup Monitors and Visualize Placement

```python
# Create monitor set
monitors = hwc.MonitorSet()

# Add monitors with automatic waveguide detection
monitors.add_monitors_at_position(
    structure=structure,
    axis='x',
    position=100,
    label='Input'
)

monitors.add_monitors_at_position(
    structure=structure,
    axis='x',
    position=400,
    label='Output'
)

# Visualize monitor positions on structure
hwc.view_monitors(structure, monitors)
print(f"Configured monitors: {monitors.list_monitors()}")
```

### 6. Run Simulation via API

```python
# Run FDTD simulation on cloud GPU
results = hwc.simulate(
    structure=structure,
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors=monitors,
    mode_info=mode_info,
    max_steps=20000,
    check_every_n=1000,
    source_ramp_periods=5.0,
    add_absorption=True,
    absorption_widths=(70, 35, 17),
    absorption_coeff=4.89e-3,
    api_key=api_key,
    gpu_type="H100"  # Options: "B200", "H200", "H100", "A100", "L40S", "L4", "A10G", "T4"
)

print(f"GPU time: {results['sim_time']:.2f}s")
print(f"Performance: {results['performance']:.2e} grid-points×steps/s")
```

**Available GPU types:**
- `B200` - NVIDIA Blackwell B200 (highest performance)
- `H200` - NVIDIA H200 (excellent performance)
- `H100` - NVIDIA H100 (recommended for large simulations)
- `A100` - NVIDIA A100 (high performance)
- `L40S` - NVIDIA L40S (balanced performance)
- `L4` - NVIDIA L4 (cost-effective)
- `A10G` - NVIDIA A10G (good performance)
- `T4` - NVIDIA T4 (entry-level)

### 7. Analyze Results

```python
# Quick visualization of all monitors
hwc.quick_view_monitors(results, component='all')  # Total field intensity
hwc.quick_view_monitors(results, component='Hz')   # Hz component

# Power analysis (already computed by API)
input_power = results['powers']['Input']
output_power = results['powers']['Output']

print(f"Input power: {jnp.mean(input_power):.4e}")
print(f"Output power: {jnp.mean(output_power):.4e}")

# Transmission analysis
transmission = results['transmissions']['transmission']
print(f"Transmission per frequency: {transmission}")
print(f"Average transmission: {jnp.mean(transmission):.4f}")
print(f"Transmission in dB: {10*jnp.log10(jnp.mean(transmission)):.2f} dB")

# Loss calculation
loss = 1 - jnp.mean(transmission)
loss_dB = -10*jnp.log10(jnp.mean(transmission))
print(f"Loss: {loss:.4f} ({loss_dB:.2f} dB)")
```

## Examples

The `examples/` directory will contain complete tutorials as Jupyter notebooks for different use cases. These notebooks are designed to run in Google Colab or local Jupyter environments.

Coming soon:
- Waveguide simulation with mode sources
- Metasurface design with Gaussian beam illumination
- GDS import/export workflows
- Custom monitor placement examples

## API Architecture

### What Runs Locally vs API

**Local (CPU):**
- Structure design (`create_structure`, `density`)
- Absorption masks (`create_absorption_mask`)
- Monitor configuration (`MonitorSet`)
- Mode source generation (`create_mode_source`) - eigenvalue solver

**API (GPU):**
- FDTD simulation (`simulate`) - main compute workload
- Gaussian source generation (`create_gaussian_source`) - requires FDTD

### Execution Summary

| Operation | Where it runs | Notes |
|-----------|---------------|-------|
| Structure creation | Local | Pure geometry |
| Mode solver | Local | Eigenvalue decomposition |
| Gaussian source | API | Requires FDTD |
| FDTD simulation | API | Main compute workload |

## API Reference

### Structure
- `density(theta, radius, alpha)` - Apply density filtering
- `create_structure(layers, vertical_radius)` - Create 3D structure from layers
- `Layer(density_pattern, permittivity_values, layer_thickness)` - Layer specification

### Sources
- `mode(freq_band, permittivity, axis, mode_num)` - Low-level mode solver
- `create_mode_source(structure, freq_band, ...)` - Generate modal source
- `create_gaussian_source(structure_shape, ...)` - Generate Gaussian source (API)

### Monitors
- `MonitorSet()` - Container for field monitors
- `add_monitors_at_position(structure, axis, position, label)` - Auto-place monitors
- `S_from_slice(field_slice)` - Calculate Poynting vector

### Data I/O
- `generate_gds_from_density(density_array, ...)` - Export density to GDS file
- `view_gds(gds_filepath, ...)` - Visualize GDS file contents
- `gds_to_theta(gds_filepath, ...)` - Import GDS file to theta array
- `component_to_theta(component, ...)` - Convert gdsfactory component to theta

### Simulation
- `simulate(structure, source_field, ..., api_key=None)` - Run FDTD simulation on GPU via API
- `generate_gaussian_source(structure_shape, ..., api_key=None)` - Generate Gaussian source via API

## Requirements

### Core Dependencies
- Python ≥ 3.9
- JAX ≥ 0.4.0 (CPU-only is sufficient)
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.4.0
- Requests ≥ 2.26.0
- gdstk ≥ 0.9.0 (GDS file I/O)
- scikit-image ≥ 0.19.0 (contour extraction)

### Optional Dependencies
- gdsfactory ≥ 7.0.0 (for `component_to_theta()` function)
  - Install with: `pip install hyperwave-community[gdsfactory]`

## License

MIT License - see LICENSE file for details

<!--
## Citation

If you use Hyperwave Community in your research, please cite:

```bibtex
@software{hyperwave_community,
  title = {Hyperwave Community: GPU-Accelerated Photonics Simulation},
  author = {Hyperwave Team},
  year = {2025},
  url = {https://github.com/yourusername/hyperwave-community}
}
```

## Support

- Documentation: https://docs.hyperwave.com
- Issues: https://github.com/yourusername/hyperwave-community/issues
- Email: support@hyperwave.com
- Discord: https://discord.gg/hyperwave

## Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines.

## Acknowledgments

Hyperwave Community builds on research in computational electromagnetics and inverse design. Special thanks to the JAX team for the excellent GPU computing framework.
-->
