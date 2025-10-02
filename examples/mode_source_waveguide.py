#!/usr/bin/env python3
"""Example: Waveguide simulation with mode source.

This example demonstrates the complete workflow:
1. Create theta (design pattern)
2. Build structure with adiabatic absorbers
3. Generate mode source
4. Setup field monitors
5. Run FDTD simulation via API
6. Analyze and visualize results
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import hyperwave_community as hwc

# =============================================================================
# 1. Configure API Access
# =============================================================================

# API access configuration will be added here
# Get your API key at: https://spinsphotonics.com/dashboard (coming soon)

# =============================================================================
# 2. Create Theta (Design Pattern)
# =============================================================================

print("Creating design pattern...")

# Create design pattern (theta) - binary pattern for waveguide
theta = jnp.zeros((500, 1000))
center_y = theta.shape[0] // 2
waveguide_width = 40
strip_start = center_y - waveguide_width // 2
strip_end = center_y + waveguide_width // 2
theta = theta.at[strip_start:strip_end, :].set(1.0)

# Visualize theta pattern
plt.imshow(theta)
plt.title('Theta: Binary Waveguide Pattern')
plt.show()

# Apply density filtering for smooth edges
waveguide_density = hwc.density(theta=theta, radius=8, alpha=0)

# Create blank density pattern for cladding layers (all SiO2)
cladding_density = hwc.density(theta=jnp.zeros_like(theta), radius=0, alpha=0)

# Visualize density pattern
plt.imshow(waveguide_density)
plt.title('Density: Smoothed Waveguide Pattern')
plt.show()

# =============================================================================
# 3. Build Structure and Visualize
# =============================================================================

print("Building structure...")

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
print(f"Structure shape: {structure.permittivity.shape}")

# =============================================================================
# 4. Create Mode Source and Visualize
# =============================================================================

print("\nGenerating mode source...")

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

# =============================================================================
# 5. Setup Monitors and Visualize Placement
# =============================================================================

print("\nSetting up monitors...")

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

# =============================================================================
# 6. Run Simulation via API
# =============================================================================

print("\nSubmitting simulation to API...")

# Run FDTD simulation on H100 GPU
results = hwc.simulate(
    structure=structure,
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors=monitors,
    mode_info=mode_info,
    max_steps=10000,
    check_every_n=1000,
    source_ramp_periods=5.0,
    add_absorption=True,
    absorption_widths=(70, 35, 17),
    absorption_coeff=4.89e-3
)

print(f"GPU time: {results['sim_time']:.2f}s")
print(f"Performance: {results['performance']:.2e} grid-points×steps/s")

# =============================================================================
# 7. Analyze Results
# =============================================================================

print("\n" + "="*60)
print("RESULTS ANALYSIS")
print("="*60)

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

print("\n" + "="*60)
print("SIMULATION COMPLETE")
print("="*60)
print("\nVisualization windows have been displayed using quick_view_monitors().")
print("Field data is available in results['monitor_data'] for custom analysis.")
