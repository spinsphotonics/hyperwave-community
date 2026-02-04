#!/usr/bin/env python3
"""
Test script for Local Workflow - validates all steps before porting to notebook.
"""

import hyperwave_community as hwc
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# ============================================================================
# IMPORTS & PDK SETUP
# ============================================================================
print("=" * 60)
print("Step 0: Imports & PDK Setup")
print("=" * 60)

# Activate generic PDK (required for gdsfactory components)
PDK = gf.gpdk.get_generic_pdk()
PDK.activate()
print("PDK activated successfully")

# ============================================================================
# STEP 1: Load GDSFactory Component
# ============================================================================
print("\n" + "=" * 60)
print("Step 1: Load GDSFactory Component")
print("=" * 60)

COMPONENT_NAME = "mmi2x2_with_sbend"
RESOLUTION_UM = 0.02  # 20nm resolution

# Load component from GDSFactory
component = gf.components.mmi2x2_with_sbend()
print(f"Loaded component: {COMPONENT_NAME}")

# Convert to theta pattern
theta, device_info = hwc.component_to_theta(
    component=component,
    resolution=RESOLUTION_UM,
)

print(f"Theta shape: {theta.shape}")
print(f"Device size: {device_info['physical_size_um']} um")

# ============================================================================
# STEP 2: Apply Density Filtering
# ============================================================================
print("\n" + "=" * 60)
print("Step 2: Apply Density Filtering")
print("=" * 60)

# Material properties
N_CORE = 3.48      # Silicon refractive index
N_CLAD = 1.4457    # SiO2 cladding refractive index

# Padding for absorbers and monitors
PADDING = (100, 100, 0, 0)  # (left, right, top, bottom) in pixels

# Apply density filtering
density_core = hwc.density(
    theta=theta,
    pad_width=PADDING,
    radius=3,  # Smoothing radius
)

density_clad = hwc.density(
    theta=jnp.zeros_like(theta),
    pad_width=PADDING,
    radius=3,
)

print(f"Density shape (with padding): {density_core.shape}")

# ============================================================================
# STEP 3: Build 3D Layer Structure
# ============================================================================
print("\n" + "=" * 60)
print("Step 3: Build 3D Layer Structure")
print("=" * 60)

# Layer dimensions
WG_HEIGHT_UM = 0.22         # Waveguide height
TOTAL_HEIGHT_UM = 4.0       # Total simulation height
VERTICAL_RADIUS = 2.0       # Vertical blur radius

# Calculate layer thicknesses in cells
wg_thickness_cells = int(np.round(WG_HEIGHT_UM / RESOLUTION_UM))
clad_thickness_cells = int(np.round((TOTAL_HEIGHT_UM - WG_HEIGHT_UM) / 2 / RESOLUTION_UM))

print(f"Waveguide thickness: {wg_thickness_cells} cells")
print(f"Cladding thickness: {clad_thickness_cells} cells")

# Define layers
waveguide_layer = hwc.Layer(
    density_pattern=density_core,
    permittivity_values=(N_CLAD**2, N_CORE**2),  # (clad, core)
    layer_thickness=wg_thickness_cells,
)

cladding_layer = hwc.Layer(
    density_pattern=density_clad,
    permittivity_values=N_CLAD**2,
    layer_thickness=clad_thickness_cells,
)

# Create 3D structure
structure = hwc.create_structure(
    layers=[cladding_layer, waveguide_layer, cladding_layer],
    vertical_radius=VERTICAL_RADIUS,
)

_, Lx, Ly, Lz = structure.permittivity.shape
print(f"Structure dimensions: ({Lx}, {Ly}, {Lz})")

# ============================================================================
# STEP 4: Add Absorbing Boundaries
# ============================================================================
print("\n" + "=" * 60)
print("Step 4: Add Absorbing Boundaries")
print("=" * 60)

# Absorber parameters (optimized for 20nm resolution)
ABS_WIDTH_X = 82
ABS_WIDTH_Y = 41
ABS_WIDTH_Z = 41
ABS_COEFF = 6.17e-4

abs_shape = (ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z)

# Create absorption mask
absorber = hwc.create_absorption_mask(
    grid_shape=(Lx, Ly, Lz),
    absorption_widths=abs_shape,
    absorption_coeff=ABS_COEFF,
    show_plots=False,  # Disable for script
)

# Add absorber to structure conductivity
structure.conductivity = jnp.zeros_like(structure.conductivity) + absorber

print(f"Absorber shape: {abs_shape}")
print(f"Absorber coefficient: {ABS_COEFF}")

# ============================================================================
# STEP 5: Create Mode Source
# ============================================================================
print("\n" + "=" * 60)
print("Step 5: Create Mode Source")
print("=" * 60)

# Wavelength and frequency settings
WL_UM = 1.55  # Wavelength in microns
wl_cells = WL_UM / RESOLUTION_UM
freq_band = (2 * jnp.pi / wl_cells, 2 * jnp.pi / wl_cells, 1)

# Source position (after absorber region)
source_pos_x = ABS_WIDTH_X + 5

print(f"Wavelength: {WL_UM} um ({wl_cells} cells)")
print(f"Source position: x={source_pos_x}")

# Create mode source
source_field, source_offset, mode_info = hwc.create_mode_source(
    structure=structure,
    freq_band=freq_band,
    mode_num=0,  # Fundamental mode
    propagation_axis="x",
    source_position=source_pos_x,
    perpendicular_bounds=(0, Ly // 2),  # Bottom half only for input waveguide
    visualize=False,
)

print(f"Source field shape: {source_field.shape}")
print(f"Source offset: {source_offset}")

# ============================================================================
# STEP 6: Set Up Monitors
# ============================================================================
print("\n" + "=" * 60)
print("Step 6: Set Up Monitors")
print("=" * 60)

# Create monitor set
monitors = hwc.MonitorSet()

# Input monitor (after source)
monitors.add_monitors_at_position(
    structure=structure,
    axis="x",
    position=ABS_WIDTH_X + 10,
    label="Input",
)

# Output monitors (before right absorber)
monitors.add_monitors_at_position(
    structure=structure,
    axis="x",
    position=Lx - (ABS_WIDTH_X + 10),
    label="Output",
)

# List all monitors
monitors.list_monitors()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("LOCAL WORKFLOW SETUP COMPLETE")
print("=" * 60)
print(f"Structure: {Lx} x {Ly} x {Lz}")
print(f"Source position: x={source_pos_x}")
print(f"Absorber widths: {abs_shape}")
print(f"Frequency band: {freq_band}")
print(f"Source field shape: {source_field.shape}")
print("\nReady for GPU simulation with hwc.simulate()")

# ============================================================================
# STEP 7: GPU SIMULATION
# ============================================================================
print("\n" + "=" * 60)
print("Step 7: GPU Simulation")
print("=" * 60)

API_KEY = "7c8aee19-01c9-4400-b5b9-6af0ccf3b118"

# Extract recipes for API
structure_recipe = structure.extract_recipe()
monitors_recipe = monitors.recipe

print(f"Structure recipe keys: {list(structure_recipe.keys())}")
print(f"Number of monitors: {len(monitors_recipe)}")

results = hwc.simulate(
    structure_recipe=structure_recipe,
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors_recipe=monitors_recipe,
    mode_info=mode_info,
    simulation_steps=20000,
    check_every_n=1000,
    source_ramp_periods=5.0,
    add_absorption=True,
    absorption_widths=abs_shape,
    absorption_coeff=ABS_COEFF,
    api_key=API_KEY,
    gpu_type="B200",
)

print(f"GPU time: {results['sim_time']:.2f}s")
print(f"Performance: {results['performance']:.2e} grid-points×steps/s")

# ============================================================================
# STEP 8: RESULT ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("Step 8: Result Analysis")
print("=" * 60)

# Get monitor data
monitor_data = results['monitor_data']
print(f"Available monitors: {list(monitor_data.keys())}")

# Helper function for Poynting vector calculation
def S_from_slice(field_slice):
    """Calculate Poynting vector from field slice."""
    E = field_slice[:, :3, :, :]
    H = field_slice[:, 3:, :, :]

    S = jnp.zeros_like(E, dtype=jnp.float32)
    S = S.at[:, 0, :, :].set(jnp.real(E[:, 1] * jnp.conj(H[:, 2]) - E[:, 2] * jnp.conj(H[:, 1])))
    S = S.at[:, 1, :, :].set(jnp.real(E[:, 2] * jnp.conj(H[:, 0]) - E[:, 0] * jnp.conj(H[:, 2])))
    S = S.at[:, 2, :, :].set(jnp.real(E[:, 0] * jnp.conj(H[:, 1]) - E[:, 1] * jnp.conj(H[:, 0])))

    return S * 0.5

# Get field data from monitors
input_fields = monitor_data['Input_bottom']
output_bottom = monitor_data['Output_bottom']
output_top = monitor_data['Output_top']

print(f"Monitor field shape: {input_fields.shape}")

# Average across monitor thickness (X dimension)
input_plane = jnp.mean(input_fields, axis=2)
out_bottom_plane = jnp.mean(output_bottom, axis=2)
out_top_plane = jnp.mean(output_top, axis=2)

# Calculate Poynting vectors
S_in = S_from_slice(input_plane)
S_out_bottom = S_from_slice(out_bottom_plane)
S_out_top = S_from_slice(out_top_plane)

# Calculate power (X-component)
power_in = jnp.abs(jnp.sum(S_in[:, 0, :, :], axis=(1, 2)))
power_out_bottom = jnp.abs(jnp.sum(S_out_bottom[:, 0, :, :], axis=(1, 2)))
power_out_top = jnp.abs(jnp.sum(S_out_top[:, 0, :, :], axis=(1, 2)))
power_out_total = power_out_bottom + power_out_top

# Calculate metrics
total_transmission = power_out_total / power_in
split_bottom = power_out_bottom / power_out_total
split_top = power_out_top / power_out_total

# Print results
print("\n" + "=" * 60)
print("TRANSMISSION ANALYSIS")
print("=" * 60)
print(f"Input Power:      {float(power_in[0]):.4e}")
print(f"Output Bottom:    {float(power_out_bottom[0]):.4e}")
print(f"Output Top:       {float(power_out_top[0]):.4e}")
print(f"Total Output:     {float(power_out_total[0]):.4e}")
print("-" * 60)
print(f"Transmission:     {float(total_transmission[0]):.1%}")
print(f"Split Ratio:      {float(split_bottom[0]):.1%} / {float(split_top[0]):.1%}")
print("=" * 60)

print("\nScript completed successfully!")
