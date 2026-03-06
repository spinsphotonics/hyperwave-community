# Grating Coupler: Comprehensive Tutorial
#
# This tutorial walks through a full grating coupler simulation using the
# HyperWave SDK. Unlike the quick-start, every parameter is set explicitly
# so you can see exactly how the simulation is configured and why.
#
# What you will learn:
#   - Load a grating coupler from gdsfactory with custom geometry
#   - Convert it to a simulation grid and apply density filtering
#   - Build a multi-layer SOI stack with custom material properties
#   - Configure absorbing boundaries with physics-based parameters
#   - Create a guided mode source at a specific wavelength
#   - Place multiple monitors to capture transmission and reflection
#   - Run the simulation with custom convergence settings
#   - Analyze results: convergence, fields, S-parameters, power budget
#   - Export data to CSV for post-processing


# %% Installation
# pip install hyperwave-community gdsfactory matplotlib


# %% Imports + Verbose Logging

import jax.numpy as jnp
import hyperwave_community as hwc

# Enable verbose logging so you can see what the SDK is doing at each step.
# set_debug() would add even more detail, useful for troubleshooting.
hwc.set_verbose()

import matplotlib  # noqa: E402
matplotlib.use("Agg")


# %% Step 1: Load Grating Coupler Component
#
# We use gdsfactory's grating_coupler_rectangular, which models a uniform
# rectangular-tooth grating. The key parameters:
#
#   n_periods:     Number of grating teeth. More periods = stronger coupling
#                  but longer device. 15 is a good balance for tutorials.
#   period:        Tooth pitch in micrometers. For 1550nm TE operation on
#                  standard SOI, 0.68-0.75 um is typical.
#   fill_factor:   Fraction of each period occupied by the tooth (etched region).
#                  0.5 = equal tooth and gap widths. Tune for coupling efficiency.
#   width_grating: Lateral width of the grating in micrometers. Should be wide
#                  enough to capture the fiber mode (~10 um for SMF-28).
#   length_taper:  Length of the taper from the waveguide to the grating region.
#                  Longer tapers reduce mode mismatch but increase device size.
#                  50 um keeps the simulation grid manageable for a tutorial.
#   wavelength:    Target wavelength (um). Affects internal geometry calculations.
#   fiber_angle:   Angle of the fiber relative to vertical (degrees). Standard
#                  packaging uses 8-15 degrees to avoid back-reflections.

import gdsfactory as gf  # noqa: E402
gf.gpdk.PDK.activate()

gc = gf.components.grating_coupler_rectangular(
    n_periods=15,
    period=0.68,
    fill_factor=0.5,
    width_grating=11.0,
    length_taper=50.0,
    wavelength=1.55,
    fiber_angle=10,
    slab_xmin=-1.0,
    slab_offset=1.0,
)

print(f"Component: {gc.name}")
print(f"Bounding box: {gc.dbbox()}")
print(f"Ports: {[p.name for p in gc.ports]}")


# %% Step 2: Convert to Simulation Grid
#
# component_to_theta rasterizes the GDS geometry onto a regular grid.
# The resolution parameter controls the grid spacing in the FINAL structure
# (not the theta array). Internally, theta is generated at 2x finer resolution
# because create_structure downsamples by 2.
#
# For 1550nm light in silicon, a resolution of 0.05 um (50 nm) gives roughly
# 31 grid points per wavelength in the highest-index material (n~3.48),
# which is above the 20-point minimum for accurate FDTD. This keeps the
# total grid under ~30M cells, fitting comfortably on a B200 GPU.

resolution_um = 0.05

theta, info = hwc.component_to_theta(
    gc,
    resolution=resolution_um,
    waveguide_value=1.0,
    background_value=0.0,
)

print(f"Theta shape: {theta.shape}")
print(f"Theta resolution: {info['theta_resolution_um']:.4f} um/pixel")
print(f"Structure resolution: {info['structure_resolution_um']:.4f} um/pixel")
print(f"Component name: {info['component_name']}")


# %% Step 3: Apply Density Filtering
#
# The density filter enforces a minimum feature size and smooths the binary
# geometry into a form suitable for gradient-based optimization. Even for
# forward-only simulations (no optimization), density filtering improves
# numerical stability by removing sub-pixel features.
#
# Parameters:
#   radius:  Filter radius in pixels. Larger = smoother features, stronger
#            minimum feature size constraint. For a 40nm grid, radius=3
#            gives a minimum feature size of ~120nm.
#   alpha:   Binarization strength [0, 1]. 0 = soft density, 1 = fully binary.
#            For forward simulation, moderate binarization (0.8) is fine.
#            For optimization, start at 0 and gradually increase.

filtered_density = hwc.density(theta, radius=3, alpha=0.8)

print(f"Filtered density shape: {filtered_density.shape}")
print(f"Density range: [{float(jnp.min(filtered_density)):.4f}, {float(jnp.max(filtered_density)):.4f}]")


# %% Step 4: Build Multi-Layer Structure
#
# A grating coupler on SOI has a specific vertical stack. From bottom to top:
#
#   1. Buried oxide (BOX): SiO2 substrate, eps = 2.085 (n=1.444)
#   2. Silicon device layer: patterned region, eps from 2.085 to 12.08 (n=3.476)
#   3. Upper cladding: SiO2 or air above the device
#
# Layer thicknesses are in grid units. At 50nm resolution:
#   - BOX:      2.0 um / 0.05 um = 40 grid units
#   - Silicon:  0.22 um / 0.05 um = 4.4 grid units (fractional is fine)
#   - Cladding: 2.0 um / 0.05 um = 40 grid units
#
# vertical_radius controls vertical smoothing between layers. A small value
# (2.0) keeps interfaces sharp, which is appropriate for SOI where the
# layer boundaries are physically abrupt.

eps_si = 12.08    # Silicon at 1550nm (n = 3.476)
eps_sio2 = 2.085  # SiO2 at 1550nm (n = 1.444)
eps_air = 1.0     # Air

# Uniform layers use a single permittivity value (no density dependence).
# The patterned layer interpolates between background and waveguide permittivity
# based on the filtered density.

box_layer = hwc.Layer(
    density_pattern=jnp.ones_like(filtered_density),
    permittivity_values=eps_sio2,
    layer_thickness=40,
)

device_layer = hwc.Layer(
    density_pattern=filtered_density,
    permittivity_values=(eps_sio2, eps_si),
    layer_thickness=4.4,
)

cladding_layer = hwc.Layer(
    density_pattern=jnp.ones_like(filtered_density),
    permittivity_values=eps_air,
    layer_thickness=40,
)

structure = hwc.create_structure(
    layers=[box_layer, device_layer, cladding_layer],
    vertical_radius=2.0,
)

print(f"Permittivity shape: {structure.permittivity.shape}")
print(f"Grid size: {structure.permittivity.shape[1]} x {structure.permittivity.shape[2]} x {structure.permittivity.shape[3]}")
print(f"Permittivity range: [{float(jnp.min(structure.permittivity)):.2f}, {float(jnp.max(structure.permittivity)):.2f}]")


# %% Visualize Structure
#
# plot_structure shows cross-sections of the permittivity distribution.
# The default dual-view gives XY (top-down) and XZ (side) slices at the
# midpoints, which is ideal for verifying the grating pattern and layer stack.

hwc.plot_structure(structure, show=False, save_path="gc_structure.png")
print("Structure visualization saved to gc_structure.png")


# %% Step 5: Add Absorbing Boundaries
#
# Absorbing boundaries prevent artificial reflections from the simulation edges.
# The SDK provides absorber_params() which computes physics-based widths and
# coefficients from the wavelength and grid spacing.
#
# For grating couplers, the Z absorber is especially important because light
# diffracts upward into the cladding. The XY absorbers handle the laterally
# scattered light and the guided mode that continues past the monitors.

grid_shape = structure.permittivity.shape[1:]  # (nx, ny, nz)

# Get physics-based absorber parameters
absorber = hwc.absorber_params(
    wavelength_um=1.55,
    dx_um=resolution_um,
    structure_dimensions=grid_shape,
)

print(f"Absorber XY width: {absorber['abs_xy_um']:.2f} um ({absorber['absorption_widths'][0]} cells)")
print(f"Absorber Z width:  {absorber['abs_z_um']:.2f} um ({absorber['absorption_widths'][2]} cells)")
print(f"Absorber coefficient: {absorber['abs_coeff']:.6f}")

absorption_widths = absorber["absorption_widths"]
absorption_coeff = absorber["abs_coeff"]

# Create the absorption mask for visualization
absorption_mask = hwc.create_absorption_mask(
    grid_shape=grid_shape,
    absorption_widths=absorption_widths,
    absorption_coeff=absorption_coeff,
)

print(f"Absorption mask shape: {absorption_mask.shape}")

hwc.plot_absorption_mask(absorption_mask, show=False, save_path="gc_absorption.png")
print("Absorption mask visualization saved to gc_absorption.png")


# %% Step 6: Create Mode Source
#
# The mode source excites the fundamental guided mode (TE0) of the waveguide.
# It is placed near the waveguide port of the grating coupler so that
# light propagates into the grating and diffracts upward.
#
# freq_band defines the simulation frequencies as angular frequencies (omega).
# For a single-wavelength simulation at 1550nm:
#   omega = 2*pi / lambda = 2*pi / 1.55 ~ 4.054
#
# source_position: X-coordinate (in grid cells) where the mode is solved.
#   Placing it a few cells inside the absorber boundary ensures a clean launch.
#
# z_bounds: Restricts the mode solver to the silicon layer region. This helps
#   it find the correct guided mode rather than substrate or cladding modes.

wavelength_um = 1.55
omega = 2 * jnp.pi / wavelength_um
freq_band = (float(omega), float(omega), 1)

# Source position: after the absorber region, near the waveguide input
source_x = absorption_widths[0] + 10

# Z bounds: limit mode solving to the silicon device layer.
# The device layer starts at z=40 and extends ~4.4 cells.
# Give it some margin above and below for the evanescent tails.
z_start = 40 - 10  # 10 cells below the device layer
z_end = 45 + 10    # 10 cells above the device layer

source_field, source_offset, mode_info = hwc.create_mode_source(
    structure,
    freq_band=freq_band,
    mode_num=0,
    propagation_axis="x",
    source_position=source_x,
    z_bounds=(z_start, z_end),
)

print(f"Source field shape: {source_field.shape}")
print(f"Source offset: {source_offset}")
print(f"Mode beta: {mode_info['beta']}")
print(f"Mode error: {mode_info['error']}")

# Visualize the mode profile
hwc.plot_mode(
    mode_info["field"],
    mode_info["beta"],
    mode_num=0,
    propagation_axis="x",
    show=False,
    save_path="gc_mode.png",
)
print("Mode profile saved to gc_mode.png")


# %% Step 7: Set Up Monitors
#
# Monitors record the electromagnetic field at specific locations during the
# simulation. For a grating coupler we want:
#
#   1. Input monitor: right after the source, measures input power (normalization)
#   2. Reflection monitor: between source and grating, measures back-reflection
#   3. Output (transmission) monitor: at the far end, after the grating
#
# add_monitors_at_position automatically detects waveguide features and
# sizes the monitors appropriately. The label prefix determines the naming
# convention used by analyze_transmission.

monitors = hwc.MonitorSet()

# Input monitor: just after the source
input_x = source_x + 15
input_names = monitors.add_monitors_at_position(
    structure,
    axis="x",
    position=input_x,
    label="Input_src",
)
print(f"Input monitors: {input_names}")

# Reflection monitor: between source and the grating region
reflect_x = source_x + 5
reflect_names = monitors.add_monitors_at_position(
    structure,
    axis="x",
    position=reflect_x,
    label="Reflect_r1",
)
print(f"Reflection monitors: {reflect_names}")

# Output monitor: at the far end of the structure (before absorber)
output_x = grid_shape[0] - absorption_widths[0] - 15
output_names = monitors.add_monitors_at_position(
    structure,
    axis="x",
    position=output_x,
    label="Output_o1",
)
print(f"Output monitors: {output_names}")

print(f"\nTotal monitors: {len(monitors.monitors)}")
print(f"Monitor mapping: {monitors.mapping}")


# %% Visualize Monitor Layout
#
# plot_monitor_layout overlays monitor rectangles on a structure cross-section.
# The Z-axis view shows the top-down layout including monitor positions and
# the source plane.

hwc.plot_monitor_layout(
    structure.permittivity,
    monitors,
    axis="z",
    source_position=source_x,
    show=False,
    save_path="gc_monitor_layout.png",
)
print("Monitor layout saved to gc_monitor_layout.png")


# %% Step 8: Configure API + Run Simulation
#
# Before running, configure your API key. You can also set it via the
# HYPERWAVE_API_KEY environment variable.
#
# The simulation uses the "local workflow": structure, source, and monitors
# are built locally, then sent to the cloud GPU for the FDTD time-stepping.
#
# Key simulation parameters:
#   simulation_steps:  Maximum number of FDTD time steps. 20000 is usually
#                      sufficient for grating couplers with early stopping.
#   convergence:       Early stopping preset. "thorough" checks every 1000 steps
#                      and requires 5 consecutive stable checks with min 5000 steps.
#   gpu_type:          GPU model. "B200" is fastest, "A100" is an alternative.

hwc.configure_api()  # Uses HYPERWAVE_API_KEY environment variable

# Get the structure recipe for the cloud
structure_recipe = structure.extract_recipe()
monitors_recipe = monitors.recipe

# Estimate cost before running
cost = hwc.estimate_cost(
    structure_shape=structure.permittivity.shape,
    max_steps=20000,
    gpu_type="B200",
)
if cost:
    print(f"Estimated cost: {cost.get('estimated_credits', 'N/A')} credits")
    print(f"Estimated time: {cost.get('estimated_seconds', 'N/A')} seconds")

# Run the simulation
results = hwc.simulate(
    structure_recipe=structure_recipe,
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors_recipe=monitors_recipe,
    mode_info=mode_info,
    simulation_steps=20000,
    convergence="thorough",
    add_absorption=True,
    absorption_widths=absorption_widths,
    absorption_coeff=absorption_coeff,
    gpu_type="B200",
)

print("\nSimulation complete!")
print(f"  GPU time: {results.get('sim_time', 'N/A')} seconds")
print(f"  Converged: {results.get('converged', 'N/A')}")
print(f"  Convergence step: {results.get('convergence_step', 'N/A')}")


# %% Step 9: Analyze Results
#
# analyze_transmission computes S-parameters (power ratios) between the input
# monitor and each output monitor. It uses Poynting vector integration for
# accurate power flow measurement.

# 9a. Convergence history
if "convergence_history" in results:
    steps = results["convergence_history"].get("steps", [])
    errors = results["convergence_history"].get("errors", [])
    if steps and errors:
        hwc.plot_convergence(
            steps, errors,
            title="Grating Coupler Convergence",
            show=False,
            save_path="gc_convergence.png",
        )
        print("Convergence plot saved to gc_convergence.png")

# 9b. Transmission analysis
# Use the actual monitor names from Step 7. For a single-waveguide grating
# coupler, add_monitors_at_position returns ["Input_src"] (one feature detected).
transmission = hwc.analyze_transmission(
    results,
    input_monitor=input_names[0],
    output_monitors=output_names,
    direction="x",
    print_results=True,
)

print("\nPower budget:")
print(f"  Total transmission: {transmission['total_transmission']:.4f}")
print(f"  Excess loss: {transmission['excess_loss_dB']:.2f} dB")

# 9c. Field visualization
hwc.plot_monitors(
    results,
    component="Hz",
    freq_idx=0,
    show=False,
    save_path="gc_fields.png",
)
print("Field plots saved to gc_fields_*.png")


# %% Step 10: Export Data
#
# export_csv writes simulation data to CSV files for post-processing in
# other tools (MATLAB, Excel, custom scripts). It automatically detects
# the data format and writes appropriate columns.

csv_path = hwc.export_csv(results, output_path="gc_results.csv")
print(f"Results exported to: {csv_path}")

# Export just the transmission data separately
csv_path_t = hwc.export_csv(transmission, output_path="gc_transmission.csv")
print(f"Transmission data exported to: {csv_path_t}")

print("\nTutorial complete! Check the output files for visualizations and data.")
