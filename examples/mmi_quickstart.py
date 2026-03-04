# %% Installation
# pip install hyperwave-community

# %% Imports
import hyperwave_community as hwc
import gdsfactory as gf
import numpy as np
import jax.numpy as jnp

PDK = gf.gpdk.get_generic_pdk()
PDK.activate()


# %% Step 1: Load MMI Component
#
# Load a 2x2 MMI with S-bends from gdsfactory. Extend ports so the mode
# source and monitors sit inside straight waveguide sections.

COMPONENT_NAME = "mmi2x2_with_sbend"
RESOLUTION_UM = 0.02          # 20 nm grid spacing
EXTENSION_LENGTH = 2.0        # Extend ports by 2 um

gf_device = gf.components.mmi2x2_with_sbend()
gf_extended = gf.c.extend_ports(gf_device, length=EXTENSION_LENGTH)

theta, device_info = hwc.component_to_theta(
    component=gf_extended,
    resolution=RESOLUTION_UM,
)

print(f"Theta shape: {theta.shape}")
print(f"Device size: {device_info['physical_size_um']} um")


# %% Step 2: Density Filtering
#
# Smooth the binary pattern with a conic filter for numerical stability
# and add padding for absorbers and monitors.

N_CORE = 3.48                  # Silicon refractive index at 1550 nm
N_CLAD = 1.45                  # SiO2 cladding

PADDING = (100, 100, 0, 0)    # (left, right, top, bottom) in theta pixels

density_core = hwc.density(theta=theta, pad_width=PADDING, radius=3)
density_clad = hwc.density(theta=jnp.zeros_like(theta), pad_width=PADDING, radius=5)

print(f"Density shape (with padding): {density_core.shape}")


# %% Step 3: Build 3D Structure
#
# Stack cladding and waveguide layers into a 3D permittivity volume.

WG_HEIGHT_UM = 0.22           # SOI waveguide height
TOTAL_HEIGHT_UM = 4.0          # Total simulation height
VERTICAL_RADIUS = 2            # Vertical blur radius

wg_thickness_cells = max(1, int(np.round(WG_HEIGHT_UM / RESOLUTION_UM)))
clad_thickness_cells = int(np.round((TOTAL_HEIGHT_UM - WG_HEIGHT_UM) / 2 / RESOLUTION_UM))

eps_core = N_CORE ** 2
eps_clad = N_CLAD ** 2

waveguide_layer = hwc.Layer(
    density_pattern=density_core,
    permittivity_values=(eps_clad, eps_core),
    layer_thickness=wg_thickness_cells,
)

cladding_layer = hwc.Layer(
    density_pattern=density_clad,
    permittivity_values=eps_clad,
    layer_thickness=clad_thickness_cells,
)

structure = hwc.create_structure(
    layers=[cladding_layer, waveguide_layer, cladding_layer],
    vertical_radius=VERTICAL_RADIUS,
)

_, Lx, Ly, Lz = structure.permittivity.shape
print(f"Structure: ({Lx}, {Ly}, {Lz}) cells")

# Visualize XY cross-section at waveguide center
z_wg_center = clad_thickness_cells + wg_thickness_cells // 2

hwc.plot_structure(
    structure,
    show_permittivity=True,
    show_conductivity=False,
    axis="z",
    position=z_wg_center,
    show=False,
)


# %% Step 4: Absorbing Boundaries
#
# Add adiabatic absorbers at grid edges to prevent reflections.

abs_params = hwc.absorber_params(
    wavelength_um=1.55,
    dx_um=RESOLUTION_UM,
    structure_dimensions=(Lx, Ly, Lz),
)

abs_widths = tuple(abs_params["absorption_widths"])
abs_coeff = abs_params["abs_coeff"]

print(f"Absorber widths (x, y, z): {abs_widths}")
print(f"Absorber coefficient: {abs_coeff:.6f}")

absorber = hwc.create_absorption_mask(
    grid_shape=(Lx, Ly, Lz),
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
)

structure.conductivity = jnp.zeros_like(structure.conductivity) + absorber

hwc.plot_absorption_mask(absorber, show=False)


# %% Step 5: Mode Source
#
# Solve for the fundamental TE mode at the input waveguide.

WL_UM = 1.55
wl_cells = WL_UM / RESOLUTION_UM
freq_band = (2 * jnp.pi / wl_cells, 2 * jnp.pi / wl_cells, 1)

# Place source just inside the absorber region
source_pos_x = abs_widths[0]

# Auto-detect waveguide at source plane to set mode-solver bounds
temp_monitors = hwc.MonitorSet()
temp_monitors.add_monitors_at_position(
    structure=structure, axis="x", position=source_pos_x, label="source_detect",
)

source_monitor = temp_monitors.monitors[0]
y_min_orig = source_monitor.offset[1]
y_max_orig = y_min_orig + source_monitor.shape[1]
z_min_orig = source_monitor.offset[2]
z_max_orig = z_min_orig + source_monitor.shape[2]

# Double the extent so the mode field decays to zero at edges
y_center = (y_min_orig + y_max_orig) // 2
z_center = (z_min_orig + z_max_orig) // 2
y_half = source_monitor.shape[1]
z_half = source_monitor.shape[2]

y_min = max(0, y_center - y_half)
y_max = min(Ly, y_center + y_half)
z_min = max(0, z_center - z_half)
z_max = min(Lz, z_center + z_half)

source_field, source_offset, mode_info = hwc.create_mode_source(
    structure=structure,
    freq_band=freq_band,
    mode_num=0,
    propagation_axis="x",
    source_position=source_pos_x,
    perpendicular_bounds=(y_min, y_max),
    z_bounds=(z_min, z_max),
)

# Trim source field to mode region (reduces data transfer to cloud)
source_field_trimmed = source_field[:, :, :, y_min:y_max, z_min:z_max]
source_offset_corrected = (source_pos_x, y_min, z_min)

print(f"Source field shape: {source_field_trimmed.shape}")

hwc.plot_mode(
    mode_field=mode_info["field"],
    beta=mode_info["beta"],
    mode_num=0,
    propagation_axis="x",
    show=False,
)


# %% Step 6: Set Up Monitors
#
# Place field monitors at input/output ports plus a full XY plane for
# field visualization.

gf_device = gf.components.mmi2x2_with_sbend()

monitors = hwc.MonitorSet()

# Map port coordinates from GDS to structure grid
y_pad_struct = PADDING[0] // 2
x_pad_struct = PADDING[2] // 2
bbox = device_info["bounding_box_um"]
x_min_um, y_min_um = bbox[0], bbox[1]
theta_res = device_info["theta_resolution_um"]

MONITOR_THICKNESS = 5
MONITOR_HALF = 35              # Half-extent in Y and Z

for port in gf_device.ports:
    px_um, py_um = port.center
    x_struct = int((px_um - x_min_um) / theta_res / 2) + x_pad_struct
    y_struct = int((py_um - y_min_um) / theta_res / 2) + y_pad_struct

    if abs(port.orientation % 360 - 180) < 1:
        label = f"Input_{port.name}"
    else:
        label = f"Output_{port.name}"

    monitor = hwc.Monitor(
        shape=(MONITOR_THICKNESS, 2 * MONITOR_HALF, 2 * MONITOR_HALF),
        offset=(x_struct, y_struct - MONITOR_HALF, z_wg_center - MONITOR_HALF),
    )
    monitors.add(monitor, label)

# Full XY plane at waveguide center for field visualization
xy_mid = hwc.Monitor(shape=(Lx, Ly, 1), offset=(0, 0, z_wg_center))
monitors.add(xy_mid, name="xy_mid")

print("Monitors:", monitors.list_monitors())

hwc.plot_monitor_layout(
    structure.permittivity,
    monitors,
    axis="z",
    position=z_wg_center,
    source_position=source_pos_x,
    show=False,
)


# %% Step 7: Configure API and Run Simulation
#
# This is the only step that uses cloud GPU and requires an API key.
# Sign up at https://spinsphotonics.com/signup to get your key.

# For Google Colab, load the key from Colab Secrets:
#   from google.colab import userdata
#   hwc.configure_api(api_key=userdata.get("HYPERWAVE_API_KEY"))
#
# Otherwise, pass your key directly:
hwc.configure_api(api_key="YOUR_API_KEY_HERE")

# Extract recipes for cloud transmission
structure_recipe = structure.extract_recipe()
monitors_recipe = monitors.recipe

results = hwc.simulate(
    structure_recipe=structure_recipe,
    source_field=source_field_trimmed,
    source_offset=source_offset_corrected,
    freq_band=freq_band,
    monitors_recipe=monitors_recipe,
    mode_info=mode_info,
    simulation_steps=20000,
    add_absorption=False,          # Already baked into the structure
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
)


# %% Step 8: Analyze Results
#
# Compute transmission coefficients and visualize field propagation.

transmission = hwc.analyze_transmission(
    results,
    input_monitor="Input_o1",
    direction="x",
    print_results=True,
)

# Plot Hz field at each monitor
hwc.plot_monitors(results, component="Hz", show=False)

# Export results to CSV
hwc.export_csv(transmission, "mmi_transmission.csv")

print("Done! Results saved to mmi_transmission.csv")
