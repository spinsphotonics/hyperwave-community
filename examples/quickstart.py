# %% [markdown]
# # Hyperwave Quickstart: 2x2 MMI with S-Bends
#
# Simulate a 2x2 multimode interference coupler using gdsfactory for layout
# and Hyperwave for cloud-accelerated 3D FDTD.
#
# **What you'll learn:**
# 1. Convert a gdsfactory component to a simulation-ready structure
# 2. Set up mode sources, monitors, and absorbing boundaries
# 3. Run a cloud GPU simulation and analyze transmission

# %% Installation
# pip install hyperwave-community gdsfactory

# %% Imports
import hyperwave_community as hwc
import gdsfactory as gf
import numpy as np
import jax.numpy as jnp

hwc.set_verbose(True)

PDK = gf.gpdk.get_generic_pdk()
PDK.activate()


# %% Step 1: Load Component
#
# Load a 2x2 MMI with S-bends from gdsfactory and extend ports so the mode
# source and monitors sit inside straight waveguide sections.

RESOLUTION_UM = 0.02          # 20 nm grid spacing
EXTENSION_LENGTH = 2.0        # Extend ports by 2 um

gf_device = gf.components.mmi2x2_with_sbend()
gf_extended = gf.c.extend_ports(gf_device, length=EXTENSION_LENGTH)

theta, device_info = hwc.component_to_theta(
    component=gf_extended,
    resolution=RESOLUTION_UM,
)


# %% Step 2: Build 3D Structure
#
# Apply density filtering, then stack cladding and waveguide layers into
# a 3D permittivity volume.

N_CORE = 3.48                  # Silicon refractive index at 1550 nm
N_CLAD = 1.45                  # SiO2 cladding
eps_core = N_CORE ** 2
eps_clad = N_CLAD ** 2

PADDING = (100, 100, 0, 0)    # (left, right, top, bottom) in theta pixels

density_core = hwc.density(theta=theta, pad_width=PADDING, radius=3)
density_clad = hwc.density(theta=jnp.zeros_like(theta), pad_width=PADDING, radius=5)

WG_HEIGHT_UM = 0.22
TOTAL_HEIGHT_UM = 4.0
wg_cells = max(1, int(np.round(WG_HEIGHT_UM / RESOLUTION_UM)))
clad_cells = int(np.round((TOTAL_HEIGHT_UM - WG_HEIGHT_UM) / 2 / RESOLUTION_UM))

structure = hwc.create_structure(
    layers=[
        hwc.Layer(density_pattern=density_clad, permittivity_values=eps_clad, layer_thickness=clad_cells),
        hwc.Layer(density_pattern=density_core, permittivity_values=(eps_clad, eps_core), layer_thickness=wg_cells),
        hwc.Layer(density_pattern=density_clad, permittivity_values=eps_clad, layer_thickness=clad_cells),
    ],
    vertical_radius=2,
)

_, Lx, Ly, Lz = structure.permittivity.shape
z_wg_center = clad_cells + wg_cells // 2

hwc.plot_structure(structure, axis="z", position=z_wg_center)


# %% Step 3: Absorbing Boundaries
#
# Add adiabatic absorbers at grid edges to prevent reflections.

abs_params = hwc.absorber_params(
    wavelength_um=1.55,
    dx_um=RESOLUTION_UM,
    structure_dimensions=(Lx, Ly, Lz),
)
abs_widths = tuple(abs_params["absorption_widths"])
abs_coeff = abs_params["abs_coeff"]

absorber = hwc.create_absorption_mask(
    grid_shape=(Lx, Ly, Lz),
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
)
structure.conductivity = jnp.zeros_like(structure.conductivity) + absorber

hwc.plot_absorption_mask(absorber)


# %% Step 4: Mode Source
#
# Solve for the fundamental TE mode at the input waveguide.

WL_UM = 1.55
wl_cells = WL_UM / RESOLUTION_UM
freq_band = (2 * jnp.pi / wl_cells, 2 * jnp.pi / wl_cells, 1)

source_pos_x = abs_widths[0]

source_field, source_offset, mode_info = hwc.create_mode_source(
    structure=structure,
    freq_band=freq_band,
    mode_num=0,
    propagation_axis="x",
    source_position=source_pos_x,
)

hwc.plot_mode(
    mode_field=mode_info["field"],
    beta=mode_info["beta"],
    mode_num=0,
    propagation_axis="x",
)


# %% Step 5: Monitors
#
# Place field monitors at each port. Names starting with "Input_" and
# "Output_" are recognized by analyze_transmission.

monitors = hwc.MonitorSet()

y_pad_struct = PADDING[0] // 2
bbox = device_info["bounding_box_um"]
x_min_um, y_min_um = bbox[0], bbox[1]
theta_res = device_info["theta_resolution_um"]

MONITOR_THICKNESS = 5
MONITOR_HALF = 35

for port in gf_device.ports:
    px_um, py_um = port.center
    x_struct = int((px_um - x_min_um) / theta_res / 2) + PADDING[2] // 2
    y_struct = int((py_um - y_min_um) / theta_res / 2) + y_pad_struct

    label = f"Input_{port.name}" if abs(port.orientation % 360 - 180) < 1 else f"Output_{port.name}"

    monitors.add(
        hwc.Monitor(
            shape=(MONITOR_THICKNESS, 2 * MONITOR_HALF, 2 * MONITOR_HALF),
            offset=(x_struct, y_struct - MONITOR_HALF, z_wg_center - MONITOR_HALF),
        ),
        label,
    )

# Full XY plane for field visualization
monitors.add(hwc.Monitor(shape=(Lx, Ly, 1), offset=(0, 0, z_wg_center)), "xy_mid")

hwc.plot_monitor_layout(
    structure.permittivity, monitors,
    axis="z", position=z_wg_center, source_position=source_pos_x,
)


# %% Step 6: Simulate
#
# Configure your API key and run the simulation on cloud GPU.
# Sign up at https://spinsphotonics.com/signup to get your key.

try:
    from google.colab import userdata
    hwc.configure_api(api_key=userdata.get("HYPERWAVE_API_KEY"))
except ImportError:
    import os
    hwc.configure_api(api_key=os.environ.get("HYPERWAVE_API_KEY"))

results = hwc.simulate(
    structure_recipe=structure.extract_recipe(),
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors_recipe=monitors.recipe,
    mode_info=mode_info,
    simulation_steps=20000,
    add_absorption=False,
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
)


# %% Step 7: Analyze Results

transmission = hwc.analyze_transmission(
    results, input_monitor="Input_o1", direction="x",
)

hwc.plot_monitors(results, component="Hz")

hwc.export_csv(transmission, "quickstart_transmission.csv")
