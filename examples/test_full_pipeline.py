# %% Test Full Pipeline (Tiny Structure)
# Exercises every SDK function with minimal GPU cost.

# %% Setup
import hyperwave_community as hwc
import gdsfactory as gf
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")

PDK = gf.gpdk.get_generic_pdk()
PDK.activate()

hwc.set_verbose(True)
hwc.set_debug(True)

# %% Component (smallest possible)
comp = gf.components.straight(length=5, width=0.5)
comp_extended = gf.c.extend_ports(comp, length=1.0)

RESOLUTION = 0.1  # 100nm grid, very coarse

theta, device_info = hwc.component_to_theta(
    component=comp_extended,
    resolution=RESOLUTION,
)
print(f"Theta shape: {theta.shape}")
print(f"Device size: {device_info['physical_size_um']} um")

# %% Density + Structure (coarse resolution, tiny grid)
N_CORE = 3.48
N_CLAD = 1.45
eps_core = N_CORE ** 2
eps_clad = N_CLAD ** 2

PADDING = (30, 30, 0, 0)  # minimal left/right padding

density_core = hwc.density(theta=theta, pad_width=PADDING, radius=2)
density_clad = hwc.density(theta=jnp.zeros_like(theta), pad_width=PADDING, radius=2)
print(f"Density shape: {density_core.shape}")

WG_HEIGHT_UM = 0.22
TOTAL_HEIGHT_UM = 2.0  # reduced from typical 4um
wg_cells = max(1, int(np.round(WG_HEIGHT_UM / RESOLUTION)))
clad_cells = int(np.round((TOTAL_HEIGHT_UM - WG_HEIGHT_UM) / 2 / RESOLUTION))

waveguide_layer = hwc.Layer(
    density_pattern=density_core,
    permittivity_values=(eps_clad, eps_core),
    layer_thickness=wg_cells,
)
cladding_layer = hwc.Layer(
    density_pattern=density_clad,
    permittivity_values=eps_clad,
    layer_thickness=clad_cells,
)

structure = hwc.create_structure(
    layers=[cladding_layer, waveguide_layer, cladding_layer],
    vertical_radius=1,
)

_, Lx, Ly, Lz = structure.permittivity.shape
z_wg_center = clad_cells + wg_cells // 2
print(f"Structure: ({Lx}, {Ly}, {Lz}) cells")
print(f"Total grid points: {Lx * Ly * Lz:,}")

# %% Absorber (minimal widths)
abs_params = hwc.absorber_params(
    wavelength_um=1.55,
    dx_um=RESOLUTION,
    structure_dimensions=(Lx, Ly, Lz),
)
abs_widths = tuple(abs_params["absorption_widths"])
abs_coeff = abs_params["abs_coeff"]
print(f"Absorber widths: {abs_widths}, coeff: {abs_coeff:.6f}")

absorber = hwc.create_absorption_mask(
    grid_shape=(Lx, Ly, Lz),
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
)
structure.conductivity = jnp.zeros_like(structure.conductivity) + absorber

# %% Source
WL_UM = 1.55
wl_cells = WL_UM / RESOLUTION
freq_band = (2 * jnp.pi / wl_cells, 2 * jnp.pi / wl_cells, 1)

source_pos_x = abs_widths[0]

source_field, source_offset, mode_info = hwc.create_mode_source(
    structure=structure,
    freq_band=freq_band,
    mode_num=0,
    propagation_axis="x",
    source_position=source_pos_x,
)
print(f"Source field shape: {source_field.shape}")

# %% Monitors (input + output)
monitors = hwc.MonitorSet()

MON_THICK = 3
MON_HALF_Y = min(15, Ly // 2)
MON_HALF_Z = min(15, Lz // 2)
y_center = Ly // 2
z_center = z_wg_center

input_mon = hwc.Monitor(
    shape=(MON_THICK, 2 * MON_HALF_Y, 2 * MON_HALF_Z),
    offset=(source_pos_x + 5, y_center - MON_HALF_Y, z_center - MON_HALF_Z),
)
monitors.add(input_mon, "Input_o1")

output_pos_x = Lx - abs_widths[0] - 5
output_mon = hwc.Monitor(
    shape=(MON_THICK, 2 * MON_HALF_Y, 2 * MON_HALF_Z),
    offset=(output_pos_x, y_center - MON_HALF_Y, z_center - MON_HALF_Z),
)
monitors.add(output_mon, "Output_o2")

print(f"Monitors: {monitors.list_monitors()}")

# %% Visualize everything (pre-sim)
hwc.plot_structure(structure, show_permittivity=True, show_conductivity=False,
                   axis="z", position=z_wg_center, show=False)
hwc.plot_absorption_mask(absorber, show=False)
hwc.plot_mode(
    mode_field=mode_info["field"],
    beta=mode_info["beta"],
    mode_num=0,
    propagation_axis="x",
    show=False,
)
hwc.plot_monitor_layout(
    structure.permittivity,
    monitors,
    axis="z",
    position=z_wg_center,
    source_position=source_pos_x,
    show=False,
)

print("\n=== Pre-sim summary ===")
print(f"Grid: {Lx} x {Ly} x {Lz} = {Lx * Ly * Lz:,} cells")
print("Pre-sim checks: PASS")

# %% API + Simulate (minimal steps)
import os  # noqa: E402
try:
    from google.colab import userdata
    hwc.configure_api(api_key=userdata.get("HYPERWAVE_API_KEY"))
except ImportError:
    hwc.configure_api(api_key=os.environ.get("HYPERWAVE_API_KEY"))

cost = hwc.estimate_cost(structure_shape=structure.permittivity.shape, max_steps=200)
print(f"Estimated cost: {cost}")

structure_recipe = structure.extract_recipe()
monitors_recipe = monitors.recipe

results = hwc.simulate(
    structure_recipe=structure_recipe,
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors_recipe=monitors_recipe,
    mode_info=mode_info,
    simulation_steps=200,
    add_absorption=False,
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
)

# %% Analyze
transmission = hwc.analyze_transmission(results, input_monitor="Input_o1", direction="x")

if "convergence_steps" in results and "convergence_errors" in results:
    hwc.plot_convergence(results["convergence_steps"], results["convergence_errors"], show=False)
else:
    print("No convergence data (too few steps), skipping plot_convergence")

hwc.plot_monitors(results, component="Hz", show=False)

# %% Export
hwc.export_csv(transmission, "test_transmission.csv")
hwc.export_csv(results, "test_results.csv")

print("\nALL FUNCTIONS PASSED")
