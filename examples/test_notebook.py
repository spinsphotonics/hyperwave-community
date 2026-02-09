"""Test script: all notebook code cells combined. Run to find bugs."""
import os
import pickle
import functools
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import matplotlib
matplotlib.use('Agg')

print = functools.partial(__builtins__.__dict__["print"], flush=True)

# === Cell 2: Imports ===
import hyperwave_community as hwc
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

hwc.configure_api(api_key=os.environ['HYPERWAVE_API_KEY'])

# === Cell 4: Physical parameters ===
import math

n_si = 3.48
n_sio2 = 1.44
n_clad = 1.44
n_air = 1.0

wavelength_um = 1.55

h_dev = 0.220
etch_depth = 0.110
h_box = 2.0
h_clad = 0.78
h_sub = 0.8
h_air = 1.0
pad = 3.0

dx = 0.070           # 70nm structure grid (fast iteration; 35nm for production)
pixel_size = dx / 2   # 35nm theta grid
domain = 40.0

wg_width = 0.5
wg_length = 5.0

beam_waist = 5.2
fiber_angle = 14.5

Lx = int(domain / dx)
Ly = Lx

theta_Lx = 2 * Lx
theta_Ly = 2 * Ly

h_p = pad / dx
h0 = h_air / dx
h1 = h_clad / dx
h2 = etch_depth / dx
h3 = (h_dev - etch_depth) / dx
h4 = h_box / dx
h5 = h_sub / dx
Lz = int(math.ceil(h_p + h0 + h1 + h2 + h3 + h4 + h5 + h_p))

z_etch = int(round(h_p + h0 + h1))
z_slab = z_etch + int(round(h2))
z_box = z_slab + int(round(h3))

wl_px = wavelength_um / dx
freq = 2 * np.pi / wl_px
freq_band = (freq, freq, 1)

eps_si = n_si**2
eps_sio2 = n_sio2**2
eps_clad = n_clad**2
eps_air = n_air**2

DENSITY_RADIUS = 3
DENSITY_ALPHA = 0.8
DESIGN_LAYER = 3

print(f"Structure grid: {Lx} x {Ly} x {Lz} ({dx * 1000:.0f} nm)")
print(f"Theta grid: {theta_Lx} x {theta_Ly} ({pixel_size * 1000:.1f} nm)")
print(f"Layers (px): pad={h_p:.2f} air={h0:.2f} clad={h1:.2f} "
      f"etch={h2:.2f} slab={h3:.2f} BOX={h4:.2f} sub={h5:.2f} pad={h_p:.2f}")

# === Cell 6: Theta ===
wg_len = int(round(wg_length / dx))
wg_hw = int(round(wg_width / 2 / dx))

wg_len_theta = int(round(wg_length / pixel_size))
wg_hw_theta = int(round(wg_width / 2 / pixel_size))

abs_margin = int(round(80 * 0.05 / dx))  # scale from 50nm reference
abs_margin_theta = 2 * abs_margin
design_region = {
    'x_start': wg_len_theta,
    'x_end': theta_Lx - abs_margin_theta,
    'y_start': abs_margin_theta,
    'y_end': theta_Ly - abs_margin_theta,
}

theta_init = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
theta_init[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0
dr = design_region
theta_init[dr['x_start']:dr['x_end'], dr['y_start']:dr['y_end']] = 0.5

print(f"Theta shape: {theta_init.shape}")
print(f"Design region: {(dr['x_end'] - dr['x_start']) * pixel_size:.1f} x "
      f"{(dr['y_end'] - dr['y_start']) * pixel_size:.1f} um")

# === Cell 8: Structure recipe ===
density_top = hwc.density(theta=jnp.array(theta_init), pad_width=0, alpha=DENSITY_ALPHA, radius=DENSITY_RADIUS)

slab_density = hwc.density(jnp.zeros((theta_Lx, theta_Ly)), pad_width=0)

# Layer 4 (dev/slab) is scalar eps_si -- continuous silicon film below partial etch
recipe = hwc.recipe_from_params(
    grid_shape=density_top.shape,
    layers=[
        {'density': np.array(slab_density),   'permittivity': eps_air,              'thickness': h_p},
        {'density': np.array(slab_density),   'permittivity': eps_air,              'thickness': h0},
        {'density': np.array(slab_density),   'permittivity': eps_clad,             'thickness': h1},
        {'density': np.array(density_top),    'permittivity': (eps_clad, eps_si),   'thickness': h2},
        {'density': np.array(slab_density),   'permittivity': eps_si,               'thickness': h3},
        {'density': np.array(slab_density),   'permittivity': eps_sio2,             'thickness': h4},
        {'density': np.array(slab_density),   'permittivity': eps_si,               'thickness': h5},
        {'density': np.array(slab_density),   'permittivity': eps_si,               'thickness': h_p},
    ],
    vertical_radius=0,
)

Lz_recipe = recipe['metadata']['final_shape'][3]
z_dev = z_etch + int(h2 // 2)

print(f"Structure grid: {Lx} x {Ly} x {Lz_recipe} ({dx * 1000:.0f} nm)")
print(f"Etch z={z_etch} ({z_etch * dx:.3f} um), slab z={z_slab} ({z_slab * dx:.3f} um)")

# === Cell 10: Source ===
source_above_surface_um = 0.05
source_z = int(round((pad + h_air - source_above_surface_um) / dx))

grating_x = int(round((dr['x_start'] + dr['x_end']) / 2 * pixel_size / dx))
grating_y = Ly // 2
waist_px = beam_waist / dx

print(f"Generating Gaussian source on cloud GPU...")
source_field, input_power = hwc.generate_gaussian_source(
    sim_shape=(Lx, Ly, Lz),
    frequencies=np.array([freq]),
    source_pos=(grating_x, grating_y, source_z),
    waist_radius=waist_px,
    theta=-fiber_angle,
    phi=0.0,
    polarization='y',
    max_steps=5000,
    gpu_type="B200",
)

source_offset = (0, 0, source_z)
input_power = float(np.mean(input_power))

print(f"Source shape: {source_field.shape}")
print(f"Source offset: {source_offset}")
print(f"Input power: {input_power:.6f}")

# === Cell 12: Mode computation ===
print("Computing waveguide mode...")

small_x_theta = 40
theta_mode_wg = np.zeros((small_x_theta, theta_Ly), dtype=np.float32)
theta_mode_wg[:, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

density_mode_top = hwc.density(
    theta=jnp.array(theta_mode_wg), pad_width=0,
    alpha=DENSITY_ALPHA, radius=DENSITY_RADIUS,
)
theta_mode_bottom = np.where(theta_mode_wg > 0, 1.0, 0.0).astype(np.float32)
density_mode_bottom = hwc.density(
    theta=jnp.array(theta_mode_bottom), pad_width=0,
    alpha=DENSITY_ALPHA, radius=DENSITY_RADIUS,
)
slab_density_mode = hwc.density(jnp.zeros((small_x_theta, theta_Ly)), pad_width=0)

wg_structure = hwc.create_structure(layers=[
    hwc.Layer(density_pattern=slab_density_mode, permittivity_values=eps_air, layer_thickness=h_p),
    hwc.Layer(density_pattern=slab_density_mode, permittivity_values=eps_air, layer_thickness=h0),
    hwc.Layer(density_pattern=slab_density_mode, permittivity_values=eps_clad, layer_thickness=h1),
    hwc.Layer(density_pattern=density_mode_top, permittivity_values=(eps_clad, eps_si), layer_thickness=h2),
    hwc.Layer(density_pattern=density_mode_bottom, permittivity_values=(eps_clad, eps_si), layer_thickness=h3),
    hwc.Layer(density_pattern=slab_density_mode, permittivity_values=eps_sio2, layer_thickness=h4),
    hwc.Layer(density_pattern=slab_density_mode, permittivity_values=eps_si, layer_thickness=h5),
    hwc.Layer(density_pattern=slab_density_mode, permittivity_values=eps_si, layer_thickness=h_p),
], vertical_radius=0)

eps_wg = np.array(wg_structure.permittivity)
Lz_wg = eps_wg.shape[3]
Ly_perm = eps_wg.shape[2]
print(f"WG structure shape: {eps_wg.shape}")

x_center = eps_wg.shape[1] // 2
eps_yz = eps_wg[0, x_center, :, :]

crop_y = min(50, Ly_perm // 4)
crop_z = min(30, Lz_wg // 4)
y_center_idx = Ly_perm // 2
y_start = y_center_idx - crop_y
y_end = y_center_idx + crop_y
z_start = max(0, z_etch - crop_z)
z_end = min(Lz_wg, z_box + crop_z)

eps_cropped = eps_yz[y_start:y_end, z_start:z_end]
print(f"Cropped eps for mode solve: {eps_cropped.shape}")

eps_4d = jnp.stack([jnp.array(eps_cropped)] * 3, axis=0)[:, jnp.newaxis, :, :]

from hyperwave_community.mode_solver import mode as hwc_mode
mode_E_field, beta_arr, _err = hwc_mode(
    freq_band=freq_band,
    permittivity=eps_4d,
    axis=0,
    mode_num=0,
)
n_eff_mode = float(beta_arr[0]) / (2 * np.pi / wl_px)
print(f"n_eff = {n_eff_mode:.4f}")
assert 2.0 < n_eff_mode < 3.0, f"n_eff={n_eff_mode:.4f} out of range"

mode_E_cropped = mode_E_field[0:1, 0:3, :, :, :]
mode_full_cropped = hwc.mode_convert(
    mode_E_field=mode_E_cropped,
    freq_band=freq_band,
    permittivity_slice=np.array(eps_cropped),
    propagation_axis='x',
    gpu_type="B200",
)
print(f"Mode field (cropped): {mode_full_cropped.shape}")

mode_e = np.array(mode_full_cropped[0, 0:3, 0, :, :])
mode_h = np.array(mode_full_cropped[0, 3:6, 0, :, :])
cross = np.cross(mode_e, np.conj(mode_h), axis=0)
P_mode_cross = float(np.abs(np.real(np.sum(cross[0, :, :]))))
print(f"P_mode_cross = {P_mode_cross:.6f}")

mode_field_full = np.zeros((1, 6, 1, Ly_perm, Lz_wg), dtype=np.complex64)
mode_field_full[:, :, :, y_start:y_end, z_start:z_end] = np.array(mode_full_cropped)

print(f"Mode field shape: {mode_field_full.shape}")
print(f"P_mode_cross: {P_mode_cross:.6f}")
print(f"n_eff: {n_eff_mode:.4f}")

# === Cell 14: Forward simulation ===
abs_params = hwc.get_optimized_absorber_params(
    resolution_nm=dx * 1000,
    wavelength_um=wavelength_um,
    structure_dimensions=(Lx, Ly, Lz),
)
abs_widths = abs_params['absorption_widths']
abs_coeff = abs_params['absorber_coeff']

monitors = hwc.MonitorSet()
output_x = abs_widths[0] + 10

monitors.add(hwc.Monitor(shape=(Lx, Ly, 1), offset=(0, 0, z_dev)), name='Output_xy_device')
monitors.add(hwc.Monitor(shape=(Lx, 1, Lz), offset=(0, Ly // 2, 0)), name='Output_xz_center')
monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(Lx // 2, 0, 0)), name='Output_yz_center')
monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(output_x, 0, 0)), name='Output_wg_output')

print(f"Running forward simulation...")
fwd_results = hwc.simulate(
    structure_recipe=recipe,
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors_recipe=monitors.recipe,
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
    gpu_type="B200",
    simulation_steps=10000,
)
print(f"Forward sim complete: {fwd_results['sim_time']:.1f}s GPU time")

wg_field = np.array(fwd_results['monitor_data']['Output_wg_output'])
S = hwc.S_from_slice(jnp.mean(jnp.array(wg_field), axis=2))
power = float(jnp.abs(jnp.sum(S[0, 0, :, :])))
print(f"Waveguide output power: {power:.6f}")
print(f"Coupling (approx): {power / input_power * 100:.1f}%")

# === Cell 17: Optimization setup ===
structure_spec = {
    'layers_info': [
        {'permittivity_values': float(eps_air),              'layer_thickness': float(h_p), 'density_radius': 0, 'density_alpha': 0},
        {'permittivity_values': float(eps_air),              'layer_thickness': float(h0),  'density_radius': 0, 'density_alpha': 0},
        {'permittivity_values': float(eps_clad),             'layer_thickness': float(h1),  'density_radius': 0, 'density_alpha': 0},
        {'permittivity_values': [float(eps_clad), float(eps_si)], 'layer_thickness': float(h2),  'density_radius': DENSITY_RADIUS, 'density_alpha': DENSITY_ALPHA},
        {'permittivity_values': float(eps_si),               'layer_thickness': float(h3),  'density_radius': 0, 'density_alpha': 0},
        {'permittivity_values': float(eps_sio2),             'layer_thickness': float(h4),  'density_radius': 0, 'density_alpha': 0},
        {'permittivity_values': float(eps_si),               'layer_thickness': float(h5),  'density_radius': 0, 'density_alpha': 0},
        {'permittivity_values': float(eps_si),               'layer_thickness': float(h_p), 'density_radius': 0, 'density_alpha': 0},
    ],
    'construction_params': {'vertical_radius': 0},
}

loss_monitor_shape = (1, Ly, Lz)
loss_monitor_offset = (output_x, 0, 0)

design_monitor_shape = (Lx, Ly, int(round(h2)))
design_monitor_offset = (0, 0, z_etch)

waveguide_mask = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
waveguide_mask[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

NUM_STEPS = 2
LR = 0.01
GRAD_CLIP = 1.0

print(f"Loss monitor at x={loss_monitor_offset[0]} ({loss_monitor_offset[0] * dx:.1f} um)")
print(f"Design monitor: {design_monitor_shape}")

# === Cell 19: Optimization loop ===
print(f"Running optimization ({NUM_STEPS} steps)...")
print(f"Loss: Mode coupling (maximize efficiency)")
results = []

for step_result in hwc.run_optimization(
    theta=theta_init,
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    structure_spec=structure_spec,
    loss_monitor_shape=loss_monitor_shape,
    loss_monitor_offset=loss_monitor_offset,
    design_monitor_shape=design_monitor_shape,
    design_monitor_offset=design_monitor_offset,
    mode_field=mode_field_full,
    input_power=input_power,
    mode_cross_power=P_mode_cross,
    mode_axis=0,
    waveguide_mask=waveguide_mask,
    num_steps=NUM_STEPS,
    learning_rate=LR,
    grad_clip_norm=GRAD_CLIP,
    cosine_decay_alpha=0.1,
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
    gpu_type="B200",
):
    results.append(step_result)
    loss = step_result['loss']
    eff_pct = abs(loss) * 100
    print(f"Step {step_result['step']:3d}/{NUM_STEPS}:  eff = {eff_pct:.2f}%  "
          f"|grad|_max = {step_result['grad_max']:.3e}  ({step_result['step_time']:.1f}s)")

efficiencies = [abs(r['loss']) * 100 for r in results]
best_idx = int(np.argmax(efficiencies))
best_eff = efficiencies[best_idx]
print(f"\nBest: {best_eff:.2f}% ({-10 * np.log10(max(best_eff / 100, 1e-10)):.2f} dB) at step {best_idx + 1}")
theta_final = results[-1]['theta']

# === Cell 21: Results ===
best_theta = results[best_idx]['theta']
print(f"Best theta shape: {best_theta.shape}")

# === Cell 23: Verification forward sim ===
density_best = hwc.density(theta=jnp.array(best_theta), pad_width=0, alpha=DENSITY_ALPHA, radius=DENSITY_RADIUS)

recipe_best = hwc.recipe_from_params(
    grid_shape=density_best.shape,
    layers=[
        {'density': np.array(slab_density),  'permittivity': eps_air,            'thickness': h_p},
        {'density': np.array(slab_density),  'permittivity': eps_air,            'thickness': h0},
        {'density': np.array(slab_density),  'permittivity': eps_clad,           'thickness': h1},
        {'density': np.array(density_best),  'permittivity': (eps_clad, eps_si), 'thickness': h2},
        {'density': np.array(slab_density),  'permittivity': eps_si,             'thickness': h3},
        {'density': np.array(slab_density),  'permittivity': eps_sio2,           'thickness': h4},
        {'density': np.array(slab_density),  'permittivity': eps_si,             'thickness': h5},
        {'density': np.array(slab_density),  'permittivity': eps_si,             'thickness': h_p},
    ],
    vertical_radius=0,
)

print(f"Running verification forward sim...")
opt_results = hwc.simulate(
    structure_recipe=recipe_best,
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors_recipe=monitors.recipe,
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
    gpu_type="B200",
    simulation_steps=10000,
)

wg_field = np.array(opt_results['monitor_data']['Output_wg_output'])
S = hwc.S_from_slice(jnp.mean(jnp.array(wg_field), axis=2))
wg_power = float(jnp.abs(jnp.sum(S[0, 0, :, :])))
eff_pct = wg_power / input_power * 100
loss_dB = -10 * np.log10(max(eff_pct / 100, 1e-10))
print(f"Coupling efficiency: {eff_pct:.2f}% ({loss_dB:.2f} dB)")

print("\n=== ALL CELLS PASSED ===")
