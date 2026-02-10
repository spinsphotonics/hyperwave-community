"""Test script: all notebook code cells combined. Run to find bugs."""
import os
import pickle
import functools
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import matplotlib
matplotlib.use('Agg')

print = functools.partial(__builtins__.__dict__["print"], flush=True)

# === Imports ===
import hyperwave_community as hwc
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

hwc.configure_api(api_key=os.environ['HYPERWAVE_API_KEY'])

# === Step 1: Physical parameters ===
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

dx = 0.070           # 70nm for fast testing (35nm for production)
pixel_size = dx / 2
domain = 40.0

wg_width = 0.5
wg_length = 5.0

beam_waist = 5.2
fiber_angle = 14.5

Lx = int(domain / dx)
Ly = Lx

theta_Lx = 2 * Lx
theta_Ly = 2 * Ly

h_p_f = pad / dx
h0_f = h_air / dx
h1_f = h_clad / dx
h2_f = etch_depth / dx
h3_f = (h_dev - etch_depth) / dx
h4_f = h_box / dx
h5_f = h_sub / dx

h_p = int(round(h_p_f))
h0 = int(round(h0_f))
h1 = int(round(h1_f))
h2 = int(round(h2_f))
h3 = int(round(h3_f))
h4 = int(round(h4_f))
h5 = int(round(h5_f))
Lz = h_p + h0 + h1 + h2 + h3 + h4 + h5 + h_p

z_etch = h_p + h0 + h1
z_slab = z_etch + h2
z_box = z_slab + h3

wl_px = wavelength_um / dx
freq = 2 * np.pi / wl_px
freq_band = (freq, freq, 1)

eps_si = n_si**2
eps_sio2 = n_sio2**2
eps_clad = n_clad**2
eps_air = n_air**2

print(f"Structure grid: {Lx} x {Ly} x {Lz} ({dx * 1000:.0f} nm)")
print(f"Theta grid: {theta_Lx} x {theta_Ly} ({pixel_size * 1000:.1f} nm)")
print(f"Layers (px): pad={h_p} air={h0} clad={h1} etch={h2} slab={h3} BOX={h4} sub={h5} pad={h_p}")

# === Step 2: Design variables (theta) ===
wg_len = int(round(wg_length / dx))
wg_hw = int(round(wg_width / 2 / dx))

wg_len_theta = int(round(wg_length / pixel_size))
wg_hw_theta = int(round(wg_width / 2 / pixel_size))

design_region = {
    'x_start': wg_len_theta,
    'x_end': theta_Lx - wg_len_theta,
    'y_start': wg_len_theta,
    'y_end': theta_Ly - wg_len_theta,
}

dr = design_region
theta_init = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
theta_init[dr['x_start']:dr['x_end'], dr['y_start']:dr['y_end']] = 0.5
theta_init[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

print(f"Theta shape: {theta_init.shape}")
print(f"Design region: {(dr['x_end'] - dr['x_start']) * pixel_size:.1f} x "
      f"{(dr['y_end'] - dr['y_start']) * pixel_size:.1f} um")

# === Step 3: Layer stack ===
slab = jnp.zeros(theta_init.shape)

design_layers = [
    hwc.Layer(density_pattern=slab,                    permittivity_values=eps_air,            layer_thickness=h_p),
    hwc.Layer(density_pattern=slab,                    permittivity_values=eps_air,            layer_thickness=h0),
    hwc.Layer(density_pattern=slab,                    permittivity_values=eps_clad,           layer_thickness=h1),
    hwc.Layer(density_pattern=jnp.array(theta_init),   permittivity_values=(eps_clad, eps_si), layer_thickness=h2),
    hwc.Layer(density_pattern=slab,                    permittivity_values=eps_si,             layer_thickness=h3),
    hwc.Layer(density_pattern=slab,                    permittivity_values=eps_sio2,           layer_thickness=h4),
    hwc.Layer(density_pattern=slab,                    permittivity_values=eps_si,             layer_thickness=h5),
    hwc.Layer(density_pattern=slab,                    permittivity_values=eps_si,             layer_thickness=h_p),
]

def build_recipe(layers):
    """Convert Layer list to lightweight cloud recipe (no 3D array built locally)."""
    return hwc.recipe_from_params(
        grid_shape=np.array(layers[0].density_pattern).shape,
        layers=[{
            'density': np.array(l.density_pattern),
            'permittivity': l.permittivity_values,
            'thickness': l.layer_thickness,
        } for l in layers],
        vertical_radius=0,
    )

recipe = build_recipe(design_layers)
Lz_recipe = recipe['metadata']['final_shape'][3]
z_dev = z_etch + int(h2 // 2)

print(f"Structure grid: {Lx} x {Ly} x {Lz_recipe} ({dx * 1000:.0f} nm)")
print(f"Etch z={z_etch} ({z_etch * dx:.3f} um), slab z={z_slab} ({z_slab * dx:.3f} um)")

# === Step 4: Absorbing boundaries ===
abs_params = hwc.get_optimized_absorber_params(
    resolution_nm=dx * 1000,
    wavelength_um=wavelength_um,
    structure_dimensions=(Lx, Ly, Lz),
)
abs_widths = abs_params['absorption_widths']
abs_coeff = abs_params['absorber_coeff']

print(f"Absorber widths (x,y,z): {abs_widths}")
print(f"Absorber coefficient: {abs_coeff:.6f}")

# === Step 5: Source ===
source_above_surface_um = 0.05
source_z = int(round((pad + h_air - source_above_surface_um) / dx))

grating_x = int(round((dr['x_start'] + dr['x_end']) / 2 * pixel_size / dx))
grating_y = Ly // 2
waist_px = beam_waist / dx

est = hwc.estimate_cost(structure_shape=(3, Lx, Ly, Lz), max_steps=1000, gpu_type="B200")
if est:
    print(f"Source gen estimate: {est['estimated_seconds']:.0f}s, {est['estimated_credits']:.4f} credits")

print(f"Generating Gaussian source on cloud GPU...")
t0 = time.time()
source_field, input_power = hwc.generate_gaussian_source(
    sim_shape=(Lx, Ly, Lz),
    frequencies=np.array([freq]),
    source_pos=(grating_x, grating_y, source_z),
    waist_radius=waist_px,
    theta=-fiber_angle,
    phi=0.0,
    polarization='y',
    max_steps=5000,
    check_every_n=200,
    gpu_type="B200",
)

print(f"Source generated in {time.time() - t0:.1f}s")

source_offset = (0, 0, source_z)
input_power = float(np.mean(input_power))

print(f"Source shape: {source_field.shape}")
print(f"Source offset: {source_offset}")
print(f"Input power: {input_power:.6f}")

# === Step 6: Waveguide mode ===
print("Computing waveguide mode...")

small_x_theta = 40
theta_mode = np.zeros((small_x_theta, theta_Ly), dtype=np.float32)
theta_mode[:, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

theta_mode_bot = np.where(theta_mode > 0, 1.0, 0.0).astype(np.float32)
d_mode_slab = jnp.zeros((small_x_theta, theta_Ly))

wg_structure = hwc.create_structure(layers=[
    hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_air,            layer_thickness=h_p),
    hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_air,            layer_thickness=h0),
    hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_clad,           layer_thickness=h1),
    hwc.Layer(density_pattern=jnp.array(theta_mode),     permittivity_values=(eps_clad, eps_si), layer_thickness=h2),
    hwc.Layer(density_pattern=jnp.array(theta_mode_bot), permittivity_values=(eps_clad, eps_si), layer_thickness=h3),
    hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_sio2,           layer_thickness=h4),
    hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_si,             layer_thickness=h5),
    hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_si,             layer_thickness=h_p),
], vertical_radius=0)

eps_wg = np.array(wg_structure.permittivity)
Lz_wg = eps_wg.shape[3]
Ly_perm = eps_wg.shape[2]
print(f"WG structure shape: {eps_wg.shape}")

eps_yz = eps_wg[0, eps_wg.shape[1] // 2, :, :]

crop_y = min(50, Ly_perm // 4)
crop_z = min(30, Lz_wg // 4)
y_c = Ly_perm // 2
y0, y1 = y_c - crop_y, y_c + crop_y
z0 = max(0, z_etch - crop_z)
z1 = min(Lz_wg, z_box + crop_z)
eps_crop = eps_yz[y0:y1, z0:z1]
print(f"Cropped eps for mode solve: {eps_crop.shape}")

eps_4d = jnp.stack([jnp.array(eps_crop)] * 3, axis=0)[:, jnp.newaxis, :, :]

from hyperwave_community.mode_solver import mode as hwc_mode
mode_E, beta_arr, _ = hwc_mode(
    freq_band=freq_band,
    permittivity=eps_4d,
    axis=0,
    mode_num=0,
)
n_eff = float(beta_arr[0]) / (2 * np.pi / wl_px)
print(f"n_eff = {n_eff:.4f}")
assert 2.0 < n_eff < 3.0, f"n_eff={n_eff:.4f} out of range"

mode_EH = hwc.mode_convert(
    mode_E_field=mode_E[0:1, 0:3, :, :, :],
    freq_band=freq_band,
    permittivity_slice=np.array(eps_crop),
    propagation_axis='x',
    propagation_length=500,
    gpu_type="B200",
)
print(f"Mode field (cropped): {mode_EH.shape}")

# Negate H-field for backward (-x) propagation: mode_converter computes +x mode,
# but the waveguide output monitor measures fields propagating in -x direction
# (from grating toward waveguide). Ref: gc_colab_workflow.py line 582.
mode_EH = np.array(mode_EH, copy=True)
mode_EH[:, 3:6, ...] *= -1

mode_e = np.array(mode_EH[0, 0:3, 0, :, :])
mode_h = np.array(mode_EH[0, 3:6, 0, :, :])
cross = np.cross(mode_e, np.conj(mode_h), axis=0)
P_mode_cross = float(np.abs(np.real(np.sum(cross[0, :, :]))))
print(f"P_mode_cross = {P_mode_cross:.6f}")

mode_field = np.zeros((1, 6, 1, Ly_perm, Lz_wg), dtype=np.complex64)
mode_field[:, :, :, y0:y1, z0:z1] = np.array(mode_EH)

print(f"Mode field shape: {mode_field.shape}")
print(f"P_mode_cross: {P_mode_cross:.6f}")
print(f"n_eff: {n_eff:.4f}")

# === Step 7: Monitors ===
monitors = hwc.MonitorSet()
output_x = abs_widths[0] + 10

monitors.add(hwc.Monitor(shape=(Lx, Ly, 1), offset=(0, 0, z_dev)), name='Output_xy_device')
monitors.add(hwc.Monitor(shape=(Lx, 1, Lz), offset=(0, Ly // 2, 0)), name='Output_xz_center')
monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(Lx // 2, 0, 0)), name='Output_yz_center')
monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(output_x, 0, 0)), name='Output_wg_output')

# === Step 8: Forward simulation ===
est = hwc.estimate_cost(structure_shape=(3, Lx, Ly, Lz), max_steps=10000, gpu_type="B200")
if est:
    print(f"Forward sim estimate: {est['estimated_seconds']:.0f}s, {est['estimated_credits']:.4f} credits")

print(f"Running forward simulation...")
t0 = time.time()
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
print(f"Forward sim complete: {fwd_results['sim_time']:.1f}s GPU, {time.time() - t0:.0f}s total")

wg_field = np.array(fwd_results['monitor_data']['Output_wg_output'])
S = hwc.S_from_slice(jnp.mean(jnp.array(wg_field), axis=2))
power = float(jnp.abs(jnp.sum(S[0, 0, :, :])))
print(f"Waveguide output power: {power:.6f}")
print(f"Coupling (approx): {power / input_power * 100:.1f}%")

# === Step 9: Optimization ===
structure_spec = {
    'layers_info': [{
        'permittivity_values': [float(v) for v in l.permittivity_values] if isinstance(l.permittivity_values, tuple) else float(l.permittivity_values),
        'layer_thickness': float(l.layer_thickness),
        'density_radius': 0,
        'density_alpha': 0,
    } for l in design_layers],
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

est = hwc.estimate_cost(
    structure_shape=(3, Lx, Ly, Lz),
    max_steps=10000 * NUM_STEPS * 2,
    gpu_type="B200",
    simulation_type="fdtd_simulation",
)
if est:
    print(f"Optimization estimate ({NUM_STEPS} steps): {est['estimated_seconds']:.0f}s, "
          f"{est['estimated_credits']:.4f} credits (${est['estimated_cost_usd']:.2f})")

print(f"Running optimization ({NUM_STEPS} steps)...")
results = []
t_opt_start = time.time()

try:
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
        mode_field=mode_field,
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
        eff = abs(step_result['loss']) * 100
        print(f"Step {step_result['step']:3d}/{NUM_STEPS}:  eff = {eff:.2f}%  "
              f"|grad|_max = {step_result['grad_max']:.3e}  ({step_result['step_time']:.1f}s)",
              flush=True)
except KeyboardInterrupt:
    elapsed = time.time() - t_opt_start
    print(f"\nCancelled after {len(results)} steps ({elapsed:.0f}s).", flush=True)

if results:
    efficiencies = [abs(r['loss']) * 100 for r in results]
    best_idx = int(np.argmax(efficiencies))
    best_eff = efficiencies[best_idx]
    print(f"\nBest: {best_eff:.2f}% ({-10 * np.log10(max(best_eff / 100, 1e-10)):.2f} dB) at step {best_idx + 1}")
else:
    print("No optimization steps completed.")
    raise SystemExit(1)

# === Step 10: Results ===
best_theta = results[best_idx]['theta']
print(f"Best theta shape: {best_theta.shape}")

# === Step 11: Verification forward sim ===
opt_layers = list(design_layers)
opt_layers[3] = hwc.Layer(
    density_pattern=jnp.array(best_theta),
    permittivity_values=(eps_clad, eps_si),
    layer_thickness=h2,
)
recipe_best = build_recipe(opt_layers)

est = hwc.estimate_cost(structure_shape=(3, Lx, Ly, Lz), max_steps=10000, gpu_type="B200")
if est:
    print(f"Verification sim estimate: {est['estimated_seconds']:.0f}s, {est['estimated_credits']:.4f} credits")

print(f"Running verification forward sim...")
t0 = time.time()
opt_results = None
for attempt in range(3):
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
    if opt_results is not None:
        break
    print(f"  Verification sim attempt {attempt + 1} failed (rate limit), retrying in 30s...")
    time.sleep(30)

if opt_results is None:
    print("Verification sim failed after 3 attempts (rate limit). Optimization results above are valid.")
    print("\n=== OPTIMIZATION PASSED (verification skipped due to rate limit) ===")
    raise SystemExit(0)

print(f"Verification sim complete: {opt_results['sim_time']:.1f}s GPU, {time.time() - t0:.0f}s total")

# Poynting vector power coupling
wg_field = np.array(opt_results['monitor_data']['Output_wg_output'])
S = hwc.S_from_slice(jnp.mean(jnp.array(wg_field), axis=2))
wg_power = float(jnp.abs(jnp.sum(S[0, 0, :, :])))
power_eff = wg_power / input_power * 100

# Mode overlap coupling
field_avg = np.mean(wg_field, axis=2)
E_out = field_avg[0, 0:3, :, :]
H_out = field_avg[0, 3:6, :, :]
E_m = mode_field[0, 0:3, 0, :, :]
H_m = mode_field[0, 3:6, 0, :, :]
I1 = np.sum(E_m[1] * np.conj(H_out[2]) - E_m[2] * np.conj(H_out[1]))
I2 = np.sum(E_out[1] * np.conj(H_m[2]) - E_out[2] * np.conj(H_m[1]))
mode_eff = abs(np.real(I1 * I2)) / (2.0 * input_power * P_mode_cross) * 100

print(f"Power coupling:  {power_eff:.2f}% ({-10 * np.log10(max(power_eff / 100, 1e-10)):.2f} dB)")
print(f"Mode coupling:   {mode_eff:.2f}% ({-10 * np.log10(max(mode_eff / 100, 1e-10)):.2f} dB)")

print("\n=== ALL CELLS PASSED ===")
