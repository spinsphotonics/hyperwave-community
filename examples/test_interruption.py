"""Test WebSocket interruption: start 5-step optimization, cancel after step 1."""
import os
import sys
import signal
import functools
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import matplotlib
matplotlib.use('Agg')

print = functools.partial(__builtins__.__dict__["print"], flush=True)

import hyperwave_community as hwc
import numpy as np
import jax.numpy as jnp
import time

_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"=== test_interruption.py  run={_run_id} ===")

hwc.configure_api(api_key=os.environ['HYPERWAVE_API_KEY'])

# === Same physical setup as test_notebook.py, abbreviated ===
n_si, n_sio2, n_clad, n_air = 3.48, 1.44, 1.44, 1.0
wavelength_um = 1.55
dx = 0.070
pixel_size = dx / 2
domain = 40.0
wg_width, wg_length = 0.5, 5.0
beam_waist, fiber_angle = 5.2, 14.5

Lx = Ly = int(domain / dx)
theta_Lx = theta_Ly = 2 * Lx
eps_si, eps_sio2, eps_clad, eps_air = n_si**2, n_sio2**2, n_clad**2, n_air**2

h_p = int(round(3.0 / dx))
h0 = int(round(1.0 / dx))
h1 = int(round(0.78 / dx))
h2 = int(round(0.110 / dx))
h3 = int(round(0.110 / dx))
h4 = int(round(2.0 / dx))
h5 = int(round(0.8 / dx))
Lz = h_p + h0 + h1 + h2 + h3 + h4 + h5 + h_p
z_etch = h_p + h0 + h1
z_slab = z_etch + h2
z_box = z_slab + h3
z_dev = z_etch + int(h2 // 2)

wl_px = wavelength_um / dx
freq = 2 * np.pi / wl_px
freq_band = (freq, freq, 1)

# Theta
wg_len_theta = int(round(wg_length / pixel_size))
wg_hw_theta = int(round(wg_width / 2 / pixel_size))
dr = {
    'x_start': wg_len_theta, 'x_end': theta_Lx - wg_len_theta,
    'y_start': wg_len_theta, 'y_end': theta_Ly - wg_len_theta,
}
theta_init = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
theta_init[dr['x_start']:dr['x_end'], dr['y_start']:dr['y_end']] = 0.5
theta_init[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

# Layers
slab = jnp.zeros(theta_init.shape)
design_layers = [
    hwc.Layer(density_pattern=slab, permittivity_values=eps_air, layer_thickness=h_p),
    hwc.Layer(density_pattern=slab, permittivity_values=eps_air, layer_thickness=h0),
    hwc.Layer(density_pattern=slab, permittivity_values=eps_clad, layer_thickness=h1),
    hwc.Layer(density_pattern=jnp.array(theta_init), permittivity_values=(eps_clad, eps_si), layer_thickness=h2),
    hwc.Layer(density_pattern=slab, permittivity_values=eps_si, layer_thickness=h3),
    hwc.Layer(density_pattern=slab, permittivity_values=eps_sio2, layer_thickness=h4),
    hwc.Layer(density_pattern=slab, permittivity_values=eps_si, layer_thickness=h5),
    hwc.Layer(density_pattern=slab, permittivity_values=eps_si, layer_thickness=h_p),
]

structure_spec = {
    'layers_info': [{
        'permittivity_values': [float(v) for v in l.permittivity_values] if isinstance(l.permittivity_values, tuple) else float(l.permittivity_values),
        'layer_thickness': float(l.layer_thickness),
        'density_radius': 0, 'density_alpha': 0,
    } for l in design_layers],
    'construction_params': {'vertical_radius': 0},
}

abs_params = hwc.get_optimized_absorber_params(
    resolution_nm=dx * 1000, wavelength_um=wavelength_um,
    structure_dimensions=(Lx, Ly, Lz),
)
abs_widths = abs_params['absorption_widths']
abs_coeff = abs_params['absorber_coeff']

# Source
source_z = int(round((3.0 + 1.0 - 0.05) / dx))
grating_x = int(round((dr['x_start'] + dr['x_end']) / 2 * pixel_size / dx))
grating_y = Ly // 2
waist_px = beam_waist / dx

print("Generating source on cloud GPU...")
t0 = time.time()
source_field, input_power = hwc.generate_gaussian_source(
    sim_shape=(Lx, Ly, Lz), frequencies=np.array([freq]),
    source_pos=(grating_x, grating_y, source_z), waist_radius=waist_px,
    theta=-fiber_angle, phi=0.0, polarization='y', max_steps=5000,
    absorption_widths=(30, 30, 20), absorption_coeff=1e-4, gpu_type="B200",
)
print(f"Source generated in {time.time() - t0:.1f}s")
source_offset = (0, 0, source_z)
input_power = float(np.mean(input_power))

# Mode (simplified - use local solver)
from hyperwave_community.mode_solver import mode as hwc_mode
small_x_theta = 40
theta_mode = np.zeros((small_x_theta, theta_Ly), dtype=np.float32)
theta_mode[:, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0
theta_mode_bot = np.where(theta_mode > 0, 1.0, 0.0).astype(np.float32)
d_mode_slab = jnp.zeros((small_x_theta, theta_Ly))
wg_structure = hwc.create_structure(layers=[
    hwc.Layer(density_pattern=d_mode_slab, permittivity_values=eps_air, layer_thickness=h_p),
    hwc.Layer(density_pattern=d_mode_slab, permittivity_values=eps_air, layer_thickness=h0),
    hwc.Layer(density_pattern=d_mode_slab, permittivity_values=eps_clad, layer_thickness=h1),
    hwc.Layer(density_pattern=jnp.array(theta_mode), permittivity_values=(eps_clad, eps_si), layer_thickness=h2),
    hwc.Layer(density_pattern=jnp.array(theta_mode_bot), permittivity_values=(eps_clad, eps_si), layer_thickness=h3),
    hwc.Layer(density_pattern=d_mode_slab, permittivity_values=eps_sio2, layer_thickness=h4),
    hwc.Layer(density_pattern=d_mode_slab, permittivity_values=eps_si, layer_thickness=h5),
    hwc.Layer(density_pattern=d_mode_slab, permittivity_values=eps_si, layer_thickness=h_p),
], vertical_radius=0)
eps_wg = np.array(wg_structure.permittivity)
Lz_wg = eps_wg.shape[3]
Ly_perm = eps_wg.shape[2]
eps_yz = eps_wg[0, eps_wg.shape[1] // 2, :, :]
crop_y = min(50, Ly_perm // 4)
crop_z = min(30, Lz_wg // 4)
y_c = Ly_perm // 2
y0, y1 = y_c - crop_y, y_c + crop_y
z0, z1 = max(0, z_etch - crop_z), min(Lz_wg, z_box + crop_z)
eps_crop = eps_yz[y0:y1, z0:z1]
eps_4d = jnp.stack([jnp.array(eps_crop)] * 3, axis=0)[:, jnp.newaxis, :, :]
mode_E, beta_arr, _ = hwc_mode(freq_band=freq_band, permittivity=eps_4d, axis=0, mode_num=0)
n_eff = float(beta_arr[0]) / (2 * np.pi / wl_px)
print(f"n_eff = {n_eff:.4f}")

print("Computing mode on cloud GPU...")
t0 = time.time()
mode_EH = hwc.mode_convert(
    mode_E_field=mode_E[0:1, 0:3, :, :, :], freq_band=freq_band,
    permittivity_slice=np.array(eps_crop), propagation_axis='x',
    propagation_length=500, gpu_type="B200",
)
print(f"Mode convert: {time.time() - t0:.1f}s")
mode_EH = np.array(mode_EH, copy=True)
mode_EH[:, 3:6, ...] *= -1  # Negate H for backward propagation
mode_e = np.array(mode_EH[0, 0:3, 0, :, :])
mode_h = np.array(mode_EH[0, 3:6, 0, :, :])
cross = np.cross(mode_e, np.conj(mode_h), axis=0)
P_mode_cross = float(np.abs(np.real(np.sum(cross[0, :, :]))))
mode_field = np.zeros((1, 6, 1, Ly_perm, Lz_wg), dtype=np.complex64)
mode_field[:, :, :, y0:y1, z0:z1] = np.array(mode_EH)
print(f"P_mode_cross = {P_mode_cross:.6f}")

# Optimization params
output_x = abs_widths[0] + 10
loss_monitor_shape = (1, Ly, Lz)
loss_monitor_offset = (output_x, 0, 0)
design_monitor_shape = (Lx, Ly, int(round(h2)))
design_monitor_offset = (0, 0, z_etch)
waveguide_mask = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
waveguide_mask[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

NUM_STEPS = 5  # Request 5 steps, but we'll stop after 1
print(f"\n=== INTERRUPTION TEST: requesting {NUM_STEPS} steps, will cancel after step 1 ===\n")

results = []
t_opt_start = time.time()

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
    learning_rate=0.01,
    grad_clip_norm=1.0,
    cosine_decay_alpha=0.1,
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
    gpu_type="B200",
):
    results.append(step_result)
    eff = abs(step_result['loss']) * 100
    wall = time.time() - t_opt_start
    print(f"Step {step_result['step']:3d}/{NUM_STEPS}:  eff = {eff:.2f}%  "
          f"|grad|_max = {step_result['grad_max']:.3e}  (wall={wall:.1f}s)",
          flush=True)

    if len(results) >= 1:
        print(f"\n=== INTERRUPTING after step {len(results)} ===")
        print(f"Breaking out of generator (this should close WebSocket and cancel GPU task)")
        break

elapsed = time.time() - t_opt_start
print(f"\nInterruption test complete: {len(results)} steps received in {elapsed:.0f}s")

if len(results) == 1:
    print("SUCCESS: Received exactly 1 step before cancellation")
    print(f"  Step 1 eff: {abs(results[0]['loss']) * 100:.4f}%")
    print(f"  Theta shape: {results[0]['theta'].shape}")
else:
    print(f"UNEXPECTED: Received {len(results)} steps (expected 1)")

# Wait for GPU to shut down, then verify via account info
time.sleep(10)
print("\nVerifying billing state...")
try:
    info = hwc.configure_api(api_key=os.environ['HYPERWAVE_API_KEY'], validate=True)
    if info:
        print(f"  Credits remaining: {info.get('credits', '?')}")
except Exception as e:
    print(f"  Could not verify: {e}")

print("\n=== INTERRUPTION TEST DONE ===")
