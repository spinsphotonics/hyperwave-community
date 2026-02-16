"""
Test script for inverse_design_workflow notebook against localhost:8000.
Converted from inverse_design_workflow.ipynb with Colab-specific code removed.
"""

import os
import sys
import math
import time
import pickle
import traceback

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import hyperwave_community as hwc

# --- API Configuration (replaces Colab userdata) ---
hwc.configure_api(
    api_key=os.environ.get('HYPERWAVE_API_KEY'),
    api_url='http://localhost:8000',
)

# ============================================================
print("=== Step 1: Physical Parameters ===")
# ============================================================
try:
    # Materials
    n_si = 3.48
    n_sio2 = 1.44
    n_clad = 1.44
    n_air = 1.0

    # Wavelength
    wavelength_um = 1.55

    # Layer thicknesses (um)
    h_dev = 0.220        # total silicon device layer
    etch_depth = 0.110   # partial etch depth
    h_box = 2.0          # buried oxide
    h_clad = 0.78        # SiO2 cladding
    h_sub = 0.8          # silicon substrate
    h_air = 1.0          # air above cladding
    pad = 3.0            # absorber padding (top and bottom)

    # Grid resolution
    dx = 0.035           # 35nm structure grid
    pixel_size = dx / 2  # 17.5nm theta grid (2x for subpixel averaging)
    domain = 20.0        # um total domain

    # Waveguide
    wg_width = 0.5       # um
    wg_length = 2.5      # um

    # Fiber
    beam_waist = 5.2     # um (SMF-28 mode field radius at 1550nm)
    fiber_angle = 14.5   # degrees from vertical

    # Structure grid dimensions
    Lx = int(domain / dx)
    Ly = Lx

    # Theta grid dimensions (2x structure)
    theta_Lx = 2 * Lx
    theta_Ly = 2 * Ly

    # Layer thicknesses in pixels (float for mode solve structure)
    h_p_f = pad / dx
    h0_f = h_air / dx
    h1_f = h_clad / dx
    h2_f = etch_depth / dx
    h3_f = (h_dev - etch_depth) / dx
    h4_f = h_box / dx
    h5_f = h_sub / dx

    # Integer pixel thicknesses for simulation structure
    h_p = int(round(h_p_f))
    h0 = int(round(h0_f))
    h1 = int(round(h1_f))
    h2 = int(round(h2_f))
    h3 = int(round(h3_f))
    h4 = int(round(h4_f))
    h5 = int(round(h5_f))
    Lz = h_p + h0 + h1 + h2 + h3 + h4 + h5 + h_p

    # Key Z positions
    z_etch = h_p + h0 + h1
    z_slab = z_etch + h2
    z_box = z_slab + h3

    # Frequency
    wl_px = wavelength_um / dx
    freq = 2 * np.pi / wl_px
    freq_band = (freq, freq, 1)

    # Permittivity
    eps_si = n_si**2
    eps_sio2 = n_sio2**2
    eps_clad = n_clad**2
    eps_air = n_air**2

    print(f"Structure grid: {Lx} x {Ly} x {Lz} ({dx * 1000:.0f} nm)")
    print(f"Theta grid: {theta_Lx} x {theta_Ly} ({pixel_size * 1000:.1f} nm)")
    print(f"Layers (px): pad={h_p} air={h0} clad={h1} etch={h2} slab={h3} BOX={h4} sub={h5} pad={h_p}")
except Exception as e:
    print(f"ERROR in Step 1: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 2: Design Variables (Theta) ===")
# ============================================================
try:
    # Waveguide in structure pixels (35nm grid)
    wg_len = int(round(wg_length / dx))
    wg_hw = int(round(wg_width / 2 / dx))

    # Waveguide in theta pixels (17.5nm grid, 2x structure)
    wg_len_theta = int(round(wg_length / pixel_size))
    wg_hw_theta = int(round(wg_width / 2 / pixel_size))

    # Design region with uniform 5um margin on all sides
    design_region = {
        'x_start': wg_len_theta,
        'x_end': theta_Lx - wg_len_theta,
        'y_start': wg_len_theta,
        'y_end': theta_Ly - wg_len_theta,
    }

    # Build theta: zeros -> fill design region with 0.5 -> stamp waveguide as 1.0
    dr = design_region
    theta_init = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
    theta_init[dr['x_start']:dr['x_end'], dr['y_start']:dr['y_end']] = 0.5
    theta_init[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

    # Plot
    extent = [0, theta_Lx * pixel_size, 0, theta_Ly * pixel_size]
    plt.figure(figsize=(6, 6))
    plt.imshow(theta_init.T, origin='lower', cmap='gray', vmin=0, vmax=1, extent=extent)
    plt.colorbar(label='theta')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.title('Initial Theta')
    plt.tight_layout()
    plt.savefig('/tmp/plot_step2_initial_theta.png')
    plt.close()

    print(f"Theta shape: {theta_init.shape}")
    print(f"Design region: {(dr['x_end'] - dr['x_start']) * pixel_size:.1f} x "
          f"{(dr['y_end'] - dr['y_start']) * pixel_size:.1f} um")
except Exception as e:
    print(f"ERROR in Step 2: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 3: Layer Stack ===")
# ============================================================
try:
    # Slab pattern (uniform zero for non-design layers)
    slab = jnp.zeros(theta_init.shape)

    # 8-layer SOI stack
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
except Exception as e:
    print(f"ERROR in Step 3: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 4: Absorbing Boundaries ===")
# ============================================================
try:
    abs_params = hwc.get_optimized_absorber_params(
        resolution_nm=dx * 1000,
        wavelength_um=wavelength_um,
        structure_dimensions=(Lx, Ly, Lz),
    )
    abs_widths = abs_params['absorption_widths']
    abs_coeff = abs_params['absorber_coeff']

    print(f"Absorber widths (x,y,z): {abs_widths}")
    print(f"Absorber coefficient: {abs_coeff:.6f}")
except Exception as e:
    print(f"ERROR in Step 4: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 5: API Configuration + Account Info ===")
# ============================================================
try:
    info = hwc.get_account_info()
    print(f"Account info: {info}")
except Exception as e:
    print(f"ERROR in Step 5 (account info): {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 6: Source Generation ===")
# ============================================================
try:
    # Source position: in the air gap, 50nm above cladding surface
    source_above_surface_um = 0.05
    source_z = int(round((pad + h_air - source_above_surface_um) / dx))

    # Grating center in structure pixels
    grating_x = int(round((dr['x_start'] + dr['x_end']) / 2 * pixel_size / dx))
    grating_y = Ly // 2
    waist_px = beam_waist / dx

    # Estimate cost
    est = hwc.estimate_cost(
        structure_shape=(3, Lx, Ly, Lz),
        max_steps=5000,
        gpu_type="B200",
        simulation_type="fdtd_simulation",
    )
    if est:
        print(f"Source gen estimate: {est['estimated_seconds']:.0f}s, "
              f"{est['estimated_credits']:.4f} credits")

    # Generate Gaussian source on cloud GPU (wave equation error method)
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
        gpu_type="B200",
    )
    print(f"Source generated in {time.time() - t0:.1f}s")

    source_offset = (0, 0, source_z)
    input_power = float(np.mean(input_power))

    # Plot source
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, idx, name in [(axes[0], 1, '|Ey|'), (axes[1], 3, '|Hx|')]:
        ax.imshow(np.abs(np.array(source_field[0, idx, :, :, 0])).T,
                  origin='lower', cmap='hot', extent=[0, Lx * dx, 0, Ly * dx])
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_title(f'Source {name}')
    plt.tight_layout()
    plt.savefig('/tmp/plot_step6_source.png')
    plt.close()

    print(f"Source shape: {source_field.shape}")
    print(f"Source offset: {source_offset}")
    print(f"Input power: {input_power:.6f}")
except Exception as e:
    print(f"ERROR in Step 6: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 7: Waveguide Mode ===")
# ============================================================
try:
    # Build small waveguide structure for mode solving (narrow x, full y)
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
    print(f"WG structure: {eps_wg.shape}")

    # Crop YZ cross-section around waveguide core for eigenmode solve
    eps_yz = eps_wg[0, eps_wg.shape[1] // 2, :, :]
    crop_y = min(50, Ly_perm // 4)
    crop_z = min(30, Lz_wg // 4)
    y_c = Ly_perm // 2
    y0, y1 = y_c - crop_y, y_c + crop_y
    z0 = max(0, z_etch - crop_z)
    z1 = min(Lz_wg, z_box + crop_z)
    eps_crop = eps_yz[y0:y1, z0:z1]
    print(f"Cropped eps: {eps_crop.shape}")

    # Solve E-only eigenmode locally (no GPU needed)
    from hyperwave_community.mode_solver import mode as hwc_mode
    eps_4d = jnp.stack([jnp.array(eps_crop)] * 3, axis=0)[:, jnp.newaxis, :, :]
    mode_E, beta_arr, _ = hwc_mode(freq_band=freq_band, permittivity=eps_4d, axis=0, mode_num=0)
    n_eff = float(beta_arr[0]) / (2 * np.pi / wl_px)
    print(f"n_eff = {n_eff:.4f}")
    assert 2.0 < n_eff < 3.0, f"n_eff={n_eff:.4f} out of range"

    # Convert E-only -> full E+H via short FDTD propagation on cloud GPU
    mode_EH = hwc.mode_convert(
        mode_E_field=mode_E[0:1, 0:3, :, :, :],
        freq_band=freq_band,
        permittivity_slice=np.array(eps_crop),
        propagation_axis='x',
        propagation_length=500,
        gpu_type="B200",
    )

    # Negate H for backward (-x) propagation
    mode_EH = np.array(mode_EH, copy=True)
    mode_EH[:, 3:6, ...] *= -1

    # P_mode_cross: mode self-overlap integral for normalization
    mode_e = np.array(mode_EH[0, 0:3, 0, :, :])
    mode_h = np.array(mode_EH[0, 3:6, 0, :, :])
    cross = np.cross(mode_e, np.conj(mode_h), axis=0)
    P_mode_cross = float(np.abs(np.real(np.sum(cross[0, :, :]))))
    print(f"P_mode_cross = {P_mode_cross:.6f}")

    # Pad mode to full YZ domain
    mode_field = np.zeros((1, 6, 1, Ly_perm, Lz_wg), dtype=np.complex64)
    mode_field[:, :, :, y0:y1, z0:z1] = np.array(mode_EH)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ext_crop = [y0 * dx, y1 * dx, z0 * dx, z1 * dx]

    axes[0].imshow(eps_crop.T, origin='lower', cmap='viridis', extent=ext_crop)
    axes[0].set_title('Cropped permittivity')

    axes[1].imshow(np.abs(mode_e[1]).T, origin='lower', cmap='hot', extent=ext_crop)
    axes[1].set_title(f'Mode |Ey| (n_eff={n_eff:.4f})')

    E_mag = np.sqrt(np.sum(np.abs(mode_e)**2, axis=0))
    axes[2].imshow(E_mag.T, origin='lower', cmap='hot', extent=ext_crop)
    axes[2].set_title(f'Mode |E| (P_cross={P_mode_cross:.4f})')

    for ax in axes:
        ax.set_xlabel('y (um)')
        ax.set_ylabel('z (um)')
    plt.tight_layout()
    plt.savefig('/tmp/plot_step7_waveguide_mode.png')
    plt.close()
except Exception as e:
    print(f"ERROR in Step 7: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 8: Monitors ===")
# ============================================================
try:
    monitors = hwc.MonitorSet()
    output_x = abs_widths[0] + 10

    monitors.add(hwc.Monitor(shape=(Lx, Ly, 1), offset=(0, 0, z_dev)), name='Output_xy_device')
    monitors.add(hwc.Monitor(shape=(Lx, 1, Lz), offset=(0, Ly // 2, 0)), name='Output_xz_center')
    monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(Lx // 2, 0, 0)), name='Output_yz_center')
    monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(output_x, 0, 0)), name='Output_wg_output')

    monitors.list_monitors()
except Exception as e:
    print(f"ERROR in Step 8: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 9: Forward Simulation (Initial Design) ===")
# ============================================================
try:
    est = hwc.estimate_cost(
        structure_shape=(3, Lx, Ly, Lz),
        max_steps=10000,
        gpu_type="B200",
        simulation_type="fdtd_simulation",
    )
    if est:
        print(f"Forward sim estimate: {est['estimated_seconds']:.0f}s, "
              f"{est['estimated_credits']:.4f} credits (${est['estimated_cost_usd']:.2f})")

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

    # Plot |E|^2 cross-sections
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    slices = [
        ('Output_xy_device', f'|E|^2 XY (z={z_dev * dx:.2f} um)', [0, Lx * dx, 0, Ly * dx]),
        ('Output_xz_center', '|E|^2 XZ (y=center)', [0, Lx * dx, 0, Lz * dx]),
        ('Output_yz_center', '|E|^2 YZ (x=center)', [0, Ly * dx, 0, Lz * dx]),
    ]

    for ax, (name, title, ext) in zip(axes, slices):
        field = np.array(fwd_results['monitor_data'][name])
        E2 = np.sum(np.abs(field[0, 0:3])**2, axis=0).squeeze()
        ax.imshow(E2.T, origin='lower', cmap='hot', extent=ext,
                  aspect='auto', vmax=np.percentile(E2, 95) * 4)
        ax.set_title(title)

    plt.suptitle('Forward simulation (initial design)')
    plt.tight_layout()
    plt.savefig('/tmp/plot_step9_forward_sim.png')
    plt.close()

    # Check power at waveguide output
    wg_field = np.array(fwd_results['monitor_data']['Output_wg_output'])
    S = hwc.S_from_slice(jnp.mean(jnp.array(wg_field), axis=2))
    power = float(jnp.abs(jnp.sum(S[0, 0, :, :])))
    print(f"Waveguide output power: {power:.6f}")
    print(f"Coupling (approx): {power / input_power * 100:.1f}%")
except Exception as e:
    print(f"ERROR in Step 9: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 10: Optimization Setup ===")
# ============================================================
try:
    # Structure spec
    structure_spec = {
        'layers_info': [{
            'permittivity_values': [float(v) for v in l.permittivity_values] if isinstance(l.permittivity_values, tuple) else float(l.permittivity_values),
            'layer_thickness': float(l.layer_thickness),
            'density_radius': 0,
            'density_alpha': 0,
        } for l in design_layers],
        'construction_params': {'vertical_radius': 0},
    }

    # Loss monitor
    loss_monitor_shape = (1, Ly, Lz)
    loss_monitor_offset = (output_x, 0, 0)

    # Design monitor
    dr_x0 = dr['x_start'] // 2
    dr_x1 = dr['x_end'] // 2
    dr_y0 = dr['y_start'] // 2
    dr_y1 = dr['y_end'] // 2
    design_monitor_shape = (dr_x1 - dr_x0, dr_y1 - dr_y0, int(round(h2)))
    design_monitor_offset = (dr_x0, dr_y0, z_etch)

    # Waveguide mask
    waveguide_mask = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
    waveguide_mask[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

    # Optimizer settings
    NUM_STEPS = 5
    LR = 0.1
    GRAD_CLIP = 1.0

    est = hwc.estimate_cost(
        structure_shape=(3, Lx, Ly, Lz),
        max_steps=10000 * NUM_STEPS * 2,
        gpu_type="B200",
        simulation_type="fdtd_simulation",
    )
    if est:
        print(f"Optimization estimate ({NUM_STEPS} steps): {est['estimated_seconds']:.0f}s, "
              f"{est['estimated_credits']:.4f} credits (${est['estimated_cost_usd']:.2f})")

    print(f"Loss monitor at x={loss_monitor_offset[0]} ({loss_monitor_offset[0] * dx:.1f} um)")
    print(f"Design monitor: {design_monitor_shape} at offset {design_monitor_offset}")
    print(f"  Design region: {(dr_x1 - dr_x0) * dx:.1f} x {(dr_y1 - dr_y0) * dx:.1f} um")
    print(f"Optimizer: Adam, LR={LR}, grad_clip={GRAD_CLIP}, {NUM_STEPS} steps")
except Exception as e:
    print(f"ERROR in Step 10: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 11: Optimization Loop ===")
# ============================================================
results = []
theta_final = theta_init
try:
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
        learning_rate=LR,
        grad_clip_norm=GRAD_CLIP,
        absorption_widths=abs_widths,
        absorption_coeff=abs_coeff,
        gpu_type="B200",
    ):
        results.append(step_result)
        eff = abs(step_result['loss']) * 100
        print(f"Step {step_result['step']:3d}/{NUM_STEPS}:  eff = {eff:.2f}%  "
              f"|grad|_max = {step_result['grad_max']:.3e}  ({step_result['step_time']:.1f}s)",
              flush=True)

    # Summary
    if results:
        efficiencies = [abs(r['loss']) * 100 for r in results]
        best_idx = int(np.argmax(efficiencies))
        best_eff = efficiencies[best_idx]
        loss_dB = -10 * np.log10(max(best_eff / 100, 1e-10))
        print(f"\nBest: {best_eff:.2f}% ({loss_dB:.2f} dB) at step {best_idx + 1}")
        theta_final = results[-1]['theta']
    else:
        print("No optimization steps completed.")
except Exception as e:
    print(f"ERROR in Step 11: {e}")
    traceback.print_exc()
    if results:
        theta_final = results[-1]['theta']

# ============================================================
print("\n=== Step 12: Optimization Results ===")
# ============================================================
try:
    if results:
        efficiencies = [abs(r['loss']) * 100 for r in results]
        best_idx = int(np.argmax(efficiencies))
        best_theta = results[best_idx]['theta']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Efficiency curve
        axes[0].plot(range(1, len(efficiencies) + 1), efficiencies, 'b-o', markersize=3)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Efficiency (%)')
        axes[0].set_title('Mode Coupling Efficiency')
        axes[0].grid(True, alpha=0.3)

        # Initial vs best theta
        extent = [0, theta_Lx * pixel_size, 0, theta_Ly * pixel_size]
        axes[1].imshow(theta_init.T, origin='lower', cmap='gray', vmin=0, vmax=1, extent=extent)
        axes[1].set_title('Initial')

        axes[2].imshow(best_theta.T, origin='lower', cmap='gray', vmin=0, vmax=1, extent=extent)
        axes[2].set_title(f'Best (step {best_idx + 1})')

        for ax in axes[1:]:
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')

        plt.tight_layout()
        plt.savefig('/tmp/plot_step12_optimization_results.png')
        plt.close()
    else:
        print("No results to plot.")
        best_theta = theta_init
except Exception as e:
    print(f"ERROR in Step 12: {e}")
    traceback.print_exc()

# ============================================================
print("\n=== Step 13: Verification (Forward Sim with Optimized Design) ===")
# ============================================================
try:
    # Build recipe with optimized theta
    opt_layers = list(design_layers)
    opt_layers[3] = hwc.Layer(
        density_pattern=jnp.array(best_theta),
        permittivity_values=(eps_clad, eps_si),
        layer_thickness=h2,
    )
    recipe_best = build_recipe(opt_layers)

    est = hwc.estimate_cost(structure_shape=(3, Lx, Ly, Lz), max_steps=10000, gpu_type="B200")
    if est:
        print(f"Verification sim estimate: {est['estimated_seconds']:.0f}s, "
              f"{est['estimated_credits']:.4f} credits")

    t0 = time.time()
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
    print(f"Verification sim complete: {opt_results['sim_time']:.1f}s GPU, {time.time() - t0:.0f}s total")

    # Plot |E|^2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    slices = [
        ('Output_xy_device', f'|E|^2 XY (z={z_dev * dx:.2f} um)', [0, Lx * dx, 0, Ly * dx]),
        ('Output_xz_center', '|E|^2 XZ (y=center)', [0, Lx * dx, 0, Lz * dx]),
        ('Output_yz_center', '|E|^2 YZ (x=center)', [0, Ly * dx, 0, Lz * dx]),
    ]

    for ax, (name, title, ext) in zip(axes, slices):
        field = np.array(opt_results['monitor_data'][name])
        E2 = np.sum(np.abs(field[0, 0:3])**2, axis=0).squeeze()
        ax.imshow(E2.T, origin='lower', cmap='hot', extent=ext,
                  aspect='auto', vmax=np.percentile(E2, 95) * 4)
        ax.set_title(title)

    plt.suptitle('Optimized design fields')
    plt.tight_layout()
    plt.savefig('/tmp/plot_step13_verification.png')
    plt.close()

    # Power coupling via Poynting vector
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

    print(f"Power coupling:  {power_eff:.2f}% ({-10*np.log10(max(power_eff/100, 1e-10)):.2f} dB)")
    print(f"Mode coupling:   {mode_eff:.2f}% ({-10*np.log10(max(mode_eff/100, 1e-10)):.2f} dB)")
except Exception as e:
    print(f"ERROR in Step 13: {e}")
    traceback.print_exc()

print("\n=== DONE ===")
