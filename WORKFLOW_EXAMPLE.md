# Hyperwave SDK Workflow Example

This example shows how to use the Hyperwave SDK functions for photonics simulations.

## Complete Workflow

```python
import hyperwave_community as hwc

# 1. Configure API
hwc.configure_api(api_key='your-api-key')

# 2. Build structure recipe from GDSFactory component
recipe_result = hwc.build_recipe(
    component_name="mmi2x2",
    resolution_nm=20,
    n_core=3.48,
    n_clad=1.4457,
    wg_height_um=0.22,
    total_height_um=4.0,
    extension_length=2.0,
    padding=(100, 100, 0, 0),
    density_radius=3,
    vertical_radius=2.0,
)
print(f"Dimensions: {recipe_result['dimensions']}")

# 3. Build monitors at port locations
monitor_result = hwc.build_monitors(
    port_info=recipe_result['port_info'],
    dimensions=recipe_result['dimensions'],
    source_port="o1",
    structure_recipe=recipe_result['recipe'],
    show_structure=True,  # Visualize structure with monitors
)
print(f"Source port: {monitor_result['source_port_name']}")

# 4. Compute frequency band
freq_result = hwc.compute_freq_band(
    wl_min_um=1.55,
    wl_max_um=1.55,
    n_freqs=1,
    resolution_um=recipe_result['resolution_um'],
)
print(f"Frequency band: {freq_result['freq_band']}")

# 5. Solve waveguide mode at source port
source_result = hwc.solve_mode_source(
    density_core=recipe_result['density_core'],
    density_clad=recipe_result['density_clad'],
    source_x_position=monitor_result['source_position'],
    mode_bounds=monitor_result['mode_bounds'],
    layer_config=recipe_result['layer_config'],
    eps_values=recipe_result['eps_values'],
    freq_band=freq_result['freq_band'],
    mode_num=0,
    show_mode=True,  # Visualize mode profile
)
print(f"Source field shape: {source_result['source_field'].shape}")

# 6. Run GPU simulation (CONSUMES CREDITS)
results = hwc.run_simulation(
    device_type="mmi2x2",
    recipe_result=recipe_result,
    monitor_result=monitor_result,
    freq_result=freq_result,
    source_result=source_result,
    num_steps=20000,
    gpu_type="B200",
    convergence="default",
)
print(f"Simulation time: {results['sim_time']:.1f}s")
print(f"Converged: {results.get('converged', False)}")

# 7. Analyze transmission (runs locally, free)
transmission = hwc.analyze_transmission(
    results,
    input_monitor="Input_o1",
    output_monitors=["Output_o3", "Output_o4"],
)
print(f"Total transmission: {transmission['total_transmission']:.4f}")
print(f"Excess loss: {transmission['excess_loss_dB']:.2f} dB")

# 8. Visualize field intensity (runs locally, free)
import matplotlib.pyplot as plt

field_data = hwc.get_field_intensity_2d(
    results,
    monitor_name='xy_mid',
    dimensions=recipe_result['dimensions'],
    resolution_um=recipe_result['resolution_um'],
    freq_band=freq_result['freq_band'],
)

plt.figure(figsize=(12, 5))
plt.imshow(field_data['intensity'], origin='upper', extent=field_data['extent'], cmap='jet')
plt.colorbar(label='|E|²')
plt.title(f"|E|² at λ = {field_data['wavelength_nm']:.1f} nm")
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.show()
```

## Cost Information

| Step | Function | Cost |
|------|----------|------|
| 1 | `configure_api()` | Free |
| 2 | `build_recipe()` | Free |
| 3 | `build_monitors()` | Free |
| 4 | `compute_freq_band()` | Free |
| 5 | `solve_mode_source()` | Free |
| 6 | `run_simulation()` | **Credits** |
| 7 | `analyze_transmission()` | Free |
| 8 | `get_field_intensity_2d()` | Free |

Only Step 6 (`run_simulation`) consumes GPU credits. All other steps run on CPU and are free.

## Convergence Options

Simple presets for early stopping:
- `"quick"` - Fast, 2 stability checks at 2000 step intervals
- `"default"` - Balanced, 3 stability checks at 1000 step intervals
- `"thorough"` - Conservative, 5 stability checks, min 5000 steps
- `"full"` - No early stopping, run all steps

All presets use 1% relative threshold for convergence detection.
