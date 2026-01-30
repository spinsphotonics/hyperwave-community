# Granular Workflow Example

This example shows how to use the new fine-grained granular SDK functions for maximum control over the simulation pipeline.

## Complete Workflow

```python
import hyperwave_community as hwc
import numpy as np

# 1. Configure API
hwc.configure_api(api_key='your-api-key')

# 2. Load component metadata
component_data = hwc.load_component(
    name="mmi2x2",
    extension_length=2.0,
    component_kwargs={}
)
print(f"Ports: {list(component_data['port_info'].keys())}")

# 3. Create structure recipe
recipe_data = hwc.create_structure_recipe(
    component_data=component_data,
    resolution_nm=20,
    n_core=3.48,
    n_clad=1.4457,
    wg_height_um=0.22,
    clad_top_um=1.89,
    clad_bot_um=2.0,
    padding=(100, 100, 0, 0),
    density_radius=3,
    vertical_radius=2.0,
)
print(f"Dimensions: {recipe_data['dimensions']}")

# 4. Create monitors
monitors_data = hwc.create_monitors(
    structure_recipe_data=recipe_data,
    source_port="o2",
    monitor_x_um=0.1,
    monitor_y_um=1.5,
    monitor_z_um=1.5,
    source_offset_cells=5,
)
print(f"Source port: {monitors_data['source_port_name']}")

# 5. Solve mode at source port
source_data = hwc.solve_mode(
    structure_recipe=recipe_data['structure_recipe'],
    mode_solve_params=monitors_data['mode_solve_params'],
    freq_band=recipe_data['freq_band'],
    mode_num=0,
)
print(f"Mode n_eff: {source_data['mode_info']['n_eff']}")

# 6. Run GPU simulation (this consumes credits)
results = hwc.run_gpu_simulation(
    structure_recipe=recipe_data['structure_recipe'],
    source_data=source_data,
    monitors=monitors_data['monitors'],
    gpu_type="H100",
    max_steps=20000,
    check_every_n=1000,
    source_ramp_periods=10.0,
)
print(f"Simulation time: {results['sim_time']:.1f}s")
print(f"Converged: {results['converged']}")

# 7. Analyze transmission
trans_data = hwc.analyze_transmission(results)
print(f"Total transmission: {trans_data['total_transmission']:.2f}")
print(f"Excess loss: {trans_data['excess_loss_dB']:.2f} dB")

# 8. Extract field slice for visualization
field_data = hwc.get_field_slice(
    simulation_results=results,
    monitor_name="xy_mid",
    freq_idx=0,
)

# 9. Visualize field
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(field_data['intensity_2d'], cmap='hot')
plt.colorbar(label='Field Intensity')
plt.title(f'Field Slice: {field_data["monitor_name"]}')
plt.xlabel('X (cells)')
plt.ylabel('Y (cells)')
plt.show()
```

## API Endpoints Called

These SDK functions call the following hyperwave-cloud API endpoints:

1. `POST /granular/component/load` - Load component metadata
2. `POST /granular/structure/create` - Create structure recipe
3. `POST /granular/monitors/create` - Create monitors
4. `POST /granular/mode/solve` - Solve waveguide mode
5. `POST /granular/simulation/run` - Run GPU simulation (CONSUMES CREDITS)
6. `POST /granular/analysis/transmission` - Analyze transmission
7. `POST /granular/analysis/field_slice` - Extract field slice

## Cost Information

Only step 6 (`run_gpu_simulation`) consumes GPU credits. All other steps run on CPU and are free.

## Comparison with Two-Stage Workflow

### Two-Stage (Recommended for most users)
```python
# Stage 1: Prepare (CPU, free)
setup = hwc.prepare_simulation(
    device_type="mmi2x2",
    pdk_config=pdk_config,
    source_port="o1",
    wavelength_um=1.55,
)

# Stage 2: Run (GPU, consumes credits)
results = hwc.run_simulation(
    device_type="mmi2x2",
    setup_data=setup['setup_data'],
    num_steps=20000,
)
```

### Granular (Fine-grained control)
```python
# 7 separate steps with full control at each stage
component_data = hwc.load_component("mmi2x2")
recipe_data = hwc.create_structure_recipe(component_data, ...)
monitors_data = hwc.create_monitors(recipe_data, ...)
source_data = hwc.solve_mode(recipe_data['structure_recipe'], ...)
results = hwc.run_gpu_simulation(...)  # Only this step consumes credits
trans_data = hwc.analyze_transmission(results)
field_data = hwc.get_field_slice(results, ...)
```

## When to Use Granular Workflow

Use the granular workflow when you need:
- Custom structure modifications between steps
- Access to intermediate data (mode fields, monitor configs, etc.)
- Fine-tuned control over each parameter
- Integration with custom analysis pipelines
- Debugging or research into specific simulation stages

For most production use cases, the two-stage workflow is simpler and sufficient.
