Local Workflow
==============

In the local workflow, all CPU steps run on your machine (or in Google Colab). You build the
simulation structure step by step, giving you full access to intermediate arrays -- theta patterns,
density fields, permittivity distributions -- at every stage. This is ideal for custom structures,
inverse design, and debugging, because you can inspect and modify any intermediate result before
proceeding to the next step. Only the GPU FDTD simulation step (Step 7) requires an API call and
costs credits (1 credit = $25 = 1 hour of B200 compute); everything else runs locally and is free.

**Open in Google Colab**: `local_workflow.ipynb <https://colab.research.google.com/github/spinsphotonics/hyperwave-community/blob/main/examples/local_workflow.ipynb>`_

`Download the notebook <https://github.com/spinsphotonics/hyperwave-community/blob/main/examples/local_workflow.ipynb>`_

.. contents:: Steps
   :local:
   :depth: 1

Imports
-------

The local workflow uses GDSFactory for photonic component layout, JAX for array operations (which
are GPU-compatible and differentiable), and matplotlib for visualization. NumPy is used alongside
JAX for operations that do not require automatic differentiation. The generic PDK activation call
at the end is required before loading any GDSFactory components -- it registers the standard layer
stack and cross-sections that ``component_to_theta()`` relies on.

.. code-block:: python

   import hyperwave_community as hwc
   import gdsfactory as gf
   import matplotlib.pyplot as plt
   import numpy as np
   import jax.numpy as jnp

   # Activate generic PDK (required for gdsfactory)
   PDK = gf.gpdk.get_generic_pdk()
   PDK.activate()

Step 1: Load GDSFactory Component
---------------------------------

The first step converts a photonic component into a 2D binary pattern called "theta." Theta is a
2D array where 1 represents the core material (silicon) and 0 represents the cladding (SiO2). The
function ``component_to_theta()`` rasterizes the GDS polygons onto a regular grid at the specified
resolution, producing a pixel-level representation of the device geometry.

There are several key parameters to understand:

- **RESOLUTION_UM**: The grid cell size in microns. A value of 0.02 (20 nm) is standard for silicon
  photonics at 1550 nm wavelength. Finer resolution increases accuracy but also increases the
  simulation grid size and therefore the compute cost.
- **EXTENSION_LENGTH**: Extends waveguide ports outward by this distance (in microns). This ensures
  the mode source and monitors are placed in straight waveguide regions, well away from the device's
  active area where the geometry is changing. Without sufficient extension, the source mode can
  couple poorly into the device.

.. code-block:: python

   # Component settings
   COMPONENT_NAME = "mmi2x2_with_sbend"
   RESOLUTION_UM = 0.02  # 20nm resolution
   EXTENSION_LENGTH = 2.0  # Extend ports by 2um

   # Load component from GDSFactory
   component = gf.components.mmi2x2_with_sbend()

   # Extend ports to ensure proper mode coupling
   component = gf.c.extend_ports(component, length=EXTENSION_LENGTH)

   # Convert to theta pattern
   theta, device_info = hwc.component_to_theta(
       component=component,
       resolution=RESOLUTION_UM,
   )

   print(f"Theta shape: {theta.shape}")
   print(f"Device size: {device_info['physical_size_um']} um")

   # Visualize
   plt.imshow(theta.T, cmap='gray', aspect='equal')
   plt.title('Theta Pattern')
   plt.show()

Custom Theta Patterns
^^^^^^^^^^^^^^^^^^^^^

You are not limited to GDSFactory components. Any 2D binary array can serve as a theta pattern.
This is particularly useful for inverse design workflows, where theta is treated as a continuous
design variable and optimized with gradients flowing through the entire simulation pipeline --
from theta, through density filtering, structure construction, FDTD simulation, and finally to
the objective function.

.. code-block:: python

   # Create a custom waveguide pattern
   theta = np.zeros((500, 1000))
   center_y = theta.shape[0] // 2
   wg_width = 25  # pixels

   # Straight waveguide
   theta[center_y - wg_width//2 : center_y + wg_width//2, :] = 1.0

Step 2: Apply Density Filtering
-------------------------------

Density filtering smooths the binary theta pattern using a conic filter, converting the sharp 0/1
boundaries into gradual transitions. This serves two purposes. First, it improves numerical
stability in the FDTD simulation by avoiding abrupt permittivity jumps at material interfaces.
Second, for inverse design, it enforces minimum feature sizes -- the filter radius sets the
smallest feature that can appear in the optimized structure.

Before filtering, the theta pattern is padded with extra cells using the ``pad_width`` parameter.
The padding of 100 cells on the left and right provides space for absorbing boundaries and field
monitors along the propagation axis. The top and bottom padding is set to 0 because the Y
boundaries are already far from the waveguide core and do not need additional space.

Two separate density patterns are created: one for the waveguide core and one for the cladding.
The ``radius`` parameter controls the smoothing strength. A smaller radius (3) is used for the
waveguide core to preserve fine geometric features, while a larger radius (5) is used for the
uniform cladding where sharp features are not needed.

.. code-block:: python

   # Material properties
   N_CORE = 3.48      # Silicon refractive index at 1550nm
   N_CLAD = 1.45      # SiO2 cladding refractive index at 1550nm

   # Padding for absorbers and monitors
   PADDING = (100, 100, 0, 0)  # (left, right, top, bottom) in pixels

   # Apply density filtering
   density_core = hwc.density(
       theta=theta,
       pad_width=PADDING,
       radius=3,  # Smoothing radius for waveguide
   )

   density_clad = hwc.density(
       theta=jnp.zeros_like(theta),
       pad_width=PADDING,
       radius=5,  # Wider smoothing for uniform cladding
   )

   print(f"Density shape (with padding): {density_core.shape}")

Step 3: Build 3D Layer Structure
--------------------------------

This step extrudes the 2D density patterns into a full 3D permittivity distribution by stacking
layers vertically. The structure consists of three layers: bottom cladding, waveguide core, and
top cladding. Each layer maps its density pattern to permittivity values. The waveguide layer
interpolates between cladding permittivity and core permittivity based on the density value at
each grid cell, while the cladding layers use a uniform permittivity. The ``vertical_radius``
parameter applies vertical blurring between layers to smooth the transitions in the z-direction,
avoiding abrupt permittivity changes between adjacent layers.

There are two key parameters controlling the vertical geometry:

- **WG_HEIGHT_UM**: The waveguide core thickness. A value of 0.22 um (220 nm) is the standard
  silicon-on-insulator (SOI) thickness used across most silicon photonics foundries.
- **TOTAL_HEIGHT_UM**: The total simulation domain height. This must be large enough for the
  optical mode fields to decay to near-zero at the top and bottom boundaries. A value of 4.0 um
  provides sufficient margin for typical single-mode waveguide structures.

.. code-block:: python

   # Layer dimensions
   WG_HEIGHT_UM = 0.22
   TOTAL_HEIGHT_UM = 4.0
   VERTICAL_RADIUS = 2.0

   # Calculate layer thicknesses in cells
   wg_thickness = int(np.round(WG_HEIGHT_UM / RESOLUTION_UM))
   clad_thickness = int(np.round((TOTAL_HEIGHT_UM - WG_HEIGHT_UM) / 2 / RESOLUTION_UM))

   # Define layers
   waveguide_layer = hwc.Layer(
       density_pattern=density_core,
       permittivity_values=(N_CLAD**2, N_CORE**2),
       layer_thickness=wg_thickness,
   )

   cladding_layer = hwc.Layer(
       density_pattern=density_clad,
       permittivity_values=N_CLAD**2,
       layer_thickness=clad_thickness,
   )

   # Create 3D structure
   structure = hwc.create_structure(
       layers=[cladding_layer, waveguide_layer, cladding_layer],
       vertical_radius=VERTICAL_RADIUS,
   )

   _, Lx, Ly, Lz = structure.permittivity.shape
   print(f"Structure dimensions: ({Lx}, {Ly}, {Lz})")

   # Visualize
   structure.view(axis="z", position=Lz // 2)

Step 4: Add Absorbing Boundaries
--------------------------------

FDTD simulations use a finite computational domain, so absorbing boundaries are needed at the
domain edges to prevent artificial reflections from corrupting the results. Without them, outgoing
light would bounce off the grid boundaries and interfere with the physical fields inside the
simulation region.

The function ``get_optimized_absorber_params()`` returns Bayesian-optimized absorber width and
coefficient values that are scaled for your specific grid resolution. These pre-optimized
parameters ensure good absorption performance without excessive computational overhead. The
returned absorption widths are asymmetric: the x-direction (propagation axis) gets a wider
absorber because light travels primarily along that axis, while the y and z directions receive
narrower absorbers since less energy reaches those boundaries.

The absorber is applied to the structure's conductivity field, creating a gradually increasing
loss region near each boundary. Electromagnetic waves entering this region are progressively
attenuated and effectively absorbed before reaching the grid edge.

.. code-block:: python

   # Get Bayesian-optimized absorber parameters scaled for our resolution
   abs_params = hwc.get_optimized_absorber_params(
       resolution_nm=RESOLUTION_UM * 1000,
       structure_dimensions=(Lx, Ly, Lz),
   )

   ABS_COEFF = abs_params['absorber_coeff']
   abs_shape = abs_params['absorption_widths']

   print(f"Absorber width: {abs_params['absorber_width']} cells")
   print(f"Absorber coeff: {ABS_COEFF:.6f}")
   print(f"Absorption widths (x,y,z): {abs_shape}")

   # Create absorption mask
   absorber = hwc.create_absorption_mask(
       grid_shape=(Lx, Ly, Lz),
       absorption_widths=abs_shape,
       absorption_coeff=ABS_COEFF,
       show_plots=True,
   )

   # Add absorber to structure conductivity
   structure.conductivity = jnp.zeros_like(structure.conductivity) + absorber

Step 5: Create Mode Source
--------------------------

The FDTD source is a waveguide mode profile computed by solving a 2D cross-section eigenvalue
problem. Rather than injecting a simple plane wave, this approach excites a physically accurate
waveguide mode that matches the structure, resulting in clean coupling with minimal reflections.

The process works in several stages. First, waveguide positions are auto-detected by placing
temporary monitors at the source plane -- this finds where the high-index core regions are. Next,
the mode-solving region is expanded to 2x the detected waveguide size. This expansion is critical:
it ensures the mode field fully decays to zero at the boundaries of the solving region, preventing
truncation artifacts that would introduce spurious reflections. After the eigenvalue solve, the
source field is trimmed back to the expanded region to minimize the amount of data that must be
transferred to the GPU.

Key parameters to be aware of:

- **mode_num=0**: Selects the fundamental TE mode. Use 1, 2, etc. to select higher-order modes.
- **source_pos_x**: The x-position where the source is placed, set just inside the absorber
  boundary so the source excites a clean waveguide mode in the straight extension region.
- **visualize=True**: Displays the solved mode profile and permittivity cross-section so you can
  visually verify that the mode solver found the correct mode.

.. code-block:: python

   # Wavelength and frequency settings
   WL_UM = 1.55
   wl_cells = WL_UM / RESOLUTION_UM
   freq_band = (2 * jnp.pi / wl_cells, 2 * jnp.pi / wl_cells, 1)

   # Source position (after absorber region)
   source_pos_x = abs_shape[0]

   # Auto-detect waveguide bounds using temporary monitor placement
   temp_monitors = hwc.MonitorSet()
   temp_monitors.add_monitors_at_position(
       structure=structure,
       axis="x",
       position=source_pos_x,
       label="source_detect",
   )

   # Use the first detected waveguide (bottom input port)
   source_monitor = temp_monitors.monitors[0]

   # Expand bounds to 2x for mode solving (ensures mode field decays to zero)
   y_min_orig = source_monitor.offset[1]
   y_max_orig = y_min_orig + source_monitor.shape[1]
   z_min_orig = source_monitor.offset[2]
   z_max_orig = z_min_orig + source_monitor.shape[2]

   y_center = (y_min_orig + y_max_orig) // 2
   z_center = (z_min_orig + z_max_orig) // 2
   y_half = source_monitor.shape[1]
   z_half = source_monitor.shape[2]

   y_min = max(0, y_center - y_half)
   y_max = min(Ly, y_center + y_half)
   z_min = max(0, z_center - z_half)
   z_max = min(Lz, z_center + z_half)

   # Create mode source in expanded region
   source_field, source_offset, mode_info = hwc.create_mode_source(
       structure=structure,
       freq_band=freq_band,
       mode_num=0,
       propagation_axis="x",
       source_position=source_pos_x,
       perpendicular_bounds=(y_min, y_max),
       z_bounds=(z_min, z_max),
       visualize=True,
       visualize_permittivity=True,
   )

   # Trim source field to mode region (reduces data transfer)
   source_field_trimmed = source_field[:, :, :, y_min:y_max, z_min:z_max]
   source_offset_corrected = (source_pos_x, y_min, z_min)

   print(f"Source field shape: {source_field_trimmed.shape}")

Step 6: Set Up Monitors
-----------------------

Monitors are 3D volumes that record the electromagnetic field at specified locations during the
simulation. This step places monitors at each port of the original (un-extended) GDSFactory
component by converting port positions from physical coordinates (microns) to structure grid
coordinates. Ports facing left (180-degree orientation) are labeled as inputs, and ports facing
right (0-degree orientation) are labeled as outputs.

In addition to port monitors, an ``xy_mid`` monitor is added that captures a full XY plane slice
at the waveguide center height. This monitor is used for visualizing the field propagation pattern
across the entire device after simulation.

The monitor sizing parameters control the spatial extent of each port monitor:

- **MONITOR_THICKNESS**: The number of cells along the propagation axis (x). Averaging the
  recorded field across this thickness reduces noise in the transmission calculation.
- **MONITOR_HALF**: The half-extent of the monitor in the Y and Z directions. This must be large
  enough to capture the full spatial extent of the waveguide mode field, including its evanescent
  tails in the cladding.

.. code-block:: python

   # Create monitor set
   monitors = hwc.MonitorSet()

   # Coordinate mapping from device_info
   y_pad_struct = PADDING[0] // 2
   x_pad_struct = PADDING[2] // 2
   bbox = device_info['bounding_box_um']
   x_min_um, y_min_um = bbox[0], bbox[1]
   theta_res = device_info['theta_resolution_um']

   # Monitor sizing
   MONITOR_THICKNESS = 5
   MONITOR_HALF = 35
   z_wg_center = clad_thickness + wg_thickness // 2

   # Place monitors at original device ports
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
           offset=(x_struct, y_struct - MONITOR_HALF, z_wg_center - MONITOR_HALF)
       )
       monitors.add(monitor, label)

   # Add xy_mid monitor for field visualization
   xy_mid_monitor = hwc.Monitor(
       shape=(Lx, Ly, 1),
       offset=(0, 0, z_wg_center)
   )
   monitors.add(xy_mid_monitor, name="xy_mid")

   monitors.list_monitors()

   # Visualize
   monitors.view(
       structure=structure,
       axis="z",
       position=z_wg_center,
       source_position=source_pos_x,
       absorber_boundary=absorber,
   )

Before running, preview the estimated cost:

.. code-block:: python

   dims = structure.permittivity.shape
   cost = hwc.estimate_cost(
       grid_points=dims[1] * dims[2] * dims[3],
       max_steps=20000,
   )
   print(f"Estimated time: {cost['estimated_seconds']:.0f}s")
   print(f"Estimated cost: ${cost['estimated_cost_usd']:.2f}")

See :ref:`gpu-cost-estimation` for details.

Step 7: Run GPU Simulation
--------------------------

This is the only step that costs credits (1 credit = $25 = 1 hour of B200 compute). Everything up to this point has been free local computation; now the prepared structure is sent to a cloud GPU for the actual FDTD time-stepping.

The function ``structure.extract_recipe()`` serializes the full 3D permittivity and conductivity
arrays into a compact format suitable for transfer to the cloud. The ``simulate()`` function then
sends this structure recipe along with the source field and monitor configuration to the server,
which reconstructs the structure on the GPU and runs the FDTD time-stepping algorithm.

Key parameters that control the simulation:

- **simulation_steps**: The maximum number of FDTD time steps. A value of 20000 is typically
  sufficient for most photonic devices to reach steady state.
- **check_every_n**: How often (in time steps) to check for convergence. Every 1000 steps, the
  solver evaluates whether the fields have settled and can terminate early if convergence criteria
  are met.
- **source_ramp_periods**: The number of optical periods over which to gradually ramp the source
  amplitude from zero to full power. This gradual turn-on reduces transient artifacts that would
  otherwise contaminate the field monitors.

.. code-block:: python

   import hyperwave_community as hwc

   # Configure and validate API key
   from google.colab import userdata
   hwc.configure_api(api_key=userdata.get('HYPERWAVE_API_KEY'))
   hwc.get_account_info()

   # Extract recipes for API
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
       check_every_n=1000,
       source_ramp_periods=5.0,
       add_absorption=True,
       absorption_widths=abs_shape,
       absorption_coeff=ABS_COEFF,
   )

   print(f"GPU time: {results['sim_time']:.2f}s")
   print(f"Performance: {results['performance']:.2e} grid-points*steps/s")

Step 8: Analyze Results
-----------------------

Monitor data is returned as arrays of shape ``(N_freq, 6, ...)`` containing all six electromagnetic
field components (Ex, Ey, Ez, Hx, Hy, Hz) at each frequency point. The analysis proceeds in three
stages: first, the field is averaged across the monitor thickness (the x-dimension) to produce a
single 2D cross-section per monitor; second, the Poynting vector ``S = 0.5 * Re(E x H*)`` is
computed using ``hwc.S_from_slice()``; and third, the x-component of S (the component along the
propagation direction) is integrated over the monitor cross-section to obtain the total power
flowing through each port.

For the MMI 2x2 splitter simulated here, ``Input_o1`` excites the bottom input port. The power is
then measured at ``Output_o3`` (the cross port, on the opposite side of the MMI) and ``Output_o4``
(the bar port, on the same side). An ideal 50:50 splitter produces -3 dB at each output port with
zero excess loss. Deviation from -3 dB indicates either imbalance between the outputs or excess
scattering/radiation loss within the device.

.. code-block:: python

   # Get monitor data
   monitor_data = results['monitor_data']
   print(f"Available monitors: {list(monitor_data.keys())}")

   # MMI 2x2 port mapping:
   #   Input_o1 (bottom) ---> Output_o4 (bottom, bar)
   #   Input_o2 (top)    ---> Output_o3 (top, bar)
   input_name = "Input_o1"
   bar_name = "Output_o4"
   cross_name = "Output_o3"

   input_fields = monitor_data[input_name]
   bar_fields = monitor_data[bar_name]
   cross_fields = monitor_data[cross_name]

   # Average across monitor thickness (X dimension)
   input_plane = jnp.mean(input_fields, axis=2)
   bar_plane = jnp.mean(bar_fields, axis=2)
   cross_plane = jnp.mean(cross_fields, axis=2)

   # Calculate Poynting vectors
   S_in = hwc.S_from_slice(input_plane)
   S_bar = hwc.S_from_slice(bar_plane)
   S_cross = hwc.S_from_slice(cross_plane)

   # Calculate power (X-component for x-propagating mode)
   power_in = jnp.abs(jnp.sum(S_in[:, 0, :, :], axis=(1, 2)))
   power_bar = jnp.abs(jnp.sum(S_bar[:, 0, :, :], axis=(1, 2)))
   power_cross = jnp.abs(jnp.sum(S_cross[:, 0, :, :], axis=(1, 2)))
   power_out_total = power_bar + power_cross

   # Transmission metrics
   T_bar = power_bar / power_in
   T_cross = power_cross / power_in
   T_total = power_out_total / power_in
   excess_loss_dB = 10 * jnp.log10(T_total)

   print(f"Bar port ({bar_name}):    T = {float(T_bar[0]):.5f}")
   print(f"Cross port ({cross_name}): T = {float(T_cross[0]):.5f}")
   print(f"Total transmission:        {float(T_total[0]):.5f}")
   print(f"Excess loss:               {float(excess_loss_dB[0]):.2f} dB")

Summary
-------

.. list-table::
   :header-rows: 1

   * - Step
     - Function
     - Runs On
     - Cost
   * - 1
     - ``component_to_theta()``
     - Local
     - Free
   * - 2
     - ``density()``
     - Local
     - Free
   * - 3
     - ``Layer()``, ``create_structure()``
     - Local
     - Free
   * - 4
     - ``create_absorption_mask()``
     - Local
     - Free
   * - 5
     - ``create_mode_source()``
     - Local
     - Free
   * - 6
     - ``MonitorSet()``
     - Local
     - Free
   * - 7
     - ``simulate()``
     - Cloud GPU
     - Credits ($25/hr)
   * - 8
     - Analysis
     - Local
     - Free

When to Use Local Workflow
--------------------------

* **Custom structures**: Create theta patterns not available in GDSFactory
* **Inverse design**: Optimize theta as a design variable with gradients
* **Debugging**: Inspect all intermediate arrays (density, permittivity, etc.)
* **Large structures**: Pre-process locally before committing to GPU time

Next Steps
----------

* :doc:`../gpu_options` - GPU performance and cost reference
* :doc:`../convergence` - Configure early stopping
* :doc:`../api` - Full API reference
* :doc:`api_workflow` - Simpler workflow using API functions
