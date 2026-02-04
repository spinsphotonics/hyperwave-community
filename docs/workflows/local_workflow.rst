Local Workflow Tutorial
=======================

This tutorial walks through the local workflow for FDTD photonics simulations. The local workflow runs all CPU steps on your machine using hyperwave functions directly. Only the GPU simulation requires an API call.

**Download the notebook**: `getting_started_local.ipynb <https://github.com/spinsphotonics/hyperwave-community/blob/main/examples/getting_started_local.ipynb>`_

.. contents:: Steps
   :local:
   :depth: 1

Imports
-------

.. code-block:: python

   import hyperwave_community as hwc
   import gdsfactory as gf
   import matplotlib.pyplot as plt
   import numpy as np
   import jax.numpy as jnp

Step 1: Load GDSFactory Component
---------------------------------

Load a photonic component from GDSFactory and convert it to a binary theta pattern.

.. code-block:: python

   # Component settings
   COMPONENT_NAME = "mmi2x2_with_sbend"
   RESOLUTION_UM = 0.02  # 20nm resolution

   # Load component from GDSFactory
   component = gf.components.mmi2x2_with_sbend()

   # Convert to theta pattern
   theta, device_info = hwc.component_to_theta(
       component=component,
       resolution=RESOLUTION_UM,
   )

   print(f"Theta shape: {theta.shape}")
   print(f"Device size: {device_info['physical_size_um']} um")

   # Visualize
   plt.imshow(theta, cmap='gray', aspect='equal')
   plt.title('Theta Pattern')
   plt.show()

Custom Theta Patterns
^^^^^^^^^^^^^^^^^^^^^

You can also create custom theta patterns for inverse design:

.. code-block:: python

   # Create a custom waveguide pattern
   theta = np.zeros((500, 1000))
   center_y = theta.shape[0] // 2
   wg_width = 25  # pixels

   # Straight waveguide
   theta[center_y - wg_width//2 : center_y + wg_width//2, :] = 1.0

Step 2: Apply Density Filtering
-------------------------------

Smooth the theta pattern with density filtering for numerical stability.

.. code-block:: python

   # Material properties
   N_CORE = 3.48      # Silicon refractive index
   N_CLAD = 1.4457    # SiO2 cladding refractive index

   # Padding for absorbers and monitors
   PADDING = (100, 100, 0, 0)  # (left, right, top, bottom)

   # Apply density filtering
   density_core = hwc.density(
       theta=theta,
       pad_width=PADDING,
       radius=3,
   )

   density_clad = hwc.density(
       theta=jnp.zeros_like(theta),
       pad_width=PADDING,
       radius=3,
   )

   print(f"Density shape: {density_core.shape}")

Step 3: Build 3D Layer Structure
--------------------------------

Stack layers to create the 3D permittivity structure.

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

Add PML absorbing boundaries to prevent reflections.

.. code-block:: python

   # Absorber parameters
   ABS_WIDTH_X = 82
   ABS_WIDTH_Y = 41
   ABS_WIDTH_Z = 41
   ABS_COEFF = 6.17e-4

   abs_shape = (ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z)

   # Create absorption mask
   absorber = hwc.create_absorption_mask(
       grid_shape=(Lx, Ly, Lz),
       absorption_widths=abs_shape,
       absorption_coeff=ABS_COEFF,
   )

   # Add absorber to structure
   structure.conductivity = jnp.zeros_like(structure.conductivity) + absorber

Step 5: Create Mode Source
--------------------------

Solve for the fundamental waveguide mode at the input port.

.. code-block:: python

   # Wavelength and frequency settings
   WL_UM = 1.55
   wl_cells = WL_UM / RESOLUTION_UM
   freq_band = (2 * jnp.pi / wl_cells, 2 * jnp.pi / wl_cells, 1)

   # Source position (after absorber)
   source_pos_x = ABS_WIDTH_X + 5

   # Create mode source
   source_field, source_offset, mode_info = hwc.create_mode_source(
       structure=structure,
       freq_band=freq_band,
       mode_num=0,
       propagation_axis="x",
       source_position=source_pos_x,
       perpendicular_bounds=(0, Ly // 2),  # Bottom waveguide only
       visualize=True,
   )

   print(f"Source field shape: {source_field.shape}")

Step 6: Set Up Monitors
-----------------------

Configure field monitors at input and output ports.

.. code-block:: python

   # Create monitor set
   monitors = hwc.MonitorSet()

   # Input monitor
   monitors.add_monitors_at_position(
       structure=structure,
       axis="x",
       position=ABS_WIDTH_X + 10,
       label="Input",
   )

   # Output monitors
   monitors.add_monitors_at_position(
       structure=structure,
       axis="x",
       position=Lx - (ABS_WIDTH_X + 10),
       label="Output",
   )

   monitors.list_monitors()

   # Visualize
   monitors.view(
       structure=structure,
       axis="z",
       position=Lz // 2,
       source_position=source_pos_x,
   )

Step 7: Run GPU Simulation
--------------------------

This step runs on cloud GPU and requires an API key.

.. code-block:: python

   API_KEY = "your-api-key-here"

   results = hwc.simulate(
       structure=structure,
       source_field=source_field,
       source_offset=source_offset,
       freq_band=freq_band,
       monitors=monitors,
       mode_info=mode_info,
       simulation_steps=20000,
       check_every_n=1000,
       source_ramp_periods=5.0,
       add_absorption=True,
       absorption_widths=abs_shape,
       absorption_coeff=ABS_COEFF,
       api_key=API_KEY,
       gpu_type="B200",
   )

   print(f"GPU time: {results['sim_time']:.2f}s")

Step 8: Analyze Results
-----------------------

Analyze transmission using Poynting flux calculations.

.. code-block:: python

   # Get monitor data
   monitor_data = results['monitor_data']

   # Poynting vector calculation
   def S_from_slice(field_slice):
       E = field_slice[:, :3, :, :]
       H = field_slice[:, 3:, :, :]
       S = jnp.zeros_like(E, dtype=jnp.float32)
       S = S.at[:, 0].set(jnp.real(E[:, 1] * jnp.conj(H[:, 2]) - E[:, 2] * jnp.conj(H[:, 1])))
       return S * 0.5

   # Calculate power
   input_plane = jnp.mean(monitor_data['Input_bottom'], axis=2)
   S_in = S_from_slice(input_plane)
   power_in = jnp.abs(jnp.sum(S_in[:, 0], axis=(1, 2)))

   # ... similar for outputs

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
     - Credits
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

* :doc:`../gpu_options` - Choose the right GPU for your simulation
* :doc:`../convergence` - Configure early stopping
* :doc:`../api` - Full API reference
* :doc:`api_workflow` - Simpler workflow using API functions
