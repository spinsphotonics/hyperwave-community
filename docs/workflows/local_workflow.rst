Local Workflow Tutorial
=======================

This tutorial walks through the local workflow for simulating an MMI 2x2 splitter. The local workflow gives you full control over structure creation, allowing custom theta patterns and inspection of intermediate data.

**Download the notebook**: `getting_started_local.ipynb <https://github.com/spinsphotonics/hyperwave-community/blob/main/examples/getting_started_local.ipynb>`_

.. contents:: Steps
   :local:
   :depth: 1

Configure API
-------------

First, configure your API key. Get one from `spinsphotonics.com <https://spinsphotonics.com>`_.

.. code-block:: python

   import hyperwave_community as hwc

   hwc.configure_api(api_key="your-api-key-here")
   hwc.get_account_info()  # Check your credits

Step 1a: Load Component (Theta)
-------------------------------

Load a GDSFactory component and convert it to a binary theta pattern. This step runs entirely locally.

.. code-block:: python

   # Component settings
   COMPONENT_NAME = "mmi2x2_with_sbend"
   RESOLUTION_NM = 20
   EXTENSION_LENGTH = 2.0

   # Load component to theta
   theta_result = hwc.load_component(
       component_name=COMPONENT_NAME,
       resolution_nm=RESOLUTION_NM,
       extension_length=EXTENSION_LENGTH,
       show_plot=True,  # Visualize the pattern
   )

   print(f"Theta shape: {theta_result['theta'].shape}")
   print(f"Ports: {list(theta_result['port_info'].keys())}")

**Output:**

.. code-block:: text

   Theta shape: (3600, 700)
   Ports: ['o1', 'o2', 'o3', 'o4']

Exploring Components
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # List available components
   print("Available components:")
   for comp in hwc.list_components()[:10]:
       print(f"  - {comp}")

   # Get parameters for a component
   params = hwc.get_component_params("mmi2x2")
   print(f"Parameters: {params}")

   # Preview with custom parameters
   preview = hwc.preview_component(
       component_name="mmi2x2",
       component_kwargs={"width_mmi": 6.0, "length_mmi": 10.0},
       extension_length=2.0,
       show_plot=True,
   )

Custom Theta Patterns
^^^^^^^^^^^^^^^^^^^^^

For inverse design or custom structures, you can create theta directly:

.. code-block:: python

   import numpy as np

   # Create a custom waveguide pattern
   theta = np.zeros((500, 1000))
   center_y = theta.shape[0] // 2
   wg_width = 25  # pixels

   # Straight waveguide
   theta[center_y - wg_width//2 : center_y + wg_width//2, :] = 1.0

   # Use with build_recipe_from_theta by creating a theta_result dict

Step 1b: Build Recipe from Theta
--------------------------------

Convert the theta pattern to a 3D structure recipe. This step processes the pattern locally.

.. code-block:: python

   # Material properties
   N_CORE = 3.48      # Silicon
   N_CLAD = 1.4457    # SiO2

   # Build recipe from theta
   recipe_result = hwc.build_recipe_from_theta(
       theta_result=theta_result,
       n_core=N_CORE,
       n_clad=N_CLAD,
       wg_height_um=0.22,
       total_height_um=4.0,
       padding=(100, 100, 0, 0),
       density_radius=3,
       vertical_radius=2.0,
       show_structure=True,  # Visualize the 3D structure
   )

   print(f"Structure dimensions: {recipe_result['dimensions']}")
   print(f"Ports: {list(recipe_result['port_info'].keys())}")

**Output:**

.. code-block:: text

   Structure dimensions: (1800, 350, 199)
   Ports: ['o1', 'o2', 'o3', 'o4']

Inspecting Intermediate Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The local workflow gives you access to all intermediate arrays:

.. code-block:: python

   import numpy as np

   # Access the structure object for visualization
   structure = recipe_result['structure']

   # Access density arrays (before 3D construction)
   print(f"density_core shape: {recipe_result['density_core'].shape}")
   print(f"density_core dtype: {recipe_result['density_core'].dtype}")
   print(f"density_core size: {np.array(recipe_result['density_core']).nbytes / 1024 / 1024:.2f} MB")

   # Layer configuration
   print(f"layer_config: {recipe_result['layer_config']}")

Comparing Local vs API
^^^^^^^^^^^^^^^^^^^^^^

You can verify that local processing matches the API:

.. code-block:: python

   # Build via API for comparison
   api_recipe_result = hwc.build_recipe(
       component_name=COMPONENT_NAME,
       resolution_nm=RESOLUTION_NM,
       extension_length=EXTENSION_LENGTH,
       n_core=N_CORE,
       n_clad=N_CLAD,
       wg_height_um=0.22,
       total_height_um=4.0,
       padding=(100, 100, 0, 0),
       density_radius=3,
       vertical_radius=2.0,
   )

   # Compare dimensions
   print(f"Local: {recipe_result['dimensions']}")
   print(f"API:   {api_recipe_result['dimensions']}")

Step 2: Build Monitors
----------------------

Set up field monitors at input and output ports.

.. code-block:: python

   SOURCE_PORT = "o1"

   monitor_result = hwc.build_monitors(
       port_info=recipe_result['port_info'],
       dimensions=recipe_result['dimensions'],
       source_port=SOURCE_PORT,
       resolution_um=recipe_result['resolution_um'],
       structure_recipe=recipe_result['recipe'],
       show_structure=True,
   )

   print(f"Monitors: {list(monitor_result['monitor_names'].keys())}")
   print(f"Source port: {monitor_result['source_port_name']}")
   print(f"Source position: x={monitor_result['source_position']}")

Step 3: Compute Frequency Band
------------------------------

.. code-block:: python

   WL_CENTER_UM = 1.55
   N_FREQS = 1

   freq_result = hwc.compute_freq_band(
       wl_min_um=WL_CENTER_UM,
       wl_max_um=WL_CENTER_UM,
       n_freqs=N_FREQS,
       resolution_um=recipe_result['resolution_um'],
   )

   print(f"Frequency band: {freq_result['freq_band']}")
   print(f"Wavelengths: {freq_result['wavelengths_um']}")

Step 4: Solve Waveguide Mode
----------------------------

.. code-block:: python

   # Debug info available in local workflow
   print(f"density_core shape: {recipe_result['density_core'].shape}")
   print(f"mode_bounds: {monitor_result['mode_bounds']}")
   print(f"source_position: {monitor_result['source_position']}")

   MODE_NUM = 0

   source_result = hwc.solve_mode_source(
       density_core=recipe_result['density_core'],
       density_clad=recipe_result['density_clad'],
       source_x_position=monitor_result['source_position'],
       mode_bounds=monitor_result['mode_bounds'],
       layer_config=recipe_result['layer_config'],
       eps_values=recipe_result['eps_values'],
       freq_band=freq_result['freq_band'],
       mode_num=MODE_NUM,
       show_mode=True,
   )

   print(f"Source field shape: {source_result['source_field'].shape}")
   print(f"Source offset: {source_result['source_offset']}")

Step 5: Run Simulation
----------------------

This step uses GPU credits. See :doc:`../gpu_options` and :doc:`../convergence` for options.

.. code-block:: python

   NUM_STEPS = 20000
   GPU_TYPE = "B200"

   # Get optimized absorber parameters
   absorber_params = hwc.get_optimized_absorber_params(
       resolution_nm=RESOLUTION_NM,
       wavelength_um=WL_CENTER_UM,
       structure_dimensions=recipe_result['dimensions'],
   )
   print(f"Absorber params: {absorber_params}")

   # Run simulation
   results = hwc.run_simulation(
       device_type=COMPONENT_NAME,
       recipe_result=recipe_result,
       monitor_result=monitor_result,
       freq_result=freq_result,
       source_result=source_result,
       num_steps=NUM_STEPS,
       gpu_type=GPU_TYPE,
       absorption_widths=absorber_params['absorption_widths'],
       absorption_coeff=absorber_params['absorber_coeff'],
       convergence="default",
   )

   print(f"Simulation time: {results['sim_time']:.1f}s")
   print(f"Total execution time: {results['total_time']:.1f}s")
   if results.get('converged'):
       print(f"Converged at step: {results['convergence_step']}")

Step 6: Analyze Results
-----------------------

Transmission Analysis
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   transmission = hwc.analyze_transmission(
       results,
       input_monitor="Input_o1",
       output_monitors=["Output_o3", "Output_o4"],
   )

   print(f"Input power: {transmission['power_in']:.4f}")
   print(f"Total transmission: {transmission['total_transmission']:.4f}")
   print(f"Excess loss: {transmission['excess_loss_dB']:.2f} dB")

Field Visualization
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt

   field_data = hwc.get_field_intensity_2d(
       results,
       monitor_name='xy_mid',
       dimensions=recipe_result['dimensions'],
       resolution_um=recipe_result['resolution_um'],
       freq_band=freq_result['freq_band'],
   )

   plt.figure(figsize=(12, 5))
   plt.imshow(
       field_data['intensity'],
       origin='upper',
       extent=field_data['extent'],
       cmap='jet',
       aspect='equal'
   )
   plt.xlabel('x (μm)')
   plt.ylabel('y (μm)')
   plt.title(f"|E|² at λ = {field_data['wavelength_nm']:.1f} nm")
   plt.colorbar(label='|E|²')
   plt.show()

Summary
-------

.. list-table::
   :header-rows: 1

   * - Step
     - Function
     - Cost
     - Notes
   * - 1a
     - ``load_component()``
     - Free
     - Local processing
   * - 1b
     - ``build_recipe_from_theta()``
     - Free
     - Local processing
   * - 2
     - ``build_monitors()``
     - Free
     - API call
   * - 3
     - ``compute_freq_band()``
     - Free
     - API call
   * - 4
     - ``solve_mode_source()``
     - Free
     - API call
   * - 5
     - ``run_simulation()``
     - Credits
     - GPU simulation
   * - 6
     - ``analyze_transmission()``
     - Free
     - Local analysis

When to Use Local Workflow
--------------------------

* **Custom structures**: Create theta patterns that aren't available in GDSFactory
* **Inverse design**: Optimize theta as a design variable
* **Debugging**: Inspect intermediate arrays and verify processing
* **Large-scale studies**: Process structures locally before committing to GPU time

Next Steps
----------

* :doc:`../gpu_options` - Choose the right GPU for your simulation
* :doc:`../convergence` - Configure early stopping
* :doc:`../api` - Full API reference
* :doc:`api_workflow` - Simpler workflow for standard components
