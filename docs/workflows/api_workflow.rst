API Workflow Tutorial
=====================

This tutorial walks through the API workflow for simulating an MMI 2x2 splitter. The API workflow is the simplest way to run FDTD simulations using standard GDSFactory components.

**Download the notebook**: `getting_started.ipynb <https://github.com/spinsphotonics/hyperwave-community/blob/main/examples/getting_started.ipynb>`_

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

Step 1: Build Structure Recipe
------------------------------

Build a 3D photonic structure from a GDSFactory component with a single API call.

.. code-block:: python

   # Component settings
   COMPONENT_NAME = "mmi2x2_with_sbend"
   RESOLUTION_NM = 20
   EXTENSION_LENGTH = 2.0

   # Material properties
   N_CORE = 3.48      # Silicon
   N_CLAD = 1.4457    # SiO2

   # Build the recipe
   recipe_result = hwc.build_recipe(
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

   print(f"Structure dimensions: {recipe_result['dimensions']}")
   print(f"Ports: {list(recipe_result['port_info'].keys())}")

**Output:**

.. code-block:: text

   Structure dimensions: (1800, 350, 199)
   Ports: ['o1', 'o2', 'o3', 'o4']

Available Components
^^^^^^^^^^^^^^^^^^^^

To see available GDSFactory components:

.. code-block:: python

   # List all components
   for comp in hwc.list_components()[:10]:
       print(f"  - {comp}")

   # Get parameters for a component
   params = hwc.get_component_params("mmi2x2")
   print(params)

Step 2: Build Monitors
----------------------

Set up field monitors at input and output ports.

.. code-block:: python

   SOURCE_PORT = "o1"  # Input port

   monitor_result = hwc.build_monitors(
       port_info=recipe_result['port_info'],
       dimensions=recipe_result['dimensions'],
       source_port=SOURCE_PORT,
       resolution_um=recipe_result['resolution_um'],
       structure_recipe=recipe_result['recipe'],
       show_structure=True,  # Visualize
   )

   print(f"Monitors: {list(monitor_result['monitor_names'].keys())}")
   print(f"Source position: x={monitor_result['source_position']}")

**Output:**

.. code-block:: text

   Monitors: ['Input_o1', 'Output_o3', 'Output_o4', 'xy_mid', 'xz_mid']
   Source position: x=95

Step 3: Compute Frequency Band
------------------------------

Convert wavelengths to the frequency band used internally.

.. code-block:: python

   WL_CENTER_UM = 1.55  # 1550 nm
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

Compute the fundamental waveguide mode for the source.

.. code-block:: python

   MODE_NUM = 0  # Fundamental TE mode

   source_result = hwc.solve_mode_source(
       density_core=recipe_result['density_core'],
       density_clad=recipe_result['density_clad'],
       source_x_position=monitor_result['source_position'],
       mode_bounds=monitor_result['mode_bounds'],
       layer_config=recipe_result['layer_config'],
       eps_values=recipe_result['eps_values'],
       freq_band=freq_result['freq_band'],
       mode_num=MODE_NUM,
       show_mode=True,  # Visualize mode profile
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

**Output:**

.. code-block:: text

   Simulation completed in 62.5s (total: 128.4s)
   Converged at step 12000

Step 6: Analyze Results
-----------------------

Analysis runs locally and is free.

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

**Output:**

.. code-block:: text

   ============================================================
   Transmission Analysis (Input: Input_o1)
   Monitor              Transmission         dB
   ------------------------------------------------------------
   Output_o3                  0.4657      -3.32
   Output_o4                  0.4753      -3.23
   ------------------------------------------------------------
   Total                      0.9410      -0.26
   ============================================================

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
   * - 1
     - ``build_recipe()``
     - Free
   * - 2
     - ``build_monitors()``
     - Free
   * - 3
     - ``compute_freq_band()``
     - Free
   * - 4
     - ``solve_mode_source()``
     - Free
   * - 5
     - ``run_simulation()``
     - Credits
   * - 6
     - ``analyze_transmission()``
     - Free

Next Steps
----------

* :doc:`../gpu_options` - Choose the right GPU for your simulation
* :doc:`../convergence` - Configure early stopping
* :doc:`../api` - Full API reference
* :doc:`local_workflow` - Try the local workflow for more control
