Quick Start Tutorial
====================

This tutorial walks through a complete FDTD simulation of an MMI 2x2 splitter.

**Download the notebook**: `getting_started.ipynb <https://github.com/spinsphotonics/hyperwave-community/blob/main/examples/getting_started.ipynb>`_

Configure API
-------------

Get your API key from `spinsphotonics.com <https://spinsphotonics.com>`_.

.. code-block:: python

   import hyperwave_community as hwc

   hwc.configure_api(api_key="your-api-key-here")
   hwc.get_account_info()  # Check your credits

Step 1: Build Structure Recipe
------------------------------

Build a 3D photonic structure from a GDSFactory component.

.. code-block:: python

   recipe_result = hwc.build_recipe(
       component_name="mmi2x2_with_sbend",
       resolution_nm=20,
       n_core=3.48,
       n_clad=1.4457,
       wg_height_um=0.22,
       total_height_um=4.0,
       extension_length=2.0,
       padding=[100, 100, 0, 0],
       density_radius=3,
       vertical_radius=2.0,
   )

   print(f"Structure dimensions: {recipe_result['dimensions']}")
   print(f"Ports: {list(recipe_result['port_info'].keys())}")

Step 2: Build Monitors
----------------------

.. code-block:: python

   monitor_result = hwc.build_monitors(
       port_info=recipe_result['port_info'],
       dimensions=recipe_result['dimensions'],
       source_port="o1",
       structure_recipe=recipe_result['recipe'],
       show_structure=True,  # Visualize structure with monitors
   )

   print(f"Monitors: {list(monitor_result['monitor_names'].keys())}")

Step 3: Compute Frequency Band
------------------------------

.. code-block:: python

   freq_result = hwc.compute_freq_band(
       wl_min_um=1.55,
       wl_max_um=1.55,
       n_freqs=1,
       resolution_um=recipe_result['resolution_um'],
   )

Step 4: Solve Waveguide Mode
----------------------------

.. code-block:: python

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

Step 5: Run Simulation
----------------------

This step uses GPU credits. See :doc:`gpu_options` for available GPUs and :doc:`convergence` for early stopping options.

.. code-block:: python

   results = hwc.run_simulation(
       device_type="mmi2x2_with_sbend",
       recipe_result=recipe_result,
       monitor_result=monitor_result,
       freq_result=freq_result,
       source_result=source_result,
       num_steps=20000,
       gpu_type="H100",
       convergence="default",
   )

   print(f"Simulation time: {results['sim_time']:.1f}s")
   if results.get('converged'):
       print(f"Converged at step: {results['convergence_step']}")

Step 6: Analyze Results
-----------------------

Analysis runs locally and is free.

.. code-block:: python

   import matplotlib.pyplot as plt

   # Transmission analysis
   transmission = hwc.analyze_transmission(
       results,
       input_monitor="Input_o1",
       output_monitors=["Output_o3", "Output_o4"],
   )

   print(f"Total transmission: {transmission['total_transmission']:.4f}")
   print(f"Excess loss: {transmission['excess_loss_dB']:.2f} dB")

   # Field intensity visualization
   field_data = hwc.get_field_intensity_2d(
       results,
       monitor_name='xy_mid',
       dimensions=recipe_result['dimensions'],
       resolution_um=recipe_result['resolution_um'],
       freq_band=freq_result['freq_band'],
   )

   plt.figure(figsize=(12, 5))
   plt.imshow(field_data['intensity'], origin='upper', extent=field_data['extent'], cmap='jet')
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
