API Workflow
============

This tutorial walks through the API workflow for simulating an MMI 2x2 splitter. The API workflow is the simplest way to run FDTD simulations using standard GDSFactory components.

In the API workflow, all CPU-intensive steps (structure creation, density filtering, layer stacking) run on Modal servers provided by SPINs. You only need to specify the component and parameters; the server handles the rest. Only the GPU FDTD simulation step costs credits (1 credit = $25 = 1 hour of compute).

**Open in Google Colab**: `api_workflow.ipynb <https://colab.research.google.com/github/spinsphotonics/hyperwave-community/blob/main/examples/api_workflow.ipynb>`_

`Download the notebook <https://github.com/spinsphotonics/hyperwave-community/blob/main/examples/api_workflow.ipynb>`_

.. contents:: Steps
   :local:
   :depth: 1

Configure API
-------------

Your API key authenticates requests to the SPINs servers and tracks your credit balance. You can get one by signing up at `spinsphotonics.com <https://spinsphotonics.com>`_. After configuring the key, calling ``get_account_info()`` displays your remaining credits and account details so you can verify your setup before running any simulations.

.. code-block:: python

   import hyperwave_community as hwc

   from google.colab import userdata
   hwc.configure_api(api_key=userdata.get('HYPERWAVE_API_KEY'))
   hwc.get_account_info()  # Check your credits

Step 1: Build Structure Recipe
------------------------------

``build_recipe()`` is the core of the API workflow. A single call loads a GDSFactory component, converts it to a 2D design pattern (theta), applies density filtering, and builds the full 3D permittivity structure on the server. This is where the photonic device geometry is fully defined.

The key configurable parameters are:

- ``component_name``: Any GDSFactory component (e.g. ``"mmi1x2"``, ``"coupler"``, ``"bend_euler"``).
- ``resolution_nm``: Grid cell size in nanometers. Smaller values produce more accurate results but increase simulation time and memory. 20 nm is a good default for silicon photonics at 1550 nm.
- ``extension_length``: How far to extend waveguide ports, in micrometers. Longer extensions help with mode coupling and reduce reflections at the simulation boundary.
- ``n_core`` / ``n_clad``: Refractive indices of the core and cladding materials (e.g. 3.48 for silicon, 1.4457 for SiO2).
- ``padding``: Extra grid cells around the structure (left, right, top, bottom) to accommodate absorbing boundary layers and monitors.
- ``density_radius``: Smoothing radius for the density filter, which controls the minimum feature size.
- ``wg_height_um`` / ``total_height_um``: Waveguide slab thickness and total simulation domain height, both in micrometers.

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

Monitors are field sampling planes placed at specific positions in the simulation domain. They record the electromagnetic field at each frequency point during the simulation, which is later used to compute transmission and visualize field patterns. ``build_monitors()`` automatically places monitors at each port detected from the GDSFactory component.

The ``source_port`` parameter specifies which port to excite. Monitors at other ports measure the output transmission. Set ``show_structure=True`` to visualize where monitors are placed relative to the structure, which is useful for verifying the simulation setup before committing compute time.

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

FDTD simulations operate in normalized frequency units internally. ``compute_freq_band()`` converts physical wavelengths (in micrometers) to these internal units so that the solver can correctly discretize the source pulse and frequency-domain monitors.

For single-wavelength simulations, set ``wl_min_um`` equal to ``wl_max_um`` as shown below. For broadband sweeps, specify a wavelength range and set ``n_freqs`` greater than 1 to sample multiple frequency points in a single simulation run, which is far more efficient than running separate simulations at each wavelength.

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

Before running the FDTD simulation, we need to define the electromagnetic source that will excite the device. ``solve_mode_source()`` computes the fundamental waveguide mode profile at the source port by solving a 2D cross-section eigenvalue problem. This mode profile is then injected as a continuous-wave source during the simulation, ensuring that only the desired guided mode is launched into the structure.

Set ``mode_num=0`` for the fundamental TE mode, ``mode_num=1`` for the first higher-order mode, and so on. Set ``show_mode=True`` to visualize the solved mode profile, which is helpful for confirming that the correct mode was selected.

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

Estimate Cost
^^^^^^^^^^^^^

Before running, preview the estimated cost (see :ref:`gpu-cost-estimation` for details):

.. code-block:: python

   dims = recipe_result['dimensions']
   cost = hwc.estimate_cost(
       grid_points=dims[0] * dims[1] * dims[2],
       max_steps=20000,
   )
   print(f"Estimated time: {cost['estimated_seconds']:.0f}s")
   print(f"Estimated cost: ${cost['estimated_cost_usd']:.2f}")

Step 5: Run Simulation
----------------------

This is the only step that consumes credits (1 credit = $25 = 1 hour of B200 compute). The FDTD simulation runs on a cloud GPU, stepping Maxwell's equations forward in time until the fields converge or the maximum number of steps is reached. All previous steps (structure building, monitor placement, mode solving) were free server-side computations that prepared the inputs for this step.

The key parameters to configure are:

- ``num_steps``: Maximum number of simulation time steps. The simulation may stop early if convergence is detected.
- ``convergence``: Set to ``"default"`` for automatic early stopping when the fields have decayed, or ``"none"`` to run all steps regardless. See :doc:`../convergence` for details.
- Absorber parameters (boundary layer widths and absorption coefficients) are automatically optimized for your resolution and wavelength using ``get_optimized_absorber_params()``.

.. code-block:: python

   NUM_STEPS = 20000

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

All analysis runs locally on your machine and is free. No server calls or credits are consumed for this step.

Transmission Analysis
^^^^^^^^^^^^^^^^^^^^^

``analyze_transmission()`` computes the power flowing through each output monitor using Poynting flux integration and normalizes by the input power. The result includes per-port transmission in both linear and dB units, total transmission across all output ports, and excess loss. For an ideal 2x2 MMI splitter, each output should receive approximately 50% of the input power (-3 dB), with minimal excess loss indicating low scattering and absorption.

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

``get_field_intensity_2d()`` extracts the electric field intensity |E|^2 from a 2D monitor slice and returns it along with physical coordinates for plotting. The ``xy_mid`` monitor captures the field pattern at the middle of the waveguide layer, showing how light propagates through the input waveguide, distributes across the multimode interference region, and splits into the two output ports.

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
     - Credits ($25/hr)
   * - 6
     - ``analyze_transmission()``
     - Free

Next Steps
----------

* :doc:`../gpu_options` - GPU performance and cost reference
* :doc:`../convergence` - Configure early stopping
* :doc:`../api` - Full API reference
* :doc:`local_workflow` - Try the local workflow for more control
