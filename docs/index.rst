Hyperwave Community Documentation
==================================

Welcome to the Hyperwave Community documentation. This package provides an open-source photonics simulation toolkit with GPU-accelerated FDTD via cloud API.

Features
--------

* **GDSFactory Integration**: Import photonic components directly from GDSFactory
* **Granular Workflow**: Step-by-step control with free CPU preprocessing
* **GPU-Accelerated Simulation**: Run FDTD simulations on cloud-based GPUs (B200, H200, H100, A100, etc.)
* **Early Stopping**: Smart convergence detection to save credits
* **Power Analysis**: Poynting flux calculations and transmission spectra
* **Visualization**: Built-in plotting for structures, modes, and field intensities

Installation
------------

.. code-block:: bash

   pip install hyperwave-community

Or install from source:

.. code-block:: bash

   git clone https://github.com/spinsphotonics/hyperwave-community.git
   cd hyperwave-community
   pip install -e .

Quick Start (Granular Workflow)
-------------------------------

The granular workflow is recommended for most users. CPU steps are **free** (require valid API key),
only the GPU simulation step consumes credits.

1. Configure API
~~~~~~~~~~~~~~~~

Get your API key from `spinsphotonics.com <https://spinsphotonics.com>`_.

.. code-block:: python

   import hyperwave_community as hwc

   # Configure and validate API key
   hwc.configure_api(api_key="your-api-key-here")
   hwc.get_account_info()  # Check your credits

2. Build Structure Recipe (Free)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a 3D photonic structure from a GDSFactory component.

.. code-block:: python

   # Build structure recipe from GDSFactory component
   recipe_result = hwc.build_recipe(
       component_name="mmi2x2_with_sbend",  # Any gdsfactory component
       resolution_nm=20,           # Grid resolution in nm
       n_core=3.48,                # Silicon refractive index
       n_clad=1.4457,              # SiO2 cladding refractive index
       wg_height_um=0.22,          # Waveguide core height in um
       total_height_um=4.0,        # Total simulation height
       extension_length=2.0,       # Port extension length in um
       padding=[100, 100, 0, 0],   # Padding cells (left, right, top, bottom)
   )

   print(f"Structure dimensions: {recipe_result['dimensions']}")
   print(f"Ports: {list(recipe_result['port_info'].keys())}")

3. Build Monitors (Free)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Build monitors from port information
   monitor_result = hwc.build_monitors(
       port_info=recipe_result['port_info'],
       dimensions=recipe_result['dimensions'],
       source_port="o1",           # Input port name
       structure_recipe=recipe_result['recipe'],
       show_structure=True,        # Visualize structure with monitors
   )

4. Compute Frequency Band (Free)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert wavelength to frequency band
   freq_result = hwc.compute_freq_band(
       wl_min_um=1.55,             # Center wavelength
       wl_max_um=1.55,
       n_freqs=1,
       resolution_um=recipe_result['resolution_um'],
   )

5. Solve Waveguide Mode (Free)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Solve for waveguide mode at source port
   source_result = hwc.solve_mode_source(
       density_core=recipe_result['density_core'],
       density_clad=recipe_result['density_clad'],
       source_x_position=monitor_result['source_position'],
       mode_bounds=monitor_result['mode_bounds'],
       layer_config=recipe_result['layer_config'],
       eps_values=recipe_result['eps_values'],
       freq_band=freq_result['freq_band'],
       mode_num=0,                 # Fundamental TE mode
       show_mode=True,             # Visualize mode profile
   )

6. Run Simulation (Uses Credits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run FDTD simulation - pass granular results directly
   results = hwc.run_simulation(
       device_type="mmi2x2_with_sbend",
       recipe_result=recipe_result,
       monitor_result=monitor_result,
       freq_result=freq_result,
       source_result=source_result,
       num_steps=20000,
       gpu_type="H100",            # Options: B200, H200, H100, A100, etc.
       convergence="default",      # or "quick", "thorough", "full"
   )

   print(f"Simulation time: {results['sim_time']:.1f}s")
   if results.get('converged'):
       print(f"Converged at step: {results['convergence_step']}")

7. Analyze Results (Free)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Transmission analysis
   transmission = hwc.analyze_transmission(
       results,
       input_monitor="Input_o1",
       output_monitors=["Output_o3", "Output_o4"],
   )

   # Field intensity visualization
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

Convergence Presets
-------------------

Control early stopping behavior to save credits:

- ``"quick"`` - Fast, fewer stability checks (2 checks at 2000 step intervals)
- ``"default"`` - Balanced approach (3 checks at 1000 step intervals)
- ``"thorough"`` - Conservative (5 checks, min 5000 steps)
- ``"full"`` - No early stopping, run all steps

All presets use 1% relative threshold.

For custom configuration:

.. code-block:: python

   convergence = hwc.ConvergenceConfig(
       check_every_n=500,
       relative_threshold=0.005,   # 0.5% threshold
       min_stable_checks=5,
       min_steps=3000,
   )

   results = hwc.run_simulation(..., convergence=convergence)

GPU Options
-----------

Available GPU types (in order of performance):

- ``"B200"`` - NVIDIA B200 (fastest)
- ``"H200"`` - NVIDIA H200
- ``"H100"`` - NVIDIA H100
- ``"A100-80GB"`` - NVIDIA A100 80GB
- ``"A100-40GB"`` - NVIDIA A100 40GB
- ``"L40S"`` - NVIDIA L40S
- ``"L4"`` - NVIDIA L4
- ``"A10G"`` - NVIDIA A10G
- ``"T4"`` - NVIDIA T4 (most economical)

.. toctree::
   :maxdepth: 2

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
