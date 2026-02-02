Hyperwave Community Documentation
==================================

GPU-accelerated FDTD photonics simulation via cloud API.

Features
--------

* **GDSFactory Integration**: Import photonic components directly from GDSFactory
* **GPU-Accelerated Simulation**: Run FDTD simulations on cloud GPUs (B200, H200, H100, A100, etc.)
* **Early Stopping**: Smart convergence detection to optimize simulation time
* **Power Analysis**: Poynting flux calculations and transmission spectra
* **Visualization**: Built-in plotting for structures, modes, and field intensities

Installation
------------

.. code-block:: bash

   pip install hyperwave-community

Or install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/spinsphotonics/hyperwave-community.git

Workflow Overview
-----------------

The SDK workflow consists of:

1. **CPU Steps (free)**: Build structure, monitors, frequency band, and mode source
2. **GPU Step (uses credits)**: Run FDTD simulation
3. **Analysis (free, local)**: Analyze transmission and visualize fields

.. code-block:: python

   import hyperwave_community as hwc

   # Configure API
   hwc.configure_api(api_key="your-api-key")

   # CPU Steps (free)
   recipe_result = hwc.build_recipe(component_name="mmi2x2", ...)
   monitor_result = hwc.build_monitors(port_info=recipe_result['port_info'], ...)
   freq_result = hwc.compute_freq_band(wl_min_um=1.55, wl_max_um=1.55, ...)
   source_result = hwc.solve_mode_source(density_core=recipe_result['density_core'], ...)

   # GPU Step (uses credits)
   results = hwc.run_simulation(
       device_type="mmi2x2",
       recipe_result=recipe_result,
       monitor_result=monitor_result,
       freq_result=freq_result,
       source_result=source_result,
       gpu_type="H100",
       convergence="default",
   )

   # Analysis (free, local)
   transmission = hwc.analyze_transmission(results, input_monitor="Input_o1", ...)

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   api
   convergence
   gpu_options

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
