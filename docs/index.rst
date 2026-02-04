Hyperwave Community
===================

GPU-accelerated FDTD photonics simulation via cloud API.

Features
--------

* **GDSFactory Integration**: Import photonic components directly from GDSFactory
* **GPU-Accelerated Simulation**: Run FDTD simulations on cloud GPUs (B200, H200, H100, A100)
* **Two Workflows**: Simple API workflow or detailed local workflow
* **Early Stopping**: Smart convergence detection to optimize simulation time
* **Power Analysis**: Poynting flux calculations and transmission spectra
* **Visualization**: Built-in plotting for structures, modes, and field intensities

Quick Example
-------------

.. code-block:: python

   import hyperwave_community as hwc

   # Configure API
   hwc.configure_api(api_key="your-api-key")

   # Build structure from GDSFactory component
   recipe = hwc.build_recipe(component_name="mmi2x2", resolution_nm=20, ...)

   # Set up monitors and source
   monitors = hwc.build_monitors(port_info=recipe['port_info'], ...)
   freq = hwc.compute_freq_band(wl_min_um=1.55, wl_max_um=1.55, ...)
   source = hwc.solve_mode_source(density_core=recipe['density_core'], ...)

   # Run simulation (uses credits)
   results = hwc.run_simulation(recipe_result=recipe, ..., gpu_type="H100")

   # Analyze results (free)
   transmission = hwc.analyze_transmission(results, ...)
   print(f"Transmission: {transmission['total_transmission']:.2%}")

Getting Started
---------------

1. :doc:`installation` - Install the package and get an API key
2. :doc:`workflows/index` - Choose between API and local workflows
3. :doc:`workflows/api_workflow` - Quick start tutorial (recommended for beginners)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   workflows/index
   workflows/api_workflow
   workflows/local_workflow

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   convergence
   gpu_options

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
