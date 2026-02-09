Hyperwave Community
===================

GPU-accelerated FDTD photonics simulation via cloud API.

Features
--------

* **GDSFactory Integration**: Import photonic components directly from GDSFactory
* **GPU-Accelerated Simulation**: Run FDTD simulations on cloud GPUs (B200, H200, H100, A100)
* **Two Workflows**: API workflow (Modal CPU) or local workflow (your CPU)
* **Early Stopping**: Smart convergence detection to optimize simulation time
* **Power Analysis**: Poynting flux calculations and transmission spectra
* **Visualization**: Built-in plotting for structures, modes, and field intensities

Getting Started
---------------

1. :doc:`installation` - Install the package and get an API key
2. :doc:`workflows/index` - Choose between API and local workflows
3. :doc:`workflows/api_workflow` - API workflow tutorial
4. :doc:`workflows/local_workflow` - Local workflow tutorial

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   workflows/index
   workflows/api_workflow
   workflows/local_workflow

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Reference

   convergence
   gpu_options
   getting_started/colab_secrets

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
