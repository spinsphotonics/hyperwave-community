Hyperwave Community
===================

GPU-accelerated FDTD photonics simulation via cloud API.

Features
--------

* **GDSFactory Integration**: Import photonic components directly from GDSFactory
* **GPU-Accelerated Simulation**: Run FDTD simulations on cloud NVIDIA B200 GPUs
* **Two Workflows**: Local workflow for full control, or API workflow for integration
* **Early Stopping**: Smart convergence detection to optimize simulation time
* **Power Analysis**: Poynting flux calculations and transmission spectra
* **Visualization**: Built-in plotting for structures, modes, and field intensities

Getting Started
---------------

1. :doc:`installation` - Install the package and get an API key
2. :doc:`workflows/local_workflow` - **Start here**: step-by-step tutorial with full control
3. :doc:`workflows/api_workflow` - Single-call workflow for integration into existing systems
4. :doc:`workflows/inverse_design` - Inverse design optimization tutorial

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   workflows/index
   workflows/local_workflow
   workflows/api_workflow
   Inverse Design <workflows/inverse_design>

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

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
