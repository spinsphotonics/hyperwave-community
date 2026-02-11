Workflows
=========

Hyperwave Community offers two workflows for running FDTD photonics simulations. Both produce identical results - choose based on where you want CPU work to run.

.. contents:: On this page
   :local:
   :depth: 2

API Workflow
------------

All CPU steps (structure creation, mode solving, etc.) run on Modal servers provided by SPINs. You send a single ``build_recipe()`` call and get back the full structure.

**Use this workflow when:**

* You want to simulate standard GDSFactory components (MMIs, couplers, bends, etc.)
* You want minimal code
* You don't need to inspect intermediate data

**Example:**

.. code-block:: python

   import hyperwave_community as hwc

   hwc.configure_api(api_key="your-key")

   # Single call creates the entire structure
   recipe_result = hwc.build_recipe(
       component_name="mmi2x2_with_sbend",
       resolution_nm=20,
       n_core=3.48,
       n_clad=1.4457,
       ...
   )

:doc:`api_workflow` - Full tutorial

Local Workflow
--------------

All CPU steps run locally on your machine (or Colab). You build the structure step by step using hyperwave functions directly. Only the GPU simulation requires an API call.

**Use this workflow when:**

* You need custom structures not available in GDSFactory
* You're doing inverse design / optimization
* You want to inspect or modify intermediate arrays (theta, density, permittivity)
* You're running in Colab and want to use Colab's CPU

**Example:**

.. code-block:: python

   import hyperwave_community as hwc
   import gdsfactory as gf

   # Load component and convert to theta pattern
   component = gf.components.mmi2x2_with_sbend()
   theta, device_info = hwc.component_to_theta(
       component=component,
       resolution=0.02,
   )

   # Build structure step by step
   density_core = hwc.density(theta=theta, pad_width=(100, 100, 0, 0), radius=3)
   structure = hwc.create_structure(layers=[...], vertical_radius=2.0)

:doc:`local_workflow` - Full tutorial

Workflow Comparison
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - **API Workflow**
     - **Local Workflow**
   * - CPU work runs on
     - Modal (SPINs servers)
     - Your machine / Colab
   * - Structure creation
     - Single API call
     - Step-by-step local functions
   * - GDSFactory components
     - Built-in support
     - Full customization
   * - Custom theta patterns
     - Not supported
     - Full support
   * - Intermediate inspection
     - Limited
     - Full access to all arrays
   * - Lines of code
     - ~30 lines
     - ~50 lines

Shared Steps
------------

Both workflows share the same simulation and analysis steps:

.. code-block:: python

   # Build monitors (free)
   monitor_result = hwc.build_monitors(...)

   # Compute frequency band (free)
   freq_result = hwc.compute_freq_band(...)

   # Solve mode source (free)
   source_result = hwc.solve_mode_source(...)

   # Run simulation (costs credits: 1 credit = $25 = 1 hr)
   results = hwc.run_simulation(...)

   # Analyze results (free, local)
   transmission = hwc.analyze_transmission(...)

Cost Structure
--------------

Both workflows have the same cost structure:

.. list-table::
   :header-rows: 1

   * - Step
     - Cost
   * - Structure creation (Steps 1-4)
     - Free
   * - GPU Simulation (Step 5)
     - Credits ($25/hr)
   * - Analysis (Step 6)
     - Free

Credits are only consumed when running ``run_simulation()``. All simulations run on NVIDIA B200 GPUs at $25 per compute hour (1 credit = $25 = 1 hour). All other functions are free but require a valid API key.

.. toctree::
   :maxdepth: 1
   :hidden:

   api_workflow
   local_workflow
   inverse_design
