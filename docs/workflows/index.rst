Workflows
=========

Hyperwave Community offers two workflows for running FDTD photonics simulations. Both produce identical results on NVIDIA B200 GPUs.

.. contents:: On this page
   :local:
   :depth: 2

Local Workflow
--------------

The local workflow is the primary way to use Hyperwave. All CPU steps run locally on your machine or in Google Colab, giving you full control over intermediate arrays (theta, density, permittivity) at every stage. Only the GPU simulation requires an API call.

**Start here for:**

* Learning how the solver works
* Testing and validating simulations
* Custom structures and parameter sweeps
* Inverse design and optimization
* Any workflow where you want to inspect or modify intermediate data

**Example:**

.. code-block:: python

   import hyperwave_community as hwc
   import gdsfactory as gf

   from google.colab import userdata
   hwc.configure_api(api_key=userdata.get('HYPERWAVE_API_KEY'))

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

You can also use the API Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are integrating Hyperwave into an existing application, UI, or automated pipeline, the API workflow provides a single server-side call that handles structure creation for you. See :doc:`api_workflow` for details.

Workflow Comparison
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - **Local Workflow**
     - **API Workflow**
   * - Primary use
     - Learning, testing, custom work
     - Integration into existing systems
   * - CPU work runs on
     - Your machine / Colab
     - Modal (SPINs servers)
   * - Structure creation
     - Step-by-step local functions
     - Single API call
   * - Custom theta patterns
     - Full support
     - Not supported
   * - Intermediate inspection
     - Full access to all arrays
     - Limited
   * - Lines of code
     - ~50 lines
     - ~30 lines

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

   local_workflow
   api_workflow
   inverse_design
