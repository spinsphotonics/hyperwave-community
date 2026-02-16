Workflows
=========

Hyperwave Community offers three workflows for running FDTD photonics simulations. All produce identical results on NVIDIA B200 GPUs.

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

Inverse Design Workflow
-----------------------

Adjoint-method gradient-based optimization on cloud GPUs. The optimizer iteratively
updates a 2D design pattern (theta) by running forward and adjoint FDTD simulations
to compute gradients of a loss function with respect to the design variables. You
define a design region, choose a loss function (mode coupling, Poynting power, or
field intensity), and the optimizer updates the design pattern to maximize your
objective.

Builds on the local workflow: you set up the same layer stack and mode solve locally,
then hand the structure specification to ``run_optimization()`` which handles the
forward/adjoint loop on cloud GPU.

**Example:**

.. code-block:: python

   import hyperwave_community as hwc

   from google.colab import userdata
   hwc.configure_api(api_key=userdata.get('HYPERWAVE_API_KEY'))

   # Run optimization loop on cloud GPU
   for step_result in hwc.run_optimization(
       theta=theta_init,
       source_field=source_field,
       source_offset=source_offset,
       freq_band=freq_band,
       structure_spec=structure_spec,
       mode_field=mode_field,        # built-in mode coupling loss
       input_power=input_power,
       mode_cross_power=P_mode_cross,
       num_steps=50,
       learning_rate=0.1,
       ...
   ):
       print(f"Step {step_result['step']}: efficiency = {abs(step_result['loss']) * 100:.2f}%")

:doc:`inverse_design` - Full tutorial

API Workflow
------------

If you are integrating Hyperwave into an existing application, UI, or automated
pipeline, the API workflow provides a single server-side call that handles structure
creation for you. All CPU-side steps (theta conversion, density filtering, layer
stacking) run on the server, so you only need a GDS component and an API key.

:doc:`api_workflow` - Full tutorial

Workflow Comparison
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * -
     - **Local Workflow**
     - **API Workflow**
     - **Inverse Design**
   * - Primary use
     - Learning, testing, custom work
     - Integration into existing systems
     - Topology optimization
   * - CPU work runs on
     - Your machine / Colab
     - Modal (SPINs servers)
     - Your machine + cloud GPU
   * - Structure creation
     - Step-by-step local functions
     - Single API call
     - Step-by-step with theta optimization
   * - Custom theta patterns
     - Full support
     - Not supported
     - Required (design variable)
   * - Intermediate inspection
     - Full access to all arrays
     - Limited
     - Full access to all arrays
   * - Optimization
     - Manual only
     - Not supported
     - Adjoint-method gradient-based
   * - Lines of code
     - ~50 lines
     - ~30 lines
     - ~80 lines

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

All workflows share the same cost structure:

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
   * - Optimization (inverse design only)
     - Credits, 2 sims per step ($25/hr)

Credits are only consumed when running ``run_simulation()`` or ``run_optimization()``. All simulations run on NVIDIA B200 GPUs at $25 per compute hour (1 credit = $25 = 1 hour). All other functions are free but require a valid API key.

.. toctree::
   :maxdepth: 1

   local_workflow
   inverse_design
   api_workflow
