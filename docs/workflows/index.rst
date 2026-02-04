Workflows
=========

Hyperwave Community offers two workflows for running FDTD photonics simulations. Both produce identical results - choose based on your needs.

.. contents:: On this page
   :local:
   :depth: 2

Workflow Comparison
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - **API Workflow**
     - **Local Workflow**
   * - Best for
     - Quick simulations, beginners
     - Custom structures, optimization
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

API Workflow (Recommended for Beginners)
----------------------------------------

The API workflow uses a single ``build_recipe()`` call to create structures from GDSFactory components. This is the simplest way to run simulations.

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

Local Workflow (For Advanced Users)
-----------------------------------

The local workflow separates structure creation into distinct steps: loading a component (or creating custom theta), applying density filtering, and building the recipe. This gives you full control over the process.

**Use this workflow when:**

* You need custom structures not available in GDSFactory
* You're doing inverse design / optimization
* You want to inspect or modify intermediate arrays (theta, density)
* You need to compare local vs API outputs

**Example:**

.. code-block:: python

   import hyperwave_community as hwc

   hwc.configure_api(api_key="your-key")

   # Step 1a: Load component to theta (or create custom theta)
   theta_result = hwc.load_component(
       component_name="mmi2x2_with_sbend",
       resolution_nm=20,
       show_plot=True,  # Inspect the pattern
   )

   # Step 1b: Build recipe from theta (local processing)
   recipe_result = hwc.build_recipe_from_theta(
       theta_result=theta_result,
       n_core=3.48,
       n_clad=1.4457,
       ...
   )

:doc:`local_workflow` - Full tutorial

Shared Steps
------------

Both workflows share the same steps after building the recipe:

.. code-block:: python

   # Step 2: Build monitors (free)
   monitor_result = hwc.build_monitors(...)

   # Step 3: Compute frequency band (free)
   freq_result = hwc.compute_freq_band(...)

   # Step 4: Solve mode source (free)
   source_result = hwc.solve_mode_source(...)

   # Step 5: Run simulation (uses credits)
   results = hwc.run_simulation(...)

   # Step 6: Analyze results (free, local)
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
     - Credits
   * - Analysis (Step 6)
     - Free

Credits are only consumed when running ``run_simulation()``. All other functions are free but require a valid API key.

.. toctree::
   :maxdepth: 1
   :hidden:

   api_workflow
   local_workflow
