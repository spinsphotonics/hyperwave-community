.. _gpu-options:

GPU Options
===========

Select GPU hardware for your FDTD simulations based on performance and cost needs.

Available GPUs
--------------

.. list-table::
   :header-rows: 1

   * - GPU Type
     - VRAM
     - Performance
     - Best For
   * - ``"B200"``
     - 192 GB
     - Highest
     - Large structures, production runs
   * - ``"H200"``
     - 141 GB
     - Very High
     - Large structures, complex simulations
   * - ``"H100"``
     - 80 GB
     - High
     - Most simulations (recommended default)
   * - ``"A100-80GB"``
     - 80 GB
     - Medium-High
     - Standard simulations
   * - ``"A100"``
     - 40 GB
     - Medium
     - Smaller structures, testing

Usage
-----

.. code-block:: python

   results = hwc.run_simulation(
       ...,
       gpu_type="H100",  # or "B200", "H200", "A100-80GB", etc.
   )

Choosing a GPU
--------------

**For development and testing:**

Use ``"A100"`` or ``"A100-80GB"`` for lower cost while iterating on designs.

**For production runs:**

Use ``"H100"`` (default) for a good balance of performance and availability.

**For large structures:**

Use ``"H200"`` or ``"B200"`` when simulating large photonic structures that require more VRAM.

Cost Estimation
---------------

Use ``estimate_cost()`` to get the estimated credit cost before running:

.. code-block:: python

   cost = hwc.estimate_cost(
       num_steps=20000,
       gpu_type="H100",
       dimensions=recipe_result['dimensions'],
   )
   print(f"Estimated cost: {cost['credits']} credits")
