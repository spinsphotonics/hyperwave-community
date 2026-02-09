.. _gpu-options:

GPU Options
===========

All simulations run exclusively on NVIDIA B200 GPUs.

.. note::

   Effective February 2026, all simulations run exclusively on NVIDIA B200 GPUs. Previous GPU
   tiers have been consolidated into a single, streamlined pricing tier at $25 per compute hour.
   The per-hour cost for B200 simulations remains unchanged from prior pricing.

.. _gpu-performance:

Performance Reference
---------------------

.. list-table::
   :header-rows: 1

   * - GPU
     - Speed (Gcell/s)
     - Max Footprint (um)
     - Cost / hr
   * - ``"B200"``
     - 25
     - 36 x 36
     - $25.00

| \* Max footprint assumes 20 nm mesh and 4.2 um stack height.
| \* 1 credit = $25 = 1 hour of compute.
| \* See :ref:`gpu-specs` for full hardware specifications.

.. _gpu-cost-estimation:

Cost Estimation
---------------

:func:`~hyperwave_community.estimate_cost` previews time and cost before
running. Free, no authentication required.

.. code-block:: python

   dims = recipe_result['dimensions']  # e.g. [1800, 350, 199]
   grid_points = dims[0] * dims[1] * dims[2]

   cost = hwc.estimate_cost(
       grid_points=grid_points,
       max_steps=20000,
   )
   print(f"Estimated time: {cost['estimated_seconds']:.0f}s")
   print(f"Estimated cost: ${cost['estimated_cost_usd']:.2f}")

Returns:

- ``estimated_seconds``: estimated simulation wall time
- ``estimated_credits``: estimated credits (1 credit = $25 = 1 hour)
- ``estimated_cost_usd``: estimated cost in USD
- ``grid_points``: total grid points in the simulation

.. note::

   Actual cost is often lower because simulations typically converge before
   reaching ``max_steps``.

.. _gpu-calculator:

Cost Calculator
---------------

Preview estimated time and cost for your simulation dimensions.

.. raw:: html

   <form id="gpu-calc-form">
     <div class="gpu-calc-inputs">
       <div class="gpu-calc-field">
         <label for="calc-x">X (um)</label>
         <input type="number" id="calc-x" value="10" min="0.1" step="any">
       </div>
       <div class="gpu-calc-field">
         <label for="calc-y">Y (um)</label>
         <input type="number" id="calc-y" value="10" min="0.1" step="any">
       </div>
       <div class="gpu-calc-field">
         <label for="calc-z">Z (um)</label>
         <input type="number" id="calc-z" value="4.2" min="0.1" step="any">
       </div>
       <div class="gpu-calc-field">
         <label for="calc-res">Resolution (nm)</label>
         <input type="number" id="calc-res" value="20" min="1" step="1">
       </div>
       <div class="gpu-calc-field">
         <label for="calc-steps">Max Steps</label>
         <input type="number" id="calc-steps" value="10000" min="100" step="100">
       </div>
     </div>
   </form>
   <div id="gpu-calc-results"></div>

----

.. _gpu-specs:

Appendix: GPU Specifications
----------------------------

Full hardware and billing reference. Back to :ref:`gpu-performance`.

.. list-table::
   :header-rows: 1

   * - GPU
     - VRAM (GB)
     - Speed (Gcell/s)
     - Max Cells (M)
     - Cost / hr
   * - ``"B200"``
     - 192
     - 25
     - 700
     - $25.00

| \* Max Cells assumes 20 nm mesh resolution and 4.2 um stack height (210 vertical cells).
| \* Non-square footprints are possible. The constraint is total cell count, not aspect ratio.
| \* 1 credit = $25 = 1 hour of compute.
