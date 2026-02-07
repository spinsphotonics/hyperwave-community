.. _gpu-options:

GPU Options
===========

Hyperwave runs FDTD simulations on cloud GPUs. **B200** is the recommended
default -- fastest, largest capacity, and lowest cost per simulation.

.. _gpu-usage:

Usage
-----

Pass ``gpu_type`` to :func:`~hyperwave_community.run_simulation`:

.. code-block:: python

   results = hwc.run_simulation(
       ...,
       gpu_type="B200",   # recommended default
   )

See :ref:`gpu-available` for all supported GPUs, or :ref:`gpu-cost-estimation`
to preview cost before running.

.. _gpu-available:

Available GPUs
--------------

Click any column header to sort.

| \* Max footprint assumes 20 nm mesh and 4.2 um stack height.
| \* Credits (10 x 10 um) = estimated cost for a 10 x 10 um device at 10,000 steps.
| \* See :ref:`gpu-specs` for full hardware specifications.

.. list-table::
   :header-rows: 1
   :class: sortable

   * - GPU
     - Speed (Gcell/s)
     - Max Footprint (um)
     - Credits / hr
     - Credits (10 x 10 um)
   * - ``"B200"``
     - 25
     - 36 x 36
     - 2.50
     - 0.015
   * - ``"H200"``
     - 16
     - 31 x 31
     - 2.00
     - 0.018
   * - ``"H100"``
     - 13
     - 23 x 23
     - 1.50
     - 0.017
   * - ``"A100-80GB"``
     - 9
     - 23 x 23
     - 1.00
     - 0.016
   * - ``"A100-40GB"``
     - 7
     - 16 x 16
     - 0.80
     - 0.017
   * - ``"L40S"``
     - 5
     - 18 x 18
     - 0.70
     - 0.020
   * - ``"A10G"``
     - 3
     - 12 x 12
     - 0.40
     - 0.020
   * - ``"T4"``
     - 2
     - 10 x 10
     - 0.30
     - 0.022

- **B200** -- Recommended default. Fastest, largest capacity, lowest cost per simulation.
- **H200** -- Good alternative when B200 availability is limited.
- **H100, A100-80GB, A100-40GB, L40S, A10G, T4** -- Architecture benchmarking.

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
       gpu_type="B200",
   )
   print(f"Estimated time: {cost['estimated_seconds']:.0f}s")
   print(f"Estimated cost: ${cost['estimated_cost_usd']:.2f}")

Returns:

- ``estimated_seconds`` -- estimated simulation wall time
- ``estimated_credits`` -- estimated credit cost
- ``estimated_cost_usd`` -- estimated cost in USD
- ``gpu_type`` -- the GPU type used for the estimate
- ``grid_points`` -- total grid points in the simulation

.. note::

   Actual cost is often lower because simulations typically converge before
   reaching ``max_steps``.

.. _gpu-calculator:

Cost Calculator
---------------

Compare estimated time and cost across all GPUs. Rows marked **OOM** exceed
that GPU's VRAM capacity.

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

Full hardware and billing reference. Back to :ref:`gpu-available`.

.. list-table::
   :header-rows: 1

   * - GPU
     - VRAM (GB)
     - Speed (Gcell/s)
     - Max Cells (M)
   * - ``"B200"``
     - 192
     - 25
     - 700
   * - ``"H200"``
     - 141
     - 16
     - 510
   * - ``"H100"``
     - 80
     - 13
     - 290
   * - ``"A100-80GB"``
     - 80
     - 9
     - 290
   * - ``"A100-40GB"``
     - 40
     - 7
     - 145
   * - ``"L40S"``
     - 48
     - 5
     - 175
   * - ``"A10G"``
     - 24
     - 3
     - 85
   * - ``"T4"``
     - 16
     - 2
     - 58

| \* Max Cells assumes 20 nm mesh resolution and 4.2 um stack height (210 vertical cells).
| \* Non-square footprints are possible -- the constraint is total cell count, not aspect ratio.
