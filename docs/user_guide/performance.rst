Performance
===========

This section covers optimization of simulation speed, accuracy, and cost.

GPU Selection
-------------

Hyperwave Community supports multiple GPU tiers:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - GPU
     - Memory
     - Speed
     - Cost
   * - B200
     - 192 GB
     - Fastest
     - Lowest per FLOP
   * - H200
     - 141 GB
     - Fast
     - Medium
   * - H100
     - 80 GB
     - Fast
     - Medium
   * - A100
     - 80 GB
     - Moderate
     - Higher per FLOP

.. note::

   Use ``gpu_type="B200"`` for best performance and cost efficiency.
   Only fall back to other GPUs if B200 is unavailable.

Grid Resolution
---------------

Grid resolution directly affects accuracy and simulation cost:

* **Coarser grids** (30-40nm): Fast, good for initial exploration
* **Medium grids** (20-25nm): Balanced accuracy and speed
* **Fine grids** (10-15nm): High accuracy, significantly more expensive

Rule of thumb: use at least 10-20 grid points per wavelength in the highest
refractive index material.

.. code-block:: python

   # Example: calculate minimum resolution
   wavelength = 1.55  # um
   n_max = 3.48       # silicon
   lambda_min = wavelength / n_max  # ~0.445 um

   # 20 points per wavelength
   dx = lambda_min / 20  # ~22nm

Convergence Settings
--------------------

The FDTD simulation runs until fields converge or a maximum step count is reached.
Use ``ConvergenceConfig`` to control this:

.. code-block:: python

   # Use a preset
   config = hwc.CONVERGENCE_PRESETS['default']

   # Or customize
   config = hwc.ConvergenceConfig(
       max_steps=10000,
       check_every_n=100,
       tolerance=1e-6,
   )

Available presets:

* **fast**: Fewer steps, looser tolerance. Good for quick checks.
* **default**: Balanced settings for most simulations.
* **accurate**: More steps, tighter tolerance. Use for final results.

Cost Estimation
---------------

Estimate simulation cost before running:

.. code-block:: python

   cost = hwc.estimate_cost(
       structure=structure,
       gpu_type='B200',
       max_steps=10000,
   )
   print(f"Estimated cost: {cost['credits']} credits")

Tips for Reducing Cost
~~~~~~~~~~~~~~~~~~~~~~

1. **Start with coarse grids** to validate setup before running at full resolution.
2. **Use convergence presets** to avoid running more steps than necessary.
3. **Minimize monitor volume** -- only extract fields where needed.
4. **Use B200 GPUs** for best cost per FLOP.

Inverse Design Optimization
----------------------------

.. note::

   The inverse design workflow is currently a preview. Binarization schedules
   and fabrication constraint enforcement are in development. Optimized designs
   will contain grayscale density values and are not directly fabrication-ready.

For adjoint-based inverse design, each optimization step requires two simulations
(forward and adjoint). ``run_optimization()`` runs the full loop on a cloud GPU
and returns results after each completed step.

Tips for efficient optimization:

* Start with ``density_alpha=0`` (freeform) and increase later for binarization.
* Use ``learning_rate=0.01`` as a starting point.
* Run 50-100 steps initially to assess convergence behavior.

.. code-block:: python

   # Run optimization -- one GPU call, results per step
   results = []
   try:
       for step_result in hwc.run_optimization(
           theta=theta_init,
           source_field=source,
           source_offset=source_offset,
           freq_band=freq_band,
           structure_spec=structure_spec,
           loss_monitor_shape=loss_monitor_shape,
           loss_monitor_offset=loss_monitor_offset,
           design_monitor_shape=design_monitor_shape,
           design_monitor_offset=design_monitor_offset,
           mode_field=mode_full,
           input_power=input_power,
           mode_cross_power=P_mode_cross,
           num_steps=50,
           learning_rate=0.01,
           gpu_type="B200",
       ):
           results.append(step_result)
           eff = step_result['efficiency'] * 100
           print(f"Step {step_result['step']}: {eff:.2f}%")
   except KeyboardInterrupt:
       print(f"Stopped after {len(results)} steps.")

Cancellation and Billing
~~~~~~~~~~~~~~~~~~~~~~~~

Interrupting the kernel (or breaking out of the loop) closes the connection,
which cancels the GPU task. You are only charged for steps that completed
before the interruption.

For single-step gradient access (advanced), use ``compute_adjoint_gradient()``
which runs one forward + adjoint pair and returns the raw gradient.
