.. _convergence-configuration:

Convergence Options
===================

Control early stopping behavior to optimize simulation time.

Presets
-------

Use a preset string for common configurations:

.. list-table::
   :header-rows: 1

   * - Preset
     - Description
     - Check Interval
     - Stable Checks Required
   * - ``"quick"``
     - Fast, fewer stability checks
     - 2000 steps
     - 2
   * - ``"default"``
     - Balanced approach
     - 1000 steps
     - 3
   * - ``"thorough"``
     - Conservative, more checks
     - 1000 steps
     - 5 (min 5000 steps)
   * - ``"full"``
     - No early stopping
     - N/A
     - N/A

All presets use **1% relative threshold** for convergence detection.

Usage:

.. code-block:: python

   results = hwc.run_simulation(
       ...,
       convergence="default",  # or "quick", "thorough", "full"
   )

Custom Configuration
--------------------

For fine-grained control, use ``ConvergenceConfig``:

.. code-block:: python

   convergence = hwc.ConvergenceConfig(
       check_every_n=500,            # Steps between convergence checks (default: 1000)
       relative_threshold=0.005,     # Relative power change threshold (default: 0.01 = 1%)
       min_stable_checks=5,          # Consecutive stable checks required (default: 3)
       min_steps=3000,               # Minimum steps before checking (default: 0)
       power_threshold=1e-7,         # Ignore ports with power below this (default: 1e-6)
       monitors=["Output_o3"],       # Specific monitors to check (default: None = all outputs)
   )

   results = hwc.run_simulation(..., convergence=convergence)

Parameters
~~~~~~~~~~

``check_every_n``
   Number of FDTD steps between convergence checks. Lower values check more frequently but add overhead.

``relative_threshold``
   Maximum relative change in power between checks to be considered "stable". Default is 0.01 (1%).

``min_stable_checks``
   Number of consecutive stable checks required before stopping. Higher values are more conservative.

``min_steps``
   Minimum number of steps to run before checking convergence. Useful to ensure the simulation has reached steady state.

``power_threshold``
   Ignore monitors with power below this value when checking convergence. Helps avoid noise-dominated convergence decisions.

``monitors``
   List of specific monitor names to check. If None, all output monitors are checked.

How Convergence Works
---------------------

1. After ``min_steps``, the simulation checks power at output monitors every ``check_every_n`` steps
2. If the relative change in power is below ``relative_threshold`` for all monitored ports, that's a "stable" check
3. After ``min_stable_checks`` consecutive stable checks, the simulation stops early
4. If convergence is not reached, the simulation runs for the full ``num_steps``
