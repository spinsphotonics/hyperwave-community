Running Simulations
===================

Functions for configuring the API, estimating costs, and running GPU-accelerated FDTD
simulations via the Hyperwave cloud.

The SDK provides two simulation interfaces:

- **High-level** (:func:`~hyperwave_community.api_client.simulate`): Pass a structure
  and get results back in one call. Handles all intermediate steps automatically.
- **Step-by-step**: Build the recipe, monitors, and source separately using the CPU-step
  functions, then call :func:`~hyperwave_community.api_client.run_simulation`.

Configuration and Account
-------------------------

.. autofunction:: hyperwave_community.api_client.configure_api

.. autofunction:: hyperwave_community.api_client.get_account_info

.. autofunction:: hyperwave_community.api_client.estimate_cost

High-Level Simulation
---------------------

.. autofunction:: hyperwave_community.api_client.simulate

GPU Step
--------

.. autofunction:: hyperwave_community.api_client.run_simulation

CPU Steps (Free)
----------------

These functions run on cloud CPUs and do not consume credits.

.. autofunction:: hyperwave_community.api_client.build_recipe

.. autofunction:: hyperwave_community.api_client.build_monitors

.. autofunction:: hyperwave_community.api_client.compute_freq_band

.. autofunction:: hyperwave_community.api_client.solve_mode_source

.. autofunction:: hyperwave_community.api_client.get_default_absorber_params

.. _convergence-configuration:

Convergence Configuration
-------------------------

Control early-stopping behavior with convergence presets or a custom configuration.

**Presets:**

- ``"quick"`` -- Fast, stops early with fewer checks (2 stable checks at 2000-step intervals)
- ``"default"`` -- Balanced approach (3 stable checks at 1000-step intervals)
- ``"thorough"`` -- Conservative, more checks before stopping (5 stable checks, min 5000 steps)
- ``"full"`` -- No early stopping, runs all steps

All presets use a 1% relative threshold for convergence detection.

**Custom configuration:**

For fine-grained control, create a :class:`~hyperwave_community.api_client.ConvergenceConfig`:

.. code-block:: python

   convergence = hwc.ConvergenceConfig(
       check_every_n=500,
       relative_threshold=0.005,
       min_stable_checks=5,
       min_steps=3000,
       power_threshold=1e-7,
       monitors=["Output_o3"],
   )
   results = hwc.run_simulation(..., convergence=convergence)

See also: :doc:`/convergence` for a detailed guide on convergence tuning.

.. autoclass:: hyperwave_community.api_client.ConvergenceConfig
   :members:
   :undoc-members:
   :no-index:

Inverse Design
--------------

.. autofunction:: hyperwave_community.api_client.compute_adjoint_gradient

.. autofunction:: hyperwave_community.api_client.run_optimization

Component Library
-----------------

Browse and preview built-in photonic components.

.. autofunction:: hyperwave_community.api_client.list_components

.. autofunction:: hyperwave_community.api_client.get_component_params

.. autofunction:: hyperwave_community.api_client.preview_component
