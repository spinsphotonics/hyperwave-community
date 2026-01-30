API Client
==========

The API client module provides GPU-accelerated FDTD simulation via the Hyperwave cloud API.

.. automodule:: hyperwave_community.api_client
   :no-members:

Workflows Overview
------------------

The SDK supports multiple workflow levels:

1. **Granular Workflow** (Recommended): Maximum control with step-by-step functions
2. **Two-Stage Workflow**: Separate setup and simulation phases
3. **One-Shot Workflow**: Quick single-function simulation

Configuration & Account
-----------------------

.. autofunction:: hyperwave_community.api_client.configure_api
.. autofunction:: hyperwave_community.api_client.get_account_info
.. autofunction:: hyperwave_community.api_client.estimate_cost

Granular Workflow (Recommended)
-------------------------------

The granular workflow gives you maximum control over the simulation process.
CPU steps are free (require valid API key), GPU steps consume credits.

**CPU Steps (Free):**

.. autofunction:: hyperwave_community.api_client.build_recipe
.. autofunction:: hyperwave_community.api_client.build_monitors
.. autofunction:: hyperwave_community.api_client.compute_freq_band
.. autofunction:: hyperwave_community.api_client.solve_mode_source
.. autofunction:: hyperwave_community.api_client.get_default_absorber_params

**GPU Step (Uses Credits):**

.. autofunction:: hyperwave_community.api_client.run_simulation

Two-Stage Workflow
------------------

The two-stage workflow separates setup (CPU) from simulation (GPU).

.. autofunction:: hyperwave_community.api_client.prepare_simulation

One-Shot Workflow
-----------------

For quick tests, use the one-shot workflow that combines everything.

.. autofunction:: hyperwave_community.api_client.simulate

Convergence Configuration
-------------------------

Control early stopping behavior with convergence presets or custom configuration.

**Presets:**

- ``"quick"`` - Fast, stops early with fewer checks (2 stable checks at 2000 step intervals)
- ``"default"`` - Balanced approach (3 stable checks at 1000 step intervals)
- ``"thorough"`` - Conservative, more checks before stopping (5 stable checks, min 5000 steps)
- ``"full"`` - No early stopping, runs all steps

All presets use 1% relative threshold for convergence detection.

.. autoclass:: hyperwave_community.api_client.ConvergenceConfig
   :members:
   :undoc-members:

.. autodata:: hyperwave_community.api_client.CONVERGENCE_PRESETS

Analysis Functions (Local)
--------------------------

These functions run locally and do not consume API credits.

.. autofunction:: hyperwave_community.api_client.analyze_transmission
.. autofunction:: hyperwave_community.api_client.get_field_intensity_2d
.. autofunction:: hyperwave_community.api_client.compute_poynting_vector
.. autofunction:: hyperwave_community.api_client.compute_monitor_power

Visualization Functions
-----------------------

.. autofunction:: hyperwave_community.api_client.visualize_structure
.. autofunction:: hyperwave_community.api_client.visualize_mode_source

Utility Functions
-----------------

.. autofunction:: hyperwave_community.api_client.encode_array
.. autofunction:: hyperwave_community.api_client.decode_array

Advanced Granular Workflow
--------------------------

For fine-grained control over individual API calls:

.. autofunction:: hyperwave_community.api_client.load_component
.. autofunction:: hyperwave_community.api_client.create_structure_recipe
.. autofunction:: hyperwave_community.api_client.create_monitors
.. autofunction:: hyperwave_community.api_client.solve_mode
.. autofunction:: hyperwave_community.api_client.run_gpu_simulation
.. autofunction:: hyperwave_community.api_client.get_field_slice
