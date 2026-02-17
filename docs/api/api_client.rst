API Client
==========

The API client module provides GPU-accelerated FDTD simulation via the Hyperwave cloud API.

.. automodule:: hyperwave_community.api_client
   :no-members:

Workflow Overview
-----------------

The SDK workflow consists of:

1. **CPU Steps (free)**: Build structure, monitors, frequency band, and mode source
2. **GPU Step (costs credits, $25/hr on B200)**: Run FDTD simulation
3. **Analysis (free, local)**: Analyze transmission and visualize fields

Configuration & Account
-----------------------

.. autofunction:: hyperwave_community.api_client.configure_api
.. autofunction:: hyperwave_community.api_client.get_account_info
.. autofunction:: hyperwave_community.api_client.estimate_cost

CPU Steps (Free)
----------------

These functions run on Modal CPU and do not consume credits.

.. autofunction:: hyperwave_community.api_client.build_recipe
.. autofunction:: hyperwave_community.api_client.build_monitors
.. autofunction:: hyperwave_community.api_client.compute_freq_band
.. autofunction:: hyperwave_community.api_client.solve_mode_source
.. autofunction:: hyperwave_community.api_client.get_default_absorber_params

GPU Step (Costs Credits, $25/hr)
---------------------------------

.. autofunction:: hyperwave_community.api_client.run_simulation

.. _convergence-configuration:

Convergence Configuration
-------------------------

Control early stopping behavior with convergence presets or custom configuration.

**Presets:**

- ``"quick"`` - Fast, stops early with fewer checks (2 stable checks at 2000 step intervals)
- ``"default"`` - Balanced approach (3 stable checks at 1000 step intervals)
- ``"thorough"`` - Conservative, more checks before stopping (5 stable checks, min 5000 steps)
- ``"full"`` - No early stopping, runs all steps

All presets use 1% relative threshold for convergence detection.

**Custom Configuration:**

For fine-grained control, create a custom ``ConvergenceConfig``:

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

.. autoclass:: hyperwave_community.api_client.ConvergenceConfig
   :members:
   :undoc-members:
   :no-index:


Analysis Functions (Free, Local)
--------------------------------

These functions run locally for analyzing simulation results.

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
