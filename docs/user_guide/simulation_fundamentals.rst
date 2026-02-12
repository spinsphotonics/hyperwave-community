Simulation Fundamentals
=======================

This section covers running FDTD simulations via the Hyperwave cloud API,
extracting field data with monitors, and computing power flow.

Cloud Simulation Workflow
-------------------------

Hyperwave Community runs GPU-accelerated FDTD on remote servers. The typical
workflow is:

1. Build the structure and source locally
2. Submit to the cloud GPU
3. Receive field data at monitor locations
4. Analyze results locally

.. code-block:: python

   import hyperwave_community as hwc

   # 1. Configure API
   hwc.configure_api(api_key='your-key-here')

   # 2. Build recipe and monitors
   recipe = hwc.build_recipe(structure)
   monitors_recipe = hwc.build_monitors(monitors)

   # 3. Run simulation on cloud GPU
   results = hwc.simulate(
       structure=structure,
       source_field=source,
       source_offset=offset,
       freq_band=freq_band,
       monitors=monitors,
   )

Monitors
--------

Monitors define 3D volumes where field data is extracted during simulation.
Fields are only saved at monitor locations to minimize data transfer.

Creating Monitors
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Single monitor
   mon = hwc.Monitor(shape=(1, 50, 50), offset=(100, 0, 0))

   # Monitor set with named monitors
   monitors = hwc.MonitorSet()
   monitors.add(hwc.Monitor(shape=(1, 50, 50), offset=(100, 0, 0)), name='output')
   monitors.add(hwc.Monitor(shape=(1, 50, 50), offset=(20, 0, 0)), name='input')

Monitor Placement
~~~~~~~~~~~~~~~~~

Use ``add_monitors_at_position()`` for automatic monitor placement:

.. code-block:: python

   monitors = hwc.MonitorSet()
   monitors.add_monitors_at_position(
       structure,
       axis='x',
       position=100,
       label='Output',
   )

.. note::

   Monitor shapes must have positive integer dimensions. The offset can be
   negative if the monitor extends beyond the default origin.

Power Analysis
--------------

Poynting Vector
~~~~~~~~~~~~~~~

The Poynting vector gives the direction and magnitude of power flow.
Use ``S_from_slice()`` to compute it from a 2D field slice:

.. code-block:: python

   # Extract monitor field data
   field_slice = results['output']  # shape: (N_freq, 6, Ny, Nz)

   # Average over one spatial dimension if needed
   field_2d = jnp.mean(field_slice, axis=2)  # average over y

   # Compute Poynting vector
   S = hwc.S_from_slice(field_2d)  # shape: (N_freq, 3, Ny, Nz)

   # Sum power in propagation direction
   power = jnp.abs(jnp.sum(S[:, 0, :, :], axis=(1, 2)))

Transmission
~~~~~~~~~~~~

Use ``analyze_transmission()`` for quick transmission calculations:

.. code-block:: python

   transmission = hwc.analyze_transmission(results)

Field Visualization
-------------------

Extract and plot field intensity:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Electric field intensity
   E2 = hwc.get_electric_field_intensity(field)  # |E|^2

   # Plot XY cross-section
   plt.figure(figsize=(8, 6))
   plt.imshow(E2[0, :, :, nz//2].T, origin='lower', cmap='hot')
   plt.colorbar(label='|E|^2')
   plt.xlabel('x (grid units)')
   plt.ylabel('y (grid units)')
   plt.title('Electric field intensity')
   plt.show()

Available field analysis functions:

* ``get_field_intensity(field)``: Total intensity |E|^2 + |H|^2
* ``get_electric_field_intensity(field)``: Electric intensity |E|^2
* ``get_magnetic_field_intensity(field)``: Magnetic intensity |H|^2
* ``get_field_slice(field, axis, position)``: Extract 2D cross-section

Absorbing Boundaries
--------------------

Absorbing boundaries prevent reflections from the simulation domain edges.
Use the Bayesian-optimized parameters for best results:

.. code-block:: python

   # Get optimized absorber parameters for your resolution
   abs_params = hwc.get_optimized_absorber_params(
       resolution_nm=20.0,
       structure_dimensions=(nx, ny, nz),
   )

   # Create absorption mask
   absorption = hwc.create_absorption_mask(
       grid_shape=(nx, ny, nz),
       absorption_widths=abs_params['absorption_widths'],
       absorption_coeff=abs_params['absorber_coeff'],
       show_plots=False,
   )

.. warning::

   Insufficient absorber width causes reflections from domain boundaries,
   producing standing wave artifacts in the field data. If you see unexpected
   oscillations, try increasing the absorber width.
