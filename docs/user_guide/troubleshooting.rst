Troubleshooting
===============

This section covers common issues and solutions when using Hyperwave Community.

API Connection Issues
---------------------

**Authentication Error**

.. code-block:: text

   Error: Invalid API key

* Verify your API key is correct: ``hwc.configure_api(api_key='...')``
* Check your account status: ``hwc.get_account_info()``
* Ensure your key has not expired

**Timeout Error**

.. code-block:: text

   Error: Request timed out

* Check your internet connection
* Large simulations may take several minutes to start
* The cloud GPU pool may be temporarily at capacity; retry after a few minutes

Shape Mismatch Errors
---------------------

**Layer dimensions must be even**

.. code-block:: text

   ValueError: dimension 201 is not even

All density patterns must have even dimensions. Use ``density()`` to auto-trim:

.. code-block:: python

   # This may produce odd dimensions
   theta = jnp.ones((201, 201))

   # density() auto-trims to even
   filtered = hwc.density(theta, radius=8)  # shape will be even

**Source and structure shape mismatch**

The source field must be compatible with the structure dimensions. If using
``create_mode_source()``, the source is automatically sized. For manual sources,
ensure the transverse dimensions match.

Convergence Issues
------------------

**Simulation does not converge**

If the field error does not decrease:

1. **Increase max_steps**: The simulation may need more time steps.
2. **Check absorber parameters**: Insufficient absorption causes reflections.
3. **Verify source placement**: Sources inside absorbers are attenuated.
4. **Check for resonances**: High-Q resonators need many more steps.

.. code-block:: python

   # Increase simulation time
   results = hwc.simulate(
       ...,
       convergence_config=hwc.ConvergenceConfig(max_steps=20000),
   )

**Fields grow exponentially**

This typically indicates a numerical instability:

* Grid resolution may be too coarse for the materials used
* Check that permittivity values are physical (positive, reasonable magnitude)

Monitor Issues
--------------

**Empty or zero field data**

* Verify monitor is inside the simulation domain
* Check that monitor does not overlap with absorbing boundaries
* Ensure the simulation ran enough steps for fields to reach the monitor

**Unexpected field patterns**

* Plot field cross-sections in all three planes (XY, XZ, YZ) to identify issues
* Check source polarization and propagation direction
* Verify the structure permittivity looks correct

Density and Structure Issues
----------------------------

**Density values outside [0, 1]**

The density filter may produce values slightly outside [0, 1]. This is normal
for small excursions. Large deviations indicate a problem:

.. code-block:: python

   # Check density range
   d = hwc.density(theta, radius=8)
   print(f"Density range: [{float(d.min()):.4f}, {float(d.max()):.4f}]")

**Structure looks wrong in cross-section**

* Verify layer order (layers are stacked bottom-to-top in z)
* Check layer thicknesses are in grid units, not physical units
* Use ``view_structure()`` to inspect all three cross-sections

Inverse Design Issues
---------------------

**Gradient is zero or NaN**

* Check that monitors are placed correctly
* Verify the objective function is differentiable
* Ensure the source reaches the device region

**Optimization makes no progress**

* Learning rate may be too small (try increasing by 2-5x)
* Gradient may be dominated by noise (try averaging over multiple frequencies)
* Check that the design region is correctly defined in the theta array

**Efficiency exceeds 100%**

* Incorrect power normalization (P_mode_cross)
* Wrong input power calculation
* Verify mode computation gives reasonable n_eff (typically 2.0-3.0 for silicon)

Getting Help
------------

If you encounter issues not covered here:

1. Check the `API Reference <../api.html>`_ for function signatures and parameters
2. Review the example notebooks in the ``examples/`` directory
3. Report issues at https://github.com/spinsphotonics/hyperwave-community/issues

Include the following when reporting:

* Python version and package version (``hwc.__version__``)
* Complete error traceback
* Minimal code example that reproduces the issue
* Structure dimensions and simulation parameters
