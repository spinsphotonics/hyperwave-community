Structure Modeling
==================

This section covers how to define photonic device structures using layers,
density filtering, and permittivity construction.

Layer-Based Design
------------------

Photonic devices in Hyperwave Community are built from stacked ``Layer`` objects.
Each layer defines a 2D density pattern and material properties.

.. code-block:: python

   import hyperwave_community as hwc
   import jax.numpy as jnp

   # Create density patterns
   nx, ny = 200, 200
   ones = jnp.ones((nx, ny))
   zero = jnp.zeros((nx, ny))

   # Define layers (bottom to top)
   layers = [
       hwc.Layer(zero, (1.44**2, 3.48**2), 10),   # BOX: SiO2 to Si interpolation
       hwc.Layer(ones, (1.44**2, 3.48**2), 8),     # Device: solid silicon
       hwc.Layer(zero, 1.44**2, 20),                # Cladding: uniform SiO2
   ]

Layer Parameters
~~~~~~~~~~~~~~~~

Each ``Layer`` takes four arguments:

1. **density_pattern** (2D array): Values in [0, 1] controlling material interpolation.
   Shape must have even dimensions.
2. **permittivity_values**: Either a ``(low, high)`` tuple for interpolation, or a single
   float for uniform permittivity.
3. **layer_thickness** (int or float): Thickness in grid units. Float values enable
   subpixel averaging at interfaces.
4. **conductivity_values** (optional): Same format as permittivity. Defaults to 0.

.. note::

   When ``permittivity_values`` is a tuple ``(low, high)``, the final permittivity at
   each pixel is: ``eps = low + density * (high - low)``.

Density Filtering
-----------------

The ``density()`` function converts raw optimization variables into filtered density
fields with minimum feature size constraints.

.. code-block:: python

   # Raw optimization variables
   theta = jnp.ones((400, 400)) * 0.5

   # Apply density filtering
   filtered = hwc.density(theta, radius=8, alpha=4.0)

Parameters
~~~~~~~~~~

* **theta**: Raw optimization variable array (2D).
* **radius** (float): Filter radius controlling minimum feature size. Larger radius
  produces smoother features. Default: 8.
* **alpha** (float): Projection strength controlling binarization. 0 = no projection
  (smooth), higher values push toward binary. Default: 0.
* **pad_width** (int or tuple): Padding applied before filtering.
* **c**, **eta**, **eta_lo**, **eta_hi**: Advanced projection parameters.

.. warning::

   The output array may be slightly smaller than the input due to the filtering
   kernel. Always check the output shape and ensure dimensions remain even.

Creating Structures
-------------------

The ``create_structure()`` function assembles layers into a 3D permittivity
distribution.

.. code-block:: python

   structure = hwc.create_structure(layers=layers, vertical_radius=5.0)

   # Access the arrays
   print(f"Permittivity shape: {structure.permittivity.shape}")
   print(f"Conductivity shape: {structure.conductivity.shape}")

The returned ``Structure`` object contains:

* **permittivity**: Shape ``(3, nx, ny, nz)`` for the three field components
* **conductivity**: Same shape as permittivity
* **recipe**: Serializable metadata for reconstructing the structure

Vertical Blur
~~~~~~~~~~~~~

The ``vertical_radius`` parameter controls smoothing between layers in the
z-direction. A value of 0 creates sharp interfaces; larger values create
gradual transitions.

Visualizing Structures
----------------------

Use ``view_structure()`` to inspect the 3D permittivity distribution:

.. code-block:: python

   hwc.view_structure(structure)

For more control, extract cross-sections manually:

.. code-block:: python

   import matplotlib.pyplot as plt

   eps = structure.permittivity
   nx, ny, nz = eps.shape[1:]

   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   # XY cross-section at device layer
   axes[0].imshow(eps[0, :, :, nz//2].T, origin='lower')
   axes[0].set_title('XY (z=center)')
   axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

   # XZ cross-section
   axes[1].imshow(eps[0, :, ny//2, :].T, origin='lower')
   axes[1].set_title('XZ (y=center)')
   axes[1].set_xlabel('x'); axes[1].set_ylabel('z')

   # YZ cross-section
   axes[2].imshow(eps[0, nx//2, :, :].T, origin='lower')
   axes[2].set_title('YZ (x=center)')
   axes[2].set_xlabel('y'); axes[2].set_ylabel('z')

   plt.tight_layout()
   plt.show()

GDS Export
----------

.. warning::

   GDS export should only be used after the design has been fully binarized
   and fabrication constraints have been applied. Exporting a grayscale
   (non-binary) design to GDS will produce a geometry that does not match
   the simulated performance. Binarization penalties and fabrication-rule
   enforcement are still in development.

Once a design is binary, export to GDSII format for fabrication:

.. code-block:: python

   # Convert binary density to GDS
   hwc.generate_gds_from_density(
       density_array=final_density,
       level=0.5,
       output_filename='device.gds',
       resolution=0.020,  # 20nm grid
   )

   # Visualize the GDS
   hwc.view_gds('device.gds')

   # Import GDS back to density array
   theta_imported, info = hwc.gds_to_theta('device.gds', resolution=0.020)
