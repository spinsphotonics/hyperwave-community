Mode Solving
============

This section covers waveguide mode computation and source injection for
photonic simulations.

Waveguide Modes
---------------

Waveguide modes are eigensolutions of the wave equation for a given cross-section.
Each mode has a propagation constant (beta) and field profile that describes how
light travels along the waveguide.

Computing Modes with ``create_mode_source``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``create_mode_source()`` function computes waveguide eigenmodes and returns
a source field ready for injection into an FDTD simulation.

.. code-block:: python

   import hyperwave_community as hwc
   import jax.numpy as jnp

   # Define frequency band
   wavelength = 1.55  # um
   freq = 2 * jnp.pi / wavelength
   freq_band = (freq, freq, 1)  # single frequency

   # Compute mode source
   source, offset, mode_info = hwc.create_mode_source(
       structure,
       freq_band,
       mode_num=0,              # fundamental mode
       propagation_axis='x',    # propagation direction
       source_position=80,      # position along propagation axis
   )

The function returns:

1. **source**: Field array with shape ``(N_freq, 6, sx, sy, sz)``
2. **offset**: Tuple ``(x, y, z)`` specifying where the source is placed
3. **mode_info**: Dictionary with mode computation details

.. note::

   The returned source has E-field components but H-fields set to zero.
   For simulations requiring proper H-fields, use the ``mode_converter``.

Mode Converter
--------------

The ``mode_converter()`` converts an E-only mode field into a full E+H field
by propagating through a short waveguide segment.

.. code-block:: python

   from hyperwave_community.sources import mode_converter

   # Convert E-only to full E+H
   full_mode = mode_converter(
       mode_E_field=mode_info['mode_field'],
       freq_band=freq_band,
       permittivity_slice=eps_slice,
       propagation_axis='x',
       visualize=False,
   )

   # full_mode shape: (N_freq, 6, 1, ny, nz)

This is useful for:

* Computing accurate mode overlap integrals
* Calculating coupling efficiency between a simulated field and a waveguide mode
* Verifying that the mode solver produced correct field profiles

Gaussian Sources
----------------

For free-space excitation (e.g., fiber coupling to grating couplers), use
``create_gaussian_source()``:

.. code-block:: python

   source = hwc.create_gaussian_source(
       frequency=freq,
       beam_waist=5.2,        # beam waist in grid units
       polarization='ey',     # E-field polarization
       grid_shape=(nx, ny),   # transverse dimensions
       center=(nx//2, ny//2), # beam center
   )

Source Placement
----------------

Source fields are injected at a specific position in the simulation domain.
The ``offset`` tuple controls placement:

.. code-block:: python

   # Source placed at x=80, spanning full y and z
   source_offset = (80, 0, 0)

.. warning::

   Sources must be placed inside the simulation domain but outside the
   absorbing boundary region. Placing sources inside absorbers will cause
   them to be attenuated before reaching the device.

Power Normalization
-------------------

When computing coupling efficiency, normalize against the input power.
The mode cross-power integral ``P_mode_cross`` quantifies the power carried
by a mode:

.. code-block:: python

   import numpy as np

   # Extract E and H from mode field
   E = full_mode[0, 0:3, 0, :, :]  # (3, ny, nz)
   H = full_mode[0, 3:6, 0, :, :]  # (3, ny, nz)

   # Cross product surface integral
   Sx = np.real(E[1] * np.conj(H[2]) - E[2] * np.conj(H[1]))
   P_mode_cross = float(np.abs(np.sum(Sx)))

   print(f"P_mode_cross = {P_mode_cross:.4f}")
