.. _inverse-design-workflow:

Inverse Design Workflow
=======================

Gradient-based topology optimization of a silicon photonic grating coupler using the adjoint
method on cloud GPUs. Each optimization step runs a forward and adjoint FDTD simulation to
compute the gradient of a loss function with respect to the 2D design pattern (theta). The
optimizer (Adam with cosine learning rate decay) updates theta to maximize mode coupling
efficiency from a fiber into a waveguide.

Lightweight operations (parameter setup, layer stacking, mode solving, monitor placement) run
locally or in Colab and are free. Cloud GPU steps (source generation, forward simulation,
optimization) consume credits at $25/hr on NVIDIA B200.

**Open in Google Colab**: `gc_inverse_design.ipynb <https://colab.research.google.com/github/spinsphotonics/hyperwave-community/blob/main/examples/gc_inverse_design.ipynb>`_

`Download the notebook <https://github.com/spinsphotonics/hyperwave-community/blob/main/examples/gc_inverse_design.ipynb>`_

.. contents:: Steps
   :local:
   :depth: 1

Prerequisites
-------------

Install the package with the GDS extra (needed for GDSFactory and JAX dependencies) and
configure your API key:

.. code-block:: python

   %pip install "hyperwave-community[gds] @ git+https://github.com/spinsphotonics/hyperwave-community.git" -q

The ``[gds]`` extra installs JAX, GDSFactory, and their dependencies. JAX is used for array
operations that support automatic differentiation through the density-to-structure pipeline.
GDSFactory is optional for inverse design (you build theta manually), but importing the full
package ensures all utilities are available.

.. code-block:: python

   import hyperwave_community as hwc
   import numpy as np
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import time

   from google.colab import userdata
   hwc.configure_api(api_key=userdata.get('HYPERWAVE_API_KEY'))
   hwc.get_account_info()

Step 1: Physical Parameters
---------------------------

Define the silicon-on-insulator (SOI) platform parameters for a partial-etch grating coupler
at 1550 nm wavelength. These parameters control the physical materials, device geometry, and
simulation grid resolution.

Materials and Wavelength
^^^^^^^^^^^^^^^^^^^^^^^^

The material refractive indices define the permittivity distribution of the simulation domain.
Standard 220 nm SOI uses silicon (n=3.48) as the core and silicon dioxide (n=1.44) as the
cladding and buried oxide. The target wavelength of 1.55 um corresponds to the telecom C-band
center.

- ``n_si`` (3.48): Silicon refractive index at 1550 nm. Used for the device layer, slab, and
  substrate.
- ``n_sio2`` (1.44): Silicon dioxide refractive index. Used for the buried oxide (BOX) and
  cladding layers.
- ``n_clad`` (1.44): Cladding refractive index, set equal to SiO2 for an oxide-clad grating.
  Change to 1.0 for an air-clad design.
- ``n_air`` (1.0): Air refractive index. Used for the region above the cladding and the top
  absorber padding.

Layer Thicknesses
^^^^^^^^^^^^^^^^^

The vertical layer stack defines the SOI cross-section. Each thickness is specified in microns:

- ``h_dev`` (0.220 um): Total silicon device layer thickness. This is the standard 220 nm SOI
  foundry thickness. The top portion is partially etched to form the grating; the bottom portion
  remains as a continuous slab.
- ``etch_depth`` (0.110 um): Partial etch depth. The optimizer controls a 2D design pattern
  that determines where the top 110 nm of silicon is etched away, exposing the cladding.
  The remaining 110 nm (h_dev - etch_depth) forms the unetched slab beneath the grating.
- ``h_box`` (2.0 um): Buried oxide thickness. Separates the silicon device layer from the
  silicon substrate. Must be thick enough to prevent leakage into the substrate.
- ``h_clad`` (0.78 um): SiO2 cladding thickness above the silicon device layer.
- ``h_sub`` (0.8 um): Silicon substrate below the BOX layer.
- ``h_air`` (1.0 um): Air gap above the cladding, where the fiber source is placed.
- ``pad`` (3.0 um): Absorber padding on top and bottom of the domain. These regions contain the
  adiabatic absorbers that prevent reflections at the vertical boundaries.

Grid Resolution
^^^^^^^^^^^^^^^

The grid resolution controls the accuracy and computational cost of the simulation. Two grids
are used: the structure grid (for FDTD) and the theta grid (for the design variables).

- ``dx`` (0.035 um = 35 nm): Structure grid cell size. This is the FDTD grid resolution. For
  grating coupler simulations, 35 nm provides approximately 12 cells per wavelength in silicon
  (1550 nm / 3.48 / 35 nm), which is sufficient for grating structures with feature sizes above
  ~100 nm.
- ``pixel_size`` (dx / 2 = 17.5 nm): Theta grid cell size. The design variables operate at 2x
  the structure resolution. When ``create_structure()`` builds the 3D permittivity from theta,
  it downsamples via subpixel averaging. This provides smoother gradients during optimization
  and allows finer geometric control over grating tooth positions and widths.
- ``domain`` (20.0 um): Total lateral extent of the simulation domain. Must be large enough to
  contain the grating pattern, waveguide, fiber beam footprint, and absorber margins.

Waveguide and Fiber
^^^^^^^^^^^^^^^^^^^

- ``wg_width`` (0.5 um): Output waveguide width. Standard single-mode silicon waveguide width.
- ``wg_length`` (2.5 um): Length of the fixed waveguide segment at the edge of the design
  region. This section is not optimized (held at theta=1) and provides a clean transition from
  the grating into the output waveguide.
- ``beam_waist`` (5.2 um): Gaussian beam waist radius, corresponding to the SMF-28 mode field
  diameter of ~10.4 um at 1550 nm.
- ``fiber_angle`` (14.5 degrees): Fiber tilt angle from vertical. Tilting the fiber breaks
  symmetry and suppresses second-order back-reflections from the grating. The negative sign
  applied later (``theta=-fiber_angle``) tilts the beam toward the waveguide output direction.

.. code-block:: python

   import math

   # Materials
   n_si = 3.48
   n_sio2 = 1.44
   n_clad = 1.44
   n_air = 1.0

   # Wavelength
   wavelength_um = 1.55

   # Layer thicknesses (um)
   h_dev = 0.220        # total silicon device layer
   etch_depth = 0.110   # partial etch depth
   h_box = 2.0          # buried oxide
   h_clad = 0.78        # SiO2 cladding
   h_sub = 0.8          # silicon substrate
   h_air = 1.0          # air above cladding
   pad = 3.0            # absorber padding (top and bottom)

   # Grid resolution
   dx = 0.035           # 35nm structure grid
   pixel_size = dx / 2  # 17.5nm theta grid (2x for subpixel averaging)
   domain = 20.0        # um total domain

   # Waveguide
   wg_width = 0.5       # um
   wg_length = 2.5      # um

   # Fiber
   beam_waist = 5.2     # um (SMF-28 mode field radius at 1550nm)
   fiber_angle = 14.5   # degrees from vertical

Derived Grid Dimensions
^^^^^^^^^^^^^^^^^^^^^^^

Layer thicknesses are converted from physical units (microns) to pixel counts by dividing by
the grid cell size ``dx`` and rounding to the nearest integer. The total vertical extent ``Lz``
is the sum of all layer thicknesses plus two absorber pads. The frequency is computed in
normalized FDTD units from the wavelength: ``freq = 2*pi / (wavelength / dx)``.

.. code-block:: python

   # Structure grid dimensions
   Lx = int(domain / dx)
   Ly = Lx

   # Theta grid dimensions (2x structure)
   theta_Lx = 2 * Lx
   theta_Ly = 2 * Ly

   # Layer thicknesses in pixels
   h_p = int(round(pad / dx))
   h0 = int(round(h_air / dx))
   h1 = int(round(h_clad / dx))
   h2 = int(round(etch_depth / dx))
   h3 = int(round((h_dev - etch_depth) / dx))
   h4 = int(round(h_box / dx))
   h5 = int(round(h_sub / dx))
   Lz = h_p + h0 + h1 + h2 + h3 + h4 + h5 + h_p

   # Key Z positions
   z_etch = h_p + h0 + h1
   z_slab = z_etch + h2
   z_box = z_slab + h3

   # Frequency in normalized units
   wl_px = wavelength_um / dx
   freq = 2 * np.pi / wl_px
   freq_band = (freq, freq, 1)

   # Permittivity values
   eps_si = n_si**2
   eps_sio2 = n_sio2**2
   eps_clad = n_clad**2
   eps_air = n_air**2

**Output:**

.. code-block:: text

   Structure grid: 571 x 571 x 253 (35 nm)
   Theta grid: 1142 x 1142 (17.5 nm)

Step 2: Design Variables (Theta)
--------------------------------

The 2D ``theta`` array is the design variable that the optimizer modifies at each step. It
controls the etch layer geometry: ``theta=1`` means unetched silicon (the full 220 nm device
layer is intact), and ``theta=0`` means etched to the cladding (only the bottom 110 nm slab
remains). Intermediate values (0 < theta < 1) represent partially etched regions, which appear
naturally during optimization before the design converges toward binary features.

Theta operates at 2x the structure resolution (17.5 nm vs 35 nm). When ``create_structure()``
builds the 3D permittivity, it downsamples theta via subpixel averaging. This 2x oversampling
provides two benefits: first, smoother gradients that help the optimizer converge; second,
finer positional control over grating tooth edges, which is critical for grating coupler
performance.

The initial design has three regions:

1. **Waveguide region** (theta=1.0): A fixed waveguide strip on the left side of the domain,
   extending from x=0 to x=wg_length. This region is held at theta=1 (fully unetched silicon)
   and is not modified during optimization. It provides a clean interface between the grating
   and the output waveguide.

2. **Design region** (theta=0.5): The central region where the optimizer will sculpt the grating
   pattern. Initialized to 0.5 (uniform gray), which represents a 50/50 mix of silicon and
   cladding. This starting point provides gradients in both directions (toward more or less
   silicon) at every pixel.

3. **Margin** (theta=0.0): A uniform margin on all sides of the design region keeps the
   optimized pattern away from the absorber boundaries. The margin width equals the waveguide
   length (2.5 um), which provides enough clearance for the absorbers to work effectively.

.. code-block:: python

   # Waveguide dimensions in theta pixels (17.5nm grid)
   wg_len_theta = int(round(wg_length / pixel_size))
   wg_hw_theta = int(round(wg_width / 2 / pixel_size))

   # Design region with uniform margin on all sides
   design_region = {
       'x_start': wg_len_theta,
       'x_end': theta_Lx - wg_len_theta,
       'y_start': wg_len_theta,
       'y_end': theta_Ly - wg_len_theta,
   }

   # Build theta: zeros -> fill design region with 0.5 -> stamp waveguide as 1.0
   dr = design_region
   theta_init = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
   theta_init[dr['x_start']:dr['x_end'], dr['y_start']:dr['y_end']] = 0.5
   theta_init[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

Step 3: Layer Stack
-------------------

Build an 8-layer SOI stack using ``hwc.Layer`` and ``hwc.create_structure()``. This is the same
layer-stacking approach as the :doc:`local_workflow`. Each ``Layer`` object specifies a 2D
density pattern, the permittivity value(s) for that layer, and the layer thickness in pixels.

Only the etch layer (layer index 3) is controlled by the design variable ``theta``. All other
layers have fixed, uniform permittivity: their density patterns are set to all-zeros, which
means they take the single permittivity value directly.

The ``Layer`` constructor accepts either a single permittivity value (for uniform layers) or a
tuple of two values (for design layers). When a tuple ``(eps_low, eps_high)`` is provided, the
permittivity at each grid cell is interpolated between the two values based on the density
pattern: ``eps = eps_low + density * (eps_high - eps_low)``. For the etch layer, ``eps_low`` is
the cladding permittivity (etched regions) and ``eps_high`` is the silicon permittivity
(unetched regions).

The ``vertical_radius`` parameter in ``create_structure()`` controls vertical blurring between
adjacent layers. For grating couplers, this is set to 0 because the layer transitions (air to
cladding, cladding to etch, etch to slab) are physically abrupt in real SOI fabrication.

.. list-table:: Layer Stack
   :header-rows: 1

   * - Layer
     - Material
     - Thickness
   * - pad (absorber)
     - air
     - 3.0 um
   * - air
     - air
     - 1.0 um
   * - cladding
     - SiO2
     - 0.78 um
   * - **etch (design)**
     - **SiO2/Si**
     - **0.11 um**
   * - slab
     - Si
     - 0.11 um
   * - BOX
     - SiO2
     - 2.0 um
   * - substrate
     - Si
     - 0.8 um
   * - pad (absorber)
     - Si
     - 3.0 um

.. code-block:: python

   # Slab pattern (uniform zero for non-design layers)
   slab = jnp.zeros(theta_init.shape)

   # 8-layer SOI stack
   design_layers = [
       hwc.Layer(density_pattern=slab,                    permittivity_values=eps_air,            layer_thickness=h_p),
       hwc.Layer(density_pattern=slab,                    permittivity_values=eps_air,            layer_thickness=h0),
       hwc.Layer(density_pattern=slab,                    permittivity_values=eps_clad,           layer_thickness=h1),
       hwc.Layer(density_pattern=jnp.array(theta_init),   permittivity_values=(eps_clad, eps_si), layer_thickness=h2),
       hwc.Layer(density_pattern=slab,                    permittivity_values=eps_si,             layer_thickness=h3),
       hwc.Layer(density_pattern=slab,                    permittivity_values=eps_sio2,           layer_thickness=h4),
       hwc.Layer(density_pattern=slab,                    permittivity_values=eps_si,             layer_thickness=h5),
       hwc.Layer(density_pattern=slab,                    permittivity_values=eps_si,             layer_thickness=h_p),
   ]

   # Build 3D structure locally
   structure = hwc.create_structure(layers=design_layers, vertical_radius=0)

Visualize cross-sections to verify the layer stack is correct. The XY slice at ``z_dev`` shows
the etch layer pattern (the design region and waveguide), while the XZ slice at the domain
center shows the vertical layer arrangement:

.. code-block:: python

   z_dev = z_etch + int(h2 // 2)
   structure.view(show_permittivity=True, show_conductivity=False, axis="z", position=z_dev)
   structure.view(show_permittivity=True, show_conductivity=False, axis="x", position=Lx // 2)

Step 4: Absorbing Boundaries
-----------------------------

Adiabatic absorbers prevent reflections at the simulation domain edges. Without them, outgoing
light would reflect off the grid boundaries and interfere with the physical fields inside the
simulation, producing incorrect results.

``hwc.absorber_params()`` returns auto-tuned absorber widths and absorption coefficient derived
from power-law fits to Bayesian-optimized parameters. The widths are scaled for your specific
grid resolution, wavelength, and domain size. The returned parameters are:

- ``absorption_widths``: A list of three integers ``[wx, wy, wz]`` specifying the absorber
  width in pixels along each axis. Wider absorbers provide better absorption but reduce the
  usable simulation domain.
- ``abs_coeff``: The peak absorption coefficient. The absorber ramps up from zero at the inner
  boundary to this value at the grid edge, following an adiabatic (gradual) profile that
  minimizes reflections at the absorber interface.

The absorber is applied by creating a 3D conductivity mask with
``hwc.create_absorption_mask()`` and adding it to the structure's conductivity field. Any
electromagnetic energy entering the absorber region is progressively attenuated.

.. code-block:: python

   ap = hwc.absorber_params(wavelength_um, dx, structure_dimensions=(Lx, Ly, Lz))
   abs_widths = ap["absorption_widths"]
   abs_coeff = ap["abs_coeff"]

   # Create absorption mask and add to structure
   absorber = hwc.create_absorption_mask(
       grid_shape=(Lx, Ly, Lz),
       absorption_widths=abs_widths,
       absorption_coeff=abs_coeff,
   )
   structure.conductivity = jnp.zeros_like(structure.conductivity) + absorber

**Output:**

.. code-block:: text

   Absorber widths (x,y,z): [85, 85, 85] px = (2.97, 2.97, 2.97) um
   Absorber coefficient: 0.003500

Step 5: Gaussian Source
-----------------------

Generate a Gaussian beam source on a cloud GPU using the wave equation error method. Unlike
the mode source used in the :doc:`local_workflow` (which excites a waveguide mode), the
Gaussian source simulates a fiber illuminating the grating coupler from above. The beam
propagates downward at the specified fiber angle, producing a tilted Gaussian intensity profile
at the grating surface.

This step consumes credits because it runs a short FDTD simulation on a cloud GPU to propagate
the Gaussian beam through the grid and record the resulting complex field distribution.

The key parameters for ``hwc.generate_gaussian_source()`` are:

- ``sim_shape``: ``(Lx, Ly, Lz)`` tuple defining the simulation grid dimensions. Must match
  the structure grid.
- ``frequencies``: Array of normalized frequencies. Use ``np.array([freq])`` for a single
  frequency simulation.
- ``source_pos``: ``(x, y, z)`` tuple in pixel coordinates specifying the beam center position.
  The x and y coordinates are set to the center of the grating design region. The z coordinate
  is placed 50 nm above the cladding surface, inside the air gap where a physical fiber tip
  would sit.
- ``waist_radius``: Gaussian beam waist in pixels (``beam_waist / dx``). For SMF-28 fiber at
  1550 nm, this is 5.2 um / 0.035 um = ~149 pixels.
- ``theta``: Beam tilt angle in degrees. A negative value tilts the beam toward the -x
  direction (toward the waveguide output). The standard 14.5-degree tilt suppresses
  second-order back-reflections from the grating.
- ``phi``: Azimuthal tilt angle (0.0 for tilt in the XZ plane only).
- ``polarization``: ``'y'`` for TE polarization (electric field along the grating teeth).
- ``max_steps``: Maximum FDTD steps for the beam generation simulation. 5000 is typically
  sufficient for the beam to propagate through the grid.
- ``wavelength_um`` / ``dx_um``: Physical wavelength and grid cell size, used internally by the
  auto-tuned absorber parameters.
- ``gpu_type``: GPU model to use. ``"B200"`` is the default NVIDIA B200 GPU.

.. code-block:: python

   # Source position: in the air gap, 50nm above cladding surface
   source_above_surface_um = 0.05
   source_z = int(round((pad + h_air - source_above_surface_um) / dx))

   # Grating center in structure pixels
   grating_x = int(round((dr['x_start'] + dr['x_end']) / 2 * pixel_size / dx))
   grating_y = Ly // 2
   waist_px = beam_waist / dx

   # Generate Gaussian source on cloud GPU
   source_field, input_power = hwc.generate_gaussian_source(
       sim_shape=(Lx, Ly, Lz),
       frequencies=np.array([freq]),
       source_pos=(grating_x, grating_y, source_z),
       waist_radius=waist_px,
       theta=-fiber_angle,     # negative = tilt toward waveguide (-x)
       phi=0.0,                # tilt in XZ plane
       polarization='y',       # TE polarization
       max_steps=5000,
       wavelength_um=wavelength_um,
       dx_um=dx,
       gpu_type="B200",
   )

   source_offset = (0, 0, source_z)
   input_power = float(np.mean(input_power))

The returned ``source_field`` has shape ``(1, 6, Lx, Ly, 1)``: one frequency, six field
components (Ex, Ey, Ez, Hx, Hy, Hz), the full XY plane, and a single Z slice. The
``source_offset`` tuple tells the simulator where to inject this field in the 3D grid.
``input_power`` is a scalar representing the total beam power, used later to normalize the
coupling efficiency.

Step 6: Waveguide Mode
-----------------------

Compute the fundamental TE0 waveguide mode at the output port. This mode profile defines the
target for the loss function: the optimizer will maximize the overlap between the scattered
field from the grating and this waveguide mode.

The process has two stages:

1. **E-only eigenmode solve (local, free)**: Build a small waveguide structure, extract the
   YZ permittivity cross-section, and solve the 2D eigenvalue problem using
   ``hwc_mode()``. This produces the electric field profile and effective index of the
   fundamental mode. The solve runs locally on CPU with no GPU cost.

2. **E-to-EH conversion (cloud GPU, credits)**: The eigenmode solver only produces E-field
   components. A short FDTD propagation via ``hwc.mode_convert()`` generates the
   corresponding H-field components by propagating the E-only mode through a short waveguide
   segment. This step runs on a cloud GPU and consumes a small amount of credits.

Building the Mode Solve Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A narrow waveguide structure is built for the mode solve. The x dimension is kept small (40
theta pixels, about 0.7 um) because the mode profile is uniform along the propagation
direction and only the YZ cross-section matters. The full Y dimension is preserved to capture
the complete mode field.

The waveguide structure uses the same layer stack as the main simulation, with two important
differences: the etch layer contains only the waveguide pattern (a strip at theta=1), and the
slab layer (layer index 4) also contains a waveguide pattern to ensure the mode solver sees
the correct 220 nm waveguide cross-section rather than just the 110 nm slab.

.. code-block:: python

   from hyperwave_community.mode_solver import mode as hwc_mode

   # Build a small waveguide structure for mode solving
   small_x_theta = 40
   theta_mode = np.zeros((small_x_theta, theta_Ly), dtype=np.float32)
   theta_mode[:, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

   theta_mode_bot = np.where(theta_mode > 0, 1.0, 0.0).astype(np.float32)
   d_mode_slab = jnp.zeros((small_x_theta, theta_Ly))

   wg_structure = hwc.create_structure(layers=[
       hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_air,            layer_thickness=h_p),
       hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_air,            layer_thickness=h0),
       hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_clad,           layer_thickness=h1),
       hwc.Layer(density_pattern=jnp.array(theta_mode),     permittivity_values=(eps_clad, eps_si), layer_thickness=h2),
       hwc.Layer(density_pattern=jnp.array(theta_mode_bot), permittivity_values=(eps_clad, eps_si), layer_thickness=h3),
       hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_sio2,           layer_thickness=h4),
       hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_si,             layer_thickness=h5),
       hwc.Layer(density_pattern=d_mode_slab,               permittivity_values=eps_si,             layer_thickness=h_p),
   ], vertical_radius=0)

Eigenmode Solve and Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The YZ permittivity cross-section is cropped around the waveguide core to reduce the eigenvalue
problem size. The crop region extends 50 pixels in Y and 30 pixels in Z around the core center,
which is sufficient to capture the mode's evanescent tails while keeping the solve fast.

The ``hwc_mode()`` function takes a 4D permittivity tensor ``(3, 1, y, z)`` (three polarization
components, one x-slice) and returns the E-field modes, propagation constants (beta), and mode
indices. ``mode_num=0`` selects the fundamental mode. The effective index ``n_eff`` should be
between 2.0 and 3.0 for a standard 220 nm SOI waveguide at 1550 nm.

After the eigenmode solve, ``hwc.mode_convert()`` propagates the E-only field through a 500-pixel
waveguide segment on a cloud GPU to generate the corresponding H-field components. The resulting
``mode_EH`` array has all six field components and can be used directly in the overlap integral.

The H-field is negated (``mode_EH[:, 3:6, ...] *= -1``) because the mode propagates in the -x
direction (from the grating toward the waveguide output), which is opposite to the standard
positive-x convention.

.. code-block:: python

   # Crop YZ cross-section around waveguide core
   eps_yz = wg_structure.permittivity[0, wg_structure.permittivity.shape[1] // 2, :, :]
   crop_y, crop_z = min(50, Ly // 4), min(30, Lz // 4)
   y_c = eps_yz.shape[0] // 2
   y0, y1 = y_c - crop_y, y_c + crop_y
   z0 = max(0, z_etch - crop_z)
   z1 = min(eps_yz.shape[1], z_box + crop_z)
   eps_crop = eps_yz[y0:y1, z0:z1]

   # Solve E-only eigenmode locally (no GPU needed)
   eps_4d = jnp.stack([jnp.array(eps_crop)] * 3, axis=0)[:, jnp.newaxis, :, :]
   mode_E, beta_arr, _ = hwc_mode(freq_band=freq_band, permittivity=eps_4d, axis=0, mode_num=0)
   n_eff = float(beta_arr[0]) / (2 * np.pi / wl_px)

   # Convert E-only -> full E+H via short FDTD propagation on cloud GPU
   mode_EH = hwc.mode_convert(
       mode_E_field=mode_E[0:1, 0:3, :, :, :],
       freq_band=freq_band,
       permittivity_slice=np.array(eps_crop),
       propagation_axis='x',
       propagation_length=500,
       gpu_type="B200",
   )

   # Negate H for backward (-x) propagation
   mode_EH = np.array(mode_EH, copy=True)
   mode_EH[:, 3:6, ...] *= -1

Mode Self-Overlap (P_mode_cross)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mode self-overlap integral ``P_mode_cross`` normalizes the coupling efficiency calculation.
It is the integral of ``Re(E_mode x H_mode*)`` over the mode cross-section, representing the
total power carried by the mode. This value appears in the denominator of the efficiency
formula in Step 9.

After computing the self-overlap, the mode field is zero-padded to the full YZ domain size so
it can be passed directly to the optimizer.

.. code-block:: python

   mode_e = np.array(mode_EH[0, 0:3, 0, :, :])
   mode_h = np.array(mode_EH[0, 3:6, 0, :, :])
   cross = np.cross(mode_e, np.conj(mode_h), axis=0)
   P_mode_cross = float(np.abs(np.real(np.sum(cross[0, :, :]))))

   # Pad mode to full YZ domain
   mode_field = np.zeros((1, 6, 1, Ly, Lz), dtype=np.complex64)
   mode_field[:, :, :, y0:y1, z0:z1] = np.array(mode_EH)

Step 7: Monitors
----------------

Set up field monitors for visualization and optimization. Monitors are 3D volumes that record
the electromagnetic field at specified locations during the simulation. Four categories of
monitors are used in the inverse design workflow:

Visualization Monitors
^^^^^^^^^^^^^^^^^^^^^^

Full-plane slices used to plot field intensity after simulation. These monitors capture the
entire XY, XZ, or YZ cross-section at a specified position, providing a complete picture of
how light propagates through the device.

- ``Output_xy_device``: XY plane at the etch layer height (``z_dev``). Shows the in-plane field
  pattern across the grating and waveguide.
- ``Output_xz_center``: XZ plane at the domain center (``Ly // 2``). Shows the vertical field
  distribution and how the beam couples down into the grating.
- ``Output_yz_center``: YZ plane at the domain center (``Lx // 2``). Shows the cross-sectional
  field distribution at the grating center.

Waveguide Output Monitor
^^^^^^^^^^^^^^^^^^^^^^^^^

A YZ cross-section placed at the waveguide output, just inside the absorber boundary
(``abs_widths[0] + 10`` pixels from the x=0 edge). This monitor captures the field that has
coupled into the waveguide mode, which is compared against the target mode from Step 6 to
compute coupling efficiency.

Loss and Design Monitors
^^^^^^^^^^^^^^^^^^^^^^^^^

These monitors are defined separately (not added to the ``MonitorSet``) and passed directly to
the optimizer:

- **Loss monitor**: Positioned at the same location as the waveguide output monitor. The
  optimizer evaluates the mode overlap integral on this plane at each step.
- **Design monitor**: A 3D volume covering only the etch layer within the design region. This
  restricts gradient computation to the region where theta is being optimized, which is far
  more memory-efficient than differentiating through the full 3D grid. The design monitor shape
  and offset are computed by converting the theta-grid design region coordinates to structure-grid
  coordinates (dividing by 2).

.. code-block:: python

   monitors = hwc.MonitorSet()

   # Visualization monitors (full-plane slices)
   monitors.add(hwc.Monitor(shape=(Lx, Ly, 1), offset=(0, 0, z_dev)), name='Output_xy_device')
   monitors.add(hwc.Monitor(shape=(Lx, 1, Lz), offset=(0, Ly // 2, 0)), name='Output_xz_center')
   monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(Lx // 2, 0, 0)), name='Output_yz_center')

   # Waveguide output monitor
   output_x = abs_widths[0] + 10
   monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(output_x, 0, 0)), name='Output_wg_output')

   monitors.list_monitors()

Visualize monitor positions overlaid on the structure:

.. code-block:: python

   monitors.view(
       structure=structure,
       axis="z",
       position=z_dev,
       absorber_boundary=absorber,
   )

Define the loss and design monitors for the optimizer:

.. code-block:: python

   # Loss monitor: waveguide output plane
   loss_monitor_shape = (1, Ly, Lz)
   loss_monitor_offset = (output_x, 0, 0)

   # Design monitor: etch layer volume (gradients computed here only)
   dr_x0 = dr['x_start'] // 2  # theta to structure coordinates
   dr_x1 = dr['x_end'] // 2
   dr_y0 = dr['y_start'] // 2
   dr_y1 = dr['y_end'] // 2
   design_monitor_shape = (dr_x1 - dr_x0, dr_y1 - dr_y0, int(round(h2)))
   design_monitor_offset = (dr_x0, dr_y0, z_etch)

Step 8: Forward Simulation
--------------------------

Run a forward FDTD simulation with the initial design to verify the setup before committing to
a full optimization run. This costs one GPU simulation in credits, but it catches errors in
source placement, monitor positions, layer stack configuration, and absorber sizing early,
before you spend credits on multiple optimization steps.

The structure recipe is extracted using ``structure.extract_recipe()``, which serializes the 3D
permittivity and conductivity arrays into a compact format for transfer to the cloud GPU. This
is the same serialization used in the :doc:`local_workflow`.

Key parameters for ``hwc.simulate()``:

- ``structure_recipe``: Serialized 3D permittivity and conductivity from ``extract_recipe()``.
- ``source_field`` / ``source_offset``: The Gaussian beam field and its injection position from
  Step 5.
- ``freq_band``: Normalized frequency band from Step 1.
- ``monitors_recipe``: Serialized monitor positions and shapes.
- ``absorption_widths`` / ``absorption_coeff``: Absorber parameters from Step 4.
- ``simulation_steps``: Maximum number of FDTD time steps. 10000 is used for this forward
  verification run (fewer than the optimization, which uses its own internal step count).
- ``gpu_type``: ``"B200"`` for NVIDIA B200 GPU.

.. code-block:: python

   # Extract recipe from local structure
   structure_recipe = structure.extract_recipe()

   fwd_results = hwc.simulate(
       structure_recipe=structure_recipe,
       source_field=source_field,
       source_offset=source_offset,
       freq_band=freq_band,
       monitors_recipe=monitors.recipe,
       absorption_widths=abs_widths,
       absorption_coeff=abs_coeff,
       gpu_type="B200",
       simulation_steps=10000,
   )

Visualize the field distribution across all monitors and check the initial coupling into the
waveguide using Poynting flux:

.. code-block:: python

   hwc.quick_view_monitors(fwd_results, component='all')

   # Check power at waveguide output
   wg_field = np.array(fwd_results['monitor_data']['Output_wg_output'])
   S = hwc.S_from_slice(jnp.mean(jnp.array(wg_field), axis=2))
   power = float(jnp.abs(jnp.sum(S[0, 0, :, :])))
   print(f"Coupling (approx): {power / input_power * 100:.1f}%")

The initial coupling will be low (typically <5%) because the uniform 0.5 theta does not form
a coherent grating pattern. The optimizer will improve this in the next step.

Step 9: Optimization
--------------------

``hwc.run_optimization()`` runs the full optimization loop on a cloud GPU and streams results
back after each step via WebSocket (with SSE fallback). Each step runs a forward and adjoint
FDTD simulation to compute the gradient of the loss function with respect to theta, then
updates theta using the Adam optimizer with cosine learning rate decay.

Interrupting the kernel (pressing the stop button in Colab) cancels the GPU task immediately.
You are only charged for completed steps, and all step results received so far are kept in the
``results`` list.

Loss Function
^^^^^^^^^^^^^

The mode coupling loss measures how efficiently the scattered field couples into the target
waveguide mode (computed in Step 6). It uses the standard bidirectional overlap integral from
coupled mode theory:

.. code-block:: text

   eta = |Re(I1 * I2)| / (2 * P_in * P_mode)

where the overlap integrals are:

.. code-block:: text

   I1 = integral over A of (E_mode x H_sim*) . n dA
   I2 = integral over A of (E_sim x H_mode*) . n dA

and the normalization terms are:

- ``P_in``: Input source power from Step 5 (``input_power``).
- ``P_mode``: Mode self-overlap integral from Step 6 (``P_mode_cross``).

For TE polarization propagating along x, only the Ey*Hz - Ez*Hy components contribute to
the cross products. This is exactly the formula used in Step 11 (verification) to independently
check the result.

``run_optimization`` supports three built-in loss types, selected by which keyword arguments
are provided:

.. list-table::
   :header-rows: 1

   * - Loss Type
     - Parameter
     - Use Case
   * - Mode coupling
     - ``mode_field=...``
     - Waveguide coupling (this tutorial)
   * - Poynting power
     - ``power_axis=0``
     - Maximize directional power flow
   * - Intensity
     - ``intensity_component='Ey'``
     - Maximize field at a point

Structure Specification
^^^^^^^^^^^^^^^^^^^^^^^

The ``structure_spec`` dict provides the layer stack template so the cloud GPU can reconstruct
the 3D permittivity from any theta value at each optimization step. It contains:

- ``layers_info``: A list of dicts, one per layer, each with ``permittivity_values`` (single
  float or list of two floats), ``layer_thickness`` (int), ``density_radius`` (set to 0 for no
  density filtering), and ``density_alpha`` (set to 0).
- ``construction_params``: Dict with ``vertical_radius`` (set to 0 for grating couplers).

The waveguide mask is a 2D array of the same shape as theta that forces theta=1 in the
waveguide region during optimization. Any pixel where the mask is 1.0 will not be modified by
the gradient update, preserving the output waveguide geometry.

.. code-block:: python

   # Structure spec: layer stack template for GPU reconstruction
   structure_spec = {
       'layers_info': [{
           'permittivity_values': [float(v) for v in l.permittivity_values]
               if isinstance(l.permittivity_values, tuple) else float(l.permittivity_values),
           'layer_thickness': float(l.layer_thickness),
           'density_radius': 0,
           'density_alpha': 0,
       } for l in design_layers],
       'construction_params': {'vertical_radius': 0},
   }

   # Waveguide mask: forces theta=1 in waveguide region (not optimized)
   waveguide_mask = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
   waveguide_mask[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

Optimizer Settings
^^^^^^^^^^^^^^^^^^

- ``NUM_STEPS`` (5): Number of optimization steps. Use 5 for a quick demo; increase to 50-100
  for production designs that converge to high efficiency.
- ``LR`` (0.1): Initial learning rate for the Adam optimizer. The learning rate decays via a
  cosine schedule from this initial value to ``LR * 0.1`` over the full optimization run.
- ``GRAD_CLIP`` (1.0): Maximum gradient norm. Gradients exceeding this threshold are clipped to
  prevent unstable updates in early optimization steps.

.. code-block:: python

   NUM_STEPS = 5       # increase to 50-100 for production
   LR = 0.1
   GRAD_CLIP = 1.0

Run Optimization
^^^^^^^^^^^^^^^^

The ``run_optimization()`` call returns a generator that yields a ``step_result`` dict after
each completed step. Each dict contains:

- ``step``: Current step number (1-indexed).
- ``loss``: The loss function value (negative of efficiency). Take ``abs(loss)`` to get the
  coupling efficiency as a fraction.
- ``theta``: The updated 2D design array after this step's gradient update.
- ``grad_max``: Maximum absolute gradient value, useful for monitoring convergence.
- ``step_time``: Wall-clock time for this step in seconds.

Each step runs a forward and adjoint FDTD simulation, so the GPU cost is roughly 2x a single
simulation per step.

Key parameters for ``run_optimization()``:

- ``theta``: Initial design array (2D, float32).
- ``source_field`` / ``source_offset``: From Step 5.
- ``freq_band``: Normalized frequency band from Step 1.
- ``structure_spec``: Layer stack template from above.
- ``loss_monitor_shape`` / ``loss_monitor_offset``: Where to evaluate the loss function.
- ``design_monitor_shape`` / ``design_monitor_offset``: Volume where gradients are computed.
- ``mode_field``: Target waveguide mode (6 field components) from Step 6.
- ``input_power``: Source power normalization scalar from Step 5.
- ``mode_cross_power``: Mode self-overlap integral from Step 6.
- ``mode_axis``: Propagation axis index (0 for x-direction).
- ``waveguide_mask``: 2D mask protecting the waveguide from optimization.
- ``num_steps``: Total number of optimization steps.
- ``learning_rate``: Initial Adam learning rate.
- ``grad_clip_norm``: Maximum gradient norm for clipping.
- ``absorption_widths`` / ``absorption_coeff``: Absorber parameters from Step 4.
- ``gpu_type``: ``"B200"`` for NVIDIA B200 GPU.

.. code-block:: python

   results = []

   for step_result in hwc.run_optimization(
       theta=theta_init,
       source_field=source_field,
       source_offset=source_offset,
       freq_band=freq_band,
       structure_spec=structure_spec,
       loss_monitor_shape=loss_monitor_shape,
       loss_monitor_offset=loss_monitor_offset,
       design_monitor_shape=design_monitor_shape,
       design_monitor_offset=design_monitor_offset,
       mode_field=mode_field,
       input_power=input_power,
       mode_cross_power=P_mode_cross,
       mode_axis=0,
       waveguide_mask=waveguide_mask,
       num_steps=NUM_STEPS,
       learning_rate=LR,
       grad_clip_norm=GRAD_CLIP,
       absorption_widths=abs_widths,
       absorption_coeff=abs_coeff,
       gpu_type="B200",
   ):
       results.append(step_result)
       eff = abs(step_result['loss']) * 100
       print(f"Step {step_result['step']:3d}/{NUM_STEPS}:  eff = {eff:.2f}%  "
             f"|grad|_max = {step_result['grad_max']:.3e}  ({step_result['step_time']:.1f}s)")

Step 10: Results
----------------

After optimization completes, plot the efficiency curve and compare the initial and optimized
designs. The efficiency at each step is ``abs(loss) * 100`` (percentage). The best step is
selected by maximum efficiency, which may not be the final step if the optimizer overshoots.

.. code-block:: python

   efficiencies = [abs(r['loss']) * 100 for r in results]
   best_idx = int(np.argmax(efficiencies))
   best_theta = results[best_idx]['theta']

   fig, axes = plt.subplots(1, 3, figsize=(18, 5))

   # Efficiency curve
   axes[0].plot(range(1, len(efficiencies) + 1), efficiencies, 'b-o', markersize=3)
   axes[0].set_xlabel('Step')
   axes[0].set_ylabel('Efficiency (%)')
   axes[0].set_title('Mode Coupling Efficiency')
   axes[0].grid(True, alpha=0.3)

   # Initial vs best theta
   extent = [0, theta_Lx * pixel_size, 0, theta_Ly * pixel_size]
   axes[1].imshow(theta_init.T, origin='upper', cmap='gray', vmin=0, vmax=1, extent=extent)
   axes[1].set_title('Initial')
   axes[2].imshow(best_theta.T, origin='upper', cmap='gray', vmin=0, vmax=1, extent=extent)
   axes[2].set_title(f'Best (step {best_idx + 1})')

   for ax in axes[1:]:
       ax.set_xlabel('x (um)')
       ax.set_ylabel('y (um)')
   plt.tight_layout()
   plt.show()

Step 11: Verification
---------------------

Run a final forward simulation with the best theta to independently verify the coupling
efficiency. This is important because the optimizer reports efficiency *before* each step's
gradient update: the loss value at step N corresponds to the theta from step N-1. The
``best_theta`` includes the final gradient update, so the verified efficiency will typically be
slightly higher than the last reported value.

The verification simulation uses ``convergence="full"`` to ensure it uses the same FDTD solver
(``mem_efficient_multi_freq``) as the optimizer in Step 9. Without this parameter,
``simulate()`` defaults to ``convergence="default"``, which routes to the
``early_stopping_solve`` solver. This alternative solver can produce slightly different results
(~0.5% gap) due to different field accumulation strategies.

.. code-block:: python

   # Rebuild structure with optimized theta
   opt_layers = list(design_layers)
   opt_layers[3] = hwc.Layer(
       density_pattern=jnp.array(best_theta),
       permittivity_values=(eps_clad, eps_si),
       layer_thickness=h2,
   )
   opt_structure = hwc.create_structure(layers=opt_layers, vertical_radius=0)
   opt_structure.conductivity = jnp.zeros_like(opt_structure.conductivity) + absorber

   opt_recipe = opt_structure.extract_recipe()

   opt_results = hwc.simulate(
       structure_recipe=opt_recipe,
       source_field=source_field,
       source_offset=source_offset,
       freq_band=freq_band,
       monitors_recipe=monitors.recipe,
       absorption_widths=abs_widths,
       absorption_coeff=abs_coeff,
       gpu_type="B200",
       simulation_steps=10000,
       convergence="full",
   )

Compute the mode coupling efficiency using the overlap integral (the same formula the optimizer
uses internally). For TE polarization along x, the relevant cross-product components are
Ey*Hz and Ez*Hy:

.. code-block:: python

   wg_field = np.array(opt_results['monitor_data']['Output_wg_output'])
   field_avg = np.mean(wg_field, axis=2)
   E_out = field_avg[0, 0:3, :, :]
   H_out = field_avg[0, 3:6, :, :]
   E_m = mode_field[0, 0:3, 0, :, :]
   H_m = mode_field[0, 3:6, 0, :, :]

   I1 = np.sum(E_m[1] * np.conj(H_out[2]) - E_m[2] * np.conj(H_out[1]))
   I2 = np.sum(E_out[1] * np.conj(H_m[2]) - E_out[2] * np.conj(H_m[1]))
   mode_eff = abs(np.real(I1 * I2)) / (2.0 * input_power * P_mode_cross) * 100

   print(f"Mode coupling: {mode_eff:.2f}% ({-10*np.log10(max(mode_eff/100, 1e-10)):.2f} dB)")

Summary
-------

.. list-table::
   :header-rows: 1

   * - Step
     - Function
     - Runs On
     - Cost
   * - 1-2
     - Physical parameters, theta design variables
     - Local
     - Free
   * - 3
     - ``Layer()``, ``create_structure()``
     - Local
     - Free
   * - 4
     - ``absorber_params()``, ``create_absorption_mask()``
     - Local
     - Free
   * - 5
     - ``generate_gaussian_source()``
     - Cloud GPU
     - Credits ($25/hr)
   * - 6
     - ``mode()``, ``mode_convert()``
     - Local + Cloud GPU
     - Credits ($25/hr)
   * - 7
     - ``MonitorSet()``, ``Monitor()``
     - Local
     - Free
   * - 8
     - ``simulate()`` (forward verification)
     - Cloud GPU
     - Credits ($25/hr)
   * - 9
     - ``run_optimization()`` (adjoint)
     - Cloud GPU
     - Credits, 2 sims/step ($25/hr)
   * - 10-11
     - Analysis and verification
     - Local + Cloud GPU
     - Credits ($25/hr)

Next Steps
----------

* Increase ``NUM_STEPS`` to 50-100 for production designs
* Try different initial theta values (0.3, 0.5, 0.7, or random) to explore different local optima
* Decrease ``dx`` (e.g., 25 nm) for finer resolution and more accurate results
* Expand ``freq_band`` to multiple wavelengths (e.g., 1530-1570 nm) for broadband optimization
* Fabrication constraints (density filtering, binarization, minimum feature size) are in development
* :doc:`local_workflow` - Step-by-step local workflow with full control
* :doc:`api_workflow` - Cloud-based forward simulations
* :doc:`../gpu_options` - GPU performance and cost reference
* :doc:`../convergence` - Early stopping configuration
* :doc:`../api` - Full API reference
