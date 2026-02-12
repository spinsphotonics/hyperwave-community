.. _inverse-design-workflow:

Inverse Design Workflow
=======================

This tutorial walks through gradient-based topology optimization of a silicon-on-insulator (SOI) grating coupler using the adjoint method on cloud GPUs. Each optimization step runs a forward and adjoint FDTD simulation to compute the gradient of a loss function with respect to design variables. The optimizer (Adam with cosine learning rate decay) updates the 2D design pattern to maximize mode coupling efficiency from a fiber into a waveguide.

**Open in Google Colab**: `gc_inverse_design.ipynb <https://colab.research.google.com/github/spinsphotonics/hyperwave-community/blob/main/examples/gc_inverse_design.ipynb>`_

`Download the notebook <https://github.com/spinsphotonics/hyperwave-community/blob/main/examples/gc_inverse_design.ipynb>`_

.. contents:: Steps
   :local:
   :depth: 1

Prerequisites
-------------

Install the package and configure your API key:

.. code-block:: python

   %pip install "hyperwave-community @ git+https://github.com/spinsphotonics/hyperwave-community.git@main" -q

   import hyperwave_community as hwc

   from google.colab import userdata
   hwc.configure_api(api_key=userdata.get('HYPERWAVE_API_KEY'))
   hwc.get_account_info()

You will also need these imports throughout the workflow:

.. code-block:: python

   import math
   import numpy as np
   import jax.numpy as jnp
   import matplotlib.pyplot as plt

Step 1: Physical Parameters
---------------------------

Define the SOI platform parameters. Standard 220nm silicon-on-insulator at 1550nm wavelength with a 35nm structure grid. The theta grid operates at 2x resolution (17.5nm) for subpixel averaging.

.. code-block:: python

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
   domain = 40.0        # um total domain

   # Waveguide
   wg_width = 0.5       # um
   wg_length = 5.0      # um

   # Fiber
   beam_waist = 5.2     # um (SMF-28 mode field radius at 1550nm)
   fiber_angle = 14.5   # degrees from vertical

Layer thicknesses are converted to pixel units as floats (not integers) to preserve subpixel averaging at layer interfaces. This is critical for thin layers like the etch region, where rounding would shift interfaces by up to half a grid cell.

.. code-block:: python

   # Layer thicknesses in pixels (FLOAT for subpixel averaging)
   h_p = pad / dx
   h0 = h_air / dx
   h1 = h_clad / dx
   h2 = etch_depth / dx
   h3 = (h_dev - etch_depth) / dx
   h4 = h_box / dx
   h5 = h_sub / dx
   Lz = int(math.ceil(h_p + h0 + h1 + h2 + h3 + h4 + h5 + h_p))

   # Permittivity values
   eps_si = n_si**2
   eps_sio2 = n_sio2**2
   eps_clad = n_clad**2
   eps_air = n_air**2

   # Frequency in normalized units
   wl_px = wavelength_um / dx
   freq = 2 * np.pi / wl_px
   freq_band = (freq, freq, 1)

**Output:**

.. code-block:: text

   Structure grid: 1142 x 1142 x 253 (35 nm)
   Theta grid: 2284 x 2284 (17.5 nm)

Step 2: Grid and Layer Stack
----------------------------

Build an 8-layer SOI stack using ``hwc.Layer``. The etch layer (layer 3) is controlled by the design variable ``theta`` with density filtering (``radius=3``, ``alpha=0.8``) for minimum feature size control.

Layers from top to bottom: pad (air, 3um), air (1um), cladding (SiO2, 0.78um), etch (SiO2/Si, 0.11um, designable), slab (Si, 0.11um), BOX (SiO2, 2um), substrate (Si, 0.8um), pad (Si, 3um).

.. code-block:: python

   DENSITY_RADIUS = 3
   DENSITY_ALPHA = 0.8
   DESIGN_LAYER = 3  # etch layer index

   density_etch = hwc.density(jnp.array(theta_init), radius=DENSITY_RADIUS, alpha=DENSITY_ALPHA)
   zero = jnp.zeros((theta_Lx, theta_Ly))
   ones = jnp.ones((theta_Lx, theta_Ly))

   # hwc.Layer(density_pattern, permittivity_values, layer_thickness)
   # Float thicknesses enable subpixel averaging at layer interfaces.
   layers = [
       hwc.Layer(zero,         eps_air,            h_p),  # pad (top)
       hwc.Layer(zero,         eps_air,            h0),   # air
       hwc.Layer(zero,         eps_clad,           h1),   # cladding
       hwc.Layer(density_etch, (eps_clad, eps_si), h2),   # etch (designable)
       hwc.Layer(ones,         eps_si,             h3),   # slab (solid Si)
       hwc.Layer(zero,         eps_sio2,           h4),   # BOX
       hwc.Layer(zero,         eps_si,             h5),   # substrate
       hwc.Layer(zero,         eps_si,             h_p),  # pad (bottom)
   ]
   structure = hwc.create_structure(layers=layers, vertical_radius=0)

The etch layer uses a tuple ``(eps_clad, eps_si)`` for its permittivity values, meaning that where the density pattern is 0 (etched) the permittivity is ``eps_clad``, and where it is 1 (unetched) the permittivity is ``eps_si``.

Step 3: Initial Design
----------------------

The 2D ``theta`` array controls the etch layer geometry. ``theta=1`` means unetched silicon, ``theta=0`` means etched to cladding. Theta operates at 2x the structure resolution (17.5nm vs 35nm); ``create_structure`` downsamples via subpixel averaging in the ``epsilon()`` function, providing smoother gradients and finer geometric control.

The initial design has a fixed waveguide on the left side and a rectangular design region initialized to 0.5 (uniform gray).

.. code-block:: python

   # Theta grid dimensions (17.5nm, 2x structure)
   theta_Lx = 2 * Lx
   theta_Ly = 2 * Ly

   # Design region (in theta pixels)
   abs_margin = 80  # structure pixels
   abs_margin_theta = 2 * abs_margin
   design_region = {
       'x_start': wg_len_theta,
       'x_end': theta_Lx - abs_margin_theta,
       'y_start': abs_margin_theta,
       'y_end': theta_Ly - abs_margin_theta,
   }

   # Build initial theta
   theta_init = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
   # Fixed waveguide
   theta_init[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0
   # Design region at 0.5
   dr = design_region
   theta_init[dr['x_start']:dr['x_end'], dr['y_start']:dr['y_end']] = 0.5

Step 4: Gaussian Source
-----------------------

Create a unidirectional Gaussian beam using ``hwc.create_gaussian_source``. The beam propagates downward (-z) at 14.5 degrees from vertical, simulating fiber illumination from above at the standard SMF coupling angle. A negative ``theta`` parameter tilts the beam toward the waveguide (in the -x direction).

.. code-block:: python

   # Source position: in the air gap, 50nm above cladding surface
   source_above_surface_um = 0.05
   source_z = int(round((pad + h_air - source_above_surface_um) / dx))

   # Grating center in structure pixels
   grating_x = int(round((dr['x_start'] + dr['x_end']) / 2 * pixel_size / dx))
   grating_y = Ly // 2
   waist_px = beam_waist / dx

   source_field, input_power = hwc.create_gaussian_source(
       sim_shape=(Lx, Ly, Lz),
       frequencies=jnp.array([freq]),
       source_pos=(grating_x, grating_y, source_z),
       waist_radius=waist_px,
       theta=-fiber_angle,  # negative tilts beam toward waveguide
       phi=0.0,
       polarization='y',
       x_span=float(Lx),
       y_span=float(Ly),
       max_steps=5000,
       check_every_n=200,
       show_plots=False,
   )

   source_offset = (grating_x - Lx // 2, grating_y - Ly // 2, source_z)

Step 5: Waveguide Mode
----------------------

Solve for the fundamental TE eigenmode at the waveguide cross-section using ``hwc.create_mode_source``. Then use ``mode_converter`` to obtain the full E+H field profile, which is needed for mode overlap loss computation during optimization.

.. code-block:: python

   from hyperwave_community.sources import mode_converter

   source_mode, offset_mode, mode_info = hwc.create_mode_source(
       structure=structure,
       freq_band=freq_band,
       mode_num=0,
       propagation_axis='x',
       source_position=mode_x_pos,
       perpendicular_bounds=(Ly // 2 - 50, Ly // 2 + 50),
       z_bounds=(z_etch - 10, z_box + 10),
       visualize=False,
   )

   # Convert E-only mode to full E+H via short waveguide propagation
   eps_slice = structure.permittivity[:, mode_x_pos, :, :]
   mode_E = mode_info['field']

   mode_full = mode_converter(
       mode_E_field=mode_E,
       freq_band=freq_band,
       permittivity_slice=eps_slice,
       propagation_axis='x',
       visualize=False,
   )

The mode overlap cross-power ``P_mode_cross`` is computed from the mode's E and H fields and used later for normalization in the loss function.

Step 6: Forward Simulation
--------------------------

Run a forward simulation with the initial design to verify the setup before committing to a full optimization run. This costs one GPU simulation (billed at $25/hr on B200) but catches errors in source placement, monitor positions, and layer stack configuration early.

.. code-block:: python

   # Absorber parameters (auto-scaled for resolution)
   abs_params = hwc.get_optimized_absorber_params(
       resolution_nm=dx * 1000,
       wavelength_um=wavelength_um,
       structure_dimensions=(Lx, Ly, Lz),
   )
   abs_widths = abs_params['absorption_widths']
   abs_coeff = abs_params['absorber_coeff']

   # Set up monitors for field extraction
   monitors = hwc.MonitorSet()
   monitors.add(hwc.Monitor(shape=(Lx, Ly, 1), offset=(0, 0, z_dev)), name='xy_device')
   monitors.add(hwc.Monitor(shape=(Lx, 1, Lz), offset=(0, Ly // 2, 0)), name='xz_center')
   monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(Lx // 2, 0, 0)), name='yz_center')
   monitors.add(hwc.Monitor(shape=(1, Ly, Lz), offset=(output_x, 0, 0)), name='wg_output')

   # Run forward simulation
   recipe = structure.extract_recipe()
   fwd_results = hwc.simulate(
       structure_recipe=recipe,
       source_field=source_field,
       source_offset=source_offset,
       freq_band=freq_band,
       monitors_recipe=monitors.recipe,
       absorption_widths=abs_widths,
       absorption_coeff=abs_coeff,
   )

Step 7: Optimization Setup
--------------------------

Three components are needed for inverse design: a structure specification, monitors, and a loss function.

Structure Spec
^^^^^^^^^^^^^^

For forward simulation, we passed a pre-built ``Structure`` with the permittivity already computed. For inverse design, the cloud GPU needs to rebuild the permittivity from a new ``theta`` at every step. The ``structure_spec`` provides the layer stack template (materials, thicknesses, filter parameters) so the GPU knows how to reconstruct permittivity from any theta.

.. code-block:: python

   structure_spec = make_structure_spec(
       layers, DESIGN_LAYER, DENSITY_RADIUS, DENSITY_ALPHA
   )

Loss and Design Monitors
^^^^^^^^^^^^^^^^^^^^^^^^

The loss monitor is placed at the waveguide output, where the output field is sampled for loss computation. The design monitor covers the full XY domain but only the etch layer in Z (approximately 3 pixels thick). Gradients are computed only inside the design monitor volume, which is far more memory-efficient than differentiating through the full 3D grid.

.. code-block:: python

   # Loss monitor: waveguide output
   loss_monitor_shape = (1, Ly, Lz)
   loss_monitor_offset = (abs_widths[0] + 10, 0, 0)

   # Design monitor: etch layer volume
   design_monitor_shape = (Lx, Ly, int(round(h2)))
   design_monitor_offset = (0, 0, z_etch)

Loss Function
^^^^^^^^^^^^^

The mode overlap loss function computes the coupling efficiency between the simulated output field and the target waveguide mode, then negates it. Since the optimizer minimizes the loss, negating the efficiency maximizes coupling into the waveguide.

.. code-block:: python

   def mode_overlap_loss(loss_field):
       """Negative mode coupling efficiency (minimize to maximize coupling)."""
       import jax.numpy as jnp

       alpha = 1.0 / jnp.sqrt(_input_power)
       beta = jnp.sqrt(2.0 / _P_mode_cross)

       f = jnp.mean(loss_field * alpha, axis=2)
       m = jnp.array(_mode_np) * beta

       e0, h0 = m[0, 0:3, 0, :, :], m[0, 3:6, 0, :, :]
       e1, h1 = f[0, 0:3, :, :], f[0, 3:6, :, :]

       cross_e0h1 = jnp.sum(jnp.cross(e0, jnp.conj(h1), axis=0)[0, :, :])
       cross_e1h0 = jnp.sum(jnp.cross(e1, jnp.conj(h0), axis=0)[0, :, :])

       eff = jnp.abs(jnp.real(cross_e0h1 * cross_e1h0)) / 4.0
       return -eff

This function is serialized via ``cloudpickle`` and sent to the GPU. It captures the mode field (~125KB), input power, and mode cross-power in its closure.

Optimizer Config
^^^^^^^^^^^^^^^^

.. code-block:: python

   NUM_STEPS = 5          # increase to 50-100 for production
   LR = 0.01
   GRAD_CLIP = 0.5

   # Waveguide mask forces theta=1 in waveguide region during optimization
   waveguide_mask = np.zeros((theta_Lx, theta_Ly), dtype=np.float32)
   waveguide_mask[:wg_len_theta, theta_Ly // 2 - wg_hw_theta : theta_Ly // 2 + wg_hw_theta] = 1.0

Step 8: Estimate Cost
---------------------

Estimate the GPU cost before running optimization. Each optimization step runs a forward and adjoint simulation, so the cost is roughly 2x a single simulation per step (billed at $25/hr on B200). See :ref:`gpu-cost-estimation` for details on how cost is computed.

.. code-block:: python

   grid_points = Lx * Ly * Lz
   cost = hwc.estimate_cost(
       grid_points=grid_points,
       max_steps=10000,
   )
   # Inverse design: forward + adjoint per step, roughly 2x
   per_step_cost = cost['estimated_cost_usd'] * 2
   total_estimate = per_step_cost * NUM_STEPS
   print(f"Per step: ~${per_step_cost:.3f}")
   print(f"Total ({NUM_STEPS} steps): ~${total_estimate:.2f}")

Step 9: Run Optimization
------------------------

``hwc.run_optimization()`` runs the full optimization loop on a cloud GPU and streams results back after each step. The optimizer uses Adam with cosine learning rate decay. Interrupting the kernel cancels the GPU task immediately, and you are only charged for completed steps.

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
       loss_fn=mode_overlap_loss,
       num_steps=NUM_STEPS,
       learning_rate=LR,
       grad_clip_norm=GRAD_CLIP,
       waveguide_mask=waveguide_mask,
       absorption_widths=abs_widths,
       absorption_coeff=abs_coeff,
   ):
       results.append(step_result)
       step = step_result['step']
       loss = step_result['loss']
       eff_pct = abs(loss) * 100
       print(f"Step {step}: coupling = {eff_pct:.2f}%")

Each ``step_result`` dict contains ``step``, ``loss``, ``theta`` (the updated design), ``grad_max``, and ``step_time``.

Step 10: Results
----------------

After optimization completes, plot the efficiency curve and compare the initial and optimized designs.

.. code-block:: python

   efficiencies = [abs(r['loss']) * 100 for r in results]
   best_idx = int(np.argmax(efficiencies))
   best_theta = results[best_idx]['theta']

   fig, axes = plt.subplots(1, 3, figsize=(18, 5))

   # Efficiency curve
   axes[0].plot(range(1, len(efficiencies) + 1), efficiencies, 'b-o', markersize=3)
   axes[0].set_xlabel('Step')
   axes[0].set_ylabel('Coupling Efficiency (%)')
   axes[0].set_title('Mode Coupling Efficiency')
   axes[0].grid(True, alpha=0.3)

   # Initial vs best theta
   extent = [0, theta_Lx * pixel_size, 0, theta_Ly * pixel_size]
   for ax, th, title in [(axes[1], theta_init, 'Initial'),
                          (axes[2], best_theta, f'Best (step {best_idx + 1})')]:
       ax.imshow(th.T, origin='lower', cmap='PuOr', vmin=0, vmax=1, extent=extent)
       ax.set_xlabel('x (um)')
       ax.set_ylabel('y (um)')
       ax.set_title(title)

   plt.tight_layout()
   plt.show()

Step 11: Verify Optimized Design
--------------------------------

Run a final forward simulation with the best theta to visualize the optimized field distribution and compute the mode coupling efficiency.

.. code-block:: python

   # Build structure with best theta
   density_best = hwc.density(jnp.array(best_theta), radius=DENSITY_RADIUS, alpha=DENSITY_ALPHA)

   layers_best = [
       hwc.Layer(zero,         eps_air,            h_p),
       hwc.Layer(zero,         eps_air,            h0),
       hwc.Layer(zero,         eps_clad,           h1),
       hwc.Layer(density_best, (eps_clad, eps_si), h2),
       hwc.Layer(ones,         eps_si,             h3),
       hwc.Layer(zero,         eps_sio2,           h4),
       hwc.Layer(zero,         eps_si,             h5),
       hwc.Layer(zero,         eps_si,             h_p),
   ]
   structure_best = hwc.create_structure(layers=layers_best, vertical_radius=0)

   # Forward simulation with same monitors
   recipe_best = structure_best.extract_recipe()
   opt_results = hwc.simulate(
       structure_recipe=recipe_best,
       source_field=source_field,
       source_offset=source_offset,
       freq_band=freq_band,
       monitors_recipe=monitors.recipe,
       absorption_widths=abs_widths,
       absorption_coeff=abs_coeff,
   )

Compute the mode coupling efficiency using the mode overlap integral:

.. code-block:: python

   from hyperwave_community.monitors import mode_coupling_efficiency

   eff_lin, eff_dB = mode_coupling_efficiency(
       output_field=jnp.array(wg_field),
       mode_field=jnp.array(mode_full),
       input_power=input_power,
       mode_cross_power=P_mode_cross,
       axis=0,
   )
   eff_pct = float(eff_lin[0]) * 100
   loss_dB = float(eff_dB[0])
   print(f"Mode coupling efficiency: {eff_pct:.2f}% ({loss_dB:.2f} dB)")

Notebook Download
-----------------

`Open in Google Colab <https://colab.research.google.com/github/spinsphotonics/hyperwave-community/blob/main/examples/gc_inverse_design.ipynb>`_
| `Download from GitHub <https://github.com/spinsphotonics/hyperwave-community/blob/main/examples/gc_inverse_design.ipynb>`_

Summary
-------

.. list-table::
   :header-rows: 1

   * - Step
     - Function
     - Cost
   * - 1-3
     - Physical parameters, layer stack, initial design
     - Free (local)
   * - 4
     - ``hwc.create_gaussian_source()``
     - Free (local)
   * - 5
     - ``hwc.create_mode_source()``, ``mode_converter()``
     - Free (local)
   * - 6
     - ``hwc.simulate()`` (verification)
     - Credits, 1 sim ($25/hr)
   * - 7
     - Optimization setup
     - Free (local)
   * - 8
     - ``hwc.estimate_cost()``
     - Free
   * - 9
     - ``hwc.run_optimization()``
     - Credits, 2 sims per step ($25/hr)
   * - 10-11
     - Results and verification
     - Credits, 1 sim ($25/hr)

Next Steps
----------

* Increase ``NUM_STEPS`` to 50-100 for production designs
* Gradually increase ``density_alpha`` during optimization for binarization
* Expand ``freq_band`` to multiple wavelengths (e.g., 1530-1570nm) for broadband optimization
* Try different initial theta values (0.3, 0.5, 0.7, or random) to explore different local optima
* :doc:`api_workflow` - Cloud-based forward simulations
* :doc:`local_workflow` - Step-by-step local workflow with full control
* :doc:`../gpu_options` - GPU performance and cost reference
* :doc:`../convergence` - Early stopping configuration
* :doc:`../api` - Full API reference
