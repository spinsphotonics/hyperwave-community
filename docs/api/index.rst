Function Index
==============

Quick reference for all public functions and classes in the ``hyperwave_community`` SDK,
organized by workflow step.

.. list-table:: Building Structures
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community.structure.Layer`
     - Define a layer with permittivity values and thickness
   * - :func:`~hyperwave_community.structure.create_structure`
     - Assemble layers into a 3D simulation structure
   * - :func:`~hyperwave_community.structure.density`
     - Apply density filtering and projection to a design parameter array
   * - :func:`~hyperwave_community.structure.Structure`
     - Data class holding the assembled structure
   * - :func:`~hyperwave_community.structure.recipe_from_params`
     - Create a recipe dict from raw structure parameters

.. list-table:: Sources
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community.simulate.create_mode_source`
     - Create a waveguide mode source via the cloud mode solver
   * - :func:`~hyperwave_community.sources.generate_gaussian_source`
     - Generate a Gaussian pulse source field

.. list-table:: Monitors
   :widths: 40 60
   :header-rows: 0

   * - :class:`~hyperwave_community.monitors.Monitor`
     - Single field monitor at a spatial position
   * - :class:`~hyperwave_community.monitors.MonitorSet`
     - Collection of monitors with waveguide detection
   * - :func:`~hyperwave_community.monitors.S_from_slice`
     - Compute Poynting vector from a field slice
   * - :func:`~hyperwave_community.monitors.power_from_a_box`
     - Compute total power flowing through a box of monitors
   * - :func:`~hyperwave_community.monitors.get_field_slice`
     - Extract a 2D field slice from 3D field data
   * - :func:`~hyperwave_community.monitors.get_power_through_plane`
     - Compute power through a planar cross-section
   * - :func:`~hyperwave_community.monitors.get_field_intensity`
     - Compute total field intensity (E + H)
   * - :func:`~hyperwave_community.monitors.get_electric_field_intensity`
     - Compute electric field intensity
   * - :func:`~hyperwave_community.monitors.get_magnetic_field_intensity`
     - Compute magnetic field intensity

.. list-table:: Absorbing Boundaries
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community.absorption.absorber_params`
     - Compute absorber width and smoothness for a given grid
   * - :func:`~hyperwave_community.absorption.create_absorption_mask`
     - Generate a 3D absorption mask array

.. list-table:: Running Simulations
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community.api_client.configure_api`
     - Set API key, URL, and GPU preferences
   * - :func:`~hyperwave_community.api_client.get_account_info`
     - Retrieve account credits and usage
   * - :func:`~hyperwave_community.api_client.estimate_cost`
     - Estimate simulation cost before running
   * - :func:`~hyperwave_community.api_client.simulate`
     - High-level simulation (structure in, results out)
   * - :func:`~hyperwave_community.api_client.run_simulation`
     - Run FDTD simulation with a prepared recipe
   * - :func:`~hyperwave_community.api_client.build_recipe`
     - Build a simulation recipe on the cloud (CPU, free)
   * - :func:`~hyperwave_community.api_client.build_monitors`
     - Build monitor configuration on the cloud (CPU, free)
   * - :func:`~hyperwave_community.api_client.compute_freq_band`
     - Compute frequency band on the cloud (CPU, free)
   * - :func:`~hyperwave_community.api_client.solve_mode_source`
     - Solve for a waveguide mode on the cloud (CPU, free)
   * - :func:`~hyperwave_community.api_client.get_default_absorber_params`
     - Get default absorber parameters from the cloud
   * - :class:`~hyperwave_community.api_client.ConvergenceConfig`
     - Custom convergence/early-stopping configuration
   * - :func:`~hyperwave_community.api_client.compute_adjoint_gradient`
     - Compute adjoint gradient for inverse design
   * - :func:`~hyperwave_community.api_client.run_optimization`
     - Run a full inverse-design optimization loop

.. list-table:: Analysis
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community.api_client.analyze_transmission`
     - Compute S-parameters and transmission from simulation results
   * - :func:`~hyperwave_community.api_client.get_field_intensity_2d`
     - Extract a 2D field intensity slice
   * - :func:`~hyperwave_community.api_client.compute_poynting_vector`
     - Compute the Poynting vector from 3D field data
   * - :func:`~hyperwave_community.api_client.compute_monitor_power`
     - Compute power at each monitor from simulation results
   * - :func:`~hyperwave_community.api_client.mode_convert`
     - Compute mode overlap / conversion efficiency

.. list-table:: Visualization
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community.visualization.plot_structure`
     - Plot permittivity cross-sections of a structure
   * - :func:`~hyperwave_community.visualization.plot_structure_3d`
     - 3D volume rendering of a structure
   * - :func:`~hyperwave_community.visualization.plot_fields`
     - Plot electric and magnetic field components
   * - :func:`~hyperwave_community.visualization.plot_mode`
     - Plot a waveguide mode profile
   * - :func:`~hyperwave_community.visualization.plot_monitors`
     - Plot monitor transmission spectra
   * - :func:`~hyperwave_community.visualization.plot_monitor_layout`
     - Visualize monitor positions on the structure
   * - :func:`~hyperwave_community.visualization.plot_convergence`
     - Plot simulation convergence over time steps
   * - :func:`~hyperwave_community.visualization.plot_absorption_mask`
     - Plot the absorbing boundary mask
   * - :func:`~hyperwave_community.visualization.plot_simulation_overview`
     - Combined overview: structure, fields, and spectra
   * - :func:`~hyperwave_community.visualization.plot_gds`
     - Plot a GDS layout file
   * - :func:`~hyperwave_community.api_client.visualize_structure`
     - Cloud-rendered structure visualization
   * - :func:`~hyperwave_community.api_client.visualize_mode_source`
     - Cloud-rendered mode source visualization

.. list-table:: Data Import/Export
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community.data_io.component_to_theta`
     - Convert a GDSFactory component to a design array
   * - :func:`~hyperwave_community.data_io.gds_to_theta`
     - Convert a GDS file to a design array
   * - :func:`~hyperwave_community.data_io.generate_gds_from_density`
     - Export a density array to a GDS file
   * - :func:`~hyperwave_community.data_io.export_csv`
     - Export simulation results to CSV files

.. list-table:: Component Library
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community.api_client.list_components`
     - List available built-in photonic components
   * - :func:`~hyperwave_community.api_client.get_component_params`
     - Get default parameters for a component
   * - :func:`~hyperwave_community.api_client.preview_component`
     - Preview a component with given parameters

.. list-table:: Utilities
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community._logging.set_verbose`
     - Enable or disable verbose logging output
   * - :func:`~hyperwave_community._logging.set_debug`
     - Enable or disable debug-level logging
   * - :func:`~hyperwave_community.api_client.encode_array`
     - Encode a NumPy array to base64
   * - :func:`~hyperwave_community.api_client.decode_array`
     - Decode a base64 string to a NumPy array

.. list-table:: Metasurface
   :widths: 40 60
   :header-rows: 0

   * - :func:`~hyperwave_community.metasurface.create_circle_array`
     - Create a single circle pattern on a grid
   * - :func:`~hyperwave_community.metasurface.create_circle_grid`
     - Create a periodic grid of circles
