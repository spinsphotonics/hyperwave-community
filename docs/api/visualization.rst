Visualization
=============

Plotting functions for structures, fields, monitors, and simulation results.

All ``plot_*`` functions run locally using matplotlib. The ``visualize_*`` functions
render on the cloud and return images.

Local Plotting (matplotlib)
---------------------------

.. autofunction:: hyperwave_community.visualization.plot_structure

.. autofunction:: hyperwave_community.visualization.plot_structure_3d

.. autofunction:: hyperwave_community.visualization.plot_fields

.. autofunction:: hyperwave_community.visualization.plot_mode

.. autofunction:: hyperwave_community.visualization.plot_monitors

.. autofunction:: hyperwave_community.visualization.plot_monitor_layout

.. autofunction:: hyperwave_community.visualization.plot_convergence

.. autofunction:: hyperwave_community.visualization.plot_absorption_mask

.. autofunction:: hyperwave_community.visualization.plot_simulation_overview

.. autofunction:: hyperwave_community.visualization.plot_gds

Cloud Rendering
---------------

These functions use the cloud API to render visualizations.

.. autofunction:: hyperwave_community.api_client.visualize_structure

.. autofunction:: hyperwave_community.api_client.visualize_mode_source
