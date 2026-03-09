Set Up Simulation
=================

Functions for converting GDS layouts to simulation-ready 3D structures
with sources, monitors, and absorbing boundaries.

Structure
---------

.. autofunction:: hyperwave_community.component_to_theta
.. autofunction:: hyperwave_community.gds_to_theta

.. autofunction:: hyperwave_community.density

.. autoclass:: hyperwave_community.Layer
   :members:
   :undoc-members:

.. autofunction:: hyperwave_community.create_structure

Absorption
----------

.. autofunction:: hyperwave_community.get_optimized_absorber_params

.. autofunction:: hyperwave_community.create_absorption_mask

Source
------

.. autofunction:: hyperwave_community.create_mode_source

Monitors
--------

.. autofunction:: hyperwave_community.create_port_monitors

.. autoclass:: hyperwave_community.Monitor
   :members:

.. autoclass:: hyperwave_community.MonitorSet
   :members:
