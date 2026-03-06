Monitors
========

Classes and functions for placing field monitors and extracting power and field data
from simulation results.

Use :class:`~hyperwave_community.monitors.MonitorSet` to define a collection of monitors,
then pass it to the simulation. After simulation, use the analysis functions to extract
S-parameters, field slices, and power data.

Classes
-------

.. autoclass:: hyperwave_community.monitors.Monitor
   :members:
   :special-members: __init__

.. autoclass:: hyperwave_community.monitors.MonitorSet
   :members:
   :special-members: __init__

Field Analysis Functions
------------------------

.. autofunction:: hyperwave_community.monitors.S_from_slice

.. autofunction:: hyperwave_community.monitors.power_from_a_box

.. autofunction:: hyperwave_community.monitors.get_field_slice

.. autofunction:: hyperwave_community.monitors.get_power_through_plane

.. autofunction:: hyperwave_community.monitors.get_field_intensity

.. autofunction:: hyperwave_community.monitors.get_electric_field_intensity

.. autofunction:: hyperwave_community.monitors.get_magnetic_field_intensity
