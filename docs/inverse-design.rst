Inverse Design
==============

End-to-end topology optimization pipeline. Build a device, run multi-phase
optimization on cloud GPU, then clean up and export to GDS locally.

Device Configuration
--------------------

.. autofunction:: hyperwave_community.build_device

.. autoclass:: hyperwave_community.DeviceConfig
   :members:

Waveguide Mode Solver
---------------------

.. autofunction:: hyperwave_community.solve_waveguide_mode

Optimization
------------

.. autofunction:: hyperwave_community.optimize

Surgery
-------

.. autofunction:: hyperwave_community.surgery

DRC Check
---------

.. autofunction:: hyperwave_community.check_drc

GDS Export
----------

.. autofunction:: hyperwave_community.export_gds
