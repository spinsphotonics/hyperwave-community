Data Import/Export
==================

Functions for converting between GDS layout files, GDSFactory components, and the
design parameter arrays (theta) used by the simulator. Also includes CSV export for
sharing results.

GDS Import
----------

.. autofunction:: hyperwave_community.data_io.component_to_theta

.. autofunction:: hyperwave_community.data_io.gds_to_theta

GDS Export
----------

.. autofunction:: hyperwave_community.data_io.generate_gds_from_density

CSV Export
----------

.. autofunction:: hyperwave_community.data_io.export_csv
