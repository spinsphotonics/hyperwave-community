Absorbing Boundaries
====================

Functions for computing adiabatic absorbing boundary parameters and generating
absorption masks for the simulation domain.

Absorbing boundaries prevent reflections at the edges of the simulation grid.
Use :func:`~hyperwave_community.absorption.absorber_params` to compute the width and
smoothness, then :func:`~hyperwave_community.absorption.create_absorption_mask` to
generate the 3D mask array.

Functions
---------

.. autofunction:: hyperwave_community.absorption.absorber_params

.. autofunction:: hyperwave_community.absorption.create_absorption_mask
