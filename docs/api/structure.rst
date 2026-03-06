Building Structures
===================

Functions for defining layers, applying density filtering, and assembling simulation structures.

The typical workflow is:

1. Create a design parameter array (``theta``) representing your device geometry.
2. Apply :func:`~hyperwave_community.structure.density` to filter and project into a binary structure.
3. Wrap each layer in a :class:`~hyperwave_community.structure.Layer` with material properties.
4. Call :func:`~hyperwave_community.structure.create_structure` to assemble the full 3D structure.

Classes
-------

.. autoclass:: hyperwave_community.structure.Layer
   :members:
   :special-members: __init__

.. autoclass:: hyperwave_community.structure.Structure
   :members:
   :exclude-members: __init__, permittivity, conductivity, layers_info, construction_params, metadata

Functions
---------

.. autofunction:: hyperwave_community.structure.density

.. autofunction:: hyperwave_community.structure.create_structure

.. autofunction:: hyperwave_community.structure.recipe_from_params
