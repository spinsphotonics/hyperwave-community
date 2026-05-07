Objectives
==========

Composable, safe loss functions for inverse design. Objectives are expression
trees built with operator overloading, serialized to JSON, and evaluated on the
GPU with full JAX autodiff support. No user code runs on the GPU.

Constructors
------------

.. autofunction:: hyperwave_community.objectives.mode_coupling
.. autofunction:: hyperwave_community.objectives.power
.. autofunction:: hyperwave_community.objectives.intensity
.. autofunction:: hyperwave_community.objectives.field
.. autofunction:: hyperwave_community.objectives.const

Combinators
-----------

.. autofunction:: hyperwave_community.objectives.min_of
.. autofunction:: hyperwave_community.objectives.max_of

Math Operations
---------------

.. autofunction:: hyperwave_community.objectives.abs_val
.. autofunction:: hyperwave_community.objectives.log
.. autofunction:: hyperwave_community.objectives.log10
.. autofunction:: hyperwave_community.objectives.sqrt
.. autofunction:: hyperwave_community.objectives.relu
.. autofunction:: hyperwave_community.objectives.real
.. autofunction:: hyperwave_community.objectives.imag
.. autofunction:: hyperwave_community.objectives.conj

Spatial Reductions
------------------

.. autofunction:: hyperwave_community.objectives.sum_spatial
.. autofunction:: hyperwave_community.objectives.mean_spatial

Base Class
----------

.. autoclass:: hyperwave_community.objectives.Objective
   :members: serialize
   :special-members: __add__, __mul__, __neg__, __sub__, __truediv__, __pow__
