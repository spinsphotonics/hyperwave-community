Types and Checkpoints
=====================

Data types that flow between pipeline stages, and checkpoint utilities
for saving and resuming optimization runs.

Types
-----

.. autoclass:: hyperwave_community.Design
   :members:

.. autoclass:: hyperwave_community.OptimizationResult
   :members:

.. autoclass:: hyperwave_community.DrcReport
   :members:

Checkpoints
-----------

.. autofunction:: hyperwave_community.save_checkpoint
.. autofunction:: hyperwave_community.load_checkpoint
.. autofunction:: hyperwave_community.list_checkpoints
