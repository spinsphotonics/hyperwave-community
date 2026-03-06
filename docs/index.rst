Hyperwave Community
===================

Python SDK for GPU-accelerated FDTD photonics simulation via cloud API.

Installation
------------

.. code-block:: bash

   pip install hyperwave-community

Quick Start
-----------

.. code-block:: python

   import hyperwave_community as hwc

   hwc.set_device("auto")   # auto-detects GPU/CPU, installs correct JAX
   hwc.set_verbose(True)

.. note::

   See the full `Quickstart Notebook <https://colab.research.google.com/github/spinsphotonics/hyperwave-community/blob/main/examples/quickstart.ipynb>`_
   for a complete walkthrough.

API Reference
-------------

.. toctree::
   :maxdepth: 2

   configuration
   building
   simulation
   visualization
