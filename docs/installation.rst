:tocdepth: 1

Installation
============

Requirements
------------

- Python 3.9 or later
- A Google account (for cloud GPU access)

Install from PyPI
-----------------

.. code-block:: bash

   pip install hyperwave-community

Device Setup
------------

After installing, configure your compute device before running any simulations:

.. code-block:: python

   import hyperwave_community as hwc

   hwc.set_device("auto")   # auto-detects GPU/CPU, installs correct JAX

.. note::

   ``set_device("auto")`` will automatically detect available hardware and install
   the appropriate JAX backend (GPU or CPU). Call this once at the start of your
   script or notebook.

Next Steps
----------

See the full `Quickstart Notebook <https://colab.research.google.com/github/spinsphotonics/hyperwave-community/blob/main/examples/quickstart.ipynb>`_
for a complete walkthrough of your first simulation.
