Installation
============

Requirements
------------

* Python 3.9 or higher
* Internet connection (for API calls)

The easiest way to get started is with `Google Colab <https://colab.research.google.com>`_, which provides a free hosted notebook environment with all dependencies pre-installed. All of our tutorials include "Open in Colab" links that let you start running simulations immediately.

Install in Google Colab
-----------------------

Run this in the first cell of your Colab notebook:

.. code-block:: python

   %pip install git+https://github.com/spinsphotonics/hyperwave-community.git -q

Install Locally
---------------

If you prefer to work on your own machine:

.. code-block:: bash

   pip install git+https://github.com/spinsphotonics/hyperwave-community.git

.. note::
   You may need to restart the kernel after installation in Jupyter.

Dependencies
------------

The following packages are installed automatically:

* **jax** / **jaxlib** - GPU-accelerated array operations
* **numpy** / **scipy** - Numerical computing
* **matplotlib** - Visualization
* **gdsfactory** / **gdstk** - Photonic component library and GDS handling
* **requests** - API communication
* **scikit-image** - Image processing for density filtering
* **cloudpickle** - Serialization for API transport

Get an API Key
--------------

To run simulations, you need an API key from `spinsphotonics.com <https://spinsphotonics.com>`_.

1. Sign up at `spinsphotonics.com <https://spinsphotonics.com>`_
2. Find your API key in the dashboard under Settings
3. Configure the SDK:

.. code-block:: python

   import hyperwave_community as hwc

   hwc.configure_api(api_key="your-api-key-here")
   hwc.get_account_info()  # Verify your key and check credits

.. tip::
   If you are using Google Colab, store your API key in **Colab Secrets** instead of pasting it directly in your notebook. See :doc:`getting_started/colab_secrets` for setup instructions.

Next Steps
----------

Choose a workflow to get started:

* :doc:`workflows/index` - Overview of available workflows
* :doc:`workflows/local_workflow` - Step-by-step tutorial with full control
* :doc:`workflows/api_workflow` - Single-call workflow for integration into existing systems
