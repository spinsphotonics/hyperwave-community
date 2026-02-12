Installation
============

Requirements
------------

* Python 3.9 or higher
* Internet connection (for API calls)

Install from GitHub
-------------------

.. code-block:: bash

   pip install git+https://github.com/spinsphotonics/hyperwave-community.git

Install in Jupyter/Colab
------------------------

In a Jupyter notebook or Google Colab, use:

.. code-block:: python

   %pip install git+https://github.com/spinsphotonics/hyperwave-community.git -q

.. note::
   You may need to restart the kernel after installation.

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

1. Create an account at spinsphotonics.com
2. Navigate to your dashboard to find your API key
3. Configure the SDK:

.. code-block:: python

   import hyperwave_community as hwc

   hwc.configure_api(api_key="your-api-key-here")
   hwc.get_account_info()  # Verify your key and check credits

Next Steps
----------

Choose a workflow to get started:

* :doc:`workflows/index` - Overview of available workflows
* :doc:`workflows/local_workflow` - Step-by-step tutorial with full control
* :doc:`workflows/api_workflow` - Single-call workflow for integration into existing systems
