Sources
=======

Functions for creating electromagnetic sources for FDTD simulations.

Two source types are available:

- **Mode source**: Excites a specific waveguide mode. Uses the cloud mode solver via
  :func:`~hyperwave_community.simulate.create_mode_source`.
- **Gaussian source**: A broadband Gaussian pulse, computed locally via
  :func:`~hyperwave_community.sources.generate_gaussian_source`.

Functions
---------

.. autofunction:: hyperwave_community.simulate.create_mode_source

.. autofunction:: hyperwave_community.sources.generate_gaussian_source
