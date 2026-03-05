User Guide
==========

This section covers the key concepts and workflows for designing and simulating
photonic devices with Hyperwave Community.

.. toctree::
   :maxdepth: 2

   structure_modeling
   mode_solving
   simulation_fundamentals
   performance
   troubleshooting

Structure Modeling
~~~~~~~~~~~~~~~~~~

Learn how to define photonic device geometries:

* **Layers**: Multi-layer material stacks with density patterns
* **Density Filtering**: Minimum feature size constraints for fabricability
* **Permittivity**: Automatic 3D permittivity construction from layers
* **Visualization**: Inspect structure cross-sections before simulation

Mode Solving
~~~~~~~~~~~~

Understand waveguide modes and source injection:

* **Eigenmode Solver**: Computing propagation constants and field profiles
* **Mode Sources**: Injecting waveguide modes into simulations
* **Mode Converter**: Converting E-only modes to full E+H fields
* **Gaussian Sources**: Free-space beam excitation

Simulation Fundamentals
~~~~~~~~~~~~~~~~~~~~~~~

Run GPU-accelerated FDTD simulations via the cloud API:

* **Cloud Workflow**: Submitting simulations to GPU servers
* **Monitors**: Extracting field data at specific locations
* **Power Analysis**: Poynting flux and transmission calculations
* **Convergence**: Early stopping and convergence detection

Performance
~~~~~~~~~~~

Optimize simulation speed and accuracy:

* **GPU Selection**: Choosing the right GPU tier
* **Grid Resolution**: Balancing accuracy and cost
* **Convergence Settings**: Tuning early stopping parameters
* **Cost Estimation**: Predicting simulation credits

Troubleshooting
~~~~~~~~~~~~~~~

Common issues and solutions:

* **API Connection**: Authentication and network errors
* **Shape Mismatches**: Array dimension problems
* **Convergence Issues**: Simulations that don't settle
* **Memory Limits**: Handling large structures
