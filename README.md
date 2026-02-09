# Hyperwave Community

GPU-accelerated FDTD photonics simulation via cloud API. Design photonic structures locally, run simulations on cloud GPUs (B200, H200, H100, A100), and analyze results with built-in visualization.

## Installation

```bash
pip install hyperwave-community
```

With GDSFactory support:

```bash
pip install "hyperwave-community[gdsfactory]"
```

## Quick Start

[Sign up](https://spinsphotonics.com/signup) to get your API key, then try one of the workflow tutorials:

- [API Workflow](https://colab.research.google.com/github/spinsphotonics/hyperwave-community/blob/main/examples/api_workflow.ipynb) (recommended, runs entirely in Colab)
- [Local Workflow](https://colab.research.google.com/github/spinsphotonics/hyperwave-community/blob/main/examples/local_workflow.ipynb) (structure building on your machine, GPU sim via API)

Both notebooks are Colab-ready. Paste your API key and run.

## How It Works

All structure creation, source generation, and analysis run locally on CPU for free. Only the GPU simulation step uses credits.

| Step | Runs on | Cost |
|------|---------|------|
| Structure design, layer stacking | Local CPU | Free |
| Mode solving, source generation | Local CPU | Free |
| Monitor configuration | Local CPU | Free |
| FDTD simulation | Cloud GPU | Credits |
| Power analysis, visualization | Local CPU | Free |

Use `hwc.estimate_cost()` before running a simulation to check the expected cost.

## Documentation

Full documentation, tutorials, and API reference:

- [Workflows and Tutorials](https://hyperwave-community.readthedocs.io/en/latest/workflows/index.html)
- [API Reference](https://hyperwave-community.readthedocs.io/en/latest/api.html)
- [GPU Options and Pricing](https://hyperwave-community.readthedocs.io/en/latest/gpu_options.html)
- [Installation Guide](https://hyperwave-community.readthedocs.io/en/latest/installation.html)

## License

MIT
