# Hyperwave Community Dev Log

## 2026-02-12: Merge Inverse Design System

- Merged Jim's feb-inverse-design-notebook branch
- Added WebSocket-based optimization streaming (WS-first, SSE fallback)
- New SDK functions: run_optimization, compute_adjoint_gradient, mode_convert, generate_gaussian_source, recipe_from_params
- New notebook: examples/inverse_design_workflow.ipynb (grating coupler inverse design)
- Added gateway_url parameter to configure_api() for split routing flexibility
- Removed solve.py (local FDTD) in favor of cloud-only mode conversion
- GPU defaults: B200 across all endpoints
- New dependency: websocket-client>=1.6.0
- New unit tests: absorption, data_io, monitors, structure

## Initial State (pre 2026-02-09)

Python SDK + Jupyter notebooks + Sphinx docs for the HyperWave simulation platform.

**Repo role:** Client-facing SDK. Users install this package, call `run_simulation()` or `compute_adjoint_gradient()`, and the SDK sends HTTP requests to the hyperwave-cloud gateway (Railway). Gateway handles auth/billing via firebase-first, dispatches GPU work to Modal.

**What existed before today:**
- SDK (`api_client.py`): `run_simulation`, `simulate`, `simulate_one_shot`, `run_optimization`, `compute_adjoint_gradient`, `estimate_cost`, `get_optimized_absorber_params`
- Granular workflow functions: `load_component`, `build_recipe_from_theta`, `build_monitors_local`
- 3 example notebooks: api_workflow, local_workflow, inverse_design_workflow (grating coupler)
- Sphinx docs on ReadTheDocs with workflow tutorials, GPU options reference, Colab secrets guide
- API base URL migrated through Render -> Cloud Run -> Modal -> Railway
- gdsfactory integrated as core dependency for photonic component definitions
- Gzip support for large simulation responses
- Mode solver extracted from private hyperwave repo into this package

## 2026-02-09

### B200-only GPU consolidation
- Removed `gpu_type` parameter from all SDK functions. GPU hardcoded to B200 internally.
- Cleaned stale gpu_type refs from dev notebooks, sources.py, docs.
- Updated pricing in Sphinx docs: 1 credit = $25 = 1 hour B200.

### Infrastructure
- API base URL now points to Railway gateway (was Modal).
- Both public Colab notebooks install directly from GitHub main branch.

### Merged feb_mvp to main
- All changes above deployed via merge to main. Railway auto-deploys from main.

## 2026-02-10

### Notebook install string cleanup
- Simplified install string in api_workflow.ipynb and local_workflow.ipynb from `%pip install "hyperwave-community[gds] @ git+https://..." --no-cache-dir -q` to `%pip install git+https://github.com/spinsphotonics/hyperwave-community.git -q`.
- The `[gds]` extra never existed in setup.py (gdsfactory is in base install_requires).

### Merged feb_mvp to main
