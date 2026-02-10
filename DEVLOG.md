# Hyperwave Community Dev Log

## 2026-02-09

### B200-only GPU consolidation
- Removed `gpu_type` parameter from all SDK functions (`estimate_cost`, `run_simulation`, `simulate`, `simulate_one_shot`, `run_optimization`). GPU is now hardcoded to B200 internally.
- Updated all notebooks (api_workflow, local_workflow, inverse_design_workflow) to remove GPU selection.
- Updated Sphinx docs: gpu_options.rst now reference-only, workflow RSTs updated with $25/hr pricing.
- Updated sources.py defaults from H100 to B200.
- Pricing: 1 credit = $25 = 1 hour of B200 compute.

### Notebook distribution
- Both public Colab notebooks now install directly from GitHub repo main branch.
- Links always point to latest version, replacing the manual Colab upload workflow.

### Infrastructure
- Migrated API gateway from Modal (serverless, cold starts) to Railway (always-on containers).
- SDK default API URL: https://hyperwave-gateway-production.up.railway.app
