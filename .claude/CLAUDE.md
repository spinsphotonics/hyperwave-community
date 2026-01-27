# CLAUDE.md

## Project Overview

Python client library for Hyperwave GPU simulation services. Works with the hyperwave-cloud API gateway.

## Important Notes

### Notebooks
- `examples/getting_started.ipynb` - Customer-facing demo notebook (tracked in git)
- `examples/getting_started_dev.ipynb` - Local dev/testing copy (gitignored)

**When updating notebooks, update BOTH files** to keep them in sync.

### Do NOT commit without explicit permission
Never commit or push changes unless the user explicitly asks.

## Related Repos
- **hyperwave-cloud** (sibling repo): FastAPI gateway that this client talks to
- **hyperwave**: Modal functions for GPU simulations

## Key Files
- `hyperwave_community/api_client.py` - All API endpoint functions
- `hyperwave_community/__init__.py` - Public exports
- `hyperwave_community/simulate.py` - Simulation functions
