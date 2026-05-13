"""Hyperwave agent package -- MCP server mirroring the hwc SDK.

Entry points:
    python -m hyperwave_community.agent.mcp_server
    python -m hyperwave_community.agent
"""

from .tools import (
    solve_waveguide_mode,
    optimize,
    surgery,
    check_drc,
    export_gds,
    estimate_cost,
    configure_api,
)
