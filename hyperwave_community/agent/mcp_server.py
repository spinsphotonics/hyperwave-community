"""Hyperwave MCP server -- mirrors the hwc SDK exactly.

Each tool has the same name and signature as the corresponding
hyperwave_community function. Agents write the same code humans do.

Entry points:
    python -m hyperwave_community.agent.mcp_server   (stdio)
    python -m hyperwave_community.agent               (stdio)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from . import tools

mcp = FastMCP(
    "Hyperwave",
    instructions=(
        "Hyperwave is a photonics inverse design platform. "
        "These tools mirror the hwc (hyperwave_community) Python SDK exactly. "
        "Use them the same way a human would use hwc in a Jupyter notebook."
    ),
)


@mcp.tool()
def solve_waveguide_mode(
    grid: float,
    waveguide_width: float,
    waveguide_height: float,
    n_core: float,
    n_clad: float,
    wavelength: float,
    mode_number: int = 0,
    propagation_axis: int = 0,
    cross_section_size: int = 80,
) -> Dict[str, Any]:
    """Solve for a waveguide eigenmode. Returns n_eff and mode field.

    mode_number: 0=TE0 (fundamental), 1=TM0, 2=TE1, etc.
    """
    return tools.solve_waveguide_mode(
        grid, waveguide_width, waveguide_height,
        n_core, n_clad, wavelength, mode_number,
        propagation_axis, cross_section_size,
    )


@mcp.tool()
def optimize(
    layers: List[Dict[str, Any]],
    theta_b64: Dict[str, str],
    grid: float,
    wavelength: float,
    source_b64: str,
    mode_b64: str,
    phase: str = "freeform",
    n_steps: int = 100,
    initial_design_json: Optional[Dict[str, Any]] = None,
    input_power: float = 1.0,
    beta_init: Optional[float] = None,
    beta_max: Optional[float] = None,
    learning_rate: Optional[float] = None,
    disk_radius: int = 3,
    gpu_type: str = "B200",
) -> Dict[str, Any]:
    """Run inverse design optimization on cloud GPU.

    Standard pipeline: theta -> density -> Layer -> create_structure (for viz),
    then pass layers + theta to optimize for GPU execution.

    Phases (run in order): freeform, binarize, dfm, recovery.
    """
    return tools.optimize(
        layers=layers, theta_b64=theta_b64, grid=grid, wavelength=wavelength,
        source_b64=source_b64, mode_b64=mode_b64, phase=phase, n_steps=n_steps,
        initial_design_json=initial_design_json, input_power=input_power,
        beta_init=beta_init, beta_max=beta_max, learning_rate=learning_rate,
        disk_radius=disk_radius, gpu_type=gpu_type,
    )


@mcp.tool()
def surgery(
    design_json: Dict[str, Any],
    min_feature_size: float = 0.105,
    pixel_size: float = 0.0175,
) -> Dict[str, Any]:
    """Remove small isolated features and fill small holes. CPU, instant, free."""
    return tools.surgery(design_json, min_feature_size, pixel_size)


@mcp.tool()
def check_drc(
    design_json: Dict[str, Any],
    disk_radius: int = 3,
    pixel_size: float = 0.0175,
) -> Dict[str, Any]:
    """Check design rule compliance via morphological operations. CPU, instant, free."""
    return tools.check_drc(design_json, disk_radius, pixel_size)


@mcp.tool()
def export_gds(
    design_json: Dict[str, Any],
    filename: str = "output.gds",
    pixel_size: float = 0.0175,
    layer_name: Optional[str] = None,
    beta: float = 64.0,
    eta: Optional[float] = None,
    smooth_nm: Optional[float] = None,
) -> Dict[str, Any]:
    """Export design to GDSII for fabrication. CPU, instant, free."""
    return tools.export_gds(design_json, filename, pixel_size, layer_name=layer_name, beta=beta, eta=eta, smooth_nm=smooth_nm)


@mcp.tool()
def estimate_cost(
    n_steps: int = 100,
    phase: str = "freeform",
    structure_shape: Optional[List[int]] = None,
    gpu_type: str = "B200",
) -> Dict[str, Any]:
    """Estimate GPU cost before running optimization. Free, no credits charged.

    Pass structure_shape from build_device result for accurate estimates.
    """
    return tools.estimate_cost(n_steps, phase, structure_shape=structure_shape, gpu_type=gpu_type)


@mcp.tool()
def configure_api(
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Configure the Hyperwave API connection with an API key."""
    return tools.configure_api(api_key)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Start the MCP server on stdio transport."""
    api_key = os.environ.get("HYPERWAVE_API_KEY")
    if api_key:
        import hyperwave_community as hwc
        hwc.configure_api(api_key=api_key, validate=False)
    else:
        print(
            "Warning: HYPERWAVE_API_KEY not set. "
            "Local tools (build_device, solve_waveguide_mode) will work. "
            "Cloud tools (optimize) require an API key.",
            file=sys.stderr,
        )

    mcp.run()


if __name__ == "__main__":
    main()
