"""MCP tool functions that mirror the hwc (hyperwave_community) SDK exactly.

Each function here has the same name and accepts the same parameters as the
corresponding hwc function. The only difference is serialization: MCP tools
accept/return JSON-serializable types, so DeviceConfig, Design, and numpy
arrays are represented as dicts and base64 strings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _serialize_device(device) -> Dict[str, Any]:
    """Convert DeviceConfig to a JSON-serializable dict."""
    from hyperwave_community.api_client import encode_array

    design_info = []
    for d in device.design_layers_info:
        info = {k: v for k, v in d.items() if k not in ("theta", "waveguide_mask")}
        if "theta" in d:
            info["theta_shape"] = list(np.array(d["theta"]).shape)
            info["theta_b64"] = encode_array(np.array(d["theta"]))
        design_info.append(info)

    return {
        "shape": list(device.shape),
        "grid": device.grid,
        "pixel_size": device.pixel_size,
        "vertical_radius": device.vertical_radius,
        "freq_band": [float(x) for x in device.freq_band],
        "design_layers_info": design_info,
    }


def _deserialize_design(design_json: Dict[str, Any]):
    """Convert a JSON dict back to a Design object."""
    from hyperwave_community.api_client import decode_array
    from hyperwave_community.types import Design

    thetas = {}
    for name, b64 in design_json.get("thetas", {}).items():
        thetas[name] = decode_array(b64)

    return Design(
        thetas=thetas,
        density_radii=design_json.get("density_radii", {}),
        efficiency=design_json.get("efficiency", 0.0),
        phase=design_json.get("phase", ""),
        step=design_json.get("step", 0),
    )


def _serialize_design(design) -> Dict[str, Any]:
    """Convert a Design object to a JSON-serializable dict."""
    from hyperwave_community.api_client import encode_array
    return {
        "thetas": {name: encode_array(np.array(t)) for name, t in design.thetas.items()},
        "density_radii": design.density_radii,
        "efficiency": float(design.efficiency) if design.efficiency else 0.0,
        "phase": design.phase,
        "step": design.step,
        "removed_islands": getattr(design, "removed_islands", 0),
        "filled_holes": getattr(design, "filled_holes", 0),
    }


# build_device removed from public API.
# Users define theta, layers, and pass them directly to optimize().


# ---------------------------------------------------------------------------
# solve_waveguide_mode - same as hwc.solve_waveguide_mode()
# ---------------------------------------------------------------------------

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
    """Solve for a waveguide eigenmode.

    Computes the electromagnetic mode profile and effective index for a
    rectangular waveguide cross-section.

    Args:
        grid: grid spacing in um
        waveguide_width: waveguide width in um
        waveguide_height: waveguide core height in um
        n_core: core refractive index
        n_clad: cladding refractive index
        wavelength: operating wavelength in um
        mode_number: 0=TE0, 1=TM0, 2=TE1, etc.
        propagation_axis: 0=x (default)
        cross_section_size: mode solver grid size in pixels

    Returns:
        Dict with: n_eff, mode_field_shape, mode_field_b64
    """
    from hyperwave_community.waveguide_mode import solve_waveguide_mode as _solve
    from hyperwave_community.api_client import encode_array

    mode_field, n_eff = _solve(
        grid=grid,
        waveguide_width=waveguide_width,
        waveguide_height=waveguide_height,
        n_core=n_core,
        n_clad=n_clad,
        wavelength=wavelength,
        mode_number=mode_number,
        propagation_axis=propagation_axis,
        cross_section_size=cross_section_size,
    )

    mode_np = np.array(mode_field)
    return {
        "status": "ok",
        "n_eff": float(n_eff),
        "mode_field_shape": list(mode_np.shape),
        "mode_field_b64": encode_array(mode_np),
    }


# ---------------------------------------------------------------------------
# optimize - same as hwc.optimize()
# ---------------------------------------------------------------------------

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

    Standard pipeline: define layers and theta, then call optimize.
    Each step runs a forward and adjoint FDTD simulation.

    Args:
        layers: Layer stack as list of dicts (name, thickness, index,
            and design=True with density_radius for design layers).
        theta_b64: Per-layer initial theta as base64-encoded numpy arrays.
            Keys are design layer names, e.g. {"etch": "<base64>"}.
        grid: FDTD grid spacing in um.
        wavelength: Operating wavelength in um.
        source_b64: base64-encoded source field array.
        mode_b64: base64-encoded mode field array.
        phase: "freeform", "binarize", "dfm", or "recovery".
        n_steps: number of optimization steps.
        initial_design_json: serialized Design from a previous phase.
        input_power: input power normalization.
        beta_init: binarization start sharpness (binarize phase).
        beta_max: binarization end sharpness (binarize phase).
        learning_rate: override default LR.
        disk_radius: morphological disk radius for DFM.
        gpu_type: GPU type ("B200", "H100", "A100_80GB").

    Returns:
        Dict with: design (serialized), history, best_efficiency, phase.
    """
    from hyperwave_community.api_client import decode_array
    from hyperwave_community.pipeline import optimize as _optimize

    source = decode_array(source_b64)
    mode = decode_array(mode_b64)

    theta = {name: decode_array(b64) for name, b64 in theta_b64.items()}

    initial_design = None
    if initial_design_json:
        initial_design = _deserialize_design(initial_design_json)

    result = _optimize(
        layers=layers,
        theta=theta,
        grid=grid,
        wavelength=wavelength,
        source=source,
        mode=mode,
        phase=phase,
        n_steps=n_steps,
        initial_design=initial_design,
        input_power=input_power,
        beta_init=beta_init,
        beta_max=beta_max,
        learning_rate=learning_rate,
        disk_radius=disk_radius,
        gpu_type=gpu_type,
    )

    return {
        "status": "ok",
        "design": _serialize_design(result.design),
        "history": result.history,
        "best_efficiency": float(result.best_efficiency),
        "phase": result.phase,
        "n_steps": result.n_steps,
    }


# ---------------------------------------------------------------------------
# surgery - same as hwc.surgery()
# ---------------------------------------------------------------------------

def surgery(
    design_json: Dict[str, Any],
    min_feature_size: float = 0.105,
    pixel_size: float = 0.0175,
) -> Dict[str, Any]:
    """Remove small isolated features and fill small holes.

    Args:
        design_json: serialized Design from optimize()
        min_feature_size: minimum feature size in um
        pixel_size: pixel size in um

    Returns:
        Dict with: design (serialized), removed_islands, filled_holes
    """
    from hyperwave_community.pipeline import surgery as _surgery

    design = _deserialize_design(design_json)
    cleaned = _surgery(
        design=design,
        min_feature_size=min_feature_size,
        pixel_size=pixel_size,
    )

    return {
        "status": "ok",
        "design": _serialize_design(cleaned),
        "removed_islands": getattr(cleaned, "removed_islands", 0),
        "filled_holes": getattr(cleaned, "filled_holes", 0),
    }


# ---------------------------------------------------------------------------
# check_drc - same as hwc.check_drc()
# ---------------------------------------------------------------------------

def check_drc(
    design_json: Dict[str, Any],
    disk_radius: int = 3,
    pixel_size: float = 0.0175,
) -> Dict[str, Any]:
    """Check design rule compliance via morphological operations.

    Args:
        design_json: serialized Design from optimize() or surgery()
        disk_radius: morphological disk radius in pixels
        pixel_size: pixel size in um

    Returns:
        Dict with per-layer DRC results: cd_violations, gap_violations, etc.
    """
    from hyperwave_community.pipeline import check_drc as _check_drc

    design = _deserialize_design(design_json)
    report = _check_drc(
        design=design,
        disk_radius=disk_radius,
        pixel_size=pixel_size,
    )

    if isinstance(report, dict):
        return {
            "status": "ok",
            "layers": {
                name: {
                    "cd_violations": r.cd_violations,
                    "cd_pct": float(r.cd_pct),
                    "gap_violations": r.gap_violations,
                    "gap_pct": float(r.gap_pct),
                    "binarization_score": float(r.binarization_score),
                    "passed": r.cd_pct < 1.0 and r.gap_pct < 1.0,
                }
                for name, r in report.items()
            },
        }

    return {
        "status": "ok",
        "cd_violations": report.cd_violations,
        "cd_pct": float(report.cd_pct),
        "gap_violations": report.gap_violations,
        "gap_pct": float(report.gap_pct),
        "binarization_score": float(report.binarization_score),
        "passed": report.cd_pct < 1.0 and report.gap_pct < 1.0,
    }


# ---------------------------------------------------------------------------
# export_gds - same as hwc.export_gds()
# ---------------------------------------------------------------------------

def export_gds(
    design_json: Dict[str, Any],
    filename: str = "output.gds",
    pixel_size: float = 0.0175,
    layer_name: Optional[str] = None,
    beta: float = 64.0,
    eta: Optional[float] = None,
    smooth_nm: Optional[float] = None,
) -> Dict[str, Any]:
    """Export design to GDSII format for fabrication.

    Args:
        design_json: serialized Design
        filename: output filename
        pixel_size: pixel size in um
        layer_name: which design layer to export (None = first)
        beta: Heaviside projection sharpness
        eta: projection threshold
        smooth_nm: optional KLayout smoothing radius in nm

    Returns:
        Dict with: filename, n_polygons
    """
    from hyperwave_community.pipeline import export_gds as _export_gds

    design = _deserialize_design(design_json)
    gds_path = _export_gds(
        design=design,
        filename=filename,
        pixel_size=pixel_size,
        layer_name=layer_name,
        beta=beta,
        eta=eta,
        smooth_nm=smooth_nm,
    )

    return {
        "status": "ok",
        "filename": str(gds_path),
    }


# ---------------------------------------------------------------------------
# estimate_cost - same as hwc.estimate_cost()
# ---------------------------------------------------------------------------

def estimate_cost(
    n_steps: int = 100,
    phase: str = "freeform",
    grid_points: Optional[int] = None,
    structure_shape: Optional[List[int]] = None,
    gpu_type: str = "B200",
) -> Dict[str, Any]:
    """Estimate the GPU cost of a simulation or optimization run.

    Provide either grid_points or structure_shape for accurate estimates.

    Args:
        n_steps: number of steps
        phase: pipeline phase
        grid_points: total grid cells (e.g. 200*100*123)
        structure_shape: [nx, ny, nz] from build_device result
        gpu_type: GPU type

    Returns:
        Dict with: estimated_cost_usd, estimated_credits, etc.
    """
    import hyperwave_community as hwc

    if grid_points is None and structure_shape is not None:
        grid_points = 1
        for d in structure_shape:
            grid_points *= d

    result = hwc.estimate_cost(
        max_steps=n_steps,
        gpu_type=gpu_type,
        grid_points=grid_points,
    )
    if result:
        return {"status": "ok", **result}
    return {"status": "ok", "estimated_cost_usd": 0.0, "note": "Provide grid_points or structure_shape for accurate estimate"}


# ---------------------------------------------------------------------------
# configure_api - same as hwc.configure_api()
# ---------------------------------------------------------------------------

def configure_api(
    api_key: Optional[str] = None,
    validate: bool = False,
) -> Dict[str, Any]:
    """Configure the Hyperwave API connection.

    Args:
        api_key: API key for authentication
        validate: whether to validate the key against the server

    Returns:
        Dict with account info if validate=True
    """
    import hyperwave_community as hwc
    result = hwc.configure_api(api_key=api_key, validate=validate)
    if result:
        return {"status": "ok", **result}
    return {"status": "ok", "message": "API configured"}
