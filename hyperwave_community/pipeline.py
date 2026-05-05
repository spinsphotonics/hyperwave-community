"""Pipeline functions for inverse design.

Local (CPU, free): surgery(), check_drc(), export_gds()
Cloud (GPU, credits): optimize()
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hyperwave_community.types import Design, DrcReport, OptimizationResult


# ---------------------------------------------------------------------------
# optimize() - cloud GPU
# ---------------------------------------------------------------------------

def optimize(
    device: Any,
    source: np.ndarray,
    mode: np.ndarray,
    objective: Any = None,
    phase: str = "freeform",
    n_steps: int = 100,
    density_filter_radius: int = 6,
    initial_design: Optional[Design] = None,
    input_power: float = 1.0,
    mode_cross_power: Optional[float] = None,
    beta_init: Optional[float] = None,
    beta_max: Optional[float] = None,
    learning_rate: Optional[float] = None,
    disk_radius: int = 3,
    gap_radius: int = 3,
    fab_eta_lo: Optional[float] = None,
    fab_eta_hi: Optional[float] = None,
    gpu_type: str = "B200",
    api_key: Optional[str] = None,
) -> OptimizationResult:
    """Run an optimization phase on cloud GPU.

    Args:
        device: GridInfo from LayerStack.build(), or a dict with
            design_layers, freq_band, source_offset, recipe_params,
            absorption_widths, absorption_coeff, output_monitor_pos,
            output_monitor_shape, design_xy_range, max_steps, check_every_n.
        source: Source field array.
        mode: Target mode field array.
        objective: Objective expression tree (from hwc.objectives).
            If None, uses mode coupling with the provided mode field.
        phase: "freeform", "binarize", "dfm", or "recovery".
        n_steps: Number of optimization steps.
        density_filter_radius: Conic filter radius in pixels.
        initial_design: Design from a previous phase.
        input_power: Source input power for normalization.
        mode_cross_power: Mode self-overlap power.
        beta_init: Override beta start value.
        beta_max: Override beta end value.
        learning_rate: Override learning rate.
        disk_radius: Morphological disk radius for dfm/recovery.
        gap_radius: Gap disk radius for dfm/recovery.
        fab_eta_lo: Override solid detection eta.
        fab_eta_hi: Override void detection eta.
        gpu_type: GPU type (default "B200").
        api_key: API key (overrides configured key).

    Returns:
        OptimizationResult with .design and .history.
    """
    from hyperwave_community.api_client import (
        _API_CONFIG, encode_array, decode_array, _handle_api_error,
    )
    from hyperwave_community._logging import logger
    import requests
    import json
    import gzip
    import threading
    import time as _time

    effective_api_key = api_key or _API_CONFIG.get('api_key')
    if not effective_api_key:
        raise ValueError(
            "API key required. Call hwc.configure_api(api_key='...') first, "
            "or sign up at spinsphotonics.com")

    API_URL = _API_CONFIG.get('api_url', '')

    # Extract device info
    if hasattr(device, 'design_layers_info'):
        # GridInfo from LayerStack.build()
        design_layers_raw = device.design_layers_info
        freq_band = list(device.freq_band)
        recipe_params = device.recipe_params
        # These need to be provided separately or have defaults
        source_offset = [0, 0, 0]
        absorption_widths = [70, 35, 17]
        absorption_coeff = 0.00489
        output_monitor_pos = [10, 0, 0]
        output_monitor_shape = [1, device.shape[1], device.shape[2]]
        design_xy_range = [[0, device.recipe_params['grid_shape'][0]],
                           [0, device.recipe_params['grid_shape'][1]]]
        max_steps = 20000
        check_every_n = 500
        enforce_symmetry = False
    else:
        # Raw dict
        design_layers_raw = device.get('design_layers', [])
        freq_band = list(device.get('freq_band', [0.1, 0.1, 1]))
        recipe_params = device.get('recipe_params', {})
        source_offset = list(device.get('source_offset', [0, 0, 0]))
        absorption_widths = list(device.get('absorption_widths', [70, 35, 17]))
        absorption_coeff = device.get('absorption_coeff', 0.00489)
        output_monitor_pos = list(device.get('output_monitor_pos', [10, 0, 0]))
        output_monitor_shape = list(device.get('output_monitor_shape', [1, 1, 1]))
        design_xy_range = [list(r) for r in device.get('design_xy_range', [[0, 1], [0, 1]])]
        max_steps = device.get('max_steps', 20000)
        check_every_n = device.get('check_every_n', 500)
        enforce_symmetry = device.get('enforce_symmetry', False)

    # Build design_layers with encoded thetas
    design_layers = []
    if initial_design is not None:
        for dl in design_layers_raw:
            name = dl['name']
            layer = dict(dl)
            if name in initial_design.thetas:
                layer['theta_b64'] = encode_array(np.array(initial_design.thetas[name]))
                layer['theta_shape'] = list(initial_design.thetas[name].shape)
            design_layers.append(layer)
    else:
        for dl in design_layers_raw:
            layer = dict(dl)
            if 'theta' in layer:
                layer['theta_b64'] = encode_array(np.array(layer.pop('theta')))
                layer['theta_shape'] = list(np.array(layer.get('theta_b64', '')).shape) if 'theta' in dl else [0]
            design_layers.append(layer)

    # Serialize objective
    objective_spec = None
    objective_arrays_b64 = None
    if objective is not None:
        spec, arrays = objective.serialize()
        objective_spec = spec
        objective_arrays_b64 = [encode_array(a) for a in arrays]

    # Build request
    request_data = {
        "phase": phase,
        "n_steps": n_steps,
        "design_layers": design_layers,
        "freq_band": freq_band,
        "source_field_b64": encode_array(np.array(source)),
        "source_field_shape": list(np.array(source).shape),
        "source_offset": source_offset,
        "recipe_params": recipe_params,
        "absorption_widths": absorption_widths,
        "absorption_coeff": absorption_coeff,
        "mode_field_b64": encode_array(np.array(mode)),
        "mode_field_shape": list(np.array(mode).shape),
        "input_power": input_power,
        "output_monitor_pos": output_monitor_pos,
        "output_monitor_shape": output_monitor_shape,
        "design_xy_range": design_xy_range,
        "density_filter_radius": density_filter_radius,
        "disk_radius": disk_radius,
        "gap_radius": gap_radius,
        "max_steps": max_steps,
        "check_every_n": check_every_n,
        "enforce_symmetry": enforce_symmetry,
        "gpu_type": gpu_type,
    }

    if mode_cross_power is not None:
        request_data["mode_cross_power"] = mode_cross_power
    if beta_init is not None:
        request_data["beta_init"] = beta_init
    if beta_max is not None:
        request_data["beta_max"] = beta_max
    if learning_rate is not None:
        request_data["learning_rate"] = learning_rate
    if fab_eta_lo is not None:
        request_data["fab_eta_lo"] = fab_eta_lo
    if fab_eta_hi is not None:
        request_data["fab_eta_hi"] = fab_eta_hi
    if objective_spec is not None:
        request_data["objective_spec"] = objective_spec
        request_data["objective_arrays_b64"] = objective_arrays_b64

    # WebSocket transport (same pattern as _run_optimization_ws)
    history = []
    last_thetas = {}

    try:
        import websocket as _ws_lib

        headers = {
            "X-API-Key": effective_api_key,
            "Content-Type": "application/json",
        }
        body = json.dumps(request_data).encode()
        compressed = gzip.compress(body)
        if len(compressed) < len(body):
            headers["Content-Encoding"] = "gzip"
            body = compressed

        logger.info("Starting pipeline optimize (phase=%s, n_steps=%d)...", phase, n_steps)
        t0 = _time.time()
        response = requests.post(
            f"{API_URL}/pipeline_optimize_start",
            data=body, headers=headers, timeout=(60, 300))
        response.raise_for_status()
        session_id = response.json()["session_id"]
        logger.info("  Session started in %.1fs: %s...", _time.time() - t0, session_id[:8])

        ws_url = API_URL.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/inverse_design_ws?session_id={session_id}"

        ws = _ws_lib.create_connection(
            ws_url, header={"X-API-Key": effective_api_key}, timeout=30)
        ws.settimeout(600)

        stop_ping = threading.Event()

        def _pinger():
            while not stop_ping.is_set():
                stop_ping.wait(30)
                if not stop_ping.is_set():
                    try:
                        ws.send(json.dumps({"type": "ping"}))
                    except Exception:
                        break

        ping_thread = threading.Thread(target=_pinger, daemon=True)
        ping_thread.start()

        try:
            while True:
                raw = ws.recv()
                if not raw:
                    continue
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "error":
                    raise RuntimeError(msg.get("message", "Server error"))
                if msg_type == "done":
                    break
                if msg_type == "step":
                    step = msg.get("step", 0)
                    eff = msg.get("efficiency", 0.0)
                    history.append({
                        "step": step,
                        "efficiency": eff,
                        "loss": msg.get("loss", 0.0),
                        "fab_loss": msg.get("fab_loss"),
                        "time": msg.get("step_time", 0.0),
                    })
                    # Decode thetas if present
                    if "theta_b64" in msg and isinstance(msg["theta_b64"], dict):
                        last_thetas = {
                            name: decode_array(b64)
                            for name, b64 in msg["theta_b64"].items()
                        }
                    elif "theta_b64" in msg and isinstance(msg["theta_b64"], str):
                        last_thetas = {"design": decode_array(msg["theta_b64"])}

                    print(f"Step {step:3d}/{n_steps}: efficiency={eff*100:.2f}%  "
                          f"time={msg.get('step_time', 0):.0f}s", flush=True)

        finally:
            stop_ping.set()
            ping_thread.join(timeout=2)
            try:
                ws.close()
            except Exception:
                pass

    except ImportError:
        raise ImportError(
            "websocket-client required for optimize(). "
            "Install with: pip install websocket-client")
    except requests.HTTPError as e:
        _handle_api_error(e, "pipeline optimize")
        raise

    # Build result
    if not last_thetas and initial_design is not None:
        last_thetas = initial_design.thetas

    final_eff = history[-1]["efficiency"] if history else 0.0
    final_step = history[-1]["step"] if history else 0

    design = Design(
        thetas=last_thetas,
        density_filter_radius=density_filter_radius,
        efficiency=final_eff,
        phase=phase,
        step=final_step,
    )

    efficiencies = [h["efficiency"] for h in history]
    best_idx = int(np.argmax(efficiencies)) if efficiencies else 0
    best_eff = efficiencies[best_idx] if efficiencies else 0.0
    best_step = history[best_idx]["step"] if history else 0

    schedule_config = {
        "phase": phase,
        "density_filter_radius": density_filter_radius,
        "disk_radius": disk_radius,
    }
    if beta_init is not None:
        schedule_config["beta_init"] = beta_init
    if beta_max is not None:
        schedule_config["beta_max"] = beta_max
    if learning_rate is not None:
        schedule_config["learning_rate"] = learning_rate

    return OptimizationResult(
        design=design,
        history=history,
        phase=phase,
        n_steps=len(history),
        best_efficiency=best_eff,
        best_step=best_step,
        schedule_config=schedule_config,
        n_steps_planned=n_steps,
    )


# ---------------------------------------------------------------------------
# surgery()
# ---------------------------------------------------------------------------

def surgery(
    design: Design,
    min_feature_size: float = 0.105,
    pixel_size: float = 0.0175,
) -> Design:
    """Remove small features and fill small holes.

    Non-differentiable cleanup step. Runs locally on CPU, no credits.

    Args:
        design: Design to clean up.
        min_feature_size: Minimum feature size in um.
        pixel_size: Physical pixel size in um.

    Returns:
        New Design with cleaned thetas and surgery metadata.
    """
    from scipy import ndimage
    from hyperwave_community.structure import density

    min_area_px = int(np.pi * (min_feature_size / pixel_size / 2) ** 2)
    min_area_px = max(min_area_px, 1)

    new_thetas = {}
    total_removed = 0
    total_filled = 0

    for name, theta in design.thetas.items():
        mask = design.design_mask(name)

        import jax.numpy as jnp
        d = np.array(density(jnp.array(theta), radius=float(design.density_filter_radius)))
        binary = (d >= 0.5).astype(np.int32)
        binary_design = binary.copy()
        binary_design[~mask] = 0

        # Count and remove small solid islands
        solid_labeled, n_solid = ndimage.label(binary_design & mask)
        solid_sizes = ndimage.sum(
            np.ones_like(binary_design), solid_labeled, range(1, n_solid + 1))
        removed = sum(1 for s in solid_sizes if s < min_area_px)

        patched = binary_design.copy()
        for i, size in enumerate(solid_sizes):
            if size < min_area_px:
                patched[solid_labeled == (i + 1)] = 0

        # Count and fill small holes
        void_region = (1 - binary_design) & mask
        void_labeled, n_void = ndimage.label(void_region)
        void_sizes = ndimage.sum(
            np.ones_like(binary_design), void_labeled, range(1, n_void + 1))
        filled = sum(1 for s in void_sizes if s < min_area_px)

        for i, size in enumerate(void_sizes):
            if size < min_area_px:
                patched[void_labeled == (i + 1)] = 1

        # Replace design region with patched binary
        theta_clean = np.array(theta, dtype=np.float32, copy=True)
        theta_clean[mask] = patched[mask].astype(np.float32)

        new_thetas[name] = theta_clean
        total_removed += removed
        total_filled += filled

    return Design(
        thetas=new_thetas,
        density_filter_radius=design.density_filter_radius,
        efficiency=design.efficiency,
        phase="surgery",
        step=design.step,
        removed_islands=total_removed,
        filled_holes=total_filled,
    )


# ---------------------------------------------------------------------------
# check_drc()
# ---------------------------------------------------------------------------

def check_drc(
    design: Design,
    disk_radius: int = 3,
    pixel_size: float = 0.0175,
) -> DrcReport:
    """Run design rule check via morphological opening.

    Checks minimum feature size (CD) and minimum gap. Runs locally
    on CPU, no credits charged.

    Args:
        design: Design to check.
        disk_radius: Morphological disk radius in pixels.
        pixel_size: Physical pixel size in um.

    Returns:
        DrcReport with violation counts, percentages, and pass/fail.
    """
    from skimage.morphology import disk, opening
    from hyperwave_community.structure import density

    name = design.layer_names[0]
    theta = design.thetas[name]
    mask = design.design_mask(name)

    import jax.numpy as jnp
    d = np.array(density(jnp.array(theta), radius=float(design.density_filter_radius)))
    binary = (d >= 0.5).astype(bool)
    binary_design = binary & mask
    void_design = (~binary) & mask

    design_pixels = int(mask.sum())
    d_vals = d[mask]
    binarization_score = float(1.0 - np.mean(4 * d_vals * (1 - d_vals)))

    selem = disk(disk_radius)
    solid_opened = opening(binary_design, selem)
    cd_violations = int((binary_design & ~solid_opened).sum())

    void_opened = opening(void_design, selem)
    gap_violations = int((void_design & ~void_opened).sum())

    min_feature_nm = (2 * disk_radius + 1) * pixel_size * 1000
    cd_pct = 100.0 * cd_violations / design_pixels if design_pixels > 0 else 0.0
    gap_pct = 100.0 * gap_violations / design_pixels if design_pixels > 0 else 0.0

    return DrcReport(
        cd_violations=cd_violations,
        cd_pct=cd_pct,
        gap_violations=gap_violations,
        gap_pct=gap_pct,
        binarization_score=binarization_score,
        disk_radius=disk_radius,
        min_feature_nm=min_feature_nm,
        min_gap_nm=min_feature_nm,
        design_pixels=design_pixels,
    )


# ---------------------------------------------------------------------------
# export_gds()
# ---------------------------------------------------------------------------

def export_gds(
    design: Design,
    filename: str = "output.gds",
    layer: Tuple[int, int] = (1, 0),
    pixel_size: float = 0.0175,
) -> str:
    """Export design to GDSII file.

    Runs locally on CPU, no credits charged.

    Args:
        design: Design to export.
        filename: Output GDS filename.
        layer: GDS layer tuple (layer_number, datatype).
        pixel_size: Physical pixel size in um.

    Returns:
        Absolute path to the generated GDS file.
    """
    from hyperwave_community.structure import density
    from hyperwave_community.data_io import generate_gds_from_density

    theta = design.theta
    import jax.numpy as jnp
    d = np.array(density(jnp.array(theta), radius=float(design.density_filter_radius)))

    return generate_gds_from_density(
        density_array=d,
        level=0.5,
        output_filename=filename,
        resolution=pixel_size,
    )
