"""Pipeline functions for inverse design.

Local (CPU, free): surgery(), check_drc(), export_gds()
Cloud (GPU, credits): optimize()
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from hyperwave_community.types import Design, DrcReport, OptimizationResult


def compute_mode_cross_power(mode_field: np.ndarray) -> float:
    """Compute mode self-overlap power from a mode field array.

    Args:
        mode_field: Mode field with shape (n_freq, 6, ...) or (n_freq, 3, ...).
            If 3 components (E only), returns 1.0 as fallback.

    Returns:
        Scalar mode cross power |Re(sum(E x H*)_x)|.
    """
    if mode_field.shape[1] < 6:
        return 1.0
    e = mode_field[0, :3]
    h = mode_field[0, 3:]
    cross = np.cross(e, np.conj(h), axis=0)
    return float(np.abs(np.real(np.sum(cross[0]))))


# ---------------------------------------------------------------------------
# optimize() - cloud GPU
# ---------------------------------------------------------------------------

def optimize(
    layers: Any = None,
    theta: Optional[Dict[str, np.ndarray]] = None,
    grid: Optional[float] = None,
    wavelength: Optional[float] = None,
    source: Optional[np.ndarray] = None,
    mode: Optional[np.ndarray] = None,
    objective: Any = None,
    phase: str = "freeform",
    n_steps: int = 100,
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
    device: Any = None,
) -> OptimizationResult:
    """Run an optimization phase on cloud GPU.

    The standard pipeline:
        theta -> density(theta, radius) -> Layer -> create_structure (for viz)
        Then: optimize(layers=..., theta=..., grid=..., wavelength=...,
                       source=..., mode=..., phase=..., n_steps=...)

    Args:
        layers: Layer stack as list of dicts. Each dict has name, thickness,
            index. Design layers also have: design=True, density_radius.
        theta: Initial design variables per layer, e.g. {"etch": np.array(...)}.
            If None, uses initial_value from layer specs (default 0.5).
        grid: FDTD grid spacing in um.
        wavelength: Operating wavelength in um.
        source: Source field array.
        mode: Target mode field array.
        objective: Objective expression tree (from hwc.objectives).
            If None, uses mode coupling with the provided mode field.
        phase: "freeform", "binarize", "dfm", or "recovery".
        n_steps: Number of optimization steps.
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
        device: (Deprecated) DeviceConfig from build_device(). Use layers=
            and theta= instead.

    Returns:
        OptimizationResult with .design, .history, .save().
    """
    # Handle backward compat: if device= passed, use old path
    if device is not None:
        if hasattr(device, 'design_layers_info'):
            # DeviceConfig object
            pass
        elif isinstance(device, dict):
            pass
        else:
            raise TypeError(f"Unsupported device type: {type(device)}")
    elif layers is not None:
        # New path: construct DeviceConfig from primitives
        from hyperwave_community.device import _build_device_from_specs

        if source is None or mode is None:
            raise ValueError("source and mode are required")
        if grid is None or wavelength is None:
            raise ValueError("grid and wavelength are required")

        # Determine nx, ny from theta or layer specs
        nx = ny = None
        if theta:
            first_theta = next(iter(theta.values()))
            nx, ny = first_theta.shape
        else:
            raise ValueError(
                "theta is required. Create your design variables with "
                "np.full((nx, ny), 0.5) and pass as theta={'layer_name': array}")

        device = _build_device_from_specs(
            layers=layers, grid=grid, wavelength=wavelength,
            nx=nx, ny=ny,
        )

        # Override theta in device with user-provided theta
        if theta:
            for dl in device.design_layers_info:
                if dl["name"] in theta:
                    dl["theta"] = np.array(theta[dl["name"]])
    else:
        raise TypeError(
            "Pass layers= and theta= to define the device. Example:\n"
            "  hwc.optimize(\n"
            "      layers=[{'name':'box','thickness':2.0,'index':1.44}, ...],\n"
            "      theta={'etch': np.full((nx,ny), 0.5)},\n"
            "      grid=0.035, wavelength=1.55,\n"
            "      source=source_field, mode=mode_field,\n"
            "      phase='freeform', n_steps=100)"
        )

    if mode_cross_power is None:
        mode_cross_power = compute_mode_cross_power(np.array(mode))
        if mode_cross_power == 0.0:
            mode_cross_power = 1.0

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
        design_layers_raw = device.design_layers_info
        freq_band = list(device.freq_band)
        recipe_params = device.recipe_params
        source_offset = [0, 0, 0]
        absorption_widths = [70, 35, 17]
        absorption_coeff = 0.00489
        output_monitor_pos = [10, 0, 0]
        output_monitor_shape = [1, int(device.shape[1]), int(device.shape[2])]
        design_xy_range = [[0, int(device.recipe_params['grid_shape'][0])],
                           [0, int(device.recipe_params['grid_shape'][1])]]
        max_steps = 20000
        check_every_n = 500
        enforce_symmetry = False
    else:
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

    # Build design_layers with encoded arrays (theta, waveguide_mask)
    design_layers = []
    if initial_design is not None:
        for dl in design_layers_raw:
            name = dl['name']
            layer = dict(dl)
            if name in initial_design.thetas:
                layer['theta_b64'] = encode_array(np.array(initial_design.thetas[name]))
                layer['theta_shape'] = list(initial_design.thetas[name].shape)
                layer.pop('theta', None)
            if 'waveguide_mask' in layer and isinstance(layer['waveguide_mask'], np.ndarray):
                wm = layer.pop('waveguide_mask')
                layer['waveguide_mask_b64'] = encode_array(wm.astype(np.float32))
                layer['waveguide_mask_shape'] = list(wm.shape)
            design_layers.append(layer)
    else:
        for dl in design_layers_raw:
            layer = dict(dl)
            if 'theta' in layer:
                theta_arr = np.array(layer.pop('theta'))
                layer['theta_b64'] = encode_array(theta_arr)
                layer['theta_shape'] = list(theta_arr.shape)
            if 'waveguide_mask' in layer and isinstance(layer['waveguide_mask'], np.ndarray):
                wm = layer.pop('waveguide_mask')
                layer['waveguide_mask_b64'] = encode_array(wm.astype(np.float32))
                layer['waveguide_mask_shape'] = list(wm.shape)
            # Convert tuples to lists for JSON
            if 'eps_range' in layer and isinstance(layer['eps_range'], tuple):
                layer['eps_range'] = list(layer['eps_range'])
            if 'z_range' in layer and isinstance(layer['z_range'], tuple):
                layer['z_range'] = list(layer['z_range'])
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

    # WebSocket transport
    try:
        import websocket as _ws_lib
    except ImportError:
        raise ImportError(
            "websocket-client required for optimize(). "
            "Install with: pip install websocket-client")

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

    try:
        response = requests.post(
            f"{API_URL}/pipeline_optimize_start",
            data=body, headers=headers, timeout=(60, 300))
        response.raise_for_status()
    except requests.HTTPError as e:
        _handle_api_error(e, "pipeline optimize")
        raise

    session_id = response.json()["session_id"]
    logger.info("  Session started in %.1fs: %s...", _time.time() - t0, session_id[:8])

    ws_url = API_URL.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/inverse_design_ws?session_id={session_id}"

    ws = _ws_lib.create_connection(
        ws_url, header={"X-API-Key": effective_api_key}, timeout=30)
    ws.settimeout(600)

    stop_ping = threading.Event()
    ws_lock = threading.Lock()

    def _pinger():
        while not stop_ping.is_set():
            stop_ping.wait(30)
            if not stop_ping.is_set():
                try:
                    with ws_lock:
                        ws.send(json.dumps({"type": "ping"}))
                except Exception:
                    break

    ping_thread = threading.Thread(target=_pinger, daemon=True)
    ping_thread.start()

    history = []
    current_thetas = (dict(initial_design.thetas) if initial_design else {})
    cancelled = False

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
                step_num = msg.get("step", 0)
                eff = msg.get("efficiency", 0.0)

                history.append({
                    "step": step_num,
                    "efficiency": eff,
                    "loss": msg.get("loss", 0.0),
                    "fab_loss": msg.get("fab_loss"),
                    "time": msg.get("step_time", 0.0),
                })

                # Decode thetas if present
                if "theta_b64" in msg and isinstance(msg["theta_b64"], dict):
                    current_thetas = {
                        name: decode_array(b64)
                        for name, b64 in msg["theta_b64"].items()
                    }
                elif "theta_b64" in msg and isinstance(msg["theta_b64"], str):
                    current_thetas = {"design": decode_array(msg["theta_b64"])}

                eff_db = -10 * np.log10(max(eff, 1e-10))
                print(f"Step {step_num:3d}/{n_steps}: "
                      f"efficiency={eff*100:6.2f}% ({eff_db:.2f} dB)  "
                      f"time={msg.get('step_time', 0):.0f}s",
                      flush=True)

    except KeyboardInterrupt:
        cancelled = True
        n_done = len(history)
        print(f"\nCancelled after {n_done} steps. "
              f"Completed steps are kept, GPU job stopped.", flush=True)
        try:
            with ws_lock:
                ws.send(json.dumps({"type": "cancel"}))
        except Exception:
            pass

    finally:
        stop_ping.set()
        if ping_thread is not None:
            ping_thread.join(timeout=2)
        try:
            ws.close()
        except Exception:
            pass

    # Build result from collected history
    final_eff = history[-1]["efficiency"] if history else 0.0
    final_step = history[-1]["step"] if history else 0

    # Extract per-layer density radii from device config
    _density_radii = {}
    for dl in design_layers_raw:
        if "density_radius" in dl:
            _density_radii[dl["name"]] = int(dl["density_radius"])

    design = Design(
        thetas=current_thetas,
        density_radii=_density_radii,
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
        "disk_radius": disk_radius,
    }
    if beta_init is not None:
        schedule_config["beta_init"] = beta_init
    if beta_max is not None:
        schedule_config["beta_max"] = beta_max
    if learning_rate is not None:
        schedule_config["learning_rate"] = learning_rate

    elapsed = _time.time() - t0
    if not cancelled:
        print(f"\nCompleted {len(history)} steps in {elapsed:.0f}s. "
              f"Best: {best_eff*100:.2f}% at step {best_step}.", flush=True)

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
    min_area_nm2: float = 30_000,
    pixel_size: float = 0.0175,
    min_feature_size: float = None,
) -> Design:
    """Remove small features and fill small holes.

    Non-differentiable cleanup step. Runs locally on CPU, no credits.
    Scans the binarized density for solid islands and interior holes
    smaller than ``min_area_nm2`` and removes/fills them.

    Args:
        design: Design to clean up.
        min_area_nm2: Minimum feature area in nm^2. Solid islands and
            interior holes below this area are removed or filled.
            Default: 30,000 nm^2.
        pixel_size: Physical pixel size in um.
        min_feature_size: Deprecated. Use ``min_area_nm2`` instead.

    Returns:
        New Design with cleaned thetas and surgery metadata.
    """
    from scipy import ndimage
    from hyperwave_community.structure import density

    if min_feature_size is not None:
        import warnings
        warnings.warn(
            "min_feature_size is deprecated. Use min_area_nm2 instead.",
            DeprecationWarning, stacklevel=2)
        min_area_px = int(np.pi * (min_feature_size / pixel_size / 2) ** 2)
    else:
        px_nm = pixel_size * 1000
        min_area_px = int(round(min_area_nm2 / (px_nm ** 2)))
    min_area_px = max(min_area_px, 1)

    new_thetas = {}
    total_removed = 0
    total_filled = 0

    for name, theta in design.thetas.items():
        mask = design.design_mask(name)
        layer_radius = float(design.density_radii.get(name, 6))

        import jax.numpy as jnp
        d = np.array(density(jnp.array(theta), radius=layer_radius))
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
        density_radii=dict(design.density_radii),
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
    fab_rules: Optional[Dict[str, Dict[str, int]]] = None,
):
    """Run design rule check via morphological opening.

    Runs locally on CPU, no credits charged.

    Single-layer (simple):
        drc = check_drc(design, disk_radius=3)
        drc.passed  # True/False

    Multi-layer with per-layer rules:
        drc = check_drc(design, fab_rules={
            "W1": {"cd_radius": 4, "gap_radius": 4},
            "W2": {"cd_radius": 4, "gap_radius": 6},
        })
        drc["W1"].passed
        drc["W2"].gap_pct

    Args:
        design: Design to check.
        disk_radius: Default morphological disk radius (used when
            fab_rules not provided, or as fallback for missing layers).
        pixel_size: Physical pixel size in um.
        fab_rules: Per-layer fab rules. Dict of layer_name -> dict with
            optional keys: cd_radius, gap_radius (default to disk_radius).

    Returns:
        DrcReport if single-layer or no fab_rules.
        Dict[str, DrcReport] if fab_rules provided (keyed by layer name).
    """
    if fab_rules is not None:
        results = {}
        for name in design.layer_names:
            rules = fab_rules.get(name, {})
            cd_r = rules.get("cd_radius", disk_radius)
            gap_r = rules.get("gap_radius", disk_radius)
            results[name] = _check_drc_layer(
                design, name, cd_r, gap_r, pixel_size)
        return results

    if len(design.layer_names) == 1:
        name = design.layer_names[0]
        return _check_drc_layer(design, name, disk_radius, disk_radius, pixel_size)

    # Multi-layer without fab_rules: check each with same radius
    import warnings
    warnings.warn(
        f"Design has {len(design.layer_names)} layers but no fab_rules provided. "
        f"Using disk_radius={disk_radius} for all layers. "
        f"Pass fab_rules={{...}} for per-layer DRC specs.",
        stacklevel=2,
    )
    results = {}
    for name in design.layer_names:
        results[name] = _check_drc_layer(
            design, name, disk_radius, disk_radius, pixel_size)
    return results


def _check_drc_layer(
    design: Design,
    layer_name: str,
    cd_radius: int,
    gap_radius: int,
    pixel_size: float,
) -> DrcReport:
    """Check DRC for a single layer with separate CD and gap radii."""
    from skimage.morphology import disk, opening
    from hyperwave_community.structure import density
    import jax.numpy as jnp

    theta = design.thetas[layer_name]
    mask = design.design_mask(layer_name)

    layer_radius = float(design.density_radii.get(layer_name, 6))
    d = np.array(density(jnp.array(theta), radius=layer_radius))
    binary = (d >= 0.5).astype(bool)
    binary_design = binary & mask
    void_design = (~binary) & mask

    design_pixels = int(mask.sum())
    d_vals = d[mask]
    binarization_score = float(1.0 - np.mean(4 * d_vals * (1 - d_vals)))

    # CD violations: solid features smaller than cd_radius
    cd_selem = disk(cd_radius)
    solid_opened = opening(binary_design, cd_selem)
    cd_violations = int((binary_design & ~solid_opened).sum())

    # Gap violations: void features smaller than gap_radius
    gap_selem = disk(gap_radius)
    void_opened = opening(void_design, gap_selem)
    gap_violations = int((void_design & ~void_opened).sum())

    min_feature_nm = (2 * cd_radius + 1) * pixel_size * 1000
    min_gap_nm = (2 * gap_radius + 1) * pixel_size * 1000
    cd_pct = 100.0 * cd_violations / design_pixels if design_pixels > 0 else 0.0
    gap_pct = 100.0 * gap_violations / design_pixels if design_pixels > 0 else 0.0

    return DrcReport(
        cd_violations=cd_violations,
        cd_pct=cd_pct,
        gap_violations=gap_violations,
        gap_pct=gap_pct,
        binarization_score=binarization_score,
        disk_radius=cd_radius,
        min_feature_nm=min_feature_nm,
        min_gap_nm=min_gap_nm,
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
    layer_name: Optional[str] = None,
    beta: float = 64.0,
    eta: Optional[float] = None,
    smooth_nm: Optional[float] = None,
) -> str:
    """Export design to GDSII file.

    Uses the subpixel-accurate pipeline validated by Jon Roth:
    theta -> conic filter -> tanh Heaviside projection -> marching squares
    contour -> optional KLayout smoothing.

    The tanh projection at beta=64 keeps the density mostly binary but
    leaves a thin gradient ring at each contour, giving marching squares
    sub-pixel interpolation accuracy. This produces smooth contours
    without pixel laddering artifacts.

    Runs locally on CPU, no credits charged.

    Args:
        design: Design to export.
        filename: Output GDS filename.
        layer: GDS layer tuple (layer_number, datatype).
        pixel_size: Physical pixel size in um.
        layer_name: Which design layer to export. Defaults to first layer.
        beta: Tanh Heaviside projection sharpness. Default 64.0 (matches
            typical optimization final beta). Higher = sharper edges.
        eta: Projection threshold. If None, uses the layer's density_eta
            from build_device(). Controls solid/void boundary position.
        smooth_nm: Optional KLayout smoothing distance in nm. Removes
            remaining staircase vertices. 5nm works well for 12.5nm grids.
            None = no smoothing (raw marching squares output).

    Returns:
        Absolute path to the generated GDS file.
    """
    from hyperwave_community.structure import density as _density_filter
    from hyperwave_community.data_io import generate_gds_from_density

    name = layer_name or design.layer_names[0]
    if len(design.layer_names) > 1 and layer_name is None:
        import warnings
        warnings.warn(
            f"Multi-layer design, exporting first layer '{name}'. "
            f"Pass layer_name= to select a specific layer.",
            stacklevel=2,
        )
    theta = design.thetas[name]
    layer_radius = float(design.density_radii.get(name, 6))

    if eta is None:
        eta = 0.5

    import jax.numpy as jnp

    # Step 1: Conic filter (alpha=0 skips internal projection)
    ufilt = np.array(_density_filter(jnp.array(theta), radius=layer_radius,
                                     alpha=0.0, eta=eta))

    # Step 2: Tanh Heaviside projection (Wang 2011 / Hammond 2022)
    # Keeps thin gradient ring at contours for sub-pixel marching squares
    num = np.tanh(beta * eta) + np.tanh(beta * (ufilt - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    d = num / den

    # Step 3: Marching squares contour at 0.5
    gds_path = generate_gds_from_density(
        density_array=d,
        level=0.5,
        output_filename=filename,
        resolution=pixel_size,
    )

    # Step 4: Optional KLayout smoothing
    if smooth_nm is not None and smooth_nm > 0:
        try:
            from hyperwave.data_io import smooth_gds
            smooth_gds(
                input_path=gds_path,
                output_path=gds_path,
                smooth_d_nm=float(smooth_nm),
            )
        except ImportError:
            import warnings
            warnings.warn(
                "KLayout not available for smoothing. Install with: "
                "pip install klayout. Returning unsmoothed GDS.",
                stacklevel=2,
            )

    return gds_path
