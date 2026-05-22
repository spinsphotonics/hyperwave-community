"""End-to-end integration test for hwc.optimize() against production API.

Verifies the full pipeline: WebSocket connection to Modal, GPU job dispatch,
step streaming, and result parsing. Runs ONE 2-step optimization and checks
all assertions on the single result to minimize GPU cost.

Usage:
    HYPERWAVE_API_KEY=<key> python -m pytest tests/integration/test_optimize_e2e.py -v -s

Skipped by default unless HYPERWAVE_API_KEY is set.
"""

import os
import numpy as np
import pytest

HAVE_KEY = bool(os.environ.get("HYPERWAVE_API_KEY"))

pytestmark = pytest.mark.skipif(
    not HAVE_KEY,
    reason="HYPERWAVE_API_KEY not set (real GPU test, costs credits)"
)


def test_optimize_e2e():
    """Single end-to-end optimize: connect, run 2 GPU steps, validate result."""
    import hyperwave_community as hwc

    hwc.configure_api(
        api_key=os.environ["HYPERWAVE_API_KEY"],
        validate=False,
    )

    voxel_size = 0.025
    wavelength = 1.55
    n_sin, n_sio2 = 2.0, 1.44
    wg_width_um = 1.0
    wg_height_um = 0.4

    layers = [
        {"name": "box", "thickness": 1.0, "index": n_sio2},
        {"name": "sin", "thickness": wg_height_um, "index": n_sin,
         "design": True, "density_radius": 4},
        {"name": "clad", "thickness": 1.0, "index": n_sio2},
    ]

    nx, ny = 400, 200
    theta = np.random.rand(nx, ny).astype(np.float32) * 0.5 + 0.25

    # Build device to get the actual YZ cross-section size
    from hyperwave_community.device import _build_device_from_specs
    dev = _build_device_from_specs(
        layers=layers, grid=voxel_size, wavelength=wavelength,
        nx=nx, ny=ny)
    _, Ly, Lz = dev.shape

    source_field, _ = hwc.solve_waveguide_mode(
        grid=voxel_size, waveguide_width=wg_width_um,
        waveguide_height=wg_height_um, n_core=n_sin,
        n_clad=n_sio2, wavelength=wavelength, mode_number=0,
        cross_section_size=max(Ly, Lz))
    # Pad/crop to match structure cross-section
    _, _, _, my, mz = source_field.shape
    field = np.zeros((1, 6, 1, Ly, Lz), dtype=np.complex64)
    y0 = (Ly - my) // 2
    z0 = (Lz - mz) // 2
    y1, z1 = y0 + my, z0 + mz
    ys, ye = max(0, y0), min(Ly, y1)
    zs, ze = max(0, z0), min(Lz, z1)
    sys, sye = ys - y0, ye - y0
    szs, sze = zs - z0, ze - z0
    field[:, :, :, ys:ye, zs:ze] = source_field[:, :, :, sys:sye, szs:sze]
    source_field = field
    mode_field = source_field

    result = hwc.optimize(
        layers=layers,
        theta={"sin": theta},
        grid=voxel_size,
        wavelength=wavelength,
        source=source_field,
        mode=mode_field,
        n_steps=2,
        phase="freeform",
        gpu_type="B200",
    )

    # Completion
    assert result.n_steps == 2, f"Expected 2 steps, got {result.n_steps}"
    assert result.phase == "freeform"

    # History
    assert len(result.history) == 2
    for entry in result.history:
        assert "step" in entry
        assert "efficiency" in entry
        assert "loss" in entry
        assert "time" in entry
        assert isinstance(entry["step"], int)
        assert isinstance(entry["efficiency"], float)

    # Design
    assert result.design is not None
    assert result.design.thetas is not None
    assert len(result.design.thetas) > 0
    for name, t in result.design.thetas.items():
        assert isinstance(t, np.ndarray), f"theta[{name}] not ndarray"
        assert t.ndim == 2, f"theta[{name}] shape={t.shape}"

    # Best tracking
    assert result.best_efficiency >= 0.0
    assert result.best_step >= 1
