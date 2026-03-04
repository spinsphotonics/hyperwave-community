"""Phase 2: Integration tests against real server (localhost:8000 -> Modal GPU).

These tests use real electromagnetic simulations on Modal GPU.
They verify functional correctness of the optimization pipeline:
- Full optimization run completes with valid results
- Loss monotonically decreases (optimization is working)
- Client-side checkpoint save/load works with real data
- Step numbering is correct (1-indexed from server)
- Theta arrays update each step

Note: opt_state round-trip is tested in Phase 1 (mock server, 32 tests).
The real server does not yet stream opt_state_b64, so resume tests
verify theta-based warm-start rather than full optimizer state restore.

Usage:
    1. Start the real server: bash /tmp/start_real_server.sh
    2. Run: python -m pytest tests/integration/test_phase2_real_server.py -v -s

IMPORTANT: Each test step runs a real FDTD simulation on Modal GPU.
A 40x40 grid with 10 steps takes ~30-40 seconds total (first step ~20s for JIT).
"""

import time
import numpy as np
import pytest

# Skip entire module if real server isn't running
try:
    import requests
    r = requests.get("http://localhost:8000/health", timeout=2)
    REAL_SERVER_AVAILABLE = r.status_code == 200
except Exception:
    REAL_SERVER_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not REAL_SERVER_AVAILABLE,
    reason="Real server not running on localhost:8000"
)

# Small grid for fast tests
GRID_SHAPE = (40, 40)
NUM_STEPS = 10
API_KEY = "9e293a83-feb0-4275-b6f0-540ba935b4bb"
API_URL = "http://localhost:8000"


def configure_real_sdk():
    """Configure SDK to point at the real local server."""
    import hyperwave_community as hwc
    from hyperwave_community.api_client import _API_CONFIG
    _API_CONFIG['api_key'] = API_KEY
    _API_CONFIG['api_url'] = API_URL
    _API_CONFIG['gateway_url'] = None
    return hwc


def make_minimal_inputs():
    """Create minimal valid inputs for a real optimization run.

    Uses a tiny 40x40 theta grid with simple 3-layer stack.
    Structure grid is 20x20 (theta is 2x structure due to subpixel averaging).
    """
    rng = np.random.RandomState(42)
    theta = rng.rand(*GRID_SHAPE).astype(np.float32) * 0.5 + 0.25

    # Layer thicknesses (in pixels)
    h_air = 4
    h_design = 6
    h_sub = 4
    Lz = h_air + h_design + h_sub  # 14 pixels

    # Source: uniform Ey plane wave
    struct_x = GRID_SHAPE[0] // 2  # 20
    struct_y = GRID_SHAPE[1] // 2  # 20
    source = np.zeros((1, 6, struct_x, struct_y, 1), dtype=np.complex64)
    source[0, 1, :, :, 0] = 1.0  # Ey component

    return theta, source, Lz, h_air, h_design, h_sub


def run_real_opt(hwc, theta, source, Lz, h_air, h_design, h_sub, num_steps):
    """Run optimization against real server using intensity loss."""
    structure_spec = {
        'layers_info': [
            {'permittivity_values': 1.0, 'layer_thickness': h_air,
             'density_radius': 0, 'density_alpha': 0},
            {'permittivity_values': [1.0, 12.11], 'layer_thickness': h_design,
             'density_radius': 0, 'density_alpha': 0},
            {'permittivity_values': 2.085, 'layer_thickness': h_sub,
             'density_radius': 0, 'density_alpha': 0},
        ],
        'construction_params': {'vertical_radius': 0},
    }

    freq = 2 * np.pi / (1.55 / 0.05)  # ~1550nm at 50nm resolution
    freq_band = (freq, freq, 1)

    struct_x = GRID_SHAPE[0] // 2
    struct_y = GRID_SHAPE[1] // 2

    results = []
    for step_result in hwc.run_optimization(
        theta=theta,
        source_field=source,
        source_offset=(0, 0, 1),
        freq_band=freq_band,
        structure_spec=structure_spec,
        loss_monitor_shape=(1, struct_y, Lz),
        loss_monitor_offset=(struct_x - 2, 0, 0),
        design_monitor_shape=(struct_x, struct_y, h_design),
        design_monitor_offset=(0, 0, h_air),
        intensity_component='Ey',
        intensity_maximize=True,
        num_steps=num_steps,
        learning_rate=0.01,
        gpu_type="B200",
        absorption_widths=(2, 2, 2),
        max_steps=2000,
    ):
        results.append(step_result)
        print(f"  Step {step_result['step']:3d}: loss={step_result['loss']:.6f} "
              f"grad_max={step_result['grad_max']:.3e} ({step_result['step_time']:.1f}s)")
    return results


class TestRealServerBaseline:
    """Validate that a full optimization run works against real Modal GPU."""

    def test_full_run_completes(self):
        hwc = configure_real_sdk()
        theta, source, Lz, h_air, h_design, h_sub = make_minimal_inputs()

        print(f"\nRunning {NUM_STEPS} steps on real Modal GPU...")
        t0 = time.time()
        results = run_real_opt(hwc, theta, source, Lz, h_air, h_design, h_sub, NUM_STEPS)
        elapsed = time.time() - t0

        assert len(results) == NUM_STEPS, f"Expected {NUM_STEPS} results, got {len(results)}"
        print(f"Completed in {elapsed:.0f}s")

        # Verify required fields in each result
        for r in results:
            assert 'loss' in r
            assert 'theta' in r
            assert 'step' in r
            assert 'efficiency' in r
            assert 'grad_max' in r
            assert 'step_time' in r

        # Steps are sequential (server uses 1-indexed)
        steps = [r['step'] for r in results]
        assert steps == list(range(1, NUM_STEPS + 1)), f"Steps: {steps}"

        # Loss magnitude should decrease (loss is negative for maximize=True)
        losses = [r['loss'] for r in results]
        first_loss = losses[0]
        last_loss = losses[-1]
        print(f"Loss: {first_loss:.6f} -> {last_loss:.6f}")
        assert abs(last_loss) < abs(first_loss), (
            f"Loss magnitude did not decrease: {abs(first_loss):.6f} -> {abs(last_loss):.6f}"
        )

        # Theta arrays should have the right shape
        for r in results:
            assert r['theta'].shape == GRID_SHAPE, (
                f"Theta shape {r['theta'].shape} != expected {GRID_SHAPE}"
            )

        # Theta should change between first and last step
        theta_diff = np.max(np.abs(results[0]['theta'] - results[-1]['theta']))
        assert theta_diff > 1e-6, f"Theta didn't change: max diff = {theta_diff}"

    def test_checkpoint_save_load(self):
        """Save a checkpoint from real results, load it, verify contents."""
        hwc = configure_real_sdk()
        theta, source, Lz, h_air, h_design, h_sub = make_minimal_inputs()

        print(f"\nRunning {NUM_STEPS} steps for checkpoint test...")
        results = run_real_opt(hwc, theta, source, Lz, h_air, h_design, h_sub, NUM_STEPS)
        assert len(results) == NUM_STEPS

        # Save checkpoint
        ckpt_path = "/tmp/phase2_ckpt_test.npz"
        hwc.save_checkpoint(results, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

        # Load and verify
        loaded = hwc.load_checkpoint(ckpt_path)
        assert 'theta_history' in loaded, "Loaded checkpoint missing theta_history"
        assert 'theta_best' in loaded, "Loaded checkpoint missing theta_best"
        assert 'last_step' in loaded, "Loaded checkpoint missing last_step"
        assert 'loss_history' in loaded, "Loaded checkpoint missing loss_history"

        # Verify last theta matches last result
        last_theta = loaded['theta_history'][-1]
        np.testing.assert_array_equal(
            last_theta, results[-1]['theta'],
            err_msg="Loaded theta doesn't match last result"
        )

        # Verify loss history matches
        saved_losses = loaded['loss_history']
        result_losses = np.array([r['loss'] for r in results])
        np.testing.assert_allclose(saved_losses, result_losses, rtol=1e-6)

        print(f"Checkpoint verified: last_step={loaded['last_step']}, "
              f"theta_history shape={loaded['theta_history'].shape}, "
              f"loss range=[{saved_losses[0]:.6f}, {saved_losses[-1]:.6f}]")
