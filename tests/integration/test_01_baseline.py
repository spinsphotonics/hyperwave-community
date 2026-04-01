"""Test 01: Full 30-step baseline run."""

import numpy as np
from .helpers import make_test_inputs, run_opt, NUM_STEPS, THETA_SHAPE, BASELINE_FILE


class TestBaseline:
    def test_runs_all_steps(self, configure_sdk):
        theta, source = make_test_inputs()
        results = run_opt(configure_sdk, theta, source, NUM_STEPS)
        assert len(results) == NUM_STEPS

    def test_loss_decreases(self, configure_sdk):
        theta, source = make_test_inputs()
        results = run_opt(configure_sdk, theta, source, NUM_STEPS)
        losses = [r['loss'] for r in results]
        assert losses[-1] < losses[0]

    def test_step_numbers_sequential(self, configure_sdk):
        theta, source = make_test_inputs()
        results = run_opt(configure_sdk, theta, source, NUM_STEPS)
        steps = [r['step'] for r in results]
        assert steps == list(range(NUM_STEPS))

    def test_theta_present(self, configure_sdk):
        theta, source = make_test_inputs()
        results = run_opt(configure_sdk, theta, source, NUM_STEPS)
        for r in results:
            assert 'theta' in r
            assert r['theta'].shape == THETA_SHAPE

    def test_opt_state_present(self, configure_sdk):
        theta, source = make_test_inputs()
        results = run_opt(configure_sdk, theta, source, NUM_STEPS)
        for r in results:
            assert 'opt_state_b64' in r

    def test_save_baseline(self, configure_sdk):
        """Save baseline for comparison tests."""
        theta, source = make_test_inputs()
        results = run_opt(configure_sdk, theta, source, NUM_STEPS)
        np.savez(
            BASELINE_FILE,
            losses=np.array([r['loss'] for r in results]),
            efficiencies=np.array([r['efficiency'] for r in results]),
            grad_maxes=np.array([r['grad_max'] for r in results]),
            thetas=np.stack([r['theta'] for r in results]),
        )
