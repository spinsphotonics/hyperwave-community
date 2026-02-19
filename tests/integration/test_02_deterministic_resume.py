"""Test 02: Deterministic resume - THE KILLER TEST.

With constant LR (vanilla Adam), a 15+15 resumed run must produce
identical results to a 30-step straight run. The optimizer state
(mu, nu, count) is fully restored from checkpoint, and LR is constant,
so there is nothing to cause divergence.
"""

import os
import numpy as np
import pytest
from .helpers import make_test_inputs, run_opt, NUM_STEPS, THETA_SHAPE, BASELINE_FILE

SPLIT_STEP = 15


class TestDeterministicResume:
    def _get_baseline(self):
        if not os.path.exists(BASELINE_FILE):
            pytest.skip("Baseline not found. Run test_01 first.")
        return np.load(BASELINE_FILE, allow_pickle=True)

    def test_resume_losses_match_baseline(self, configure_sdk, tmp_dir):
        """15 + 15 resumed losses must exactly match 30-step straight run."""
        baseline = self._get_baseline()
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt.npz")

        r1 = run_opt(hwc, theta, source, SPLIT_STEP)
        assert len(r1) == SPLIT_STEP
        hwc.save_checkpoint(r1, ckpt)

        r2 = run_opt(hwc, None, source, NUM_STEPS, checkpoint=ckpt)
        assert len(r2) == NUM_STEPS - SPLIT_STEP

        all_losses = np.array([r['loss'] for r in r1 + r2])
        np.testing.assert_allclose(all_losses, baseline['losses'], rtol=1e-5,
                                   err_msg="Resume diverged from baseline")

    def test_resume_thetas_match_baseline(self, configure_sdk, tmp_dir):
        """15 + 15 resumed thetas must exactly match 30-step straight run."""
        baseline = self._get_baseline()
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt.npz")

        r1 = run_opt(hwc, theta, source, SPLIT_STEP)
        hwc.save_checkpoint(r1, ckpt)
        r2 = run_opt(hwc, None, source, NUM_STEPS, checkpoint=ckpt)

        resume_thetas = np.stack([r['theta'] for r in r1 + r2])
        np.testing.assert_allclose(resume_thetas, baseline['thetas'], rtol=1e-5,
                                   err_msg="Theta arrays diverged")

    def test_step_numbers_correct(self, configure_sdk, tmp_dir):
        """Step numbers should be continuous across resume."""
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt.npz")

        r1 = run_opt(hwc, theta, source, SPLIT_STEP)
        hwc.save_checkpoint(r1, ckpt)
        r2 = run_opt(hwc, None, source, NUM_STEPS, checkpoint=ckpt)

        all_steps = [r['step'] for r in r1 + r2]
        assert all_steps == list(range(NUM_STEPS))
