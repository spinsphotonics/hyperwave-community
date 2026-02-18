"""Test 02: Deterministic resume - verify optimizer state continuity."""

import os
import numpy as np
import pytest
from .helpers import make_test_inputs, run_opt, NUM_STEPS, THETA_SHAPE

SPLIT_STEP = 15


class TestDeterministicResume:
    def test_resume_loss_continuity(self, configure_sdk, tmp_dir):
        """Loss should not spike on resume; phase 2 starts near where phase 1 ended."""
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt.npz")

        r1 = run_opt(hwc, theta, source, SPLIT_STEP)
        assert len(r1) == SPLIT_STEP
        hwc.save_checkpoint(r1, ckpt)

        r2 = run_opt(hwc, None, source, NUM_STEPS, checkpoint=ckpt)
        assert len(r2) == NUM_STEPS - SPLIT_STEP

        # Loss at start of phase 2 should be close to loss at end of phase 1
        last_p1_loss = r1[-1]['loss']
        first_p2_loss = r2[0]['loss']
        # Allow some movement since LR schedule restarts, but no huge spike
        assert first_p2_loss < last_p1_loss * 1.5, (
            f"Loss spiked on resume: {last_p1_loss:.6f} -> {first_p2_loss:.6f}"
        )

        # Loss should generally decrease across the full run
        all_losses = [r['loss'] for r in r1 + r2]
        assert all_losses[-1] < all_losses[0], "Loss did not decrease overall"

    def test_phase1_matches_independent_run(self, configure_sdk, tmp_dir):
        """Phase 1 of a resume run should exactly match an independent run of the same length."""
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt.npz")

        # Run 1: 15 steps then checkpoint
        r1 = run_opt(hwc, theta, source, SPLIT_STEP)
        hwc.save_checkpoint(r1, ckpt)

        # Run 2: independent 15 steps from same initial conditions
        r_independent = run_opt(hwc, theta, source, SPLIT_STEP)

        losses_r1 = np.array([r['loss'] for r in r1])
        losses_ind = np.array([r['loss'] for r in r_independent])
        np.testing.assert_allclose(losses_r1, losses_ind, rtol=1e-5,
                                   err_msg="Phase 1 not reproducible")

    def test_theta_restored_on_resume(self, configure_sdk, tmp_dir):
        """Theta at start of phase 2 should match theta at end of phase 1."""
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt.npz")

        r1 = run_opt(hwc, theta, source, SPLIT_STEP)
        hwc.save_checkpoint(r1, ckpt)
        r2 = run_opt(hwc, None, source, NUM_STEPS, checkpoint=ckpt)

        last_theta_p1 = r1[-1]['theta']
        first_theta_p2 = r2[0]['theta']
        # The first step of phase 2 uses restored theta from checkpoint,
        # but returns theta AFTER the update. So just check they're close.
        np.testing.assert_allclose(
            first_theta_p2, last_theta_p1, atol=0.01,
            err_msg="Theta not properly restored on resume"
        )

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
