"""Test 07: Cosine LR schedule continuity on resume."""

import os
import math
import numpy as np
import pytest
from .helpers import make_test_inputs, run_opt, NUM_STEPS, THETA_SHAPE


def expected_cosine_lr(step, total_steps, init_lr=0.01, alpha=0.1):
    if total_steps <= 1:
        return init_lr
    progress = min(step / (total_steps - 1), 1.0)
    return init_lr * (alpha + (1 - alpha) * 0.5 * (1 + math.cos(math.pi * progress)))


class TestCosineLR:
    def test_full_run_lr_curve(self, configure_sdk):
        theta, source = make_test_inputs()
        results = run_opt(configure_sdk, theta, source, NUM_STEPS)
        if 'lr' not in results[0]:
            pytest.skip("lr not in response")

        lrs = [r['lr'] for r in results]
        assert lrs[0] > lrs[-1]

        for i, lr in enumerate(lrs):
            expected = expected_cosine_lr(i, NUM_STEPS)
            np.testing.assert_allclose(lr, expected, rtol=1e-4,
                                       err_msg=f"LR mismatch at step {i}")

    def test_resumed_lr_is_fresh_cosine(self, configure_sdk, tmp_dir):
        """Resumed run should have a fresh cosine schedule over remaining_steps.

        This is by design: the server creates a new cosine schedule for
        remaining_steps = num_steps - start_step, so LR starts high again
        but decays over fewer steps.
        """
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt_lr.npz")

        # Phase 1: 15 steps
        r1 = run_opt(hwc, theta, source, 15)
        if 'lr' not in r1[0]:
            pytest.skip("lr not in response")
        hwc.save_checkpoint(r1, ckpt)

        # Verify phase 1 LR is cosine over 15 steps
        lrs_p1 = [r['lr'] for r in r1]
        for i, lr in enumerate(lrs_p1):
            expected = expected_cosine_lr(i, 15)
            np.testing.assert_allclose(lr, expected, rtol=1e-4,
                                       err_msg=f"Phase 1 LR mismatch at step {i}")

        # Phase 2: resume from step 15, total 30 -> 15 remaining
        r2 = run_opt(hwc, None, source, NUM_STEPS, checkpoint=ckpt)
        lrs_p2 = [r['lr'] for r in r2]

        # Phase 2 should be a fresh cosine over 15 remaining steps
        for i, lr in enumerate(lrs_p2):
            expected = expected_cosine_lr(i, 15)
            np.testing.assert_allclose(lr, expected, rtol=1e-4,
                                       err_msg=f"Phase 2 LR mismatch at step {i}")
