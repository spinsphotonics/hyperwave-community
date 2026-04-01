"""Test 05: Multiple save/load cycles (10+10+10) must match 30 straight.

With constant LR, each resumed segment continues with the exact same
optimizer state and LR, so the combined result must be identical to
a single uninterrupted run.
"""

import os
import numpy as np
import pytest
from .helpers import make_test_inputs, run_opt, NUM_STEPS, THETA_SHAPE, BASELINE_FILE


class TestMultiSegment:
    def test_three_segments_match_baseline(self, configure_sdk, tmp_dir):
        """10+10+10 resumed losses must exactly match 30-step straight run."""
        if not os.path.exists(BASELINE_FILE):
            pytest.skip("Baseline not found")
        baseline = np.load(BASELINE_FILE, allow_pickle=True)
        hwc = configure_sdk
        theta, source = make_test_inputs()

        ckpt1 = os.path.join(tmp_dir, "seg1.npz")
        ckpt2 = os.path.join(tmp_dir, "seg2.npz")

        seg1 = run_opt(hwc, theta, source, num_steps=10)
        hwc.save_checkpoint(seg1, ckpt1)

        seg2 = run_opt(hwc, None, source, num_steps=20, checkpoint=ckpt1)
        hwc.save_checkpoint(seg1 + seg2, ckpt2)

        seg3 = run_opt(hwc, None, source, num_steps=30, checkpoint=ckpt2)

        all_losses = np.array([r['loss'] for r in seg1 + seg2 + seg3])
        assert len(all_losses) == 30
        np.testing.assert_allclose(all_losses, baseline['losses'], rtol=1e-5,
                                   err_msg="3-segment run diverged from baseline")

    def test_step_continuity(self, configure_sdk, tmp_dir):
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "seg.npz")

        seg1 = run_opt(hwc, theta, source, num_steps=10)
        hwc.save_checkpoint(seg1, ckpt)

        seg2 = run_opt(hwc, None, source, num_steps=20, checkpoint=ckpt)
        hwc.save_checkpoint(seg1 + seg2, ckpt)

        seg3 = run_opt(hwc, None, source, num_steps=30, checkpoint=ckpt)

        all_steps = [r['step'] for r in seg1 + seg2 + seg3]
        assert all_steps == list(range(30))

    def test_segment_lengths_correct(self, configure_sdk, tmp_dir):
        """Each segment should have the correct number of steps."""
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "seg.npz")

        seg1 = run_opt(hwc, theta, source, num_steps=10)
        assert len(seg1) == 10
        hwc.save_checkpoint(seg1, ckpt)

        seg2 = run_opt(hwc, None, source, num_steps=20, checkpoint=ckpt)
        assert len(seg2) == 10
        hwc.save_checkpoint(seg1 + seg2, ckpt)

        seg3 = run_opt(hwc, None, source, num_steps=30, checkpoint=ckpt)
        assert len(seg3) == 10
