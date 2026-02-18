"""Test 05: Multiple save/load cycles (10+10+10)."""

import os
import numpy as np
import pytest
from .helpers import make_test_inputs, run_opt, NUM_STEPS, THETA_SHAPE


class TestMultiSegment:
    def test_three_segments_loss_decreasing(self, configure_sdk, tmp_dir):
        """Loss should decrease across all 3 segments despite LR restarts."""
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

        # Overall loss should decrease
        assert all_losses[-1] < all_losses[0], "Loss did not decrease overall"

        # No huge spikes at resume boundaries
        last_seg1 = seg1[-1]['loss']
        first_seg2 = seg2[0]['loss']
        assert first_seg2 < last_seg1 * 1.5, (
            f"Loss spike at seg1->seg2: {last_seg1:.6f} -> {first_seg2:.6f}"
        )

        last_seg2 = seg2[-1]['loss']
        first_seg3 = seg3[0]['loss']
        assert first_seg3 < last_seg2 * 1.5, (
            f"Loss spike at seg2->seg3: {last_seg2:.6f} -> {first_seg3:.6f}"
        )

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
