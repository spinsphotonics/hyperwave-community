"""Test 04: Boundary edge cases for resume."""

import os
import numpy as np
import pytest
from .helpers import make_test_inputs, run_opt, THETA_SHAPE


class TestBoundaryResume:
    def test_single_step(self, configure_sdk):
        hwc = configure_sdk
        theta, source = make_test_inputs()
        results = run_opt(hwc, theta, source, num_steps=1)
        assert len(results) == 1
        assert results[0]['step'] == 0

    def test_resume_last_step(self, configure_sdk, tmp_dir):
        """Resume from step 28 for total 30 should yield 1 step."""
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt29.npz")

        r1 = run_opt(hwc, theta, source, num_steps=29)
        assert len(r1) == 29
        hwc.save_checkpoint(r1, ckpt)

        r2 = run_opt(hwc, None, source, num_steps=30, checkpoint=ckpt)
        assert len(r2) == 1

    def test_double_checkpoint(self, configure_sdk, tmp_dir):
        """Save, resume, save again, resume again."""
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt1 = os.path.join(tmp_dir, "ckpt1.npz")
        ckpt2 = os.path.join(tmp_dir, "ckpt2.npz")

        r1 = run_opt(hwc, theta, source, num_steps=10)
        hwc.save_checkpoint(r1, ckpt1)

        r2 = run_opt(hwc, None, source, num_steps=15, checkpoint=ckpt1)
        assert len(r2) == 5
        hwc.save_checkpoint(r1 + r2, ckpt2)

        r3 = run_opt(hwc, None, source, num_steps=20, checkpoint=ckpt2)
        assert len(r3) == 5

        all_steps = [r['step'] for r in r1 + r2 + r3]
        assert all_steps == list(range(20))
