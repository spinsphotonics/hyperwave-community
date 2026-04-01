"""Test 07: Constant LR (vanilla Adam) - verify LR is constant across run and resume."""

import os
import numpy as np
import pytest
from .helpers import make_test_inputs, run_opt, NUM_STEPS


class TestConstantLR:
    def test_full_run_constant_lr(self, configure_sdk):
        """All steps should use the same learning rate."""
        theta, source = make_test_inputs()
        results = run_opt(configure_sdk, theta, source, NUM_STEPS)
        if 'lr' not in results[0]:
            pytest.skip("lr not in response")

        lrs = [r['lr'] for r in results]
        for i, lr in enumerate(lrs):
            np.testing.assert_allclose(lr, 0.01, rtol=1e-4,
                                       err_msg=f"LR not constant at step {i}")

    def test_resumed_lr_same_as_initial(self, configure_sdk, tmp_dir):
        """Resumed run should use the same constant LR."""
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt_lr.npz")

        r1 = run_opt(hwc, theta, source, 15)
        if 'lr' not in r1[0]:
            pytest.skip("lr not in response")
        hwc.save_checkpoint(r1, ckpt)

        r2 = run_opt(hwc, None, source, NUM_STEPS, checkpoint=ckpt)
        lrs = [r['lr'] for r in r1 + r2]
        for i, lr in enumerate(lrs):
            np.testing.assert_allclose(lr, 0.01, rtol=1e-4,
                                       err_msg=f"LR not constant at step {i}")
