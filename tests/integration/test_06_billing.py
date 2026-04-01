"""Test 06: Billing verification."""

import os
import numpy as np
import pytest
import requests
from .helpers import make_test_inputs, run_opt, THETA_SHAPE


class TestBilling:
    def test_full_run_billed(self, configure_sdk, mock_server):
        theta, source = make_test_inputs()
        run_opt(configure_sdk, theta, source, 30)
        billing = requests.get(f"{mock_server}/billing").json()
        assert len(billing) == 1
        assert billing[0]['status'] == 'completed'
        assert billing[0]['steps_completed'] == 30

    def test_partial_run_billed(self, configure_sdk, mock_server):
        hwc = configure_sdk
        theta, source = make_test_inputs()
        count = 0
        for step_result in hwc.run_optimization(
            theta=theta,
            source_field=source,
            source_offset=(0, 0, 0),
            freq_band=(0.196, 0.209, 2),
            structure_spec={"layers_info": [], "construction_params": {"grid_shape": list(THETA_SHAPE)}},
            loss_monitor_shape=(1, 1, 1),
            loss_monitor_offset=(0, 0, 0),
            design_monitor_shape=(1, 1, 1),
            design_monitor_offset=(0, 0, 0),
            mode_field=source,
            input_power=1.0,
            mode_cross_power=0.5,
            num_steps=30,
            learning_rate=0.01,
            gpu_type="B200",
            auto_checkpoint=False,
        ):
            count += 1
            if count >= 10:
                break

        billing = requests.get(f"{mock_server}/billing").json()
        assert len(billing) >= 1
        last = billing[-1]
        assert last['steps_completed'] >= 10

    def test_resume_billing_separate(self, configure_sdk, mock_server, tmp_dir):
        hwc = configure_sdk
        theta, source = make_test_inputs()
        ckpt = os.path.join(tmp_dir, "ckpt.npz")

        r1 = run_opt(hwc, theta, source, 15)
        hwc.save_checkpoint(r1, ckpt)
        r2 = run_opt(hwc, None, source, 30, checkpoint=ckpt)

        billing = requests.get(f"{mock_server}/billing").json()
        assert len(billing) == 2
        assert billing[0]['steps_completed'] == 15
        assert billing[1]['steps_completed'] == 15
