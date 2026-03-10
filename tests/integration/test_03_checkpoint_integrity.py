"""Test 03: Checkpoint integrity."""

import os
import numpy as np
import pytest
from .helpers import make_test_inputs, run_opt, THETA_SHAPE


class TestCheckpointIntegrity:
    def _run_and_save(self, hwc, path, num_steps=15):
        theta, source = make_test_inputs()
        results = run_opt(hwc, theta, source, num_steps)
        hwc.save_checkpoint(results, path)
        return results

    def test_file_created(self, configure_sdk, tmp_dir):
        path = os.path.join(tmp_dir, "ckpt.npz")
        self._run_and_save(configure_sdk, path)
        assert os.path.exists(path)

    def test_required_keys(self, configure_sdk, tmp_dir):
        path = os.path.join(tmp_dir, "ckpt.npz")
        self._run_and_save(configure_sdk, path)
        ckpt = configure_sdk.load_checkpoint(path)
        for key in ['theta_history', 'theta_best', 'best_step', 'last_step',
                     'loss_history', 'efficiency_history', 'grad_max_history']:
            assert key in ckpt, f"Missing: {key}"

    def test_theta_history_shape(self, configure_sdk, tmp_dir):
        path = os.path.join(tmp_dir, "ckpt.npz")
        self._run_and_save(configure_sdk, path, num_steps=15)
        ckpt = configure_sdk.load_checkpoint(path)
        assert ckpt['theta_history'].shape == (15, *THETA_SHAPE)

    def test_histories_length(self, configure_sdk, tmp_dir):
        path = os.path.join(tmp_dir, "ckpt.npz")
        self._run_and_save(configure_sdk, path, num_steps=15)
        ckpt = configure_sdk.load_checkpoint(path)
        assert len(ckpt['loss_history']) == 15
        assert len(ckpt['efficiency_history']) == 15
        assert len(ckpt['grad_max_history']) == 15

    def test_best_step_valid(self, configure_sdk, tmp_dir):
        path = os.path.join(tmp_dir, "ckpt.npz")
        self._run_and_save(configure_sdk, path, num_steps=15)
        ckpt = configure_sdk.load_checkpoint(path)
        assert 0 <= ckpt['best_step'] < 15
        np.testing.assert_array_equal(ckpt['theta_best'], ckpt['theta_history'][ckpt['best_step']])

    def test_opt_state_present(self, configure_sdk, tmp_dir):
        path = os.path.join(tmp_dir, "ckpt.npz")
        self._run_and_save(configure_sdk, path)
        ckpt = configure_sdk.load_checkpoint(path)
        assert 'opt_state_b64' in ckpt
        assert len(ckpt['opt_state_b64']) > 0

    def test_metadata(self, configure_sdk, tmp_dir):
        path = os.path.join(tmp_dir, "ckpt.npz")
        self._run_and_save(configure_sdk, path)
        ckpt = configure_sdk.load_checkpoint(path)
        assert 'metadata_json' in ckpt
        assert 'timestamp' in ckpt['metadata_json']

    def test_empty_results_raises(self, configure_sdk):
        with pytest.raises(ValueError, match="empty"):
            configure_sdk.save_checkpoint([], "/tmp/bad.npz")

    def test_missing_file_raises(self, configure_sdk):
        with pytest.raises(FileNotFoundError):
            configure_sdk.load_checkpoint("/tmp/nonexistent_xyz.npz")
