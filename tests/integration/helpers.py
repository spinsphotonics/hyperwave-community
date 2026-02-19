"""Shared test constants and helpers."""

import numpy as np

NUM_STEPS = 30
THETA_SHAPE = (20, 20)
BASELINE_FILE = "/tmp/test_baseline_30steps.npz"


def make_test_inputs():
    rng = np.random.RandomState(123)
    theta = rng.rand(*THETA_SHAPE).astype(np.float32) * 0.5 + 0.25
    source = rng.rand(2, 6, 1, 2, 2).astype(np.float32)
    return theta, source


def run_opt(hwc, theta, source, num_steps, checkpoint=None, auto_checkpoint=False, checkpoint_path=None):
    kwargs = dict(
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
        num_steps=num_steps,
        learning_rate=0.01,
        gpu_type="B200",
        auto_checkpoint=auto_checkpoint,
    )
    if checkpoint_path:
        kwargs['checkpoint_path'] = checkpoint_path
    if checkpoint is not None:
        kwargs['checkpoint'] = checkpoint
    else:
        kwargs['theta'] = theta
    return list(hwc.run_optimization(**kwargs))
