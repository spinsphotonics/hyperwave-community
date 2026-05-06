"""Checkpoint save/load system for optimization state.

Saves everything needed to resume an optimization exactly where it left
off: thetas, optimizer state, schedule position, and full config. Stored
locally by default as a directory of portable files.

Directory format:
    checkpoint_dir/
        meta.json           # step, phase, schedule config, efficiency, run_id
        theta_{name}.npy    # per-layer design variables (numpy, portable)
        optimizer_state.pkl # optax state (pickle, version-dependent)
        history.json        # per-step metrics

Usage:
    # Save after optimization
    results = optimize(device, source, mode, n_steps=100, ...)
    save_checkpoint(results, "~/checkpoints/my_run")

    # Resume later (any machine with same hyperwave version)
    ckpt = load_checkpoint("~/checkpoints/my_run")
    results2 = optimize(device, source, mode, n_steps=100,
                        resume_from=ckpt)
    # Beta picks up at step 100's value, Adam moments preserved

Modal Volume storage:
    Each run gets a UUID-based run_id. Checkpoints on Modal go to
    /checkpoints/{run_id}/step_{N:04d}/ -- never overwritten because
    run_id is unique per optimize() call. No more shared run_name
    collisions.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Checkpoint:
    """Complete optimization state for save/load/resume.

    Contains everything needed to continue an optimization exactly
    where it left off: design variables, optimizer moments, schedule
    position, and full configuration.

    Attributes:
        run_id: Unique identifier for this optimization run (UUID).
        step: Current optimization step.
        n_steps_planned: Total steps the schedule was designed for.
            Used to compute position in beta/lambda ramps on resume.
        phase: Which phase produced this checkpoint.
        efficiency: Efficiency at this step.

        thetas: Per-layer design variables (layer_name -> 2D array).
        density_radii: Per-layer conic filter radii.

        schedule_config: Full schedule parameters used for this run.
            Includes beta_init, beta_max, learning_rate, fab params, etc.
            On resume, the system uses step/n_steps_planned to compute
            where in the ramp to continue.

        optimizer_state_bytes: Serialized optax optimizer state (pickle).
            If None or deserialization fails, optimizer is reinitialized.
            This means you lose Adam momentum but can still resume.

        history: Per-step metrics from the run so far.
    """

    # Identity
    run_id: str
    step: int
    n_steps_planned: int
    phase: str
    efficiency: float

    # Design state
    thetas: Dict[str, np.ndarray]
    density_radii: Dict[str, int]

    # Schedule (for resuming ramps)
    schedule_config: Dict[str, Any]

    # Optimizer state (opaque, version-dependent)
    optimizer_state_bytes: Optional[bytes] = None

    # History
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: str = ""
    hyperwave_version: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def beta_current(self) -> float:
        """Compute current beta value based on position in schedule."""
        cfg = self.schedule_config
        beta_init = cfg.get("beta_init", 1.0)
        beta_max = cfg.get("beta_max", 64.0)
        schedule = cfg.get("beta_schedule", "linear")

        if self.n_steps_planned <= 1:
            return beta_max

        progress = min(self.step / max(self.n_steps_planned - 1, 1), 1.0)

        if schedule == "linear":
            return beta_init + progress * (beta_max - beta_init)
        elif schedule == "exponential":
            return beta_init * (beta_max / beta_init) ** progress
        else:
            return beta_init + progress * (beta_max - beta_init)

    @property
    def layer_names(self) -> List[str]:
        return list(self.thetas.keys())

    @property
    def density_filter_radius(self) -> int:
        """First layer's density radius (backward compat)."""
        return next(iter(self.density_radii.values()))


def generate_run_id() -> str:
    """Generate a unique run ID: timestamp + short UUID.

    Format: YYYYMMDD_HHMMSS_xxxxxxxx
    Timestamp for human readability, UUID suffix for uniqueness.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"{ts}_{uid}"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    result: Any,  # OptimizationResult or Checkpoint
    path: str,
    schedule_config: Optional[Dict[str, Any]] = None,
    n_steps_planned: Optional[int] = None,
    run_id: Optional[str] = None,
) -> str:
    """Save optimization state to a local directory.

    Args:
        result: An OptimizationResult (from optimize()) or a Checkpoint.
        path: Directory path to save to. Created if it doesn't exist.
            If path already exists, a step-numbered subdirectory is created
            to avoid overwriting.
        schedule_config: Schedule parameters (beta_init, beta_max, etc.).
            Required when saving an OptimizationResult. Ignored for Checkpoint.
        n_steps_planned: Total steps the schedule was designed for.
            Required when saving an OptimizationResult.
        run_id: Unique run identifier. Auto-generated if not provided.

    Returns:
        Absolute path to the saved checkpoint directory.
    """
    from hyperwave_community.types import OptimizationResult

    path = os.path.expanduser(path)

    if isinstance(result, Checkpoint):
        ckpt = result
    elif isinstance(result, OptimizationResult):
        # Pull from result fields if not explicitly provided
        _run_id = run_id or getattr(result, "run_id", "") or generate_run_id()
        _schedule = schedule_config or getattr(result, "schedule_config", {})
        _n_planned = n_steps_planned or getattr(result, "n_steps_planned", 0) or result.n_steps
        ckpt = Checkpoint(
            run_id=_run_id,
            step=result.design.step,  # actual step number, not count
            n_steps_planned=_n_planned,
            phase=result.phase,
            efficiency=result.design.efficiency,
            thetas={name: np.array(arr) for name, arr in result.design.thetas.items()},
            density_radii=dict(result.design.density_radii),
            schedule_config=_schedule,
            optimizer_state_bytes=getattr(result, "optimizer_state_bytes", None),
            history=result.history,
        )
    else:
        raise TypeError(f"Expected OptimizationResult or Checkpoint, got {type(result)}")

    # Create step-numbered subdirectory to prevent overwrites
    save_dir = os.path.join(path, f"step_{ckpt.step:04d}")
    os.makedirs(save_dir, exist_ok=True)

    # Save metadata as JSON (sanitize NaN/inf to null for JSON compat)
    def _sanitize(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, dict):
            return {k: _sanitize(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_sanitize(val) for val in v]
        return v

    meta = _sanitize({
        "run_id": ckpt.run_id,
        "step": ckpt.step,
        "n_steps_planned": ckpt.n_steps_planned,
        "phase": ckpt.phase,
        "efficiency": ckpt.efficiency,
        "density_radii": ckpt.density_radii,
        "schedule_config": ckpt.schedule_config,
        "layer_names": ckpt.layer_names,
        "created_at": ckpt.created_at,
        "has_optimizer_state": ckpt.optimizer_state_bytes is not None,
    })
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save thetas as portable numpy files
    for name, theta in ckpt.thetas.items():
        np.save(os.path.join(save_dir, f"theta_{name}.npy"), np.array(theta))

    # Save optimizer state (pickle, version-dependent)
    if ckpt.optimizer_state_bytes is not None:
        with open(os.path.join(save_dir, "optimizer_state.pkl"), "wb") as f:
            f.write(ckpt.optimizer_state_bytes)

    # Save history
    if ckpt.history:
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(ckpt.history, f, indent=2)

    abs_path = os.path.abspath(save_dir)
    print(f"Checkpoint saved: {abs_path}")
    print(f"  Run ID: {ckpt.run_id}")
    print(f"  Step: {ckpt.step}/{ckpt.n_steps_planned}")
    print(f"  Efficiency: {ckpt.efficiency:.4f}")
    print(f"  Layers: {ckpt.layer_names}")
    print(f"  Beta: {ckpt.beta_current:.1f}")
    if ckpt.optimizer_state_bytes:
        print(f"  Optimizer state: saved ({len(ckpt.optimizer_state_bytes)} bytes)")
    else:
        print("  Optimizer state: not saved (will reinitialize on resume)")

    return abs_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> Checkpoint:
    """Load a checkpoint from a local directory.

    Args:
        path: Path to the checkpoint directory (the step_NNNN dir).
            If path points to a parent run directory, loads the latest step.

    Returns:
        Checkpoint object ready to pass to optimize(resume_from=...).
    """
    path = os.path.expanduser(path)

    # If path is a parent directory, find the latest step subdirectory
    if not os.path.exists(os.path.join(path, "meta.json")):
        step_dirs = sorted([
            d for d in os.listdir(path)
            if d.startswith("step_") and os.path.isdir(os.path.join(path, d))
        ])
        if not step_dirs:
            raise FileNotFoundError(
                f"No checkpoint found at {path}. "
                f"Expected a directory with meta.json or step_NNNN subdirectories."
            )
        path = os.path.join(path, step_dirs[-1])
        print(f"Loading latest checkpoint: {path}")

    # Load metadata
    with open(os.path.join(path, "meta.json"), "r") as f:
        meta = json.load(f)

    # Load thetas
    thetas = {}
    for name in meta["layer_names"]:
        theta_path = os.path.join(path, f"theta_{name}.npy")
        thetas[name] = np.load(theta_path)

    # Load optimizer state (with fallback)
    optimizer_state_bytes = None
    opt_path = os.path.join(path, "optimizer_state.pkl")
    if os.path.exists(opt_path):
        try:
            with open(opt_path, "rb") as f:
                optimizer_state_bytes = f.read()
        except Exception as e:
            print(f"Warning: could not load optimizer state: {e}")
            print("  Optimizer will be reinitialized on resume.")

    # Load history
    history = []
    hist_path = os.path.join(path, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            history = json.load(f)

    ckpt = Checkpoint(
        run_id=meta["run_id"],
        step=meta["step"],
        n_steps_planned=meta["n_steps_planned"],
        phase=meta["phase"],
        efficiency=meta["efficiency"],
        thetas=thetas,
        density_radii=meta.get("density_radii", {"default": meta.get("density_filter_radius", 6)}),
        schedule_config=meta.get("schedule_config", {}),
        optimizer_state_bytes=optimizer_state_bytes,
        history=history,
        created_at=meta.get("created_at", ""),
    )

    print(f"Checkpoint loaded: {path}")
    print(f"  Run ID: {ckpt.run_id}")
    print(f"  Step: {ckpt.step}/{ckpt.n_steps_planned}")
    print(f"  Efficiency: {ckpt.efficiency:.4f}")
    print(f"  Beta at checkpoint: {ckpt.beta_current:.1f}")
    if optimizer_state_bytes:
        print(f"  Optimizer state: loaded ({len(optimizer_state_bytes)} bytes)")
    else:
        print("  Optimizer state: not available (will reinitialize)")

    return ckpt


# ---------------------------------------------------------------------------
# List checkpoints
# ---------------------------------------------------------------------------

def list_checkpoints(path: str) -> List[Dict[str, Any]]:
    """List all checkpoints in a directory.

    Args:
        path: Parent directory containing step_NNNN subdirectories.

    Returns:
        List of dicts with step, efficiency, phase, created_at for each.
    """
    path = os.path.expanduser(path)
    results = []

    if not os.path.exists(path):
        return results

    for entry in sorted(os.listdir(path)):
        meta_path = os.path.join(path, entry, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            results.append({
                "dir": entry,
                "step": meta["step"],
                "efficiency": meta["efficiency"],
                "phase": meta["phase"],
                "run_id": meta["run_id"],
                "created_at": meta.get("created_at", ""),
            })

    return results
