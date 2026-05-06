"""Core types for the inverse design API.

Design flows between optimization phases. OptimizationResult wraps what
optimize() returns. DrcReport wraps what check_drc() returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Design:
    """Optimization state that flows between phases.

    Carries the design variables (theta arrays per layer), the filter
    settings, and metadata about which phase produced this design.
    Pass between optimize() calls, and to surgery(), check_drc(), export_gds().
    """

    thetas: Dict[str, np.ndarray]
    density_radii: Dict[str, int]
    efficiency: float = 0.0
    phase: str = ""
    step: int = 0

    removed_islands: int = 0
    filled_holes: int = 0

    @property
    def theta(self) -> np.ndarray:
        """First (or only) layer's theta array."""
        return next(iter(self.thetas.values()))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.theta.shape

    @property
    def layer_names(self) -> List[str]:
        return list(self.thetas.keys())

    @property
    def density_filter_radius(self) -> int:
        """First layer's density radius (backward compat)."""
        return next(iter(self.density_radii.values()))

    def design_mask(self, layer_name: Optional[str] = None) -> np.ndarray:
        name = layer_name or self.layer_names[0]
        return np.ones(self.thetas[name].shape, dtype=bool)


@dataclass
class OptimizationResult:
    """Result from optimize()."""

    design: Design
    history: List[Dict[str, Any]]
    phase: str
    n_steps: int
    best_efficiency: float = 0.0
    best_step: int = 0
    schedule_config: Dict[str, Any] = field(default_factory=dict)
    n_steps_planned: int = 0
    run_id: str = ""
    optimizer_state_bytes: Optional[bytes] = None

    def save(self, path: str) -> str:
        from hyperwave_community.checkpoint import save_checkpoint
        return save_checkpoint(self, path)


@dataclass
class DrcReport:
    """Result from check_drc()."""

    cd_violations: int
    cd_pct: float
    gap_violations: int
    gap_pct: float
    binarization_score: float
    disk_radius: int
    min_feature_nm: float
    min_gap_nm: float
    design_pixels: int
    status: str = ""
    passed: bool = False

    def __post_init__(self):
        self.passed = self.cd_pct < 1.0 and self.gap_pct < 1.0
        self.status = (
            "PASS (< 1% violations)" if self.passed
            else f"FAIL (CD {self.cd_pct:.1f}%, gap {self.gap_pct:.1f}%)"
        )
