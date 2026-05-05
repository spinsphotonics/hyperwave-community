"""High-level layer stack builder for inverse design.

Wraps the existing Layer/create_structure into a simpler interface
matching the tutorial API.

Usage:
    layer_stack = hwc.LayerStack(
        grid=0.035,          # FDTD grid spacing in um
        wavelength=1.55,     # um
        total_height=4.22,   # um
    )
    layer_stack.add_layer("box", thickness=2.0, index=1.44)
    layer_stack.add_layer("etch", thickness=0.110, index=3.48,
                          design_layer=True, initial_value=0.5)
    layer_stack.add_layer("clad", thickness=2.0, index=1.44)

    grid = layer_stack.build()
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from hyperwave_community.structure import Layer, create_structure, density


@dataclass
class _LayerSpec:
    name: str
    thickness: float
    index: float
    design_layer: bool = False
    initial_value: float = 0.5
    density_eta: float = 0.5


@dataclass
class GridInfo:
    """Result of LayerStack.build()."""
    layers: List[Layer]
    shape: Tuple[int, int, int]
    design_layers_info: List[Dict[str, Any]]
    recipe_params: Dict[str, Any]
    freq_band: Tuple[float, float, int]
    pixel_size: float
    grid: float


class LayerStack:
    """High-level layer stack builder.

    Args:
        grid: FDTD grid spacing in um (e.g. 0.035 for 35nm).
        wavelength: Operating wavelength in um (e.g. 1.55).
        total_height: Total stack height in um (computed from layers if None).
    """

    def __init__(self, grid: float, wavelength: float,
                 total_height: Optional[float] = None):
        self.grid = grid
        self.pixel_size = grid / 2
        self.wavelength = wavelength
        self.total_height = total_height
        self._layers: List[_LayerSpec] = []

    def add_layer(
        self,
        name: str,
        thickness: float,
        index: float,
        design_layer: bool = False,
        initial_value: float = 0.5,
        density_eta: float = 0.5,
    ) -> None:
        """Add a layer to the stack.

        Args:
            name: Layer name (e.g. "etch", "slab", "box").
            thickness: Physical thickness in um.
            index: Refractive index of the material.
            design_layer: If True, this layer has optimizable design variables.
            initial_value: Initial theta value for design layers (0-1).
            density_eta: Projection threshold for this layer. Controls the
                solid/void balance in the Heaviside projection:
                  0.50 = balanced (equal solid and void). Default, safe.
                  0.55-0.60 = wider gaps (use for layers with larger min
                              gap spec, e.g. Cisco W2/W3 at 150nm gap).
                  0.45-0.50 = wider features (for layers needing thicker
                              solid regions).
                Must be in [0.45, 0.75]. Values outside this range produce
                degenerate projections.
        """
        if not 0.45 <= density_eta <= 0.75:
            raise ValueError(
                f"density_eta must be in [0.45, 0.75], got {density_eta}. "
                f"Use 0.5 for balanced, 0.55-0.60 for wider gaps.")
        self._layers.append(_LayerSpec(
            name=name, thickness=thickness, index=index,
            design_layer=design_layer, initial_value=initial_value,
            density_eta=density_eta,
        ))

    def build(self, nx: Optional[int] = None, ny: Optional[int] = None) -> GridInfo:
        """Build the layer stack into a structure.

        Args:
            nx: X dimension in theta pixels. If None, must be set later.
            ny: Y dimension in theta pixels. If None, uses nx.

        Returns:
            GridInfo with layers, shape, design info, and recipe params.
        """
        if nx is None:
            if self.total_height is not None:
                nx = int(round(self.total_height / self.pixel_size))
                nx = nx + (nx % 2)
            else:
                raise ValueError("nx required when total_height not set")
        if ny is None:
            ny = nx
        ny = ny + (ny % 2)
        nx = nx + (nx % 2)

        dx = self.grid
        ps = self.pixel_size
        wl_px = self.wavelength / dx
        freq = 2 * np.pi / wl_px
        freq_band = (float(freq), float(freq), 1)

        hw_layers = []
        design_info = []
        z_cursor = 0

        for spec in self._layers:
            h_px = int(round(spec.thickness / dx))
            eps = spec.index ** 2

            if spec.design_layer:
                theta = jnp.full((nx, ny), spec.initial_value, dtype=jnp.float32)
                perm_values = (1.0, eps)
                design_info.append({
                    "name": spec.name,
                    "z_range": (z_cursor, z_cursor + h_px),
                    "eps_range": perm_values,
                    "thickness_px": h_px,
                    "index": spec.index,
                    "density_eta": spec.density_eta,
                })
            else:
                theta = jnp.zeros((nx, ny), dtype=jnp.float32)
                perm_values = eps

            hw_layers.append(Layer(
                density_pattern=theta,
                permittivity_values=perm_values,
                layer_thickness=h_px,
            ))
            z_cursor += h_px

        structure = create_structure(layers=hw_layers, vertical_radius=0)
        Lx = structure.permittivity.shape[1]
        Ly = structure.permittivity.shape[2]
        Lz = structure.permittivity.shape[3]

        layers_template = []
        for i, spec in enumerate(self._layers):
            lyr = hw_layers[i]
            if isinstance(lyr.permittivity_values, tuple):
                pv = [float(v) for v in lyr.permittivity_values]
            else:
                pv = float(lyr.permittivity_values)
            layers_template.append({
                "permittivity_values": pv,
                "layer_thickness": float(lyr.layer_thickness),
                "density_radius": 0,
                "density_alpha": 0,
            })

        recipe_params = {
            "grid_shape": [nx, ny],
            "layers_template": layers_template,
        }

        return GridInfo(
            layers=hw_layers,
            shape=(Lx, Ly, Lz),
            design_layers_info=design_info,
            recipe_params=recipe_params,
            freq_band=freq_band,
            pixel_size=ps,
            grid=dx,
        )
