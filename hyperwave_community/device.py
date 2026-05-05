"""Device configuration for inverse design.

build_device() takes a declarative list of layer specs and produces
everything optimize() needs. One function call, no builder pattern.

Grid convention (matching hyperwave internal):
    grid = permittivity voxel size (pv) in um (e.g. 0.035 = 35nm FDTD)
    theta pixel = grid / 2 (e.g. 17.5nm). Theta is 2x oversampled.
    nx, ny = theta grid dimensions (2x permittivity grid)
    layer thickness = thickness_um / grid (permittivity pixels)
    density_radius = in theta pixels (conic filter operates on theta grid)

Usage:
    device = hwc.build_device(
        layers=[
            {"name": "box", "thickness": 2.0, "index": 1.44},
            {"name": "etch", "thickness": 0.11, "index": 3.48,
             "design": True, "density_radius": 6},
            {"name": "clad", "thickness": 2.0, "index": 1.44},
        ],
        grid=0.035, wavelength=1.55, nx=2000,
    )
    result = hwc.optimize(device, source, mode, phase="freeform", n_steps=100)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from hyperwave_community.structure import Layer, create_structure


@dataclass
class DeviceConfig:
    """Result of build_device(). Pass to optimize()."""
    layers: List[Layer]
    shape: Tuple[int, int, int]
    design_layers_info: List[Dict[str, Any]]
    recipe_params: Dict[str, Any]
    freq_band: Tuple[float, float, int]
    pixel_size: float
    grid: float


def build_device(
    layers: List[Dict[str, Any]],
    grid: float,
    wavelength: float,
    nx: int,
    ny: Optional[int] = None,
) -> DeviceConfig:
    """Build a device configuration from layer specifications.

    Args:
        layers: List of layer dicts. Each dict has:
            name (str): Layer name, e.g. "etch", "box", "W1".
            thickness (float): Physical thickness in um.
            index (float): Refractive index of the material.
            design (bool): If True, this is an optimizable layer.
                Default False.
            initial_value (float): Initial theta for design layers (0-1).
                Default 0.5.
            density_radius (int): Conic filter radius in theta pixels.
                Required for design layers. The theta grid is 2x finer
                than the FDTD grid, so R=8 at 17.5nm theta pixel gives
                min feature = 8 * 17.5nm = 140nm.
            density_eta (float): Projection threshold (0.45-0.75).
                Controls solid/void balance:
                  0.50 = balanced (default, safe)
                  0.55-0.60 = wider gaps (for larger min gap specs)
                Default 0.5.
        grid: FDTD grid spacing in um (permittivity voxel size).
            E.g. 0.035 for 35nm. Theta grid = grid/2.
        wavelength: Operating wavelength in um.
        nx: X dimension in theta pixels (2x FDTD grid).
        ny: Y dimension in theta pixels. Defaults to nx.

    Returns:
        DeviceConfig to pass to optimize().
    """
    if ny is None:
        ny = nx
    nx = nx + (nx % 2)
    ny = ny + (ny % 2)

    dx = grid
    pixel_size = dx / 2
    wl_px = wavelength / dx
    freq = 2 * np.pi / wl_px
    freq_band = (float(freq), float(freq), 1)

    hw_layers = []
    design_info = []
    z_cursor = 0

    for spec in layers:
        name = spec["name"]
        thickness = spec["thickness"]
        index = spec["index"]
        is_design = spec.get("design", False)
        h_px = int(round(thickness / dx))
        eps = index ** 2

        if is_design:
            initial_value = spec.get("initial_value", 0.5)
            density_radius = spec.get("density_radius")
            density_eta = spec.get("density_eta", 0.5)

            if density_radius is None:
                raise ValueError(
                    f"Layer '{name}': density_radius is required for design layers. "
                    f"It sets the conic filter radius in theta pixels "
                    f"(theta pixel = {pixel_size*1000:.1f}nm).")

            if not 0.45 <= density_eta <= 0.75:
                raise ValueError(
                    f"Layer '{name}': density_eta must be in [0.45, 0.75], "
                    f"got {density_eta}.")

            theta = jnp.full((nx, ny), initial_value, dtype=jnp.float32)
            perm_values = (1.0, eps)

            design_info.append({
                "name": name,
                "theta": np.array(theta),
                "z_range": (z_cursor, z_cursor + h_px),
                "eps_range": perm_values,
                "density_radius": int(density_radius),
                "density_eta": float(density_eta),
                "waveguide_mask": None,
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
    for i, spec in enumerate(layers):
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

    return DeviceConfig(
        layers=hw_layers,
        shape=(Lx, Ly, Lz),
        design_layers_info=design_info,
        recipe_params=recipe_params,
        freq_band=freq_band,
        pixel_size=pixel_size,
        grid=dx,
    )
