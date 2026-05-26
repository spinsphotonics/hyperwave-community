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

from dataclasses import dataclass
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
    vertical_radius: int = 2

    def get_source_config(
        self,
        layer_name: str,
        waveguide_y_center: Optional[float] = None,
        waveguide_y_width: Optional[float] = None,
        x_position: Optional[int] = None,
        mode_pad: int = 15,
    ) -> Dict[str, Any]:
        """Compute mode source bounds from device geometry.

        Returns a dict with perpendicular_bounds, z_bounds, and
        source_position ready to pass to create_mode_source().

        Args:
            layer_name: Design layer name (e.g. "sin", "etch").
            waveguide_y_center: Waveguide center in um. If None, uses
                the center stored in the layer spec from build_device().
            waveguide_y_width: Waveguide width in um. If None, uses
                the width stored in the layer spec from build_device().
            x_position: Source x position in permittivity pixels.
                If None, defaults to 5 pixels past the absorber width.
            mode_pad: Padding around the waveguide for mode solve (pixels).

        Returns:
            Dict with: source_position, perpendicular_bounds, z_bounds
        """
        dl = None
        for d in self.design_layers_info:
            if d["name"] == layer_name:
                dl = d
                break
        if dl is None:
            raise ValueError(f"Layer '{layer_name}' not found in design_layers_info")

        Ly = self.shape[1]
        Lz = self.shape[2]
        z_start, z_end = dl["z_range"]
        z_mid = (z_start + z_end) // 2
        wg_height_px = z_end - z_start

        if waveguide_y_center is None:
            waveguide_y_center = dl.get("waveguide_y_center_px")
        else:
            waveguide_y_center = int(round(waveguide_y_center / self.grid))
        if waveguide_y_width is None:
            waveguide_y_width = dl.get("waveguide_y_width_px")
        else:
            waveguide_y_width = int(round(waveguide_y_width / self.grid))

        if waveguide_y_center is None or waveguide_y_width is None:
            raise ValueError(
                f"Layer '{layer_name}': waveguide_y_center and waveguide_y_width "
                f"not set. Pass them to build_device() in the layer spec or "
                f"provide them directly to get_source_config().")

        y_half = waveguide_y_width // 2
        yl = max(0, waveguide_y_center - y_half - mode_pad)
        yh = min(Ly, waveguide_y_center + y_half + mode_pad)
        zl = max(0, z_mid - wg_height_px // 2 - mode_pad)
        zh = min(Lz, z_mid + wg_height_px // 2 + mode_pad)

        if yl >= yh:
            raise ValueError(
                f"Waveguide center ({waveguide_y_center}px) is outside "
                f"grid Y bounds (0-{Ly}px).")
        if zl >= zh:
            raise ValueError(
                f"Layer z_range ({dl['z_range']}) produces invalid "
                f"z_bounds ({zl}, {zh}).")

        if x_position is None:
            x_position = min(75, self.shape[0] - 10)

        return {
            "source_position": x_position,
            "perpendicular_bounds": (yl, yh),
            "z_bounds": (zl, zh),
        }


def _build_device_from_specs(
    layers: List[Dict[str, Any]],
    grid: float,
    wavelength: float,
    nx: int,
    ny: Optional[int] = None,
    vertical_radius: int = 2,
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
            waveguide_width (float): Waveguide width in um (optional).
                Used by get_source_config() to compute mode solve bounds.
            waveguide_y_center (float): Waveguide center Y position in um
                (optional). Defaults to center of the Y dimension.
        grid: FDTD grid spacing in um (permittivity voxel size).
            E.g. 0.035 for 35nm. Theta grid = grid/2.
        wavelength: Operating wavelength in um.
        nx: X dimension in theta pixels (2x FDTD grid).
        ny: Y dimension in theta pixels. Defaults to nx.
        vertical_radius: Vertical blur radius at layer interfaces (pixels).
            Smooths permittivity transitions between layers. Default 2.

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
    _eps_clad_cache = {}

    def _infer_eps_clad(spec, idx, layers):
        """Infer cladding permittivity for a design layer from adjacent layers."""
        explicit = spec.get("cladding_index")
        if explicit is not None:
            return explicit ** 2
        for adj in [idx - 1, idx + 1]:
            if 0 <= adj < len(layers) and not layers[adj].get("design", False):
                return layers[adj]["index"] ** 2
        import warnings
        warnings.warn(
            f"Layer '{spec['name']}': no adjacent non-design layer found. "
            f"Using eps_clad=1.0 (vacuum). Set cladding_index in the layer "
            f"spec to override.", stacklevel=3)
        return 1.0

    for i, spec in enumerate(layers):
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
            eps_clad = _infer_eps_clad(spec, i, layers)
            _eps_clad_cache[i] = eps_clad
            perm_values = (eps_clad, eps)

            wg_mask = spec.get("waveguide_mask")
            if wg_mask is None:
                wg_mask = np.zeros((nx, ny), dtype=bool)
            else:
                wg_mask = np.array(wg_mask, dtype=bool)

            wg_width_um = spec.get("waveguide_width")
            wg_y_center_um = spec.get("waveguide_y_center")
            wg_y_center_px = None
            wg_y_width_px = None
            if wg_width_um is not None:
                wg_y_width_px = int(round(wg_width_um / dx))
                if wg_y_center_um is not None:
                    wg_y_center_px = int(round(wg_y_center_um / dx))
                else:
                    wg_y_center_px = (ny // 2) // 2

            dl_info = {
                "name": name,
                "theta": np.array(theta),
                "z_range": (z_cursor, z_cursor + h_px),
                "eps_range": perm_values,
                "density_radius": int(density_radius),
                "density_eta": float(density_eta),
                "waveguide_mask": wg_mask,
                "waveguide_y_center_px": wg_y_center_px,
                "waveguide_y_width_px": wg_y_width_px,
            }
            design_info.append(dl_info)
        else:
            theta = jnp.zeros((nx, ny), dtype=jnp.float32)
            perm_values = eps

        hw_layers.append(Layer(
            density_pattern=theta,
            permittivity_values=perm_values,
            layer_thickness=h_px,
        ))
        z_cursor += h_px

    structure = create_structure(layers=hw_layers, vertical_radius=vertical_radius)
    Lx = structure.permittivity.shape[1]
    Ly = structure.permittivity.shape[2]
    Lz = structure.permittivity.shape[3]

    # Build layers_template in the format the deployed Modal optimizer expects:
    # {"layer_type": "design_N"|"slab", "params": {"permittivity": ..., "thickness": ...}}
    layers_template = []
    design_idx = 0
    for i, spec in enumerate(layers):
        is_design = spec.get("design", False)
        eps = spec["index"] ** 2
        h_px = int(round(spec["thickness"] / dx))

        if is_design:
            eps_clad_t = _eps_clad_cache.get(i, _infer_eps_clad(spec, i, layers))
            layers_template.append({
                "layer_type": f"design_{design_idx}",
                "params": {
                    "permittivity": (float(eps_clad_t), float(eps)),
                    "thickness": h_px,
                }
            })
            design_idx += 1
        else:
            layers_template.append({
                "layer_type": "slab",
                "params": {
                    "permittivity": float(eps),
                    "thickness": h_px,
                }
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
        vertical_radius=vertical_radius,
    )


def build_device(*args, **kwargs) -> DeviceConfig:
    """Deprecated. Use optimize(layers=..., theta=...) directly."""
    import warnings
    warnings.warn(
        "build_device() is deprecated. Pass layers= and theta= to optimize() directly.",
        DeprecationWarning, stacklevel=2,
    )
    return _build_device_from_specs(*args, **kwargs)
