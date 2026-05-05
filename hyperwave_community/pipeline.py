"""Local pipeline functions for inverse design.

surgery(), check_drc(), and export_gds() run locally on CPU.
No cloud call, no credits charged.

optimize() calls the cloud and is implemented separately
(requires the cloud endpoint to be deployed).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from hyperwave_community.types import Design, DrcReport


# ---------------------------------------------------------------------------
# surgery()
# ---------------------------------------------------------------------------

def surgery(
    design: Design,
    min_feature_size: float = 0.105,
    pixel_size: float = 0.0175,
) -> Design:
    """Remove small features and fill small holes.

    Non-differentiable cleanup step. Runs locally on CPU, no credits.

    Args:
        design: Design to clean up.
        min_feature_size: Minimum feature size in um.
        pixel_size: Physical pixel size in um.

    Returns:
        New Design with cleaned thetas and surgery metadata.
    """
    from scipy import ndimage
    from hyperwave_community.structure import density

    min_area_px = int(np.pi * (min_feature_size / pixel_size / 2) ** 2)
    min_area_px = max(min_area_px, 1)

    new_thetas = {}
    total_removed = 0
    total_filled = 0

    for name, theta in design.thetas.items():
        mask = design.design_mask(name)

        import jax.numpy as jnp
        d = np.array(density(jnp.array(theta), radius=float(design.density_filter_radius)))
        binary = (d >= 0.5).astype(np.int32)
        binary_design = binary.copy()
        binary_design[~mask] = 0

        # Count and remove small solid islands
        solid_labeled, n_solid = ndimage.label(binary_design & mask)
        solid_sizes = ndimage.sum(
            np.ones_like(binary_design), solid_labeled, range(1, n_solid + 1))
        removed = sum(1 for s in solid_sizes if s < min_area_px)

        patched = binary_design.copy()
        for i, size in enumerate(solid_sizes):
            if size < min_area_px:
                patched[solid_labeled == (i + 1)] = 0

        # Count and fill small holes
        void_region = (1 - binary_design) & mask
        void_labeled, n_void = ndimage.label(void_region)
        void_sizes = ndimage.sum(
            np.ones_like(binary_design), void_labeled, range(1, n_void + 1))
        filled = sum(1 for s in void_sizes if s < min_area_px)

        for i, size in enumerate(void_sizes):
            if size < min_area_px:
                patched[void_labeled == (i + 1)] = 1

        # Replace design region with patched binary
        theta_clean = np.array(theta, dtype=np.float32, copy=True)
        theta_clean[mask] = patched[mask].astype(np.float32)

        new_thetas[name] = theta_clean
        total_removed += removed
        total_filled += filled

    return Design(
        thetas=new_thetas,
        density_filter_radius=design.density_filter_radius,
        efficiency=design.efficiency,
        phase="surgery",
        step=design.step,
        removed_islands=total_removed,
        filled_holes=total_filled,
    )


# ---------------------------------------------------------------------------
# check_drc()
# ---------------------------------------------------------------------------

def check_drc(
    design: Design,
    disk_radius: int = 3,
    pixel_size: float = 0.0175,
) -> DrcReport:
    """Run design rule check via morphological opening.

    Checks minimum feature size (CD) and minimum gap. Runs locally
    on CPU, no credits charged.

    Args:
        design: Design to check.
        disk_radius: Morphological disk radius in pixels.
        pixel_size: Physical pixel size in um.

    Returns:
        DrcReport with violation counts, percentages, and pass/fail.
    """
    from skimage.morphology import disk, opening
    from hyperwave_community.structure import density

    name = design.layer_names[0]
    theta = design.thetas[name]
    mask = design.design_mask(name)

    import jax.numpy as jnp
    d = np.array(density(jnp.array(theta), radius=float(design.density_filter_radius)))
    binary = (d >= 0.5).astype(bool)
    binary_design = binary & mask
    void_design = (~binary) & mask

    design_pixels = int(mask.sum())
    d_vals = d[mask]
    binarization_score = float(1.0 - np.mean(4 * d_vals * (1 - d_vals)))

    selem = disk(disk_radius)
    solid_opened = opening(binary_design, selem)
    cd_violations = int((binary_design & ~solid_opened).sum())

    void_opened = opening(void_design, selem)
    gap_violations = int((void_design & ~void_opened).sum())

    min_feature_nm = (2 * disk_radius + 1) * pixel_size * 1000
    cd_pct = 100.0 * cd_violations / design_pixels if design_pixels > 0 else 0.0
    gap_pct = 100.0 * gap_violations / design_pixels if design_pixels > 0 else 0.0

    return DrcReport(
        cd_violations=cd_violations,
        cd_pct=cd_pct,
        gap_violations=gap_violations,
        gap_pct=gap_pct,
        binarization_score=binarization_score,
        disk_radius=disk_radius,
        min_feature_nm=min_feature_nm,
        min_gap_nm=min_feature_nm,
        design_pixels=design_pixels,
    )


# ---------------------------------------------------------------------------
# export_gds()
# ---------------------------------------------------------------------------

def export_gds(
    design: Design,
    filename: str = "output.gds",
    layer: Tuple[int, int] = (1, 0),
    pixel_size: float = 0.0175,
) -> str:
    """Export design to GDSII file.

    Runs locally on CPU, no credits charged.

    Args:
        design: Design to export.
        filename: Output GDS filename.
        layer: GDS layer tuple (layer_number, datatype).
        pixel_size: Physical pixel size in um.

    Returns:
        Absolute path to the generated GDS file.
    """
    from hyperwave_community.structure import density
    from hyperwave_community.data_io import generate_gds_from_density

    theta = design.theta
    import jax.numpy as jnp
    d = np.array(density(jnp.array(theta), radius=float(design.density_filter_radius)))

    return generate_gds_from_density(
        density_array=d,
        level=0.5,
        output_filename=filename,
        resolution=pixel_size,
    )
