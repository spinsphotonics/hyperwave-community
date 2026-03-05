"""Consolidated visualization module for HyperWave SDK.

All plotting functions in one place. Each function defers its matplotlib
import so the module is importable without a display backend.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Branding helper
# ---------------------------------------------------------------------------

def _apply_branding(fig):
    """Add a subtle 'HyperWave' watermark in the bottom-right corner."""
    fig.text(
        0.99, 0.01, "HyperWave",
        fontsize=7,
        color="#cccccc",
        ha="right",
        va="bottom",
        transform=fig.transFigure,
    )


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def plot_convergence(
    steps,
    errors,
    *,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Convergence history",
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot FDTD convergence history.

    Args:
        steps: List or array of step numbers where convergence was checked.
        errors: List or array of error values at each check (can be per-freq
            arrays or scalars).
        figsize: Figure size in inches.
        title: Plot title.
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    steps_arr = np.asarray(steps)
    errors_arr = np.asarray(errors)

    # errors may be (n_checks, n_freq) or (n_checks,)
    if errors_arr.ndim == 2:
        for freq_idx in range(errors_arr.shape[1]):
            ax.semilogy(
                steps_arr,
                errors_arr[:, freq_idx],
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=f"Freq {freq_idx}",
            )
        ax.legend(fontsize=9)
    else:
        # Scalar per check, or max across freqs already taken
        if errors_arr.ndim == 1 and len(errors_arr) > 0:
            # Each element might itself be an array (max-error per freq)
            try:
                max_errors = [float(np.max(np.asarray(e))) for e in errors]
            except Exception:
                max_errors = errors_arr
            ax.semilogy(
                steps_arr,
                max_errors,
                marker="o",
                markersize=3,
                linewidth=1.2,
                color="#2563eb",
            )
        else:
            ax.semilogy(
                steps_arr,
                errors_arr,
                marker="o",
                markersize=3,
                linewidth=1.2,
                color="#2563eb",
            )

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Max error", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="medium")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Fields (monitor data from simulation results)
# ---------------------------------------------------------------------------

def plot_fields(
    monitor_data,
    monitor_names,
    *,
    field_component: str = "all",
    freq_idx: int = 0,
    figsize=None,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot monitor field intensity from simulation results.

    Iterates over named monitors and displays a 2D heatmap of the chosen
    field component for the given frequency index.

    Args:
        monitor_data: List of arrays, one per monitor.  Each array has shape
            ``(n_freq, 6, nx, ny, nz)``.
        monitor_names: Dict mapping monitor name to index in *monitor_data*.
        field_component: ``'all'`` (total intensity), ``'E'``, ``'H'``, or
            one of ``'Ex','Ey','Ez','Hx','Hy','Hz'``.
        freq_idx: Frequency index to plot.
        figsize: Figure size (auto-computed if *None*).
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    n_monitors = len(monitor_names)
    if figsize is None:
        figsize = (6 * min(n_monitors, 3), 5 * max(1, (n_monitors + 2) // 3))

    cols = min(n_monitors, 3)
    rows = max(1, (n_monitors + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True, squeeze=False)

    comp_map = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}

    for plot_idx, (name, mon_idx) in enumerate(monitor_names.items()):
        ax = axes[plot_idx // cols][plot_idx % cols]
        data = np.asarray(monitor_data[mon_idx])

        # Extract the requested component
        if field_component in comp_map:
            field_3d = np.abs(data[freq_idx, comp_map[field_component]])
            label = f"|{field_component}|"
        elif field_component == "E":
            field_3d = np.sqrt(np.sum(np.abs(data[freq_idx, 0:3]) ** 2, axis=0))
            label = "|E|"
        elif field_component == "H":
            field_3d = np.sqrt(np.sum(np.abs(data[freq_idx, 3:6]) ** 2, axis=0))
            label = "|H|"
        else:
            # 'all': total intensity |E|^2 + |H|^2
            field_3d = np.sum(np.abs(data[freq_idx, 0:3]) ** 2, axis=0) + \
                       np.sum(np.abs(data[freq_idx, 3:6]) ** 2, axis=0)
            label = "Intensity"

        # Collapse to 2D by squeezing or averaging the thinnest axis
        field_2d, xlabel, ylabel = _collapse_to_2d(field_3d)

        im = ax.imshow(field_2d.T, cmap="inferno", origin="upper", aspect="auto")
        ax.set_title(f"{name} - {label}", fontsize=13, fontweight="medium")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused axes
    for idx in range(n_monitors, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Mode profile
# ---------------------------------------------------------------------------

def plot_mode(
    mode_field,
    beta,
    mode_num,
    *,
    propagation_axis: str = "x",
    figsize: Tuple[int, int] = (14, 8),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot mode field components (Ex, Ey, Ez magnitudes).

    Args:
        mode_field: Array of shape ``(n_freq, 6, x, y, z)`` or
            ``(n_freq, 3, x, y, z)`` (E-only).
        beta: Propagation constant (scalar or per-freq array).
        mode_num: Mode number label.
        propagation_axis: ``'x'`` or ``'y'``.
        figsize: Figure size.
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    mode_field = np.asarray(mode_field)
    beta_val = float(np.asarray(beta).ravel()[0])

    # Squeeze along the propagation axis to get a 2D cross-section
    if propagation_axis == "x":
        mode_slice = np.squeeze(mode_field[0, :3, 0, :, :])  # (3, y, z)
        xlabel, ylabel = "Y (cells)", "Z (cells)"
    else:
        mode_slice = np.squeeze(mode_field[0, :3, :, 0, :])  # (3, x, z)
        xlabel, ylabel = "X (cells)", "Z (cells)"

    component_names = ["Ex", "Ey", "Ez"]

    # Auto-crop to the region where the mode energy is concentrated.
    total_mag = np.sqrt(sum(np.abs(mode_slice[i]) ** 2 for i in range(3)))
    threshold = 0.01 * float(np.max(total_mag))
    nonzero = np.argwhere(total_mag > threshold)
    if len(nonzero) > 0:
        margin = max(5, int(0.1 * max(total_mag.shape)))
        r_min = max(0, int(nonzero[:, 0].min()) - margin)
        r_max = min(total_mag.shape[0], int(nonzero[:, 0].max()) + margin + 1)
        c_min = max(0, int(nonzero[:, 1].min()) - margin)
        c_max = min(total_mag.shape[1], int(nonzero[:, 1].max()) + margin + 1)
    else:
        r_min, r_max = 0, total_mag.shape[0]
        c_min, c_max = 0, total_mag.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle(
        f"Mode {mode_num} E-field profile (beta = {beta_val:.4f})",
        fontsize=13,
        fontweight="medium",
    )

    for i, (ax, comp) in enumerate(zip(axes, component_names)):
        mag = np.abs(mode_slice[i])[r_min:r_max, c_min:c_max]
        vmax = float(np.max(mag)) or 1.0
        im = ax.imshow(mag.T, cmap="viridis", origin="upper", vmin=0, vmax=vmax, aspect="equal")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{comp} magnitude", fontsize=13, fontweight="medium")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Quick monitor view
# ---------------------------------------------------------------------------

def plot_monitors(
    results,
    *,
    component: str = "Hz",
    freq_idx: int = 0,
    cmap: str = "inferno",
    figsize: Tuple[int, int] = (8, 5),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Quick view of each monitor's field data.

    Produces one figure per monitor with a 2D slice of the chosen component.

    Args:
        results: Dict returned by ``simulate()`` with keys
            ``'monitor_data'``, ``'monitor_names'``.
        component: ``'Ex','Ey','Ez','Hx','Hy','Hz'``, ``'|E|'``, ``'|H|'``,
            or ``'all'`` (total intensity).
        freq_idx: Frequency index.
        cmap: Matplotlib colormap.
        figsize: Per-monitor figure size.
        show: Whether to call ``plt.show()``.
        save_path: If given, save the last figure. For multiple monitors
            the path is suffixed with the monitor name.

    Returns:
        List of matplotlib ``Figure`` objects (one per monitor).
    """
    import matplotlib.pyplot as plt

    comp_map = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
    names = list(results["monitor_names"])
    n_monitors = len(names)

    if n_monitors == 0:
        return None

    # Compute grid dimensions (prefer 2 columns)
    cols = min(n_monitors, 2)
    rows = max(1, (n_monitors + cols - 1) // cols)
    fig_w, fig_h = figsize
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_w * cols, fig_h * rows),
        constrained_layout=True,
        squeeze=False,
    )

    for plot_idx, name in enumerate(names):
        ax = axes[plot_idx // cols][plot_idx % cols]
        data = np.asarray(results["monitor_data"][name])

        # Extract field
        if component in comp_map:
            field_3d = data[freq_idx, comp_map[component]]
        elif component == "|E|":
            field_3d = np.sqrt(np.sum(np.abs(data[freq_idx, 0:3]) ** 2, axis=0))
        elif component == "|H|":
            field_3d = np.sqrt(np.sum(np.abs(data[freq_idx, 3:6]) ** 2, axis=0))
        elif component == "all":
            field_3d = np.sqrt(
                np.sum(np.abs(data[freq_idx, 0:3]) ** 2, axis=0)
                + np.sum(np.abs(data[freq_idx, 3:6]) ** 2, axis=0)
            )
        else:
            raise ValueError(f"Unknown component: {component}")

        # For complex data, take magnitude
        if np.iscomplexobj(field_3d):
            field_3d = np.abs(field_3d)

        field_2d, xlabel, ylabel = _collapse_to_2d(field_3d)

        im = ax.imshow(field_2d.T, cmap=cmap, origin="upper", aspect="auto")
        ax.set_title(f"{name} - {component} (freq {freq_idx})", fontsize=13, fontweight="medium")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused axes
    for idx in range(n_monitors, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Monitor layout (structure cross-section + monitor overlay)
# ---------------------------------------------------------------------------

def plot_monitor_layout(
    permittivity,
    monitors,
    *,
    axis: str = "z",
    position: Optional[int] = None,
    source_position: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot structure cross-section with monitor rectangles overlaid.

    Args:
        permittivity: Array of shape ``(3, nx, ny, nz)``.
        monitors: A ``MonitorSet`` object, or a list of ``Monitor`` objects.
            If a MonitorSet, names are read from its mapping.
        axis: Slice axis (``'x'``, ``'y'``, or ``'z'``).
        position: Slice position. Defaults to the midpoint.
        source_position: Optional X position to draw a source-plane line.
        figsize: Figure size.
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    perm = np.asarray(permittivity)
    if perm.ndim == 4:
        nx, ny, nz = perm.shape[1], perm.shape[2], perm.shape[3]
        eps_real = np.real(perm[0])
    else:
        nx, ny, nz = perm.shape
        eps_real = np.real(perm)

    # Resolve MonitorSet vs list
    if hasattr(monitors, "monitors") and hasattr(monitors, "mapping"):
        monitor_list = monitors.monitors
        monitor_mapping = monitors.mapping
    else:
        monitor_list = monitors
        monitor_mapping = None

    if position is None:
        position = {"x": nx // 2, "y": ny // 2, "z": nz // 2}[axis]

    # Slice the structure
    if axis == "x":
        struct_slice = eps_real[position, :, :]
        extent = [0, ny, nz, 0]
        xlabel, ylabel = "Y (cells)", "Z (cells)"
    elif axis == "y":
        struct_slice = eps_real[:, position, :]
        extent = [0, nx, nz, 0]
        xlabel, ylabel = "X (cells)", "Z (cells)"
    else:
        struct_slice = eps_real[:, :, position]
        extent = [0, nx, ny, 0]
        xlabel, ylabel = "X (cells)", "Y (cells)"

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(struct_slice.T, extent=extent, cmap="PuOr", alpha=0.3, aspect="auto", origin="upper")

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

    for i, mon in enumerate(monitor_list):
        shape = mon.shape
        offset = mon.offset

        # Resolve name
        mon_name = None
        if monitor_mapping:
            for mname, midx in monitor_mapping.items():
                if midx == i:
                    mon_name = mname
                    break
        if mon_name is None:
            mon_name = f"Monitor {i}"

        mx_s, my_s, mz_s = offset
        mx_e = mx_s + shape[0]
        my_e = my_s + shape[1]
        mz_e = mz_s + shape[2]

        # Only draw monitors that intersect the slice plane
        rect_params = None
        if axis == "x" and shape[0] > 1 and mx_s <= position <= mx_e:
            rect_params = ((my_s, mz_s), my_e - my_s, mz_e - mz_s)
        elif axis == "y" and shape[1] > 1 and my_s <= position <= my_e:
            rect_params = ((mx_s, mz_s), mx_e - mx_s, mz_e - mz_s)
        elif axis == "z" and mz_s <= position < mz_e:
            rect_params = ((mx_s, my_s), mx_e - mx_s, my_e - my_s)

        if rect_params is None:
            continue

        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            rect_params[0],
            rect_params[1],
            rect_params[2],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            label=f"{mon_name} {shape}@{offset}",
        )
        ax.add_patch(rect)
        text_x = rect_params[0][0] + rect_params[1] + 5
        text_y = rect_params[0][1] + rect_params[2] / 2
        ax.text(text_x, text_y, mon_name, ha="left", va="center", fontsize=9, color=color)

    # Source line
    if source_position is not None:
        if axis in ("y", "z"):
            ax.axvline(x=source_position, color="yellow", linewidth=3, linestyle="--", alpha=0.8, label=f"Source (X={source_position})")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"Monitor layout ({axis.upper()} = {position}) | {nx} x {ny} x {nz}",
        fontsize=13,
        fontweight="medium",
    )
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_aspect("equal")

    fig.set_tight_layout(True)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Absorption mask
# ---------------------------------------------------------------------------

def plot_absorption_mask(
    absorption_mask,
    *,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = "Greys",
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot absorption mask slices (XY, XZ, YZ at center).

    Args:
        absorption_mask: Array of shape ``(3, xx, yy, zz)``.  Only the first
            component (Ex) is visualized.
        figsize: Figure size.
        cmap: Colormap.
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    mask = np.asarray(absorption_mask)
    if mask.ndim != 4 or mask.shape[0] < 1:
        raise ValueError(f"absorption_mask must have shape (3, xx, yy, zz), got {mask.shape}")

    _, xx, yy, zz = mask.shape
    data = np.sqrt(np.abs(mask[0]))  # sqrt scaling for visibility
    vmin, vmax = float(data.min()), float(data.max())

    x_mid, y_mid, z_mid = xx // 2, yy // 2, zz // 2

    slices = [
        ("XY", data[:, :, z_mid], (0, xx - 1, 0, yy - 1), "X index", "Y index"),
        ("XZ", data[:, y_mid, :], (0, xx - 1, 0, zz - 1), "X index", "Z index"),
        ("YZ", data[x_mid, :, :], (0, yy - 1, 0, zz - 1), "Y index", "Z index"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle("Absorption mask (sqrt-scaled)", fontsize=13, fontweight="medium")

    for ax, (label, sl, ext, xlab, ylab) in zip(axes, slices):
        im = ax.imshow(sl.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, aspect="equal")
        ax.set_title(f"{label} slice", fontsize=13, fontweight="medium")
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(r"$\sqrt{\alpha}$", fontsize=11)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig




# ---------------------------------------------------------------------------
# Theta (2D layout)
# ---------------------------------------------------------------------------

def plot_theta(
    theta,
    *,
    figsize: Tuple[int, int] = (10, 4),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot a 2D density layout (theta).

    Args:
        theta: 2D array of material densities.
        figsize: Figure size.
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(np.asarray(theta).T, cmap="gray", origin="lower", aspect="equal")
    ax.set_title("2D Layout (theta)", fontsize=13, fontweight="medium")
    ax.set_xlabel("x (cells)", fontsize=11)
    ax.set_ylabel("y (cells)", fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Material density", fontsize=11)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig

# ---------------------------------------------------------------------------
# Structure (permittivity)
# ---------------------------------------------------------------------------

def plot_structure(
    permittivity,
    conductivity=None,
    *,
    show_permittivity: bool = True,
    show_conductivity: bool = False,
    axis: Optional[str] = None,
    position: Optional[int] = None,
    view_mode: str = "2d",
    figsize=None,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot structure permittivity cross-sections.

    When *axis* is ``None`` the default dual-view is shown (XY at mid-Z and
    XZ at mid-Y). When an axis is specified a single slice is produced.
    When *view_mode* is ``"3d"``, three orthogonal cross-sections are shown
    on matplotlib 3D axes.

    Args:
        permittivity: Array with shape ``(3, nx, ny, nz)`` or a ``Structure``
            object (the ``.permittivity`` attribute is used).
        conductivity: Deprecated, ignored.
        axis: ``'x'``, ``'y'``, ``'z'``, or *None* for default dual view.
        position: Slice position along the chosen axis.
        view_mode: ``"2d"`` (default) or ``"3d"`` for a 3D orthogonal view.
        figsize: Figure size (auto-computed if *None*).
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    # Accept a Structure object
    if hasattr(permittivity, "permittivity"):
        struct = permittivity
        perm_arr = np.asarray(struct.permittivity)
    else:
        perm_arr = np.asarray(permittivity)

    nx, ny, nz = perm_arr.shape[1], perm_arr.shape[2], perm_arr.shape[3]
    cmap_p = "PuOr"

    if view_mode == "3d":
        return _plot_structure_3d_mpl(
            perm_arr, nx, ny, nz, cmap_p,
            figsize=figsize, show=show, save_path=save_path,
        )

    def _get_slice(arr, ax, pos):
        if ax == "x":
            return arr[0, pos, :, :], "y", "z"
        elif ax == "y":
            return arr[0, :, pos, :], "x", "z"
        else:
            return arr[0, :, :, pos], "x", "y"

    if axis is not None:
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
        if position is None:
            position = {"x": nx, "y": ny, "z": nz}[axis] // 2

        if figsize is None:
            figsize = (6, 5)
        fig, ax_obj = plt.subplots(figsize=figsize, constrained_layout=True)

        sl, xlab, ylab = _get_slice(perm_arr, axis, position)
        _plot_slice(ax_obj, sl, cmap_p, perm_arr.min(), perm_arr.max(),
                    f"Permittivity: {xlab}-{ylab} at {axis}={position}", xlab, ylab, fig)
    else:
        # Default dual view
        mid_z = nz // 2
        mid_y = ny // 2
        if figsize is None:
            figsize = (12, 5)
        fig, axes_arr = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, squeeze=False)

        _plot_slice(axes_arr[0, 0], perm_arr[0, :, :, mid_z], cmap_p,
                    perm_arr.min(), perm_arr.max(),
                    f"Permittivity: x-y at z={mid_z}", "x", "y", fig)
        _plot_slice(axes_arr[0, 1], perm_arr[0, :, mid_y, :], cmap_p,
                    perm_arr.min(), perm_arr.max(),
                    f"Permittivity: x-z at y={mid_y}", "x", "z", fig)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Simulation overview (fields + absorbers + structure)
# ---------------------------------------------------------------------------

def plot_simulation_overview(
    fields,
    absorption_mask,
    permittivity,
    *,
    freq_idx: int = 0,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Combined overview: structure, absorbers, E-field, H-field, line cuts.

    Args:
        fields: Full field output of shape ``(n_freq, 6, nx, ny, nz)``.
        absorption_mask: Absorption array of shape ``(3, nx, ny, nz)``.
        permittivity: Permittivity array of shape ``(3, nx, ny, nz)``.
        freq_idx: Frequency index to visualize.
        figsize: Figure size.
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    fields_np = np.asarray(fields)
    perm_np = np.asarray(permittivity)
    abs_np = np.asarray(absorption_mask)

    z_mid = perm_np.shape[3] // 2
    y_mid = perm_np.shape[2] // 2

    # Derived quantities
    perm_xy = np.real(perm_np[0, :, :, z_mid])
    abs_xy = abs_np[0, :, :, z_mid]
    E_mag = np.sqrt(np.sum(np.abs(fields_np[freq_idx, 0:3, :, :, z_mid]) ** 2, axis=0))
    Hz = np.abs(fields_np[freq_idx, 5, :, :, z_mid])

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle("Simulation overview", fontsize=13, fontweight="medium")

    # Row 1: structure, absorbers, |E|
    im0 = axes[0, 0].imshow(perm_xy.T, origin="upper", cmap="viridis")
    axes[0, 0].set_title("Permittivity", fontsize=13, fontweight="medium")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(abs_xy.T, origin="upper", cmap="plasma")
    axes[0, 1].set_title("Absorbers", fontsize=13, fontweight="medium")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[0, 2].imshow(E_mag.T, origin="upper", cmap="viridis")
    axes[0, 2].set_title("|E|", fontsize=13, fontweight="medium")
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    # Row 2: |Hz|, |E| line cut, absorption line cut
    im3 = axes[1, 0].imshow(Hz.T, origin="upper", cmap="plasma")
    axes[1, 0].set_title("|Hz|", fontsize=13, fontweight="medium")
    fig.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    center_y = E_mag.shape[1] // 2
    E_line = E_mag[:, center_y]
    axes[1, 1].plot(E_line, color="#2563eb", linewidth=1.2)
    axes[1, 1].set_title("|E| along X", fontsize=13, fontweight="medium")
    axes[1, 1].set_xlabel("X position", fontsize=11)
    axes[1, 1].set_ylabel("Magnitude", fontsize=11)
    axes[1, 1].grid(True, alpha=0.2, linewidth=0.5)

    abs_line = abs_xy[:, center_y]
    axes[1, 2].plot(abs_line, color="#dc2626", linewidth=1.2)
    axes[1, 2].set_title("Absorption along X", fontsize=13, fontweight="medium")
    axes[1, 2].set_xlabel("X position", fontsize=11)
    axes[1, 2].set_ylabel("Conductivity", fontsize=11)
    axes[1, 2].grid(True, alpha=0.2, linewidth=0.5)

    for row in axes:
        for ax in row:
            ax.set_xlabel(ax.get_xlabel() or "X", fontsize=11)
            ax.set_ylabel(ax.get_ylabel() or "Y", fontsize=11)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# 3D structure (plotly)
# ---------------------------------------------------------------------------

def plot_structure_3d(
    permittivity,
    conductivity=None,
    *,
    n_isosurfaces: int = 5,
    save_html: Optional[str] = None,
    show: bool = True,
):
    """Interactive 3D visualization of permittivity isosurfaces using plotly.

    Args:
        permittivity: Array with shape ``(3, nx, ny, nz)`` or a ``Structure``.
        conductivity: Optional conductivity array of same shape.
        n_isosurfaces: Number of isosurface levels.
        save_html: If given, save to an HTML file.
        show: Whether to display the figure.

    Returns:
        A plotly ``Figure`` object.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for 3D visualization. Install with: pip install plotly")

    # Accept Structure
    if hasattr(permittivity, "permittivity"):
        struct = permittivity
        cond_arr = np.asarray(struct.conductivity) if conductivity is None else np.asarray(conductivity)
        perm_arr = np.asarray(struct.permittivity)
    else:
        perm_arr = np.asarray(permittivity)
        cond_arr = np.asarray(conductivity) if conductivity is not None else None

    nx, ny, nz = perm_arr.shape[1], perm_arr.shape[2], perm_arr.shape[3]
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    perm_values = perm_arr[0]

    pmin, pmax = float(perm_values.min()), float(perm_values.max())
    isovalues = np.linspace(pmin, pmax, n_isosurfaces + 2)[1:-1]

    fig = go.Figure()

    for i, iso in enumerate(isovalues):
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=perm_values.flatten(),
            isomin=float(iso) - 0.05, isomax=float(iso) + 0.05,
            opacity=0.3,
            colorscale="Viridis",
            name=f"eps {float(iso):.2f}",
            showscale=(i == 0),
            surface_count=1,
        ))

    fig.update_layout(
        title=dict(text="3D Structure", x=0.5, font=dict(size=14)),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="cube",
        ),
        width=800, height=500,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    if save_html:
        fig.write_html(save_html)
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# GDS layout
# ---------------------------------------------------------------------------

def plot_gds(
    gds_filepath,
    density_array=None,
    *,
    figsize: Tuple[int, int] = (12, 6),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot GDS layout with optional density comparison.

    Args:
        gds_filepath: Path to a ``.gds`` file.
        density_array: Optional 2D numpy array for side-by-side comparison.
        figsize: Figure size.
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    try:
        import gdstk
    except ImportError:
        raise ImportError("gdstk is required for GDS visualization. Install with: pip install gdstk")

    lib = gdstk.read_gds(gds_filepath)
    cell = lib.top_level()[0]
    polygons = cell.get_polygons()

    if density_array is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        density_array = np.asarray(density_array)
        im = ax1.imshow(
            density_array, cmap="gray", origin="upper",
            extent=[0, density_array.shape[1], 0, density_array.shape[0]],
        )
        ax1.set_title(
            f"Original density ({density_array.shape[0]} x {density_array.shape[1]})",
            fontsize=13, fontweight="medium",
        )
        ax1.set_xlabel("X (px)", fontsize=11)
        ax1.set_ylabel("Y (px)", fontsize=11)
        ax1.grid(True, alpha=0.2, linewidth=0.5)
        fig.colorbar(im, ax=ax1, shrink=0.8, label="Density")
        ax = ax2
    else:
        fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]), constrained_layout=True)

    for poly in polygons:
        patch = plt.Polygon(poly.points, alpha=0.7, edgecolor="none", facecolor="#3b82f6", linewidth=0)
        ax.add_patch(patch)

    # Set limits
    if density_array is not None:
        ax.set_xlim(0, density_array.shape[1])
        ax.set_ylim(0, density_array.shape[0])
    elif polygons:
        all_pts = np.vstack([p.points for p in polygons])
        margin = 1
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

    ax.set_aspect("equal")
    ax.set_xlabel("X (GDS units)", fontsize=11)
    ax.set_ylabel("Y (GDS units)", fontsize=11)
    ax.set_title(f"GDS polygons ({len(polygons)})", fontsize=13, fontweight="medium")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plot_structure_3d_mpl(perm_arr, nx, ny, nz, cmap, *, figsize=None, show=True, save_path=None):
    """Render a volumetric 3D view of the device structure using isosurfaces."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if figsize is None:
        figsize = (10, 8)

    eps = np.real(perm_arr[0])  # shape (nx, ny, nz)

    # Downsample to at most ~80 cells per axis for performance.
    max_cells = 80
    factors = [max(1, s // max_cells) for s in eps.shape]
    if any(f > 1 for f in factors):
        from scipy.ndimage import zoom
        ds = zoom(eps, [1.0 / f for f in factors], order=1)
    else:
        ds = eps

    # Threshold: midpoint between cladding (low eps) and core (high eps).
    eps_min, eps_max = float(ds.min()), float(ds.max())
    threshold = (eps_min + eps_max) / 2.0

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    try:
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(ds, level=threshold)
        # Scale vertices back to original grid coordinates.
        scale = np.array([factors[0], factors[1], factors[2]], dtype=float)
        verts = verts * scale
        mesh = Poly3DCollection(
            verts[faces],
            alpha=0.7,
            edgecolor=(0.2, 0.2, 0.2, 0.15),
            linewidth=0.1,
        )
        mesh.set_facecolor("#4a90d9")
        ax.add_collection3d(mesh)
    except Exception:
        # Fallback: voxel plot of high-permittivity regions.
        vox_max = 40
        vox_factors = [max(1, s // vox_max) for s in eps.shape]
        if any(f > 1 for f in vox_factors):
            from scipy.ndimage import zoom as _zoom
            vox = _zoom(eps, [1.0 / f for f in vox_factors], order=1)
        else:
            vox = eps
        vox_min, vox_max_val = float(vox.min()), float(vox.max())
        vox_thresh = (vox_min + vox_max_val) / 2.0
        mask = vox > vox_thresh
        colors = np.empty(mask.shape, dtype=object)
        colors[mask] = "#4a90d9"
        ax.voxels(mask, facecolors=colors, edgecolor=(0.3, 0.3, 0.3, 0.1), alpha=0.7)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_xlabel("X (cells)", fontsize=11)
    ax.set_ylabel("Y (cells)", fontsize=11)
    ax.set_zlabel("Z (cells)", fontsize=11)
    ax.set_title("3D Structure", fontsize=13, fontweight="medium")

    # Semi-transparent cladding bounding box.
    box_verts = np.array([
        [0, 0, 0], [nx, 0, 0], [nx, ny, 0], [0, ny, 0],  # bottom
        [0, 0, nz], [nx, 0, nz], [nx, ny, nz], [0, ny, nz],  # top
    ], dtype=float)
    faces_idx = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]
    box_faces = [[box_verts[i] for i in face] for face in faces_idx]
    cladding = Poly3DCollection(
        box_faces,
        alpha=0.12,
        facecolor="#c8a882",
        edgecolor=(0.6, 0.5, 0.4, 0.3),
        linewidth=0.5,
    )
    ax.add_collection3d(cladding)

    _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


def _collapse_to_2d(field_3d):
    """Reduce a 3D field to 2D for display, returning (array, xlabel, ylabel).

    Squeezes singleton dimensions; if none, averages the smallest axis.
    """
    field_3d = np.asarray(field_3d)
    if field_3d.shape[0] == 1:
        return field_3d[0, :, :], "Y", "Z"
    if field_3d.shape[1] == 1:
        return field_3d[:, 0, :], "X", "Z"
    if field_3d.shape[2] == 1:
        return field_3d[:, :, 0], "X", "Y"

    min_dim = int(np.argmin(field_3d.shape))
    if min_dim == 0:
        return np.mean(field_3d, axis=0), "Y", "Z"
    elif min_dim == 1:
        return np.mean(field_3d, axis=1), "X", "Z"
    else:
        return np.mean(field_3d, axis=2), "X", "Y"


def _plot_slice(ax, data, cmap, vmin, vmax, title, xlabel, ylabel, fig):
    """Render a single 2D slice onto an axis."""
    im = ax.imshow(data.T, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    ax.set_title(title, fontsize=13, fontweight="medium")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)


def _split_save_path(path: str):
    """Split 'path/to/file.png' into ('path/to/file', '.png')."""
    import os
    base, ext = os.path.splitext(path)
    return base, ext or ".png"
