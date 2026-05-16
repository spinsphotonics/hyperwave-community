"""Consolidated visualization module for HyperWave SDK.

All plotting functions in one place. Each function defers its matplotlib
import so the module is importable without a display backend.
"""

from __future__ import annotations

import os
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

    Example::

        hwc.plot_mode(
            mode_field=mode_info["field"],
            beta=mode_info["beta"],
            mode_num=0,
            propagation_axis="x",
        )
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

    Produces two separate figures: port monitors in a 2-column grid,
    and field slice monitors (xy_mid, xz_mid, etc.) full-width below.

    Args:
        results: Dict returned by ``simulate()`` with keys
            ``'monitor_data'``, ``'monitor_names'``.
        component: ``'Ex','Ey','Ez','Hx','Hy','Hz'``, ``'|E|'``, ``'|H|'``,
            or ``'all'`` (total intensity).
        freq_idx: Frequency index.
        cmap: Matplotlib colormap.
        figsize: Per-subplot figure size (width, height).
        show: Whether to call ``plt.show()``.
        save_path: If given, saves with ``_ports`` / ``_fields`` suffixes.

    Returns:
        When ``show=False``: tuple of (ports_fig, fields_fig), or a single
        Figure if only one category exists. ``None`` when ``show=True``.

    Example::

        hwc.plot_monitors(results, component="Hz")
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    comp_map = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
    names = list(results["monitor_names"])
    n_monitors = len(names)

    if n_monitors == 0:
        return None

    def _extract_field_2d(data, component, freq_idx):
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
        if np.iscomplexobj(field_3d):
            field_3d = np.abs(field_3d)
        return _collapse_to_2d(field_3d)

    # Separate port monitors from field slice monitors (like xy_mid)
    port_names = [n for n in names if not n.startswith("xy_") and not n.startswith("xz_") and not n.startswith("yz_")]
    field_names = [n for n in names if n.startswith("xy_") or n.startswith("xz_") or n.startswith("yz_")]

    fig_w, fig_h = figsize
    n_ports = len(port_names)
    n_fields = len(field_names)
    figs = []

    # --- Figure 1: Port monitors in 2-column grid ---
    if n_ports > 0:
        cols = min(n_ports, 2)
        port_rows = (n_ports + cols - 1) // cols
        fig_ports = plt.figure(figsize=(fig_w * cols, fig_h * port_rows))
        gs_ports = gridspec.GridSpec(
            port_rows, cols,
            figure=fig_ports,
            hspace=0.3,
            wspace=0.3,
        )
        for plot_idx, name in enumerate(port_names):
            ax = fig_ports.add_subplot(gs_ports[plot_idx // cols, plot_idx % cols])
            data = np.asarray(results["monitor_data"][name])
            field_2d, xlabel, ylabel = _extract_field_2d(data, component, freq_idx)
            im = ax.imshow(field_2d.T, cmap=cmap, origin="upper", aspect="equal")
            ax.set_title(f"{name} - {component} (freq {freq_idx})", fontsize=13, fontweight="medium")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(False)
            fig_ports.colorbar(im, ax=ax, shrink=0.8)
        for idx in range(n_ports, port_rows * cols):
            ax = fig_ports.add_subplot(gs_ports[idx // cols, idx % cols])
            ax.set_visible(False)
        _apply_branding(fig_ports)
        if save_path:
            base, ext = os.path.splitext(save_path)
            fig_ports.savefig(f"{base}_ports{ext}", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        figs.append(fig_ports)

    # --- Figure 2: Field slice monitors (xy_mid, xz_mid, etc.) ---
    if n_fields > 0:
        # Compute true aspect ratios from data to size the figure correctly
        field_data_shapes = []
        for name in field_names:
            data = np.asarray(results["monitor_data"][name])
            field_2d, _, _ = _extract_field_2d(data, component, freq_idx)
            field_data_shapes.append((field_2d.shape[0], field_2d.shape[1]))
        total_fig_w = fig_w * 2
        total_fig_h = 0
        field_height_ratios = []
        for (dw, dh) in field_data_shapes:
            ratio = dh / dw if dw > 0 else 0.5
            field_height_ratios.append(ratio)
            total_fig_h += total_fig_w * ratio
        total_fig_h += fig_h * 0.3 * (n_fields - 1)  # spacing
        fig_fields = plt.figure(figsize=(total_fig_w, max(total_fig_h, fig_h)))
        gs_fields = gridspec.GridSpec(
            n_fields, 1,
            figure=fig_fields,
            height_ratios=field_height_ratios,
            hspace=0.3,
        )
        for field_idx, name in enumerate(field_names):
            ax = fig_fields.add_subplot(gs_fields[field_idx, 0])
            data = np.asarray(results["monitor_data"][name])
            field_2d, xlabel, ylabel = _extract_field_2d(data, component, freq_idx)
            im = ax.imshow(field_2d.T, cmap=cmap, origin="upper", aspect="equal")
            ax.set_title(f"{name} - {component} (freq {freq_idx})", fontsize=13, fontweight="medium")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(False)
            fig_fields.colorbar(im, ax=ax, shrink=0.8)
        _apply_branding(fig_fields)
        if save_path:
            base, ext = os.path.splitext(save_path)
            fig_fields.savefig(f"{base}_fields{ext}", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        figs.append(fig_fields)

    if not figs:
        return None
    if show:
        for f in figs:
            plt.close(f)
        return None
    return tuple(figs) if len(figs) > 1 else figs[0]


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

    Example::

        hwc.plot_monitor_layout(
            structure.permittivity, monitors,
            axis="z", position=z_wg_center, source_position=abs_widths[0],
        )
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

    Example::

        hwc.plot_absorption_mask(absorber)
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
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = "2D Layout (theta)",
    xlabel: Optional[str] = "x (cells)",
    ylabel: Optional[str] = "y (cells)",
    colorbar: bool = True,
    colorbar_label: str = "Material density",
    aspect: str = "equal",
    xlim: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax=None,
    show: bool = True,
    save_path: Optional[str] = None,
    save_dpi: int = 150,
    return_data: bool = False,
):
    """Plot a 2D density layout (theta or density).

    Args:
        theta: 2D array of material densities.
        cmap: Matplotlib colormap name.
        vmin: Min value for colormap. Auto if None.
        vmax: Max value for colormap. Auto if None.
        title: Plot title. None to hide.
        xlabel: X-axis label. None to hide.
        ylabel: Y-axis label. None to hide.
        colorbar: Whether to show colorbar.
        colorbar_label: Label for colorbar.
        aspect: ``"equal"`` (square pixels) or ``"auto"`` (stretch to fill).
        xlim: ``(x_min, x_max)`` pixel range to display.
        ylim: ``(y_min, y_max)`` pixel range to display.
        figsize: Figure size. Auto if None.
        ax: Existing matplotlib Axes to plot on.
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure.
        save_dpi: DPI for saved figure.
        return_data: If True, also return the 2D array.

    Returns:
        The matplotlib ``Figure``, or ``(Figure, ndarray)`` if return_data.

    Example::

        hwc.plot_theta(theta)
        hwc.plot_theta(density, cmap="viridis", vmin=0, vmax=1, title="Filtered density")
        hwc.plot_theta(theta, xlim=(200, 800), ylim=(400, 600), aspect="auto")
    """
    import matplotlib.pyplot as plt

    data = np.asarray(theta)
    own_fig = ax is None
    if own_fig:
        if figsize is None:
            nx, ny = data.shape
            ratio = nx / max(1, ny)
            figsize = (max(6, min(12, ratio * 6)), 6)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    im = ax.imshow(data.T, cmap=cmap, origin="lower", aspect=aspect, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=13, fontweight="medium")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label, fontsize=11)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if own_fig:
        _apply_branding(fig)

    if save_path:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        if return_data:
            return None, data
        return None
    if return_data:
        return fig, data
    return fig


def animate_optimization(
    thetas,
    *,
    layer: Optional[str] = None,
    steps: Optional[List[int]] = None,
    step_stride: int = 1,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    title_fn=None,
    figsize: Optional[Tuple[int, int]] = None,
    interval: int = 200,
    colorbar: bool = True,
    save_path: Optional[str] = None,
    save_dpi: int = 100,
    show: bool = True,
):
    """Create an animation (GIF) of theta evolution during optimization.

    Args:
        thetas: List of 2D arrays (one per step), OR a dict mapping step
            numbers to 2D arrays, OR a list of file paths to ``.npy`` files.
        layer: If thetas is a dict of dicts (multi-layer), which layer to show.
        steps: Specific step indices to include. None = all.
        step_stride: Show every Nth step (e.g. 5 = every 5th). Ignored if steps is set.
        cmap: Colormap.
        vmin: Min value for colormap.
        vmax: Max value for colormap.
        title_fn: Callable ``(step_idx, total) -> str`` for frame titles.
            Default shows "Step {i}/{total}".
        figsize: Figure size. Auto if None.
        interval: Milliseconds between frames.
        colorbar: Show colorbar.
        save_path: Save as GIF/MP4. Extension determines format.
        save_dpi: DPI for saved animation.
        show: Whether to display in notebook (uses HTML for inline display).

    Returns:
        The matplotlib ``FuncAnimation`` object.

    Example::

        # From optimization result
        hwc.animate_optimization(result.history, layer="etch")

        # From list of arrays
        hwc.animate_optimization([theta_step0, theta_step10, theta_step50])

        # Every 5th step, save as GIF
        hwc.animate_optimization(thetas, step_stride=5, save_path="opt.gif")

        # Custom frame titles
        hwc.animate_optimization(thetas,
            title_fn=lambda i, n: f"Iteration {i*10}/{n*10} | eta={effs[i]:.1%}")
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Normalize thetas to a list of 2D arrays
    if isinstance(thetas, dict):
        sorted_keys = sorted(thetas.keys())
        frames = [np.asarray(thetas[k]) for k in sorted_keys]
    elif isinstance(thetas, list) and len(thetas) > 0:
        if isinstance(thetas[0], (str, bytes)):
            frames = [np.load(p) for p in thetas]
        else:
            frames = [np.asarray(t) for t in thetas]
    else:
        raise ValueError("thetas must be a list of arrays, dict, or list of file paths")

    # Handle multi-layer (each frame is a dict)
    if isinstance(frames[0], dict):
        if layer is None:
            layer = list(frames[0].keys())[0]
        frames = [np.asarray(f[layer]) for f in frames]

    # Select steps
    if steps is not None:
        frames = [frames[i] for i in steps if i < len(frames)]
    elif step_stride > 1:
        frames = frames[::step_stride]

    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames to animate")

    nx, ny = frames[0].shape
    if figsize is None:
        ratio = nx / max(1, ny)
        figsize = (max(5, min(10, ratio * 5)), 5)

    if title_fn is None:
        title_fn = lambda i, n: f"Step {i + 1}/{n}"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(frames[0].T, cmap=cmap, origin="lower", aspect="equal",
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ttl = ax.set_title(title_fn(0, n_frames))
    if colorbar:
        fig.colorbar(im, ax=ax)
    fig.tight_layout()

    def update(frame_idx):
        im.set_data(frames[frame_idx].T)
        ttl.set_text(title_fn(frame_idx, n_frames))
        return [im, ttl]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", dpi=save_dpi)
        else:
            anim.save(save_path, dpi=save_dpi)
        print(f"Saved animation to {save_path} ({n_frames} frames)")

    if show:
        try:
            from IPython.display import HTML, display
            display(HTML(anim.to_jshtml()))
        except ImportError:
            plt.show()
        plt.close(fig)
        return None

    return anim

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

    Example::

        hwc.plot_structure(structure, axis="z", position=z_wg_center)
        hwc.plot_structure(structure, view_mode="3d")
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


def plot_structure_slice(
    layers,
    *,
    axis: str = "xz",
    position: Optional[int] = None,
    xlim: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    zlim: Optional[Tuple[int, int]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize=None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colorbar: bool = True,
    colorbar_label: str = "permittivity",
    aspect: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    save_dpi: int = 150,
    cmap: str = "PuOr",
    ax=None,
    return_data: bool = False,
):
    """Plot a 2D cross-section of a layer stack WITHOUT materializing the full 3D array.

    Computes only the requested slice from the layer definitions, using O(nx*nz)
    or O(nx*ny) memory instead of O(nx*ny*nz). For large devices this can be
    1000x less memory than ``plot_structure(create_structure(layers))``.

    Args:
        layers: List of ``Layer`` objects (same as passed to ``create_structure``),
            ordered bottom-to-top.
        axis: ``"xz"`` for XZ cross-section at a given y (default),
            ``"xy"`` for XY cross-section at a given z,
            ``"yz"`` for YZ cross-section at a given x.
        position: Slice position in pixels along the sliced axis.
            Defaults to the midpoint.
        xlim: ``(x_min, x_max)`` pixel range to display on x-axis.
        ylim: ``(y_min, y_max)`` pixel range to display on y-axis.
        zlim: ``(z_min, z_max)`` pixel range to display on z-axis.
        vmin: Min permittivity value for colormap. Auto if None.
        vmax: Max permittivity value for colormap. Auto if None.
        figsize: Figure size tuple. Auto-computed if None.
        title: Plot title. Auto-generated if None.
        xlabel: X-axis label. Auto if None.
        ylabel: Y-axis label. Auto if None.
        colorbar: Whether to show the colorbar.
        colorbar_label: Label for the colorbar.
        aspect: Aspect ratio for imshow (``"auto"`` or ``"equal"``).
        show: Whether to call ``plt.show()``.
        save_path: If given, save the figure to this path.
        save_dpi: DPI for saved figure.
        cmap: Matplotlib colormap name.
        ax: Existing matplotlib Axes to plot on. Creates new figure if None.
        return_data: If True, also return the 2D slice array.

    Returns:
        The matplotlib ``Figure``, or ``(Figure, ndarray)`` if ``return_data=True``.

    Example::

        layers = [clad, etch_layer, slab_layer, box, substrate]
        hwc.plot_structure_slice(layers)
        hwc.plot_structure_slice(layers, axis="xz", zlim=(80, 100), cmap="viridis")
        hwc.plot_structure_slice(layers, axis="xy", position=z_etch, xlim=(200, 800))

        # Plot on existing axes for subplots
        fig, axes = plt.subplots(1, 2)
        hwc.plot_structure_slice(layers, axis="xz", ax=axes[0], show=False)
        hwc.plot_structure_slice(layers, axis="xy", position=z, ax=axes[1])

        # Get raw data
        fig, data = hwc.plot_structure_slice(layers, return_data=True, show=False)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    layer_info = []
    for layer in layers:
        d = np.asarray(layer.density_pattern)
        pv = layer.permittivity_values
        h = int(np.ceil(layer.layer_thickness))
        if isinstance(pv, (tuple, list)):
            eps_lo, eps_hi = float(pv[0]), float(pv[1])
        else:
            eps_lo = eps_hi = float(pv)
        layer_info.append((d, eps_lo, eps_hi, h))

    dnx, dny = layer_info[0][0].shape
    total_z = sum(h for _, _, _, h in layer_info)

    if axis == "xz":
        mid_y = position if position is not None else dny // 2
        slc = np.zeros((dnx, total_z))
        z = 0
        for density, eps_lo, eps_hi, h in layer_info:
            col = density[:, mid_y]
            for zi in range(h):
                slc[:, z + zi] = eps_lo + (eps_hi - eps_lo) * col
            z += h

        auto_title = f"XZ cross-section at y={mid_y}"
        auto_xlabel, auto_ylabel = "x (px)", "z (px)"
        # Effective display range (after xlim/zlim crops)
        ew = (xlim[1] - xlim[0]) if xlim else dnx
        eh = (zlim[1] - zlim[0]) if zlim else total_z
        ratio = ew / max(1, eh)
        auto_figsize = (min(16, max(6, ratio * 3)), 3)
        display_xlim, display_ylim = xlim, zlim

    elif axis == "xy":
        z_target = position if position is not None else total_z // 2
        z = 0
        slc = None
        for density, eps_lo, eps_hi, h in layer_info:
            if z_target >= z and z_target < z + h:
                slc = eps_lo + (eps_hi - eps_lo) * density
                break
            z += h
        if slc is None:
            slc = np.full((dnx, dny), layer_info[-1][1])

        auto_title = f"XY cross-section at z={z_target}"
        auto_xlabel, auto_ylabel = "x (px)", "y (px)"
        ew = (xlim[1] - xlim[0]) if xlim else dnx
        eh = (ylim[1] - ylim[0]) if ylim else dny
        ratio = ew / max(1, eh)
        auto_figsize = (max(5, min(10, ratio * 6)), 6)
        display_xlim, display_ylim = xlim, ylim

    elif axis == "yz":
        mid_x = position if position is not None else dnx // 2
        slc = np.zeros((dny, total_z))
        z = 0
        for density, eps_lo, eps_hi, h in layer_info:
            col = density[mid_x, :]
            for zi in range(h):
                slc[:, z + zi] = eps_lo + (eps_hi - eps_lo) * col
            z += h

        auto_title = f"YZ cross-section at x={mid_x}"
        auto_xlabel, auto_ylabel = "y (px)", "z (px)"
        ew = (ylim[1] - ylim[0]) if ylim else dny
        eh = (zlim[1] - zlim[0]) if zlim else total_z
        ratio = ew / max(1, eh)
        auto_figsize = (min(16, max(6, ratio * 3)), 3)
        display_xlim, display_ylim = ylim, zlim

    else:
        raise ValueError(f"axis must be 'xz', 'xy', or 'yz', got '{axis}'")

    # Flip z-axis so last layer (typically substrate) is at bottom,
    # first layer (typically air) is at top -- natural cross-section view
    if axis in ("xz", "yz"):
        slc = slc[:, ::-1]

    # Auto-select aspect: "auto" when zooming (so zoomed region fills the figure),
    # "equal" when showing the full view (so pixels are square)
    if aspect is None:
        is_zoomed = (xlim is not None or ylim is not None or zlim is not None)
        aspect = "auto" if is_zoomed else "equal"

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize or auto_figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(slc.T, origin="lower", aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel or auto_xlabel)
    ax.set_ylabel(ylabel or auto_ylabel)
    ax.set_title(title or auto_title)
    if colorbar:
        plt.colorbar(im, ax=ax, label=colorbar_label)
    if display_xlim:
        ax.set_xlim(display_xlim)
    if display_ylim:
        ax.set_ylim(display_ylim)

    if own_fig:
        plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    if show:
        plt.show()
        plt.close(fig)
        if return_data:
            return None, slc
        return None
    if return_data:
        return fig, slc
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
        _cond_arr = np.asarray(struct.conductivity) if conductivity is None else np.asarray(conductivity)
        perm_arr = np.asarray(struct.permittivity)
    else:
        perm_arr = np.asarray(permittivity)
        _cond_arr = np.asarray(conductivity) if conductivity is not None else None

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
    ax.set_box_aspect([nx, ny, nz])
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


# ---------------------------------------------------------------------------
# Pipeline visualization
# ---------------------------------------------------------------------------

def plot_phase_summary(results, title=None, show_fields=False, figsize=(10, 4),
                       show=True, save_path=None):
    """Plot efficiency vs step for a single optimization phase.

    Args:
        results: OptimizationResult from optimize().
        title: Plot title. Defaults to phase name.
        show_fields: If True, show final density alongside curve.
        figsize: Figure size.
        show: Whether to display the plot.
        save_path: If provided, save to this path.
    """
    import matplotlib.pyplot as plt

    efficiencies = [h.get("efficiency", 0) * 100 for h in results.history]
    steps = [h.get("step", i) for i, h in enumerate(results.history)]

    if show_fields:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    ax1.plot(steps, efficiencies, "b-", linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Efficiency (%)")
    ax1.set_title(title or results.phase.capitalize())
    ax1.grid(True, alpha=0.3)
    if efficiencies:
        ax1.axhline(max(efficiencies), color="r", linestyle="--", alpha=0.5,
                     label=f"Best: {max(efficiencies):.1f}%")
        ax1.legend()

    if show_fields and hasattr(results.design, "theta"):
        ax2.imshow(results.design.theta.T, origin="lower", cmap="viridis",
                   vmin=0, vmax=1)
        ax2.set_title("Final density")
        ax2.set_xlabel("x (px)")
        ax2.set_ylabel("y (px)")

    plt.tight_layout()
    if save_path:
        base, ext = _split_save_path(save_path)
        plt.savefig(f"{base}{ext}", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_pipeline_summary(phases, labels=None, figsize=(12, 4),
                          show=True, save_path=None):
    """Plot efficiency across multiple phases on a single timeline.

    Args:
        phases: List of OptimizationResult objects.
        labels: List of phase labels. Defaults to result.phase.
        figsize: Figure size.
        show: Whether to display.
        save_path: If provided, save to this path.
    """
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [r.phase.capitalize() for r in phases]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = ["#2563eb", "#7c3aed", "#d97706", "#059669"]
    step_offset = 0

    for i, (result, label) in enumerate(zip(phases, labels)):
        efficiencies = [h.get("efficiency", 0) * 100 for h in result.history]
        steps = [step_offset + h.get("step", j) for j, h in enumerate(result.history)]
        color = colors[i % len(colors)]
        ax.plot(steps, efficiencies, color=color, linewidth=1.5, label=label)
        if steps:
            step_offset = steps[-1]

    ax.set_xlabel("Total steps")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Pipeline Summary")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        base, ext = _split_save_path(save_path)
        plt.savefig(f"{base}{ext}", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# 3D device viewer (emits JSON for the standalone UI)
# ---------------------------------------------------------------------------

# Material name aliases -> canonical short keys used by GDSFactoryDevice3D
_MATERIAL_ALIAS = {
    "silicon": "si",
    "si": "si",
    "silicon_nitride": "sin",
    "sin": "sin",
    "si3n4": "sin",
    "silicon_dioxide": "sio2",
    "sio2": "sio2",
    "oxide": "sio2",
    "air": "air",
}


def show_device_3d(density, layers, pixel_size, mode="auto", monitors=None, gds_polygons=None):
    """Emit geometry data for the standalone UI 3D device viewer.

    Args:
        density: 2D numpy array (nx, ny) with values in [0, 1].
        layers: list of dicts with keys: name, thickness, material, index.
            Design layers also have ``is_design=True``.
        pixel_size: um per pixel.
        mode: ``"auto"`` (default), ``"contour"``, or ``"slab"``.
            - ``"contour"``: extract smooth contours at level 0.5 (binary).
            - ``"slab"``: render design layer as full rectangle (grayscale).
            - ``"auto"``: contour if binarization > 0.8, else slab.
        monitors: optional list of dicts with keys:
            - ``name`` (str): e.g. ``"Input_te0"``, ``"Output_te1"``
            - ``x`` (float): x position in um
            - ``y`` (float): y center in um
            - ``width`` (float): monitor width in um
            - ``orientation`` (float): angle in degrees (0=along y, 90=along x)
    """
    import json

    density = np.asarray(density, dtype=float)
    if density.ndim != 2:
        raise ValueError(f"density must be 2D, got shape {density.shape}")

    nx, ny = density.shape
    x_max = nx * pixel_size
    y_max = ny * pixel_size

    # Always generate density texture (SiN/SiO2 colors with alpha)
    import io
    import base64
    from PIL import Image

    sin_rgb = np.array([176, 176, 168], dtype=np.float64) / 255.0
    sio2_rgb = np.array([224, 232, 240], dtype=np.float64) / 255.0
    d = density.T
    rgb = sio2_rgb + d[..., None] * (sin_rgb - sio2_rgb)
    alpha = d
    rgba_uint8 = (np.concatenate([rgb, alpha[..., None]], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    texture_b64 = base64.b64encode(buf.getvalue()).decode()

    # Binary texture: crisp threshold at 0.5
    d_bin = (d >= 0.5).astype(np.float64)
    rgb_bin = sin_rgb * np.ones_like(rgb)
    rgba_bin = (np.concatenate([rgb_bin, d_bin[..., None]], axis=-1) * 255).astype(np.uint8)
    img_bin = Image.fromarray(rgba_bin)
    buf_bin = io.BytesIO()
    img_bin.save(buf_bin, format="PNG")
    binary_texture_b64 = base64.b64encode(buf_bin.getvalue()).decode()

    def _gds_to_viewer_paths(gds_polys):
        paths = []
        for poly in gds_polys:
            pts = np.asarray(poly.points if hasattr(poly, 'points') else poly)
            if pts.ndim == 2 and len(pts) >= 3:
                paths.append([[float(p[1]), float(p[0])] for p in pts])
        return paths

    # Smooth contour at 0.5 for clean 2D outline
    from skimage.measure import find_contours as _find_contours
    _padded = np.pad(density, 1, mode="constant", constant_values=0)
    _raw = _find_contours(_padded, 0.5)
    smooth_contour = []
    for c in _raw:
        p = (c - 1.0) * pixel_size
        pl = [[float(pt[0]), float(pt[1])] for pt in p]
        if len(pl) >= 3:
            smooth_contour.append(pl)

    # GDS polygons: clean fabrication geometry from gdstk
    if gds_polygons is not None:
        contour_paths = _gds_to_viewer_paths(gds_polygons)
        design_paths = contour_paths

        # Multi-level contours: high-res find_contours + gdstk boolean for holes
        from skimage.measure import find_contours as _fc
        import gdstk as _gdstk
        from .data_io import _build_containment_hierarchy, _is_clockwise
        _pad_d = np.pad(density, 1, mode="constant", constant_values=0)
        levels = [0.2, 0.4, 0.6, 0.8]
        density_contours = []
        for level in levels:
            raw = _fc(_pad_d, level)
            if not raw:
                continue
            gds_polys = [_gdstk.Polygon((c[:, ::-1] - 1) * pixel_size) for c in raw if len(c) >= 3]
            roots, hierarchy = _build_containment_hierarchy(raw)
            final = []
            for root_idx in roots:
                result = [gds_polys[root_idx]]
                children = list(hierarchy[root_idx])
                while children:
                    child_idx = children.pop(0)
                    op = "not" if not _is_clockwise(raw[child_idx]) else "or"
                    result = _gdstk.boolean(result, [gds_polys[child_idx]], op)
                    children.extend(hierarchy[child_idx])
                final.extend(result)
            level_paths = _gds_to_viewer_paths(final)
            if level_paths:
                density_contours.append({"level": level, "paths": level_paths})
    else:
        density_contours = []
        from skimage.measure import find_contours
        padded = np.pad(density, 1, mode="constant", constant_values=0)

        def _contours_at(level):
            raw = find_contours(padded, level)
            paths = []
            for c in raw:
                p = (c - 1.0) * pixel_size
                pl = [[float(pt[0]), float(pt[1])] for pt in p]
                if len(pl) >= 3:
                    paths.append(pl)
            return paths

        contour_paths = _contours_at(0.5)

        if mode == "auto":
            bscore = 1.0 - float(np.mean(4 * density * (1 - density)))
            mode = "contour" if bscore > 0.8 else "slab"

        if mode == "contour":
            design_paths = contour_paths
        else:
            design_paths = [[[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]]]

    slab_rect = [[[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]]]

    polygon_layers = []
    z_cursor = 0.0
    for layer in layers:
        mat_raw = layer.get("material", "").lower()
        mat_key = _MATERIAL_ALIAS.get(mat_raw, mat_raw)
        thickness = float(layer["thickness"])

        z_min = float(layer.get("z_min", z_cursor))
        z_max = z_min + thickness
        z_cursor = z_max

        if mat_key == "air":
            continue
        is_design = layer.get("is_design", False)

        layer_data = {
            "layer_name": layer["name"],
            "z_min": z_min,
            "z_max": z_max,
            "material": mat_key,
            "refractive_index": float(layer["index"]),
            "paths": design_paths if is_design else slab_rect,
        }
        if is_design:
            layer_data["texture_b64"] = texture_b64
            layer_data["binary_texture_b64"] = binary_texture_b64
            layer_data["texture_size"] = [int(nx), int(ny)]
            layer_data["contour_paths"] = contour_paths
            layer_data["smooth_contour"] = smooth_contour
            if density_contours:
                layer_data["density_contours"] = density_contours
            max_verts = 200
            sx = max(1, nx // max_verts)
            sy = max(1, ny // max_verts)
            ds = density[::sx, ::sy]
            layer_data["heightmap"] = ds.tolist()
            layer_data["heightmap_size"] = [int(ds.shape[0]), int(ds.shape[1])]
        polygon_layers.append(layer_data)

    # Build port/monitor list
    ports = []
    if monitors:
        for m in monitors:
            is_input = m["name"].lower().startswith("input")
            ports.append({
                "name": m["name"],
                "center": [float(m["x"]), float(m["y"])],
                "width": float(m.get("width", y_max * 0.8)),
                "orientation": float(m.get("orientation", 0)),
                "layer": "WG" if is_input else "WG",
            })

    data = {
        "polygons": polygon_layers,
        "ports": ports,
        "bounds": {
            "x_min": 0,
            "x_max": float(x_max),
            "y_min": 0,
            "y_max": float(y_max),
        },
    }

    print("__GEOMETRY_UPDATE__" + json.dumps(data))
