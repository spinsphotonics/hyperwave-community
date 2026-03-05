# Monitor Auto-Placement Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace 50+ lines of rogue monitor/mode-source code in quickstart with clean SDK functions.

**Architecture:** Add `create_port_monitors()` to monitors.py for gdsfactory port-based monitor creation with collision detection. Enhance `create_mode_source()` in simulate.py to auto-detect waveguide bounds when not provided. Deprecate `add_monitors_at_position()`. Restructure quickstart cells to be self-contained.

**Tech Stack:** Python, JAX, gdsfactory, hyperwave-community SDK

---

## Design Decisions

- Port metadata (from gdsfactory) is the primary input for auto-placement
- Mode source waveguide detection is baked into `create_mode_source` (auto when bounds not provided)
- Monitor collisions are auto-shrunk with a warning telling users to manually override if needed
- The `xy_mid` visualization plane is always included in auto-generated monitor sets
- Manual `monitors.add(Monitor(...), name)` is the path for custom (non-gdsfactory) designs
- `add_monitors_at_position()` is deprecated (still works, emits DeprecationWarning)
- Two paths only: `create_port_monitors()` (gdsfactory) and `monitors.add()` (manual)
- Each notebook cell should be self-contained: variables defined where they are used

---

### Task 1: Add `create_port_monitors()` to monitors.py

**Files:**
- Modify: `hyperwave_community/monitors.py` (add function at bottom, before `add_monitors_at_position`)

**Step 1: Write the function**

Add this function to `monitors.py`:

```python
import warnings

def create_port_monitors(
    component,
    structure,
    device_info: dict,
    padding: tuple,
    absorption_widths: tuple,
    monitor_thickness: int = 5,
    monitor_half_extent: int = 35,
    z_wg_center: Optional[int] = None,
    input_label_prefix: str = "Input_",
    output_label_prefix: str = "Output_",
) -> 'MonitorSet':
    """Create monitors at all ports of a gdsfactory component.

    Converts gdsfactory port positions to structure pixel coordinates and
    creates appropriately sized monitors. Automatically detects Input vs
    Output based on port orientation. Detects and resolves monitor collisions
    by shrinking overlapping monitors.

    Always includes an xy_mid visualization plane monitor.

    Args:
        component: gdsfactory component with .ports attribute.
        structure: Structure object with permittivity attribute.
        device_info: Dict from component_to_theta() containing
            'bounding_box_um' and 'theta_resolution_um'.
        padding: (left, right, top, bottom) padding used in density().
        absorption_widths: Tuple from absorber_params() for grid dimensions.
        monitor_thickness: Thickness along propagation axis (pixels). Default 5.
        monitor_half_extent: Half-size in y and z (pixels). Default 35.
        z_wg_center: Z position of waveguide center. If None, auto-detects
            from structure by finding z with highest permittivity contrast.
        input_label_prefix: Prefix for input port monitors. Default "Input_".
        output_label_prefix: Prefix for output port monitors. Default "Output_".

    Returns:
        MonitorSet with monitors at each port plus an xy_mid plane.
    """
    # Get structure dimensions
    if len(structure.permittivity.shape) == 4:
        _, Lx, Ly, Lz = structure.permittivity.shape
    else:
        Lx, Ly, Lz = structure.permittivity.shape

    # Auto-detect z_wg_center if not provided
    if z_wg_center is None:
        eps = structure.permittivity
        if len(eps.shape) == 4:
            eps = eps[0]
        # Find z with max permittivity variance (waveguide core layer)
        z_variance = jnp.var(eps, axis=(0, 1))
        z_wg_center = int(jnp.argmax(z_variance))
        logger.info("Auto-detected z_wg_center=%d", z_wg_center)

    # Extract coordinate mapping from device_info
    bbox = device_info["bounding_box_um"]
    x_min_um, y_min_um = bbox[0], bbox[1]
    theta_res = device_info["theta_resolution_um"]
    y_pad_struct = padding[0] // 2  # left padding in structure pixels

    # Build monitor specs from ports
    monitor_specs = []
    for port in component.ports:
        px_um, py_um = port.center
        x_struct = int((px_um - x_min_um) / theta_res / 2) + padding[2] // 2
        y_struct = int((py_um - y_min_um) / theta_res / 2) + y_pad_struct

        is_input = abs(port.orientation % 360 - 180) < 1
        prefix = input_label_prefix if is_input else output_label_prefix
        label = f"{prefix}{port.name}"

        monitor_specs.append({
            "label": label,
            "x": x_struct,
            "y": y_struct,
            "half_extent": monitor_half_extent,
        })

    # Detect and resolve y-axis collisions
    # Sort by y position to find adjacent monitors
    monitor_specs.sort(key=lambda m: m["y"])
    for i in range(len(monitor_specs) - 1):
        a = monitor_specs[i]
        b = monitor_specs[i + 1]
        a_top = a["y"] + a["half_extent"]
        b_bottom = b["y"] - b["half_extent"]
        if a_top > b_bottom:
            # Collision: shrink both to midpoint
            midpoint = (a["y"] + b["y"]) // 2
            old_half_a = a["half_extent"]
            old_half_b = b["half_extent"]
            a["half_extent"] = midpoint - a["y"]
            b["half_extent"] = b["y"] - midpoint
            # Ensure minimum size of 5 pixels
            a["half_extent"] = max(5, a["half_extent"])
            b["half_extent"] = max(5, b["half_extent"])
            logger.warning(
                "Monitors %s and %s were auto-shrunk to avoid overlap "
                "(%d -> %d and %d -> %d pixels half-extent). "
                "If transmission looks wrong, manually adjust with "
                "monitors.remove(name) and monitors.add(...).",
                a["label"], b["label"],
                old_half_a, a["half_extent"],
                old_half_b, b["half_extent"],
            )

    # Create MonitorSet
    monitors = MonitorSet()
    for spec in monitor_specs:
        y_half = spec["half_extent"]
        z_half = min(monitor_half_extent, z_wg_center, Lz - z_wg_center)
        shape = (monitor_thickness, 2 * y_half, 2 * z_half)
        offset = (spec["x"], spec["y"] - y_half, z_wg_center - z_half)
        # Clamp offset to valid range
        offset = (
            max(0, min(offset[0], Lx - shape[0])),
            max(0, offset[1]),
            max(0, offset[2]),
        )
        monitors.add(Monitor(shape=shape, offset=offset), spec["label"])

    # Always add xy_mid visualization plane
    monitors.add(
        Monitor(shape=(Lx, Ly, 1), offset=(0, 0, z_wg_center)),
        "xy_mid",
    )

    return monitors
```

**Step 2: Run quickstart to verify it works (deferred to Task 5)**

**Step 3: Commit**

```bash
git add hyperwave_community/monitors.py
git commit -m "Add create_port_monitors() for gdsfactory port-based monitor placement"
```

---

### Task 2: Deprecate `add_monitors_at_position()`

**Files:**
- Modify: `hyperwave_community/monitors.py:566-649` (MonitorSet.add_monitors_at_position method)
- Modify: `hyperwave_community/monitors.py:904-917` (module-level add_monitors_at_position function)

**Step 1: Add deprecation warning to both functions**

Add `import warnings` at top of file (if not already present).

In the module-level `add_monitors_at_position()` function (line ~918), add as the first line of the function body:

```python
    warnings.warn(
        "add_monitors_at_position() is deprecated. "
        "Use create_port_monitors() for gdsfactory designs or "
        "monitors.add(Monitor(...), name) for manual placement.",
        DeprecationWarning,
        stacklevel=2,
    )
```

In `MonitorSet.add_monitors_at_position()` method (line ~566), add as the first line:

```python
        warnings.warn(
            "add_monitors_at_position() is deprecated. "
            "Use create_port_monitors() for gdsfactory designs or "
            "monitors.add(Monitor(...), name) for manual placement.",
            DeprecationWarning,
            stacklevel=2,
        )
```

**Step 2: Commit**

```bash
git add hyperwave_community/monitors.py
git commit -m "Deprecate add_monitors_at_position() in favor of create_port_monitors()"
```

---

### Task 3: Enhance `create_mode_source()` with auto-detect

**Files:**
- Modify: `hyperwave_community/simulate.py:22-182`

**Step 1: Add auto-detect logic**

In `create_mode_source()`, after line 71 (`_, full_x_size, full_y_size, full_z_size = structure.permittivity.shape`), add auto-detect logic:

```python
    # Auto-detect waveguide bounds if not provided
    if perpendicular_bounds is None or z_bounds is None:
        from .monitors import _detect_waveguides

        if propagation_axis == 'x':
            waveguides = _detect_waveguides(
                structure, x_position=source_position, z_position=None, axis='y'
            )
            if waveguides:
                wg = waveguides[0]  # Use first (closest to center)
                if perpendicular_bounds is None:
                    # Expand 2x around waveguide
                    y_center = wg['center']
                    y_expand = wg['width']
                    perpendicular_bounds = (
                        max(0, y_center - y_expand),
                        min(full_y_size, y_center + y_expand),
                    )
                    logger.info(
                        "Auto-detected waveguide at y=%d (width=%d), "
                        "using perpendicular_bounds=(%d, %d)",
                        wg['center'], wg['width'],
                        perpendicular_bounds[0], perpendicular_bounds[1],
                    )
                if z_bounds is None and 'z_core' in wg:
                    z_center = wg['z_core']
                    z_expand = wg['width']  # Use same expansion as y
                    z_bounds = (
                        max(0, z_center - z_expand),
                        min(full_z_size, z_center + z_expand),
                    )
                    logger.info(
                        "Auto-detected z_core=%d, using z_bounds=(%d, %d)",
                        z_center, z_bounds[0], z_bounds[1],
                    )
        else:  # propagation_axis == 'y'
            waveguides = _detect_waveguides(
                structure, y_position=source_position, z_position=None, axis='x'
            )
            if waveguides:
                wg = waveguides[0]
                if perpendicular_bounds is None:
                    x_center = wg['center']
                    x_expand = wg['width']
                    perpendicular_bounds = (
                        max(0, x_center - x_expand),
                        min(full_x_size, x_center + x_expand),
                    )
                    logger.info(
                        "Auto-detected waveguide at x=%d (width=%d), "
                        "using perpendicular_bounds=(%d, %d)",
                        wg['center'], wg['width'],
                        perpendicular_bounds[0], perpendicular_bounds[1],
                    )
                if z_bounds is None and 'z_core' in wg:
                    z_center = wg['z_core']
                    z_expand = wg['width']
                    z_bounds = (
                        max(0, z_center - z_expand),
                        min(full_z_size, z_center + z_expand),
                    )
                    logger.info(
                        "Auto-detected z_core=%d, using z_bounds=(%d, %d)",
                        z_center, z_bounds[0], z_bounds[1],
                    )
```

**Step 2: Add auto-trim of source field**

After the existing padding logic (line ~170, after `source_field = jnp.concatenate(...)`), add trimming when bounds were auto-detected:

Replace the current source_field creation and return block (lines 171-182) with:

```python
    # Create full source field (E and H components)
    source_field = jnp.concatenate([mode_E_field, jnp.zeros_like(mode_E_field)], axis=1)

    # Auto-trim: if bounds were used, trim source field to just the bounds region
    # This reduces data sent to cloud
    if perpendicular_bounds is not None or z_bounds is not None:
        if propagation_axis == 'x':
            y_min_used, y_max_used = perpendicular_bounds if perpendicular_bounds else (0, full_y_size)
            z_min_used, z_max_used = z_bounds if z_bounds else (0, full_z_size)
            source_field = source_field[:, :, :, y_min_used:y_max_used, z_min_used:z_max_used]
            source_offset = (source_position, y_min_used, z_min_used)
        else:
            x_min_used, x_max_used = perpendicular_bounds if perpendicular_bounds else (0, full_x_size)
            z_min_used, z_max_used = z_bounds if z_bounds else (0, full_z_size)
            source_field = source_field[:, :, x_min_used:x_max_used, :, z_min_used:z_max_used]
            source_offset = (x_min_used, source_position, z_min_used)

    mode_info = {'field': mode_E_field, 'beta': beta, 'error': err}

    logger.info("Source: shape=%s, offset=%s, beta=%s",
                source_field.shape, source_offset, beta)

    return source_field, source_offset, mode_info
```

**Step 3: Commit**

```bash
git add hyperwave_community/simulate.py
git commit -m "Auto-detect waveguide bounds in create_mode_source when not provided"
```

---

### Task 4: Export `create_port_monitors` from `__init__.py`

**Files:**
- Modify: `hyperwave_community/__init__.py`

**Step 1: Add import and export**

Add to imports section (near line 70 where Monitor and MonitorSet are imported):
```python
from .monitors import create_port_monitors
```

Add to `__all__` list (near line 199 where Monitor and MonitorSet are listed):
```python
"create_port_monitors",
```

**Step 2: Commit**

```bash
git add hyperwave_community/__init__.py
git commit -m "Export create_port_monitors from SDK"
```

---

### Task 5: Restructure quickstart.py cells

**Files:**
- Modify: `examples/quickstart.py`

**Step 1: Rewrite quickstart.py**

The new quickstart should have self-contained cells. Here is the full file:

```python
# %% [markdown]
# # Hyperwave Quickstart: 2x2 MMI with S-Bends
#
# Simulate a 2x2 multimode interference coupler using gdsfactory for layout
# and Hyperwave for cloud-accelerated 3D FDTD.
#
# **What you'll learn:**
# 1. Convert a gdsfactory component to a simulation-ready structure
# 2. Set up mode sources, monitors, and absorbing boundaries
# 3. Run a cloud GPU simulation and analyze transmission

# %% Installation
# pip install hyperwave-community gdsfactory

# %% Imports and Configuration
import hyperwave_community as hwc
import gdsfactory as gf
import numpy as np
import jax.numpy as jnp

hwc.set_verbose(True)

PDK = gf.gpdk.get_generic_pdk()
PDK.activate()

RESOLUTION_UM = 0.02          # 20 nm grid spacing
N_CORE = 3.48                 # Silicon refractive index at 1550 nm
N_CLAD = 1.45                 # SiO2 cladding
WL_UM = 1.55                  # Wavelength
WG_HEIGHT_UM = 0.22           # Waveguide core height
TOTAL_HEIGHT_UM = 4.0         # Total stack height
PADDING = (100, 100, 0, 0)   # (left, right, top, bottom) in theta pixels


# %% Step 1: Load Component
#
# Load a 2x2 MMI with S-bends from gdsfactory and extend ports so the mode
# source and monitors sit inside straight waveguide sections.

EXTENSION_LENGTH = 2.0        # Extend ports by 2 um

gf_device = gf.components.mmi2x2_with_sbend()
gf_extended = gf.c.extend_ports(gf_device, length=EXTENSION_LENGTH)

theta, device_info = hwc.component_to_theta(
    component=gf_extended,
    resolution=RESOLUTION_UM,
)


# %% Step 2: Build 3D Structure
#
# Apply density filtering, then stack cladding and waveguide layers into
# a 3D permittivity volume.

eps_core = N_CORE ** 2
eps_clad = N_CLAD ** 2

density_core = hwc.density(theta=theta, pad_width=PADDING, radius=3)
density_clad = hwc.density(theta=jnp.zeros_like(theta), pad_width=PADDING, radius=5)

wg_cells = max(1, int(np.round(WG_HEIGHT_UM / RESOLUTION_UM)))
clad_cells = int(np.round((TOTAL_HEIGHT_UM - WG_HEIGHT_UM) / 2 / RESOLUTION_UM))

structure = hwc.create_structure(
    layers=[
        hwc.Layer(density_pattern=density_clad, permittivity_values=eps_clad, layer_thickness=clad_cells),
        hwc.Layer(density_pattern=density_core, permittivity_values=(eps_clad, eps_core), layer_thickness=wg_cells),
        hwc.Layer(density_pattern=density_clad, permittivity_values=eps_clad, layer_thickness=clad_cells),
    ],
    vertical_radius=2,
)

z_wg_center = clad_cells + wg_cells // 2
hwc.plot_structure(structure, axis="z", position=z_wg_center)


# %% Step 3: Absorbing Boundaries
#
# Add adiabatic absorbers at grid edges to prevent reflections.

_, Lx, Ly, Lz = structure.permittivity.shape

abs_params = hwc.absorber_params(
    wavelength_um=WL_UM,
    dx_um=RESOLUTION_UM,
    structure_dimensions=(Lx, Ly, Lz),
)
abs_widths = tuple(abs_params["absorption_widths"])
abs_coeff = abs_params["abs_coeff"]

absorber = hwc.create_absorption_mask(
    grid_shape=(Lx, Ly, Lz),
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
)
structure.conductivity = jnp.zeros_like(structure.conductivity) + absorber

hwc.plot_absorption_mask(absorber)


# %% Step 4: Mode Source
#
# Solve for the fundamental TE mode at the input waveguide.
# The SDK automatically detects the waveguide cross-section at the
# source plane and constrains the mode solver to that region.

wl_cells = WL_UM / RESOLUTION_UM
freq_band = (2 * jnp.pi / wl_cells, 2 * jnp.pi / wl_cells, 1)

source_field, source_offset, mode_info = hwc.create_mode_source(
    structure=structure,
    freq_band=freq_band,
    mode_num=0,
    propagation_axis="x",
    source_position=abs_widths[0],
)

hwc.plot_mode(
    mode_field=mode_info["field"],
    beta=mode_info["beta"],
    mode_num=0,
    propagation_axis="x",
)


# %% Step 5: Monitors
#
# Place field monitors at each port. The SDK reads port positions from the
# gdsfactory component and creates appropriately sized, non-overlapping monitors.

monitors = hwc.create_port_monitors(
    component=gf_device,
    structure=structure,
    device_info=device_info,
    padding=PADDING,
    absorption_widths=abs_widths,
)

hwc.plot_monitor_layout(
    structure.permittivity, monitors,
    axis="z", position=z_wg_center, source_position=abs_widths[0],
)


# %% Step 6: Simulate
#
# Configure your API key and run the simulation on cloud GPU.
# Sign up at https://spinsphotonics.com/signup to get your key.

try:
    from google.colab import userdata
    hwc.configure_api(api_key=userdata.get("HYPERWAVE_API_KEY"))
except ImportError:
    import os
    hwc.configure_api(api_key=os.environ.get("HYPERWAVE_API_KEY"))

results = hwc.simulate(
    structure_recipe=structure.extract_recipe(),
    source_field=source_field,
    source_offset=source_offset,
    freq_band=freq_band,
    monitors_recipe=monitors.recipe,
    mode_info=mode_info,
    simulation_steps=20000,
    add_absorption=False,
    absorption_widths=abs_widths,
    absorption_coeff=abs_coeff,
)


# %% Step 7: Analyze Results

transmission = hwc.analyze_transmission(
    results, input_monitor="Input_o1", direction="x",
)

hwc.plot_monitors(results, component="Hz")

hwc.export_csv(transmission, "quickstart_transmission.csv")
```

**Step 2: Convert to notebook and execute**

```bash
cd /home/dq4443/dev/work/hyperwave-community
source /home/dq4443/.claude-mcp-venv/bin/activate
jupytext --to notebook examples/quickstart.py -o examples/quickstart.ipynb
export HYPERWAVE_API_KEY=<key>
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
    examples/quickstart.ipynb --output quickstart.ipynb
```

**Step 3: Verify results**

Check notebook outputs:
- Each plot cell has exactly 1 image (no doubles)
- Transmission T_o4 is approximately 1.0
- No errors or tracebacks
- Monitor collision warning appears if applicable

**Step 4: Commit**

```bash
git add examples/quickstart.py examples/quickstart.ipynb
git commit -m "Restructure quickstart to use create_port_monitors and auto-detect mode source"
```

---

### Task 6: Update test_full_pipeline.py

**Files:**
- Modify: `examples/test_full_pipeline.py` (if it references deprecated functions)

**Step 1: Check if test_full_pipeline uses add_monitors_at_position**

Search for `add_monitors_at_position` in test files. If used, update to use either
`create_port_monitors()` or manual `monitors.add()`.

**Step 2: Run test_full_pipeline.py to verify no regressions**

```bash
cd /home/dq4443/dev/work/hyperwave-community
source /home/dq4443/.claude-mcp-venv/bin/activate
python examples/test_full_pipeline.py
```

**Step 3: Commit if changed**

```bash
git add examples/test_full_pipeline.py
git commit -m "Update test_full_pipeline to use non-deprecated monitor APIs"
```

---

### Task 7: Push and verify

**Step 1: Push all commits**

```bash
git push
```

**Step 2: Verify CI passes (if applicable)**

Check that the branch builds and any CI checks pass.
