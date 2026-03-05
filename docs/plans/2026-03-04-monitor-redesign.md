# Monitor Auto-Placement Redesign

## Problem

The quickstart notebook has ~30 lines of manual monitor placement code and ~15 lines of mode source waveguide detection code. This "rogue code" is confusing for users trying to follow the tutorial. Variables are defined in one cell and used in another with no clear purpose. Adjacent port monitors can collide when ports are close together, with no detection or handling.

## Design Decisions

- Port metadata (from gdsfactory) is the primary input for auto-placement, not permittivity scanning
- Mode source waveguide detection is baked into `create_mode_source` (auto-detect when bounds not provided)
- Monitor collisions are auto-shrunk with a warning telling users to manually override if results look wrong
- The `xy_mid` visualization plane is always included in auto-generated monitor sets
- Manual `monitors.add(Monitor(...), name)` is the recommended path for custom (non-gdsfactory) designs
- `add_monitors_at_position()` (permittivity auto-detect) remains available but is deprioritized (not featured in tutorials)
- Each notebook cell should be self-contained: variables defined where they are used

## New SDK Functions

### 1. `hwc.create_port_monitors()`

Creates a MonitorSet with monitors at every port of a gdsfactory component.

```python
def create_port_monitors(
    component,              # gdsfactory component (has .ports)
    structure,              # Structure object (for grid dimensions, z_wg_center)
    device_info: dict,      # From component_to_theta() (bounding_box_um, theta_resolution_um)
    padding: tuple,         # (left, right, top, bottom) padding used in density()
    absorption_widths: tuple,  # From absorber_params(), needed for source_position
    monitor_thickness: int = 5,
    monitor_half_extent: int = 35,  # Half-size in y and z (pixels)
    z_wg_center: int = None,  # If None, auto-detect from structure
    input_label_prefix: str = "Input_",
    output_label_prefix: str = "Output_",
) -> MonitorSet:
```

**Behavior:**
1. Iterate over `component.ports`
2. Convert each port's (x_um, y_um) to structure pixel coordinates using `device_info`
3. Determine Input vs Output based on port orientation (180 deg = input, 0 deg = output)
4. Create monitor with `monitor_half_extent` around port center
5. Check for collisions between monitors on the same side (input or output)
6. If collision detected: shrink both monitors to midpoint, log warning with guidance
7. Add `xy_mid` plane monitor (full XY at z_wg_center)
8. Return populated MonitorSet

**Collision detection:**
- For each pair of monitors, check if their y-ranges overlap
- If overlap: set each monitor's edge to the midpoint between their centers
- Log: `"WARNING: Monitors {name_a} and {name_b} were auto-shrunk to avoid overlap ({old_size} -> {new_size} pixels). If transmission looks wrong, manually adjust with monitors.remove(name) and monitors.add(...)"`

### 2. `hwc.create_mode_source()` Enhancement

Add auto-detect when `perpendicular_bounds` and `z_bounds` are not provided.

**Current signature** (unchanged):
```python
def create_mode_source(
    structure,
    freq_band,
    mode_num: int = 0,
    propagation_axis: str = "x",
    source_position: int = 0,
    perpendicular_bounds: tuple = None,  # (min, max) in pixels
    z_bounds: tuple = None,              # (min, max) in pixels
) -> Tuple[source_field, source_offset, mode_info]:
```

**New behavior when bounds are None:**
1. Use `_detect_waveguides()` to find waveguide at `source_position`
2. Take the first detected waveguide (closest to center if multiple)
3. Expand 2x around detected waveguide center (so mode field decays to zero)
4. Use expanded region as perpendicular_bounds and z_bounds
5. After computing mode, auto-trim source_field to the bounds region
6. Return trimmed field with corrected offset
7. Log the detected bounds for transparency

**When bounds ARE provided:** Identical to current behavior. No change for power users.

## Monitor Placement Priority (for docs/tutorials)

1. **gdsfactory designs**: `hwc.create_port_monitors()` -- primary, featured in quickstart
2. **Custom designs**: `monitors.add(Monitor(shape=..., offset=...), name)` -- manual, explicit, recommended for non-gdsfactory
3. **Auto-detect**: `monitors.add_monitors_at_position()` -- available, not deprecated, but not featured

## Quickstart Cell Restructure

### Before (current)
- Cell 4 (Mode Source): 25 lines, temp MonitorSet hack, manual expand, manual trim
- Cell 5 (Monitors): 30 lines, manual port iteration, coordinate math, manual Monitor creation

### After
- Cell 4 (Mode Source): ~5 lines
  ```python
  source_field, source_offset, mode_info = hwc.create_mode_source(
      structure=structure,
      freq_band=freq_band,
      mode_num=0,
      propagation_axis="x",
      source_position=abs_widths[0],
  )
  hwc.plot_mode(mode_field=mode_info["field"], beta=mode_info["beta"],
                mode_num=0, propagation_axis="x")
  ```

- Cell 5 (Monitors): ~6 lines
  ```python
  monitors = hwc.create_port_monitors(
      component=gf_device,
      structure=structure,
      device_info=device_info,
      padding=PADDING,
      absorption_widths=abs_widths,
  )
  hwc.plot_monitor_layout(structure.permittivity, monitors,
                          axis="z", position=z_wg_center, source_position=abs_widths[0])
  ```

## Files to Modify

1. `hyperwave_community/monitors.py` -- add `create_port_monitors()`
2. `hyperwave_community/simulate.py` -- enhance `create_mode_source()` auto-detect
3. `hyperwave_community/__init__.py` -- export `create_port_monitors`
4. `examples/quickstart.py` -- restructure cells, use new APIs
5. `examples/quickstart.ipynb` -- regenerate from .py

## Out of Scope

- Deprecating `add_monitors_at_position()` (stays as-is)
- Changing Monitor/MonitorSet data structures
- Mode overlap transmission (currently uses Poynting vector, that's fine)
- Fixing `add_absorption=True` server bug (separate issue)
