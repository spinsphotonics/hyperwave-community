# GPU Benchmarking Implementation Plan


**Goal:** Benchmark all 8 supported GPU types for max structure size (OOM limit), simulation speed (Gcell/s), and credit cost, saving full artifacts (images, stats, transmission) in a well-organized directory structure.

**Architecture:** A Jupyter dev notebook (`examples/gpu_benchmark_dev.ipynb`) orchestrates the benchmarks. Each run calls the full 5-step hyperwave pipeline (build_recipe -> build_monitors -> compute_freq_band -> solve_mode_source -> run_simulation) and saves all artifacts to `benchmarks/<GPU>/<aspect>/pad_<N>/`. ThreadPoolExecutor with 8 workers runs all GPUs in parallel. Results persist as JSON after every run so the notebook is resumable.

**Tech Stack:** hyperwave_community (FDTD sim library), Jupyter notebook, matplotlib (Agg backend for thread safety), concurrent.futures, json

---

## Background

### The hyperwave_community API pipeline

Every simulation follows this exact sequence:

```python
import hyperwave_community as hwc
from hyperwave_community import ConvergenceConfig

# Step 1: Build structure (Modal CPU, FREE)
recipe_result = hwc.build_recipe(
    component_name="mmi2x2_with_sbend",
    resolution_nm=20,
    extension_length=2.0,
    n_core=3.48, n_clad=1.4457,
    wg_height_um=0.22, total_height_um=4.0,
    padding=(100, 100, 0, 0),
    density_radius=3, vertical_radius=2.0,
)
# Returns: recipe, density_core, density_clad, dimensions, port_info,
#          layer_config, eps_values, resolution_um

# Step 2: Build monitors (Modal CPU, FREE)
monitor_result = hwc.build_monitors(
    port_info=recipe_result['port_info'],
    dimensions=recipe_result['dimensions'],
    source_port="o1",
    resolution_um=recipe_result['resolution_um'],
    structure_recipe=recipe_result['recipe'],
    show_structure=False,
    include_field_monitor=True,
)
# Returns: monitors, monitor_names, source_position, mode_bounds

# Step 3: Compute frequency band (Modal CPU, FREE)
freq_result = hwc.compute_freq_band(
    wl_min_um=1.55, wl_max_um=1.55, n_freqs=1,
    resolution_um=recipe_result['resolution_um'],
)
# Returns: freq_band, wavelengths_um

# Step 4: Solve mode source (Modal CPU, FREE)
source_result = hwc.solve_mode_source(
    density_core=recipe_result['density_core'],
    density_clad=recipe_result['density_clad'],
    source_x_position=monitor_result['source_position'],
    mode_bounds=monitor_result['mode_bounds'],
    layer_config=recipe_result['layer_config'],
    eps_values=recipe_result['eps_values'],
    freq_band=freq_result['freq_band'],
    mode_num=0,
    show_mode=False,
)
# Returns: source_field, source_offset

# Step 5: Get optimized absorbers (LOCAL, FREE)
absorber_params = hwc.get_optimized_absorber_params(
    resolution_nm=20,
    wavelength_um=1.55,
    structure_dimensions=recipe_result['dimensions'],
)
# Returns: absorber_width, absorber_coeff, absorption_widths

# Step 6: Run simulation (GPU, COSTS CREDITS)
results = hwc.run_simulation(
    device_type="mmi2x2_with_sbend",
    recipe_result=recipe_result,
    monitor_result=monitor_result,
    freq_result=freq_result,
    source_result=source_result,
    num_steps=40000,
    gpu_type="H100",
    convergence=ConvergenceConfig(
        check_every_n=1000,
        min_stable_checks=3,
        min_steps=5000,
        power_threshold=1e-6,
    ),
    absorption_widths=absorber_params['absorption_widths'],
    absorption_coeff=absorber_params['absorber_coeff'],
)
# Returns: sim_time, total_time, performance, converged, convergence_step,
#          monitor_data, powers, s_parameters, field_intensity
```

### GPU types and multipliers

| GPU | VRAM | Credit Multiplier |
|-----|------|-------------------|
| B200 | 192 GB | 2.5x |
| H200 | 141 GB | 2.0x |
| H100 | 80 GB | 1.5x |
| A100-80GB | 80 GB | 1.0x |
| A100 | 40 GB | 0.8x |
| L40S | 48 GB | 0.7x |
| A10 | 24 GB | 0.4x |
| T4 | 16 GB | 0.3x |

### Key constraints

- The baseline MMI (`mmi2x2_with_sbend` at 20nm, padding=(100,100,0,0)) produces dimensions (1800, 350, 199) = ~125M voxels. This is B200-scale. Smaller GPUs will OOM at this size.
- `run_simulation()` returns `None` on OOM or any error.
- `convergence="full"` disables early stopping. Use `ConvergenceConfig` for real convergence.
- Convergence options: `"quick"`, `"default"`, `"thorough"`, `"full"`, or `ConvergenceConfig` object.
- `performance` field in results = grid_points * steps / second. Divide by 1e9 for Gcell/s.
- `visualize_structure(save_path=..., show=False)` and `visualize_mode_source(save_path=..., show=False)` save PNGs without displaying.
- `estimate_cost(grid_points=N, max_steps=S, gpu_type=G)` previews credit cost (free, no auth needed).
- The notebook is gitignored by `examples/*_dev.ipynb` pattern.

### File locations

- Repo root: `/home/dq4443/dev/work/hyperwave-community/`
- API client: `hyperwave_community/api_client.py`
- Absorption: `hyperwave_community/absorption.py`
- Exports: `hyperwave_community/__init__.py`
- Existing workflow: `examples/api_workflow.ipynb`

---

## Task 1: Create benchmark directory structure and notebook skeleton

**Files:**
- Create: `examples/gpu_benchmark_dev.ipynb`
- Create: `benchmarks/.gitkeep` (ensure dir exists)

**Step 1: Create benchmarks directory**

```bash
cd /home/dq4443/dev/work/hyperwave-community
mkdir -p benchmarks
```

**Step 2: Create notebook with cells 1-3 (Setup, Constants, Persistence)**

Create `examples/gpu_benchmark_dev.ipynb` with these cells:

**Cell 1 (code): Setup and Imports**
```python
import hyperwave_community as hwc
from hyperwave_community import ConvergenceConfig
import json
import time
import threading
import io
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hwc.configure_api(api_key="9e293a83-feb0-4275-b6f0-540ba935b4bb")
hwc.get_account_info()
```

**Cell 2 (code): Constants and GPU Config**
```python
GPU_TYPES = ["B200", "H200", "H100", "A100-80GB", "A100", "L40S", "A10", "T4"]

GPU_INFO = {
    "B200":      {"vram_gb": 192, "multiplier": 2.5},
    "H200":      {"vram_gb": 141, "multiplier": 2.0},
    "H100":      {"vram_gb": 80,  "multiplier": 1.5},
    "A100-80GB": {"vram_gb": 80,  "multiplier": 1.0},
    "A100":      {"vram_gb": 40,  "multiplier": 0.8},
    "L40S":      {"vram_gb": 48,  "multiplier": 0.7},
    "A10":       {"vram_gb": 24,  "multiplier": 0.4},
    "T4":        {"vram_gb": 16,  "multiplier": 0.3},
}

# Simulation parameters
COMPONENT = "mmi2x2_with_sbend"
RESOLUTION_NM = 20
WL_UM = 1.55
N_CORE = 3.48
N_CLAD = 1.4457
WG_HEIGHT_UM = 0.22
TOTAL_HEIGHT_UM = 4.0
EXTENSION_LENGTH = 2.0
DENSITY_RADIUS = 3
VERTICAL_RADIUS = 2.0
SOURCE_PORT = "o1"
MODE_NUM = 0
MAX_STEPS = 40000

CONV_CONFIG = ConvergenceConfig(
    check_every_n=1000,
    min_stable_checks=3,
    min_steps=5000,
    power_threshold=1e-6,
)

BENCHMARK_DIR = Path("../benchmarks")

# Starting padding by VRAM tier
# Large GPUs start at baseline, small GPUs start small
def get_start_padding(gpu_type):
    vram = GPU_INFO[gpu_type]["vram_gb"]
    if vram >= 80:
        return 100  # baseline MMI size
    elif vram >= 40:
        return 40
    elif vram >= 24:
        return 20
    else:
        return 10
```

**Cell 3 (code): Persistence and Directory Helpers**
```python
results_lock = threading.Lock()

def ensure_run_dir(gpu_type, category, label):
    """Create and return path like benchmarks/H100/elongated/pad_200/"""
    run_dir = BENCHMARK_DIR / gpu_type / category / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def load_summary():
    """Load the shared summary.json, or return empty dict."""
    summary_path = BENCHMARK_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {"runs": {}, "metadata": {}}

def save_summary(summary):
    """Thread-safe save of summary.json."""
    summary["metadata"]["last_updated"] = datetime.now().isoformat()
    with results_lock:
        summary_path = BENCHMARK_DIR / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

def save_run_stats(run_dir, stats):
    """Save stats.json for a single run."""
    with open(run_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

def is_run_done(gpu_type, category, label):
    """Check if a run already has stats.json (for resumability)."""
    stats_path = BENCHMARK_DIR / gpu_type / category / label / "stats.json"
    return stats_path.exists()

def capture_stdout(func, *args, **kwargs):
    """Call func, capture its stdout, return (result, captured_text)."""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return result, buffer.getvalue()
```

**Step 3: Verify notebook is gitignored**

Confirm `examples/*_dev.ipynb` pattern in `.gitignore` covers `gpu_benchmark_dev.ipynb`.

**Step 4: Commit**

```bash
git add benchmarks/.gitkeep
git commit -m "feat: add benchmarks directory for GPU benchmarking"
```

---

## Task 2: Write calibration cell

**Files:**
- Modify: `examples/gpu_benchmark_dev.ipynb` (add cell 4)

**Step 1: Add calibration cell**

**Cell 4 (code): Calibration - understand padding-to-dimensions mapping**
```python
print("=== Calibration: padding -> dimensions mapping ===\n")

test_paddings = [10, 20, 40, 60, 80, 100, 150, 200, 300]
calibration_data = {}

for pad in test_paddings:
    padding = (pad, pad, 0, 0)
    result, output = capture_stdout(
        hwc.build_recipe,
        component_name=COMPONENT,
        resolution_nm=RESOLUTION_NM,
        extension_length=EXTENSION_LENGTH,
        n_core=N_CORE, n_clad=N_CLAD,
        wg_height_um=WG_HEIGHT_UM,
        total_height_um=TOTAL_HEIGHT_UM,
        padding=padding,
        density_radius=DENSITY_RADIUS,
        vertical_radius=VERTICAL_RADIUS,
    )
    if result is None:
        print(f"  pad={pad}: FAILED")
        continue

    dims = result['dimensions']
    voxels = dims[0] * dims[1] * dims[2]
    calibration_data[pad] = {
        "dims": dims,
        "voxels": voxels,
        "voxels_M": round(voxels / 1e6, 1),
    }
    print(f"  pad={pad:>4d} -> {dims[0]:>5d} x {dims[1]:>4d} x {dims[2]:>3d} = {voxels/1e6:>8.1f}M voxels")

# Also test balanced padding
print("\n=== Balanced padding (all sides) ===\n")
for pad in [10, 20, 40, 60, 80, 100]:
    padding = (pad, pad, pad, pad)
    result, output = capture_stdout(
        hwc.build_recipe,
        component_name=COMPONENT,
        resolution_nm=RESOLUTION_NM,
        extension_length=EXTENSION_LENGTH,
        n_core=N_CORE, n_clad=N_CLAD,
        wg_height_um=WG_HEIGHT_UM,
        total_height_um=TOTAL_HEIGHT_UM,
        padding=padding,
        density_radius=DENSITY_RADIUS,
        vertical_radius=VERTICAL_RADIUS,
    )
    if result is None:
        print(f"  pad={pad}: FAILED")
        continue

    dims = result['dimensions']
    voxels = dims[0] * dims[1] * dims[2]
    print(f"  pad={pad:>4d} -> {dims[0]:>5d} x {dims[1]:>4d} x {dims[2]:>3d} = {voxels/1e6:>8.1f}M voxels")

print("\nCalibration complete.")
```

**Step 2: Run cell 4 to see results**

Expected output: a table showing how padding maps to dimensions. The baseline pad=100 should give ~(1800, 350, 199). Smaller paddings give smaller structures.

---

## Task 3: Write the core `run_benchmark()` function

**Files:**
- Modify: `examples/gpu_benchmark_dev.ipynb` (add cells 5-6)

**Step 1: Add run_benchmark function cell**

**Cell 5 (code): Core benchmark pipeline**
```python
def run_benchmark(gpu_type, padding_tuple, category, label, num_steps=MAX_STEPS):
    """Run the full FDTD pipeline and save all artifacts.

    Args:
        gpu_type: GPU string e.g. "H100"
        padding_tuple: (left, right, top, bottom) in theta pixels
        category: "elongated", "balanced", "validation", or "speed_reference"
        label: directory name e.g. "pad_100"
        num_steps: max simulation steps

    Returns:
        dict with stats on success, None on failure (OOM)
    """
    run_dir = ensure_run_dir(gpu_type, category, label)

    # Skip if already done
    if (run_dir / "stats.json").exists():
        print(f"[{gpu_type}/{category}/{label}] Already done, skipping")
        with open(run_dir / "stats.json") as f:
            return json.load(f)

    stats = {
        "gpu_type": gpu_type,
        "category": category,
        "label": label,
        "padding": list(padding_tuple),
        "timestamp": datetime.now().isoformat(),
        "status": "started",
    }

    try:
        # --- Step 1: Build recipe (FREE) ---
        recipe_result, recipe_output = capture_stdout(
            hwc.build_recipe,
            component_name=COMPONENT,
            resolution_nm=RESOLUTION_NM,
            extension_length=EXTENSION_LENGTH,
            n_core=N_CORE, n_clad=N_CLAD,
            wg_height_um=WG_HEIGHT_UM,
            total_height_um=TOTAL_HEIGHT_UM,
            padding=padding_tuple,
            density_radius=DENSITY_RADIUS,
            vertical_radius=VERTICAL_RADIUS,
        )
        if recipe_result is None:
            stats["status"] = "failed_build_recipe"
            stats["error"] = recipe_output
            save_run_stats(run_dir, stats)
            print(f"[{gpu_type}/{category}/{label}] build_recipe FAILED")
            return None

        dims = recipe_result['dimensions']
        voxels = dims[0] * dims[1] * dims[2]
        stats["dimensions"] = list(dims)
        stats["voxels"] = voxels
        stats["voxels_M"] = round(voxels / 1e6, 1)
        print(f"[{gpu_type}/{category}/{label}] Structure: {dims[0]}x{dims[1]}x{dims[2]} = {voxels/1e6:.1f}M voxels")

        # --- Step 2: Build monitors (FREE) ---
        monitor_result, _ = capture_stdout(
            hwc.build_monitors,
            port_info=recipe_result['port_info'],
            dimensions=recipe_result['dimensions'],
            source_port=SOURCE_PORT,
            resolution_um=recipe_result['resolution_um'],
            structure_recipe=recipe_result['recipe'],
            show_structure=False,
            include_field_monitor=True,
        )
        if monitor_result is None:
            stats["status"] = "failed_build_monitors"
            save_run_stats(run_dir, stats)
            print(f"[{gpu_type}/{category}/{label}] build_monitors FAILED")
            return None

        stats["monitor_names"] = list(monitor_result['monitor_names'].keys())

        # --- Save structure visualization ---
        try:
            hwc.visualize_structure(
                structure_recipe=recipe_result['recipe'],
                monitors=monitor_result['monitors'],
                monitor_names=monitor_result['monitor_names'],
                dimensions=recipe_result['dimensions'],
                source_position=monitor_result['source_position'],
                show=False,
                save_path=str(run_dir / "structure.png"),
            )
        except Exception as e:
            stats["structure_viz_error"] = str(e)

        # --- Step 3: Compute frequency band (FREE) ---
        freq_result, _ = capture_stdout(
            hwc.compute_freq_band,
            wl_min_um=WL_UM, wl_max_um=WL_UM, n_freqs=1,
            resolution_um=recipe_result['resolution_um'],
        )

        # --- Step 4: Solve mode source (FREE) ---
        source_result, _ = capture_stdout(
            hwc.solve_mode_source,
            density_core=recipe_result['density_core'],
            density_clad=recipe_result['density_clad'],
            source_x_position=monitor_result['source_position'],
            mode_bounds=monitor_result['mode_bounds'],
            layer_config=recipe_result['layer_config'],
            eps_values=recipe_result['eps_values'],
            freq_band=freq_result['freq_band'],
            mode_num=MODE_NUM,
            show_mode=False,
        )
        if source_result is None:
            stats["status"] = "failed_solve_mode"
            save_run_stats(run_dir, stats)
            print(f"[{gpu_type}/{category}/{label}] solve_mode_source FAILED")
            return None

        # --- Save mode visualization ---
        try:
            hwc.visualize_mode_source(
                source_field=source_result['source_field'],
                source_offset=source_result['source_offset'],
                show=False,
                save_path=str(run_dir / "mode_profile.png"),
            )
        except Exception as e:
            stats["mode_viz_error"] = str(e)

        # --- Step 5: Get optimized absorber params (LOCAL, FREE) ---
        absorber_params = hwc.get_optimized_absorber_params(
            resolution_nm=RESOLUTION_NM,
            wavelength_um=WL_UM,
            structure_dimensions=recipe_result['dimensions'],
        )
        stats["absorber_width"] = absorber_params['absorber_width']
        stats["absorber_coeff"] = absorber_params['absorber_coeff']
        stats["absorption_widths"] = list(absorber_params['absorption_widths'])

        # --- Step 6: Estimate cost (FREE, no auth) ---
        cost_estimate = hwc.estimate_cost(
            grid_points=voxels,
            max_steps=num_steps,
            gpu_type=gpu_type,
        )
        stats["estimated_cost"] = cost_estimate
        print(f"[{gpu_type}/{category}/{label}] Estimated cost: {cost_estimate}")

        # --- Step 7: Run simulation (GPU, COSTS CREDITS) ---
        sim_start = time.time()
        results, sim_output = capture_stdout(
            hwc.run_simulation,
            device_type=COMPONENT,
            recipe_result=recipe_result,
            monitor_result=monitor_result,
            freq_result=freq_result,
            source_result=source_result,
            num_steps=num_steps,
            gpu_type=gpu_type,
            convergence=CONV_CONFIG,
            absorption_widths=absorber_params['absorption_widths'],
            absorption_coeff=absorber_params['absorber_coeff'],
        )
        sim_wall_time = time.time() - sim_start
        stats["sim_output"] = sim_output

        if results is None:
            stats["status"] = "failed_simulation"
            stats["sim_wall_time"] = sim_wall_time
            save_run_stats(run_dir, stats)
            print(f"[{gpu_type}/{category}/{label}] SIMULATION FAILED (likely OOM) after {sim_wall_time:.1f}s")
            return None

        # --- Record simulation stats ---
        stats["sim_time"] = results.get('sim_time', 0)
        stats["total_time"] = results.get('total_time', 0)
        stats["sim_wall_time"] = sim_wall_time
        stats["performance"] = results.get('performance', 0)
        stats["gcells_per_s"] = results.get('performance', 0) / 1e9
        stats["converged"] = results.get('converged', False)
        stats["convergence_step"] = results.get('convergence_step', num_steps)

        print(f"[{gpu_type}/{category}/{label}] SIM OK: {stats['sim_time']:.1f}s, "
              f"{stats['gcells_per_s']:.2f} Gcell/s, "
              f"converged={stats['converged']} at step {stats['convergence_step']}")

        # --- Step 8: Analyze transmission ---
        try:
            transmission, trans_output = capture_stdout(
                hwc.analyze_transmission,
                results,
                input_monitor="Input_o1",
                print_results=True,
            )
            stats["transmission"] = {
                "power_in": transmission.get('power_in'),
                "total_transmission": transmission.get('total_transmission'),
                "excess_loss_dB": transmission.get('excess_loss_dB'),
                "per_port": {k: float(v) for k, v in transmission.get('transmissions', {}).items()},
            }
            with open(run_dir / "transmission.txt", "w") as f:
                f.write(trans_output)
        except Exception as e:
            stats["transmission_error"] = str(e)

        # --- Step 9: Save field intensity plot ---
        try:
            field_data = hwc.get_field_intensity_2d(
                results,
                monitor_name='xy_mid',
                dimensions=recipe_result['dimensions'],
                resolution_um=recipe_result['resolution_um'],
                freq_band=freq_result['freq_band'],
            )
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(
                field_data['intensity'],
                origin='upper',
                extent=field_data.get('extent'),
                cmap='jet',
                aspect='equal',
            )
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
            wl_nm = field_data.get('wavelength_nm', WL_UM * 1000)
            ax.set_title(f"|E|^2 at {wl_nm:.1f} nm - {gpu_type} pad={label}")
            fig.savefig(run_dir / "field_intensity.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            stats["field_viz_error"] = str(e)

        stats["status"] = "success"
        save_run_stats(run_dir, stats)
        return stats

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        save_run_stats(run_dir, stats)
        print(f"[{gpu_type}/{category}/{label}] UNEXPECTED ERROR: {e}")
        return None
```

**Step 2: Verify cell parses correctly (no syntax errors)**

Run the cell - it only defines the function, no execution yet.

---

## Task 4: Write the validation cell (test all 8 GPUs in parallel)

**Files:**
- Modify: `examples/gpu_benchmark_dev.ipynb` (add cell 6)

**Step 1: Add validation cell**

**Cell 6 (code): Validation - one test per GPU, all in parallel**
```python
print("=== Validation: testing all 8 GPUs in parallel ===\n")

def run_validation(gpu_type):
    """Run one test simulation on a GPU at a safe size."""
    pad = get_start_padding(gpu_type)
    padding = (pad, pad, 0, 0)
    label = f"pad_{pad}"
    return run_benchmark(gpu_type, padding, "validation", label)

validation_results = {}

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {
        pool.submit(run_validation, gpu): gpu
        for gpu in GPU_TYPES
    }
    for future in as_completed(futures):
        gpu = futures[future]
        try:
            result = future.result()
            status = result["status"] if result else "FAILED"
            validation_results[gpu] = result
            print(f"\n>>> {gpu}: {status}")
            if result and result["status"] == "success":
                print(f"    Dims: {result['dimensions']}, "
                      f"Gcell/s: {result['gcells_per_s']:.2f}, "
                      f"Converged: {result['converged']} at step {result['convergence_step']}")
        except Exception as e:
            print(f"\n>>> {gpu}: EXCEPTION - {e}")
            validation_results[gpu] = None

print("\n=== Validation Summary ===")
for gpu in GPU_TYPES:
    r = validation_results.get(gpu)
    if r and r["status"] == "success":
        print(f"  {gpu:<12s} OK  {r['voxels_M']:>8.1f}M voxels  {r['gcells_per_s']:>6.2f} Gcell/s")
    else:
        print(f"  {gpu:<12s} FAILED")
```

**Step 2: Run cell 6**

This is the first time we spend credits. All 8 GPUs run in parallel. Each runs one simulation at a safe size. Expected: all succeed (or we learn which GPUs have issues).

**Step 3: Review results**

Check `benchmarks/<GPU>/validation/pad_<N>/` directories. Each should contain:
- `stats.json` with full metrics
- `structure.png`
- `mode_profile.png`
- `field_intensity.png`
- `transmission.txt`

Report results to user before proceeding.

---

## Task 5: Write binary search and benchmark execution cells

**Files:**
- Modify: `examples/gpu_benchmark_dev.ipynb` (add cells 7-8)

**Step 1: Add binary search function cell**

**Cell 7 (code): Binary search for max structure size**
```python
def find_max_size(gpu_type, aspect, max_iterations=10):
    """Binary search for maximum structure size on a GPU.

    Args:
        gpu_type: GPU string
        aspect: "elongated" or "balanced"
        max_iterations: max binary search iterations

    Returns:
        dict with best successful result, or None
    """
    vram = GPU_INFO[gpu_type]["vram_gb"]

    # Determine search range based on VRAM
    # pad_low is known to work (from validation or conservative estimate)
    # pad_high is aggressive upper bound
    if vram >= 140:
        pad_low, pad_high = 100, 1500
    elif vram >= 80:
        pad_low, pad_high = 80, 800
    elif vram >= 40:
        pad_low, pad_high = 20, 400
    elif vram >= 24:
        pad_low, pad_high = 10, 200
    else:
        pad_low, pad_high = 5, 100

    best_result = None
    summary = load_summary()

    for iteration in range(max_iterations):
        pad_mid = (pad_low + pad_high) // 2

        # Don't test if range is too narrow
        if pad_high - pad_low <= 5:
            print(f"[{gpu_type}/{aspect}] Converged: pad range [{pad_low}, {pad_high}]")
            break

        if aspect == "elongated":
            padding = (pad_mid, pad_mid, 0, 0)
        else:
            padding = (pad_mid, pad_mid, pad_mid, pad_mid)

        label = f"pad_{pad_mid}"
        print(f"\n[{gpu_type}/{aspect}] Iter {iteration}: trying pad={pad_mid} "
              f"(range [{pad_low}, {pad_high}])")

        result = run_benchmark(gpu_type, padding, aspect, label)

        if result is not None and result.get("status") == "success":
            best_result = result
            pad_low = pad_mid
            print(f"[{gpu_type}/{aspect}] SUCCESS at pad={pad_mid} "
                  f"({result['voxels_M']}M voxels)")
        else:
            pad_high = pad_mid
            print(f"[{gpu_type}/{aspect}] FAILED at pad={pad_mid}")

    # Record best result in summary
    key = f"{gpu_type}_{aspect}"
    with results_lock:
        summary = load_summary()
        summary["runs"][key] = best_result
        save_summary(summary)

    if best_result:
        print(f"\n[{gpu_type}/{aspect}] BEST: pad={best_result['padding']}, "
              f"{best_result['voxels_M']}M voxels, "
              f"{best_result['gcells_per_s']:.2f} Gcell/s")
    else:
        print(f"\n[{gpu_type}/{aspect}] No successful runs!")

    return best_result
```

**Step 2: Add benchmark execution cell**

**Cell 8 (code): Run all GPU benchmarks in parallel**
```python
print("=== Phase 1: Max Size Benchmarks (all GPUs in parallel) ===\n")

benchmark_results = {}

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {}
    for gpu in GPU_TYPES:
        for aspect in ["elongated", "balanced"]:
            f = pool.submit(find_max_size, gpu, aspect)
            futures[f] = (gpu, aspect)

    for future in as_completed(futures):
        gpu, aspect = futures[future]
        try:
            result = future.result()
            benchmark_results[(gpu, aspect)] = result
            status = "OK" if result else "FAILED"
            voxels = result['voxels_M'] if result else 0
            print(f"\n>>> COMPLETED: {gpu}/{aspect} = {status} ({voxels}M voxels)")
        except Exception as e:
            print(f"\n>>> ERROR: {gpu}/{aspect} - {e}")
            benchmark_results[(gpu, aspect)] = None

print("\n=== Phase 1 Complete ===")
print(f"\n{'GPU':<12} {'Elong (M vox)':>15} {'Balanced (M vox)':>18} {'Gcell/s':>10}")
print("-" * 60)
for gpu in GPU_TYPES:
    e = benchmark_results.get((gpu, "elongated"))
    b = benchmark_results.get((gpu, "balanced"))
    e_vox = f"{e['voxels_M']:.1f}" if e else "FAIL"
    b_vox = f"{b['voxels_M']:.1f}" if b else "FAIL"
    gcells = f"{e['gcells_per_s']:.2f}" if e else "N/A"
    print(f"{gpu:<12} {e_vox:>15} {b_vox:>18} {gcells:>10}")
```

---

## Task 6: Write speed benchmark cell

**Files:**
- Modify: `examples/gpu_benchmark_dev.ipynb` (add cell 9)

**Step 1: Add speed benchmark cell**

**Cell 9 (code): Speed benchmarks at reference size**
```python
print("=== Phase 2: Speed Benchmarks ===\n")

# Find the smallest successful max size across all GPUs
summary = load_summary()
all_max_voxels = []
for key, result in summary.get("runs", {}).items():
    if result and result.get("status") == "success":
        all_max_voxels.append(result["voxels"])

if not all_max_voxels:
    print("ERROR: No successful benchmark runs found! Run Phase 1 first.")
else:
    # Use 80% of the smallest max as reference
    reference_voxels = int(min(all_max_voxels) * 0.8)
    print(f"Reference size: {reference_voxels/1e6:.1f}M voxels "
          f"(80% of smallest max: {min(all_max_voxels)/1e6:.1f}M)")

    # Find a padding that gives approximately this voxel count
    # Use calibration data or binary search build_recipe
    # For simplicity, use the validation padding of the smallest GPU
    # (guaranteed to fit on all GPUs)
    smallest_gpu = min(GPU_TYPES, key=lambda g: GPU_INFO[g]["vram_gb"])
    ref_pad = get_start_padding(smallest_gpu)
    ref_padding = (ref_pad, ref_pad, 0, 0)

    print(f"Using padding=({ref_pad},{ref_pad},0,0) for speed reference\n")

    speed_results = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for gpu in GPU_TYPES:
            f = pool.submit(
                run_benchmark, gpu, ref_padding,
                "speed_reference", f"pad_{ref_pad}",
            )
            futures[f] = gpu

        for future in as_completed(futures):
            gpu = futures[future]
            try:
                result = future.result()
                speed_results[gpu] = result
                if result and result["status"] == "success":
                    print(f">>> {gpu}: {result['gcells_per_s']:.2f} Gcell/s "
                          f"({result['convergence_step']} steps)")
                else:
                    print(f">>> {gpu}: FAILED")
            except Exception as e:
                print(f">>> {gpu}: ERROR - {e}")

    print(f"\n{'GPU':<12} {'Gcell/s':>10} {'Sim Time':>10} {'Steps':>8}")
    print("-" * 45)
    for gpu in GPU_TYPES:
        r = speed_results.get(gpu)
        if r and r["status"] == "success":
            print(f"{gpu:<12} {r['gcells_per_s']:>10.2f} {r['sim_time']:>8.1f}s {r['convergence_step']:>8d}")
        else:
            print(f"{gpu:<12} {'FAIL':>10}")
```

---

## Task 7: Write summary generation cell

**Files:**
- Modify: `examples/gpu_benchmark_dev.ipynb` (add cell 10)

**Step 1: Add summary cell**

**Cell 10 (code): Generate summary report**
```python
print("=== Generating Summary ===\n")

summary = load_summary()

# Build final table
rows = []
for gpu in GPU_TYPES:
    info = GPU_INFO[gpu]
    e_key = f"{gpu}_elongated"
    b_key = f"{gpu}_balanced"
    e = summary["runs"].get(e_key)
    b = summary["runs"].get(b_key)
    speed_dir = BENCHMARK_DIR / gpu / "speed_reference"
    speed_stats = None
    for p in speed_dir.glob("*/stats.json"):
        with open(p) as f:
            speed_stats = json.load(f)
        break

    rows.append({
        "gpu": gpu,
        "vram_gb": info["vram_gb"],
        "multiplier": info["multiplier"],
        "max_voxels_elongated": e["voxels_M"] if e else None,
        "max_dims_elongated": e["dimensions"] if e else None,
        "max_voxels_balanced": b["voxels_M"] if b else None,
        "max_dims_balanced": b["dimensions"] if b else None,
        "gcells_per_s": speed_stats["gcells_per_s"] if speed_stats else None,
        "convergence_step": speed_stats["convergence_step"] if speed_stats else None,
    })

summary["final_table"] = rows
summary["metadata"]["component"] = COMPONENT
summary["metadata"]["resolution_nm"] = RESOLUTION_NM
summary["metadata"]["wavelength_um"] = WL_UM
summary["metadata"]["convergence_config"] = {
    "check_every_n": CONV_CONFIG.check_every_n,
    "min_stable_checks": CONV_CONFIG.min_stable_checks,
    "min_steps": CONV_CONFIG.min_steps,
    "power_threshold": CONV_CONFIG.power_threshold,
}
summary["metadata"]["max_steps"] = MAX_STEPS
save_summary(summary)

# Generate markdown report
md_lines = [
    "# GPU Benchmark Results\n",
    f"**Component:** {COMPONENT}",
    f"**Resolution:** {RESOLUTION_NM}nm",
    f"**Wavelength:** {WL_UM}um",
    f"**Max Steps:** {MAX_STEPS}",
    f"**Convergence:** check_every={CONV_CONFIG.check_every_n}, "
    f"min_stable={CONV_CONFIG.min_stable_checks}, min_steps={CONV_CONFIG.min_steps}",
    f"**Generated:** {datetime.now().isoformat()}\n",
    "## Results\n",
    "| GPU | VRAM | Mult | Max Voxels (Elong) | Max Voxels (Balanced) | Gcell/s | Steps |",
    "|-----|------|------|--------------------|-----------------------|---------|-------|",
]

for r in rows:
    e_v = f"{r['max_voxels_elongated']:.1f}M" if r['max_voxels_elongated'] else "N/A"
    b_v = f"{r['max_voxels_balanced']:.1f}M" if r['max_voxels_balanced'] else "N/A"
    gcells = f"{r['gcells_per_s']:.2f}" if r['gcells_per_s'] else "N/A"
    steps = str(r['convergence_step']) if r['convergence_step'] else "N/A"
    md_lines.append(
        f"| {r['gpu']} | {r['vram_gb']}GB | {r['multiplier']}x | {e_v} | {b_v} | {gcells} | {steps} |"
    )

md_lines.append("\n*Gcell/s = billion grid-point-step updates per second (higher is better)*")

report = "\n".join(md_lines)

with open(BENCHMARK_DIR / "summary.md", "w") as f:
    f.write(report)

print(report)
print(f"\nSaved to {BENCHMARK_DIR / 'summary.md'}")
```

---

## Task 8: Run validation and report

**Step 1: Run cells 1-4** (setup, constants, persistence, calibration)

No credits consumed. Verify calibration output makes sense.

**Step 2: Run cell 6** (validation - all 8 GPUs in parallel)

This is the first credit-consuming step. Watch for:
- All 8 GPUs should succeed at their safe starting size
- Any GPU that fails needs its start padding reduced further
- Check artifact files in `benchmarks/GPU/validation/` dirs

**Step 3: Report results to user**

Show the validation summary table and wait for user approval before proceeding to the full binary search (cells 7-8).
