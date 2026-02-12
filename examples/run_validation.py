"""
Standalone script extracted from gpu_benchmark_dev.ipynb cells 0-5.
Runs the validation benchmarks (all 8 GPUs in parallel).

Usage:
    cd /home/dq4443/dev/work/hyperwave-community/examples
    python run_validation.py
"""

# ---------------------------------------------------------------------------
# Cell 1: Setup
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')

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
import matplotlib.pyplot as plt

hwc.configure_api(api_key="9e293a83-feb0-4275-b6f0-540ba935b4bb")
print(hwc.get_account_info())

# ---------------------------------------------------------------------------
# Cell 2: Constants
# ---------------------------------------------------------------------------
GPU_TYPES = ["B200", "H200", "H100", "A100-80GB", "A100-40GB", "L40S", "A10G", "T4"]

GPU_INFO = {
    "B200":      {"vram_gb": 192, "multiplier": 2.5},
    "H200":      {"vram_gb": 141, "multiplier": 2.0},
    "H100":      {"vram_gb": 80,  "multiplier": 1.5},
    "A100-80GB": {"vram_gb": 80,  "multiplier": 1.0},
    "A100-40GB": {"vram_gb": 40,  "multiplier": 0.8},
    "L40S":      {"vram_gb": 48,  "multiplier": 0.7},
    "A10G":      {"vram_gb": 24,  "multiplier": 0.4},
    "T4":        {"vram_gb": 16,  "multiplier": 0.3},
}

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


def get_start_padding(gpu_type):
    vram = GPU_INFO[gpu_type]["vram_gb"]
    if vram >= 80:
        return 100
    elif vram >= 40:
        return 40
    elif vram >= 24:
        return 20
    else:
        return 10


# ---------------------------------------------------------------------------
# Cell 3: Persistence helpers
# ---------------------------------------------------------------------------
results_lock = threading.Lock()


def ensure_run_dir(gpu_type, category, label):
    run_dir = BENCHMARK_DIR / gpu_type / category / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_summary():
    summary_path = BENCHMARK_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {"runs": {}, "metadata": {}}


def save_summary(summary):
    summary["metadata"]["last_updated"] = datetime.now().isoformat()
    with results_lock:
        summary_path = BENCHMARK_DIR / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)


def update_summary(key, value):
    """Thread-safe read-modify-write for summary.json."""
    with results_lock:
        summary = load_summary()
        summary["runs"][key] = value
        summary["metadata"]["last_updated"] = datetime.now().isoformat()
        summary_path = BENCHMARK_DIR / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)


def save_run_stats(run_dir, stats):
    with open(run_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)


def capture_stdout(func, *args, **kwargs):
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return result, buffer.getvalue()


def retry_api_call(func, *args, max_retries=4, base_delay=3.0, **kwargs):
    """Retry an API call with exponential backoff on failure."""
    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            if result is not None:
                return result
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"  [retry] {func.__name__} returned None, retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                return None
        except Exception as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"  [retry] {func.__name__} raised {type(e).__name__}: {e}, "
                      f"retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"  [retry] {func.__name__} failed after {max_retries} retries: {e}")
                raise
    return None


# ---------------------------------------------------------------------------
# Cell 4: Calibration
# ---------------------------------------------------------------------------
print("=== Calibration: padding -> dimensions mapping ===\n")

test_paddings = [10, 20, 40, 60, 80, 100, 150, 200, 300]
calibration_data = {}

for pad in test_paddings:
    padding = (pad, pad, 0, 0)
    result = hwc.build_recipe(
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

print("\n=== Balanced padding (all sides) ===\n")
for pad in [10, 20, 40, 60, 80, 100]:
    padding = (pad, pad, pad, pad)
    result = hwc.build_recipe(
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


# ---------------------------------------------------------------------------
# Cell 5: run_benchmark function
# ---------------------------------------------------------------------------
def run_benchmark(gpu_type, padding_tuple, category, label, num_steps=MAX_STEPS):
    """Run the full FDTD pipeline and save all artifacts."""
    run_dir = ensure_run_dir(gpu_type, category, label)

    if (run_dir / "stats.json").exists():
        with open(run_dir / "stats.json") as f:
            existing = json.load(f)
        if existing.get("status") == "success":
            print(f"[{gpu_type}/{category}/{label}] Already done, skipping")
            return existing
        print(f"[{gpu_type}/{category}/{label}] Previous run status={existing.get('status')}, retrying")

    stats = {
        "gpu_type": gpu_type,
        "category": category,
        "label": label,
        "padding": list(padding_tuple),
        "timestamp": datetime.now().isoformat(),
        "status": "started",
    }

    try:
        recipe_result = retry_api_call(
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
            save_run_stats(run_dir, stats)
            print(f"[{gpu_type}/{category}/{label}] build_recipe FAILED")
            return None

        dims = recipe_result['dimensions']
        voxels = dims[0] * dims[1] * dims[2]
        stats["dimensions"] = list(dims)
        stats["voxels"] = voxels
        stats["voxels_M"] = round(voxels / 1e6, 1)
        print(f"[{gpu_type}/{category}/{label}] Structure: {dims[0]}x{dims[1]}x{dims[2]} = {voxels/1e6:.1f}M voxels")

        monitor_result = retry_api_call(
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

        freq_result = retry_api_call(
            hwc.compute_freq_band,
            wl_min_um=WL_UM, wl_max_um=WL_UM, n_freqs=1,
            resolution_um=recipe_result['resolution_um'],
        )
        if freq_result is None:
            stats["status"] = "failed_compute_freq"
            save_run_stats(run_dir, stats)
            print(f"[{gpu_type}/{category}/{label}] compute_freq_band FAILED")
            return None

        source_result = retry_api_call(
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

        try:
            hwc.visualize_mode_source(
                source_field=source_result['source_field'],
                source_offset=source_result['source_offset'],
                show=False,
                save_path=str(run_dir / "mode_profile.png"),
            )
        except Exception as e:
            stats["mode_viz_error"] = str(e)

        absorber_params = hwc.get_optimized_absorber_params(
            resolution_nm=RESOLUTION_NM,
            wavelength_um=WL_UM,
            structure_dimensions=recipe_result['dimensions'],
        )
        stats["absorber_width"] = absorber_params['absorber_width']
        stats["absorber_coeff"] = absorber_params['absorber_coeff']
        stats["absorption_widths"] = list(absorber_params['absorption_widths'])

        cost_estimate = hwc.estimate_cost(
            grid_points=voxels,
            max_steps=num_steps,
            gpu_type=gpu_type,
        )
        stats["estimated_cost"] = cost_estimate
        print(f"[{gpu_type}/{category}/{label}] Estimated cost: {cost_estimate}")

        sim_start = time.time()
        results = retry_api_call(
            hwc.run_simulation,
            max_retries=2,
            base_delay=5.0,
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

        if results is None:
            stats["status"] = "failed_simulation"
            stats["sim_wall_time"] = sim_wall_time
            save_run_stats(run_dir, stats)
            print(f"[{gpu_type}/{category}/{label}] SIMULATION FAILED (likely OOM) after {sim_wall_time:.1f}s")
            return None

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

        try:
            with results_lock:
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

        fig = None
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
        except Exception as e:
            stats["field_viz_error"] = str(e)
        finally:
            if fig is not None:
                plt.close(fig)

        stats["status"] = "success"
        save_run_stats(run_dir, stats)
        return stats

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        save_run_stats(run_dir, stats)
        print(f"[{gpu_type}/{category}/{label}] UNEXPECTED ERROR: {e}")
        return None


# ---------------------------------------------------------------------------
# Cell 6: Validation (all 8 GPUs in parallel)
# ---------------------------------------------------------------------------
print("=== Validation: testing all 8 GPUs in parallel ===\n")


def run_validation(gpu_type):
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
