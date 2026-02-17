"""
Probe max structure sizes on B200 and H200 to build VRAM->voxels model.
These are the cheapest GPUs to run, so we use them for data collection.
Runs several sizes in sequence per GPU (both GPUs in parallel).
"""
import matplotlib
matplotlib.use('Agg')

import hyperwave_community as hwc
from hyperwave_community import ConvergenceConfig
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

hwc.configure_api(api_key="9e293a83-feb0-4275-b6f0-540ba935b4bb")
print(hwc.get_account_info())

# Constants
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
BENCHMARK_DIR = Path("../benchmarks")

# Use fewer steps for probes -- we just need to know if it fits or OOMs.
# 5000 steps is enough to detect OOM (happens in first few seconds).
PROBE_STEPS = 5000

CONV_CONFIG = ConvergenceConfig(
    check_every_n=1000,
    min_stable_checks=3,
    min_steps=5000,
    power_threshold=1e-6,
)

results_lock = threading.Lock()


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


def probe_size(gpu_type, padding_tuple):
    """Run a short simulation to check if a size fits on a GPU.

    Returns dict with status on success, None on OOM/failure.
    """
    pad_label = f"pad_{padding_tuple[0]}"
    probe_dir = BENCHMARK_DIR / gpu_type / "size_probe" / pad_label
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    stats_path = probe_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            existing = json.load(f)
        if existing.get("status") == "success":
            print(f"[{gpu_type}/probe/{pad_label}] Cached: SUCCESS "
                  f"({existing.get('voxels_M', '?')}M voxels)")
            return existing
        if existing.get("status") == "failed_simulation":
            print(f"[{gpu_type}/probe/{pad_label}] Cached: OOM "
                  f"({existing.get('voxels_M', '?')}M voxels)")
            return None

    stats = {
        "gpu_type": gpu_type,
        "padding": list(padding_tuple),
        "timestamp": datetime.now().isoformat(),
        "probe_steps": PROBE_STEPS,
    }

    # Build recipe (FREE)
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
        stats["status"] = "failed_build"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return None

    dims = recipe_result['dimensions']
    voxels = dims[0] * dims[1] * dims[2]
    stats["dimensions"] = list(dims)
    stats["voxels"] = voxels
    stats["voxels_M"] = round(voxels / 1e6, 1)
    print(f"[{gpu_type}/probe/{pad_label}] {dims[0]}x{dims[1]}x{dims[2]} = {voxels/1e6:.1f}M voxels")

    # Build monitors (FREE)
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
        stats["status"] = "failed_monitors"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return None

    # Compute freq band (FREE)
    freq_result = retry_api_call(
        hwc.compute_freq_band,
        wl_min_um=WL_UM, wl_max_um=WL_UM, n_freqs=1,
        resolution_um=recipe_result['resolution_um'],
    )
    if freq_result is None:
        stats["status"] = "failed_freq"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return None

    # Solve mode source (FREE)
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
        stats["status"] = "failed_mode"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return None

    # Get absorber params (LOCAL, FREE)
    absorber_params = hwc.get_optimized_absorber_params(
        resolution_nm=RESOLUTION_NM,
        wavelength_um=WL_UM,
        structure_dimensions=recipe_result['dimensions'],
    )

    # Estimate cost (FREE)
    cost_estimate = hwc.estimate_cost(
        grid_points=voxels,
        max_steps=PROBE_STEPS,
        gpu_type=gpu_type,
    )
    stats["estimated_cost"] = cost_estimate
    print(f"[{gpu_type}/probe/{pad_label}] Est cost: {cost_estimate}")

    # Run short simulation (COSTS CREDITS -- but only 5000 steps)
    sim_start = time.time()
    results = retry_api_call(
        hwc.run_simulation,
        max_retries=1,
        base_delay=5.0,
        device_type=COMPONENT,
        recipe_result=recipe_result,
        monitor_result=monitor_result,
        freq_result=freq_result,
        source_result=source_result,
        num_steps=PROBE_STEPS,
        gpu_type=gpu_type,
        convergence=CONV_CONFIG,
        absorption_widths=absorber_params['absorption_widths'],
        absorption_coeff=absorber_params['absorber_coeff'],
    )
    sim_wall_time = time.time() - sim_start

    if results is None:
        stats["status"] = "failed_simulation"
        stats["sim_wall_time"] = sim_wall_time
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[{gpu_type}/probe/{pad_label}] OOM/FAILED after {sim_wall_time:.1f}s")
        return None

    stats["status"] = "success"
    stats["sim_time"] = results.get('sim_time', 0)
    stats["performance"] = results.get('performance', 0)
    stats["gcells_per_s"] = results.get('performance', 0) / 1e9
    stats["sim_wall_time"] = sim_wall_time
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[{gpu_type}/probe/{pad_label}] SUCCESS: {stats['gcells_per_s']:.2f} Gcell/s, "
          f"{stats['sim_time']:.1f}s")
    return stats


def find_oom_point(gpu_type, paddings):
    """Try sizes in order until OOM. Returns (last_success, first_fail)."""
    last_success = None
    for pad in paddings:
        padding = (pad, pad, 0, 0)
        result = probe_size(gpu_type, padding)
        if result is not None and result.get("status") == "success":
            last_success = result
        else:
            print(f"[{gpu_type}] OOM at pad={pad} ({last_success['voxels_M'] if last_success else 0}M was last OK)")
            return last_success, pad
    print(f"[{gpu_type}] All sizes fit! Last: pad={paddings[-1]}")
    return last_success, None


# Probe sizes: start from known-good (pad=100 = 125.4M) and go up
# B200 (192GB) -- can probably handle much larger
b200_pads = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500]
# H200 (141GB) -- slightly less than B200
h200_pads = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

print("=== Size Probes: Finding OOM points on B200 and H200 ===\n")
print("Using 5000-step probes (cheap, just enough to detect OOM)\n")

# Run B200 and H200 probes in parallel (sequential within each GPU)
with ThreadPoolExecutor(max_workers=2) as pool:
    f_b200 = pool.submit(find_oom_point, "B200", b200_pads)
    f_h200 = pool.submit(find_oom_point, "H200", h200_pads)

    b200_result = f_b200.result()
    h200_result = f_h200.result()

print("\n=== Results ===")
if b200_result[0]:
    print(f"B200 (192GB): max tested = {b200_result[0]['voxels_M']}M voxels "
          f"(pad={b200_result[0]['padding'][0]})")
    if b200_result[1]:
        print(f"  OOM at pad={b200_result[1]}")
if h200_result[0]:
    print(f"H200 (141GB): max tested = {h200_result[0]['voxels_M']}M voxels "
          f"(pad={h200_result[0]['padding'][0]})")
    if h200_result[1]:
        print(f"  OOM at pad={h200_result[1]}")

# Calculate bytes per voxel from OOM points
print("\n=== VRAM Model ===")
data_points = []
if b200_result[1]:
    # B200 OOMed -- calculate from last success + first fail
    ok_voxels = b200_result[0]['voxels']
    data_points.append(("B200", 192, ok_voxels, "upper"))
if h200_result[1]:
    ok_voxels = h200_result[0]['voxels']
    data_points.append(("H200", 141, ok_voxels, "upper"))

# T4 OOM at 93.1M voxels with 16GB
data_points.append(("T4", 16, 93.1e6, "lower"))

for name, vram, voxels, bound in data_points:
    bpv = (vram * 1e9) / voxels
    print(f"  {name}: {vram}GB / {voxels/1e6:.1f}M voxels = {bpv:.0f} bytes/voxel ({bound} bound)")
