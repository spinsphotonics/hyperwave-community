"""
Probe H100 (80GB) to find OOM point for VRAM->voxels model.
Runs 3 probes in parallel at a time.
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
from concurrent.futures import ThreadPoolExecutor, as_completed

hwc.configure_api(api_key="9e293a83-feb0-4275-b6f0-540ba935b4bb")
print(hwc.get_account_info())

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
PROBE_STEPS = 5000
GPU = "H100"

CONV_CONFIG = ConvergenceConfig(
    check_every_n=1000,
    min_stable_checks=3,
    min_steps=5000,
    power_threshold=1e-6,
)

results_lock = threading.Lock()


def retry_api_call(func, *args, max_retries=4, base_delay=3.0, **kwargs):
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


def probe_size(pad):
    padding = (pad, pad, 0, 0)
    pad_label = f"pad_{pad}"
    probe_dir = BENCHMARK_DIR / GPU / "size_probe" / pad_label
    probe_dir.mkdir(parents=True, exist_ok=True)

    stats_path = probe_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            existing = json.load(f)
        if existing.get("status") == "success":
            print(f"[H100/probe/{pad_label}] Cached SUCCESS ({existing.get('voxels_M', '?')}M)")
            return existing
        if existing.get("status") == "failed_simulation":
            print(f"[H100/probe/{pad_label}] Cached OOM ({existing.get('voxels_M', '?')}M)")
            return existing

    stats = {"gpu_type": GPU, "padding": list(padding), "timestamp": datetime.now().isoformat()}

    recipe_result = retry_api_call(
        hwc.build_recipe,
        component_name=COMPONENT, resolution_nm=RESOLUTION_NM,
        extension_length=EXTENSION_LENGTH, n_core=N_CORE, n_clad=N_CLAD,
        wg_height_um=WG_HEIGHT_UM, total_height_um=TOTAL_HEIGHT_UM,
        padding=padding, density_radius=DENSITY_RADIUS, vertical_radius=VERTICAL_RADIUS,
    )
    if recipe_result is None:
        stats["status"] = "failed_build"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    dims = recipe_result['dimensions']
    voxels = dims[0] * dims[1] * dims[2]
    stats["dimensions"] = list(dims)
    stats["voxels"] = voxels
    stats["voxels_M"] = round(voxels / 1e6, 1)
    print(f"[H100/probe/{pad_label}] {dims[0]}x{dims[1]}x{dims[2]} = {voxels/1e6:.1f}M voxels")

    monitor_result = retry_api_call(
        hwc.build_monitors,
        port_info=recipe_result['port_info'], dimensions=recipe_result['dimensions'],
        source_port=SOURCE_PORT, resolution_um=recipe_result['resolution_um'],
        structure_recipe=recipe_result['recipe'], show_structure=False,
        include_field_monitor=True,
    )
    if monitor_result is None:
        stats["status"] = "failed_monitors"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    freq_result = retry_api_call(
        hwc.compute_freq_band,
        wl_min_um=WL_UM, wl_max_um=WL_UM, n_freqs=1,
        resolution_um=recipe_result['resolution_um'],
    )
    if freq_result is None:
        stats["status"] = "failed_freq"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    source_result = retry_api_call(
        hwc.solve_mode_source,
        density_core=recipe_result['density_core'], density_clad=recipe_result['density_clad'],
        source_x_position=monitor_result['source_position'],
        mode_bounds=monitor_result['mode_bounds'],
        layer_config=recipe_result['layer_config'], eps_values=recipe_result['eps_values'],
        freq_band=freq_result['freq_band'], mode_num=MODE_NUM, show_mode=False,
    )
    if source_result is None:
        stats["status"] = "failed_mode"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    absorber_params = hwc.get_optimized_absorber_params(
        resolution_nm=RESOLUTION_NM, wavelength_um=WL_UM,
        structure_dimensions=recipe_result['dimensions'],
    )

    sim_start = time.time()
    results = retry_api_call(
        hwc.run_simulation,
        max_retries=1, base_delay=5.0,
        device_type=COMPONENT, recipe_result=recipe_result,
        monitor_result=monitor_result, freq_result=freq_result,
        source_result=source_result, num_steps=PROBE_STEPS,
        gpu_type=GPU, convergence=CONV_CONFIG,
        absorption_widths=absorber_params['absorption_widths'],
        absorption_coeff=absorber_params['absorber_coeff'],
    )
    sim_wall_time = time.time() - sim_start

    if results is None:
        stats["status"] = "failed_simulation"
        stats["sim_wall_time"] = sim_wall_time
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[H100/probe/{pad_label}] OOM at {voxels/1e6:.1f}M voxels")
        return stats

    stats["status"] = "success"
    stats["sim_time"] = results.get('sim_time', 0)
    stats["performance"] = results.get('performance', 0)
    stats["gcells_per_s"] = results.get('performance', 0) / 1e9
    stats["sim_wall_time"] = sim_wall_time
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[H100/probe/{pad_label}] SUCCESS: {voxels/1e6:.1f}M, {stats['gcells_per_s']:.2f} Gcell/s")
    return stats


# Round 1: Wide spread to find approximate OOM region
# H100 has 80GB. At ~275 bytes/voxel -> ~291M max.
# pad=300 -> ~197M, pad=500 -> ~269M (from B200 data), pad=600 -> ~305M
# Try 300, 450, 600 in parallel
print("=== H100 Size Probes: Round 1 (wide spread) ===\n")

round1_pads = [300, 450, 600]
round1_results = {}

with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(probe_size, p): p for p in round1_pads}
    for f in as_completed(futures):
        p = futures[f]
        round1_results[p] = f.result()

print("\n--- Round 1 Results ---")
for p in sorted(round1_results):
    r = round1_results[p]
    status = r.get("status", "unknown")
    voxels = r.get("voxels_M", "?")
    print(f"  pad={p}: {status} ({voxels}M voxels)")

# Round 2: Narrow down based on Round 1
# Find the boundary between success and failure
successes = sorted([p for p, r in round1_results.items() if r.get("status") == "success"])
failures = sorted([p for p, r in round1_results.items() if r.get("status") == "failed_simulation"])

if successes and failures:
    last_ok = max(successes)
    first_fail = min(failures)
    # Probe 3 points between last_ok and first_fail
    gap = first_fail - last_ok
    step = gap // 4
    round2_pads = [last_ok + step, last_ok + 2*step, last_ok + 3*step]
    print(f"\n=== H100 Size Probes: Round 2 (narrowing {last_ok}-{first_fail}) ===\n")
elif not failures:
    # All succeeded, go higher
    highest = max(successes) if successes else 600
    round2_pads = [highest + 100, highest + 200, highest + 300]
    print(f"\n=== H100 Size Probes: Round 2 (going higher from {highest}) ===\n")
else:
    # All failed, go lower
    lowest = min(failures) if failures else 300
    round2_pads = [lowest - 150, lowest - 100, lowest - 50]
    print(f"\n=== H100 Size Probes: Round 2 (going lower from {lowest}) ===\n")

round2_results = {}
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(probe_size, p): p for p in round2_pads}
    for f in as_completed(futures):
        p = futures[f]
        round2_results[p] = f.result()

print("\n--- Round 2 Results ---")
for p in sorted(round2_results):
    r = round2_results[p]
    status = r.get("status", "unknown")
    voxels = r.get("voxels_M", "?")
    print(f"  pad={p}: {status} ({voxels}M voxels)")

# Final summary
all_results = {**round1_results, **round2_results}
print("\n=== H100 Final Summary ===")
for p in sorted(all_results):
    r = all_results[p]
    status = r.get("status", "unknown")
    voxels = r.get("voxels_M", "?")
    gcells = r.get("gcells_per_s", "-")
    print(f"  pad={p:>4d}: {status:<20s} {str(voxels):>8s}M voxels  {str(gcells)} Gcell/s")

successes_all = {p: r for p, r in all_results.items() if r.get("status") == "success"}
failures_all = {p: r for p, r in all_results.items() if r.get("status") == "failed_simulation"}

if successes_all and failures_all:
    best = max(successes_all.items(), key=lambda x: x[0])
    worst = min(failures_all.items(), key=lambda x: x[0])
    print("\nH100 (80GB) bracket:")
    print(f"  Last success: pad={best[0]} -> {best[1]['voxels_M']}M voxels")
    print(f"  First OOM:    pad={worst[0]} -> {worst[1]['voxels_M']}M voxels")
    bpv_upper = 80e9 / (best[1]['voxels'] )
    bpv_lower = 80e9 / (worst[1]['voxels'])
    print(f"  bytes/voxel: {bpv_lower:.1f} - {bpv_upper:.1f}")
