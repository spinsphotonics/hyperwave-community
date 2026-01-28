"""Hyperwave API Client for GPU-accelerated FDTD photonics simulations.

Three workflow levels are available:

1. TWO-STAGE WORKFLOW (Recommended for most users):
    - prepare_simulation() - Setup simulation parameters (CPU on Modal)
    - run_simulation() - Run FDTD with pre-computed setup (GPU on Modal)

2. GRANULAR WORKFLOW (For advanced control):
    - build_recipe() - Create structure from GDSFactory component
    - build_monitors() - Create monitors from port information
    - compute_freq_band() - Convert wavelengths to frequencies
    - solve_mode_source() - Solve for waveguide mode
    - get_default_absorber_params() - Get absorber configuration

3. ONE-SHOT WORKFLOW (For quick tests):
    - simulate() - Combines setup + simulation in one call

Utility Functions:
    - configure_api() - Set API credentials and endpoint
    - get_account_info() - Get account info and credit balance
    - estimate_cost() - Estimate simulation cost before running

Environment Variables:
    HYPERWAVE_API_KEY: API authentication key
    HYPERWAVE_API_URL: API endpoint URL (optional, defaults to production)
"""

import os
import base64
import io
import time
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import requests


# Global API configuration
_API_CONFIG = {
    'api_key': None,
    'api_url': 'https://hyperwave-cloud.onrender.com'
}


def configure_api(api_key: Optional[str] = None, api_url: Optional[str] = None):
    """Configure API credentials and endpoint.

    Args:
        api_key: API authentication key. If None, uses HYPERWAVE_API_KEY environment variable.
        api_url: API endpoint URL. If None, uses HYPERWAVE_API_URL environment variable
            or defaults to production endpoint.

    Raises:
        ValueError: If no API key is provided and HYPERWAVE_API_KEY is not set.

    Example:
        >>> import hyperwave_community as hwc
        >>> hwc.configure_api(api_key='your-key-here')
    """
    global _API_CONFIG

    if api_key is not None:
        _API_CONFIG['api_key'] = api_key
    elif 'HYPERWAVE_API_KEY' in os.environ:
        _API_CONFIG['api_key'] = os.environ['HYPERWAVE_API_KEY']

    if api_url is not None:
        _API_CONFIG['api_url'] = api_url
    elif 'HYPERWAVE_API_URL' in os.environ:
        _API_CONFIG['api_url'] = os.environ['HYPERWAVE_API_URL']

    if _API_CONFIG['api_key'] is None:
        raise ValueError(
            "API key not provided. Set HYPERWAVE_API_KEY environment variable "
            "or call configure_api(api_key='your-key') first."
        )


def _get_api_config() -> Dict[str, str]:
    """Get current API configuration."""
    if _API_CONFIG['api_key'] is None:
        if 'HYPERWAVE_API_KEY' in os.environ:
            _API_CONFIG['api_key'] = os.environ['HYPERWAVE_API_KEY']
        else:
            raise RuntimeError(
                "API not configured. Call configure_api() or set HYPERWAVE_API_KEY "
                "environment variable first."
            )
    return _API_CONFIG


def encode_array(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string for API transmission."""
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_array(b64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    buffer = io.BytesIO(base64.b64decode(b64_str))
    return np.load(buffer)


def _handle_api_error(e: requests.exceptions.HTTPError, operation: str) -> None:
    """Handle common API errors with user-friendly messages."""
    if e.response is not None:
        status_code = e.response.status_code
        if status_code == 401:
            print("No API key detected in request.")
            print("Sign up for free at spinsphotonics.com to get your API key.")
        elif status_code == 403:
            print("Provided API key is invalid.")
            print("Please verify your API key in your dashboard at spinsphotonics.com/dashboard")
        elif status_code == 402:
            print(f"Insufficient credits for {operation}.")
            print("Add credits to your account at spinsphotonics.com/billing")
        elif status_code == 429:
            print("Too many concurrent simulations.")
        elif status_code == 502:
            print("Service temporarily unavailable. Please retry.")
        else:
            print(f"Unexpected error (Code: {status_code})")
    else:
        print("Communication error.")


# =============================================================================
# ACCOUNT INFO
# =============================================================================

def get_account_info(quiet: bool = False) -> Optional[Dict[str, Any]]:
    """Verify API key and get account information including credit balance.

    Args:
        quiet: If True, don't print the greeting message.

    Returns:
        Dictionary with valid, name, email, api_key_prefix, credits_balance, credits_balance_usd.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    try:
        response = requests.post(
            f"{API_URL}/account_info",
            params={"api_key": API_KEY},
            timeout=30
        )
        response.raise_for_status()
        info = response.json()

        if not quiet:
            name = info.get('name', 'User')
            email = info.get('email', '')
            prefix = info.get('api_key_prefix', '')
            balance = info.get('credits_balance', 0)

            print()
            print(f"  Welcome back, {name}!")
            print(f"  {email}")
            print()
            print(f"  API Key:  {prefix}...")
            print(f"  Credits:  {balance:.2f}")
            print()

        return info

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            print("Invalid API key.")
            print("Please verify your API key in your dashboard at spinsphotonics.com/dashboard")
        else:
            print(f"Error getting account info: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting account info: {e}")
        return None


# =============================================================================
# COST ESTIMATION
# =============================================================================

def estimate_cost(
    grid_points: Optional[int] = None,
    structure_shape: Optional[Tuple[int, int, int, int]] = None,
    max_steps: int = 10000,
    gpu_type: str = "H100",
    simulation_type: str = "fdtd_simulation",
) -> Optional[Dict[str, Any]]:
    """Estimate simulation cost before running (no auth required)."""
    API_URL = _API_CONFIG['api_url']

    request_data = {
        "max_steps": max_steps,
        "gpu_type": gpu_type,
        "simulation_type": simulation_type
    }

    if grid_points is not None:
        request_data["grid_points"] = grid_points
    elif structure_shape is not None:
        request_data["structure_shape"] = list(structure_shape)
    else:
        print("Either grid_points or structure_shape must be provided.")
        return None

    try:
        response = requests.post(f"{API_URL}/estimate_cost", json=request_data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error estimating cost: {e}")
        return None


# =============================================================================
# TWO-STAGE WORKFLOW (Recommended)
# =============================================================================

def prepare_simulation(
    device_type: str,
    pdk_config: Dict[str, Any],
    source_port: str = "o1",
    wavelength_um: float = 1.55,
    cells_per_wavelength: int = 25,
    mode_num: int = 0,
    device_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Stage 1: Prepare simulation inputs on Modal CPU.

    This performs all setup work needed before simulation:
    - Build structure recipe from GDSFactory component
    - Create monitors at port locations
    - Solve for waveguide mode at source port
    - Return all data needed for simulation

    Args:
        device_type: Device type name (e.g., "mmi2x2", "mmi1x2").
        pdk_config: PDK configuration dict with material parameters:
            - n_core: Core refractive index (e.g., 3.48 for silicon)
            - n_clad: Cladding refractive index (e.g., 1.45 for SiO2)
            - wg_height_um: Waveguide height in micrometers (e.g., 0.22)
            - clad_top_um: Top cladding thickness in micrometers (e.g., 1.89)
            - clad_bot_um: Bottom cladding (BOX) thickness in micrometers (e.g., 2.0)
        source_port: Input port name (e.g., "o1", "o2").
        wavelength_um: Wavelength in micrometers (default: 1.55).
        cells_per_wavelength: FDTD resolution (default: 25). Higher = more accurate but slower.
        mode_num: Mode number to solve (0 = fundamental, default: 0).
        device_params: Optional dict of device-specific parameters.

    Returns:
        Dict with:
        - mode_preview: Mode visualization data (n_eff, Ex, Ey, Ez)
        - setup_data: Pre-computed setup for run_simulation()
        - wavelength_um: Wavelength used
        - resolution_nm: Resolution in nanometers
        - source_port: Actual source port name

    Example:
        >>> pdk_config = {
        ...     "n_core": 3.48,
        ...     "n_clad": 1.45,
        ...     "wg_height_um": 0.22,
        ...     "clad_top_um": 1.89,
        ...     "clad_bot_um": 2.0,
        ... }
        >>> setup = hwc.prepare_simulation(
        ...     device_type="mmi2x2",
        ...     pdk_config=pdk_config,
        ...     source_port="o1",
        ...     wavelength_um=1.55,
        ...     cells_per_wavelength=25,
        ... )
        >>> print(f"Mode n_eff: {setup['mode_preview']['n_eff']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Validate pdk_config
    required_fields = ["n_core", "n_clad", "wg_height_um", "clad_top_um", "clad_bot_um"]
    missing = [f for f in required_fields if f not in pdk_config]
    if missing:
        raise ValueError(f"pdk_config missing required fields: {missing}")

    print(f"Preparing simulation for {device_type}...")
    print(f"  Source: {source_port}, Wavelength: {wavelength_um} um, Resolution: {cells_per_wavelength} cells/wavelength")

    body = {
        "device_type": device_type,
        "device_params": device_params or {},
        "source_port": source_port,
        "wavelength_um": wavelength_um,
        "cells_per_wavelength": cells_per_wavelength,
        "pdk_config": pdk_config,
        "mode_num": mode_num,
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/prepare_simulation",
            json=body,
            headers=headers,
            timeout=600
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") != "success":
            raise Exception(f"Failed to prepare simulation: {result}")

        mode_preview = result.get("mode_preview", {})
        n_eff = mode_preview.get("n_eff", "N/A")
        print(f"Mode solved: n_eff={n_eff}")
        print(f"Setup complete. Ready for run_simulation()")

        return {
            "mode_preview": mode_preview,
            "setup_data": result.get("setup_data"),
            "wavelength_um": result.get("wavelength_um"),
            "resolution_nm": result.get("resolution_nm"),
            "source_port": result.get("source_port"),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "prepare_simulation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error preparing simulation: {e}")
        return None


def run_simulation(
    device_type: str,
    setup_data: Dict[str, Any],
    num_steps: int = 20000,
    check_every_n: int = 1000,
    source_ramp_periods: float = 10.0,
    gpu_type: str = "H100",
    min_steps: int = 0,
    min_stable_checks: int = 3,
    absorber_width: int = 82,
    absorber_coeff: float = 0.0006173770394704579,
    significant_power_threshold: float = 1e-6,
    required_ports: Optional[List[str]] = None,
    poll_interval: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Stage 2: Run FDTD simulation on Modal GPU with pre-computed setup.

    This is the fast path when you already have setup_data from prepare_simulation().
    Skips all setup work and goes directly to GPU simulation.

    Args:
        device_type: Device type name (for tracking).
        setup_data: Pre-computed setup from prepare_simulation()['setup_data'].
        num_steps: Maximum FDTD steps (default: 20000).
        check_every_n: Convergence check interval (default: 1000).
        source_ramp_periods: Source ramp-up periods (default: 10.0).
        gpu_type: GPU type - "B200", "H200", "H100", "A100-80GB", etc.
        min_steps: Minimum steps before early stopping (default: 0).
        min_stable_checks: Consecutive stable checks for convergence (default: 3).
        absorber_width: Absorber width in cells (default: 82).
        absorber_coeff: Absorber coefficient.
        significant_power_threshold: Min power level for convergence check.
        required_ports: List of port names to check for convergence.
        poll_interval: Seconds between status polls (default: 2.0).

    Returns:
        Dict with simulation results:
        - s_parameters: Analyzed transmission data
        - field_intensity: 2D field intensity for visualization
        - port_fields: Per-port field data
        - sim_time: GPU simulation time in seconds
        - converged: Whether simulation converged

    Example:
        >>> setup = hwc.prepare_simulation(device_type="mmi2x2", pdk_config=pdk_config)
        >>> results = hwc.run_simulation(
        ...     device_type="mmi2x2",
        ...     setup_data=setup['setup_data'],
        ...     num_steps=30000,
        ...     gpu_type="H100",
        ... )
        >>> print(f"T_total: {results['s_parameters']['T_total']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    if setup_data is None:
        raise Exception(
            "setup_data is required. Use prepare_simulation() first, or use simulate() for one-shot workflow."
        )

    # Handle both formats: full prepare_simulation result or just setup_data
    if "setup_data" in setup_data and "source_field_base64" not in setup_data:
        setup_data = setup_data["setup_data"]

    print(f"Starting simulation for {device_type}...")
    print(f"  GPU: {gpu_type}, Max steps: {num_steps}")

    body = {
        "device_type": device_type,
        "setup_data": setup_data,
        "num_steps": num_steps,
        "check_every_n": check_every_n,
        "source_ramp_periods": source_ramp_periods,
        "gpu_type": gpu_type,
        "min_steps": min_steps,
        "min_stable_checks": min_stable_checks,
        "absorber_width": absorber_width,
        "absorber_coeff": absorber_coeff,
        "significant_power_threshold": significant_power_threshold,
        "required_ports": required_ports,
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        # Start the job
        response = requests.post(
            f"{API_URL}/run_simulation/start",
            json=body,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") != "success":
            raise Exception(f"Failed to start simulation: {result}")

        job_id = result.get("job_id")
        print(f"Job started: {job_id[:8]}...")

        # Poll for completion
        last_progress = ""
        while True:
            time.sleep(poll_interval)

            status_response = requests.get(
                f"{API_URL}/run_simulation/status/{job_id}",
                headers=headers,
                timeout=30
            )
            status_result = status_response.json()

            status = status_result.get("status")
            progress = status_result.get("progress", "")

            if progress and progress != last_progress:
                print(f"  {progress}")
                last_progress = progress

            if status == "completed":
                sim_result = status_result.get("result", {})
                sim_time = sim_result.get("sim_time", 0)
                converged = sim_result.get("converged", False)
                print(f"Simulation completed in {sim_time:.1f}s (converged: {converged})")
                return sim_result

            elif status in ("failed", "error"):
                error = status_result.get("error", "Unknown error")
                print(f"Simulation failed: {error}")
                raise Exception(f"Simulation failed: {error}")

            elif status not in ("starting", "running", "pending"):
                raise Exception(f"Unexpected status: {status}")

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "run_simulation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error running simulation: {e}")
        return None


# =============================================================================
# ONE-SHOT WORKFLOW
# =============================================================================

def simulate(
    device_type: str,
    pdk_config: Dict[str, Any],
    source_port: str = "o1",
    wavelength_um: float = 1.55,
    wavelength_span_nm: float = 100,
    num_wavelengths: int = 5,
    cells_per_wavelength: int = 25,
    num_steps: int = 30000,
    check_every_n: int = 1000,
    source_ramp_periods: float = 5.0,
    gpu_type: str = "H100",
    device_params: Optional[Dict[str, Any]] = None,
    min_steps: int = 0,
    min_stable_checks: int = 3,
    absorber_width: Optional[int] = None,
    absorber_coeff: Optional[float] = None,
    significant_power_threshold: float = 1e-6,
    required_ports: Optional[List[str]] = None,
    poll_interval: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """One-shot FDTD simulation on Modal GPUs (combines setup + simulation).

    NOTE: For better control, use the two-stage workflow instead:
        1. prepare_simulation() - Get mode preview and setup data
        2. run_simulation() - Run FDTD with pre-computed setup

    Args:
        device_type: Device type name (e.g., "mmi2x2", "mmi1x2").
        pdk_config: PDK configuration dict with material parameters:
            - n_core: Core refractive index (e.g., 3.48 for silicon)
            - n_clad: Cladding refractive index (e.g., 1.45 for SiO2)
            - wg_height_um: Waveguide height in micrometers (e.g., 0.22)
            - clad_top_um: Top cladding thickness in micrometers (e.g., 1.89)
            - clad_bot_um: Bottom cladding (BOX) thickness in micrometers (e.g., 2.0)
        source_port: Input port name (e.g., "o1", "o2").
        wavelength_um: Center wavelength in micrometers (default: 1.55).
        wavelength_span_nm: Wavelength span in nanometers (default: 100).
        num_wavelengths: Number of wavelength points (default: 5).
        cells_per_wavelength: FDTD resolution (default: 25).
        num_steps: Maximum FDTD steps (default: 30000).
        check_every_n: Convergence check interval (default: 1000).
        source_ramp_periods: Source ramp-up periods (default: 5.0).
        gpu_type: GPU type - "B200", "H200", "H100", "A100-80GB", etc.
        device_params: Optional dict of device-specific parameters.
        min_steps: Minimum FDTD steps before convergence check.
        min_stable_checks: Required consecutive stable checks.
        absorber_width: PML absorber width in pixels.
        absorber_coeff: PML absorption coefficient.
        significant_power_threshold: Min power for port detection.
        required_ports: List of port names that must have power.
        poll_interval: Seconds between status polls.

    Returns:
        Dict with simulation results:
        - s_parameters: Analyzed transmission data
        - field_intensity: 2D field for visualization
        - sim_time: GPU simulation time in seconds
        - converged: Whether simulation converged

    Example:
        >>> pdk_config = {
        ...     "n_core": 3.48,
        ...     "n_clad": 1.45,
        ...     "wg_height_um": 0.22,
        ...     "clad_top_um": 1.89,
        ...     "clad_bot_um": 2.0,
        ... }
        >>> results = hwc.simulate(
        ...     device_type="mmi2x2",
        ...     pdk_config=pdk_config,
        ...     source_port="o1",
        ...     wavelength_um=1.55,
        ... )
        >>> print(f"T_total: {results['s_parameters']['T_total']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Validate pdk_config
    required_fields = ["n_core", "n_clad", "wg_height_um", "clad_top_um", "clad_bot_um"]
    missing = [f for f in required_fields if f not in pdk_config]
    if missing:
        raise ValueError(f"pdk_config missing required fields: {missing}")

    print(f"Starting simulation for {device_type}...")
    print(f"  Source: {source_port}, Wavelength: {wavelength_um} um (+/- {wavelength_span_nm/2} nm)")
    print(f"  GPU: {gpu_type}")

    body = {
        "device_type": device_type,
        "source_port": source_port,
        "wavelength_um": wavelength_um,
        "wavelength_span_nm": wavelength_span_nm,
        "num_wavelengths": num_wavelengths,
        "cells_per_wavelength": cells_per_wavelength,
        "num_steps": num_steps,
        "check_every_n": check_every_n,
        "source_ramp_periods": source_ramp_periods,
        "gpu_type": gpu_type,
        "pdk_config": pdk_config,
        "device_params": device_params or {},
        "min_steps": min_steps,
        "min_stable_checks": min_stable_checks,
        "absorber_width": absorber_width,
        "absorber_coeff": absorber_coeff,
        "significant_power_threshold": significant_power_threshold,
        "required_ports": required_ports,
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        # Start the job
        response = requests.post(
            f"{API_URL}/simulate/start",
            json=body,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") != "success":
            raise Exception(f"Failed to start simulation: {result}")

        job_id = result.get("job_id")
        print(f"Job started: {job_id[:8]}...")

        # Poll for completion
        last_progress = ""
        while True:
            time.sleep(poll_interval)

            status_response = requests.get(
                f"{API_URL}/simulate/status/{job_id}",
                headers=headers,
                timeout=30
            )
            status_result = status_response.json()

            status = status_result.get("status")
            progress = status_result.get("progress", "")

            if progress and progress != last_progress:
                print(f"  {progress}")
                last_progress = progress

            if status == "completed":
                sim_result = status_result.get("result", {})
                sim_time = sim_result.get("sim_time", 0)
                converged = sim_result.get("converged", False)
                print(f"Simulation completed in {sim_time:.1f}s (converged: {converged})")
                return sim_result

            elif status in ("failed", "error"):
                error = status_result.get("error", "Unknown error")
                print(f"Simulation failed: {error}")
                raise Exception(f"Simulation failed: {error}")

            elif status not in ("starting", "running", "pending"):
                raise Exception(f"Unexpected status: {status}")

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "simulate")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error running simulation: {e}")
        return None


# =============================================================================
# GRANULAR WORKFLOW FUNCTIONS
# =============================================================================

def build_recipe(
    component_name: str,
    component_kwargs: Optional[Dict[str, Any]] = None,
    extension_length: float = 2.0,
    resolution_nm: float = 30.0,
    n_core: float = 3.48,
    n_clad: float = 1.45,
    wg_height_um: float = 0.22,
    total_height_um: float = 4.0,
    padding: Tuple[int, int, int, int] = (100, 100, 0, 0),
    density_radius: int = 3,
    vertical_radius: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Build structure recipe from GDSFactory component on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        component_name: Name of gdsfactory component (e.g., "mmi2x2").
        component_kwargs: Kwargs to pass to component constructor.
        extension_length: Length to extend ports in um.
        resolution_nm: Grid resolution in nanometers.
        n_core: Core refractive index.
        n_clad: Cladding refractive index.
        wg_height_um: Waveguide height in um.
        total_height_um: Total structure height in um.
        padding: (left, right, top, bottom) padding in theta pixels.
        density_radius: Radius for density filtering.
        vertical_radius: Vertical blur radius.

    Returns:
        Dictionary containing recipe, density_core, density_clad, dimensions,
        port_info, layer_config, eps_values, resolution_um.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Building recipe for {component_name}...")

    request_data = {
        "component_name": component_name,
        "component_kwargs": component_kwargs,
        "extension_length": extension_length,
        "resolution_nm": resolution_nm,
        "n_core": n_core,
        "n_clad": n_clad,
        "wg_height_um": wg_height_um,
        "total_height_um": total_height_um,
        "padding": list(padding),
        "density_radius": density_radius,
        "vertical_radius": vertical_radius,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/build_recipe",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        results = response.json()

        dims = results.get('dimensions', [])
        ports = list(results.get('port_info', {}).keys())
        print(f"Recipe built: {dims[0]}x{dims[1]}x{dims[2]} cells")
        print(f"Ports: {ports}")

        return {
            'recipe': results['recipe'],
            'density_core': decode_array(results['density_core_b64']),
            'density_clad': decode_array(results['density_clad_b64']),
            'dimensions': tuple(results['dimensions']),
            'port_info': results['port_info'],
            'layer_config': results['layer_config'],
            'eps_values': tuple(results['eps_values']),
            'resolution_um': results['resolution_um'],
            'padding': tuple(results['padding']),
            'device_info': results.get('device_info'),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "recipe building")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error building recipe: {e}")
        return None


def build_monitors(
    port_info: Dict[str, Any],
    dimensions: Tuple[int, int, int],
    source_port: str = "o1",
    monitor_x_um: float = 0.1,
    monitor_y_um: float = 1.5,
    monitor_z_um: float = 1.5,
    resolution_um: float = 0.03,
    source_offset_cells: int = 5,
) -> Optional[Dict[str, Any]]:
    """Build monitors from port information on Modal CPU.

    This function runs on CPU and does NOT consume credits.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Building monitors (source: {source_port})...")

    request_data = {
        "port_info": port_info,
        "dimensions": list(dimensions),
        "source_port": source_port,
        "monitor_x_um": monitor_x_um,
        "monitor_y_um": monitor_y_um,
        "monitor_z_um": monitor_z_um,
        "resolution_um": resolution_um,
        "source_offset_cells": source_offset_cells,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/build_monitors",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        print(f"Monitors built: {list(result.get('monitor_names', {}).keys())}")
        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "monitor building")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error building monitors: {e}")
        return None


def compute_freq_band(
    wl_min_um: float = 1.55,
    wl_max_um: float = 1.55,
    n_freqs: int = 1,
    resolution_um: float = 0.03,
) -> Optional[Dict[str, Any]]:
    """Compute frequency band from wavelength range on Modal CPU.

    This function runs on CPU and does NOT consume credits.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Computing freq band: {wl_min_um}-{wl_max_um} um ({n_freqs} points)...")

    request_data = {
        "wl_min_um": wl_min_um,
        "wl_max_um": wl_max_um,
        "n_freqs": n_freqs,
        "resolution_um": resolution_um,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/compute_freq_band",
            json=request_data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        results = response.json()
        print(f"Freq band computed: {results['freq_band']}")

        return {
            'freq_band': tuple(results['freq_band']),
            'wavelengths_um': results['wavelengths_um'],
            'frequencies_omega': results['frequencies_omega'],
            'resolution_um': results['resolution_um'],
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "frequency band computation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error computing freq band: {e}")
        return None


def solve_mode_source(
    density_core: np.ndarray,
    density_clad: np.ndarray,
    source_x_position: int,
    mode_bounds: Dict[str, int],
    layer_config: Dict[str, Any],
    eps_values: Tuple[float, float],
    freq_band: Tuple[float, float, int],
    slice_half_width: int = 5,
    mode_num: int = 0,
    propagation_axis: str = "x",
) -> Optional[Dict[str, Any]]:
    """Solve for waveguide mode source field on Modal CPU.

    This function runs on CPU and does NOT consume credits.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Solving mode at x={source_x_position}...")

    request_data = {
        "density_core_b64": encode_array(np.array(density_core)),
        "density_core_shape": list(density_core.shape),
        "density_clad_b64": encode_array(np.array(density_clad)),
        "density_clad_shape": list(density_clad.shape),
        "source_x_position": source_x_position,
        "mode_bounds": mode_bounds,
        "layer_config": layer_config,
        "eps_values": list(eps_values),
        "freq_band": list(freq_band),
        "slice_half_width": slice_half_width,
        "mode_num": mode_num,
        "propagation_axis": propagation_axis,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/solve_mode_source",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        results = response.json()

        mode_info = results.get('mode_info', {})
        n_eff = mode_info.get('n_eff', 'N/A')
        print(f"Mode solved: n_eff={n_eff}")

        return {
            'source_field': decode_array(results['source_field_b64']),
            'source_offset': tuple(results['source_offset']),
            'mode_info': mode_info,
            'freq_band': tuple(results['freq_band']),
            'solve_time_seconds': results.get('solve_time_seconds', 0.0),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "mode source solving")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error solving mode source: {e}")
        return None


def get_default_absorber_params(
    structure_dimensions: Tuple[int, int, int],
    absorber_fraction: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Get default absorber parameters based on structure size on Modal CPU.

    This function runs on CPU and does NOT consume credits.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Computing absorber params...")

    request_data = {
        "structure_dimensions": list(structure_dimensions),
        "absorber_fraction": absorber_fraction,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/get_default_absorber_params",
            json=request_data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        results = response.json()

        print(f"Absorber params computed")

        return {
            'absorption_widths': tuple(results['absorption_widths']),
            'absorption_coeff': results['absorption_coeff'],
            'add_absorption': results['add_absorption'],
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "absorber parameter computation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting absorber params: {e}")
        return None
